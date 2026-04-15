"""
CommitLLM — Full E2E Demo: Llama 3.1 8B W8A8

Demonstrates the complete cryptographic commit-and-audit protocol:

  1. Prover loads model and generates text with capture hooks
  2. Prover commits: Merkle root over residuals, KV transcript, embeddings
  3. Verifier generates a key from model weights (independent of prover)
  4. Verifier challenges: picks token + layers to audit
  5. Prover opens: shell opening (QKV, residuals, attention) for challenged layers
  6. Verifier replays: exact integer bridge, exact f64 attention, token identity
  7. Tamper detection: bit-flip in audit payload → verification fails

Llama 3.1 8B uses the strongest verification tier:
  - ExactReplay attention (f64 Q·K^T, no tolerance needed)
  - ExactTokenIdentity decode (logit argmax must match committed token)
  - QKV Freivalds randomized checks
  - Full residual bridge with integer arithmetic

Usage:
    modal run --detach scripts/modal/demo_llama_e2e.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("commitllm-demo-llama-e2e")

MODEL_ID = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "ca-certificates")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VERILM_CAPTURE": "1",
        "VERILM_CAPTURE_X_ATTN": "1",
    })
    .pip_install(*VERIFICATION)
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "__pycache__", "*.pdf", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

# Demo prompts covering different lengths and domains
DEMO_PROMPTS = [
    ("factual",   "What causes rainbows? Explain briefly.",                    64),
    ("reasoning", "If all roses are flowers and some flowers fade quickly, "
                  "can we conclude that some roses fade quickly? Explain.",     128),
    ("code",      "Write a Python function to check if a string is a palindrome.", 256),
]


def _banner(text, char="=", width=72):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _run_demo():
    import hashlib
    import json
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    failures = []
    timings = {}

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")
        return cond

    # ================================================================
    # STEP 1: Load Model
    # ================================================================
    _banner("STEP 1: Load Model")
    print(f"  Model: {MODEL_ID}")
    print(f"  Quantization: W8A8 (INT8 weights, INT8 activations)")
    print(f"  Verification tier: ExactReplay + ExactTokenIdentity")

    t0 = time.time()
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    load_ms = (time.time() - t0) * 1000
    timings["model_load_ms"] = load_ms

    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))
    print(f"  Layers: {n_layers}")
    print(f"  Load time: {load_ms/1000:.1f}s")

    # ================================================================
    # STEP 2: Generate Verifier Key (independent of prover)
    # ================================================================
    _banner("STEP 2: Generate Verifier Key")
    print("  The verifier key is derived from model weights + a random seed.")
    print("  Anyone with the same model can generate this key independently.")

    seed = hashlib.sha256(b"commitllm-demo-llama").digest()
    t0 = time.time()
    key_binary, artifact_binary = verilm_rs.generate_key_binary(model_dir, seed)
    keygen_ms = (time.time() - t0) * 1000
    timings["keygen_ms"] = keygen_ms

    key_mb = len(key_binary) / 1024 / 1024
    artifact_mb = len(artifact_binary) / 1024 / 1024 if artifact_binary else 0
    print(f"  Key size: {key_mb:.1f} MB")
    if artifact_binary:
        print(f"  Artifact size: {artifact_mb:.1f} MB")
    print(f"  Keygen time: {keygen_ms:.0f} ms")

    # Also generate JSON key for routine-tier comparison
    key_json = verilm_rs.generate_key(model_dir, seed)

    # ================================================================
    # STEP 3: Generate + Commit + Audit + Verify (per prompt)
    # ================================================================
    prompt_results = []
    for label, prompt, max_tokens in DEMO_PROMPTS:
        _banner(f"STEP 3: Inference + Verify — {label}", char="-")
        print(f"  Prompt: \"{prompt[:70]}{'...' if len(prompt) > 70 else ''}\"")
        print(f"  Max tokens: {max_tokens}")

        # ── 3a: Chat (capture + commit) ──
        print(f"\n  3a. Generate text with cryptographic capture...")
        t0 = time.time()
        result = server.chat(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        chat_ms = (time.time() - t0) * 1000

        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt
        text = result["generated_text"]

        print(f"  Generated {n_gen} tokens in {chat_ms:.0f} ms ({n_gen/(chat_ms/1000):.0f} tok/s)")
        print(f"  Commitment version: {commitment.get('version')}")
        print(f"  Merkle root: {commitment.get('merkle_root', '')[:32]}...")
        kv_roots = commitment.get("kv_roots", [])
        print(f"  KV roots: {len(kv_roots)} layers committed")
        print(f"  Output: \"{text[:120]}{'...' if len(text) > 120 else ''}\"")

        check(n_gen > 0, f"generated {n_gen} tokens")
        check(commitment.get("version") == "V4", "commitment version is V4")
        check(len(kv_roots) == n_layers, f"KV roots cover all {n_layers} layers")

        # ── 3b: Full-tier audit (all layers, binary) ──
        print(f"\n  3b. Full-tier audit (all {n_layers} layers, token 0)...")
        t0 = time.time()
        audit_binary = server.audit(
            request_id=request_id,
            token_index=0,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        audit_ms = (time.time() - t0) * 1000
        audit_kb = len(audit_binary) / 1024

        print(f"  Audit payload: {audit_kb:.1f} KB (bincode+zstd)")
        print(f"  Audit time: {audit_ms:.0f} ms")

        # ── 3c: Verify (binary key path) ──
        print(f"\n  3c. Verify audit (binary key path)...")
        N_REPS = 3
        verify_times = []
        for _ in range(N_REPS):
            t0 = time.time()
            report = verilm_rs.verify_v4_full_binary(
                bytes(audit_binary), key_binary, artifact_binary
            )
            verify_times.append((time.time() - t0) * 1000)

        verify_ms = sorted(verify_times)[N_REPS // 2]
        passed = report["passed"]
        checks_run = report["checks_run"]
        checks_passed = report["checks_passed"]

        status = "PASSED" if passed else "FAILED"
        print(f"  Result: {status}")
        print(f"  Checks: {checks_passed}/{checks_run}")
        print(f"  Verify time: {verify_ms:.1f} ms (median of {N_REPS})")

        check(passed, f"full-tier verification {status} ({checks_passed}/{checks_run})")

        if not passed and report.get("failures"):
            print(f"  Failures:")
            for f in report["failures"][:5]:
                print(f"    - {f}")

        # ── 3d: Routine-tier audit (subset of layers) ──
        print(f"\n  3d. Routine-tier audit (contiguous prefix)...")
        result2 = server.chat(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        challenge_seed = hashlib.sha256(f"demo-routine-{label}".encode()).digest()
        challenge = verilm_rs.build_audit_challenge(
            list(challenge_seed), result2["n_tokens"], n_layers, "routine"
        )
        routine_layers = challenge["layer_indices"]
        routine_token = challenge["token_index"]

        audit_routine_bin = server.audit(
            request_id=result2["request_id"],
            token_index=routine_token,
            layer_indices=routine_layers,
            tier="routine",
            binary=True,
        )
        report_routine = verilm_rs.verify_v4_full_binary(
            bytes(audit_routine_bin), key_binary, artifact_binary
        )
        routine_kb = len(audit_routine_bin) / 1024

        print(f"  Layers: {len(routine_layers)}/{n_layers} (token {routine_token})")
        print(f"  Payload: {routine_kb:.1f} KB (vs {audit_kb:.1f} KB full)")
        print(f"  Result: {'PASSED' if report_routine['passed'] else 'FAILED'}")
        print(f"  Checks: {report_routine['checks_passed']}/{report_routine['checks_run']}")

        check(report_routine["passed"],
              f"routine-tier verification ({report_routine['checks_passed']}/{report_routine['checks_run']})")

        prompt_results.append({
            "label": label,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "chat_ms": chat_ms,
            "full_audit_kb": audit_kb,
            "full_verify_ms": verify_ms,
            "full_passed": passed,
            "full_checks": f"{checks_passed}/{checks_run}",
            "routine_audit_kb": routine_kb,
            "routine_passed": report_routine["passed"],
            "routine_checks": f"{report_routine['checks_passed']}/{report_routine['checks_run']}",
            "text_preview": text[:100],
        })

    # ================================================================
    # STEP 4: Tamper Detection
    # ================================================================
    _banner("STEP 4: Tamper Detection")
    print("  Flip a single bit in the audit payload → verification must fail.")

    tamper_result = server.chat(prompt="What is gravity?", max_tokens=32, temperature=0.0)
    tamper_audit = server.audit(
        request_id=tamper_result["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )

    # Honest verification
    report_honest = verilm_rs.verify_v4_full_binary(
        bytes(tamper_audit), key_binary, artifact_binary
    )
    print(f"  Honest:   {'PASSED' if report_honest['passed'] else 'FAILED'} "
          f"({report_honest['checks_passed']}/{report_honest['checks_run']})")
    check(report_honest["passed"], "honest audit passes")

    # Tampered verification (flip one byte)
    tampered = bytearray(tamper_audit)
    if len(tampered) > 100:
        tampered[100] ^= 0xFF
    report_tamper = verilm_rs.verify_v4_full_binary(
        bytes(tampered), key_binary, artifact_binary
    )
    print(f"  Tampered: {'PASSED' if report_tamper['passed'] else 'FAILED'} "
          f"({report_tamper['checks_passed']}/{report_tamper['checks_run']})")
    check(not report_tamper["passed"], "tampered audit correctly rejected")

    if report_tamper.get("failures"):
        print(f"  Detection: {report_tamper['failures'][0][:80]}...")

    # ================================================================
    # STEP 5: EOS Handling (early stop)
    # ================================================================
    _banner("STEP 5: EOS Handling")
    print("  Short-answer prompt with high max_tokens → model stops early.")

    eos_result = server.chat(
        prompt="What is 2+2? Answer with just the number.",
        max_tokens=256, temperature=0.0,
    )
    eos_n = eos_result["n_tokens"]
    eos_commitment = eos_result["commitment"]
    eos_early = eos_n < 256
    print(f"  Tokens: {eos_n} (EOS {'fired early' if eos_early else 'did not fire early'})")

    eos_audit = server.audit(
        request_id=eos_result["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )
    eos_report = verilm_rs.verify_v4_full_binary(
        bytes(eos_audit), key_binary, artifact_binary
    )
    print(f"  Verify: {'PASSED' if eos_report['passed'] else 'FAILED'} "
          f"({eos_report['checks_passed']}/{eos_report['checks_run']})")
    check(eos_report["passed"], "EOS-trimmed commit verifies correctly")

    # ================================================================
    # STEP 6: Deep-Prefix Replay (committed KV transcript)
    # ================================================================
    _banner("STEP 6: Deep-Prefix Replay")
    print("  Verify KV transcript Merkle proofs against committed roots.")

    dp_result = server.chat(
        prompt="Explain what a hash function is.", max_tokens=64, temperature=0.0
    )
    dp_kv_roots = dp_result["commitment"].get("kv_roots", [])
    print(f"  KV roots: {len(dp_kv_roots)} layers")
    check(len(dp_kv_roots) == n_layers, f"KV roots cover all {n_layers} layers")

    dp_audit = server.audit(
        request_id=dp_result["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
        deep_prefix=True,
    )
    dp_report = verilm_rs.verify_v4_full_binary(
        bytes(dp_audit), key_binary, artifact_binary
    )
    kv_failures = [f for f in dp_report.get("failures", []) if "KV" in f]
    print(f"  Payload: {len(dp_audit)/1024:.1f} KB")
    print(f"  Verify: {'PASSED' if dp_report['passed'] else 'FAILED'} "
          f"({dp_report['checks_passed']}/{dp_report['checks_run']})")
    print(f"  KV Merkle failures: {len(kv_failures)}")
    check(len(kv_failures) == 0, "KV Merkle proofs verified")

    # ================================================================
    # SUMMARY
    # ================================================================
    _banner("DEMO SUMMARY", char="=")

    total_checks = sum(1 for _ in failures)  # count check() calls that failed
    all_passed = len(failures) == 0

    print(f"\n  Model:  {MODEL_ID}")
    print(f"  Layers: {n_layers}")
    print(f"  Key:    {key_mb:.1f} MB (binary)")
    print(f"  Keygen: {keygen_ms:.0f} ms")

    print(f"\n  {'Label':<12} {'Gen':>4} {'Chat ms':>8} {'Audit KB':>9} {'Verify ms':>10} {'Full':>6} {'Routine':>8}")
    print(f"  {'-'*12} {'-'*4} {'-'*8} {'-'*9} {'-'*10} {'-'*6} {'-'*8}")
    for p in prompt_results:
        full_status = "PASS" if p["full_passed"] else "FAIL"
        routine_status = "PASS" if p["routine_passed"] else "FAIL"
        print(f"  {p['label']:<12} {p['n_gen']:>4} {p['chat_ms']:>8.0f} "
              f"{p['full_audit_kb']:>9.1f} {p['full_verify_ms']:>10.1f} "
              f"{full_status:>6} {routine_status:>8}")

    print(f"\n  Tamper detection: {'OK' if not report_tamper['passed'] else 'BROKEN'}")
    print(f"  EOS handling:     {'OK' if eos_report['passed'] else 'BROKEN'}")
    print(f"  Deep-prefix KV:   {'OK' if len(kv_failures) == 0 else 'BROKEN'}")

    if all_passed:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")

    print()

    return {
        "passed": all_passed,
        "n_failures": len(failures),
        "failures": failures,
        "model": MODEL_ID,
        "n_layers": n_layers,
        "keygen_ms": keygen_ms,
        "key_mb": key_mb,
        "prompts": prompt_results,
        "tamper_detected": not report_tamper["passed"],
        "eos_ok": eos_report["passed"],
        "deep_prefix_ok": len(kv_failures) == 0,
    }


@app.function(image=image, gpu="A100-80GB", timeout=1200)
def run_demo():
    return _run_demo()


@app.local_entrypoint()
def main():
    print("=" * 72)
    print("  CommitLLM — Cryptographic Inference Verification Demo")
    print("  Model: Llama 3.1 8B Instruct (W8A8)")
    print("=" * 72)
    result = run_demo.remote()

    import json
    print("\n\nFull results JSON:")
    print(json.dumps(result, indent=2, default=str))

    if not result["passed"]:
        raise SystemExit(1)
