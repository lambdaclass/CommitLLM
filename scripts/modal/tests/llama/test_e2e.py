"""
Llama 3.1 8B W8A8 — E2E protocol test.

Exercises the full commit-and-audit pipeline the way a real verifier would:
  - Multiple requests with varied prompts and generation lengths
  - Challenge seeds derived from commitment (not hardcoded)
  - Mix of routine (~90%) and full (~10%) audits
  - Random token positions selected by the challenge
  - Tamper detection on a separate request
  - EOS early-stop handling

Llama uses the strongest verification tier:
  - ExactReplay attention (f64 Q·K^T)
  - ExactTokenIdentity decode (argmax must match)
  - QKV Freivalds checks

Usage:
    modal run --detach scripts/modal/tests/llama/test_e2e.py
"""

import sys, os

# _pins lives two directories up
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("commitllm-test-llama-e2e")

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
        ".git", "target", "**/__pycache__", "*.pyc", "*.pdf", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)


def _run():
    import hashlib
    import json
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    failures = []

    def check(cond, msg):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    # ── Load model ──
    print(f"\nLoading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))
    print(f"  {n_layers} layers\n")

    # ── Keygen ──
    print("Generating verifier key...")
    verifier_secret = hashlib.sha256(b"test-llama-e2e-verifier-secret").digest()
    t0 = time.time()
    key_bin, artifact_bin = verilm_rs.generate_key_binary(model_dir, verifier_secret)
    print(f"  {len(key_bin)/1024/1024:.1f} MB, {(time.time()-t0)*1000:.0f} ms")

    # Inspect profile to assert correct verification mode
    key_json = verilm_rs.generate_key(model_dir, verifier_secret)
    key = json.loads(key_json)
    profile = key.get("verification_profile", {})
    attn_mode = profile.get("attention_mode", "unknown")
    decode_mode = profile.get("decode_acceptance", "unknown")
    print(f"  profile: {profile.get('name', 'unknown')}")
    print(f"  attention_mode: {attn_mode}")
    print(f"  decode_acceptance: {decode_mode}")
    check(attn_mode == "ExactReplay", f"attention mode is ExactReplay (got {attn_mode})")
    check(decode_mode == "ExactTokenIdentity", f"decode acceptance is ExactTokenIdentity (got {decode_mode})")
    print()

    # ── Helper: derive challenge seed from commitment ──
    # TODO: production version should hash a canonical commitment digest
    # (e.g. the full serialized receipt bytes), not hand-concatenated fields.
    # This is sufficient for testing but two commitments with missing/renamed
    # fields could accidentally derive the same challenge input.
    def derive_test_challenge_seed(commitment, verifier_secret):
        """Test helper — derives a challenge seed from commitment fields."""
        merkle_root = commitment.get("merkle_root", "")
        io_root = commitment.get("io_root", "")
        return hashlib.sha256(
            f"{merkle_root}:{io_root}".encode() + verifier_secret
        ).digest()

    # ── Test 1: Batch of requests with mixed audit tiers ──
    # This mirrors real usage: the verifier receives multiple requests,
    # derives a challenge for each from the commitment, and picks
    # routine or full audit based on a policy (here: every 3rd is full).
    prompts = [
        ("What causes rainbows?", 64, 0.0),
        ("Explain how a CPU pipeline works.", 96, 0.0),
        ("What is the theory of general relativity?", 64, 0.0),
        ("Write a haiku about cryptography.", 48, 0.0),
        ("What is the difference between TCP and UDP?", 80, 0.0),
        ("Why is the sky blue?", 64, 0.0),
    ]

    print(f"Test 1: Mixed audit batch ({len(prompts)} requests)")
    print(f"  Policy: every 3rd request gets full audit, rest get routine\n")

    for i, (prompt, max_tok, temp) in enumerate(prompts):
        # Step 1: Provider generates response and returns commitment
        result = server.chat(prompt=prompt, max_tokens=max_tok, temperature=temp)
        commitment = result["commitment"]
        n_gen = result["n_tokens"] - commitment.get("n_prompt_tokens", 0)

        check(commitment.get("version") == "V4", f"req {i}: commitment is V4")

        # Step 2: Verifier derives challenge seed from the commitment
        challenge_seed = derive_test_challenge_seed(commitment, verifier_secret)

        # Step 3: Pick audit tier — full every 3rd request, routine otherwise
        is_full = (i % 3 == 0)
        tier = "full" if is_full else "routine"

        # Step 4: Build challenge (token position + layers from seed)
        challenge = verilm_rs.build_audit_challenge(
            list(challenge_seed), result["n_tokens"], n_layers, tier
        )
        tok_idx = challenge["token_index"]
        layers = challenge["layer_indices"]

        tier_label = f"FULL ({len(layers)}/{n_layers} layers)" if is_full else f"routine ({len(layers)}/{n_layers} layers)"
        print(f"  req {i}: \"{prompt[:40]}...\" → {n_gen} tok, {tier_label}, challenge token={tok_idx}")

        # Step 5: Provider opens the challenged position
        audit = server.audit(
            request_id=result["request_id"],
            token_index=tok_idx,
            layer_indices=layers,
            tier=tier,
            binary=True,
        )

        # Step 6: Verifier checks
        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        check(report["passed"], f"req {i} ({tier}): {report['checks_passed']}/{report['checks_run']} checks")
        if not report["passed"]:
            for f in report.get("failures", [])[:3]:
                print(f"    {f}")

    print()

    # ── Test 2: Multiple positions on same request ──
    # A verifier may challenge the same request at multiple token positions
    # (e.g., deep audit after a routine audit flags something).
    print("Test 2: Multiple challenge positions on one request")
    result_multi = server.chat(
        prompt="Write a short poem about the ocean.",
        max_tokens=128, temperature=0.0,
    )
    commitment_multi = result_multi["commitment"]
    n_prompt = commitment_multi.get("n_prompt_tokens", 0)
    n_gen = result_multi["n_tokens"] - n_prompt

    # Derive 3 different challenges by varying a counter in the seed
    for j in range(3):
        seed_j = hashlib.sha256(
            derive_test_challenge_seed(commitment_multi, verifier_secret) + j.to_bytes(4, "little")
        ).digest()
        challenge = verilm_rs.build_audit_challenge(
            list(seed_j), result_multi["n_tokens"], n_layers, "full"
        )
        tok_idx = challenge["token_index"]
        print(f"  challenge {j}: token={tok_idx}")

        a = server.audit(
            request_id=result_multi["request_id"],
            token_index=tok_idx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        r = verilm_rs.verify_v4_full_binary(bytes(a), key_bin, artifact_bin)
        check(r["passed"], f"position {tok_idx}: {r['checks_passed']}/{r['checks_run']} checks")
        if not r["passed"]:
            for f in r.get("failures", [])[:3]:
                print(f"    {f}")
    print()

    # ── Test 3: Tamper detection ──
    print("Test 3: Tamper detection (bit-flip → verifier rejects)")
    result_tamper = server.chat(prompt="What is gravity?", max_tokens=32, temperature=0.0)
    challenge_seed = derive_test_challenge_seed(result_tamper["commitment"], verifier_secret)
    challenge = verilm_rs.build_audit_challenge(
        list(challenge_seed), result_tamper["n_tokens"], n_layers, "full"
    )

    audit_honest = server.audit(
        request_id=result_tamper["request_id"],
        token_index=challenge["token_index"],
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )
    report_honest = verilm_rs.verify_v4_full_binary(bytes(audit_honest), key_bin, artifact_bin)
    check(report_honest["passed"], "honest audit passes")

    tampered = bytearray(audit_honest)
    if len(tampered) > 100:
        tampered[100] ^= 0xFF
    report_tamper = verilm_rs.verify_v4_full_binary(bytes(tampered), key_bin, artifact_bin)
    check(not report_tamper["passed"], "tampered audit rejected")
    print()

    # ── Test 4: EOS early stop ──
    print("Test 4: EOS early stop")
    eos = server.chat(prompt="What is 2+2? Just the number.", max_tokens=256, temperature=0.0)
    eos_n = eos["n_tokens"]
    print(f"  {eos_n} tokens (EOS {'early' if eos_n < 256 else 'at limit'})")

    challenge_seed = derive_test_challenge_seed(eos["commitment"], verifier_secret)
    challenge = verilm_rs.build_audit_challenge(
        list(challenge_seed), eos_n, n_layers, "full"
    )
    eos_audit = server.audit(
        request_id=eos["request_id"],
        token_index=challenge["token_index"],
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )
    eos_report = verilm_rs.verify_v4_full_binary(bytes(eos_audit), key_bin, artifact_bin)
    check(eos_report["passed"], f"EOS verify: {eos_report['checks_passed']}/{eos_report['checks_run']} checks")
    print()

    # ── Summary ──
    print("=" * 50)
    if failures:
        print(f"FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL TESTS PASSED")
    print("=" * 50)

    return {"passed": len(failures) == 0, "failures": failures}


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run():
    return _run()


@app.local_entrypoint()
def main():
    print("CommitLLM E2E — Llama 3.1 8B W8A8")
    print("=" * 50)
    result = run.remote()
    if not result["passed"]:
        raise SystemExit(1)
