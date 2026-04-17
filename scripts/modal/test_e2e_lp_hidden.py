"""
E2E test: LP hidden bf16 decode boundary verification with real Qwen W8A8 weights.

Exercises the full commit→open→verify path through the LP hidden verification mode:

  1. VerifiedInferenceServer.chat() — capture LP hidden + commit (Merkle binding)
  2. verilm_rs.generate_key() — must populate lm_head_bf16 for Qwen W8A8
  3. server.audit() — open V4 with lp_hidden_bf16 in shell opening
  4. verilm_rs.verify_v4_full_binary() — canonical verifier runs phase_lm_head_lp_hidden

Tests both greedy (argmax) and sampled (canonical sampler + committed seed) decode.

Usage:
    modal run --detach scripts/modal/test_e2e_lp_hidden.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-test-e2e-lp-hidden")

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"

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
        ".git", "target", "scripts/__pycache__", "*.pdf", "*.md", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

GREEDY_PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
]
SAMPLED_PROMPTS = [
    "Write a short poem about the ocean.",
    "Tell me a fun fact about space.",
]


def _run_e2e():
    import hashlib
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir

    failures = []

    def assert_true(cond, msg):
        if not cond:
            failures.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK: {msg}")

    # ── Step 1: Generate verifier key (binary path) ──
    # Binary key avoids JSON serialization of lm_head_bf16 (544M u16 elements
    # for Qwen 7B = ~1 GB as JSON text). Binary is ~1 GB bincode, serialized
    # in seconds instead of minutes.
    print("\n1. Generate verifier key (binary)...")
    seed = hashlib.sha256(b"lp-hidden-e2e-test").digest()
    t0 = time.time()
    key_binary, artifact_binary = verilm_rs.generate_key_binary(model_dir, seed)
    keygen_ms = (time.time() - t0) * 1000
    print(f"  keygen: {keygen_ms:.0f}ms, key size: {len(key_binary)} bytes ({len(key_binary)/1024/1024:.1f} MB)")
    if artifact_binary:
        print(f"  decode artifact: {len(artifact_binary)} bytes ({len(artifact_binary)/1024/1024:.1f} MB)")
    assert_true(
        key_binary[:4] == b"VKEY",
        f"key binary magic is VKEY (got {key_binary[:4]})"
    )

    # ── Step 2: Greedy decode — full E2E ──
    full_layers = list(range(n_layers))

    for pi, prompt in enumerate(GREEDY_PROMPTS):
        print(f"\n{'='*60}")
        print(f"2.{pi} Greedy E2E — {prompt[:50]}...")
        print(f"{'='*60}")

        result = server.chat(prompt=prompt, max_tokens=64, temperature=0.0)
        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt

        assert_true(n_gen > 0, f"generated {n_gen} tokens")
        assert_true(commitment.get("version") == "V4", f"commitment version V4")
        print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
        print(f"  text: {result['generated_text'][:80]}...")

        # Check LP hidden was committed (server should have lp_hidden_raw).
        # The LP hook fires once per LogitsProcessor call (prefill + decode).
        # After server trim, count should be n_gen or n_gen+1 (prefill row
        # may be retained depending on vLLM forward pass structure).
        lp_raw = getattr(server, '_last_lp_hidden', None) or []
        assert_true(
            len(lp_raw) in (n_gen, n_gen + 1),
            f"LP hidden captures near gen tokens: {len(lp_raw)} (expected {n_gen} or {n_gen+1})"
        )

        # Audit token 0 (full tier, binary).
        audit_binary = server.audit(
            request_id=request_id,
            token_index=0,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        assert_true(
            isinstance(audit_binary, (bytes, memoryview)),
            "binary audit returns bytes"
        )
        print(f"  binary payload: {len(audit_binary)} bytes")

        # Verify — this is the critical test. The canonical verifier must
        # run phase_lm_head_lp_hidden and confirm token identity via bf16
        # lm_head matmul.
        report = verilm_rs.verify_v4_full_binary(bytes(audit_binary), key_binary, artifact_binary)
        checks = report["checks_run"]
        passed = report["checks_passed"]
        assert_true(
            report["passed"],
            f"greedy verify passed ({passed}/{checks} checks)"
        )
        if not report["passed"]:
            for f in report["failures"]:
                print(f"    FAILURE: {f}")
        # Confirm LP hidden path was exercised (not silently skipped).
        # The LP hidden path adds checks for lp_hidden presence, size,
        # bf16 matmul, and token identity. If these are skipped, the
        # check count will be lower.
        assert_true(
            checks >= 20,
            f"sufficient checks run (got {checks}, expect >=20 with LP hidden)"
        )
        # Coverage must be full.
        cov = report.get("coverage", {})
        assert_true(cov.get("level") == "full", f"coverage: {cov}")

        # Check for LP-hidden-related skips (would indicate silent bypass).
        skipped = report.get("skipped", [])
        lp_skips = [s for s in skipped if "lp_hidden" in s.lower() or "lm_head" in s.lower()]
        assert_true(
            len(lp_skips) == 0,
            f"no LP hidden skips ({lp_skips})"
        )

    # ── Step 3: Sampled decode — exact canonical replay ──
    for pi, prompt in enumerate(SAMPLED_PROMPTS):
        print(f"\n{'='*60}")
        print(f"3.{pi} Sampled E2E — {prompt[:50]}...")
        print(f"{'='*60}")

        temperature = 0.8
        top_k = 50
        top_p = 0.9
        result = server.chat(
            prompt=prompt, max_tokens=32,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt
        gen_token_ids = result["token_ids"][n_prompt:]

        assert_true(n_gen > 0, f"generated {n_gen} tokens")
        print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
        print(f"  tokens: {gen_token_ids[:10]}...")
        print(f"  text: {result['generated_text'][:80]}...")

        lp_raw = getattr(server, '_last_lp_hidden', None) or []
        assert_true(
            len(lp_raw) in (n_gen, n_gen + 1),
            f"LP hidden captures near gen tokens: {len(lp_raw)} (expected {n_gen} or {n_gen+1})"
        )

        # Audit token 0 (binary).
        audit_binary = server.audit(
            request_id=request_id,
            token_index=0,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )

        # Verify — canonical verifier must replay sampled token identity
        # using derive_token_seed + canonical_sample over bf16 logits.
        report = verilm_rs.verify_v4_full_binary(bytes(audit_binary), key_binary, artifact_binary)
        checks = report["checks_run"]
        passed = report["checks_passed"]
        assert_true(
            report["passed"],
            f"sampled verify passed ({passed}/{checks} checks)"
        )
        if not report["passed"]:
            for f in report["failures"]:
                print(f"    FAILURE: {f}")
        assert_true(
            checks >= 20,
            f"sufficient checks run (got {checks}, expect >=20 with LP hidden)"
        )

    # ── Step 4: Verify tamper detection still works ──
    print(f"\n{'='*60}")
    print("4. Tamper detection...")
    print(f"{'='*60}")
    # Use the last sampled result — tamper the binary payload.
    # Re-audit to get a fresh binary, then flip a byte in the shell opening region.
    result_tamper = server.chat(prompt="What is 1+1?", max_tokens=16, temperature=0.0)
    audit_tamper = server.audit(
        request_id=result_tamper["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )
    # Flip a byte deep in the payload (past magic + zstd header).
    tampered = bytearray(audit_tamper)
    if len(tampered) > 100:
        tampered[100] ^= 0xFF
    report_tampered = verilm_rs.verify_v4_full_binary(bytes(tampered), key_binary, artifact_binary)
    assert_true(
        not report_tampered["passed"],
        "tampered audit correctly rejected"
    )

    # ── Step 5: EOS trim regression ──
    print(f"\n{'='*60}")
    print("5. EOS trim (LP hidden count must match after early EOS)...")
    print(f"{'='*60}")
    eos_result = server.chat(
        prompt="What is 2+2? Answer with just the number.",
        max_tokens=256, temperature=0.0,
    )
    eos_n = eos_result["n_tokens"]
    eos_commitment = eos_result["commitment"]
    n_prompt_eos = eos_commitment.get("n_prompt_tokens", 0)
    n_gen_eos = eos_n - n_prompt_eos
    lp_eos = getattr(server, '_last_lp_hidden', None) or []
    print(f"  n_gen={n_gen_eos}, lp_captures={len(lp_eos)}, early_eos={n_gen_eos < 256}")
    assert_true(eos_commitment is not None, "EOS trim: commit succeeded")
    assert_true(
        len(lp_eos) in (n_gen_eos, n_gen_eos + 1),
        f"LP hidden count near gen tokens after EOS: {len(lp_eos)} (expected {n_gen_eos} or {n_gen_eos+1})"
    )
    # Verify the EOS-trimmed commit.
    eos_audit = server.audit(
        request_id=eos_result["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )
    eos_report = verilm_rs.verify_v4_full_binary(bytes(eos_audit), key_binary, artifact_binary)
    assert_true(
        eos_report["passed"],
        f"EOS trim verify passed ({eos_report['checks_passed']}/{eos_report['checks_run']} checks)"
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    if failures:
        print(f"E2E LP HIDDEN TEST FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("E2E LP HIDDEN TEST PASSED")
    print(f"{'='*60}")

    return {
        "passed": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures,
    }


@app.function(image=image, gpu="A100-80GB", timeout=1200)
def run_e2e():
    return _run_e2e()


@app.local_entrypoint()
def main():
    print("VeriLM E2E: LP Hidden bf16 Decode Boundary Verification")
    print("=" * 60)
    result = run_e2e.remote()
    if result["passed"]:
        print("\nE2E LP HIDDEN TEST PASSED")
    else:
        print(f"\nE2E LP HIDDEN TEST FAILED — {result['n_failures']} failure(s):")
        for f in result["failures"]:
            print(f"  - {f}")
        raise SystemExit(1)
