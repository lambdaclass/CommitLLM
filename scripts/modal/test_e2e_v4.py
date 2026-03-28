"""
E2E test: chat → commit → audit → verify_v4 through the live server path.

Runs on Modal with a real W8A8 model. Exercises the full protocol:

  1. VerifiedInferenceServer.chat() — capture + commit
  2. server.audit() — open V4 with cached WeightProvider + layer_indices
  3. verilm_rs.verify_v4() — key-only verification

Tests both "full" (all layers) and "routine" (contiguous prefix) tiers.

Usage:
    modal run --detach scripts/modal_test_e2e_v4.py
"""

import modal

app = modal.App("verilm-test-e2e-v4")

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
    })
    .pip_install("vllm>=0.8", "torch", "numpy", "fastapi", "maturin")
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

PROMPT = "Explain the theory of relativity in one paragraph."
MAX_TOKENS = 32


def _run_e2e():
    import hashlib
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True, enable_prefix_caching=False)
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

    # ── Step 1: Chat (capture + commit) ──
    print("\n1. Chat (capture + commit)...")
    cap._capture_mode = "minimal"
    result = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
    request_id = result["request_id"]
    n_tokens = result["n_tokens"]
    commitment = result["commitment"]

    assert_true(n_tokens > 0, f"generated {n_tokens} tokens")
    assert_true("merkle_root" in commitment, "commitment has merkle_root")
    assert_true(commitment.get("version") == "V4", f"commitment version is V4 (got {commitment.get('version')})")
    print(f"  request_id: {request_id}")
    print(f"  n_tokens: {n_tokens}")
    print(f"  generated: {result['generated_text'][:80]}...")

    # ── Step 2: Generate verifier key ──
    print("\n2. Generate verifier key...")
    seed = hashlib.sha256(PROMPT.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)
    key = json.loads(key_json)
    assert_true(key.get("weight_hash") is not None, "key has weight_hash")
    print(f"  weight_hash: {key.get('weight_hash', '')[:16]}...")

    # ── Step 3a: Audit (full tier — all layers) ──
    print("\n3a. Audit (full tier — all layers, token 0)...")
    full_layers = list(range(n_layers))
    audit_full_json = server.audit(
        request_id=request_id,
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=False,
    )
    audit_full = json.loads(audit_full_json) if isinstance(audit_full_json, str) else audit_full_json
    shell_full = audit_full.get("shell_opening")
    assert_true(shell_full is not None, "full audit has shell_opening")
    opened_full = shell_full.get("layer_indices") if shell_full else None
    if opened_full is None:
        n_opened_full = len(shell_full.get("layers", [])) if shell_full else 0
        assert_true(n_opened_full == n_layers, f"full audit opened {n_opened_full}/{n_layers} layers")
    else:
        assert_true(len(opened_full) == n_layers, f"full audit opened {len(opened_full)}/{n_layers} layers")
    assert_true(
        shell_full is not None and shell_full.get("initial_residual") is not None,
        "full audit has initial_residual (full bridge active)"
    )
    assert_true(
        shell_full is not None and shell_full.get("embedding_proof") is not None,
        "full audit has embedding_proof"
    )
    payload_full_bytes = len(audit_full_json.encode("utf-8")) if isinstance(audit_full_json, str) else len(audit_full_json)
    print(f"  payload: {payload_full_bytes} bytes")

    # ── Step 3b: Verify full audit ──
    print("\n3b. Verify (full tier)...")
    report_full = verilm_rs.verify_v4(audit_full_json, key_json)
    assert_true(report_full["passed"], f"full verify passed ({report_full['checks_passed']}/{report_full['checks_run']} checks)")
    if not report_full["passed"]:
        print(f"  failures: {report_full['failures']}")
    # Coverage semantics (#12): full audit must report "full" coverage.
    cov_full = report_full.get("coverage", {})
    assert_true(cov_full.get("level") == "full", f"full audit coverage: {cov_full}")
    print(f"  coverage: {cov_full}")
    # Classified failures (#11): empty on pass, but field must exist.
    cf_full = report_full.get("classified_failures", None)
    assert_true(cf_full is not None, "classified_failures field present")
    assert_true(len(cf_full) == 0, f"no classified failures on pass (got {len(cf_full)})")

    # ── Step 4a: Audit (routine tier — contiguous prefix) ──
    # Need a fresh chat since audit consumes state
    print("\n4a. Chat again for routine audit...")
    result2 = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
    request_id2 = result2["request_id"]

    # Derive the routine layer set using build_audit_challenge
    challenge_seed = hashlib.sha256(b"e2e-test-verifier-secret").digest()
    challenge = verilm_rs.build_audit_challenge(
        list(challenge_seed), result2["n_tokens"], n_layers, "routine"
    )
    routine_layers = challenge["layer_indices"]
    routine_token = challenge["token_index"]
    print(f"  routine challenge: token={routine_token}, layers=0..={max(routine_layers)} ({len(routine_layers)} layers)")

    audit_routine_json = server.audit(
        request_id=request_id2,
        token_index=routine_token,
        layer_indices=routine_layers,
        tier="routine",
        binary=False,
    )
    audit_routine = json.loads(audit_routine_json) if isinstance(audit_routine_json, str) else audit_routine_json
    shell_routine = audit_routine.get("shell_opening")
    assert_true(shell_routine is not None, "routine audit has shell_opening")
    opened_routine = shell_routine.get("layer_indices") if shell_routine else None
    if opened_routine is not None:
        assert_true(
            opened_routine == routine_layers,
            f"routine audit opened layers match challenge ({len(opened_routine)} layers)"
        )
    payload_routine_bytes = len(audit_routine_json.encode("utf-8")) if isinstance(audit_routine_json, str) else len(audit_routine_json)
    print(f"  payload: {payload_routine_bytes} bytes (vs {payload_full_bytes} full)")

    # ── Step 4b: Verify routine audit ──
    print("\n4b. Verify (routine tier)...")
    report_routine = verilm_rs.verify_v4(audit_routine_json, key_json)
    assert_true(
        report_routine["passed"],
        f"routine verify passed ({report_routine['checks_passed']}/{report_routine['checks_run']} checks)"
    )
    if not report_routine["passed"]:
        print(f"  failures: {report_routine['failures']}")
    # Coverage semantics (#12): routine audit must report "routine" coverage.
    cov_routine = report_routine.get("coverage", {})
    assert_true(cov_routine.get("level") == "routine", f"routine audit coverage: {cov_routine}")
    assert_true(
        cov_routine.get("layers_checked", 0) < cov_routine.get("layers_total", 0),
        f"routine layers_checked < layers_total: {cov_routine}"
    )
    print(f"  coverage: {cov_routine}")

    # ── Step 5: Binary wire format ──
    print("\n5. Binary wire format (bincode+zstd)...")
    result3 = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
    request_id3 = result3["request_id"]
    audit_binary = server.audit(
        request_id=request_id3,
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )
    assert_true(isinstance(audit_binary, (bytes, memoryview)), "binary audit returns bytes")
    payload_binary_bytes = len(audit_binary)
    compression_ratio = payload_full_bytes / max(payload_binary_bytes, 1)
    print(f"  JSON:   {payload_full_bytes} bytes")
    print(f"  Binary: {payload_binary_bytes} bytes")
    print(f"  Ratio:  {compression_ratio:.1f}x")
    assert_true(
        payload_binary_bytes < payload_full_bytes,
        f"binary ({payload_binary_bytes}) smaller than JSON ({payload_full_bytes})"
    )

    # Verify binary audit
    report_binary = verilm_rs.verify_v4_binary(bytes(audit_binary), key_json)
    assert_true(
        report_binary["passed"],
        f"binary verify passed ({report_binary['checks_passed']}/{report_binary['checks_run']} checks)"
    )
    if not report_binary["passed"]:
        print(f"  failures: {report_binary['failures']}")
    # Coverage + classified_failures must be present on binary path too.
    cov_binary = report_binary.get("coverage", {})
    assert_true(cov_binary.get("level") == "full", f"binary audit coverage: {cov_binary}")
    assert_true(report_binary.get("classified_failures") is not None, "binary path has classified_failures")

    # ── Step 6: Verify that tampering is detected ──
    print("\n6. Tamper detection...")
    tampered = json.loads(audit_full_json)
    if tampered.get("shell_opening") and tampered["shell_opening"].get("layers"):
        layer0 = tampered["shell_opening"]["layers"][0]
        # ShellLayerOpening has attn_out (Vec<i32>), not x_attn
        if "attn_out" in layer0 and layer0["attn_out"]:
            layer0["attn_out"][0] ^= 0x7FFFFFFF
    tampered_json = json.dumps(tampered)
    report_tampered = verilm_rs.verify_v4(tampered_json, key_json)
    assert_true(not report_tampered["passed"], "tampered audit correctly rejected")
    if report_tampered["failures"]:
        print(f"  detected: {report_tampered['failures'][0][:80]}...")
    # Classified failures (#11): tamper should produce structured failure codes.
    cf_tampered = report_tampered.get("classified_failures", [])
    assert_true(len(cf_tampered) > 0, "tamper produces classified_failures")
    for cf in cf_tampered:
        assert_true("code" in cf and "category" in cf and "message" in cf,
            f"classified_failure has code/category/message: {cf}")
    print(f"  classified: {[c['code'] for c in cf_tampered]}")

    # ── Step 7: EOS trailing forward pass trim regression ──
    # When EOS fires before max_tokens, vLLM dispatches one extra decode step.
    # The trim block must trim final_residuals_raw too, or commit crashes with
    # "final_residual count (N+1) != forward pass count (N)".
    # Use a short-answer prompt with high max_tokens to trigger early EOS.
    print("\n7. EOS trailing forward pass trim (final_residual regression)...")
    eos_prompt = "What is 2+2? Answer with just the number."
    eos_result = server.chat(prompt=eos_prompt, max_tokens=256, temperature=0.0)
    eos_n = eos_result["n_tokens"]
    assert_true(eos_n < 256, f"EOS before max_tokens ({eos_n} < 256)")
    assert_true(eos_result.get("commitment") is not None, "EOS trim: commit succeeded")
    # Also verify the commitment is valid.
    eos_rid = eos_result["request_id"]
    eos_audit_json = server.audit(
        request_id=eos_rid,
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=False,
    )
    eos_report = verilm_rs.verify_v4(eos_audit_json, key_json)
    assert_true(eos_report["passed"], f"EOS trim: verify passed ({eos_report['checks_passed']}/{eos_report['checks_run']} checks)")
    if not eos_report["passed"]:
        print(f"  failures: {eos_report['failures']}")

    # ── Summary ──
    print(f"\n{'='*60}")
    if failures:
        print(f"E2E TEST FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("E2E TEST PASSED")
    print(f"{'='*60}")

    return {
        "passed": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "full_payload_bytes": payload_full_bytes,
        "routine_payload_bytes": payload_routine_bytes,
        "binary_payload_bytes": payload_binary_bytes,
        "compression_ratio": compression_ratio,
        "routine_layers": len(routine_layers),
        "full_checks": report_full["checks_run"],
        "routine_checks": report_routine["checks_run"],
    }


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_e2e():
    return _run_e2e()


@app.local_entrypoint()
def main():
    print("VeriLM E2E Test: chat → commit → audit → verify_v4")
    print("=" * 60)
    result = run_e2e.remote()
    if result["passed"]:
        print("\nE2E TEST PASSED")
        print(f"  Tokens: {result['n_tokens']}, Layers: {result['n_layers']}")
        print(f"  Full: {result['full_checks']} checks, {result['full_payload_bytes']} bytes (JSON)")
        print(f"  Routine: {result['routine_checks']} checks, {result['routine_payload_bytes']} bytes ({result['routine_layers']}/{result['n_layers']} layers)")
        print(f"  Binary: {result['binary_payload_bytes']} bytes ({result['compression_ratio']:.1f}x vs JSON)")
    else:
        print(f"\nE2E TEST FAILED — {result['n_failures']} failure(s):")
        for f in result["failures"]:
            print(f"  - {f}")
        raise SystemExit(1)
