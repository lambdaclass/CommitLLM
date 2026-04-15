"""
Llama 3.1 8B W8A8 — E2E protocol test.

Exercises the full commit-and-audit pipeline:
  chat → commit → keygen → audit → verify → tamper detection

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
        ".git", "target", "scripts/__pycache__", "*.pdf", "site",
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
    seed = hashlib.sha256(b"test-llama-e2e").digest()
    t0 = time.time()
    key_bin, artifact_bin = verilm_rs.generate_key_binary(model_dir, seed)
    print(f"  {len(key_bin)/1024/1024:.1f} MB, {(time.time()-t0)*1000:.0f} ms")

    # Inspect profile to assert correct verification mode
    key_json = verilm_rs.generate_key(model_dir, seed)
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

    # ── Test 1: Full-tier verify ──
    print("Test 1: Full-tier verify")
    result = server.chat(prompt="What causes rainbows?", max_tokens=64, temperature=0.0)
    commitment = result["commitment"]
    n_gen = result["n_tokens"] - commitment.get("n_prompt_tokens", 0)
    print(f"  generated {n_gen} tokens")
    print(f"  text: {result['generated_text'][:80]}...")

    check(commitment.get("version") == "V4", "commitment is V4")
    check(len(commitment.get("kv_roots", [])) == n_layers, "KV roots cover all layers")

    audit = server.audit(
        request_id=result["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )
    report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
    check(report["passed"], f"full verify: {report['checks_passed']}/{report['checks_run']} checks")
    if not report["passed"]:
        for f in report.get("failures", [])[:5]:
            print(f"    {f}")
    print()

    # ── Test 2: Routine-tier verify ──
    print("Test 2: Routine-tier verify")
    result2 = server.chat(prompt="What is a CPU pipeline?", max_tokens=64, temperature=0.0)
    challenge_seed = hashlib.sha256(b"test-llama-routine").digest()
    challenge = verilm_rs.build_audit_challenge(
        list(challenge_seed), result2["n_tokens"], n_layers, "routine"
    )
    routine_layers = challenge["layer_indices"]
    routine_token = challenge["token_index"]
    print(f"  challenge: token={routine_token}, {len(routine_layers)}/{n_layers} layers")

    audit2 = server.audit(
        request_id=result2["request_id"],
        token_index=routine_token,
        layer_indices=routine_layers,
        tier="routine",
        binary=True,
    )
    report2 = verilm_rs.verify_v4_full_binary(bytes(audit2), key_bin, artifact_bin)
    check(report2["passed"], f"routine verify: {report2['checks_passed']}/{report2['checks_run']} checks")
    if not report2["passed"]:
        for f in report2.get("failures", [])[:5]:
            print(f"    {f}")

    full_kb = len(audit) / 1024
    routine_kb = len(audit2) / 1024
    print(f"  payload: {routine_kb:.0f} KB routine vs {full_kb:.0f} KB full")
    print()

    # ── Test 3: Tamper detection ──
    print("Test 3: Tamper detection")
    result3 = server.chat(prompt="What is gravity?", max_tokens=32, temperature=0.0)
    audit3 = server.audit(
        request_id=result3["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )
    report_honest = verilm_rs.verify_v4_full_binary(bytes(audit3), key_bin, artifact_bin)
    check(report_honest["passed"], "honest audit passes")

    tampered = bytearray(audit3)
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

    eos_audit = server.audit(
        request_id=eos["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )
    eos_report = verilm_rs.verify_v4_full_binary(bytes(eos_audit), key_bin, artifact_bin)
    check(eos_report["passed"], f"EOS verify: {eos_report['checks_passed']}/{eos_report['checks_run']} checks")
    print()

    # ── Test 5: Multiple token positions ──
    print("Test 5: Verify at different token positions")
    result5 = server.chat(
        prompt="Write a short poem about the ocean.",
        max_tokens=128, temperature=0.0,
    )
    n5 = result5["n_tokens"]
    n5_prompt = result5["commitment"].get("n_prompt_tokens", 0)
    n5_gen = n5 - n5_prompt
    test_positions = [0]
    if n5_gen > 4:
        test_positions.append(n5_prompt + n5_gen // 2)
    if n5_gen > 2:
        test_positions.append(n5_prompt + n5_gen - 1)

    for tok_idx in test_positions:
        a = server.audit(
            request_id=result5["request_id"],
            token_index=tok_idx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        r = verilm_rs.verify_v4_full_binary(bytes(a), key_bin, artifact_bin)
        check(r["passed"], f"token {tok_idx}: {r['checks_passed']}/{r['checks_run']} checks")
        if not r["passed"]:
            for f in r.get("failures", [])[:3]:
                print(f"    {f}")
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
