"""
E2E validation: CapturedLogits decode verification on real GPU.

Exercises the full commit->open->verify path for the CapturedLogits mode:

  1. VerifiedInferenceServer.chat() — capture logits + lp_hidden + commit
  2. verilm_rs.generate_key_binary_with_profile() — captured-logits profile
  3. server.audit() — open V4 with captured_logits_f32 in shell opening
  4. verilm_rs.verify_v4_full_binary() — canonical verifier runs
     phase_lm_head_captured_logits (exact sampling + Freivalds binding)

Tests:
  - Greedy decode (confirm profile routing doesn't regress)
  - Sampled decode with random generated-token challenges
  - Repeated sampled runs on same prompts (intermittent mismatch detection)
  - Tamper rejection
  - Both Llama and Qwen models

Usage:
    modal run --detach scripts/modal/test_e2e_captured_logits.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-test-e2e-captured-logits")

MODELS = {
    "qwen": "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
    "llama": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
}

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
    "Describe a futuristic city in three sentences.",
    "What would happen if gravity suddenly doubled?",
]

# Number of repeated runs per sampled prompt to detect intermittent mismatches.
REPEAT_RUNS = 5


def _run_model(model_name, model_id):
    import hashlib
    import random
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"\nLoading {model_id}...")
    llm = LLM(model=model_id, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir

    decode_failures = []   # CapturedLogits-specific (exact sampling + Freivalds)
    attn_failures = []     # Attention layer mismatches (known gap, pre-existing)
    stats = {
        "model": model_name,
        "greedy_runs": 0,
        "sampled_runs": 0,
        "greedy_passed": 0,
        "sampled_passed": 0,
        "decode_checks_passed": 0,
        "decode_checks_total": 0,
    }

    def assert_true(cond, msg, category="decode"):
        if not cond:
            bucket = attn_failures if category == "attn" else decode_failures
            bucket.append(f"[{model_name}] {msg}")
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK: {msg}")

    def classify_report(report):
        """Separate attention failures from decode (CapturedLogits) failures."""
        attn = [f for f in report.get("failures", []) if "attention mismatch" in f]
        decode = [f for f in report.get("failures", []) if "attention mismatch" not in f]
        decode_passed = report["checks_passed"] + len(attn)  # checks that would pass without attn
        decode_total = report["checks_run"]
        return attn, decode, decode_passed, decode_total

    # -- Step 1: Generate verifier key with captured-logits profile --
    profile_name = f"{model_name}-w8a8-captured-logits"
    print(f"\n1. Generate verifier key (profile={profile_name})...")
    seed = hashlib.sha256(f"captured-logits-e2e-{model_name}".encode()).digest()
    t0 = time.time()
    key_binary, artifact_binary = verilm_rs.generate_key_binary_with_profile(
        model_dir, list(seed), profile_name,
    )
    keygen_ms = (time.time() - t0) * 1000
    print(f"  keygen: {keygen_ms:.0f}ms, key={len(key_binary)} bytes ({len(key_binary)/1024/1024:.1f} MB)")
    if artifact_binary:
        print(f"  decode artifact: {len(artifact_binary)} bytes ({len(artifact_binary)/1024/1024:.1f} MB)")
    assert_true(key_binary[:4] == b"VKEY", f"key binary magic is VKEY")
    assert_true(artifact_binary is not None, "decode artifact present (needed for Freivalds)")

    full_layers = list(range(n_layers))

    # -- Step 2: Greedy decode --
    for pi, prompt in enumerate(GREEDY_PROMPTS):
        print(f"\n{'='*60}")
        print(f"2.{pi} [{model_name}] Greedy E2E — {prompt[:50]}...")
        print(f"{'='*60}")

        result = server.chat(prompt=prompt, max_tokens=64, temperature=0.0)
        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt

        assert_true(n_gen > 0, f"generated {n_gen} tokens")
        print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
        print(f"  text: {result['generated_text'][:80]}...")

        # Check captured logits are present (sampler fires once per fwd:
        # 1 prefill + n_gen decode = n_gen+1, or n_gen if no EOS fwd).
        captured = server.sampler_hook._captured_logits
        assert_true(
            len(captured) in (n_gen, n_gen + 1),
            f"captured logits count: {len(captured)} (expected {n_gen} or {n_gen+1})"
        )

        # Audit first generated token (absolute index = n_prompt - 1).
        gen_start = n_prompt - 1
        stats["greedy_runs"] += 1
        audit_binary = server.audit(
            request_id=request_id,
            token_index=gen_start,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        print(f"  binary payload: {len(audit_binary)} bytes ({len(audit_binary)/1024:.1f} KiB)")

        report = verilm_rs.verify_v4_full_binary(bytes(audit_binary), key_binary, artifact_binary)
        attn_f, decode_f, decode_passed, decode_total = classify_report(report)
        stats["decode_checks_passed"] += decode_passed
        stats["decode_checks_total"] += decode_total

        # CapturedLogits decode checks must all pass; attention failures are tracked separately.
        assert_true(
            len(decode_f) == 0,
            f"greedy decode checks passed ({decode_passed}/{decode_total}, {len(attn_f)} attn failures excluded)"
        )
        if len(decode_f) == 0:
            stats["greedy_passed"] += 1
        for f in decode_f:
            print(f"    DECODE FAILURE: {f}")
        for f in attn_f:
            assert_true(False, f"attention: {f}", category="attn")

        # Check for captured-logits-related skips (exclude QKV/attention skips).
        skipped = report.get("skipped", [])
        logit_skips = [s for s in skipped
                       if "captured logits" in s.lower() and "wq/" not in s.lower()]
        assert_true(len(logit_skips) == 0, f"no captured-logits skips ({logit_skips})")

    # -- Step 3: Sampled decode with random token challenges + repeats --
    rng = random.Random(42)

    for pi, prompt in enumerate(SAMPLED_PROMPTS):
        for run in range(REPEAT_RUNS):
            print(f"\n{'='*60}")
            print(f"3.{pi}.{run} [{model_name}] Sampled E2E — {prompt[:40]}... (run {run+1}/{REPEAT_RUNS})")
            print(f"{'='*60}")

            result = server.chat(
                prompt=prompt, max_tokens=32,
                temperature=0.8, top_k=50, top_p=0.9,
            )
            request_id = result["request_id"]
            n_tokens = result["n_tokens"]
            commitment = result["commitment"]
            n_prompt = commitment.get("n_prompt_tokens", 0)
            n_gen = n_tokens - n_prompt
            gen_token_ids = result["token_ids"][n_prompt:]

            assert_true(n_gen > 0, f"generated {n_gen} tokens")
            print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
            print(f"  tokens: {gen_token_ids[:8]}...")
            print(f"  text: {result['generated_text'][:60]}...")

            # Pick a random generated token to challenge.
            # token_index is absolute into all_retained; gen tokens start at n_prompt-1.
            gen_start = n_prompt - 1
            if n_gen > 1:
                gen_offset = rng.randint(0, n_gen - 1)
            else:
                gen_offset = 0
            challenge_idx = gen_start + gen_offset

            stats["sampled_runs"] += 1
            audit_binary = server.audit(
                request_id=request_id,
                token_index=challenge_idx,
                layer_indices=full_layers,
                tier="full",
                binary=True,
            )
            print(f"  challenge gen_offset={gen_offset}, token_index={challenge_idx}, payload={len(audit_binary)} bytes ({len(audit_binary)/1024:.1f} KiB)")

            report = verilm_rs.verify_v4_full_binary(bytes(audit_binary), key_binary, artifact_binary)
            attn_f, decode_f, decode_passed, decode_total = classify_report(report)
            stats["decode_checks_passed"] += decode_passed
            stats["decode_checks_total"] += decode_total

            # CapturedLogits decode checks must all pass; attention failures tracked separately.
            assert_true(
                len(decode_f) == 0,
                f"sampled decode checks passed ({decode_passed}/{decode_total}, {len(attn_f)} attn failures excluded) token_index={challenge_idx}"
            )
            if len(decode_f) == 0:
                stats["sampled_passed"] += 1
            for f in decode_f:
                print(f"    DECODE FAILURE: {f}")
            for f in attn_f:
                assert_true(False, f"attention: {f}", category="attn")

    # -- Step 4: Tamper rejection --
    print(f"\n{'='*60}")
    print(f"4. [{model_name}] Tamper detection...")
    print(f"{'='*60}")
    result_tamper = server.chat(prompt="What is 1+1?", max_tokens=16, temperature=0.0)
    tamper_n_prompt = result_tamper["commitment"].get("n_prompt_tokens", 0)
    audit_tamper = server.audit(
        request_id=result_tamper["request_id"],
        token_index=tamper_n_prompt - 1,  # first gen token
        layer_indices=full_layers,
        tier="full",
        binary=True,
    )
    tampered = bytearray(audit_tamper)
    if len(tampered) > 100:
        tampered[100] ^= 0xFF
    report_tampered = verilm_rs.verify_v4_full_binary(bytes(tampered), key_binary, artifact_binary)
    assert_true(
        not report_tampered["passed"],
        "tampered audit correctly rejected"
    )

    # -- Summary --
    print(f"\n{'='*60}")
    print(f"[{model_name}] Summary:")
    print(f"  Greedy: {stats['greedy_passed']}/{stats['greedy_runs']} passed (decode checks)")
    print(f"  Sampled: {stats['sampled_passed']}/{stats['sampled_runs']} passed (decode checks)")
    print(f"  Decode checks: {stats['decode_checks_passed']}/{stats['decode_checks_total']}")
    if decode_failures:
        print(f"  DECODE FAILURES ({len(decode_failures)}):")
        for f in decode_failures:
            print(f"    - {f}")
    else:
        print(f"  DECODE: ALL PASSED (CapturedLogits exact sampling + Freivalds)")
    if attn_failures:
        print(f"  ATTENTION FAILURES ({len(attn_failures)}) [known gap, not CapturedLogits]:")
        for f in attn_failures[:5]:
            print(f"    - {f}")
        if len(attn_failures) > 5:
            print(f"    ... and {len(attn_failures) - 5} more")
    print(f"{'='*60}")

    return {
        "model": model_name,
        "passed": len(decode_failures) == 0,  # only decode failures count
        "n_decode_failures": len(decode_failures),
        "n_attn_failures": len(attn_failures),
        "decode_failures": decode_failures,
        "attn_failures": attn_failures,
        "stats": stats,
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_qwen():
    return _run_model("qwen", MODELS["qwen"])


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_llama():
    return _run_model("llama", MODELS["llama"])


@app.local_entrypoint()
def main():
    print("VeriLM E2E: CapturedLogits Decode Verification")
    print("=" * 60)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Greedy prompts: {len(GREEDY_PROMPTS)}")
    print(f"Sampled prompts: {len(SAMPLED_PROMPTS)} x {REPEAT_RUNS} repeats = {len(SAMPLED_PROMPTS) * REPEAT_RUNS} runs")
    print()

    # Run both models in parallel.
    qwen_future = run_qwen.spawn()
    llama_future = run_llama.spawn()

    qwen_result = qwen_future.get()
    llama_result = llama_future.get()

    all_decode_passed = True
    for result in [qwen_result, llama_result]:
        model = result["model"]
        stats = result["stats"]
        print(f"\n{model}: greedy {stats['greedy_passed']}/{stats['greedy_runs']}, "
              f"sampled {stats['sampled_passed']}/{stats['sampled_runs']} (decode checks)")
        print(f"  decode checks: {stats['decode_checks_passed']}/{stats['decode_checks_total']}")
        if result["n_attn_failures"] > 0:
            print(f"  attention failures: {result['n_attn_failures']} [known gap]")
        if not result["passed"]:
            all_decode_passed = False
            for f in result["decode_failures"]:
                print(f"  DECODE FAIL: {f}")

    print()
    if all_decode_passed:
        print("CAPTURED-LOGITS E2E: DECODE PASSED (both models)")
        print("(Attention layer failures are a known pre-existing gap, tracked separately)")
    else:
        total = qwen_result["n_decode_failures"] + llama_result["n_decode_failures"]
        print(f"CAPTURED-LOGITS E2E FAILED — {total} decode failure(s)")
        raise SystemExit(1)
