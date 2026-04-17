"""
Qwen 2.5 7B W8A8 — E2E protocol regression test.

Exercises the supported Qwen path the way a real verifier would:
  - Greedy and sampled decoding through LpHiddenBf16 acceptance
  - Challenge seeds derived from commitment (not hardcoded)
  - Mix of routine and full audits
  - Random generated-token positions selected by the challenge
  - Longer generations and longer prompts
  - Multi-position escalation on one request
  - Tamper detection (bit-flip -> verifier rejects)
  - EOS early-stop handling
  - Payload size and verification timing

This test intentionally does NOT claim Qwen attention strong-tier
verification. The witnessed-score f64 softmax*V replay breached the proposed
fixed tolerance in the 39-prompt sweep (global max_diff=9), so score witnessing
stays disabled here. Attention is tracked separately under the kernel-aligned
witness / deterministic attention-kernel roadmap.

Usage:
    modal run --detach scripts/modal/tests/qwen/test_e2e.py
"""

import sys, os

# _pins lives two directories up
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("commitllm-test-qwen-e2e")

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
        # Score witnesses are intentionally disabled until the Qwen attention
        # path is replaced by a kernel-aligned witness or deterministic kernel.
        # "VERILM_SCORE_WITNESS": "1",
    })
    .pip_install(*VERIFICATION)
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "**/__pycache__", "*.pyc", "*.pdf", "*.md", "site",
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

    # -- Load model --
    print(f"\nLoading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))
    print(f"  {n_layers} layers")

    buf = cap.get_capture_buffer()
    score_witness_enabled = getattr(buf, "_sw_enabled", False)
    print(f"  score witness: {'enabled' if score_witness_enabled else 'disabled'}")
    check(not score_witness_enabled, "score witnessing disabled for Qwen regression test")
    print()

    # -- Keygen --
    print("Generating verifier key...")
    verifier_secret = hashlib.sha256(b"test-qwen-e2e-verifier-secret").digest()
    t0 = time.time()
    key_bin, artifact_bin = verilm_rs.generate_key_binary(model_dir, verifier_secret)
    keygen_ms = (time.time() - t0) * 1000
    print(f"  key={len(key_bin)/1024/1024:.1f} MB, artifact="
          f"{(len(artifact_bin)/1024/1024 if artifact_bin else 0):.1f} MB, "
          f"{keygen_ms:.0f} ms")
    check(artifact_bin is not None, "decode artifact present for LpHiddenBf16")

    key_meta = verilm_rs.inspect_key_binary(key_bin)
    profile = key_meta.get("verification_profile", {})
    attn_mode = profile.get("attention_mode", "unknown")
    decode_mode = profile.get("decode_acceptance", "unknown")
    print(f"  profile: {profile.get('name', 'unknown')}")
    print(f"  attention_mode: {attn_mode}")
    print(f"  decode_acceptance: {decode_mode}")
    check(attn_mode == "WitnessedScores", f"attention mode is WitnessedScores (got {attn_mode})")
    check(decode_mode == "LpHiddenBf16", f"decode acceptance is LpHiddenBf16 (got {decode_mode})")
    print()

    # -- Helpers --
    def derive_test_challenge_seed(commitment, verifier_secret):
        """Test helper. Production should hash the canonical receipt bytes."""
        merkle_root = commitment.get("merkle_root", "")
        io_root = commitment.get("io_root", "")
        return hashlib.sha256(
            f"{merkle_root}:{io_root}".encode() + verifier_secret
        ).digest()

    def build_test_challenge(commitment, n_tokens, tier, *, generated_only=True):
        challenge_seed = derive_test_challenge_seed(commitment, verifier_secret)
        gen_start = max(commitment.get("n_prompt_tokens", 1) - 1, 0)
        for counter in range(64):
            seed = challenge_seed if counter == 0 else hashlib.sha256(
                challenge_seed + counter.to_bytes(4, "little")
            ).digest()
            challenge = verilm_rs.build_audit_challenge(
                list(seed), n_tokens, n_layers, tier
            )
            if not generated_only or challenge["token_index"] >= gen_start:
                return challenge
        raise RuntimeError("could not derive generated-token challenge")

    def check_qwen_report(report, label, *, expected_tier=None, require_qkv_skip=False):
        check(report["passed"], f"{label}: {report['checks_passed']}/{report['checks_run']} checks")
        if not report["passed"]:
            for f in report.get("failures", [])[:3]:
                print(f"    {f}")

        if expected_tier is not None:
            coverage = report.get("coverage", {})
            actual_level = coverage.get("level")
            # "full" is always acceptable — it means more coverage than requested.
            tier_ok = actual_level == expected_tier or actual_level == "full"
            check(tier_ok, f"{label}: coverage level is {expected_tier} (got {actual_level})")

        skipped = report.get("skipped", [])
        lp_skipped = [
            s for s in skipped
            if "lp_hidden" in s.lower() or "lm_head token identity" in s.lower()
        ]
        check(len(lp_skipped) == 0, f"{label}: LP-hidden decode check ran")
        if require_qkv_skip:
            qkv_skipped = any("Wq/Wk/Wv Freivalds" in s for s in skipped)
            check(qkv_skipped, f"{label}: QKV Freivalds explicitly skipped by Qwen profile")

    def audit_and_verify(
        prompt,
        max_tokens,
        temperature,
        tier,
        label,
        *,
        top_k=0,
        top_p=1.0,
        min_tokens=0,
        ignore_eos=False,
        include_kv=False,
        require_qkv_skip=False,
    ):
        t_chat = time.time()
        result = server.chat(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_tokens=min_tokens,
            ignore_eos=ignore_eos,
        )
        chat_ms = (time.time() - t_chat) * 1000
        commitment = result["commitment"]
        n_gen = len(result.get("token_ids", [])) - commitment.get("n_prompt_tokens", 0)

        check(commitment.get("version") == "V4", f"{label}: commitment is V4")
        check(len(commitment.get("kv_roots", [])) == n_layers, f"{label}: KV roots cover all layers")

        challenge = build_test_challenge(
            commitment,
            result["n_tokens"],
            tier,
            generated_only=True,
        )
        tok_idx = challenge["token_index"]
        layers = challenge["layer_indices"]

        t_audit = time.time()
        audit = server.audit(
            request_id=result["request_id"],
            token_index=tok_idx,
            layer_indices=layers,
            tier=tier,
            binary=True,
            include_kv=include_kv,
        )
        audit_ms = (time.time() - t_audit) * 1000
        audit_kb = len(audit) / 1024

        t_verify = time.time()
        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        verify_ms = (time.time() - t_verify) * 1000

        if temperature > 0:
            mode = f"T={temperature}, top_k={top_k}, top_p={top_p}"
        else:
            mode = "greedy"
        tier_str = f"{tier} ({len(layers)}/{n_layers} layers)"
        kv_str = "kv=on" if include_kv else "kv=off"
        print(f"  {label}: {n_gen} tok, {mode}, {tier_str}, "
              f"{kv_str}, token={tok_idx}, {audit_kb:.0f} KB, "
              f"chat={chat_ms:.0f}ms audit={audit_ms:.0f}ms verify={verify_ms:.0f}ms")

        check_qwen_report(
            report,
            label,
            expected_tier=tier,
            require_qkv_skip=require_qkv_skip,
        )
        return result, report, audit

    # -- Test 1: Greedy batch with mixed tiers --
    greedy_prompts = [
        ("What causes rainbows?", 64),
        ("Explain how a CPU pipeline works.", 96),
        ("What is the difference between TCP and UDP?", 80),
        ("Why is the sky blue?", 64),
    ]
    print(f"Test 1: Greedy batch ({len(greedy_prompts)} requests, mixed tiers)")
    for i, (prompt, max_tok) in enumerate(greedy_prompts):
        tier = "full" if (i % 3 == 0) else "routine"
        audit_and_verify(
            prompt,
            max_tok,
            0.0,
            tier,
            f"req {i}",
            require_qkv_skip=(i == 0),
        )
    print()

    # -- Test 2: Sampled decoding --
    sampled_prompts = [
        ("Write a creative story about a robot learning to paint.", 96, 0.8, 50, 0.9),
        ("Invent a new recipe using only five ingredients.", 80, 0.7, 40, 0.95),
        ("Describe an alien civilization in three sentences.", 64, 0.9, 0, 0.9),
    ]
    print(f"Test 2: Sampled decoding ({len(sampled_prompts)} requests)")
    for i, (prompt, max_tok, temp, top_k, top_p) in enumerate(sampled_prompts):
        tier = "full" if i == 0 else "routine"
        audit_and_verify(
            prompt,
            max_tok,
            temp,
            tier,
            f"sampled {i}",
            top_k=top_k,
            top_p=top_p,
        )
    print()

    # -- Test 3: Longer generation --
    print("Test 3: Longer generation")
    audit_and_verify(
        "Write a detailed explanation of how public-key cryptography works, "
        "including RSA key generation, encryption, decryption, and why it is secure.",
        256,
        0.0,
        "full",
        "long-gen",
        min_tokens=64,
    )
    print()

    # -- Test 4: Longer prompt --
    context_block = (
        "You are an expert systems architect. You are evaluating a cryptographic "
        "commit-and-audit protocol for LLM inference verification. The provider "
        "runs inference on GPU, commits retained state with Merkle roots, and opens "
        "challenged token positions after the verifier derives a challenge seed. "
        "The verifier checks shell matrix multiplications with Freivalds, validates "
        "decode with LP-hidden bf16 replay, and treats Qwen attention as an open "
        "problem until a kernel-aligned witness or deterministic attention kernel lands."
    )
    long_prompt = "\n\n".join([
        context_block,
        "Operational constraints: average audit bandwidth must stay low, but the "
        "verifier may escalate suspicious requests to deeper audits. The provider "
        "must not be able to predict which token or layer will open before committing.",
        "Adversarial constraints: a dishonest provider may tamper with sampling "
        "parameters, omit retained state, replay receipts, or attempt to steer hidden "
        "states while preserving local consistency.",
        "Give a concrete production audit policy for this Qwen profile and be explicit "
        "about which claims are proven today and which attention claims remain open.",
    ])
    print("Test 4: Long prompt (~350 tokens input)")
    audit_and_verify(long_prompt, 128, 0.0, "full", "long-prompt")
    print()

    # -- Test 5: Multi-position escalation --
    print("Test 5: Multi-position escalation on one request")
    result_multi = server.chat(
        prompt="Write a short poem about the ocean.",
        max_tokens=128,
        temperature=0.0,
    )
    commitment_multi = result_multi["commitment"]
    gen_start_multi = max(commitment_multi.get("n_prompt_tokens", 1) - 1, 0)
    base_seed = derive_test_challenge_seed(commitment_multi, verifier_secret)
    for j in range(2):
        challenge = None
        for attempt in range(64):
            seed_j = hashlib.sha256(
                base_seed
                + j.to_bytes(4, "little")
                + attempt.to_bytes(4, "little")
            ).digest()
            candidate = verilm_rs.build_audit_challenge(
                list(seed_j), result_multi["n_tokens"], n_layers, "full"
            )
            if candidate["token_index"] >= gen_start_multi:
                challenge = candidate
                break
        if challenge is None:
            raise RuntimeError("could not derive generated-token escalation challenge")
        tok_idx = challenge["token_index"]

        t_audit = time.time()
        audit = server.audit(
            request_id=result_multi["request_id"],
            token_index=tok_idx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
            include_kv=False,
        )
        audit_ms = (time.time() - t_audit) * 1000

        t_verify = time.time()
        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        verify_ms = (time.time() - t_verify) * 1000
        print(f"  challenge {j}: token={tok_idx}, {len(audit)/1024:.0f} KB, "
              f"audit={audit_ms:.0f}ms verify={verify_ms:.0f}ms")
        check_qwen_report(report, f"position {tok_idx}", expected_tier="full")
    print()

    # -- Test 6: Tamper detection --
    print("Test 6: Tamper detection (bit-flip -> verifier rejects)")
    result_tamper = server.chat(prompt="What is gravity?", max_tokens=32, temperature=0.0)
    challenge = build_test_challenge(
        result_tamper["commitment"],
        result_tamper["n_tokens"],
        "full",
        generated_only=True,
    )
    audit_honest = server.audit(
        request_id=result_tamper["request_id"],
        token_index=challenge["token_index"],
        layer_indices=full_layers,
        tier="full",
        binary=True,
        include_kv=False,
    )
    report_honest = verilm_rs.verify_v4_full_binary(bytes(audit_honest), key_bin, artifact_bin)
    check_qwen_report(report_honest, "honest tamper baseline", expected_tier="full")

    tampered = bytearray(audit_honest)
    if len(tampered) > 100:
        tampered[100] ^= 0xFF
    report_tamper = verilm_rs.verify_v4_full_binary(bytes(tampered), key_bin, artifact_bin)
    check(not report_tamper["passed"], "tampered audit rejected")
    print()

    # -- Test 7: EOS early stop --
    print("Test 7: EOS early stop")
    eos = server.chat(prompt="What is 2+2? Just the number.", max_tokens=256, temperature=0.0)
    print(f"  {eos['n_tokens']} audit tokens")

    challenge = build_test_challenge(
        eos["commitment"],
        eos["n_tokens"],
        "full",
        generated_only=True,
    )
    eos_audit = server.audit(
        request_id=eos["request_id"],
        token_index=challenge["token_index"],
        layer_indices=full_layers,
        tier="full",
        binary=True,
        include_kv=False,
    )
    eos_report = verilm_rs.verify_v4_full_binary(bytes(eos_audit), key_bin, artifact_bin)
    check_qwen_report(eos_report, "EOS verify", expected_tier="full")
    print()

    # -- Summary --
    print("=" * 50)
    if failures:
        print(f"FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL TESTS PASSED")
    print("=" * 50)

    return {"passed": len(failures) == 0, "failures": failures}


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run():
    return _run()


@app.local_entrypoint()
def main():
    print("CommitLLM E2E — Qwen 2.5 7B W8A8 (decode/shell regression)")
    print("=" * 50)
    result = run.remote()
    if not result["passed"]:
        raise SystemExit(1)
