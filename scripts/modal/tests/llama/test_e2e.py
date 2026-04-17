"""
Llama 3.1 8B W8A8 — E2E protocol test.

Exercises the full commit-and-audit pipeline the way a real verifier would:
  - Greedy and sampled decoding (different decode paths + seed commitment)
  - Short and long prompts with varied generation lengths
  - Challenge seeds derived from commitment (not hardcoded)
  - Mix of routine and full shell/decode audits
  - Random generated-token positions selected by the challenge
  - Token-0 exact-attention smoke check
  - Multi-position escalation on a single request
  - Tamper detection (bit-flip → verifier rejects)
  - EOS early-stop handling
  - Payload size and verification timing

Current Llama stock-kernel status:
  - ExactReplay attention is valid only as a token-0 smoke check
  - Arbitrary-position f64 attention replay diverges from GPU FlashAttention
    and must not be accepted by widening tolerance
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
    keygen_ms = (time.time() - t0) * 1000
    print(f"  {len(key_bin)/1024/1024:.1f} MB, {keygen_ms:.0f} ms")

    # Inspect profile from the binary key. Do not use JSON keygen here:
    # production keys can carry very large matrices.
    key_meta = verilm_rs.inspect_key_binary(key_bin)
    profile = key_meta.get("verification_profile", {})
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
    def derive_test_challenge_seed(commitment, verifier_secret):
        """Test helper — derives a challenge seed from commitment fields."""
        merkle_root = commitment.get("merkle_root", "")
        io_root = commitment.get("io_root", "")
        return hashlib.sha256(
            f"{merkle_root}:{io_root}".encode() + verifier_secret
        ).digest()

    # ── Helper: run one audit cycle (chat → commit → challenge → audit → verify) ──
    def build_test_challenge(commitment, n_tokens, tier, *, generated_only=True, force_token0=False):
        challenge_seed = derive_test_challenge_seed(commitment, verifier_secret)
        if force_token0:
            return verilm_rs.build_audit_challenge(
                list(challenge_seed), 1, n_layers, tier
            )

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
        force_token0=False,
    ):
        """Full cycle: generate → derive challenge → audit → verify. Returns report."""
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
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = len(result.get("token_ids", [])) - n_prompt

        check(commitment.get("version") == "V4", f"{label}: commitment is V4")

        challenge = build_test_challenge(
            commitment,
            result["n_tokens"],
            tier,
            generated_only=not force_token0,
            force_token0=force_token0,
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
            temp_str = f"T={temperature}, top_k={top_k}, top_p={top_p}"
        else:
            temp_str = "greedy"
        tier_str = f"{tier} ({len(layers)}/{n_layers} layers)"
        kv_str = "kv=on" if include_kv else "kv=off"
        print(f"  {label}: {n_gen} tok, {temp_str}, {tier_str}, {kv_str}, "
              f"token={tok_idx}, n_prompt={n_prompt}, {audit_kb:.0f} KB, "
              f"chat={chat_ms:.0f}ms audit={audit_ms:.0f}ms verify={verify_ms:.0f}ms")

        check(report["passed"],
              f"{label}: {report['checks_passed']}/{report['checks_run']} checks")
        coverage = report.get("coverage", {})
        actual_level = coverage.get("level")
        # "full" is always acceptable — routine may randomly select all layers.
        tier_ok = actual_level == tier or actual_level == "full"
        check(tier_ok, f"{label}: coverage level is {tier} (got {actual_level})")
        skipped = report.get("skipped", [])
        if include_kv:
            check(len(skipped) == 0, f"{label}: no verifier phases skipped")
        if not report["passed"]:
            for f in report.get("failures", [])[:3]:
                print(f"    {f}")

        return result, report, audit

    # ── Test 0: token-0 exact-attention smoke ──
    # Non-zero token positions diverge on stock FlashAttention. This test keeps
    # the exact-attention path alive without pretending it is solved generally.
    print("Test 0: Token-0 exact-attention smoke")
    audit_and_verify(
        "What causes rainbows?",
        32,
        0.0,
        "full",
        "attention-token0",
        include_kv=True,
        force_token0=True,
    )
    print()

    # ── Test 1: Mixed batch — greedy decoding, varied lengths ──
    # Mirrors real usage: verifier receives requests, derives challenges,
    # mixes routine and full audits.
    greedy_prompts = [
        ("What causes rainbows?", 64),
        ("Explain how a CPU pipeline works.", 96),
        ("What is the theory of general relativity?", 64),
        ("Write a haiku about cryptography.", 48),
        ("What is the difference between TCP and UDP?", 80),
        ("Why is the sky blue?", 64),
    ]

    print(f"Test 1: Greedy batch ({len(greedy_prompts)} requests, mixed tiers)")
    for i, (prompt, max_tok) in enumerate(greedy_prompts):
        tier = "full" if (i % 3 == 0) else "routine"
        audit_and_verify(prompt, max_tok, 0.0, tier, f"req {i}", include_kv=False)
    print()

    # ── Test 2: Sampled decoding ──
    # Exercises the sampled decode path: seed commitment, probability checks,
    # top_k/top_p filtering. This is a completely different verification path
    # from greedy (ExactTokenIdentity → seed + probability replay).
    sampled_prompts = [
        ("Write a creative story about a robot learning to paint.", 128, 0.8, 50, 0.9),
        ("Invent a new recipe using only five ingredients.", 96, 0.7, 40, 0.95),
        ("Describe an alien civilization in three sentences.", 64, 0.9, 0, 0.9),
    ]

    print(f"Test 2: Sampled decoding ({len(sampled_prompts)} requests)")
    for i, (prompt, max_tok, temp, top_k, top_p) in enumerate(sampled_prompts):
        tier = "full" if (i == 0) else "routine"
        audit_and_verify(
            prompt,
            max_tok,
            temp,
            tier,
            f"sampled {i}",
            top_k=top_k,
            top_p=top_p,
            include_kv=False,
        )
    print()

    # ── Test 3: Long generation ──
    # Real requests often generate 200-500+ tokens. Longer sequences stress
    # KV transcript, prefix state, and the attention replay window.
    print("Test 3: Long generation")
    audit_and_verify(
        "Write a detailed explanation of how public-key cryptography works, "
        "including RSA key generation, encryption, decryption, and why it is "
        "secure. Include mathematical intuition.",
        512, 0.0, "full", "long-gen",
        min_tokens=128,
        include_kv=False,
    )
    print()

    # ── Test 4: Long prompt (system prompt + context) ──
    # Real deployments have system prompts + user context that can be 500+ tokens.
    context_block = (
        "You are an expert systems architect. You have deep knowledge of "
        "distributed systems, cryptographic protocols, consensus mechanisms, "
        "and formal verification. The user is building a cryptographic "
        "commit-and-audit protocol for LLM inference verification. The protocol "
        "uses Merkle commitments over retained state, Freivalds checks for "
        "matrix multiplications, and bounded approximate replay for attention. "
        "The provider runs inference normally on GPU and returns a compact "
        "receipt. When challenged, the provider opens specific token positions "
        "and layer ranges. The verifier checks shell matmuls via Freivalds, "
        "exact bridge tensors by canonical recomputation, and attention by "
        "bounded replay against committed post-attention output. "
        "Given this context, what are the three biggest attack vectors "
        "a dishonest provider could exploit?"
    )
    long_prompt = "\n\n".join([
        context_block,
        "Additional deployment context: requests may include long system prompts, "
        "retrieval-augmented context, tool outputs, and user-provided documents. "
        "The verifier must derive unpredictable challenges after the provider "
        "commits, and the provider must not know which token or layers will open.",
        "Additional adversarial context: the provider may try to tamper with "
        "sampling parameters, omit retained state, replay a receipt from a "
        "different request, or bias attention while keeping the linear shell "
        "internally consistent.",
        "Additional operational context: most requests should receive routine "
        "audits, a smaller fraction should receive full audits, and suspicious "
        "traffic should escalate to multiple token positions on the same request.",
        "Answer with a concrete audit policy and explain the tradeoffs.",
    ])
    print("Test 4: Long prompt (~500 tokens input)")
    audit_and_verify(long_prompt, 128, 0.0, "full", "long-prompt", include_kv=False)
    print()

    # ── Test 5: Multiple positions on same request ──
    # A verifier may challenge the same request at multiple token positions
    # (e.g., deep audit after a routine audit flags something).
    print("Test 5: Multi-position escalation on one request")
    result_multi = server.chat(
        prompt="Write a short poem about the ocean.",
        max_tokens=128, temperature=0.0,
    )
    commitment_multi = result_multi["commitment"]
    gen_start_multi = max(commitment_multi.get("n_prompt_tokens", 1) - 1, 0)
    base_seed_multi = derive_test_challenge_seed(commitment_multi, verifier_secret)

    for j in range(3):
        challenge = None
        for attempt in range(64):
            seed_j = hashlib.sha256(
                base_seed_multi
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

        a = server.audit(
            request_id=result_multi["request_id"],
            token_index=tok_idx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
            include_kv=False,
        )
        audit_kb = len(a) / 1024
        t_verify = time.time()
        r = verilm_rs.verify_v4_full_binary(bytes(a), key_bin, artifact_bin)
        verify_ms = (time.time() - t_verify) * 1000
        print(f"  challenge {j}: token={tok_idx}, {audit_kb:.0f} KB, verify={verify_ms:.0f}ms")
        check(r["passed"], f"position {tok_idx}: {r['checks_passed']}/{r['checks_run']} checks")
        check(r.get("coverage", {}).get("level") == "full", f"position {tok_idx}: coverage level is full")
        check(len(r.get("skipped", [])) == 0, f"position {tok_idx}: no verifier phases skipped")
        if not r["passed"]:
            for f in r.get("failures", [])[:3]:
                print(f"    {f}")
    print()

    # ── Test 6: Tamper detection ──
    print("Test 6: Tamper detection (bit-flip → verifier rejects)")
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
    check(report_honest["passed"], "honest audit passes")
    check(report_honest.get("coverage", {}).get("level") == "full", "honest audit coverage level is full")
    check(len(report_honest.get("skipped", [])) == 0, "honest audit has no skipped phases")

    tampered = bytearray(audit_honest)
    if len(tampered) > 100:
        tampered[100] ^= 0xFF
    report_tamper = verilm_rs.verify_v4_full_binary(bytes(tampered), key_bin, artifact_bin)
    check(not report_tamper["passed"], "tampered audit rejected")
    print()

    # ── Test 7: EOS early stop ──
    print("Test 7: EOS early stop")
    eos = server.chat(prompt="What is 2+2? Just the number.", max_tokens=256, temperature=0.0)
    eos_n = eos["n_tokens"]
    print(f"  {eos_n} tokens (EOS {'early' if eos_n < 256 else 'at limit'})")

    challenge = build_test_challenge(
        eos["commitment"],
        eos_n,
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
    check(eos_report["passed"], f"EOS verify: {eos_report['checks_passed']}/{eos_report['checks_run']} checks")
    check(eos_report.get("coverage", {}).get("level") == "full", "EOS coverage level is full")
    check(len(eos_report.get("skipped", [])) == 0, "EOS audit has no skipped phases")
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


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run():
    return _run()


@app.local_entrypoint()
def main():
    print("CommitLLM E2E — Llama 3.1 8B W8A8")
    print("=" * 50)
    result = run.remote()
    if not result["passed"]:
        raise SystemExit(1)
