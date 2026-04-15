"""
Llama 3.1 8B W8A8 — E2E protocol test.

Exercises the full commit-and-audit pipeline the way a real verifier would:
  - Greedy and sampled decoding (different decode paths + seed commitment)
  - Short and long prompts with varied generation lengths
  - Challenge seeds derived from commitment (not hardcoded)
  - Mix of routine and full audits
  - Random token positions selected by the challenge
  - Multi-position escalation on a single request
  - Tamper detection (bit-flip → verifier rejects)
  - EOS early-stop handling
  - Payload size and verification timing

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
    keygen_ms = (time.time() - t0) * 1000
    print(f"  {len(key_bin)/1024/1024:.1f} MB, {keygen_ms:.0f} ms")

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
    def derive_test_challenge_seed(commitment, verifier_secret):
        """Test helper — derives a challenge seed from commitment fields."""
        merkle_root = commitment.get("merkle_root", "")
        io_root = commitment.get("io_root", "")
        return hashlib.sha256(
            f"{merkle_root}:{io_root}".encode() + verifier_secret
        ).digest()

    # ── Helper: run one audit cycle (chat → commit → challenge → audit → verify) ──
    def audit_and_verify(prompt, max_tokens, temperature, tier, label):
        """Full cycle: generate → derive challenge → audit → verify. Returns report."""
        t_chat = time.time()
        result = server.chat(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        chat_ms = (time.time() - t_chat) * 1000
        commitment = result["commitment"]
        n_gen = result["n_tokens"] - commitment.get("n_prompt_tokens", 0)

        check(commitment.get("version") == "V4", f"{label}: commitment is V4")

        challenge_seed = derive_test_challenge_seed(commitment, verifier_secret)
        challenge = verilm_rs.build_audit_challenge(
            list(challenge_seed), result["n_tokens"], n_layers, tier
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
        )
        audit_ms = (time.time() - t_audit) * 1000
        audit_kb = len(audit) / 1024

        t_verify = time.time()
        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        verify_ms = (time.time() - t_verify) * 1000

        temp_str = f"T={temperature}" if temperature > 0 else "greedy"
        tier_str = f"{tier} ({len(layers)}/{n_layers} layers)"
        print(f"  {label}: {n_gen} tok, {temp_str}, {tier_str}, "
              f"token={tok_idx}, {audit_kb:.0f} KB, "
              f"chat={chat_ms:.0f}ms audit={audit_ms:.0f}ms verify={verify_ms:.0f}ms")

        check(report["passed"],
              f"{label}: {report['checks_passed']}/{report['checks_run']} checks")
        if not report["passed"]:
            for f in report.get("failures", [])[:3]:
                print(f"    {f}")

        return result, report, audit

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
        audit_and_verify(prompt, max_tok, 0.0, tier, f"req {i}")
    print()

    # ── Test 2: Sampled decoding ──
    # Exercises the sampled decode path: seed commitment, probability checks,
    # top_k/top_p filtering. This is a completely different verification path
    # from greedy (ExactTokenIdentity → seed + probability replay).
    sampled_prompts = [
        ("Write a creative story about a robot learning to paint.", 128, 0.8),
        ("Invent a new recipe using only five ingredients.", 96, 0.7),
        ("Describe an alien civilization in three sentences.", 64, 0.9),
    ]

    print(f"Test 2: Sampled decoding ({len(sampled_prompts)} requests)")
    for i, (prompt, max_tok, temp) in enumerate(sampled_prompts):
        tier = "full" if (i == 0) else "routine"
        audit_and_verify(prompt, max_tok, temp, tier, f"sampled {i}")
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
    )
    print()

    # ── Test 4: Long prompt (system prompt + context) ──
    # Real deployments have system prompts + user context that can be 500+ tokens.
    long_prompt = (
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
    print("Test 4: Long prompt (~150 tokens input)")
    audit_and_verify(long_prompt, 128, 0.0, "full", "long-prompt")
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

    for j in range(3):
        seed_j = hashlib.sha256(
            derive_test_challenge_seed(commitment_multi, verifier_secret)
            + j.to_bytes(4, "little")
        ).digest()
        challenge = verilm_rs.build_audit_challenge(
            list(seed_j), result_multi["n_tokens"], n_layers, "full"
        )
        tok_idx = challenge["token_index"]

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

    # ── Test 6: Tamper detection ──
    print("Test 6: Tamper detection (bit-flip → verifier rejects)")
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
    )
    report_honest = verilm_rs.verify_v4_full_binary(bytes(audit_honest), key_bin, artifact_bin)
    check(report_honest["passed"], "honest audit passes")

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
