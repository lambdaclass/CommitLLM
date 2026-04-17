"""
Llama sampled decode diagnostic — investigate benchmark failures.

Captures per-failure:
  - n_prompt, gen_start, token_index, gen_index
  - temperature / top_k / top_p
  - claimed vs expected token_id
  - failure message from verifier
  - determinism check (replay same case 5x)

Usage:
    modal run --detach scripts/modal/tests/llama/diag_sampled.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("commitllm-diag-llama-sampled")

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

PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
    "What is the significance of the Turing test?",
    "How does public key cryptography work?",
    "Write a short function in Python that computes Fibonacci numbers.",
    "What are the key principles of operating system design?",
    "Explain how hash tables work and why they are useful.",
]

# Also include the E2E test prompts for comparison
E2E_SAMPLED_PROMPTS = [
    "Write a haiku about cryptography.",
    "Name three benefits of open-source software.",
    "What is the difference between TCP and UDP?",
]


def _run():
    import hashlib
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"\nLoading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    print(f"  {n_layers} layers\n")

    # Keygen
    verifier_secret = hashlib.sha256(b"diag-llama-sampled-secret").digest()
    key_bin, artifact_bin = verilm_rs.generate_key_binary(model_dir, verifier_secret)
    key_meta = verilm_rs.inspect_key_binary(key_bin)
    profile = key_meta.get("verification_profile", {})
    print(f"  profile: {profile.get('name')}, decode: {profile.get('decode_acceptance')}\n")

    def derive_challenge_seed(commitment):
        merkle_root = commitment.get("merkle_root", "")
        io_root = commitment.get("io_root", "")
        return hashlib.sha256(
            f"{merkle_root}:{io_root}".encode() + verifier_secret
        ).digest()

    def build_challenge(commitment, n_tokens, tier, *, generated_only=True):
        challenge_seed = derive_challenge_seed(commitment)
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

    # ── Warmup ──
    print("Warmup...")
    for _ in range(2):
        server.chat(prompt=PROMPTS[0], max_tokens=16, temperature=0.8, top_k=50, top_p=0.9)
    print()

    # ── Test 1: Run all benchmark prompts with sampled decode ──
    print("=" * 70)
    print("TEST 1: All benchmark prompts, sampled, full tier")
    print("=" * 70)
    failures = []

    for i, prompt in enumerate(PROMPTS):
        result = server.chat(
            prompt=prompt, max_tokens=32,
            temperature=0.8, top_k=50, top_p=0.9,
        )
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = len(result.get("token_ids", [])) - n_prompt
        gen_start = max(n_prompt - 1, 0)

        challenge = build_challenge(commitment, result["n_tokens"], "full")
        tok_idx = challenge["token_index"]
        gen_index = tok_idx - gen_start

        audit = server.audit(
            request_id=result["request_id"],
            token_index=tok_idx,
            layer_indices=list(range(n_layers)),
            tier="full",
            binary=True,
            include_kv=False,
        )

        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        status = "PASS" if report["passed"] else "FAIL"

        print(f"\n  [{i}] prompt={repr(prompt[:50])}...")
        print(f"      n_prompt={n_prompt} gen_start={gen_start} n_gen={n_gen}")
        print(f"      token_index={tok_idx} gen_index={gen_index}")
        print(f"      checks={report['checks_passed']}/{report['checks_run']} {status}")

        if not report["passed"]:
            print(f"      FAILURES: {report.get('failures', [])}")
            classified = report.get("classified_failures", [])
            for cf in classified:
                print(f"        code={cf.get('code')} cat={cf.get('category')} msg={cf.get('message')}")
                ctx = cf.get("context", {})
                if ctx:
                    for k, v in ctx.items():
                        print(f"          {k}={v}")
            failures.append({
                "prompt_idx": i,
                "n_prompt": n_prompt,
                "gen_start": gen_start,
                "token_index": tok_idx,
                "gen_index": gen_index,
                "failures": report.get("failures", []),
                "classified": classified,
            })

    # ── Test 2: E2E sampled prompts for comparison ──
    print("\n" + "=" * 70)
    print("TEST 2: E2E sampled prompts (same ones that pass in E2E test)")
    print("=" * 70)

    for i, prompt in enumerate(E2E_SAMPLED_PROMPTS):
        result = server.chat(
            prompt=prompt, max_tokens=64,
            temperature=0.8, top_k=50, top_p=0.9,
        )
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = len(result.get("token_ids", [])) - n_prompt
        gen_start = max(n_prompt - 1, 0)

        challenge = build_challenge(commitment, result["n_tokens"], "full")
        tok_idx = challenge["token_index"]
        gen_index = tok_idx - gen_start

        audit = server.audit(
            request_id=result["request_id"],
            token_index=tok_idx,
            layer_indices=list(range(n_layers)),
            tier="full",
            binary=True,
            include_kv=False,
        )

        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        status = "PASS" if report["passed"] else "FAIL"

        print(f"\n  [{i}] prompt={repr(prompt[:50])}")
        print(f"      n_prompt={n_prompt} gen_start={gen_start} n_gen={n_gen}")
        print(f"      token_index={tok_idx} gen_index={gen_index}")
        print(f"      checks={report['checks_passed']}/{report['checks_run']} {status}")
        if not report["passed"]:
            print(f"      FAILURES: {report.get('failures', [])}")

    # ── Test 3: Determinism check — pick first benchmark prompt, replay 10x ──
    print("\n" + "=" * 70)
    print("TEST 3: Determinism — same prompt, 10 sampled generations, full tier")
    print("=" * 70)

    test_prompt = PROMPTS[0]
    pass_count = 0
    fail_count = 0

    for i in range(10):
        result = server.chat(
            prompt=test_prompt, max_tokens=32,
            temperature=0.8, top_k=50, top_p=0.9,
        )
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        gen_start = max(n_prompt - 1, 0)

        challenge = build_challenge(commitment, result["n_tokens"], "full")
        tok_idx = challenge["token_index"]
        gen_index = tok_idx - gen_start

        audit = server.audit(
            request_id=result["request_id"],
            token_index=tok_idx,
            layer_indices=list(range(n_layers)),
            tier="full",
            binary=True,
            include_kv=False,
        )

        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        status = "PASS" if report["passed"] else "FAIL"
        if report["passed"]:
            pass_count += 1
        else:
            fail_count += 1

        print(f"  [{i}] n_prompt={n_prompt} tok_idx={tok_idx} gen_idx={gen_index} "
              f"{report['checks_passed']}/{report['checks_run']} {status}"
              + (f" {report.get('failures', [])}" if not report["passed"] else ""))

    print(f"\n  Determinism: {pass_count} pass, {fail_count} fail out of 10")

    # ── Test 4: Sweep token positions on one generation ──
    print("\n" + "=" * 70)
    print("TEST 4: Sweep token positions on one sampled generation")
    print("=" * 70)

    result = server.chat(
        prompt=PROMPTS[0], max_tokens=32,
        temperature=0.8, top_k=50, top_p=0.9,
    )
    commitment = result["commitment"]
    n_prompt = commitment.get("n_prompt_tokens", 0)
    n_tokens = result["n_tokens"]
    gen_start = max(n_prompt - 1, 0)
    token_ids = result.get("token_ids", [])
    print(f"  n_prompt={n_prompt} n_tokens={n_tokens} gen_start={gen_start}")
    print(f"  token_ids[gen_start:gen_start+5]={token_ids[gen_start:gen_start+5]}")

    for tok_idx in range(gen_start, min(n_tokens, gen_start + 10)):
        gen_index = tok_idx - gen_start
        audit = server.audit(
            request_id=result["request_id"],
            token_index=tok_idx,
            layer_indices=list(range(n_layers)),
            tier="full",
            binary=True,
            include_kv=False,
        )
        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        status = "PASS" if report["passed"] else "FAIL"
        print(f"  tok_idx={tok_idx} gen_idx={gen_index} "
              f"{report['checks_passed']}/{report['checks_run']} {status}"
              + (f" {report.get('failures', [])}" if not report["passed"] else ""))

    # ── Test 5: Greedy sanity check ──
    print("\n" + "=" * 70)
    print("TEST 5: Greedy sanity (should all pass)")
    print("=" * 70)

    for i, prompt in enumerate(PROMPTS[:4]):
        result = server.chat(prompt=prompt, max_tokens=32, temperature=0.0)
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        gen_start = max(n_prompt - 1, 0)

        challenge = build_challenge(commitment, result["n_tokens"], "full")
        tok_idx = challenge["token_index"]
        gen_index = tok_idx - gen_start

        audit = server.audit(
            request_id=result["request_id"],
            token_index=tok_idx,
            layer_indices=list(range(n_layers)),
            tier="full",
            binary=True,
            include_kv=False,
        )
        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        status = "PASS" if report["passed"] else "FAIL"
        print(f"  [{i}] greedy tok_idx={tok_idx} gen_idx={gen_index} "
              f"{report['checks_passed']}/{report['checks_run']} {status}"
              + (f" {report.get('failures', [])}" if not report["passed"] else ""))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Test 1 failures: {len(failures)}/{len(PROMPTS)}")
    if failures:
        gen_indices = [f["gen_index"] for f in failures]
        print(f"  Failing gen_indices: {gen_indices}")
        print(f"  Failing token_indices: {[f['token_index'] for f in failures]}")
        print(f"  gen_index == 0 cluster: {sum(1 for g in gen_indices if g == 0)}/{len(gen_indices)}")
        print(f"  gen_index == 1 cluster: {sum(1 for g in gen_indices if g == 1)}/{len(gen_indices)}")

    return failures


@app.function(image=image, gpu="A100-80GB", timeout=1200)
def run_diag():
    return _run()


@app.local_entrypoint()
def main():
    print("CommitLLM Llama Sampled Decode Diagnostic")
    print("=" * 60)
    failures = run_diag.remote()
    print(f"\nTotal failures: {len(failures)}")
