"""
Llama sampled decode A/B diagnostic.

Compares two decode verification paths on the same sampled generations:
  A) ExactTokenIdentity (i32 quantized logits → sample) — default Llama profile
  B) LpHiddenBf16 (captured hidden × bf16 lm_head → sample) — Qwen-style profile

For each generation, audits one random generated token and checks whether
each path reproduces the claimed token.

Fixed prompt set, fixed sampled params, many runs.

Usage:
    modal run --detach scripts/modal/tests/llama/diag_sampled_ab.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("commitllm-diag-llama-sampled-ab")

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
    "Write a haiku about cryptography.",
    "Name three benefits of open-source software.",
    "What is the difference between TCP and UDP?",
    "Describe how neural networks learn from data.",
]

N_RUNS = 30  # runs per prompt


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

    verifier_secret = hashlib.sha256(b"diag-llama-ab-secret").digest()

    # ── Key A: default Llama profile (ExactTokenIdentity) ──
    print("Generating key A (ExactTokenIdentity)...")
    t0 = time.time()
    key_a, art_a = verilm_rs.generate_key_binary(model_dir, verifier_secret)
    print(f"  key A: {len(key_a)/1024/1024:.1f} MB, {(time.time()-t0)*1000:.0f}ms")
    meta_a = verilm_rs.inspect_key_binary(key_a)
    print(f"  decode: {meta_a['verification_profile']['decode_acceptance']}")

    # ── Key B: Llama with LP-hidden bf16 profile ──
    print("Generating key B (LpHiddenBf16)...")
    t0 = time.time()
    key_b, art_b = verilm_rs.generate_key_binary_with_profile(
        model_dir, verifier_secret, "llama-w8a8-lp-hidden"
    )
    print(f"  key B: {len(key_b)/1024/1024:.1f} MB, {(time.time()-t0)*1000:.0f}ms")
    if art_b:
        print(f"  decode artifact: {len(art_b)/1024/1024:.1f} MB")
    meta_b = verilm_rs.inspect_key_binary(key_b)
    print(f"  decode: {meta_b['verification_profile']['decode_acceptance']}")

    # ── Challenge helper ──
    def derive_challenge_seed(commitment):
        merkle_root = commitment.get("merkle_root", "")
        io_root = commitment.get("io_root", "")
        return hashlib.sha256(
            f"{merkle_root}:{io_root}".encode() + verifier_secret
        ).digest()

    def build_challenge(commitment, n_tokens):
        challenge_seed = derive_challenge_seed(commitment)
        gen_start = max(commitment.get("n_prompt_tokens", 1) - 1, 0)
        for counter in range(64):
            seed = challenge_seed if counter == 0 else hashlib.sha256(
                challenge_seed + counter.to_bytes(4, "little")
            ).digest()
            challenge = verilm_rs.build_audit_challenge(
                list(seed), n_tokens, n_layers, "full"
            )
            if challenge["token_index"] >= gen_start:
                return challenge
        raise RuntimeError("could not derive generated-token challenge")

    # ── Warmup ──
    print("\nWarmup...")
    for _ in range(2):
        server.chat(prompt=PROMPTS[0], max_tokens=16, temperature=0.8, top_k=50, top_p=0.9)

    # ── A/B comparison ──
    print(f"\n{'='*70}")
    print(f"A/B DIAGNOSTIC: {N_RUNS} runs × {len(PROMPTS)} prompts = {N_RUNS*len(PROMPTS)} total")
    print(f"  Path A: ExactTokenIdentity (i32)")
    print(f"  Path B: LpHiddenBf16 (captured hidden × bf16 lm_head)")
    print(f"  Sampled: T=0.8, top_k=50, top_p=0.9")
    print(f"{'='*70}\n")

    a_pass_total = 0
    a_fail_total = 0
    b_pass_total = 0
    b_fail_total = 0

    for pi, prompt in enumerate(PROMPTS):
        a_pass = 0
        a_fail = 0
        b_pass = 0
        b_fail = 0

        for ri in range(N_RUNS):
            result = server.chat(
                prompt=prompt, max_tokens=32,
                temperature=0.8, top_k=50, top_p=0.9,
            )
            commitment = result["commitment"]
            n_prompt = commitment.get("n_prompt_tokens", 0)
            gen_start = max(n_prompt - 1, 0)

            challenge = build_challenge(commitment, result["n_tokens"])
            tok_idx = challenge["token_index"]
            gen_index = tok_idx - gen_start

            # Get binary audit
            audit = server.audit(
                request_id=result["request_id"],
                token_index=tok_idx,
                layer_indices=list(range(n_layers)),
                tier="full",
                binary=True,
                include_kv=False,
            )

            # Path A: ExactTokenIdentity
            report_a = verilm_rs.verify_v4_full_binary(bytes(audit), key_a, art_a)
            a_ok = report_a["passed"]

            # Path B: LpHiddenBf16
            report_b = verilm_rs.verify_v4_full_binary(bytes(audit), key_b, art_b)
            b_ok = report_b["passed"]

            if a_ok:
                a_pass += 1
            else:
                a_fail += 1
            if b_ok:
                b_pass += 1
            else:
                b_fail += 1

            # Print failures
            if not a_ok or not b_ok:
                a_status = "PASS" if a_ok else "FAIL"
                b_status = "PASS" if b_ok else "FAIL"
                a_msg = report_a.get("failures", []) if not a_ok else []
                b_msg = report_b.get("failures", []) if not b_ok else []
                print(f"  [{pi}][{ri}] gen_idx={gen_index} A={a_status} B={b_status}"
                      + (f" A:{a_msg}" if a_msg else "")
                      + (f" B:{b_msg}" if b_msg else ""))

        a_pass_total += a_pass
        a_fail_total += a_fail
        b_pass_total += b_pass
        b_fail_total += b_fail

        a_n = a_pass + a_fail
        b_n = b_pass + b_fail
        print(f"  prompt {pi}: A={a_pass}/{a_n} ({a_pass/a_n*100:.0f}%) "
              f"B={b_pass}/{b_n} ({b_pass/b_n*100:.0f}%)")

    # ── Summary ──
    a_total = a_pass_total + a_fail_total
    b_total = b_pass_total + b_fail_total
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total runs: {a_total}")
    print(f"  Path A (i32 ExactTokenIdentity): {a_pass_total}/{a_total} ({a_pass_total/a_total*100:.1f}%)")
    print(f"  Path B (LpHiddenBf16):           {b_pass_total}/{b_total} ({b_pass_total/b_total*100:.1f}%)")
    print(f"  Path A fail rate: {a_fail_total/a_total*100:.1f}%")
    print(f"  Path B fail rate: {b_fail_total/b_total*100:.1f}%")

    return {
        "a_pass": a_pass_total, "a_fail": a_fail_total,
        "b_pass": b_pass_total, "b_fail": b_fail_total,
        "total": a_total,
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_diag():
    return _run()


@app.local_entrypoint()
def main():
    print("CommitLLM Llama Sampled A/B Diagnostic")
    print("=" * 60)
    result = run_diag.remote()
    print(f"\nResult: {result}")
