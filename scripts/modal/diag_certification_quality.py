"""
Stock-bounded certification quality measurement.

Measures certification rate, escalation rate, and fail-closed rate
across broad random prompts with random generated-token challenges.

For each generated token, computes:
1. Top-k attention evidence (softmax concentration + tail bound)
2. Logit margin from CapturedLogits
3. Certification decision

Reports:
- Certification rate: fraction of tokens that pass statistical certification
- Escalation rate: fraction where attention is too spread (low top-k mass)
- Fail-closed rate: fraction where logit margin is too small
- Per-layer attention concentration statistics

Usage:
    modal run --detach scripts/modal/diag_certification_quality.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-certification-quality")

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
        "VERILM_SCORE_WITNESS": "1",
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

MODELS = {
    "qwen": "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
    "llama": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
}

# Diverse prompts covering different domains and lengths.
PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
    "Tell me a fun fact about space.",
    "What is the capital of France and why is it historically significant?",
    "Explain how neural networks learn from data.",
    "Write a recipe for chocolate chip cookies.",
    "What causes earthquakes and how are they measured?",
    "Describe the water cycle in simple terms.",
    "What are the benefits of regular exercise?",
    "Explain quantum computing to a 10 year old.",
    "Write a short story about a robot learning to paint.",
    "What is photosynthesis and why is it important?",
    "Describe the process of making coffee from bean to cup.",
    "What are the main programming paradigms?",
    "Explain how GPS navigation works.",
    "Write a haiku about autumn.",
    "What is the difference between a virus and a bacterium?",
    "Describe how the internet routes packets.",
]

N_CHALLENGES_PER_REQUEST = 5  # Challenge 5 random tokens per generation
MAX_TOKENS = 64
TOP_K = 16
CONCENTRATION_THRESHOLD = 0.9


def _run_model(model_name, model_id):
    import hashlib
    import json
    import random
    import time

    import numpy as np

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer
    import verilm_rs

    print(f"\n[{model_name}] Loading {model_id}...")
    llm = LLM(model=model_id, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))

    # Generate verifier key.
    seed = hashlib.sha256(f"diag-cert-{model_name}".encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, list(seed))

    # Warmup.
    print(f"[{model_name}] Warmup...")
    server.chat(prompt="Hello", max_tokens=8, temperature=0.0)

    # Measurement.
    print(f"\n{'='*60}")
    print(f"[{model_name}] Certification Quality Measurement")
    print(f"  {len(PROMPTS)} prompts, {N_CHALLENGES_PER_REQUEST} challenges/prompt")
    print(f"  top_k={TOP_K}, threshold={CONCENTRATION_THRESHOLD}")
    print(f"{'='*60}")

    rng = random.Random(42)
    all_certs = []
    all_evidence = []
    total_challenged = 0
    total_certified = 0
    total_low_concentration = 0
    total_low_margin = 0

    for i, prompt in enumerate(PROMPTS):
        result = server.chat(
            prompt=prompt, max_tokens=MAX_TOKENS,
            temperature=0.8, top_k=50, top_p=0.9,
        )
        n_tokens = result["n_tokens"]
        n_prompt = result["commitment"].get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt

        if n_gen < 2:
            print(f"  [{i}] skip: n_gen={n_gen}")
            continue

        # Challenge random decode tokens.
        challenge_offsets = sorted(rng.sample(
            range(n_gen), min(N_CHALLENGES_PER_REQUEST, n_gen)
        ))

        for tok_offset in challenge_offsets:
            challenge_idx = n_prompt - 1 + tok_offset
            total_challenged += 1

            try:
                audit_binary = server.audit(
                    request_id=result["request_id"],
                    token_index=challenge_idx,
                    layer_indices=full_layers,
                    tier="full",
                    binary=True,
                )

                # Get logit margin from captured logits.
                # The sampler hook stores captured logits.
                captured = server.sampler_hook._captured_logits
                if tok_offset < len(captured):
                    logits = np.array(captured[tok_offset], dtype=np.float32)
                    sorted_logits = np.sort(logits)[::-1]
                    logit_margin = float(sorted_logits[0] - sorted_logits[1])
                else:
                    logit_margin = 0.0

                # Compute certification.
                cert_json = verilm_rs.compute_attention_certification(
                    bytes(audit_binary), key_json, logit_margin,
                    TOP_K, CONCENTRATION_THRESHOLD,
                )
                cert = json.loads(cert_json)

                all_certs.append({
                    "prompt_idx": i,
                    "tok_offset": tok_offset,
                    "certified": cert["certified"],
                    "logit_margin": cert["logit_margin"],
                    "min_top_k_mass": cert["min_top_k_mass"],
                    "max_tail_bound": cert["max_tail_bound"],
                })

                if cert["certified"]:
                    total_certified += 1
                elif cert["min_top_k_mass"] < CONCENTRATION_THRESHOLD:
                    total_low_concentration += 1
                elif cert["logit_margin"] <= 0:
                    total_low_margin += 1

            except Exception as e:
                print(f"  [{i}:{tok_offset}] error: {e}")
                continue

        if (i + 1) % 5 == 0:
            rate = total_certified / total_challenged if total_challenged > 0 else 0
            print(f"  [{i+1}/{len(PROMPTS)}] {total_certified}/{total_challenged} certified ({rate:.1%})")

    # Summary.
    print(f"\n{'='*60}")
    print(f"[{model_name}] Certification Quality Summary")
    print(f"{'='*60}")

    if total_challenged > 0:
        cert_rate = total_certified / total_challenged
        esc_rate = total_low_concentration / total_challenged
        margin_fail_rate = total_low_margin / total_challenged
        other_fail = total_challenged - total_certified - total_low_concentration - total_low_margin

        print(f"  Total tokens challenged: {total_challenged}")
        print(f"  Certified:              {total_certified} ({cert_rate:.1%})")
        print(f"  Low concentration:      {total_low_concentration} ({esc_rate:.1%})")
        print(f"  Low margin:             {total_low_margin} ({margin_fail_rate:.1%})")
        print(f"  Other failures:         {other_fail}")

        # Logit margin statistics.
        margins = [c["logit_margin"] for c in all_certs]
        masses = [c["min_top_k_mass"] for c in all_certs]
        bounds = [c["max_tail_bound"] for c in all_certs]

        print(f"\n  Logit margin:  min={min(margins):.2f}  median={np.median(margins):.2f}  mean={np.mean(margins):.2f}  max={max(margins):.2f}")
        print(f"  Top-k mass:    min={min(masses):.4f}  median={np.median(masses):.4f}  mean={np.mean(masses):.4f}")
        print(f"  Tail bound:    min={min(bounds):.4f}  median={np.median(bounds):.4f}  max={max(bounds):.4f}")

        # Certification by logit margin bucket.
        print(f"\n  Certification by logit margin:")
        for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 10), (10, 50), (50, 1000)]:
            bucket = [c for c in all_certs if lo <= c["logit_margin"] < hi]
            if bucket:
                n_cert = sum(1 for c in bucket if c["certified"])
                print(f"    margin [{lo:3d},{hi:4d}): {n_cert}/{len(bucket)} ({n_cert/len(bucket):.0%})")

        # Certification by top-k mass bucket.
        print(f"\n  Certification by top-k mass (k={TOP_K}):")
        for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]:
            bucket = [c for c in all_certs if lo <= c["min_top_k_mass"] < hi]
            if bucket:
                n_cert = sum(1 for c in bucket if c["certified"])
                print(f"    mass [{lo:.2f},{hi:.2f}): {n_cert}/{len(bucket)} ({n_cert/len(bucket):.0%})")
    else:
        print(f"  No tokens challenged.")

    print(f"{'='*60}")

    return {
        "model": model_name,
        "total_challenged": total_challenged,
        "total_certified": total_certified,
        "total_low_concentration": total_low_concentration,
        "total_low_margin": total_low_margin,
        "certs": all_certs,
    }


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_qwen():
    return _run_model("qwen", MODELS["qwen"])


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_llama():
    return _run_model("llama", MODELS["llama"])


@app.local_entrypoint()
def main():
    import json
    print("Stock-Bounded Certification Quality Measurement")
    print("=" * 60)

    qwen_future = run_qwen.spawn()
    llama_future = run_llama.spawn()

    qwen_result = qwen_future.get()
    llama_result = llama_future.get()

    print("\n" + "=" * 60)
    print("COMBINED RESULTS")
    print("=" * 60)

    for result in [qwen_result, llama_result]:
        model = result["model"]
        tc = result["total_challenged"]
        cert = result["total_certified"]
        rate = cert / tc if tc > 0 else 0
        print(f"\n{model}: {cert}/{tc} certified ({rate:.1%})")
        print(f"  low-concentration: {result['total_low_concentration']}")
        print(f"  low-margin: {result['total_low_margin']}")
