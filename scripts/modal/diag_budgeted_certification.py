"""
Budgeted certification quality diagnostics.

Measures the new adaptive + alpha + budgeted greedy certifier across:
- Models: Llama 8B, Qwen 7B
- Decode mode: greedy (temperature=0), sampled (temperature=0.8)
- Context: short (max_tokens=32), long (max_tokens=128)

Reports:
1. Certification rate (certified / challenged)
2. Outcome breakdown (certified, budget_exhausted, margin_too_small)
3. Budget usage (median / p90 total_k_spent, chosen_k distribution)
4. Bound quality (estimated_bound / margin ratio, near-miss vs hard-fail)

Usage:
    modal run --detach scripts/modal/diag_budgeted_certification.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-budgeted-certification")

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
    "llama": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
    "qwen": "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
}

PROMPTS_SHORT = [
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
]

PROMPTS_LONG = [
    "Write a detailed essay about the history of computing from Babbage to modern GPUs. Cover the key milestones, breakthroughs, and the people behind them.",
    "Explain the entire process of how a web browser renders a page from the moment you type a URL. Include DNS, TCP, HTTP, parsing, layout, and painting.",
    "Describe the complete lifecycle of a star from nebula to its final state. Cover main sequence, red giant, supernova, and the different endpoints.",
    "Write a comprehensive guide to understanding cryptographic hash functions. Cover properties, constructions, attacks, and real-world applications.",
    "Explain the theory and practice of distributed consensus algorithms. Cover Paxos, Raft, and Byzantine fault tolerance.",
]

N_CHALLENGES_PER_REQUEST = 5
MAX_K_PER_HEAD = 128
MAX_TOTAL_K = 8192

CONFIGS = [
    # (name, prompts, max_tokens, temperature, top_k, top_p)
    ("greedy-short",  PROMPTS_SHORT, 32,  0.0, 1,  1.0),
    ("sampled-short", PROMPTS_SHORT, 32,  0.8, 50, 0.9),
    ("greedy-long",   PROMPTS_LONG,  128, 0.0, 1,  1.0),
    ("sampled-long",  PROMPTS_LONG,  128, 0.8, 50, 0.9),
]


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

    # Generate verifier key (binary — includes o_proj_alpha).
    seed = hashlib.sha256(f"diag-budgeted-{model_name}".encode()).digest()
    key_binary, _ = verilm_rs.generate_key_binary(model_dir, list(seed))
    key_binary = bytes(key_binary)

    # Inspect key to verify alpha is present.
    key_info = verilm_rs.inspect_key_binary(key_binary)
    print(f"[{model_name}] Key: {key_info['n_layers']}L, {key_info['hidden_dim']}d, "
          f"{key_info['n_q_heads']}qh, {key_info.get('vocab_size', '?')}v")

    # Warmup.
    print(f"[{model_name}] Warmup...")
    server.chat(prompt="Hello", max_tokens=8, temperature=0.0)

    all_results = {}

    for config_name, prompts, max_tokens, temperature, top_k, top_p in CONFIGS:
        print(f"\n{'='*70}")
        print(f"[{model_name}:{config_name}] {len(prompts)} prompts, "
              f"max_tokens={max_tokens}, temp={temperature}")
        print(f"  budget: max_k_per_head={MAX_K_PER_HEAD}, max_total_k={MAX_TOTAL_K}")
        print(f"{'='*70}")

        rng = random.Random(42)
        certs = []

        for i, prompt in enumerate(prompts):
            result = server.chat(
                prompt=prompt, max_tokens=max_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
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

                try:
                    audit_binary = server.audit(
                        request_id=result["request_id"],
                        token_index=challenge_idx,
                        layer_indices=full_layers,
                        tier="full",
                        binary=True,
                    )

                    # Logit margin from captured logits.
                    captured = server.sampler_hook._captured_logits
                    if tok_offset < len(captured):
                        logits = np.array(captured[tok_offset], dtype=np.float32)
                        sorted_logits = np.sort(logits)[::-1]
                        logit_margin = float(sorted_logits[0] - sorted_logits[1])
                    else:
                        logit_margin = 0.0

                    # Budgeted certification with alpha.
                    cert_json = verilm_rs.compute_attention_certification_budgeted(
                        bytes(audit_binary), key_binary, logit_margin,
                        MAX_K_PER_HEAD, MAX_TOTAL_K,
                    )
                    cert = json.loads(cert_json)

                    certs.append({
                        "prompt_idx": i,
                        "tok_offset": tok_offset,
                        "seq_len": n_tokens,
                        "outcome": cert["outcome"],
                        "logit_margin": cert["logit_margin"],
                        "estimated_logit_bound": cert["estimated_logit_bound"],
                        "total_k_spent": cert["total_k_spent"],
                        "chosen_k_per_layer": cert["chosen_k_per_layer"],
                        "reason": cert["reason"],
                    })

                except Exception as e:
                    print(f"  [{i}:{tok_offset}] error: {e}")
                    continue

            if (i + 1) % 5 == 0 or i == len(prompts) - 1:
                n_cert = sum(1 for c in certs if c["outcome"] == "certified")
                rate = n_cert / len(certs) if certs else 0
                print(f"  [{i+1}/{len(prompts)}] {n_cert}/{len(certs)} certified ({rate:.1%})")

        # Compute statistics.
        stats = _compute_stats(certs, config_name, model_name)
        all_results[config_name] = stats

    return {"model": model_name, "configs": all_results}


def _compute_stats(certs, config_name, model_name):
    import numpy as np

    n = len(certs)
    if n == 0:
        print(f"  No tokens challenged.")
        return {"n": 0}

    # 1. Outcome breakdown.
    n_certified = sum(1 for c in certs if c["outcome"] == "certified")
    n_budget = sum(1 for c in certs if c["outcome"] == "budget_exhausted")
    n_margin = sum(1 for c in certs if c["outcome"] == "margin_too_small")

    print(f"\n  --- [{model_name}:{config_name}] Outcome Breakdown ---")
    print(f"  Total challenged:     {n}")
    print(f"  Certified:            {n_certified} ({n_certified/n:.1%})")
    print(f"  Budget exhausted:     {n_budget} ({n_budget/n:.1%})")
    print(f"  Margin too small:     {n_margin} ({n_margin/n:.1%})")

    # 2. Budget usage.
    k_spent = [c["total_k_spent"] for c in certs]
    k_spent_certified = [c["total_k_spent"] for c in certs if c["outcome"] == "certified"]

    print(f"\n  --- Budget Usage (all tokens) ---")
    print(f"  total_k_spent: median={np.median(k_spent):.0f}  "
          f"p90={np.percentile(k_spent, 90):.0f}  "
          f"max={max(k_spent)}")
    if k_spent_certified:
        print(f"  (certified only): median={np.median(k_spent_certified):.0f}  "
              f"p90={np.percentile(k_spent_certified, 90):.0f}  "
              f"max={max(k_spent_certified)}")

    # Per-layer k distribution (aggregate across tokens).
    all_k_per_layer = {}
    for c in certs:
        for layer_idx, layer_ks in enumerate(c["chosen_k_per_layer"]):
            if layer_idx not in all_k_per_layer:
                all_k_per_layer[layer_idx] = []
            all_k_per_layer[layer_idx].extend(layer_ks)

    if all_k_per_layer:
        n_layers = max(all_k_per_layer.keys()) + 1
        print(f"\n  --- Per-Layer Chosen k (median / p90 / max) ---")
        for l in range(n_layers):
            ks = all_k_per_layer.get(l, [])
            if ks:
                print(f"    L{l:2d}: median={np.median(ks):5.0f}  "
                      f"p90={np.percentile(ks, 90):5.0f}  "
                      f"max={max(ks):5d}  "
                      f"(n_heads_sampled={len(ks)})")

    # 3. Bound quality.
    margins = [c["logit_margin"] for c in certs if c["logit_margin"] > 0]
    bound_ratios = [
        c["estimated_logit_bound"] / (c["logit_margin"] / 2.0)
        for c in certs
        if c["logit_margin"] > 0 and c["estimated_logit_bound"] < float("inf")
    ]

    if margins:
        print(f"\n  --- Logit Margin ---")
        print(f"  margin: min={min(margins):.3f}  median={np.median(margins):.3f}  "
              f"p90={np.percentile(margins, 90):.3f}  max={max(margins):.3f}")

    if bound_ratios:
        print(f"\n  --- Bound / (Margin/2) Ratio ---")
        print(f"  ratio: min={min(bound_ratios):.4f}  median={np.median(bound_ratios):.4f}  "
              f"p90={np.percentile(bound_ratios, 90):.4f}  max={max(bound_ratios):.4f}")
        # How many are near-miss (ratio in [0.5, 1.0]) vs hard-fail (ratio > 2)?
        near_miss = sum(1 for r in bound_ratios if 0.5 <= r < 1.0)
        marginal_fail = sum(1 for r in bound_ratios if 1.0 <= r < 2.0)
        hard_fail = sum(1 for r in bound_ratios if r >= 2.0)
        easy_cert = sum(1 for r in bound_ratios if r < 0.5)
        print(f"  easy (<0.5): {easy_cert}  near-miss (0.5-1.0): {near_miss}  "
              f"marginal-fail (1.0-2.0): {marginal_fail}  hard-fail (>2.0): {hard_fail}")

    # 4. Certification by logit margin bucket.
    print(f"\n  --- Certification by Logit Margin Bucket ---")
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 3), (3, 5), (5, 10), (10, 50), (50, 1000)]:
        bucket = [c for c in certs if lo <= c["logit_margin"] < hi]
        if bucket:
            nc = sum(1 for c in bucket if c["outcome"] == "certified")
            median_k = np.median([c["total_k_spent"] for c in bucket])
            print(f"    margin [{lo:5.1f},{hi:6.1f}): {nc:3d}/{len(bucket):3d} "
                  f"({nc/len(bucket):5.0%})  median_k={median_k:.0f}")

    return {
        "n": n,
        "n_certified": n_certified,
        "n_budget_exhausted": n_budget,
        "n_margin_too_small": n_margin,
        "cert_rate": n_certified / n,
        "k_spent_median": float(np.median(k_spent)),
        "k_spent_p90": float(np.percentile(k_spent, 90)),
        "bound_ratio_median": float(np.median(bound_ratios)) if bound_ratios else None,
        "certs": certs,
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_llama():
    return _run_model("llama", MODELS["llama"])


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_qwen():
    return _run_model("qwen", MODELS["qwen"])


@app.local_entrypoint()
def main():
    import json

    print("Budgeted Certification Quality Diagnostics")
    print(f"Budget: max_k_per_head={MAX_K_PER_HEAD}, max_total_k={MAX_TOTAL_K}")
    print("=" * 70)

    llama_future = run_llama.spawn()
    qwen_future = run_qwen.spawn()

    llama_result = llama_future.get()
    qwen_result = qwen_future.get()

    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    for result in [llama_result, qwen_result]:
        model = result["model"]
        print(f"\n{'='*70}")
        print(f"  MODEL: {model}")
        print(f"{'='*70}")
        for config_name, stats in result["configs"].items():
            if stats["n"] == 0:
                continue
            print(f"  {config_name}: "
                  f"{stats['n_certified']}/{stats['n']} certified ({stats['cert_rate']:.1%})  "
                  f"budget_exhausted={stats['n_budget_exhausted']}  "
                  f"margin_too_small={stats['n_margin_too_small']}  "
                  f"k_median={stats['k_spent_median']:.0f}  "
                  f"k_p90={stats['k_spent_p90']:.0f}"
                  + (f"  bound_ratio_median={stats['bound_ratio_median']:.4f}"
                     if stats['bound_ratio_median'] is not None else ""))
