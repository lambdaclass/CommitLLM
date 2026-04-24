"""
Directional gamma[l,h] diagnostic — the honest test of stock-bounded attention.

For each challenged token, computes the margin-directional sensitivity:

    gamma[l,h] = |δΔ| / ε

where:
  - Δ = logit[top1] - logit[top2] (winner-vs-runner-up margin)
  - δΔ is the margin shift when head h at layer l's attention output is perturbed by ε
  - The perturbation propagates through o_proj, residual connection, all subsequent
    layers, final norm, and lm_head — the full chain.

This measures g_l = ∂Δ/∂r_l projected onto each head's o_proj column space,
via finite differences.

Key question: is Σ gamma[l,h] * eps_attn[l,h,k] << margin/2?

If yes: stock-bounded attention is viable with directional certification.
If no: stop trying to rescue stock-bounded as a serious attention guarantee.

Usage:
    modal run --detach scripts/modal/diag_directional_gamma.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-directional-gamma")

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

PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
    "Tell me a fun fact about space.",
]

EPSILON = 0.1


def _find_o_proj_modules(model):
    """Find o_proj Linear modules for each transformer layer."""
    o_projs = {}
    for name, mod in model.named_modules():
        # Match patterns like "model.layers.0.self_attn.o_proj"
        if name.endswith(".o_proj") and "self_attn" in name:
            # Extract layer index
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        o_projs[layer_idx] = mod
                    except ValueError:
                        pass
    return o_projs


def _run_model(model_name, model_id):
    import time
    import numpy as np
    import torch

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from verilm.capture import get_model_from_llm

    print(f"\n[{model_name}] Loading {model_id}...")
    llm = LLM(model=model_id, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)

    model = get_model_from_llm(llm)

    # Find o_proj modules.
    o_projs = _find_o_proj_modules(model)
    n_layers = len(o_projs)
    print(f"[{model_name}] Found {n_layers} o_proj modules")

    # Get model config.
    for name, mod in model.named_modules():
        if hasattr(mod, 'num_heads') and hasattr(mod, 'head_dim'):
            n_q_heads = mod.num_heads
            d_head = mod.head_dim
            break
    else:
        # Fallback from model config
        cfg = getattr(model, 'config', None)
        if cfg:
            n_q_heads = getattr(cfg, 'num_attention_heads', 32)
            d_head = getattr(cfg, 'hidden_size', 4096) // n_q_heads
        else:
            raise RuntimeError("Cannot determine n_heads and d_head")

    hidden_dim = n_q_heads * d_head
    print(f"[{model_name}] n_layers={n_layers}, n_q_heads={n_q_heads}, "
          f"d_head={d_head}, hidden_dim={hidden_dim}")

    # Install logit capture hook.
    logit_captures = []

    def logit_hook(module, args, output):
        logit_captures.append(output.detach().float().cpu())

    logit_handle = None
    for name, mod in model.named_modules():
        if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
            logit_handle = mod.register_forward_hook(logit_hook)
            print(f"[{model_name}] Installed logit capture on {name}")
            break
    if logit_handle is None:
        raise RuntimeError("Could not find LogitsProcessor")

    # Warmup.
    params = SamplingParams(temperature=0, max_tokens=1)
    print(f"[{model_name}] Warmup...")
    logit_captures.clear()
    llm.generate(["Hello"], params)
    logit_captures.clear()

    # ---- Directional Gamma Measurement ----
    all_token_results = []
    t0 = time.time()

    for pi, prompt in enumerate(PROMPTS):
        # --- Baseline ---
        logit_captures.clear()
        llm.generate([prompt], params)
        if not logit_captures:
            print(f"  [{pi}] no logits captured, skip")
            continue
        baseline_logits = logit_captures[-1].numpy().flatten()
        top_indices = np.argsort(baseline_logits)[::-1]
        top1, top2 = int(top_indices[0]), int(top_indices[1])
        baseline_margin = float(baseline_logits[top1] - baseline_logits[top2])

        if baseline_margin <= 0:
            print(f"  [{pi}] margin={baseline_margin:.4f} <= 0, skip")
            continue

        # --- Per-layer, per-head perturbation ---
        gamma = np.zeros((n_layers, n_q_heads))

        for l in range(n_layers):
            o_proj_mod = o_projs[l]

            for h in range(n_q_heads):
                # Hook: perturb head h's o_proj input at the LAST position only.
                # With max_tokens=1, there's only a prefill pass (no separate decode).
                # We perturb the last position to measure margin sensitivity.
                applied = [False]

                def make_hook(head_idx, applied_flag):
                    def hook(module, input):
                        if applied_flag[0]:
                            return input  # Only apply once per forward pass
                        applied_flag[0] = True
                        h_tensor = input[0] if isinstance(input, tuple) else input
                        pert = torch.zeros_like(h_tensor)
                        start = head_idx * d_head
                        end = start + d_head
                        # Perturb only last token position
                        if h_tensor.dim() == 2:
                            pert[-1, start:end] = EPSILON
                        elif h_tensor.dim() == 3:
                            pert[:, -1, start:end] = EPSILON
                        else:
                            pert[..., start:end] = EPSILON
                        if isinstance(input, tuple):
                            return (h_tensor + pert,) + input[1:]
                        return h_tensor + pert
                    return hook

                handle = o_proj_mod.register_forward_pre_hook(make_hook(h, applied))

                logit_captures.clear()
                llm.generate([prompt], params)
                handle.remove()

                if not logit_captures:
                    continue

                perturbed_logits = logit_captures[-1].numpy().flatten()
                perturbed_margin = float(
                    perturbed_logits[top1] - perturbed_logits[top2]
                )
                delta_margin = abs(perturbed_margin - baseline_margin)
                gamma[l, h] = delta_margin / EPSILON

        # Also try negative perturbation for a few (l,h) to get max.
        # For speed, only do negative for layer 0 and last layer.
        for l in [0, n_layers - 1]:
            o_proj_mod = o_projs[l]
            for h in range(n_q_heads):
                applied = [False]

                def make_neg_hook(head_idx, applied_flag):
                    def hook(module, input):
                        if applied_flag[0]:
                            return input
                        applied_flag[0] = True
                        h_tensor = input[0] if isinstance(input, tuple) else input
                        pert = torch.zeros_like(h_tensor)
                        start = head_idx * d_head
                        end = start + d_head
                        if h_tensor.dim() == 2:
                            pert[-1, start:end] = -EPSILON
                        elif h_tensor.dim() == 3:
                            pert[:, -1, start:end] = -EPSILON
                        else:
                            pert[..., start:end] = -EPSILON
                        if isinstance(input, tuple):
                            return (h_tensor + pert,) + input[1:]
                        return h_tensor + pert
                    return hook

                handle = o_proj_mod.register_forward_pre_hook(make_neg_hook(h, applied))
                logit_captures.clear()
                llm.generate([prompt], params)
                handle.remove()

                if logit_captures:
                    perturbed_logits = logit_captures[-1].numpy().flatten()
                    perturbed_margin = float(
                        perturbed_logits[top1] - perturbed_logits[top2]
                    )
                    neg_gamma = abs(perturbed_margin - baseline_margin) / EPSILON
                    gamma[l, h] = max(gamma[l, h], neg_gamma)

        # Store results.
        sum_gamma = float(gamma.sum())
        per_layer_gamma = [float(gamma[l, :].sum()) for l in range(n_layers)]
        ratio = sum_gamma / (baseline_margin / 2.0)

        all_token_results.append({
            "prompt_idx": pi,
            "margin": baseline_margin,
            "top1": top1,
            "top2": top2,
            "sum_gamma": sum_gamma,
            "ratio_to_half_margin": ratio,
            "per_layer_sum_gamma": per_layer_gamma,
            "gamma_matrix": gamma.tolist(),
        })

        elapsed = time.time() - t0
        print(f"  [{pi+1}/{len(PROMPTS)}] margin={baseline_margin:.3f}  "
              f"sum_gamma={sum_gamma:.2f}  ratio={ratio:.2f}  "
              f"({elapsed:.0f}s)")

    logit_handle.remove()
    total_time = time.time() - t0

    # ---- Report ----
    print(f"\n{'='*70}")
    print(f"[{model_name}] Directional Gamma Results")
    print(f"  {len(all_token_results)} tokens, epsilon={EPSILON}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"{'='*70}")

    if not all_token_results:
        print("  No results.")
        return {"model": model_name, "results": []}

    margins = [r["margin"] for r in all_token_results]
    sum_gammas = [r["sum_gamma"] for r in all_token_results]
    ratios = [r["ratio_to_half_margin"] for r in all_token_results]

    print(f"\n  --- Margin ---")
    print(f"  min={min(margins):.3f}  median={np.median(margins):.3f}  "
          f"max={max(margins):.3f}")

    print(f"\n  --- Σ gamma[l,h] (total directional sensitivity, eps_attn=1) ---")
    print(f"  min={min(sum_gammas):.2f}  median={np.median(sum_gammas):.2f}  "
          f"max={max(sum_gammas):.2f}")

    print(f"\n  --- Ratio: Σ gamma / (margin/2) ---")
    print(f"  min={min(ratios):.2f}  median={np.median(ratios):.2f}  "
          f"max={max(ratios):.2f}")
    print(f"  (If < 1: certified without any top-k benefit)")
    print(f"  (Previous alpha-based ratio was 100-1000x)")

    # Per-layer sensitivity
    print(f"\n  --- Per-Layer Σ_h gamma[l,h] (median across tokens) ---")
    per_layer = np.array([r["per_layer_sum_gamma"] for r in all_token_results])
    for l in range(n_layers):
        col = per_layer[:, l]
        print(f"    L{l:2d}: median={np.median(col):7.3f}  "
              f"max={np.max(col):7.3f}")

    # Per-head gamma statistics (flattened across all tokens)
    all_gamma = np.array([r["gamma_matrix"] for r in all_token_results])
    gamma_flat = all_gamma.flatten()
    gamma_nonzero = gamma_flat[gamma_flat > 0]
    if len(gamma_nonzero) > 0:
        print(f"\n  --- Per-Head gamma (non-zero, across all tokens) ---")
        print(f"  n_nonzero={len(gamma_nonzero)}/{len(gamma_flat)}")
        print(f"  min={np.min(gamma_nonzero):.4f}  "
              f"median={np.median(gamma_nonzero):.4f}  "
              f"p90={np.percentile(gamma_nonzero, 90):.4f}  "
              f"max={np.max(gamma_nonzero):.4f}")

    # Interpretation
    print(f"\n  --- Interpretation ---")
    median_ratio = np.median(ratios)
    if median_ratio < 1.0:
        print(f"  EXCELLENT: directional gamma certifies without any top-k (ratio={median_ratio:.2f})")
    elif median_ratio < 10.0:
        print(f"  GOOD: directional gamma close to certifiable, top-k can close the gap (ratio={median_ratio:.2f})")
    elif median_ratio < 100.0:
        print(f"  MARGINAL: directional gamma helps but still needs significant top-k (ratio={median_ratio:.2f})")
        print(f"  Previous alpha ratio was 100-1000x. Improvement: ~{100/median_ratio:.0f}-{1000/median_ratio:.0f}x")
    else:
        print(f"  POOR: directional gamma does not help enough (ratio={median_ratio:.2f})")
        print(f"  Stock-bounded attention cannot be rescued by directional projection alone.")

    print(f"\n{'='*70}")

    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_q_heads": n_q_heads,
        "d_head": d_head,
        "results": all_token_results,
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

    print("Directional Gamma Diagnostic")
    print("=" * 70)

    llama_future = run_llama.spawn()
    qwen_future = run_qwen.spawn()

    llama_result = llama_future.get()
    qwen_result = qwen_future.get()

    print("\n" + "=" * 70)
    print("COMBINED SUMMARY")
    print("=" * 70)

    import numpy as np
    for result in [llama_result, qwen_result]:
        model = result["model"]
        results = result["results"]
        if not results:
            print(f"\n{model}: no results")
            continue
        ratios = [r["ratio_to_half_margin"] for r in results]
        sum_gammas = [r["sum_gamma"] for r in results]
        margins = [r["margin"] for r in results]
        print(f"\n{model}:")
        print(f"  margin: median={np.median(margins):.3f}")
        print(f"  Σgamma: median={np.median(sum_gammas):.2f}")
        print(f"  ratio Σgamma/(margin/2): median={np.median(ratios):.2f}  "
              f"range=[{min(ratios):.2f}, {max(ratios):.2f}]")
