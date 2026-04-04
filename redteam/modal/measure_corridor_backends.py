"""
Measure honest attention corridor under different backends.

Compares the attention output mismatch between:
1. Eager attention (attn_implementation="eager")
2. SDPA (scaled_dot_product_attention)
3. Eager + deterministic CUDA (torch.use_deterministic_algorithms)

For each backend pair, measures L∞ and L2 of the attention output difference
per layer. Also tests run-to-run determinism for each backend.

Usage:
    modal run redteam/modal/measure_corridor_backends.py
"""

import modal

app = modal.App("commitllm-corridor-backends")

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.44",
        "accelerate",
        "numpy",
        "compressed-tensors",
    )
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
    "In a groundbreaking 2024 study, researchers at MIT demonstrated that large language models can be verified using cryptographic commitment schemes. The key insight is that",
]


def capture_attn_outputs(model, tokenizer, prompts, layer_count):
    """Run prompts and capture per-layer self_attn outputs for the last token."""
    import torch

    all_results = {}
    for pi, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        attn_outputs = {}
        hooks = []

        def make_hook(idx, store):
            def fn(mod, inp, out):
                if isinstance(out, tuple):
                    t = out[0]
                else:
                    t = out
                if t is not None and t.dim() >= 2:
                    store[idx] = t[:, -1, :].detach().clone().float()
                else:
                    store[idx] = None
            return fn

        for i in range(layer_count):
            h = model.model.layers[i].self_attn.register_forward_hook(
                make_hook(i, attn_outputs))
            hooks.append(h)

        with torch.no_grad():
            output = model(**inputs)

        for h in hooks:
            h.remove()

        token = output.logits[:, -1, :].argmax(dim=-1).item()
        valid_outputs = {k: v.cpu() for k, v in attn_outputs.items() if v is not None}

        all_results[pi] = {
            'attn_outputs': valid_outputs,
            'token': token,
            'token_str': tokenizer.decode([token]),
            'n_captured': len(valid_outputs),
            'n_layers': layer_count,
        }
        print(f"  Prompt {pi}: token='{tokenizer.decode([token])}' "
              f"(captured {len(valid_outputs)}/{layer_count} layers)")

    return all_results


def run_determinism_test(model, tokenizer, layer_count, backend_name):
    """Run the same prompt twice and compare attention outputs."""
    import torch
    import numpy as np

    print(f"\n--- {backend_name} run-to-run determinism ---")
    prompt = PROMPTS[0]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    runs = []
    for run_idx in range(2):
        attn_outputs = {}
        hooks = []

        def make_hook(idx, store):
            def fn(mod, inp, out):
                if isinstance(out, tuple):
                    t = out[0]
                else:
                    t = out
                if t is not None and t.dim() >= 2:
                    store[idx] = t[:, -1, :].detach().clone().float()
            return fn

        for i in range(layer_count):
            h = model.model.layers[i].self_attn.register_forward_hook(
                make_hook(i, attn_outputs))
            hooks.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()
        runs.append({k: v.cpu() for k, v in attn_outputs.items()})

    max_diff = 0.0
    n_compared = 0
    for layer_idx in runs[0]:
        if layer_idx not in runs[1]:
            continue
        diff = (runs[0][layer_idx] - runs[1][layer_idx]).abs().max().item()
        if not np.isnan(diff):
            max_diff = max(max_diff, diff)
            n_compared += 1

    if max_diff == 0.0:
        print(f"  BIT-EXACT across two runs (L∞ = 0.0, {n_compared} layers compared)")
    else:
        print(f"  Run-to-run max L∞ = {max_diff:.6f} ({n_compared} layers compared)")

    return max_diff


@app.function(
    image=cuda_image,
    gpu="A100-80GB",
    timeout=3600,
)
def measure_backends(model_name: str = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"):
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    BACKENDS = ["eager", "sdpa"]

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = {}
    determinism = {}

    for backend in BACKENDS:
        print(f"\n{'='*70}")
        print(f"Backend: {backend}")
        print(f"{'='*70}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation=backend,
            )
            model.eval()
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        layer_count = model.config.num_hidden_layers
        backend_results = capture_attn_outputs(model, tokenizer, PROMPTS, layer_count)

        for pi, data in backend_results.items():
            results[(backend, pi)] = data

        # Run-to-run determinism test
        det_diff = run_determinism_test(model, tokenizer, layer_count, backend)
        determinism[backend] = det_diff

        # Also test with deterministic algorithms if eager
        if backend == "eager":
            print(f"\n{'='*70}")
            print(f"Backend: eager+deterministic")
            print(f"{'='*70}")

            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

            eager_det_results = capture_attn_outputs(
                model, tokenizer, PROMPTS, layer_count)
            for pi, data in eager_det_results.items():
                results[("eager_det", pi)] = data

            det_diff = run_determinism_test(
                model, tokenizer, layer_count, "eager+deterministic")
            determinism["eager_det"] = det_diff

            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass

        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # Compare backends
    # =========================================================================
    all_backends = sorted(set(b for b, _ in results.keys()))
    n_prompts = len(PROMPTS)

    print(f"\n{'='*70}")
    print("Backend comparison")
    print(f"{'='*70}")

    # Compare every pair
    for i, ref_backend in enumerate(all_backends):
        for cmp_backend in all_backends[i+1:]:
            print(f"\n--- {ref_backend} vs {cmp_backend} ---\n")
            print(f"{'prompt':>6} {'layer':>5} {'L∞':>12} {'L2':>12} "
                  f"{'frac_eq':>10} {'frac≤1':>10}")
            print("-" * 60)

            all_linf = []

            for pi in range(n_prompts):
                ref_key = (ref_backend, pi)
                cmp_key = (cmp_backend, pi)
                if ref_key not in results or cmp_key not in results:
                    print(f"  Prompt {pi}: missing data")
                    continue

                ref_data = results[ref_key]['attn_outputs']
                cmp_data = results[cmp_key]['attn_outputs']

                for layer_idx in sorted(ref_data.keys()):
                    if layer_idx not in cmp_data:
                        continue
                    diff = (ref_data[layer_idx] - cmp_data[layer_idx]).abs()
                    linf = diff.max().item()
                    l2 = diff.norm().item()

                    if np.isnan(linf) or np.isnan(l2):
                        continue

                    frac_eq = (diff == 0).float().mean().item()
                    frac_le1 = (diff <= 1.0).float().mean().item()
                    all_linf.append(linf)

                    # Print first 3, last 2, and worst layers
                    if (layer_idx <= 2 or
                        layer_idx >= len(ref_data) - 2 or
                        linf == max(all_linf)):
                        print(f"{pi:>6} {layer_idx:>5} {linf:>12.6f} {l2:>12.4f} "
                              f"{frac_eq:>9.4f} {frac_le1:>9.4f}")

                ref_tok = results[ref_key]['token_str']
                cmp_tok = results[cmp_key]['token_str']
                match = ("SAME" if ref_tok == cmp_tok
                         else f"DIFF: '{ref_tok}' vs '{cmp_tok}'")
                print(f"  Prompt {pi} token: {match}")

            if all_linf:
                arr = np.array(all_linf)
                print(f"\n  Summary: L∞ range [{arr.min():.6f}, {arr.max():.6f}], "
                      f"mean={arr.mean():.6f}, median={np.median(arr):.6f}")
                if arr.max() < 1.0:
                    print("  ALL DIFFERENCES < 1.0 — corridor gap is near zero!")
                elif arr.max() < 10.0:
                    print(f"  Max L∞ = {arr.max():.2f} — tighter than current τ=8-9.")
                else:
                    print(f"  Max L∞ = {arr.max():.2f} — NOT significantly tighter.")

    # =========================================================================
    # Determinism summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("Run-to-run determinism summary")
    print(f"{'='*70}")
    for backend, diff in sorted(determinism.items()):
        status = "BIT-EXACT" if diff == 0.0 else f"L∞={diff:.6f}"
        print(f"  {backend}: {status}")

    return dict(
        model=model_name,
        backends=all_backends,
        determinism=determinism,
    )


@app.local_entrypoint()
def main():
    for model_name in [
        "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
    ]:
        print(f"\n{'#'*70}")
        print(f"  {model_name}")
        print(f"{'#'*70}\n")
        measure_backends.remote(model_name=model_name)
