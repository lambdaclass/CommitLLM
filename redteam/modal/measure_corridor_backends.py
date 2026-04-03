"""
Measure honest attention corridor under different backends.

Compares the attention output mismatch between:
1. FlashAttention-2 (default)
2. Eager attention (attn_implementation="eager")
3. Eager + deterministic CUDA (torch.use_deterministic_algorithms)
4. SDPA (scaled_dot_product_attention)

For each backend pair, measures L∞ and L2 of the attention output difference
per layer. If eager gives L∞ ≈ 0, the corridor gap shrinks to near zero
and score witnessing becomes unnecessary.

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


@app.function(
    image=cuda_image,
    gpu="A100-40GB",
    timeout=3600,
)
def measure_backends(model_name: str = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"):
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    PROMPTS = [
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
        "In a groundbreaking 2024 study, researchers at MIT demonstrated that large language models can be verified using cryptographic commitment schemes. The key insight is that",
    ]

    BACKENDS = ["flash_attention_2", "eager", "sdpa"]

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = {}

    # Load model for each backend and capture per-layer attention outputs
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

        for pi, prompt in enumerate(PROMPTS):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Capture post-o_proj outputs at each layer
            attn_outputs = {}
            hooks = []

            def make_hook(idx):
                def fn(mod, inp, out):
                    # self_attn output is a tuple; first element is attention output
                    if isinstance(out, tuple):
                        attn_outputs[idx] = out[0][:, -1, :].detach().clone().float()
                    else:
                        attn_outputs[idx] = out[:, -1, :].detach().clone().float()
                return fn

            for i in range(layer_count):
                h = model.model.layers[i].self_attn.register_forward_hook(make_hook(i))
                hooks.append(h)

            with torch.no_grad():
                output = model(**inputs)

            for h in hooks:
                h.remove()

            token = output.logits[:, -1, :].argmax(dim=-1).item()
            key = (backend, pi)
            results[key] = {
                'attn_outputs': {k: v.cpu() for k, v in attn_outputs.items()},
                'token': token,
                'token_str': tokenizer.decode([token]),
            }
            print(f"  Prompt {pi}: token='{tokenizer.decode([token])}'")

        # Also run with deterministic mode if eager
        if backend == "eager":
            print(f"\n{'='*70}")
            print(f"Backend: eager+deterministic")
            print(f"{'='*70}")

            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

            for pi, prompt in enumerate(PROMPTS):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                attn_outputs = {}
                hooks = []

                for i in range(layer_count):
                    h = model.model.layers[i].self_attn.register_forward_hook(make_hook(i))
                    hooks.append(h)

                with torch.no_grad():
                    output = model(**inputs)

                for h in hooks:
                    h.remove()

                token = output.logits[:, -1, :].argmax(dim=-1).item()
                key = ("eager_det", pi)
                results[key] = {
                    'attn_outputs': {k: v.cpu() for k, v in attn_outputs.items()},
                    'token': token,
                    'token_str': tokenizer.decode([token]),
                }
                print(f"  Prompt {pi}: token='{tokenizer.decode([token])}'")

            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass

        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # Compare backends
    # =========================================================================
    all_backends = list(set(b for b, _ in results.keys()))
    all_backends.sort()
    n_prompts = len(PROMPTS)

    print(f"\n{'='*70}")
    print("Backend comparison")
    print(f"{'='*70}")

    # Use first available backend as reference
    ref_backend = "flash_attention_2" if any(b == "flash_attention_2" for b in all_backends) else all_backends[0]

    for cmp_backend in all_backends:
        if cmp_backend == ref_backend:
            continue

        print(f"\n--- {ref_backend} vs {cmp_backend} ---\n")
        print(f"{'prompt':>6} {'layer':>5} {'L∞':>12} {'L2':>12} {'frac_eq':>10} {'frac≤1':>10}")
        print("-" * 60)

        all_linf = []
        all_l2 = []

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
                frac_eq = (diff == 0).float().mean().item()
                frac_le1 = (diff <= 1.0).float().mean().item()

                all_linf.append(linf)
                all_l2.append(l2)

                # Only print first/last/worst layers to keep output manageable
                if layer_idx <= 2 or layer_idx >= len(ref_data) - 2 or linf == max(all_linf):
                    print(f"{pi:>6} {layer_idx:>5} {linf:>12.6f} {l2:>12.4f} "
                          f"{frac_eq:>9.4f} {frac_le1:>9.4f}")

            # Per-prompt summary
            ref_tok = results[ref_key]['token_str']
            cmp_tok = results[cmp_key]['token_str']
            match = "SAME" if ref_tok == cmp_tok else f"DIFF: '{ref_tok}' vs '{cmp_tok}'"
            print(f"  Prompt {pi} token: {match}")

        if all_linf:
            arr = np.array(all_linf)
            print(f"\n  Summary: L∞ range [{arr.min():.6f}, {arr.max():.6f}], "
                  f"mean={arr.mean():.6f}, median={np.median(arr):.6f}")
            if arr.max() < 1.0:
                print("  ALL DIFFERENCES < 1.0 — corridor gap is effectively zero!")
            elif arr.max() < 10.0:
                print(f"  Max L∞ = {arr.max():.2f} — corridor is tighter than current τ=8-9.")
            else:
                print(f"  Max L∞ = {arr.max():.2f} — corridor is NOT significantly tighter.")

    # Also compare eager with itself (run-to-run determinism)
    if "eager" in all_backends:
        print(f"\n--- Eager run-to-run determinism ---")
        print("Running eager twice and comparing...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()
        layer_count = model.config.num_hidden_layers

        for pi, prompt in enumerate(PROMPTS[:1]):  # just first prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            runs = []
            for run_idx in range(2):
                attn_outputs = {}
                hooks = []

                def make_hook2(idx):
                    def fn(mod, inp, out):
                        if isinstance(out, tuple):
                            attn_outputs[idx] = out[0][:, -1, :].detach().clone().float()
                        else:
                            attn_outputs[idx] = out[:, -1, :].detach().clone().float()
                    return fn

                for i in range(layer_count):
                    h = model.model.layers[i].self_attn.register_forward_hook(make_hook2(i))
                    hooks.append(h)

                with torch.no_grad():
                    model(**inputs)

                for h in hooks:
                    h.remove()
                runs.append({k: v.cpu() for k, v in attn_outputs.items()})

            max_diff = 0.0
            for layer_idx in runs[0]:
                diff = (runs[0][layer_idx] - runs[1][layer_idx]).abs().max().item()
                max_diff = max(max_diff, diff)

            if max_diff == 0.0:
                print(f"  Eager is BIT-EXACT across two runs (L∞ = 0.0)")
            else:
                print(f"  Eager run-to-run max L∞ = {max_diff:.6f}")

    return dict(
        model=model_name,
        backends=all_backends,
        summary={b: [] for b in all_backends},
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
