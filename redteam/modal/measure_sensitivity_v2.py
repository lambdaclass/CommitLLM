"""
Advanced corridor sensitivity v2: multi-prompt + MLP propagation.

Improvements over v1:
- Multiple diverse prompts (factual, code, math, uncertain, long)
- MLP propagation: perturbation injected at self_attn output flows through
  post-attention LayerNorm + MLP, measuring the full within-layer amplification
- All-layers-simultaneously experiment (after single-layer)
- Per-prompt logit margin tracking

Usage:
    modal run redteam/modal/measure_sensitivity_v2.py
"""

import modal

app = modal.App("commitllm-sensitivity-v2")

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
    # Short factual — high confidence expected
    "The capital of France is",
    # Uncertain / low margin
    "The best programming language for beginners is",
    # Code completion
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
    # Math
    "What is 7 * 8? The answer is",
    # Longer context
    "In a groundbreaking 2024 study, researchers at MIT demonstrated that large language models can be verified using cryptographic commitment schemes. The key insight is that",
    # Adversarial-style — model might be uncertain
    "Is the following statement true or false: 'The sun rises in the west.'",
]


@app.function(
    image=cuda_image,
    gpu="A100-40GB",
    timeout=7200,
)
def measure_sensitivity_v2(model_name: str = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"):
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    layer_count = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = hidden_dim // n_heads
    tau = 8

    print(f"Model: {layer_count} layers, dim={hidden_dim}, {n_heads} heads")

    # Precompute per-layer W_o SVD (the expensive part — do once)
    print("Computing per-layer W_o SVD...")
    wo_data = []
    for i in range(layer_count):
        layer = model.model.layers[i]
        wo = layer.self_attn.o_proj.weight.detach().float()
        U, S, Vh = torch.linalg.svd(wo, full_matrices=False)
        wo_data.append(dict(wo=wo, v_top=Vh[0, :], s_top=S[0].item()))
    print("SVD done.")

    all_prompt_results = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n{'='*85}")
        print(f"Prompt {pi}: {prompt[:60]}{'...' if len(prompt)>60 else ''}")
        print(f"{'='*85}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # =================================================================
        # Honest forward pass — capture residuals and o_proj inputs
        # =================================================================
        residuals = {}
        oproj_inputs = {}
        hooks = []

        def make_res_hook(idx):
            def fn(mod, inp, out):
                residuals[idx] = inp[0].detach().clone()
            return fn

        def make_oproj_hook(idx):
            def fn(mod, inp, out):
                oproj_inputs[idx] = inp[0][:, -1, :].detach().clone()
            return fn

        for i in range(layer_count):
            layer = model.model.layers[i]
            hooks.append(layer.register_forward_hook(make_res_hook(i)))
            hooks.append(layer.self_attn.o_proj.register_forward_hook(make_oproj_hook(i)))

        with torch.no_grad():
            honest_out = model(**inputs)
        for h in hooks:
            h.remove()

        honest_logits = honest_out.logits[:, -1, :].detach().float()
        honest_token = honest_logits.argmax(dim=-1).item()
        probs = torch.softmax(honest_logits, dim=-1)
        top5p, top5i = probs.topk(5, dim=-1)
        margin = (honest_logits[0, top5i[0, 0]] - honest_logits[0, top5i[0, 1]]).item()

        print(f"Token: '{tokenizer.decode([honest_token])}' prob={probs[0,honest_token].item():.4f} margin={margin:.4f}")

        # =================================================================
        # Per-layer: attn-only and attn+MLP finite-difference
        # =================================================================
        print(f"\n{'layer':>5} {'scale_a':>8} {'attn_Δ':>10} {'attn+mlp_Δ':>12} "
              f"{'margin%_a':>10} {'margin%_am':>11} {'flip_a':>15} {'flip_am':>15}")
        print("-" * 100)

        layer_results = []
        flips_attn = 0
        flips_attn_mlp = 0

        for j in range(layer_count):
            wo = wo_data[j]['wo']
            v_top = wo_data[j]['v_top']

            oproj_in = oproj_inputs[j].float().squeeze()
            scale_a = oproj_in.abs().max().item() / 127.0

            delta_a_i8 = tau * torch.sign(v_top)
            delta_post_wo = wo @ (scale_a * delta_a_i8)

            # --- Attn-only: hook self_attn output (post-W_o, pre-residual-add) ---
            # This perturbation enters the residual, then flows through post_norm + MLP
            # in the SAME layer, so it already includes MLP propagation within this layer.
            # The hook on self_attn captures the attention module output which gets added
            # to the residual before MLP.
            def make_attn_hook(delta):
                def fn(mod, inp, out):
                    if isinstance(out, tuple):
                        h = out[0].clone()
                        h[:, -1, :] += delta.to(h.device, h.dtype)
                        return (h,) + out[1:]
                    return out.clone() + delta.to(out.device, out.dtype).unsqueeze(0).unsqueeze(0)
                return fn

            hook = model.model.layers[j].self_attn.register_forward_hook(
                make_attn_hook(delta_post_wo))
            with torch.no_grad():
                attn_out = model(**inputs)
            hook.remove()

            attn_logits = attn_out.logits[:, -1, :].detach().float()
            attn_token = attn_logits.argmax(dim=-1).item()
            attn_dmax = (attn_logits - honest_logits).abs().max().item()
            attn_flip = attn_token != honest_token
            if attn_flip:
                flips_attn += 1

            # --- Attn+MLP bypass: hook the FULL layer output ---
            # This injects the perturbation AFTER the MLP, so it bypasses MLP
            # amplification within this layer. Comparing with attn-only (which
            # does flow through MLP) tells us whether MLP amplifies or dampens.
            def make_layer_hook(delta):
                def fn(mod, inp, out):
                    if isinstance(out, tuple):
                        h = out[0].clone()
                        h[:, -1, :] += delta.to(h.device, h.dtype)
                        return (h,) + out[1:]
                    return out.clone() + delta.to(out.device, out.dtype).unsqueeze(0).unsqueeze(0)
                return fn

            hook = model.model.layers[j].register_forward_hook(
                make_layer_hook(delta_post_wo))
            with torch.no_grad():
                layer_out = model(**inputs)
            hook.remove()

            layer_logits = layer_out.logits[:, -1, :].detach().float()
            layer_token = layer_logits.argmax(dim=-1).item()
            layer_dmax = (layer_logits - honest_logits).abs().max().item()
            layer_flip = layer_token != honest_token
            if layer_flip:
                flips_attn_mlp += 1

            m_a = attn_dmax / (abs(margin) + 1e-12) * 100
            m_am = layer_dmax / (abs(margin) + 1e-12) * 100

            fa = f"YES→'{tokenizer.decode([attn_token])}'" if attn_flip else "no"
            fam = f"YES→'{tokenizer.decode([layer_token])}'" if layer_flip else "no"

            layer_results.append(dict(
                layer=j, scale_a=scale_a,
                attn_dmax=attn_dmax, layer_dmax=layer_dmax,
                attn_margin_pct=m_a, layer_margin_pct=m_am,
                attn_flip=attn_flip, layer_flip=layer_flip,
                attn_token=attn_token, layer_token=layer_token,
                mlp_ratio=attn_dmax / (layer_dmax + 1e-12),
            ))

            print(f"{j:>5} {scale_a:>8.5f} {attn_dmax:>10.4f} {layer_dmax:>12.4f} "
                  f"{m_a:>9.1f}% {m_am:>10.1f}% {fa:>15} {fam:>15}")

        # =================================================================
        # All-layers-simultaneously
        # =================================================================
        print(f"\nAll-layers-simultaneously experiment:")

        all_hooks = []
        for j in range(layer_count):
            wo = wo_data[j]['wo']
            v_top = wo_data[j]['v_top']
            oproj_in = oproj_inputs[j].float().squeeze()
            scale_a = oproj_in.abs().max().item() / 127.0
            delta_a_i8 = tau * torch.sign(v_top)
            delta_post_wo = wo @ (scale_a * delta_a_i8)

            all_hooks.append(
                model.model.layers[j].self_attn.register_forward_hook(
                    make_attn_hook(delta_post_wo)))

        with torch.no_grad():
            all_out = model(**inputs)
        for h in all_hooks:
            h.remove()

        all_logits = all_out.logits[:, -1, :].detach().float()
        all_token = all_logits.argmax(dim=-1).item()
        all_dmax = (all_logits - honest_logits).abs().max().item()
        all_flip = all_token != honest_token
        all_margin = all_dmax / (abs(margin) + 1e-12) * 100

        print(f"  Δlogit_max={all_dmax:.4f}, margin={all_margin:.1f}%, "
              f"flip={'YES→' + repr(tokenizer.decode([all_token])) if all_flip else 'no'}")

        # =================================================================
        # Prompt summary
        # =================================================================
        mlp_ratios = [r['mlp_ratio'] for r in layer_results]
        print(f"\nPrompt {pi} summary:")
        print(f"  Margin: {margin:.4f}")
        print(f"  Attn-only flips: {flips_attn}/{layer_count}")
        print(f"  Post-layer flips: {flips_attn_mlp}/{layer_count}")
        print(f"  MLP amplification ratio (attn/layer): "
              f"mean={np.mean(mlp_ratios):.2f}, range=[{min(mlp_ratios):.2f}, {max(mlp_ratios):.2f}]")
        print(f"  All-layers Δlogit: {all_dmax:.4f} ({all_margin:.1f}% margin)")

        # Top-3 dangerous layers
        ranked = sorted(layer_results, key=lambda r: r['attn_dmax'], reverse=True)
        print(f"  Top-3 dangerous (attn-only):")
        for r in ranked[:3]:
            extra = f" FLIPS→'{tokenizer.decode([r['attn_token']])}'" if r['attn_flip'] else ""
            print(f"    Layer {r['layer']}: Δ={r['attn_dmax']:.4f}{extra}")

        all_prompt_results.append(dict(
            prompt=prompt, margin=margin, honest_token=honest_token,
            flips_attn=flips_attn, flips_layer=flips_attn_mlp,
            layer_results=layer_results,
            all_layers_dmax=all_dmax, all_layers_flip=all_flip,
        ))

    # =====================================================================
    # Cross-prompt summary
    # =====================================================================
    print(f"\n{'='*85}")
    print("CROSS-PROMPT SUMMARY")
    print(f"{'='*85}\n")

    print(f"{'prompt':>6} {'margin':>8} {'flips_attn':>11} {'flips_post':>11} "
          f"{'all_Δ':>10} {'all_flip':>9}")
    print("-" * 65)

    for i, pr in enumerate(all_prompt_results):
        af = f"YES" if pr['all_layers_flip'] else "no"
        print(f"{i:>6} {pr['margin']:>8.4f} {pr['flips_attn']:>7}/{layer_count:<3} "
              f"{pr['flips_layer']:>7}/{layer_count:<3} {pr['all_layers_dmax']:>10.4f} {af:>9}")

    # Which layers appear in top-3 most often?
    layer_danger_count = {}
    for pr in all_prompt_results:
        ranked = sorted(pr['layer_results'], key=lambda r: r['attn_dmax'], reverse=True)
        for r in ranked[:3]:
            layer_danger_count[r['layer']] = layer_danger_count.get(r['layer'], 0) + 1

    print(f"\nMost frequently dangerous layers (top-3 across all prompts):")
    for l, c in sorted(layer_danger_count.items(), key=lambda x: -x[1])[:5]:
        print(f"  Layer {l}: appears in top-3 for {c}/{len(PROMPTS)} prompts")

    # MLP amplification summary
    all_mlp_ratios = []
    for pr in all_prompt_results:
        for r in pr['layer_results']:
            all_mlp_ratios.append(r['mlp_ratio'])
    arr = np.array(all_mlp_ratios)
    print(f"\nMLP amplification (attn-path / post-layer-path):")
    print(f"  mean={arr.mean():.3f}, median={np.median(arr):.3f}, "
          f"range=[{arr.min():.3f}, {arr.max():.3f}]")
    if arr.mean() > 1.1:
        print("  MLP AMPLIFIES perturbations on average.")
    elif arr.mean() < 0.9:
        print("  MLP DAMPENS perturbations on average.")
    else:
        print("  MLP is roughly neutral on average.")

    return dict(
        model=model_name,
        prompt_results=all_prompt_results,
        layer_danger_count=layer_danger_count,
    )


@app.local_entrypoint()
def main():
    for model_name in [
        "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
    ]:
        print(f"\n{'#'*85}")
        print(f"  {model_name}")
        print(f"{'#'*85}\n")
        r = measure_sensitivity_v2.remote(model_name=model_name)
        ldc = r['layer_danger_count']
        top_layers = sorted(ldc.items(), key=lambda x: -x[1])[:3]
        print(f"\nDone. Consistently dangerous layers: {top_layers}")
