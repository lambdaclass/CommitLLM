"""
Advanced corridor sensitivity measurements on real models.

Three measurements per layer:
1. Local operator norm: ‖J_RMSNorm(x_j) · W_o^(j)‖ (tighter than crude ρ)
2. Finite-difference sensitivity: inject ±τ through W_o, measure actual logit change
3. Bad-layer localization: rank layers by impact on output

Usage:
    modal run redteam/modal/measure_sensitivity.py
"""

import modal

app = modal.App("commitllm-sensitivity")

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
def measure_sensitivity(model_name: str = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"):
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

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    layer_count = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = hidden_dim // n_heads
    print(f"Model: {layer_count} layers, hidden_dim={hidden_dim}, {n_heads} heads, head_dim={head_dim}")

    # =========================================================================
    # Phase 1: Honest forward pass — capture residuals and o_proj inputs
    # =========================================================================

    residuals = {}       # layer_idx -> residual entering the layer (batch, seq, dim)
    oproj_inputs = {}    # layer_idx -> pre-W_o attention output at last token

    hooks = []

    def make_residual_hook(idx):
        def hook_fn(module, inp, out):
            residuals[idx] = inp[0].detach().clone()
        return hook_fn

    def make_oproj_hook(idx):
        def hook_fn(module, inp, out):
            # inp[0] is the merged multi-head output before o_proj, (batch, seq, dim)
            oproj_inputs[idx] = inp[0][:, -1, :].detach().clone()
        return hook_fn

    for i in range(layer_count):
        layer = model.model.layers[i]
        hooks.append(layer.register_forward_hook(make_residual_hook(i)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_oproj_hook(i)))

    with torch.no_grad():
        honest_output = model(**inputs)

    for h in hooks:
        h.remove()

    honest_logits = honest_output.logits[:, -1, :].detach().clone().float()
    honest_token = honest_logits.argmax(dim=-1).item()
    honest_probs = torch.softmax(honest_logits, dim=-1)

    top5_probs, top5_ids = honest_probs.topk(5, dim=-1)
    logit_margin = (honest_logits[0, top5_ids[0, 0]] - honest_logits[0, top5_ids[0, 1]]).item()

    print(f"Honest token: '{tokenizer.decode([honest_token])}' (prob={honest_probs[0, honest_token].item():.4f})")
    print(f"Logit margin (top1-top2): {logit_margin:.4f}")
    for i in range(5):
        tid = top5_ids[0, i].item()
        print(f"  #{i+1} '{tokenizer.decode([tid])}': {top5_probs[0,i].item():.4f}")

    # =========================================================================
    # Phase 2: Local operator norm  ‖J_RMSNorm(x) · W_o‖
    # =========================================================================

    print(f"\n{'='*85}")
    print("Phase 2: Local operator norm (tighter than crude ρ)")
    print(f"{'='*85}\n")

    def local_operator_spectral(x_vec, gamma, wo, n_iter=80):
        """Spectral norm of J_RMSNorm(x) · W_o via power iteration.

        J_RMSNorm(x) = diag(γ)/RMS(x) · (I - x x^T / ‖x‖²)
        """
        x = x_vec.float()
        g = gamma.float()
        w = wo.float()
        rms = torch.sqrt(torch.mean(x ** 2))
        xnorm2 = torch.dot(x, x)

        def jw(v):
            z = w @ v
            return g / rms * (z - x * (torch.dot(x, z) / xnorm2))

        v = torch.randn(w.shape[1], device=w.device)
        v = v / v.norm()
        for _ in range(n_iter):
            u = jw(v)
            # For transpose: J is symmetric so J^T W_o^T = (W_o J)^T
            # u_back = W_o^T J^T u = W_o^T (same J applied to u)
            ju = g / rms * (u - x * (torch.dot(x, u) / xnorm2))
            v_new = w.T @ ju
            sigma = v_new.norm()
            v = v_new / (sigma + 1e-12)
        return jw(v).norm().item()

    print(f"{'layer':>5} {'local ‖J·Wo‖':>14} {'crude ρ':>12} {'tighter by':>12} "
          f"{'‖Wo‖_∞→∞':>12} {'per-head max':>12}")
    print("-" * 80)

    local_norms = []
    crude_rhos = []
    linf_norms = []
    head_max_norms = []

    for i in range(layer_count):
        layer = model.model.layers[i]
        wo = layer.self_attn.o_proj.weight.detach().float()
        gamma = layer.input_layernorm.weight.detach().float()
        x_last = residuals[i][0, -1, :].float()  # last token

        # Local operator norm
        ln = local_operator_spectral(x_last, gamma, wo)
        local_norms.append(ln)

        # Crude ρ
        wo_s2 = torch.linalg.svdvals(wo)[0].item()
        gmax = gamma.abs().max().item()
        rms = torch.sqrt(torch.mean(x_last ** 2)).item()
        cr = wo_s2 * gmax / rms
        crude_rhos.append(cr)

        # ∞→∞ induced norm (max absolute row sum)
        linf = wo.abs().sum(dim=1).max().item()
        linf_norms.append(linf)

        # Per-head spectral norms
        head_norms = []
        for h in range(n_heads):
            block = wo[:, h * head_dim:(h + 1) * head_dim]
            head_norms.append(torch.linalg.svdvals(block)[0].item())
        hmax = max(head_norms)
        head_max_norms.append(hmax)

        tight = cr / (ln + 1e-12)
        print(f"{i:>5} {ln:>14.4f} {cr:>12.4f} {tight:>11.1f}x {linf:>12.1f} {hmax:>12.4f}")

    # =========================================================================
    # Phase 3: Finite-difference sensitivity
    # =========================================================================

    print(f"\n{'='*85}")
    print("Phase 3: Finite-difference — inject ±τ at each layer, measure logit change")
    print(f"{'='*85}\n")

    tau = 8  # i8 corridor tolerance

    print(f"{'layer':>5} {'scale_a':>10} {'Δlogit_max':>12} {'margin%':>10} {'flip?':>20}")
    print("-" * 65)

    sensitivities = []
    flips = 0

    for j in range(layer_count):
        layer_obj = model.model.layers[j]
        wo = layer_obj.self_attn.o_proj.weight.detach().float()

        # Estimate per-tensor i8 scale from captured o_proj input
        oproj_in = oproj_inputs[j].float().squeeze()
        scale_a = oproj_in.abs().max().item() / 127.0

        # Worst-case perturbation in i8 space aligned with top singular direction
        _, _, Vh = torch.linalg.svd(wo, full_matrices=False)
        v_top = Vh[0, :]
        delta_a_i8 = tau * torch.sign(v_top)          # ‖·‖_∞ = τ
        delta_post_wo = wo @ (scale_a * delta_a_i8)    # perturbation in residual space

        # Hook self_attn output to inject perturbation
        def make_inject_hook(delta):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0].clone()
                    h[:, -1, :] += delta.to(h.device, h.dtype)
                    return (h,) + out[1:]
                h = out.clone()
                h[:, -1, :] += delta.to(h.device, h.dtype)
                return h
            return hook_fn

        hook = layer_obj.self_attn.register_forward_hook(make_inject_hook(delta_post_wo))
        with torch.no_grad():
            pert_output = model(**inputs)
        hook.remove()

        pert_logits = pert_output.logits[:, -1, :].detach().float()
        pert_token = pert_logits.argmax(dim=-1).item()

        dmax = (pert_logits - honest_logits).abs().max().item()
        flipped = pert_token != honest_token
        if flipped:
            flips += 1
        margin_pct = dmax / (abs(logit_margin) + 1e-12) * 100

        sensitivities.append(dict(
            layer=j, scale_a=scale_a, delta_max=dmax,
            margin_pct=margin_pct, flipped=flipped,
            new_token=pert_token,
        ))

        flip_str = f"YES→'{tokenizer.decode([pert_token])}'" if flipped else "no"
        print(f"{j:>5} {scale_a:>10.5f} {dmax:>12.4f} {margin_pct:>9.1f}% {flip_str:>20}")

    # =========================================================================
    # Phase 4: Summary
    # =========================================================================

    print(f"\n{'='*85}")
    print("Summary")
    print(f"{'='*85}\n")

    la = np.array(local_norms)
    ca = np.array(crude_rhos)
    print(f"Local operator norm  — range [{la.min():.4f}, {la.max():.4f}], mean {la.mean():.4f}")
    print(f"Crude ρ              — range [{ca.min():.4f}, {ca.max():.4f}], mean {ca.mean():.4f}")
    print(f"Average tightening   — {(ca / (la + 1e-12)).mean():.1f}x")
    local_contr = int((la < 1.0).sum())
    print(f"Layers with local norm < 1: {local_contr}/{layer_count}")

    sa = np.array([s['delta_max'] for s in sensitivities])
    print(f"\nLogit-change range   — [{sa.min():.4f}, {sa.max():.4f}]")
    print(f"Logit margin         — {logit_margin:.4f}")
    print(f"Token flips          — {flips}/{layer_count}")

    print(f"\nTop-5 most sensitive layers:")
    ranked = sorted(sensitivities, key=lambda s: s['delta_max'], reverse=True)
    for s in ranked[:5]:
        extra = f"  FLIPS→'{tokenizer.decode([s['new_token']])}'" if s['flipped'] else ""
        print(f"  Layer {s['layer']:>2}: Δlogit={s['delta_max']:.4f}, "
              f"margin={s['margin_pct']:.1f}%{extra}")

    worst_margin = max(s['margin_pct'] for s in sensitivities)
    print(f"\nWorst single-layer margin consumed: {worst_margin:.1f}%")
    total_margin = sum(s['margin_pct'] for s in sensitivities)
    print(f"Sum of all layers' margin (conservative): {total_margin:.1f}%")

    if worst_margin < 100 and flips == 0:
        print("SAFE: no single-layer ±τ perturbation flips the output token.")
    elif flips > 0:
        print(f"UNSAFE: {flips} layers can individually flip the output token!")
    else:
        print("BORDERLINE: no flips but margin consumed exceeds 100% on some layers.")

    return dict(
        model=model_name,
        local_norms=local_norms, crude_rhos=crude_rhos,
        linf_norms=linf_norms, head_max_norms=head_max_norms,
        sensitivities=sensitivities,
        logit_margin=logit_margin, token_flips=flips,
        honest_token=honest_token, layer_count=layer_count,
    )


@app.local_entrypoint()
def main():
    for model_name in [
        "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
    ]:
        print(f"\n{'='*85}")
        print(f"  {model_name}")
        print(f"{'='*85}\n")
        r = measure_sensitivity.remote(model_name=model_name)
        print(f"\nResult: flips={r['token_flips']}/{r['layer_count']}, "
              f"margin={r['logit_margin']:.4f}")
