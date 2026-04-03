"""
Measure RMSNorm contraction ratio ρ_j on real models.

For each layer j, computes:
    ρ_j = ‖W_o^(j)‖₂ · scale_wo · scale_a · max(γ_j) / RMS(residual^(j))

If ρ_j < 1 at every layer, the ±τ attention corridor tolerance is safe:
each per-layer injection gets dampened before it reaches the residual stream,
and accumulated drift is negligible.

This is the cheapest measurement that can resolve the corridor amplification
question (roadmap #1). One forward pass, dump residual norms and weight
spectral norms.

Usage:
    modal run redteam/modal/measure_contraction.py
"""

import modal

app = modal.App("commitllm-contraction")

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
    timeout=1800,
)
def measure_contraction(model_name: str = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"):
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

    # Use a representative prompt
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Storage for per-layer measurements
    residuals = {}
    layer_count = model.config.num_hidden_layers
    print(f"Model has {layer_count} layers, hidden_dim={model.config.hidden_size}")

    # Hook to capture residual stream before each layer's attention RMSNorm
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # input[0] is the hidden_states (residual stream) entering this layer
            hidden = input[0].detach().float()
            # RMS of the residual at this layer
            rms = torch.sqrt(torch.mean(hidden ** 2, dim=-1)).mean().item()
            residuals[layer_idx] = rms
        return hook_fn

    # Attach hooks to each decoder layer
    for i in range(layer_count):
        layer = model.model.layers[i]
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Now measure W_o spectral norms and RMSNorm weights
    print(f"\n{'layer':>5} {'‖W_o‖₂':>12} {'max(γ)':>12} {'RMS(resid)':>12} {'ρ_j':>12} {'contracts?':>10}")
    print("-" * 75)

    all_rho = []

    for i in range(layer_count):
        layer = model.model.layers[i]

        # W_o weight matrix
        wo = layer.self_attn.o_proj.weight.detach().float()
        # Spectral norm via torch.linalg.svdvals (top singular value)
        wo_spectral = torch.linalg.svdvals(wo)[0].item()

        # RMSNorm weights (gamma) — the input_layernorm normalizes before attention
        gamma = layer.input_layernorm.weight.detach().float()
        gamma_max = gamma.abs().max().item()

        # RMS of residual at this layer
        rms_resid = residuals.get(i, 1.0)

        # Contraction ratio
        # In real models, W_o operates in float16 and the result is added to
        # the residual stream. The key ratio is whether the perturbation
        # (after W_o and before RMSNorm) is smaller than the residual magnitude.
        rho = wo_spectral * gamma_max / rms_resid

        all_rho.append(rho)

        print(
            f"{i:>5} {wo_spectral:>12.2f} {gamma_max:>12.4f} "
            f"{rms_resid:>12.4f} {rho:>12.4f} {'YES' if rho < 1.0 else 'NO':>10}"
        )

    rho_arr = np.array(all_rho)
    print(f"\n=== Summary for {model_name} ===")
    print(f"ρ range: [{rho_arr.min():.4f}, {rho_arr.max():.4f}], mean={rho_arr.mean():.4f}")
    contracting = int((rho_arr < 1.0).sum())
    print(f"Contracting layers: {contracting}/{layer_count} ({100*contracting/layer_count:.0f}%)")

    if contracting == layer_count:
        print("ALL layers contract — corridor amplification decays exponentially.")
        print("The ±τ per-layer tolerance is safe. Score witnessing is optional hardening.")
    else:
        expanding = layer_count - contracting
        print(f"{expanding} layers EXPAND (ρ > 1).")
        print("Corridor amplification may be feasible. Score witnessing becomes critical.")

    # Also compute the accumulated drift bound
    # Each layer j injects ±τ which gets dampened by product of ρ from j+1 to L
    tau = 10  # typical i8 corridor tolerance
    total_drift = 0.0
    for j in range(layer_count):
        # Product of ρ from j+1 to L-1
        product = 1.0
        for k in range(j + 1, layer_count):
            product *= rho_arr[k]
        total_drift += tau * product

    print(f"\nAccumulated drift bound (τ={tau}): {total_drift:.6f}")
    print(f"For comparison, final RMS(residual) = {residuals.get(layer_count - 1, 0):.2f}")
    drift_ratio = total_drift / residuals.get(layer_count - 1, 1.0)
    print(f"Drift / final residual = {drift_ratio:.2e}")

    return {
        "model": model_name,
        "rho": all_rho,
        "residuals": dict(residuals),
        "all_contracting": contracting == layer_count,
        "total_drift": total_drift,
    }


@app.local_entrypoint()
def main():
    # Run on both target models
    for model_name in [
        "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
    ]:
        print(f"\n{'='*75}")
        print(f"  Measuring contraction for {model_name}")
        print(f"{'='*75}\n")
        result = measure_contraction.remote(model_name=model_name)
        print(f"\nResult: all_contracting={result['all_contracting']}, "
              f"total_drift={result['total_drift']:.6f}")
