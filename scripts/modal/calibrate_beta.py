"""
Calibrate per-layer beta[l] — residual-to-margin gain factor.

For each layer l, measures how much a perturbation to the residual stream
at layer l's output shifts the winner-vs-runner-up logit margin:

    beta[l] = max_samples(|δΔ| / ||δx||∞)

where Δ = logit[top1] - logit[top2].

Method:
  1. Run greedy generation (max_tokens=1) → baseline logits
  2. For each layer l, install a forward hook that adds ε * random_sign
     to layer l+1's input_layernorm input (= layer l's output residual)
  3. Re-run generation → perturbed logits
  4. Compute gain = |δΔ_perturbed - δΔ_baseline| / ε

Reports per-layer beta with statistics across prompts and perturbation seeds.

Usage:
    modal run --detach scripts/modal/calibrate_beta.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-calibrate-beta")

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
]

EPSILON = 0.1  # L∞ perturbation magnitude
N_PERTURBATION_SEEDS = 3  # random sign perturbations per (prompt, layer)


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
    # Find transformer layers
    layers = None
    for attr in ["model.layers", "transformer.h"]:
        parts = attr.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            layers = obj
            break

    if layers is None:
        raise RuntimeError("Could not find transformer layers in model")

    n_layers = len(layers)
    print(f"[{model_name}] Found {n_layers} layers")

    # Find hidden dim from first layer's input_layernorm
    hidden_dim = None
    for name, mod in model.named_modules():
        if "input_layernorm" in name and hasattr(mod, 'weight'):
            hidden_dim = mod.weight.shape[0]
            break
    print(f"[{model_name}] hidden_dim={hidden_dim}")

    # Warmup
    print(f"[{model_name}] Warmup...")
    params = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    llm.generate(["Hello"], params)

    # ---- Calibration ----
    # For each prompt, get baseline logits, then perturb each layer.
    #
    # We use vLLM's logprobs to get the top logit values. With logprobs=N,
    # we get the top-N token logprobs. For margin computation, we need top-2.
    #
    # Since full vocab logprobs are expensive, we use a different approach:
    # install a simple logit capture hook ourselves.

    logit_captures = []

    def make_logit_hook():
        """Hook on LogitsProcessor to capture raw logits."""
        def hook(module, args, output):
            # output shape: (batch * seq_len, vocab_size)
            # For decode, batch=1 seq_len=1, so output is (1, vocab_size)
            logit_captures.append(output.detach().float().cpu())
        return hook

    # Install logit capture
    logit_hook_handle = None
    for name, mod in model.named_modules():
        if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
            logit_hook_handle = mod.register_forward_hook(make_logit_hook())
            print(f"[{model_name}] Installed logit capture on {name}")
            break

    if logit_hook_handle is None:
        raise RuntimeError("Could not find LogitsProcessor for logit capture")

    params = SamplingParams(temperature=0, max_tokens=1)
    rng = np.random.RandomState(42)

    # Per-layer gain measurements: gains[l] = list of |δΔ| / ε
    gains = {l: [] for l in range(n_layers)}

    t0 = time.time()

    for pi, prompt in enumerate(PROMPTS):
        # --- Baseline ---
        logit_captures.clear()
        llm.generate([prompt], params)
        if not logit_captures:
            print(f"  [{pi}] no logits captured, skip")
            continue
        # The last captured logits are from the decode step (first generated token).
        # During prefill, LogitsProcessor may fire for the prompt; the decode logits
        # are the last capture.
        baseline_logits = logit_captures[-1].numpy().flatten()

        # Compute baseline margin
        top_indices = np.argsort(baseline_logits)[::-1]
        top1, top2 = top_indices[0], top_indices[1]
        baseline_margin = float(baseline_logits[top1] - baseline_logits[top2])

        if baseline_margin <= 0:
            print(f"  [{pi}] margin={baseline_margin:.4f} <= 0, skip")
            continue

        # --- Per-layer perturbation ---
        for l in range(n_layers):
            for seed_idx in range(N_PERTURBATION_SEEDS):
                # Generate random sign perturbation
                signs = rng.choice([-1.0, 1.0], size=hidden_dim).astype(np.float32)
                perturbation_np = EPSILON * signs
                perturbation_tensor = torch.tensor(
                    perturbation_np, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)

                # Install perturbation hook.
                # Hook the input_layernorm of layer l+1 (or final norm for last layer)
                # using a pre-hook that modifies the input.
                perturb_applied = [False]

                if l < n_layers - 1:
                    target_module = layers[l + 1].input_layernorm
                else:
                    # Last layer: hook the final norm
                    target_module = None
                    for name, mod in model.named_modules():
                        if name.endswith(".norm") and "layer" not in name:
                            target_module = mod
                            break
                    if target_module is None:
                        # Try model.model.norm
                        target_module = getattr(model, 'model', model)
                        target_module = getattr(target_module, 'norm', None)

                if target_module is None:
                    continue

                def make_perturb_hook(pert_tensor, applied_flag):
                    def hook(module, args):
                        if applied_flag[0]:
                            return args  # only perturb once per forward pass
                        applied_flag[0] = True
                        h = args[0]
                        # Add perturbation (broadcast to match shape)
                        pert = pert_tensor.to(device=h.device, dtype=h.dtype)
                        # h shape: (batch_size * seq_len, hidden_dim) during decode
                        # For decode: (1, hidden_dim)
                        # Perturbation: (1, 1, hidden_dim) → squeeze to match
                        if h.dim() == 2:
                            pert = pert.squeeze(0).squeeze(0)  # (hidden_dim,)
                        elif h.dim() == 3:
                            pert = pert  # (1, 1, hidden_dim)
                        return (h + pert,) + args[1:]
                    return hook

                perturb_handle = target_module.register_forward_pre_hook(
                    make_perturb_hook(perturbation_tensor, perturb_applied)
                )

                # Run perturbed generation
                logit_captures.clear()
                llm.generate([prompt], params)
                perturb_handle.remove()

                if not logit_captures:
                    continue

                perturbed_logits = logit_captures[-1].numpy().flatten()

                # Compute perturbed margin (for same top1/top2 tokens)
                perturbed_margin = float(
                    perturbed_logits[top1] - perturbed_logits[top2]
                )

                # Gain = |δΔ| / ε
                delta_margin = abs(perturbed_margin - baseline_margin)
                gain = delta_margin / EPSILON
                gains[l].append(gain)

        elapsed = time.time() - t0
        print(f"  [{pi+1}/{len(PROMPTS)}] {elapsed:.0f}s elapsed")

    # Remove logit capture hook
    logit_hook_handle.remove()

    total_time = time.time() - t0
    print(f"\n[{model_name}] Calibration complete in {total_time:.0f}s")

    # --- Report ---
    print(f"\n{'='*70}")
    print(f"[{model_name}] Per-Layer Beta Calibration Results")
    print(f"  epsilon={EPSILON}, {len(PROMPTS)} prompts, "
          f"{N_PERTURBATION_SEEDS} seeds/layer")
    print(f"{'='*70}")

    betas = {}
    for l in range(n_layers):
        g = gains[l]
        if not g:
            print(f"  L{l:2d}: no measurements")
            betas[l] = None
            continue
        g_arr = np.array(g)
        median_g = np.median(g_arr)
        p90_g = np.percentile(g_arr, 90)
        p99_g = np.percentile(g_arr, 99)
        max_g = np.max(g_arr)
        # Conservative beta: p99 * 2x safety factor
        beta = p99_g * 2.0
        betas[l] = beta
        print(f"  L{l:2d}: median={median_g:8.4f}  p90={p90_g:8.4f}  "
              f"p99={p99_g:8.4f}  max={max_g:8.4f}  "
              f"beta(p99*2x)={beta:8.4f}  (n={len(g)})")

    # Summary
    beta_values = [v for v in betas.values() if v is not None]
    if beta_values:
        print(f"\n  Summary:")
        print(f"    beta min:    {min(beta_values):.4f}")
        print(f"    beta median: {np.median(beta_values):.4f}")
        print(f"    beta max:    {max(beta_values):.4f}")
        print(f"    beta mean:   {np.mean(beta_values):.4f}")

        # What this means for certification
        # Current bound without beta: Σ alpha * eps
        # New bound: Σ beta[l] * alpha[l] * eps (where alpha[l] = Σ_h alpha[l,h])
        # If beta < 1, the bound shrinks
        print(f"\n  Interpretation:")
        n_below_1 = sum(1 for b in beta_values if b < 1.0)
        n_below_01 = sum(1 for b in beta_values if b < 0.1)
        print(f"    Layers with beta < 1.0: {n_below_1}/{len(beta_values)}")
        print(f"    Layers with beta < 0.1: {n_below_01}/{len(beta_values)}")
        if np.mean(beta_values) < 1.0:
            avg_reduction = 1.0 / np.mean(beta_values)
            print(f"    Average bound reduction factor: {avg_reduction:.1f}x")
        else:
            print(f"    WARNING: beta > 1 on average — bound gets LOOSER with beta")
            print(f"    This means the current bound is already too optimistic")
            print(f"    (comparing residual L∞ to logit margin without lm_head gain)")

    print(f"\n{'='*70}")

    return {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "epsilon": EPSILON,
        "n_prompts": len(PROMPTS),
        "n_seeds": N_PERTURBATION_SEEDS,
        "betas": betas,
        "gains": {l: list(g) for l, g in gains.items()},
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_llama():
    return _run_model("llama", MODELS["llama"])


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_qwen():
    return _run_model("qwen", MODELS["qwen"])


@app.local_entrypoint()
def main():
    print("Beta[l] Calibration — Residual-to-Margin Gain")
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
        betas = result["betas"]
        beta_values = [v for v in betas.values() if v is not None]
        if beta_values:
            import numpy as np
            print(f"\n{model}: beta range [{min(beta_values):.4f}, {max(beta_values):.4f}]"
                  f"  median={np.median(beta_values):.4f}  mean={np.mean(beta_values):.4f}")
            for l in sorted(betas.keys()):
                if betas[l] is not None:
                    print(f"  L{l:2d}: beta={betas[l]:.4f}")
