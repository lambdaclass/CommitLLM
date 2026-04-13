"""
Decode boundary spike: falsification test for Option C (captured final_hidden)
and cost estimation for Options A/B (captured logits / deterministic kernel).

For each generated token:
  1. Capture GPU's post-norm final_hidden (bf16 upcast to f32)
  2. Recompute lm_head matmul from captured final_hidden (i8 quantize → i32 matmul)
  3. Check if argmax of recomputed logits matches the actual generated token
  4. Measure cost: bytes/token for each capture option

Key question (Option C kill test):
  If we use the GPU's actual post-norm hidden instead of the bridge-derived one,
  does the i8→i32 lm_head matmul produce logits with the correct argmax?
  If not, the divergence is in the lm_head quantization, not upstream.

Usage:
    modal run --detach scripts/modal/diag_decode_boundary.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-decode-boundary")

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"

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
        ".git", "target", "scripts/__pycache__", "*.pdf", "site",
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
]
MAX_TOKENS = 64


def _run_diag():
    import hashlib
    import json
    import time
    import numpy as np
    import torch

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM, SamplingParams
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir

    # Get the actual model for direct access to norm + lm_head
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    final_norm = None
    lm_head_weight = None

    # Walk modules to find model.norm and lm_head
    for name, mod in model.named_modules():
        if name == "model.norm" or name.endswith(".norm") and "layer" not in name:
            if hasattr(mod, 'weight') and mod.weight is not None:
                if mod.weight.shape[0] == 3584:  # hidden_dim for Qwen 7B
                    final_norm = mod
                    print(f"  Found final_norm: {name}, weight shape={mod.weight.shape}")
    for name, param in model.named_parameters():
        if "lm_head" in name and "weight" in name:
            lm_head_weight = param
            print(f"  Found lm_head: {name}, shape={param.shape}, dtype={param.dtype}")

    if final_norm is None or lm_head_weight is None:
        print("ERROR: Could not find final_norm or lm_head")
        return {}

    # ── Step 0: Verify output[1] matches LogitsProcessor input ──
    # Gate: if these don't match, model.norm output[1] is not the decode boundary.
    print(f"\n{'='*60}")
    print("STEP 0: Verify model.norm output[1] matches LogitsProcessor input")
    print(f"{'='*60}")

    norm_out1_captures = []
    lp_input_captures = []

    def verify_norm_hook(module, args, output):
        if isinstance(output, tuple) and len(output) >= 2:
            norm_out1_captures.append(output[1].detach().cpu().float())

    def verify_lp_hook(module, args, output):
        # LogitsProcessor args: (lm_head_weight, hidden_states, sampling_metadata)
        if len(args) >= 2 and isinstance(args[1], torch.Tensor):
            lp_input_captures.append(args[1].detach().cpu().float())

    h_norm = final_norm.register_forward_hook(verify_norm_hook)

    logits_proc = None
    for name, mod in model.named_modules():
        if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
            logits_proc = mod
            break

    h_lp = logits_proc.register_forward_hook(verify_lp_hook) if logits_proc else None

    from vllm import SamplingParams as SP
    verify_out = llm.generate(["What is 2+2?"], SP(max_tokens=4, temperature=0.0))[0]
    print(f"  Verify gen: {list(verify_out.outputs[0].token_ids)}")

    h_norm.remove()
    if h_lp:
        h_lp.remove()

    boundary_ok = True
    for i, (norm_t, lp_t) in enumerate(zip(norm_out1_captures, lp_input_captures)):
        # LogitsProcessor receives pruned hidden (last row for prefill, full for decode)
        # norm output is (seq_len, hidden_dim), LP input is (1, hidden_dim) for decode
        if norm_t.shape[0] > lp_t.shape[0]:
            # Prefill: LP gets last row after _prune_hidden_states
            norm_last = norm_t[-1:, :]
            diff = (norm_last - lp_t).abs().max().item()
            print(f"  fwd {i}: norm_out1 {tuple(norm_t.shape)} → last row vs LP input {tuple(lp_t.shape)}: max_diff={diff:.6f}")
        else:
            diff = (norm_t - lp_t).abs().max().item()
            print(f"  fwd {i}: norm_out1 {tuple(norm_t.shape)} vs LP input {tuple(lp_t.shape)}: max_diff={diff:.6f}")
        if diff > 0.01:
            print(f"  FAIL: norm output[1] does NOT match LogitsProcessor input (diff={diff})")
            boundary_ok = False

    if boundary_ok:
        print("  PASS: model.norm output[1] matches LogitsProcessor input for all forward passes")
    else:
        print("  FAIL: model.norm output[1] does NOT match LogitsProcessor input")
        print("  The real decode boundary is later — should capture at LogitsProcessor instead")

    # ── Setup for main test ──
    captured_post_norm = []

    def norm_output_hook(module, input, output):
        # vLLM's fused RMSNorm returns (residual, normed).
        # output[0] is the residual passthrough (== input[0]).
        # output[1] is the actual post-norm hidden state.
        if isinstance(output, tuple) and len(output) >= 2:
            captured_post_norm.append(output[1].detach().cpu())
        elif isinstance(output, tuple):
            captured_post_norm.append(output[0].detach().cpu())
        else:
            captured_post_norm.append(output.detach().cpu())

    norm_handle = final_norm.register_forward_hook(norm_output_hook)

    seed_bytes = hashlib.sha256(PROMPTS[0].encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed_bytes)
    key = json.loads(key_json)
    vocab_size = key.get("config", {}).get("vocab_size", 152064)
    hidden_dim = key.get("config", {}).get("hidden_dim", 3584)

    print(f"\n  vocab_size={vocab_size}, hidden_dim={hidden_dim}")

    # ── Cost estimation ──
    print(f"\n{'='*60}")
    print("COST ESTIMATION (per token)")
    print(f"{'='*60}")
    print(f"  Option C (captured final_hidden bf16): {hidden_dim * 2} bytes = {hidden_dim * 2 / 1024:.1f} KB")
    print(f"  Option C (captured final_hidden fp32): {hidden_dim * 4} bytes = {hidden_dim * 4 / 1024:.1f} KB")
    print(f"  Option A (captured logits bf16):       {vocab_size * 2} bytes = {vocab_size * 2 / 1024:.1f} KB")
    print(f"  Option A (captured logits fp32):       {vocab_size * 4} bytes = {vocab_size * 4 / 1024:.1f} KB")
    print(f"  Current final_residual (fp32):         {hidden_dim * 4} bytes = {hidden_dim * 4 / 1024:.1f} KB")

    for n_tok in [32, 64, 128, 256]:
        print(f"\n  {n_tok}-token retained-state totals:")
        print(f"    Option C bf16: {n_tok * hidden_dim * 2 / 1024 / 1024:.2f} MB")
        print(f"    Option A bf16: {n_tok * vocab_size * 2 / 1024 / 1024:.2f} MB")
        print(f"    Option A fp32: {n_tok * vocab_size * 4 / 1024 / 1024:.2f} MB")

    # ── Option C falsification test ──
    print(f"\n{'='*60}")
    print("OPTION C FALSIFICATION TEST")
    print("Question: does captured post-norm final_hidden → i8 quantize → i32 lm_head")
    print("          give the correct argmax token?")
    print(f"{'='*60}")

    # Load lm_head weights as i8 for quantized matmul (same as verifier)
    lm_head_np = lm_head_weight.detach().cpu().float().numpy()
    lm_head_scale = np.abs(lm_head_np).max() / 127.0
    lm_head_i8 = np.clip(np.round(lm_head_np / lm_head_scale), -128, 127).astype(np.int8)

    # Also get the keygen's lm_head quantization for comparison
    print(f"  lm_head bf16 range: [{lm_head_np.min():.4f}, {lm_head_np.max():.4f}]")
    print(f"  lm_head i8 scale: {lm_head_scale:.6f}")

    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {pi}: {prompt[:50]}... ---")

        captured_post_norm.clear()

        # Generate with greedy (temperature=0) for clean argmax comparison
        cap._capture_mode = "minimal"
        result = server.chat(prompt=prompt, max_tokens=MAX_TOKENS, temperature=0.0)
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        gen_token_ids = list(result.get("token_ids", []))

        print(f"  n_tokens={n_tokens}, n_prompt={n_prompt}, captured_norms={len(captured_post_norm)}")

        if not captured_post_norm:
            print("  ERROR: no post-norm captures")
            continue

        # Flatten post-norm captures: prefill gives (n_prompt, hidden_dim),
        # each decode gives (1, hidden_dim)
        all_hidden = []
        for t in captured_post_norm:
            if t.dim() >= 2:
                for pos in range(t.shape[0]):
                    all_hidden.append(t[pos].float().numpy())
            else:
                all_hidden.append(t.float().numpy())

        print(f"  flattened post-norm captures: {len(all_hidden)} positions")

        # For generated tokens, the relevant post-norm hidden is at
        # positions n_prompt-1 (predicting gen[0]), n_prompt (predicting gen[1]), etc.
        # gen_start index in the capture array = n_prompt - 1
        gen_start_capture = n_prompt - 1

        exact_c = 0
        total_gen = 0

        # Also compute: what if we use the captured hidden with FLOAT lm_head (no quantization)?
        exact_float = 0

        for g in range(min(len(gen_token_ids), MAX_TOKENS)):
            capture_idx = gen_start_capture + g
            if capture_idx >= len(all_hidden):
                break

            actual_token = gen_token_ids[g]
            hidden = all_hidden[capture_idx]
            total_gen += 1

            # Option C path: captured hidden → i8 quantize → i32 matmul
            hidden_i8 = np.clip(np.round(hidden), -128, 127).astype(np.int8)
            logits_i32 = lm_head_i8.astype(np.int32) @ hidden_i8.astype(np.int32)
            argmax_i32 = int(np.argmax(logits_i32))

            # Float path: captured hidden → float lm_head matmul (no quantization)
            logits_float = lm_head_np @ hidden
            argmax_float = int(np.argmax(logits_float))

            # GPU path: captured hidden → bf16 lm_head (what GPU actually does)
            hidden_bf16 = torch.tensor(hidden).bfloat16()
            lm_head_bf16 = lm_head_weight.detach().cpu().bfloat16()
            logits_gpu_like = (lm_head_bf16 @ hidden_bf16).float().numpy()
            argmax_gpu_like = int(np.argmax(logits_gpu_like))

            is_exact_c = (argmax_i32 == actual_token)
            is_exact_float = (argmax_float == actual_token)
            is_exact_gpu = (argmax_gpu_like == actual_token)

            if is_exact_c:
                exact_c += 1
            if is_exact_float:
                exact_float += 1

            if not is_exact_c or not is_exact_float or not is_exact_gpu:
                gap_i32 = int(logits_i32[argmax_i32] - logits_i32[actual_token])
                gap_float = float(logits_float[argmax_float] - logits_float[actual_token])
                gap_gpu = float(logits_gpu_like[argmax_gpu_like] - logits_gpu_like[actual_token])
                rank_i32 = int(np.sum(logits_i32 > logits_i32[actual_token]))
                rank_float = int(np.sum(logits_float > logits_float[actual_token]))
                rank_gpu = int(np.sum(logits_gpu_like > logits_gpu_like[actual_token]))
                print(f"  gen[{g}] tid={actual_token}: "
                      f"i32={'OK' if is_exact_c else f'MISS(argmax={argmax_i32},rank={rank_i32},gap={gap_i32})'} "
                      f"fp32={'OK' if is_exact_float else f'MISS(argmax={argmax_float},rank={rank_float},gap={gap_float:.1f})'} "
                      f"bf16={'OK' if is_exact_gpu else f'MISS(argmax={argmax_gpu_like},rank={rank_gpu},gap={gap_gpu:.1f})'}")

            all_results.append({
                "prompt": pi, "gen_idx": g, "token": actual_token,
                "exact_i32": is_exact_c, "exact_float": is_exact_float,
                "exact_gpu": is_exact_gpu,
            })

        print(f"  Option C (i8→i32):   {exact_c}/{total_gen} exact ({100*exact_c/max(total_gen,1):.1f}%)")
        print(f"  Float (fp32 matmul): {exact_float}/{total_gen} exact ({100*exact_float/max(total_gen,1):.1f}%)")

    norm_handle.remove()

    # ── Summary ──
    total = len(all_results)
    total_exact_c = sum(1 for r in all_results if r["exact_i32"])
    total_exact_float = sum(1 for r in all_results if r["exact_float"])
    total_exact_gpu = sum(1 for r in all_results if r["exact_gpu"])

    print(f"\n{'='*60}")
    print(f"SUMMARY ({total} generated tokens across {len(PROMPTS)} prompts)")
    print(f"{'='*60}")
    print(f"  Option C (captured hidden → i8 → i32 lm_head): {total_exact_c}/{total} exact ({100*total_exact_c/max(total,1):.1f}%)")
    print(f"  Float path (captured hidden → fp32 lm_head):    {total_exact_float}/{total} exact ({100*total_exact_float/max(total,1):.1f}%)")
    print(f"  GPU-like (captured hidden → bf16 lm_head):      {total_exact_gpu}/{total} exact ({100*total_exact_gpu/max(total,1):.1f}%)")
    print()
    print("  DECISION:")
    if total_exact_c == total:
        print("  Option C PASSES: i8→i32 lm_head from captured hidden gives exact token identity.")
        print("  The divergence was upstream (bridge-derived hidden), not lm_head quantization.")
    elif total_exact_gpu == total:
        print("  Option C KILLED for i8→i32 path, but bf16 lm_head from captured hidden is exact.")
        print("  The divergence is in i8 quantization of lm_head, not upstream hidden state.")
        print("  → captured float logits (Option A) or deterministic lm_head (Option B) needed.")
    elif total_exact_float == total:
        print("  fp32 lm_head from captured hidden is exact, bf16 is not.")
        print("  → precision loss in bf16 lm_head matmul. Captured fp32 logits may work.")
    else:
        print("  Even fp32 lm_head from captured hidden does not give exact token identity.")
        print("  → the problem is deeper than lm_head precision. Investigate further.")

    print(f"\n{'='*60}")
    print("COST SUMMARY")
    print(f"{'='*60}")
    print(f"  Option C (final_hidden bf16): {hidden_dim * 2 / 1024:.1f} KB/token, {64 * hidden_dim * 2 / 1024 / 1024:.2f} MB for 64 tokens")
    print(f"  Option A (logits bf16):       {vocab_size * 2 / 1024:.1f} KB/token, {64 * vocab_size * 2 / 1024 / 1024:.2f} MB for 64 tokens")
    print(f"  Option A (logits fp32):       {vocab_size * 4 / 1024:.1f} KB/token, {64 * vocab_size * 4 / 1024 / 1024:.2f} MB for 64 tokens")

    return {
        "total": total,
        "exact_i32": total_exact_c,
        "exact_float": total_exact_float,
        "exact_gpu": total_exact_gpu,
    }


@app.function(image=image, gpu="A100-80GB", timeout=1200)
def run_diag():
    return _run_diag()


@app.local_entrypoint()
def main():
    print("Decode Boundary Spike: Option C falsification + cost estimation")
    print("=" * 60)
    result = run_diag.remote()
    print(f"\nResults: {result}")
