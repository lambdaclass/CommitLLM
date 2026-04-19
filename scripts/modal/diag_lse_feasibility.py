"""
LSE feasibility spike: capture softmax LSE from vLLM's bundled flash_attn.

Answers two questions:
1. Does CPU replay recover the kernel's LSE? (normalization agreement)
2. Does using captured LSE collapse the actual attn_out gap? (P@V gap)

Both questions are answered by the Rust-side `measure_corridor_with_captured_lse`
function, which computes CPU LSE from witnessed scores, compares it against
captured GPU LSE, and replays P@V using the GPU normalization constant.

The baseline comparison is `measure_corridor_witnessed_scores` which uses
standard f64 global softmax.

Hard gate: if LSE-conditioned L-inf does NOT collapse sharply vs baseline,
stop and move to deterministic attention kernels.

Usage:
    modal run --detach scripts/modal/diag_lse_feasibility.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-lse-feasibility")

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

PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
    "Tell me a fun fact about space.",
]

N_ITERS = 5
MAX_TOKENS = 32


def _run_model(model_name, model_id):
    import hashlib
    import json
    import random

    import numpy as np

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # ── Step 1: Patch vllm.vllm_flash_attn to capture LSE ──
    print(f"\n[{model_name}] Patching vllm.vllm_flash_attn for LSE capture...")

    captured_lse = []  # One entry per FA call: lse_tensor
    import vllm.vllm_flash_attn as vfa

    orig_varlen = vfa.flash_attn_varlen_func

    def patched_varlen(*args, **kwargs):
        kwargs['return_softmax_lse'] = True
        result = orig_varlen(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            captured_lse.append(result[1].detach().cpu())
            return result[0]
        return result

    vfa.flash_attn_varlen_func = patched_varlen

    orig_kvcache = vfa.flash_attn_with_kvcache

    def patched_kvcache(*args, **kwargs):
        kwargs['return_softmax_lse'] = True
        result = orig_kvcache(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            captured_lse.append(result[1].detach().cpu())
            return result[0]
        return result

    vfa.flash_attn_with_kvcache = patched_kvcache
    print(f"  Patched flash_attn_varlen_func and flash_attn_with_kvcache")

    # ── Step 2: Load model ──
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer
    import verilm_rs

    print(f"\n[{model_name}] Loading {model_id}...")
    llm = LLM(model=model_id, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    n_q_heads = cap._num_heads
    n_kv_heads = cap._num_kv_heads
    d_head = cap._head_dim
    model_dir = server._model_dir
    full_layers = list(range(n_layers))

    print(f"  {n_layers} layers, {n_q_heads}q/{n_kv_heads}kv, d_head={d_head}")

    # Generate verifier key.
    seed = hashlib.sha256(f"diag-lse-{model_name}".encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, list(seed))

    # ── Step 3: Warmup ──
    print(f"\n[{model_name}] Warmup...")
    captured_lse.clear()
    server.chat(prompt="Hello", max_tokens=8, temperature=0.0)
    warmup_count = len(captured_lse)
    print(f"  Warmup LSE captures: {warmup_count}")
    if captured_lse:
        print(f"  First LSE shape: {captured_lse[0].shape}, dtype: {captured_lse[0].dtype}")

    # ── Step 4: Measurement runs ──
    print(f"\n{'='*60}")
    print(f"[{model_name}] Measurement: {N_ITERS} iterations")
    print(f"{'='*60}")

    rng = random.Random(42)
    all_results = []

    for i in range(N_ITERS):
        prompt = PROMPTS[i % len(PROMPTS)]
        captured_lse.clear()

        result = server.chat(
            prompt=prompt, max_tokens=MAX_TOKENS,
            temperature=0.8, top_k=50, top_p=0.9,
        )
        n_tokens = result["n_tokens"]
        n_prompt = result["commitment"].get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt

        if n_gen <= 0:
            print(f"  [{i}] skip: n_gen={n_gen}")
            continue

        # How many LSE did we capture?
        n_lse = len(captured_lse)
        # Expected: n_layers calls per forward pass.
        # 1 prefill + n_gen decode = (1 + n_gen) * n_layers
        expected_lse = (1 + n_gen) * n_layers
        lse_match = "OK" if n_lse == expected_lse else f"MISMATCH({n_lse}!={expected_lse})"

        # Separate prefill vs decode LSE.
        # First n_layers captures = prefill. Remaining = decode.
        decode_lse_all = captured_lse[n_layers:]
        n_decode_tokens = len(decode_lse_all) // n_layers if n_layers > 0 else 0

        print(f"\n  [{i}] n_prompt={n_prompt}, n_gen={n_gen}, "
              f"lse_captures={n_lse} ({lse_match}), "
              f"decode_tokens_with_lse={n_decode_tokens}")

        if not decode_lse_all:
            print(f"    No decode LSE captured")
            continue

        # Pick a random decode token to challenge.
        if n_decode_tokens > 1:
            tok_offset = rng.randint(0, n_decode_tokens - 1)
        else:
            tok_offset = 0

        # Extract per-layer LSE for this decode token.
        # decode_lse_all[t*n_layers + l] is token t, layer l.
        tok_lse_per_layer = []
        for l in range(n_layers):
            idx = tok_offset * n_layers + l
            if idx < len(decode_lse_all):
                # Shape: (1, n_q_heads) or (n_q_heads,) — flatten to 1D.
                lse_np = decode_lse_all[idx].numpy().flatten()
                tok_lse_per_layer.append(lse_np.tolist())

        if len(tok_lse_per_layer) < n_layers:
            print(f"    Incomplete LSE for token {tok_offset}: {len(tok_lse_per_layer)}/{n_layers}")
            continue

        print(f"    Challenging decode token {tok_offset}")
        print(f"    LSE per layer: {len(tok_lse_per_layer)} layers, "
              f"{len(tok_lse_per_layer[0])} heads/layer")

        # Get audit for this token.
        challenge_idx = n_prompt - 1 + tok_offset  # absolute token index
        audit_binary = server.audit(
            request_id=result["request_id"],
            token_index=challenge_idx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        audit_bytes = bytes(audit_binary)

        # ── Baseline: global softmax replay (current behavior) ──
        try:
            report_baseline = json.loads(verilm_rs.measure_corridor_witnessed_scores(
                audit_bytes, key_json,
            ))
            baseline_linf = report_baseline.get("global_linf", -1)
        except Exception as e:
            print(f"    Baseline measurement failed: {e}")
            baseline_linf = -1

        # ── LSE-conditioned replay (the test) ──
        try:
            per_layer_lse_json = json.dumps(tok_lse_per_layer)
            report_lse = json.loads(verilm_rs.measure_corridor_with_captured_lse(
                audit_bytes, key_json, per_layer_lse_json,
            ))
            lse_linf = report_lse["corridor"]["global_linf"]
            q1_max_gap = report_lse["global_lse_max_gap"]
            q1_mean_gap = report_lse["global_lse_mean_gap"]

            # Per-layer details.
            lse_per_layer_linf = report_lse["corridor"].get("per_layer_max_linf", [])
            baseline_per_layer_linf = report_baseline.get("per_layer_max_linf", [])
        except Exception as e:
            print(f"    LSE corridor measurement failed: {e}")
            lse_linf = -1
            q1_max_gap = -1
            q1_mean_gap = -1
            lse_per_layer_linf = []
            baseline_per_layer_linf = []

        # Delta: how much did LSE-conditioned replay improve?
        delta = baseline_linf - lse_linf if baseline_linf >= 0 and lse_linf >= 0 else None

        print(f"    Q1 (LSE agreement): max|CPU-GPU|={q1_max_gap:.4f}, mean={q1_mean_gap:.4f}")
        print(f"    Q2 (attention gap):")
        print(f"      Baseline (global softmax): L-inf = {baseline_linf}")
        print(f"      LSE-conditioned:           L-inf = {lse_linf}")
        print(f"      Delta:                     {delta}")

        # Show worst 3 layers for both.
        if lse_per_layer_linf and baseline_per_layer_linf:
            print(f"    Per-layer comparison (worst 5):")
            paired = list(zip(range(n_layers), baseline_per_layer_linf, lse_per_layer_linf))
            paired.sort(key=lambda x: -x[1])
            for l, bl, ll in paired[:5]:
                d = bl - ll
                print(f"      layer {l:2d}: baseline={bl:4d}  lse={ll:4d}  delta={d:+d}")

        row = {
            "iter": i,
            "model": model_name,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "tok_offset": tok_offset,
            "n_lse_captured": n_lse,
            "q1_lse_max_gap": round(q1_max_gap, 6),
            "q1_lse_mean_gap": round(q1_mean_gap, 6),
            "baseline_linf": baseline_linf,
            "lse_linf": lse_linf,
            "delta": delta,
        }
        all_results.append(row)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"[{model_name}] LSE Feasibility Summary")
    print(f"{'='*60}")

    if all_results:
        lse_gaps = [r["q1_lse_max_gap"] for r in all_results if r["q1_lse_max_gap"] >= 0]
        baselines = [r["baseline_linf"] for r in all_results if r["baseline_linf"] >= 0]
        lse_linfs = [r["lse_linf"] for r in all_results if r["lse_linf"] >= 0]
        deltas = [r["delta"] for r in all_results if r["delta"] is not None]

        print(f"\n  Q1 (CPU vs GPU LSE agreement):")
        if lse_gaps:
            print(f"    max |CPU - GPU|: {max(lse_gaps):.6f}")
            print(f"    mean: {np.mean(lse_gaps):.6f}")
        else:
            print(f"    no data")

        print(f"\n  Q2 (attention output gap):")
        if baselines:
            print(f"    Baseline L-inf range: {min(baselines)} - {max(baselines)}")
        if lse_linfs:
            print(f"    LSE L-inf range:      {min(lse_linfs)} - {max(lse_linfs)}")
        if deltas:
            print(f"    Delta range:          {min(deltas)} - {max(deltas)}")
            print(f"    Mean delta:           {np.mean(deltas):.1f}")

        # Verdict.
        if deltas:
            mean_baseline = np.mean(baselines) if baselines else 0
            mean_lse = np.mean(lse_linfs) if lse_linfs else 0
            reduction_pct = (1 - mean_lse / mean_baseline) * 100 if mean_baseline > 0 else 0

            print(f"\n  VERDICT:")
            if mean_lse <= 5:
                print(f"    LSE COLLAPSES GAP ({reduction_pct:.0f}% reduction)")
                print(f"    → Normalization was the bottleneck. Stock-kernel witness viable.")
            elif reduction_pct > 50:
                print(f"    SIGNIFICANT REDUCTION ({reduction_pct:.0f}%)")
                print(f"    → Normalization matters but V aggregation also contributes.")
            elif reduction_pct > 10:
                print(f"    MODEST REDUCTION ({reduction_pct:.0f}%)")
                print(f"    → Normalization helps but V aggregation is dominant.")
            else:
                print(f"    NO IMPROVEMENT (delta ≈ 0, {reduction_pct:.0f}%)")
                print(f"    → Gap is entirely in V aggregation precision.")
                print(f"    → Stop stock-kernel work. Move to deterministic attention kernels.")
    else:
        print(f"  No results collected.")

    print(f"{'='*60}")

    # Restore originals.
    vfa.flash_attn_varlen_func = orig_varlen
    vfa.flash_attn_with_kvcache = orig_kvcache

    return {
        "model": model_name,
        "results": all_results,
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
    print("LSE Feasibility Spike — Full Measurement")
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
        rs = result["results"]
        if not rs:
            print(f"\n{model}: no results")
            continue

        lse_gaps = [r["q1_lse_max_gap"] for r in rs if r["q1_lse_max_gap"] >= 0]
        baselines = [r["baseline_linf"] for r in rs if r["baseline_linf"] >= 0]
        lse_linfs = [r["lse_linf"] for r in rs if r["lse_linf"] >= 0]
        deltas = [r["delta"] for r in rs if r["delta"] is not None]

        print(f"\n{model}:")
        if lse_gaps:
            print(f"  Q1 max |CPU-GPU LSE|: {max(lse_gaps):.6f}")
        if baselines and lse_linfs:
            print(f"  Q2 Baseline L-inf: {min(baselines)}-{max(baselines)}")
            print(f"  Q2 LSE L-inf:      {min(lse_linfs)}-{max(lse_linfs)}")
        if deltas:
            print(f"  Delta range: {min(deltas)} - {max(deltas)}")
