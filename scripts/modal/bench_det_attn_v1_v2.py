"""
Deterministic attention kernel benchmark: v1 (serial scores) vs v2 (tree-reduced scores).

Measures kernel-only latency on identical random bf16 inputs across multiple
seq_len values. Includes stock PyTorch SDPA as baseline reference.

Config: Llama 8B (n_q_heads=32, n_kv_heads=8, d_head=128)

Run: modal run --detach scripts/modal/bench_det_attn_v1_v2.py
"""

import modal

app = modal.App("det-attn-v1-v2-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
    .pip_install("numpy", "torch==2.5.1", extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .add_local_dir("kernels", remote_path="/build/kernels", copy=True)
    .run_commands(
        "nvcc -O2 --fmad=false -shared -Xcompiler -fPIC "
        "-o /build/libdet_attn_v1.so /build/kernels/deterministic_attention_v1.cu",
        "nvcc -O2 --fmad=false -shared -Xcompiler -fPIC "
        "-o /build/libdet_attn_v2.so /build/kernels/deterministic_attention.cu",
    )
)


@app.function(image=image, gpu="A100-80GB", timeout=600)
def run_benchmark():
    import ctypes
    import math
    import time

    import numpy as np
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    # ── Load kernels ──────────────────────────────────────────────
    v1_lib = ctypes.CDLL("/build/libdet_attn_v1.so")
    v2_lib = ctypes.CDLL("/build/libdet_attn_v2.so")

    for lib in [v1_lib, v2_lib]:
        lib.deterministic_attention.restype = ctypes.c_int
        lib.deterministic_attention.argtypes = [
            ctypes.c_void_p,  # q_dev
            ctypes.c_void_p,  # k_dev
            ctypes.c_void_p,  # v_dev
            ctypes.c_void_p,  # output_dev
            ctypes.c_void_p,  # weights_dev
            ctypes.c_int,     # n_q_heads
            ctypes.c_int,     # n_kv_heads
            ctypes.c_int,     # d_head
            ctypes.c_int,     # seq_len
            ctypes.c_float,   # inv_sqrt_d
        ]

    # ── Llama 8B config ──────────────────────────────────────────
    N_Q_HEADS = 32
    N_KV_HEADS = 8
    D_HEAD = 128
    INV_SQRT_D = 1.0 / math.sqrt(D_HEAD)

    SEQ_LENS = [64, 128, 256, 512, 1024, 2048]
    WARMUP = 20
    ITERS = 100

    def time_custom_kernel(lib, q_ptr, k_ptr, v_ptr, out_ptr, w_ptr, seq_len):
        """Time a custom kernel. Returns median ms over ITERS calls."""
        # Warmup
        for _ in range(WARMUP):
            rc = lib.deterministic_attention(
                q_ptr, k_ptr, v_ptr, out_ptr, w_ptr,
                N_Q_HEADS, N_KV_HEADS, D_HEAD, seq_len,
                ctypes.c_float(INV_SQRT_D),
            )
            assert rc == 0, f"Kernel failed with rc={rc}"

        # Timed runs (kernel includes cudaDeviceSynchronize)
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            rc = lib.deterministic_attention(
                q_ptr, k_ptr, v_ptr, out_ptr, w_ptr,
                N_Q_HEADS, N_KV_HEADS, D_HEAD, seq_len,
                ctypes.c_float(INV_SQRT_D),
            )
            elapsed = time.perf_counter() - t0
            assert rc == 0
            times.append(elapsed * 1000)  # ms

        times.sort()
        return times[len(times) // 2]  # median

    def time_sdpa(q_f16, k_f16, v_f16):
        """Time stock PyTorch SDPA. Returns median ms over ITERS calls."""
        # SDPA expects: q=[B, n_heads, seq_q, d_head], k/v=[B, n_heads, seq_kv, d_head]
        # For decode: seq_q=1
        # GQA: expand kv heads to match q heads
        heads_per_kv = N_Q_HEADS // N_KV_HEADS
        k_expanded = k_f16.repeat_interleave(heads_per_kv, dim=1)
        v_expanded = v_f16.repeat_interleave(heads_per_kv, dim=1)

        scale = INV_SQRT_D

        # Warmup
        for _ in range(WARMUP):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q_f16, k_expanded, v_expanded, scale=scale
            )
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = torch.nn.functional.scaled_dot_product_attention(
                q_f16, k_expanded, v_expanded, scale=scale
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)

        times.sort()
        return times[len(times) // 2]

    # ── Run benchmarks ────────────────────────────────────────────
    results = []

    # Pre-allocate Q (constant across seq_len)
    torch.manual_seed(42)
    q_bf16 = torch.randn(N_Q_HEADS, D_HEAD, device="cuda", dtype=torch.bfloat16)
    q_u16 = q_bf16.view(torch.int16).contiguous()
    q_ptr = ctypes.c_void_p(q_u16.data_ptr())

    print(f"Config: n_q_heads={N_Q_HEADS}, n_kv_heads={N_KV_HEADS}, d_head={D_HEAD}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS} (reporting median)")
    print()

    header = f"{'seq_len':>8s}  {'v1 (ms)':>10s}  {'v2 (ms)':>10s}  {'speedup':>8s}  {'SDPA (ms)':>10s}  {'v2/SDPA':>8s}"
    print(header)
    print("-" * len(header))

    for seq_len in SEQ_LENS:
        # Generate random KV data
        k_bf16 = torch.randn(seq_len, N_KV_HEADS, D_HEAD, device="cuda", dtype=torch.bfloat16)
        v_bf16 = torch.randn(seq_len, N_KV_HEADS, D_HEAD, device="cuda", dtype=torch.bfloat16)

        k_u16 = k_bf16.view(torch.int16).contiguous()
        v_u16 = v_bf16.view(torch.int16).contiguous()

        # Output buffers
        out_f32 = torch.zeros(N_Q_HEADS, D_HEAD, device="cuda", dtype=torch.float32)
        w_f32 = torch.zeros(N_Q_HEADS, seq_len, device="cuda", dtype=torch.float32)

        k_ptr = ctypes.c_void_p(k_u16.data_ptr())
        v_ptr = ctypes.c_void_p(v_u16.data_ptr())
        out_ptr = ctypes.c_void_p(out_f32.data_ptr())
        w_ptr = ctypes.c_void_p(w_f32.data_ptr())

        # Time v1
        v1_ms = time_custom_kernel(v1_lib, q_ptr, k_ptr, v_ptr, out_ptr, w_ptr, seq_len)

        # Time v2
        v2_ms = time_custom_kernel(v2_lib, q_ptr, k_ptr, v_ptr, out_ptr, w_ptr, seq_len)

        # Time SDPA baseline
        # Reshape for SDPA: [batch=1, n_heads, seq, d_head]
        q_sdpa = q_bf16.unsqueeze(0).unsqueeze(2).to(torch.float16)  # [1, 32, 1, 128]
        k_sdpa = k_bf16.permute(1, 0, 2).unsqueeze(0).to(torch.float16)  # [1, 8, seq_len, 128]
        v_sdpa = v_bf16.permute(1, 0, 2).unsqueeze(0).to(torch.float16)  # [1, 8, seq_len, 128]

        sdpa_ms = time_sdpa(q_sdpa, k_sdpa, v_sdpa)

        speedup = v1_ms / v2_ms if v2_ms > 0 else float("inf")
        v2_over_sdpa = v2_ms / sdpa_ms if sdpa_ms > 0 else float("inf")

        print(f"{seq_len:>8d}  {v1_ms:>10.3f}  {v2_ms:>10.3f}  {speedup:>7.2f}x  {sdpa_ms:>10.3f}  {v2_over_sdpa:>7.1f}x")

        results.append({
            "seq_len": seq_len,
            "v1_ms": v1_ms,
            "v2_ms": v2_ms,
            "speedup": speedup,
            "sdpa_ms": sdpa_ms,
            "v2_over_sdpa": v2_over_sdpa,
        })

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    avg_v2_sdpa = sum(r["v2_over_sdpa"] for r in results) / len(results)

    print(f"Average v1→v2 speedup: {avg_speedup:.2f}x")
    print(f"Average v2/SDPA ratio: {avg_v2_sdpa:.1f}x")
    print()

    # Decision guidance
    if avg_speedup >= 1.5:
        print("VERDICT: Meaningful speedup from score-path parallelization.")
        print("  -> Proceed to Step 3 (softmax parallelization).")
    elif avg_speedup >= 1.1:
        print("VERDICT: Modest speedup. Score path alone won't close the gap.")
        print("  -> Softmax + V tiling needed; continue but with lower expectations.")
    else:
        print("VERDICT: Negligible speedup. Score path is NOT the bottleneck.")
        print("  -> Rethink: softmax + V aggregation dominate. Consider different strategy.")

    # At-scale overhead projection
    print()
    print("--- Overhead projection (attention-only, per decode token) ---")
    for r in results:
        overhead = r["v2_ms"] - r["sdpa_ms"]
        print(f"  seq_len={r['seq_len']:>5d}: v2 overhead vs SDPA = {overhead:>+.3f} ms  ({r['v2_over_sdpa']:.1f}x)")

    return results


@app.local_entrypoint()
def main():
    import json

    print("Deterministic Attention Benchmark: v1 (serial) vs v2 (tree-reduced scores)")
    print("=" * 70)
    results = run_benchmark.remote()

    print("\n\nFull results JSON:")
    print(json.dumps(results, indent=2, default=str))
