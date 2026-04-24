"""
Deterministic attention kernel benchmark: v3 (score + softmax) vs v4 (tiled V aggregation).

Measures kernel-only latency on identical random bf16 inputs across multiple
seq_len values. Includes stock PyTorch SDPA as baseline reference.

Config: Llama 8B (n_q_heads=32, n_kv_heads=8, d_head=128)

Run: modal run --detach scripts/modal/bench_det_attn_v3_v4.py
"""

import modal

app = modal.App("det-attn-v3-v4-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
    .pip_install("numpy", "torch==2.5.1", extra_options="--index-url https://download.pytorch.org/whl/cu124")
    .add_local_dir("kernels", remote_path="/build/kernels", copy=True)
    .run_commands(
        # v3 = score + softmax parallel, sequential V
        "nvcc -O2 --fmad=false -shared -Xcompiler -fPIC "
        "-o /build/libdet_attn_v3.so /build/kernels/deterministic_attention_v3_softmax.cu",
        # v4 = score + softmax + tiled V aggregation
        "nvcc -O2 --fmad=false -shared -Xcompiler -fPIC "
        "-o /build/libdet_attn_v4.so /build/kernels/deterministic_attention.cu",
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
    print()

    # ── Load kernels ──────────────────────────────────────────────
    v3_lib = ctypes.CDLL("/build/libdet_attn_v3.so")
    v4_lib = ctypes.CDLL("/build/libdet_attn_v4.so")

    for lib in [v3_lib, v4_lib]:
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
        for _ in range(WARMUP):
            rc = lib.deterministic_attention(
                q_ptr, k_ptr, v_ptr, out_ptr, w_ptr,
                N_Q_HEADS, N_KV_HEADS, D_HEAD, seq_len,
                ctypes.c_float(INV_SQRT_D),
            )
            assert rc == 0, f"Kernel failed with rc={rc}"

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
            times.append(elapsed * 1000)

        times.sort()
        return times[len(times) // 2]

    def time_sdpa(q_f16, k_f16, v_f16):
        """Time stock PyTorch SDPA. Returns median ms."""
        heads_per_kv = N_Q_HEADS // N_KV_HEADS
        k_expanded = k_f16.repeat_interleave(heads_per_kv, dim=1)
        v_expanded = v_f16.repeat_interleave(heads_per_kv, dim=1)
        scale = INV_SQRT_D

        for _ in range(WARMUP):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q_f16, k_expanded, v_expanded, scale=scale
            )
            torch.cuda.synchronize()

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

    torch.manual_seed(42)
    q_bf16 = torch.randn(N_Q_HEADS, D_HEAD, device="cuda", dtype=torch.bfloat16)
    q_u16 = q_bf16.view(torch.int16).contiguous()
    q_ptr = ctypes.c_void_p(q_u16.data_ptr())

    print(f"Config: n_q_heads={N_Q_HEADS}, n_kv_heads={N_KV_HEADS}, d_head={D_HEAD}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS} (reporting median)")
    print()

    header = f"{'seq_len':>8s}  {'v3-smx':>10s}  {'v4-tile':>10s}  {'v4/v3':>8s}  {'SDPA':>10s}  {'v4/SDPA':>8s}"
    print(header)
    print("-" * len(header))

    for seq_len in SEQ_LENS:
        k_bf16 = torch.randn(seq_len, N_KV_HEADS, D_HEAD, device="cuda", dtype=torch.bfloat16)
        v_bf16 = torch.randn(seq_len, N_KV_HEADS, D_HEAD, device="cuda", dtype=torch.bfloat16)

        k_u16 = k_bf16.view(torch.int16).contiguous()
        v_u16 = v_bf16.view(torch.int16).contiguous()

        out_f32 = torch.zeros(N_Q_HEADS, D_HEAD, device="cuda", dtype=torch.float32)
        w_f32 = torch.zeros(N_Q_HEADS, seq_len, device="cuda", dtype=torch.float32)

        k_ptr = ctypes.c_void_p(k_u16.data_ptr())
        v_ptr = ctypes.c_void_p(v_u16.data_ptr())
        out_ptr = ctypes.c_void_p(out_f32.data_ptr())
        w_ptr = ctypes.c_void_p(w_f32.data_ptr())

        v3_ms = time_custom_kernel(v3_lib, q_ptr, k_ptr, v_ptr, out_ptr, w_ptr, seq_len)
        v4_ms = time_custom_kernel(v4_lib, q_ptr, k_ptr, v_ptr, out_ptr, w_ptr, seq_len)

        q_sdpa = q_bf16.unsqueeze(0).unsqueeze(2).to(torch.float16)
        k_sdpa = k_bf16.permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        v_sdpa = v_bf16.permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        sdpa_ms = time_sdpa(q_sdpa, k_sdpa, v_sdpa)

        speedup = v3_ms / v4_ms if v4_ms > 0 else float("inf")
        v4_over_sdpa = v4_ms / sdpa_ms if sdpa_ms > 0 else float("inf")

        print(f"{seq_len:>8d}  {v3_ms:>10.3f}  {v4_ms:>10.3f}  {speedup:>7.2f}x  {sdpa_ms:>10.3f}  {v4_over_sdpa:>7.1f}x")

        results.append({
            "seq_len": seq_len,
            "v3_softmax_ms": v3_ms,
            "v4_tiled_ms": v4_ms,
            "v4_over_v3": speedup,
            "sdpa_ms": sdpa_ms,
            "v4_over_sdpa": v4_over_sdpa,
        })

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_speedup = sum(r["v4_over_v3"] for r in results) / len(results)
    avg_v4_sdpa = sum(r["v4_over_sdpa"] for r in results) / len(results)

    print(f"Average v3→v4 speedup: {avg_speedup:.2f}x")
    print(f"Average v4/SDPA ratio: {avg_v4_sdpa:.1f}x")
    print()

    # Compare to prior steps
    print("--- Step 2 reference (v1→v2 score-path): ~3.16x speedup ---")
    print("--- Step 3 reference (v2→v3 softmax):    ~1.20x speedup ---")
    print(f"--- Step 4 result    (v3→v4 V-tiling):   {avg_speedup:.2f}x speedup ---")
    print()

    if avg_speedup >= 1.5:
        print("VERDICT: Meaningful speedup from V aggregation tiling.")
    elif avg_speedup >= 1.1:
        print("VERDICT: Modest speedup from V tiling.")
    else:
        print("VERDICT: Negligible V tiling speedup at this tile size.")
    print()

    print("--- Overhead vs SDPA (per decode token) ---")
    for r in results:
        overhead = r["v4_tiled_ms"] - r["sdpa_ms"]
        print(f"  seq_len={r['seq_len']:>5d}: v4 overhead = {overhead:>+.3f} ms  ({r['v4_over_sdpa']:.1f}x)")

    return results


@app.local_entrypoint()
def main():
    import json

    print("Deterministic Attention Benchmark: v3 (softmax) vs v4 (tiled V)")
    print("=" * 70)
    results = run_benchmark.remote()

    print("\n\nFull results JSON:")
    print(json.dumps(results, indent=2, default=str))
