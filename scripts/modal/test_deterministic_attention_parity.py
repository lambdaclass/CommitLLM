"""
Deterministic Attention Parity Test.

Compiles the CUDA kernel, generates random bf16 Q/K/V inputs,
runs both GPU kernel and CPU reference (Rust via verilm_rs),
and asserts bit-exact f32 output match.

Usage:
    modal run --detach scripts/modal/test_deterministic_attention_parity.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-test-deterministic-attention-parity")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "ca-certificates")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    })
    .pip_install(*VERIFICATION)
    # Install sidecar for verilm_rs build deps
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    # Build verilm_rs (for CPU reference)
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf", "*.md", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
    )
    # Compile the CUDA kernel (nvcc available from nvidia/cuda base image)
    .run_commands(
        "cp /build/kernels/deterministic_attention.cu /opt/det_attn.cu",
        "nvcc -O2 --fmad=false -shared -Xcompiler -fPIC -o /opt/libdet_attn.so /opt/det_attn.cu",
        "rm -rf /build",
    )
)

N_RANDOM_CASES = 10000
# Test configurations: (n_q_heads, n_kv_heads, d_head, seq_len)
TEST_CONFIGS = [
    # Small cases for debugging
    (1, 1, 4, 1),
    (1, 1, 4, 2),
    (1, 1, 4, 8),
    (2, 1, 4, 4),      # GQA
    (4, 2, 8, 16),      # GQA
    # Realistic sizes
    (32, 8, 128, 32),   # Llama 8B, short context
    (32, 8, 128, 128),  # Llama 8B, medium
    (32, 8, 128, 512),  # Llama 8B, longer
    (32, 8, 128, 1024), # Llama 8B, 1K context
    (28, 4, 128, 128),  # Qwen 7B
    (28, 4, 128, 512),  # Qwen 7B, longer
]


def _run_parity_tests():
    import ctypes
    import json
    import struct
    import random
    import numpy as np
    import verilm_rs

    # Load the CUDA kernel shared library.
    lib = ctypes.CDLL("/opt/libdet_attn.so")

    lib.deterministic_attention_host.restype = ctypes.c_int
    lib.deterministic_attention_host.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),  # q_host
        ctypes.POINTER(ctypes.c_uint16),  # k_host
        ctypes.POINTER(ctypes.c_uint16),  # v_host
        ctypes.POINTER(ctypes.c_float),   # output_host
        ctypes.POINTER(ctypes.c_float),   # weights_host
        ctypes.c_int,                     # n_q_heads
        ctypes.c_int,                     # n_kv_heads
        ctypes.c_int,                     # d_head
        ctypes.c_int,                     # seq_len
        ctypes.c_float,                   # inv_sqrt_d (precomputed)
    ]

    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    total_tests = 0
    total_pass = 0
    total_mismatches = 0

    print("=" * 70)
    print("Deterministic Attention Parity Tests")
    print(f"Configs: {len(TEST_CONFIGS)}")
    print("=" * 70)

    for cfg_idx, (n_q_heads, n_kv_heads, d_head, seq_len) in enumerate(TEST_CONFIGS):
        n_cases = 1 if seq_len >= 512 else (5 if seq_len >= 128 else 20)
        config_pass = 0
        config_fail = 0
        max_output_diff = 0.0
        max_weight_diff = 0.0

        # Precompute inv_sqrt_d ONCE — same f32 bits for both CPU and GPU.
        inv_sqrt_d = np.float32(1.0 / np.sqrt(np.float32(d_head)))
        inv_sqrt_d_bits = inv_sqrt_d.view(np.uint32).item()

        print(f"\n[{cfg_idx}] n_q_heads={n_q_heads} n_kv_heads={n_kv_heads} "
              f"d_head={d_head} seq_len={seq_len} (n_cases={n_cases}) "
              f"inv_sqrt_d={inv_sqrt_d:.8e} (0x{inv_sqrt_d_bits:08x})")

        for case in range(n_cases):
            # Generate random bf16 Q, K, V.
            # Use normal distribution, convert to bf16 by truncating f32.
            q_f32 = np_rng.randn(n_q_heads * d_head).astype(np.float32)
            k_f32 = np_rng.randn(seq_len * n_kv_heads * d_head).astype(np.float32)
            v_f32 = np_rng.randn(seq_len * n_kv_heads * d_head).astype(np.float32)

            # Convert to bf16 (truncate lower 16 bits of f32)
            def f32_to_bf16_array(arr):
                """Convert f32 array to bf16 u16 array (round-to-nearest-even)."""
                bits = arr.view(np.uint32)
                # Round-to-nearest-even: add 0x7FFF + round_bit
                round_bit = (bits >> 16) & 1
                rounded = bits + 0x7FFF + round_bit
                return (rounded >> 16).astype(np.uint16)

            q_bf16 = f32_to_bf16_array(q_f32)
            k_bf16 = f32_to_bf16_array(k_f32)
            v_bf16 = f32_to_bf16_array(v_f32)

            # --- GPU: CUDA kernel ---
            q_ct = q_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            k_ct = k_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            v_ct = v_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

            output_gpu = np.zeros(n_q_heads * d_head, dtype=np.float32)
            weights_gpu = np.zeros(n_q_heads * seq_len, dtype=np.float32)

            out_ct = output_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            w_ct = weights_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            rc = lib.deterministic_attention_host(
                q_ct, k_ct, v_ct, out_ct, w_ct,
                n_q_heads, n_kv_heads, d_head, seq_len,
                inv_sqrt_d
            )
            assert rc == 0, f"CUDA kernel failed with rc={rc}"

            # --- CPU: Rust reference (pass same inv_sqrt_d f32) ---
            q_list = q_bf16.tolist()
            k_list = [k_bf16[t * n_kv_heads * d_head:(t+1) * n_kv_heads * d_head].tolist()
                       for t in range(seq_len)]
            v_list = [v_bf16[t * n_kv_heads * d_head:(t+1) * n_kv_heads * d_head].tolist()
                       for t in range(seq_len)]

            cpu_json = verilm_rs.deterministic_attention_bf16(
                q_list, k_list, v_list,
                n_q_heads, n_kv_heads, d_head,
                float(inv_sqrt_d),
            )
            cpu_result = json.loads(cpu_json)
            # CPU returns u32 bit representations directly — no float parsing precision loss.
            out_cpu_bits = np.array(cpu_result["output_bits"], dtype=np.uint32)
            w_cpu_bits = np.array(cpu_result["weight_bits"], dtype=np.uint32)
            output_cpu = out_cpu_bits.view(np.float32)
            weights_cpu = w_cpu_bits.view(np.float32)

            # GPU output as u32 bits
            out_gpu_bits = output_gpu.view(np.uint32)
            w_gpu_bits = weights_gpu.view(np.uint32)

            output_match = np.array_equal(out_gpu_bits, out_cpu_bits)
            weight_match = np.array_equal(w_gpu_bits, w_cpu_bits)

            if not output_match:
                diffs = np.abs(output_gpu - output_cpu)
                max_out_diff = float(diffs.max())
                n_mismatch = int(np.sum(out_gpu_bits != out_cpu_bits))
                max_output_diff = max(max_output_diff, max_out_diff)
                # Find first mismatch for debugging
                idx = np.argmax(out_gpu_bits != out_cpu_bits)
                print(f"    case {case}: OUTPUT MISMATCH — {n_mismatch}/{len(output_gpu)} elements, "
                      f"max_diff={max_out_diff:.2e}, "
                      f"first@{idx}: gpu={output_gpu[idx]:.8e} cpu={output_cpu[idx]:.8e} "
                      f"(bits: {out_gpu_bits[idx]:08x} vs {out_cpu_bits[idx]:08x})")

            if not weight_match:
                diffs = np.abs(weights_gpu - weights_cpu)
                max_w_diff = float(diffs.max())
                n_mismatch = int(np.sum(w_gpu_bits != w_cpu_bits))
                max_weight_diff = max(max_weight_diff, max_w_diff)
                if output_match:  # only print weight mismatch if output matched
                    idx = np.argmax(w_gpu_bits != w_cpu_bits)
                    print(f"    case {case}: WEIGHT MISMATCH — {n_mismatch}/{len(weights_gpu)} elements, "
                          f"max_diff={max_w_diff:.2e}")

            total_tests += 1
            if output_match and weight_match:
                config_pass += 1
                total_pass += 1
            else:
                config_fail += 1
                total_mismatches += 1

        status = "PASS" if config_fail == 0 else "FAIL"
        print(f"  [{status}] {config_pass}/{config_pass + config_fail} bit-exact")
        if max_output_diff > 0:
            print(f"  max output diff: {max_output_diff:.2e}")
        if max_weight_diff > 0:
            print(f"  max weight diff: {max_weight_diff:.2e}")

    # --- Also run deterministic self-consistency: GPU twice, same output ---
    print(f"\n{'='*70}")
    print("GPU Self-Consistency (same input twice)")
    print("=" * 70)

    consist_pass = 0
    consist_fail = 0
    for cfg_idx, (n_q_heads, n_kv_heads, d_head, seq_len) in enumerate(TEST_CONFIGS[:6]):
        inv_sqrt_d = np.float32(1.0 / np.sqrt(np.float32(d_head)))

        q_f32 = np_rng.randn(n_q_heads * d_head).astype(np.float32)
        k_f32 = np_rng.randn(seq_len * n_kv_heads * d_head).astype(np.float32)
        v_f32 = np_rng.randn(seq_len * n_kv_heads * d_head).astype(np.float32)

        def f32_to_bf16_array(arr):
            bits = arr.view(np.uint32)
            round_bit = (bits >> 16) & 1
            rounded = bits + 0x7FFF + round_bit
            return (rounded >> 16).astype(np.uint16)

        q_bf16 = f32_to_bf16_array(q_f32)
        k_bf16 = f32_to_bf16_array(k_f32)
        v_bf16 = f32_to_bf16_array(v_f32)

        results = []
        for run in range(3):
            output = np.zeros(n_q_heads * d_head, dtype=np.float32)
            weights = np.zeros(n_q_heads * seq_len, dtype=np.float32)

            rc = lib.deterministic_attention_host(
                q_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                k_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                v_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_q_heads, n_kv_heads, d_head, seq_len, inv_sqrt_d
            )
            assert rc == 0
            results.append(output.view(np.uint32).copy())

        all_same = all(np.array_equal(results[0], r) for r in results[1:])
        if all_same:
            consist_pass += 1
            print(f"  [{cfg_idx}] PASS — 3 runs bit-identical")
        else:
            consist_fail += 1
            print(f"  [{cfg_idx}] FAIL — GPU not self-consistent!")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("PARITY TEST SUMMARY")
    print("=" * 70)
    print(f"  GPU vs CPU parity:     {total_pass}/{total_tests} bit-exact "
          f"({'PASS' if total_mismatches == 0 else 'FAIL'})")
    print(f"  GPU self-consistency:  {consist_pass}/{consist_pass + consist_fail} "
          f"({'PASS' if consist_fail == 0 else 'FAIL'})")
    if total_mismatches > 0:
        print(f"\n  *** {total_mismatches} PARITY FAILURES ***")
        print(f"  This means GPU and CPU compute different f32 results.")
        print(f"  Check: --fmad=false, __fmul_rn/__fadd_rn usage, exp polynomial.")
    else:
        print(f"\n  All tests bit-exact. Deterministic attention is reproducible.")
    print("=" * 70)

    return {
        "total_tests": total_tests,
        "total_pass": total_pass,
        "total_mismatches": total_mismatches,
        "consist_pass": consist_pass,
        "consist_fail": consist_fail,
    }


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_parity_tests():
    return _run_parity_tests()


@app.local_entrypoint()
def main():
    print("Deterministic Attention Parity Test")
    print("=" * 70)
    result = run_parity_tests.remote()
    print(f"\nResult: {result}")
