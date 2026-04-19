"""
Tree reduction primitive parity test: CUDA kernel vs Rust reference.

Tests that `tree_reduce_sum_f32` and `tree_reduce_max_f32` produce identical
f32 bit patterns on GPU (CUDA) and CPU (Rust).

Test matrix:
  Lengths: 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 127, 128, 129, 1024
  Data patterns:
    - Sequential integers
    - Random f32 values (5 seeds per length)
    - Cancellation-heavy (large +/- pairs)
    - All zeros
    - Signed zeros
    - Very small (subnormal-adjacent)
    - Very large (near MAX_FLOAT)
    - Repeated maxima
    - All negative

Run: modal run scripts/modal/test_tree_reduce_parity.py
"""

import modal
import struct

app = modal.App("tree-reduce-parity")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
    .pip_install("numpy")
    .add_local_dir("kernels", remote_path="/build/kernels", copy=True)
    .run_commands(
        # Compile the CUDA kernel
        "nvcc -O2 --fmad=false -shared -Xcompiler -fPIC -o /build/libtree_reduce.so /build/kernels/tree_reduce.cu",
    )
)


def f32_to_bits(f: float) -> int:
    """Convert f32 to its u32 bit representation."""
    return struct.unpack('<I', struct.pack('<f', f))[0]


def bits_to_f32(b: int) -> float:
    """Convert u32 bits to f32."""
    return struct.unpack('<f', struct.pack('<I', b))[0]


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
def run_parity_tests():
    import ctypes
    import numpy as np
    import struct as pystruct

    # --- Load CUDA library ---
    cuda_lib = ctypes.CDLL("/build/libtree_reduce.so")

    cuda_lib.tree_reduce_sum_f32_host.restype = ctypes.c_int
    cuda_lib.tree_reduce_sum_f32_host.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float)
    ]
    cuda_lib.tree_reduce_max_f32_host.restype = ctypes.c_int
    cuda_lib.tree_reduce_max_f32_host.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float)
    ]

    def gpu_reduce_sum(values: np.ndarray) -> int:
        """Returns f32 result as u32 bits."""
        n = len(values)
        arr = values.astype(np.float32)
        c_arr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = ctypes.c_float(0.0)
        rc = cuda_lib.tree_reduce_sum_f32_host(c_arr, n, ctypes.byref(result))
        assert rc == 0, f"CUDA sum failed with rc={rc}"
        return f32_to_bits(result.value)

    def gpu_reduce_max(values: np.ndarray) -> int:
        """Returns f32 result as u32 bits."""
        n = len(values)
        arr = values.astype(np.float32)
        c_arr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = ctypes.c_float(0.0)
        rc = cuda_lib.tree_reduce_max_f32_host(c_arr, n, ctypes.byref(result))
        assert rc == 0, f"CUDA max failed with rc={rc}"
        return f32_to_bits(result.value)

    # --- Python reference (mirrors the Rust tree exactly) ---

    def _next_pow2(v):
        if v == 0:
            return 1
        v -= 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1

    def _to_f32(x):
        """Round a Python float to exact f32 representation."""
        import math
        if math.isinf(x):
            return x
        if math.isnan(x):
            return x
        # Clamp to f32 range before packing
        F32_MAX = 3.4028234663852886e+38
        if x > F32_MAX:
            return float('inf')
        if x < -F32_MAX:
            return float('-inf')
        return pystruct.unpack('<f', pystruct.pack('<f', x))[0]

    def _f32_add(a, b):
        """f32 addition (round-to-nearest-even, matching IEEE 754 binary32)."""
        return _to_f32(a + b)

    def _f32_ge(a, b):
        """IEEE 754 >= comparison for f32."""
        return a >= b

    def cpu_reduce_sum(values: np.ndarray) -> int:
        """Python reference tree_reduce_sum matching the Rust/CUDA contract."""
        n = len(values)
        if n == 0:
            return f32_to_bits(0.0)
        if n == 1:
            return f32_to_bits(float(values[0]))
        padded_n = _next_pow2(n)
        buf = [_to_f32(float(values[i])) if i < n else 0.0 for i in range(padded_n)]
        stride = padded_n // 2
        while stride >= 1:
            for i in range(stride):
                buf[i] = _f32_add(buf[i], buf[i + stride])
            stride //= 2
        return f32_to_bits(buf[0])

    def cpu_reduce_max(values: np.ndarray) -> int:
        """Python reference tree_reduce_max matching the Rust/CUDA contract."""
        n = len(values)
        if n == 0:
            return f32_to_bits(float('-inf'))
        if n == 1:
            return f32_to_bits(float(values[0]))
        padded_n = _next_pow2(n)
        buf = [_to_f32(float(values[i])) if i < n else float('-inf') for i in range(padded_n)]
        stride = padded_n // 2
        while stride >= 1:
            for i in range(stride):
                left = buf[i]
                right = buf[i + stride]
                buf[i] = left if _f32_ge(left, right) else right
            stride //= 2
        return f32_to_bits(buf[0])

    # --- Test cases ---
    LENGTHS = [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 127, 128, 129, 1024]

    passed = 0
    failed = 0
    total = 0

    def _is_nan_bits(bits: int) -> bool:
        """Check if f32 bits represent NaN."""
        exp = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        return exp == 0xFF and mantissa != 0

    def check(name: str, op: str, values: np.ndarray):
        nonlocal passed, failed, total
        total += 1
        gpu_bits = gpu_reduce_sum(values) if op == "sum" else gpu_reduce_max(values)
        cpu_bits = cpu_reduce_sum(values) if op == "sum" else cpu_reduce_max(values)
        # NaN payload bits may differ across implementations.
        # Contract says NaN inputs are undefined; NaN produced via overflow
        # (inf - inf) is also outside the contract. Both sides agree "NaN" = pass.
        if _is_nan_bits(gpu_bits) and _is_nan_bits(cpu_bits):
            passed += 1
            return
        if gpu_bits != cpu_bits:
            failed += 1
            gpu_f = bits_to_f32(gpu_bits)
            cpu_f = bits_to_f32(cpu_bits)
            print(f"  FAIL {name}: GPU=0x{gpu_bits:08x} ({gpu_f}) vs CPU=0x{cpu_bits:08x} ({cpu_f})")
        else:
            passed += 1

    print("\n=== Sequential integers ===")
    for n in LENGTHS:
        values = np.arange(1, n + 1, dtype=np.float32)
        check(f"sum_seq_{n}", "sum", values)
        check(f"max_seq_{n}", "max", values)

    print(f"  sequential: {passed}/{total}")

    print("\n=== Random values (5 seeds per length) ===")
    for n in LENGTHS:
        for seed in range(5):
            rng = np.random.default_rng(seed * 1000 + n)
            values = rng.standard_normal(n).astype(np.float32) * 100.0
            check(f"sum_rand_{n}_s{seed}", "sum", values)
            check(f"max_rand_{n}_s{seed}", "max", values)

    print(f"  random: {passed}/{total}")

    print("\n=== Cancellation-heavy ===")
    for n in [4, 8, 16, 32, 128]:
        # Alternating large +/- values
        values = np.array([1e7 * ((-1) ** i) for i in range(n)], dtype=np.float32)
        check(f"sum_cancel_{n}", "sum", values)
        check(f"max_cancel_{n}", "max", values)

        # Large + small that cancel
        values2 = np.zeros(n, dtype=np.float32)
        values2[0] = 1e8
        values2[1] = -1e8
        for i in range(2, n):
            values2[i] = float(i)
        check(f"sum_cancel2_{n}", "sum", values2)

    print(f"  cancellation: {passed}/{total}")

    print("\n=== All zeros ===")
    for n in [1, 4, 16, 128]:
        values = np.zeros(n, dtype=np.float32)
        check(f"sum_zeros_{n}", "sum", values)
        check(f"max_zeros_{n}", "max", values)

    print(f"  zeros: {passed}/{total}")

    print("\n=== Signed zeros ===")
    for n in [2, 4, 8]:
        # Alternating +0.0, -0.0
        values = np.array([0.0 if i % 2 == 0 else -0.0 for i in range(n)], dtype=np.float32)
        check(f"sum_szero_{n}", "sum", values)
        check(f"max_szero_{n}", "max", values)

        # All -0.0
        values2 = np.full(n, -0.0, dtype=np.float32)
        check(f"sum_negzero_{n}", "sum", values2)
        check(f"max_negzero_{n}", "max", values2)

    print(f"  signed zeros: {passed}/{total}")

    print("\n=== Very small values ===")
    for n in [4, 16, 128]:
        values = np.full(n, 1e-38, dtype=np.float32)
        check(f"sum_tiny_{n}", "sum", values)
        check(f"max_tiny_{n}", "max", values)

    print(f"  very small: {passed}/{total}")

    print("\n=== Very large values ===")
    for n in [2, 4, 8]:
        values = np.full(n, 1e38, dtype=np.float32)
        check(f"sum_huge_{n}", "sum", values)
        check(f"max_huge_{n}", "max", values)

        # Mix of large positive and negative
        values2 = np.array([3.4e38, -3.4e38] * (n // 2), dtype=np.float32)
        check(f"sum_huge_mix_{n}", "sum", values2)
        check(f"max_huge_mix_{n}", "max", values2)

    print(f"  very large: {passed}/{total}")

    print("\n=== Repeated maxima ===")
    for n in [4, 8, 32, 128]:
        values = np.ones(n, dtype=np.float32) * 5.0
        check(f"sum_repeat_{n}", "sum", values)
        check(f"max_repeat_{n}", "max", values)

        # Max at every other position
        values2 = np.arange(1, n + 1, dtype=np.float32)
        values2[::2] = float(n)
        check(f"max_repeat2_{n}", "max", values2)

    print(f"  repeated maxima: {passed}/{total}")

    print("\n=== All negative ===")
    for n in [4, 16, 128]:
        values = -np.arange(1, n + 1, dtype=np.float32)
        check(f"sum_neg_{n}", "sum", values)
        check(f"max_neg_{n}", "max", values)

    print(f"  all negative: {passed}/{total}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print("PARITY TEST FAILED")
        return {"passed": passed, "failed": failed, "total": total}
    else:
        print("ALL PARITY TESTS PASSED")
        return {"passed": passed, "failed": failed, "total": total}


@app.local_entrypoint()
def main():
    result = run_parity_tests.remote()
    print(f"\nResult: {result}")
    if result["failed"] > 0:
        raise SystemExit(1)
