"""
Deterministic Attention Stress Test.

10K randomized parity sweep + targeted edge-case vectors.
Decode only, seqlen_q=1, head_dim=128, GQA.

Usage:
    modal run scripts/modal/stress_deterministic_attention.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-stress-deterministic-attention")

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
    )
    .run_commands(
        "cp /build/kernels/deterministic_attention.cu /opt/det_attn.cu",
        "nvcc -O2 --fmad=false -shared -Xcompiler -fPIC -o /opt/libdet_attn.so /opt/det_attn.cu",
        "rm -rf /build",
    )
)


# ─── Helpers (defined at module level, used inside remote function) ──────────

def _setup_lib():
    """Load CUDA library and set up ctypes signatures."""
    import ctypes
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
        ctypes.c_float,                   # inv_sqrt_d
    ]
    return lib


def _f32_to_bf16_array(arr):
    """Convert f32 array to bf16 u16 array (round-to-nearest-even)."""
    import numpy as np
    bits = arr.view(np.uint32)
    round_bit = (bits >> 16) & 1
    rounded = bits + 0x7FFF + round_bit
    return (rounded >> 16).astype(np.uint16)


def _run_one_case(lib, q_bf16, k_bf16, v_bf16, n_q_heads, n_kv_heads, d_head, seq_len, inv_sqrt_d):
    """Run GPU kernel and CPU reference, return (output_match, weight_match, details_dict)."""
    import ctypes
    import json
    import numpy as np
    import verilm_rs

    # GPU
    output_gpu = np.zeros(n_q_heads * d_head, dtype=np.float32)
    weights_gpu = np.zeros(n_q_heads * seq_len, dtype=np.float32)

    rc = lib.deterministic_attention_host(
        q_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        k_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        v_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        output_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_q_heads, n_kv_heads, d_head, seq_len, inv_sqrt_d,
    )
    assert rc == 0, f"CUDA kernel failed with rc={rc}"

    # CPU
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
    out_cpu_bits = np.array(cpu_result["output_bits"], dtype=np.uint32)
    w_cpu_bits = np.array(cpu_result["weight_bits"], dtype=np.uint32)

    out_gpu_bits = output_gpu.view(np.uint32)
    w_gpu_bits = weights_gpu.view(np.uint32)

    output_match = np.array_equal(out_gpu_bits, out_cpu_bits)
    weight_match = np.array_equal(w_gpu_bits, w_cpu_bits)

    details = {}
    if not output_match:
        diffs = np.abs(output_gpu - out_cpu_bits.view(np.float32))
        idx = int(np.argmax(out_gpu_bits != out_cpu_bits))
        details["output_mismatch"] = {
            "n": int(np.sum(out_gpu_bits != out_cpu_bits)),
            "total": len(output_gpu),
            "max_diff": float(diffs.max()),
            "first_idx": idx,
            "gpu_bits": f"{out_gpu_bits[idx]:08x}",
            "cpu_bits": f"{out_cpu_bits[idx]:08x}",
        }
    if not weight_match:
        diffs = np.abs(weights_gpu - w_cpu_bits.view(np.float32))
        idx = int(np.argmax(w_gpu_bits != w_cpu_bits))
        details["weight_mismatch"] = {
            "n": int(np.sum(w_gpu_bits != w_cpu_bits)),
            "total": len(weights_gpu),
            "max_diff": float(diffs.max()),
            "first_idx": idx,
            "gpu_bits": f"{w_gpu_bits[idx]:08x}",
            "cpu_bits": f"{w_cpu_bits[idx]:08x}",
        }

    return output_match, weight_match, details


def _run_stress():
    import numpy as np
    import time

    lib = _setup_lib()

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1: 10K Randomized Parity Sweep
    # ═══════════════════════════════════════════════════════════════════════
    #
    # Decode only: seqlen_q=1, d_head=128, GQA.
    # Random n_q_heads, n_kv_heads, seq_len, random bf16 inputs.

    N_RANDOM = 10000
    SEED = 12345

    # GQA configs: (n_q_heads, n_kv_heads) — realistic ratios
    GQA_CONFIGS = [
        (32, 8),   # Llama 8B
        (28, 4),   # Qwen 7B
        (32, 4),   # Mistral 7B
        (8, 1),    # Small model, MHA-like
        (16, 4),   # Generic GQA
        (32, 32),  # MHA (no grouping)
        (64, 8),   # Large model GQA
        (48, 6),   # Awkward ratio
        (12, 3),   # Awkward ratio
        (40, 5),   # Awkward ratio
    ]
    D_HEAD = 128

    rng = np.random.RandomState(SEED)
    total_pass = 0
    total_fail = 0
    failures = []

    print("=" * 70)
    print(f"SECTION 1: 10K Randomized Parity Sweep (seed={SEED})")
    print(f"  d_head={D_HEAD}, {len(GQA_CONFIGS)} GQA configs, random seq_len")
    print("=" * 70)

    t0 = time.time()
    for i in range(N_RANDOM):
        # Pick random GQA config
        cfg_idx = rng.randint(0, len(GQA_CONFIGS))
        n_q_heads, n_kv_heads = GQA_CONFIGS[cfg_idx]

        # Random seq_len: biased toward short (decode-relevant) but includes long
        # Distribution: 50% [1,64], 30% [65,256], 15% [257,1024], 5% [1025,2048]
        r = rng.random()
        if r < 0.50:
            seq_len = rng.randint(1, 65)
        elif r < 0.80:
            seq_len = rng.randint(65, 257)
        elif r < 0.95:
            seq_len = rng.randint(257, 1025)
        else:
            seq_len = rng.randint(1025, 2049)

        inv_sqrt_d = np.float32(1.0 / np.sqrt(np.float32(D_HEAD)))

        # Random bf16 inputs (normal distribution → bf16)
        q_f32 = rng.randn(n_q_heads * D_HEAD).astype(np.float32)
        k_f32 = rng.randn(seq_len * n_kv_heads * D_HEAD).astype(np.float32)
        v_f32 = rng.randn(seq_len * n_kv_heads * D_HEAD).astype(np.float32)

        q_bf16 = _f32_to_bf16_array(q_f32)
        k_bf16 = _f32_to_bf16_array(k_f32)
        v_bf16 = _f32_to_bf16_array(v_f32)

        out_ok, w_ok, details = _run_one_case(
            lib, q_bf16, k_bf16, v_bf16,
            n_q_heads, n_kv_heads, D_HEAD, seq_len, inv_sqrt_d,
        )

        if out_ok and w_ok:
            total_pass += 1
        else:
            total_fail += 1
            failure_info = {
                "case": i,
                "seed": SEED,
                "n_q_heads": n_q_heads,
                "n_kv_heads": n_kv_heads,
                "d_head": D_HEAD,
                "seq_len": seq_len,
                **details,
            }
            failures.append(failure_info)
            print(f"  FAIL case {i}: heads={n_q_heads}/{n_kv_heads} "
                  f"seq_len={seq_len} — {details}")

        # Progress every 1000
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:>5}/{N_RANDOM}] pass={total_pass} fail={total_fail} "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    s1_status = "PASS" if total_fail == 0 else "FAIL"
    print(f"\n  [{s1_status}] Random sweep: {total_pass}/{total_pass + total_fail} "
          f"bit-exact ({elapsed:.1f}s)")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2: Targeted Edge Cases
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print("SECTION 2: Targeted Edge Cases")
    print("=" * 70)

    edge_pass = 0
    edge_fail = 0
    edge_failures = []

    def run_edge(name, q_bf16, k_bf16, v_bf16, n_q, n_kv, d, sl):
        nonlocal edge_pass, edge_fail
        inv_sqrt = np.float32(1.0 / np.sqrt(np.float32(d)))
        out_ok, w_ok, details = _run_one_case(
            lib, q_bf16, k_bf16, v_bf16, n_q, n_kv, d, sl, inv_sqrt,
        )
        if out_ok and w_ok:
            edge_pass += 1
            print(f"  [PASS] {name}")
        else:
            edge_fail += 1
            edge_failures.append({"name": name, **details})
            print(f"  [FAIL] {name} — {details}")

    # Helper to make bf16 from f32 scalar
    def bf16(val):
        return _f32_to_bf16_array(np.array([val], dtype=np.float32))[0]

    def bf16_arr(vals):
        return _f32_to_bf16_array(np.array(vals, dtype=np.float32))

    # --- 2a: Repeated max scores ---
    # All K positions identical → all scores identical → uniform softmax
    print("\n  --- 2a: Repeated max scores (uniform softmax) ---")
    for sl in [1, 2, 8, 64, 256, 1024]:
        n_q, n_kv, d = 32, 8, 128
        q = bf16_arr(rng.randn(n_q * d).astype(np.float32))
        # All K positions are the same vector
        k_one = bf16_arr(rng.randn(n_kv * d).astype(np.float32))
        k = np.tile(k_one, sl)
        v = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        run_edge(f"repeated_max_scores_sl{sl}", q, k, v, n_q, n_kv, d, sl)

    # --- 2b: Near-tie scores ---
    # Q and K constructed so dot products are nearly identical
    print("\n  --- 2b: Near-tie scores ---")
    for sl in [2, 8, 64, 512]:
        n_q, n_kv, d = 32, 8, 128
        q = bf16_arr(np.ones(n_q * d, dtype=np.float32) * 0.01)
        # K: very slight perturbation across positions
        k_base = np.ones(n_kv * d, dtype=np.float32) * 0.01
        k_all = []
        for t in range(sl):
            k_t = k_base.copy()
            # Tiny perturbation in first element only
            k_t[0] = np.float32(0.01 + t * 1e-6)
            k_all.append(k_t)
        k = bf16_arr(np.concatenate(k_all))
        v = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        run_edge(f"near_tie_scores_sl{sl}", q, k, v, n_q, n_kv, d, sl)

    # --- 2c: Extreme negative tails / underflow ---
    # Q and K such that scores are very negative → exp underflows to 0
    print("\n  --- 2c: Extreme negative scores (exp underflow) ---")
    for sl in [2, 16, 128, 512]:
        n_q, n_kv, d = 32, 8, 128
        # Large Q, K pointing in opposite directions
        q = bf16_arr(np.ones(n_q * d, dtype=np.float32) * 3.0)
        k_all = []
        for t in range(sl):
            if t == 0:
                # One position has positive score (keeps softmax sum nonzero)
                k_all.append(np.ones(n_kv * d, dtype=np.float32) * 3.0)
            else:
                # All others: extreme negative
                k_all.append(np.ones(n_kv * d, dtype=np.float32) * -3.0)
        k = bf16_arr(np.concatenate(k_all))
        v = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        run_edge(f"extreme_negative_sl{sl}", q, k, v, n_q, n_kv, d, sl)

    # --- 2d: All-zero V ---
    print("\n  --- 2d: All-zero V ---")
    for sl in [1, 8, 128]:
        n_q, n_kv, d = 32, 8, 128
        q = bf16_arr(rng.randn(n_q * d).astype(np.float32))
        k = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        v = np.zeros(sl * n_kv * d, dtype=np.uint16)  # All zero bf16
        run_edge(f"zero_v_sl{sl}", q, k, v, n_q, n_kv, d, sl)

    # --- 2e: One-hot V (single element nonzero per position) ---
    print("\n  --- 2e: One-hot V ---")
    for sl in [1, 8, 64]:
        n_q, n_kv, d = 32, 8, 128
        q = bf16_arr(rng.randn(n_q * d).astype(np.float32))
        k = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        v_f32 = np.zeros(sl * n_kv * d, dtype=np.float32)
        for t in range(sl):
            for h in range(n_kv):
                idx = t * n_kv * d + h * d + (t % d)
                v_f32[idx] = 1.0
        v = _f32_to_bf16_array(v_f32)
        run_edge(f"onehot_v_sl{sl}", q, k, v, n_q, n_kv, d, sl)

    # --- 2f: Constant V (all elements same value) ---
    print("\n  --- 2f: Constant V ---")
    for val in [1.0, -1.0, 0.5, 100.0]:
        sl = 64
        n_q, n_kv, d = 32, 8, 128
        q = bf16_arr(rng.randn(n_q * d).astype(np.float32))
        k = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        v = bf16_arr(np.full(sl * n_kv * d, val, dtype=np.float32))
        run_edge(f"constant_v_{val}", q, k, v, n_q, n_kv, d, sl)

    # --- 2g: Short context (seq_len=1) ---
    print("\n  --- 2g: Minimal context (seq_len=1) ---")
    for n_q, n_kv in [(32, 8), (28, 4), (64, 8), (8, 1)]:
        d = 128
        q = bf16_arr(rng.randn(n_q * d).astype(np.float32))
        k = bf16_arr(rng.randn(1 * n_kv * d).astype(np.float32))
        v = bf16_arr(rng.randn(1 * n_kv * d).astype(np.float32))
        run_edge(f"seqlen1_{n_q}q{n_kv}kv", q, k, v, n_q, n_kv, d, 1)

    # --- 2h: Max-length context (2048) ---
    print("\n  --- 2h: Max-length context (2048) ---")
    for n_q, n_kv in [(32, 8), (28, 4)]:
        d = 128
        q = bf16_arr(rng.randn(n_q * d).astype(np.float32))
        k = bf16_arr(rng.randn(2048 * n_kv * d).astype(np.float32))
        v = bf16_arr(rng.randn(2048 * n_kv * d).astype(np.float32))
        run_edge(f"seqlen2048_{n_q}q{n_kv}kv", q, k, v, n_q, n_kv, d, 2048)

    # --- 2i: Awkward GQA ratios ---
    print("\n  --- 2i: Awkward GQA ratios ---")
    for n_q, n_kv in [(3, 1), (6, 2), (6, 3), (9, 3), (15, 5), (48, 6), (56, 7)]:
        d = 128
        sl = 64
        q = bf16_arr(rng.randn(n_q * d).astype(np.float32))
        k = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        v = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        run_edge(f"gqa_{n_q}q{n_kv}kv", q, k, v, n_q, n_kv, d, sl)

    # --- 2j: Large magnitude inputs (stress exp range) ---
    print("\n  --- 2j: Large magnitude inputs ---")
    for scale in [10.0, 50.0, 100.0]:
        sl = 64
        n_q, n_kv, d = 32, 8, 128
        q = bf16_arr((rng.randn(n_q * d) * scale).astype(np.float32))
        k = bf16_arr((rng.randn(sl * n_kv * d) * scale).astype(np.float32))
        v = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        run_edge(f"large_mag_scale{scale}", q, k, v, n_q, n_kv, d, sl)

    # --- 2k: Tiny magnitude inputs (near-zero scores) ---
    print("\n  --- 2k: Tiny magnitude inputs ---")
    for scale in [1e-3, 1e-5]:
        sl = 64
        n_q, n_kv, d = 32, 8, 128
        q = bf16_arr((rng.randn(n_q * d) * scale).astype(np.float32))
        k = bf16_arr((rng.randn(sl * n_kv * d) * scale).astype(np.float32))
        v = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        run_edge(f"tiny_mag_scale{scale}", q, k, v, n_q, n_kv, d, sl)

    # --- 2l: Single dominant position (one score >> rest) ---
    print("\n  --- 2l: Single dominant position ---")
    for sl in [8, 64, 512]:
        n_q, n_kv, d = 32, 8, 128
        q = bf16_arr(np.ones(n_q * d, dtype=np.float32))
        k_all = []
        for t in range(sl):
            if t == sl // 2:
                # Dominant position: aligned with Q
                k_all.append(np.ones(n_kv * d, dtype=np.float32) * 2.0)
            else:
                # Noise
                k_all.append(rng.randn(n_kv * d).astype(np.float32) * 0.01)
        k = bf16_arr(np.concatenate(k_all))
        v = bf16_arr(rng.randn(sl * n_kv * d).astype(np.float32))
        run_edge(f"dominant_pos_sl{sl}", q, k, v, n_q, n_kv, d, sl)

    s2_status = "PASS" if edge_fail == 0 else "FAIL"
    print(f"\n  [{s2_status}] Edge cases: {edge_pass}/{edge_pass + edge_fail} bit-exact")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    print(f"  10K random sweep:   {total_pass}/{total_pass + total_fail} bit-exact "
          f"({s1_status})")
    print(f"  Edge cases:         {edge_pass}/{edge_pass + edge_fail} bit-exact "
          f"({s2_status})")

    all_failures = failures + edge_failures
    if all_failures:
        print(f"\n  *** {len(all_failures)} TOTAL FAILURES ***")
        for f in all_failures[:20]:
            print(f"    {f}")
    else:
        print(f"\n  ALL TESTS BIT-EXACT. Arithmetic contract is stable.")
    print("=" * 70)

    return {
        "random_pass": total_pass,
        "random_fail": total_fail,
        "edge_pass": edge_pass,
        "edge_fail": edge_fail,
        "failures": all_failures[:50],
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_stress():
    return _run_stress()


@app.local_entrypoint()
def main():
    print("Deterministic Attention Stress Test")
    print("=" * 70)
    result = run_stress.remote()
    print(f"\nResult: {result}")
