#!/usr/bin/env python3
"""
Local smoke test and fast benchmark. No GPU needed.

Modes:
    python scripts/local_smoke_test.py              # quick correctness check
    python scripts/local_smoke_test.py --full       # Qwen-sized correctness check
    python scripts/local_smoke_test.py --bench      # fast benchmark (5 iterations)
    python scripts/local_smoke_test.py --bench 20   # benchmark with N iterations
"""

import hashlib
import sys
import time

import torch

PROJ_SEQUENCE = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def make_synthetic_captures(n_layers, n_fwd_passes, fwd_batch_sizes,
                            hidden_dim, kv_dim, intermediate_size):
    """Build synthetic capture tuples matching real capture hook output."""
    qkv_dim = hidden_dim + 2 * kv_dim
    gate_up_dim = 2 * intermediate_size

    proj_dims = {
        "qkv_proj": (hidden_dim, qkv_dim),
        "o_proj": (hidden_dim, hidden_dim),
        "gate_up_proj": (hidden_dim, gate_up_dim),
        "down_proj": (intermediate_size, hidden_dim),
    }

    captures = []
    for fwd_idx in range(n_fwd_passes):
        batch_size = fwd_batch_sizes[fwd_idx]
        for layer in range(n_layers):
            for proj_name in PROJ_SEQUENCE:
                in_dim, out_dim = proj_dims[proj_name]
                x_i8 = torch.randint(-128, 127, (batch_size, in_dim), dtype=torch.int8)
                acc_i32 = torch.randint(-1000, 1000, (batch_size, out_dim), dtype=torch.int32)
                scale_a = torch.tensor(0.05, dtype=torch.float32)
                captures.append((layer, proj_name, x_i8, acc_i32, scale_a))
    return captures


def _commit_params(n_traces):
    prompt = b"synthetic test prompt"
    seed = hashlib.sha256(prompt).digest()
    return {
        "token_ids": list(range(1, n_traces + 1)),
        "prompt": prompt,
        "sampling_seed": seed,
        "manifest": {
            "tokenizer_hash": "aa" * 32,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "eos_policy": "stop",
            "weight_hash": "bb" * 32,
            "quant_hash": "cc" * 32,
            "system_prompt_hash": "dd" * 32,
        },
    }


def run_smoke(n_layers, n_prompt_tokens, n_gen_tokens, hidden_dim, kv_dim, intermediate_size):
    """Single-run correctness check."""
    import json
    import verilm_rs

    q_dim = hidden_dim
    n_traces = n_prompt_tokens + n_gen_tokens - 1
    n_decode = n_gen_tokens - 1
    fwd_batch_sizes = [n_prompt_tokens] + [1] * n_decode

    print(f"Config: {n_layers}L, {hidden_dim}h, {n_prompt_tokens}p+{n_gen_tokens}g tokens")

    captures = make_synthetic_captures(
        n_layers, len(fwd_batch_sizes), fwd_batch_sizes,
        hidden_dim, kv_dim, intermediate_size,
    )

    t0 = time.monotonic()
    capture_inputs = [c[2].numpy().tobytes() for c in captures]
    capture_accs = [c[3].numpy().tobytes() for c in captures]
    capture_scales = [float(c[4].item()) for c in captures]
    t1 = time.monotonic()

    t2 = time.monotonic()
    params = _commit_params(n_traces)
    state = verilm_rs.commit_from_captures(
        capture_inputs=capture_inputs,
        capture_accumulators=capture_accs,
        capture_scales=capture_scales,
        fwd_batch_sizes=fwd_batch_sizes,
        n_layers=n_layers,
        q_dim=q_dim,
        kv_dim=kv_dim,
        intermediate_size=intermediate_size,
        level_c=True,
        **params,
    )
    t3 = time.monotonic()

    commitment = json.loads(state.commitment_json())
    assert commitment["n_tokens"] == n_traces
    proof = state.audit_stratified(0, [0, n_layers - 1], "routine")
    assert len(proof) > 0

    print(f"  to_bytes: {(t1-t0)*1000:.0f} ms  rust: {(t3-t2)*1000:.0f} ms  total: {(t3-t0)*1000:.0f} ms")
    print("  PASS")
    return True


def run_bench(n_iters, n_layers, n_prompt_tokens, n_gen_tokens,
              hidden_dim, kv_dim, intermediate_size):
    """Repeated benchmark: measures post-capture pipeline cost."""
    import verilm_rs

    q_dim = hidden_dim
    n_traces = n_prompt_tokens + n_gen_tokens - 1
    n_decode = n_gen_tokens - 1
    fwd_batch_sizes = [n_prompt_tokens] + [1] * n_decode

    print(f"Config: {n_layers}L, {hidden_dim}h, {n_prompt_tokens}p+{n_gen_tokens}g tokens")
    print(f"Iterations: {n_iters}")

    # Pre-generate synthetic data (don't measure this).
    all_captures = []
    for _ in range(n_iters):
        all_captures.append(make_synthetic_captures(
            n_layers, len(fwd_batch_sizes), fwd_batch_sizes,
            hidden_dim, kv_dim, intermediate_size,
        ))

    params = _commit_params(n_traces)
    times_bytes = []
    times_rust = []

    for i in range(n_iters):
        captures = all_captures[i]

        t0 = time.monotonic()
        capture_inputs = [c[2].numpy().tobytes() for c in captures]
        capture_accs = [c[3].numpy().tobytes() for c in captures]
        capture_scales = [float(c[4].item()) for c in captures]
        t1 = time.monotonic()
        times_bytes.append(t1 - t0)

        t2 = time.monotonic()
        state = verilm_rs.commit_from_captures(
            capture_inputs=capture_inputs,
            capture_accumulators=capture_accs,
            capture_scales=capture_scales,
            fwd_batch_sizes=fwd_batch_sizes,
            n_layers=n_layers,
            q_dim=q_dim,
            kv_dim=kv_dim,
            intermediate_size=intermediate_size,
            level_c=True,
            **params,
        )
        t3 = time.monotonic()
        times_rust.append(t3 - t2)

    def stats(t):
        t = sorted(t)
        return sum(t) / len(t), t[len(t) // 2]

    b_mean, b_med = stats(times_bytes)
    r_mean, r_med = stats(times_rust)
    total_mean = b_mean + r_mean
    total_med = b_med + r_med

    import json
    import subprocess

    print(f"\n{'='*55}")
    print(f"POST-CAPTURE PIPELINE BENCHMARK (CPU, synthetic data)")
    print(f"{'='*55}")
    print(f"{'Phase':<22} {'Mean (ms)':>10} {'Median (ms)':>12}")
    print(f"{'-'*44}")
    print(f"{'Convert to bytes':<22} {b_mean*1000:>10.1f} {b_med*1000:>12.1f}")
    print(f"{'Rust trace+commit':<22} {r_mean*1000:>10.1f} {r_med*1000:>12.1f}")
    print(f"{'Total':<22} {total_mean*1000:>10.1f} {total_med*1000:>12.1f}")
    print(f"{'='*55}")
    print(f"Captures: {len(all_captures[0])}, tokens: {n_traces}")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    result = {
        "commit": sha,
        "config": {
            "n_layers": n_layers, "hidden_dim": hidden_dim,
            "kv_dim": kv_dim, "intermediate_size": intermediate_size,
            "n_prompt_tokens": n_prompt_tokens, "n_gen_tokens": n_gen_tokens,
        },
        "iterations": n_iters,
        "to_bytes_mean_ms": round(b_mean * 1000, 1),
        "to_bytes_median_ms": round(b_med * 1000, 1),
        "rust_mean_ms": round(r_mean * 1000, 1),
        "rust_median_ms": round(r_med * 1000, 1),
        "total_mean_ms": round(total_mean * 1000, 1),
        "total_median_ms": round(total_med * 1000, 1),
    }
    print(f"\n{json.dumps(result)}")
    return True


def main():
    if "--bench" in sys.argv:
        idx = sys.argv.index("--bench")
        n_iters = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit() else 5
        ok = run_bench(
            n_iters=n_iters,
            n_layers=28, n_prompt_tokens=50, n_gen_tokens=32,
            hidden_dim=3584, kv_dim=512, intermediate_size=18944,
        )
    elif "--full" in sys.argv:
        ok = run_smoke(
            n_layers=28, n_prompt_tokens=50, n_gen_tokens=32,
            hidden_dim=3584, kv_dim=512, intermediate_size=18944,
        )
    else:
        ok = run_smoke(
            n_layers=4, n_prompt_tokens=8, n_gen_tokens=4,
            hidden_dim=256, kv_dim=64, intermediate_size=512,
        )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
