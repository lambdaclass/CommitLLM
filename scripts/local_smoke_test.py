#!/usr/bin/env python3
"""
Local smoke test: synthetic data through trace build + bytes serialize + Rust commit.

No GPU needed. Runs in seconds. Tests the full post-capture pipeline:
  synthetic captures → build_layer_traces → tobytes → verilm_rs.commit

Usage:
    python scripts/local_smoke_test.py          # quick (2 layers, 4 tokens)
    python scripts/local_smoke_test.py --full   # Qwen-sized (28 layers, 40 tokens)
"""

import hashlib
import sys
import time

import torch


def make_synthetic_captures(n_layers, n_fwd_passes, fwd_batch_sizes,
                            hidden_dim, kv_dim, intermediate_size):
    """Build synthetic capture tuples matching real capture hook output.

    Returns list of (layer, proj, x_i8, acc_i32, scale_a) tuples,
    same format as CaptureBuffer.drain().
    """
    from verilm.capture import PROJ_SEQUENCE, PROJS_PER_LAYER

    q_dim = hidden_dim  # num_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    o_dim = hidden_dim
    gate_up_dim = 2 * intermediate_size
    down_dim = hidden_dim

    proj_dims = {
        "qkv_proj": (hidden_dim, qkv_dim),
        "o_proj": (hidden_dim, o_dim),       # Attention output → hidden
        "gate_up_proj": (hidden_dim, gate_up_dim),
        "down_proj": (intermediate_size, down_dim),
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


def run_smoke(n_layers, n_prompt_tokens, n_gen_tokens, hidden_dim, kv_dim, intermediate_size):
    import verilm.capture as cap
    from verilm.trace import build_layer_traces

    # Configure capture module geometry.
    cap._n_layers = n_layers
    cap._calls_per_fwd = n_layers * cap.PROJS_PER_LAYER
    cap._configured = True
    cap._q_dim = hidden_dim
    cap._kv_dim = kv_dim
    cap._gate_up_half = intermediate_size

    total_tokens = n_prompt_tokens + n_gen_tokens
    n_traces = total_tokens - 1  # Last generated token has no forward pass

    # Prefill: 1 forward pass with batch_size = n_prompt_tokens
    # Decode: n_gen_tokens - 1 forward passes with batch_size = 1
    # (Last gen token is sampled from penultimate position's logits)
    n_decode = n_gen_tokens - 1
    fwd_batch_sizes = [n_prompt_tokens] + [1] * n_decode
    n_fwd_passes = len(fwd_batch_sizes)

    print(f"Config: {n_layers}L, {hidden_dim}h, {n_prompt_tokens}p+{n_gen_tokens}g tokens")
    print(f"  Forward passes: {n_fwd_passes} (1 prefill + {n_decode} decode)")
    print(f"  Expected traces: {n_traces}")

    # Phase 1: Build synthetic captures
    t0 = time.monotonic()
    captures = make_synthetic_captures(
        n_layers, n_fwd_passes, fwd_batch_sizes,
        hidden_dim, kv_dim, intermediate_size,
    )
    t1 = time.monotonic()
    print(f"  Synthetic captures: {len(captures)} tuples ({(t1-t0)*1000:.0f} ms)")

    # Phase 2: Build traces
    t2 = time.monotonic()
    traces = build_layer_traces(captures, n_layers=n_layers, level_c=True)
    t3 = time.monotonic()
    print(f"  Build traces: {len(traces)} tokens ({(t3-t2)*1000:.0f} ms)")

    assert len(traces) == n_traces, f"Expected {n_traces} traces, got {len(traces)}"

    # Phase 3: Serialize to bytes
    t4 = time.monotonic()
    trace_dicts = []
    for token_layers in traces:
        layer_dicts = []
        for lt in token_layers:
            d = {
                "x_attn": lt["x_attn"].numpy().tobytes(),
                "q": lt["q"].numpy().tobytes(),
                "k": lt["k"].numpy().tobytes(),
                "v": lt["v"].numpy().tobytes(),
                "a": lt["a"].numpy().tobytes(),
                "attn_out": lt["attn_out"].numpy().tobytes(),
                "x_ffn": lt["x_ffn"].numpy().tobytes(),
                "g": lt["g"].numpy().tobytes(),
                "u": lt["u"].numpy().tobytes(),
                "h": lt["h"].numpy().tobytes(),
                "ffn_out": lt["ffn_out"].numpy().tobytes(),
                "kv_cache_k": [t.numpy().tobytes() for t in lt.get("kv_cache_k", [])],
                "kv_cache_v": [t.numpy().tobytes() for t in lt.get("kv_cache_v", [])],
            }
            if "qkv_scale" in lt:
                d["scale_x_attn"] = float(lt["qkv_scale"].item())
            if "o_scale" in lt:
                d["scale_a"] = float(lt["o_scale"].item())
            if "gu_scale" in lt:
                d["scale_x_ffn"] = float(lt["gu_scale"].item())
            if "d_scale" in lt:
                d["scale_h"] = float(lt["d_scale"].item())
            layer_dicts.append(d)
        trace_dicts.append(layer_dicts)
    t5 = time.monotonic()
    print(f"  Serialize (tobytes): ({(t5-t4)*1000:.0f} ms)")

    # Phase 4: Rust commit
    import verilm_rs

    token_ids = list(range(1, n_traces + 1))  # Emitted token IDs
    prompt = b"synthetic test prompt"
    seed = hashlib.sha256(prompt).digest()
    manifest = {
        "tokenizer_hash": "aa" * 32,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "eos_policy": "stop",
        "weight_hash": "bb" * 32,
        "quant_hash": "cc" * 32,
        "system_prompt_hash": "dd" * 32,
    }

    t6 = time.monotonic()
    state = verilm_rs.commit(
        traces=trace_dicts,
        token_ids=token_ids,
        prompt=prompt,
        sampling_seed=seed,
        manifest=manifest,
    )
    t7 = time.monotonic()
    print(f"  Rust commit: ({(t7-t6)*1000:.0f} ms)")

    # Phase 5: Verify commitment
    commitment_json = state.commitment_json()
    import json
    commitment = json.loads(commitment_json)
    assert commitment["n_tokens"] == n_traces
    assert "merkle_root" in commitment
    assert "io_root" in commitment
    print(f"  Commitment OK: n_tokens={commitment['n_tokens']}")

    # Phase 6: Audit
    t8 = time.monotonic()
    proof = state.audit_stratified(0, [0, n_layers - 1], "routine")
    t9 = time.monotonic()
    assert len(proof) > 0
    print(f"  Audit (routine, 2 layers): {len(proof)} bytes ({(t9-t8)*1000:.0f} ms)")

    total = (t7 - t2)  # traces + serialize + commit
    print(f"\n  Total post-capture pipeline: {total*1000:.0f} ms")
    print("  PASS")
    return True


def main():
    full = "--full" in sys.argv

    if full:
        # Qwen2.5-7B dimensions
        ok = run_smoke(
            n_layers=28, n_prompt_tokens=50, n_gen_tokens=32,
            hidden_dim=3584, kv_dim=512, intermediate_size=18944,
        )
    else:
        # Quick: small model, few tokens
        ok = run_smoke(
            n_layers=4, n_prompt_tokens=8, n_gen_tokens=4,
            hidden_dim=256, kv_dim=64, intermediate_size=512,
        )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
