"""
Restructure raw captures into LayerTrace-compatible format.

Takes the flat (layer, proj, x_i8, acc_i32, scale_a) tuples from
capture.py and builds per-token, per-layer trace dicts matching
the Rust LayerTrace struct in crates/vi-core/src/types.rs.

Handles:
  - GQA-aware QKV split (Q width != K/V width)
  - Gate/up equal split
  - KV cache accumulation across tokens (Level C)
  - Requantization of k/v i32 → i8 (saturating clamp to [-128, 127])
"""

from typing import List, Dict, Optional, Tuple
import torch

from . import capture


def requantize_i32_to_i8(acc: torch.Tensor) -> torch.Tensor:
    """Requantize INT32 accumulator to INT8 via saturating clamp.

    Matches the Rust requantize(): clamp to [-128, 127], cast to i8.
    No scale factor — unit-scale saturating clamp.
    """
    return acc.clamp(-128, 127).to(torch.int8)


def split_qkv(acc_i32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused QKV accumulator into Q, K, V using GQA-aware dimensions.

    Q has width q_dim = num_heads * head_dim.
    K and V each have width kv_dim = num_kv_heads * head_dim.
    Total fused width = q_dim + 2 * kv_dim.
    """
    q_dim = capture._q_dim
    kv_dim = capture._kv_dim

    if q_dim == 0 or kv_dim == 0:
        raise RuntimeError(
            "verilm: model geometry not configured — call configure_from_model() first"
        )

    expected_width = q_dim + 2 * kv_dim
    actual_width = acc_i32.shape[-1]
    if actual_width != expected_width:
        raise ValueError(
            f"verilm: QKV split mismatch — got width {actual_width}, "
            f"expected {expected_width} (q_dim={q_dim}, kv_dim={kv_dim})"
        )

    q = acc_i32[..., :q_dim]
    k = acc_i32[..., q_dim:q_dim + kv_dim]
    v = acc_i32[..., q_dim + kv_dim:]
    return q, k, v


def split_gate_up(acc_i32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fused gate_up accumulator into gate and up (equal halves)."""
    half = acc_i32.shape[-1] // 2
    return acc_i32[..., :half], acc_i32[..., half:]


def build_layer_traces(
    captures: list,
    n_layers: int = 0,
    level_c: bool = False,
) -> List[List[dict]]:
    """Build per-token, per-layer trace dicts from raw captures.

    Args:
        captures: list of (layer, proj, x_i8, acc_i32, scale_a) tuples
            from CaptureBuffer.drain().
        n_layers: number of transformer layers. If 0, uses capture._n_layers.
        level_c: if True, accumulate full KV cache snapshots per token.

    Returns:
        List of tokens, each containing a list of layer dicts matching
        the Rust LayerTrace fields:
            x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
            kv_cache_k, kv_cache_v
    """
    if n_layers == 0:
        n_layers = capture._n_layers
    if n_layers == 0:
        raise RuntimeError("verilm: layer count not configured")

    calls_per_fwd = n_layers * capture.PROJS_PER_LAYER
    n_captures = len(captures)

    if n_captures == 0:
        return []
    if n_captures % calls_per_fwd != 0:
        raise ValueError(
            f"verilm: capture count {n_captures} not a multiple of "
            f"calls_per_fwd {calls_per_fwd}"
        )

    n_tokens = n_captures // calls_per_fwd

    # KV cache accumulator for Level C: kv_cache[layer] = (list_of_k_i8, list_of_v_i8)
    if level_c:
        kv_cache = [([], []) for _ in range(n_layers)]

    all_tokens = []
    for t in range(n_tokens):
        token_layers = []
        for l in range(n_layers):
            base = t * calls_per_fwd + l * capture.PROJS_PER_LAYER
            # Unpack the 4 projections in order: qkv, o, gate_up, down
            _, _, qkv_x, qkv_acc, qkv_scale = captures[base + 0]
            _, _, o_x, o_acc, o_scale = captures[base + 1]
            _, _, gu_x, gu_acc, gu_scale = captures[base + 2]
            _, _, d_x, d_acc, d_scale = captures[base + 3]

            # GQA-aware QKV split
            q, k, v = split_qkv(qkv_acc)

            # Gate/up equal split
            g, u = split_gate_up(gu_acc)

            layer_dict = {
                "x_attn": qkv_x,       # INT8 input to attention block
                "q": q,                  # W_q output (i32)
                "k": k,                  # W_k output (i32)
                "v": v,                  # W_v output (i32)
                "a": o_x,               # attention output, input to W_o (INT8)
                "attn_out": o_acc,       # W_o output (i32)
                "x_ffn": gu_x,          # INT8 input to FFN block
                "g": g,                  # W_gate output (i32)
                "u": u,                  # W_up output (i32)
                "h": d_x,               # SiLU(gate)*up, requantized (INT8)
                "ffn_out": d_acc,        # W_down output (i32)
                # Scales (not in Rust LayerTrace but needed for commitment)
                "qkv_scale": qkv_scale,
                "o_scale": o_scale,
                "gu_scale": gu_scale,
                "d_scale": d_scale,
            }

            if level_c:
                # Accumulate KV cache: requantize k/v and append
                k_i8 = requantize_i32_to_i8(k)
                v_i8 = requantize_i32_to_i8(v)
                kv_cache[l][0].append(k_i8)
                kv_cache[l][1].append(v_i8)
                # Snapshot: clone the accumulated cache at this point
                layer_dict["kv_cache_k"] = [t.clone() for t in kv_cache[l][0]]
                layer_dict["kv_cache_v"] = [t.clone() for t in kv_cache[l][1]]
            else:
                layer_dict["kv_cache_k"] = []
                layer_dict["kv_cache_v"] = []

            token_layers.append(layer_dict)
        all_tokens.append(token_layers)

    return all_tokens
