"""
Measure the witnessed-score corridor vs standard replay corridor.

Same-run A/B comparison: for each model/workload, measures both the
standard committed-KV corridor (replay from Q·K^T) and the score-witnessed
corridor (softmax(GPU-scores) @ V).

Reports: L-inf comparison, payload MB, verifier CPU time.

Models:
  - neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8
  - neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8

Usage:
    modal run --detach scripts/modal/measure_corridor_witnessed.py
"""

import os

import modal

app = modal.App("verilm-measure-corridor-witnessed")

VLLM_SPEC = os.environ.get("VERILM_VLLM_SPEC", "vllm==0.18.0")
TORCH_SPEC = os.environ.get("VERILM_TORCH_SPEC", "torch")
TRANSFORMERS_SPEC = os.environ.get("VERILM_TRANSFORMERS_SPEC", "transformers<5")
COMPRESSED_TENSORS_SPEC = os.environ.get(
    "VERILM_COMPRESSED_TENSORS_SPEC", "compressed-tensors==0.13.0"
)

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
        "VERILM_SCORE_WITNESS": "1",
    })
    .pip_install(
        VLLM_SPEC,
        TORCH_SPEC,
        TRANSFORMERS_SPEC,
        COMPRESSED_TENSORS_SPEC,
        "numpy",
        "fastapi",
        "maturin",
    )
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts", "*.pdf", "site", ".cache",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

# Two context sizes: 4K (default) and 32K.
CONTEXT_CONFIGS = [
    {"name": "4k", "max_model_len": 4096},
    {"name": "32k", "max_model_len": 32768},
]

# Workloads: short for quick sanity, long for meaningful corridor measurement.
WORKLOADS_4K = [
    {"name": "short", "prompt": "What is 2+2?", "max_tokens": 32},
    {
        "name": "medium",
        "prompt": "Explain the theory of relativity in one paragraph.",
        "max_tokens": 128,
    },
    {
        "name": "long",
        "prompt": (
            "You are a computer science professor. Explain in detail how modern "
            "transformer architectures work, starting from the attention mechanism, "
            "covering multi-head attention, positional encodings, layer normalization, "
            "feed-forward networks, and the training process."
        ),
        "max_tokens": 512,
    },
]

WORKLOADS_32K = [
    {
        "name": "long_context",
        "prompt": (
            "You are a senior software architect writing a comprehensive design "
            "document covering: (1) microservices vs monolithic trade-offs; "
            "(2) event-driven systems with Kafka; (3) database sharding; "
            "(4) observability with OpenTelemetry; (5) CI/CD zero-downtime. "
            "Be extremely thorough with concrete examples and code snippets."
        ),
        "max_tokens": 2048,
    },
]


def _extract_weight_scales(llm, n_layers, cfg):
    """Extract per-channel weight scales for faithful verifier replay."""
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    hidden_dim = cfg["hidden_dim"]
    kv_dim = cfg["kv_dim"]
    scales = {"wq": [], "wk": [], "wv": []}
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        ws = layer.self_attn.qkv_proj.weight_scale
        ws_np = ws.detach().cpu().float().numpy().flatten()
        wq = ws_np[:hidden_dim]
        wk = ws_np[hidden_dim:hidden_dim + kv_dim]
        wv = ws_np[hidden_dim + kv_dim:]
        scales["wq"].append(wq.tolist())
        scales["wk"].append(wk.tolist())
        scales["wv"].append(wv.tolist())
    return scales


def _measure_one_ab(server, wl, full_layers, key_json, model_id, buf, scale_overrides_json):
    """Same-run A/B: standard corridor vs witnessed-score corridor for one workload.

    Uses the SAME inference run — both corridors measured from the same
    committed state, ensuring fair comparison.
    """
    import json
    import time

    import verilm_rs

    chat_r = server.chat(prompt=wl["prompt"], max_tokens=wl["max_tokens"], temperature=0.0)
    n_tokens = chat_r["n_tokens"]
    request_id = chat_r["request_id"]
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]

    # Audit the last generated token (most meaningful for score witnessing).
    gen_token_pos = n_tokens - 1
    if gen_token_pos < 0:
        print(f"  WARN: no tokens generated for {wl['name']}, skipping")
        return None

    audit_binary = server.audit(
        request_id=request_id,
        token_index=gen_token_pos,
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )
    audit_size_mb = len(audit_binary) / 1e6

    # Compute incremental score payload size from model config.
    # witnessed_scores: n_layers × (n_q_heads × seq_len × 4 bytes f32)
    # seq_len = token position + 1 (all KV positions up to generated token)
    n_q_heads_model = buf._sw_n_q_heads if buf._sw_enabled else 0
    n_layers_model = len(full_layers)
    seq_len_for_scores = gen_token_pos + 1  # prompt + all prior generated
    score_payload_bytes = n_layers_model * n_q_heads_model * seq_len_for_scores * 4
    score_payload_mb = score_payload_bytes / 1e6

    # ── A: Standard committed-KV corridor ──
    t0 = time.monotonic()
    report_json_std = verilm_rs.measure_corridor_committed_kv(
        audit_binary, key_json, scale_overrides_json,
    )
    dt_standard_ms = (time.monotonic() - t0) * 1000
    report_std = json.loads(report_json_std)

    # ── B: Witnessed-score corridor ──
    t0 = time.monotonic()
    report_json_ws = verilm_rs.measure_corridor_witnessed_scores(
        audit_binary, key_json, scale_overrides_json,
    )
    dt_witnessed_ms = (time.monotonic() - t0) * 1000
    report_ws = json.loads(report_json_ws)

    # ── C: Score anchoring — f64 (baseline) ──
    t0 = time.monotonic()
    anchor_json = verilm_rs.verify_witnessed_score_anchoring(
        audit_binary, key_json, scale_overrides_json,
    )
    dt_anchor_ms = (time.monotonic() - t0) * 1000
    anchor = json.loads(anchor_json)

    # ── D: Score anchoring — GPU-like precision (f32 + fp16 truncation) ──
    t0 = time.monotonic()
    anchor_gpu_json = verilm_rs.verify_witnessed_score_anchoring_gpu_like(
        audit_binary, key_json, scale_overrides_json,
    )
    dt_anchor_gpu_ms = (time.monotonic() - t0) * 1000
    anchor_gpu = json.loads(anchor_gpu_json)

    # ── E: Score anchoring — bf16-matched precision ──
    t0 = time.monotonic()
    anchor_bf16_json = verilm_rs.verify_witnessed_score_anchoring_bf16(
        audit_binary, key_json, scale_overrides_json,
    )
    dt_anchor_bf16_ms = (time.monotonic() - t0) * 1000
    anchor_bf16 = json.loads(anchor_bf16_json)

    result = {
        "model": model_id,
        "workload": wl["name"],
        "max_tokens": wl["max_tokens"],
        "n_tokens": n_tokens,
        "n_prompt_tokens": n_prompt,
        "token_position": gen_token_pos,
        "audit_size_mb": round(audit_size_mb, 3),
        "score_payload_mb": round(score_payload_mb, 3),
        "standard": {
            "global_linf": report_std["global_linf"],
            "per_layer_max_linf": report_std["per_layer_max_linf"],
            "verifier_ms": round(dt_standard_ms, 1),
        },
        "witnessed": {
            "global_linf": report_ws["global_linf"],
            "per_layer_max_linf": report_ws["per_layer_max_linf"],
            "verifier_ms": round(dt_witnessed_ms, 1),
        },
        "anchoring_f64": {
            "global_max_gap": anchor["global_max_gap"],
            "n_layers": anchor["n_layers"],
            "verifier_ms": round(dt_anchor_ms, 1),
            "per_layer_max_gap": [l["max_gap"] for l in anchor["layers"]],
        },
        "anchoring_gpu_like": {
            "global_max_gap": anchor_gpu["global_max_gap"],
            "n_layers": anchor_gpu["n_layers"],
            "verifier_ms": round(dt_anchor_gpu_ms, 1),
            "per_layer_max_gap": [l["max_gap"] for l in anchor_gpu["layers"]],
        },
        "anchoring_bf16": {
            "global_max_gap": anchor_bf16["global_max_gap"],
            "n_layers": anchor_bf16["n_layers"],
            "verifier_ms": round(dt_anchor_bf16_ms, 1),
            "per_layer_max_gap": [l["max_gap"] for l in anchor_bf16["layers"]],
        },
    }

    # Per-layer comparison summary.
    std_linfs = report_std["per_layer_max_linf"]
    ws_linfs = report_ws["per_layer_max_linf"]
    print(
        f"  {wl['name']:>12s}  pos={gen_token_pos:4d}  n_tok={n_tokens:4d}  "
        f"audit={audit_size_mb:.2f}MB  scores={score_payload_mb:.2f}MB  "
        f"std_linf={report_std['global_linf']:3d}  "
        f"ws_linf={report_ws['global_linf']:3d}  "
        f"anchor_f64={anchor['global_max_gap']:.4f}  "
        f"anchor_gpu={anchor_gpu['global_max_gap']:.4f}  "
        f"anchor_bf16={anchor_bf16['global_max_gap']:.4f}  "
        f"ws_ms={dt_witnessed_ms:.1f}"
    )

    # Print worst layers for both
    if std_linfs:
        worst_std = sorted(range(len(std_linfs)), key=lambda i: std_linfs[i], reverse=True)[:3]
        worst_ws = sorted(range(len(ws_linfs)), key=lambda i: ws_linfs[i], reverse=True)[:3]
        print(
            f"    worst std layers: {[(i, std_linfs[i]) for i in worst_std]}  "
            f"worst ws layers: {[(i, ws_linfs[i]) for i in worst_ws]}"
        )

    return result


def _diagnose_anchor_mismatch(server, buf, full_layers, key_json, model_id, scale_overrides_json, cfg):
    """Diagnose score anchoring mismatch by computing three score variants.

    For the worst anchoring layer:
      A: Q_gpu × K_gpu           (witnessed scores from GPU)
      B: Q_verifier × K_gpu      (cross: verifier Q, GPU K)
      C: Q_verifier × K_committed (canonical scores from verifier)

    If A ≈ B but B ≠ C → K is the problem (GPU K ≠ committed K)
    If A ≠ B but B ≈ C → Q is the problem (GPU Q ≠ verifier Q)
    If all differ → both Q and K contribute
    """
    import json
    import math

    import numpy as np
    import torch
    import verilm_rs

    n_q_heads = cfg["n_q_heads"]
    n_kv_heads = cfg["n_kv_heads"]
    d_head = cfg["d_head"]
    heads_per_kv = n_q_heads // n_kv_heads
    half = d_head // 2
    scale = 1.0 / math.sqrt(d_head)

    # Run a short inference to get fresh capture + audit.
    chat_r = server.chat(prompt="Explain TCP three-way handshake.", max_tokens=32, temperature=0.0)
    n_tokens = chat_r["n_tokens"]
    request_id = chat_r["request_id"]
    gen_pos = n_tokens - 1
    if gen_pos < 0:
        print("  DIAG: no tokens generated, skipping")
        return

    audit_binary = server.audit(
        request_id=request_id,
        token_index=gen_pos,
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )

    # Find worst anchoring layer.
    anchor_json = verilm_rs.verify_witnessed_score_anchoring(
        audit_binary, key_json, scale_overrides_json,
    )
    anchor = json.loads(anchor_json)
    if not anchor["layers"]:
        print("  DIAG: no anchoring layers, skipping")
        return

    worst_layer = max(anchor["layers"], key=lambda l: l["max_gap"])
    layer_idx = worst_layer["layer"]
    print(f"\n  DIAG: worst anchor layer={layer_idx}  gap={worst_layer['max_gap']:.4f}")

    # Get GPU Q/K pre-RoPE from drain snapshot.
    try:
        q_gpu_pre, k_gpu_pre = buf.get_diagnostic_qk(layer_idx)
    except (RuntimeError, IndexError) as e:
        print(f"  DIAG: no GPU Q/K for layer {layer_idx}: {e}")
        return

    # Get verifier intermediates from Rust.
    diag_json = verilm_rs.diagnose_score_anchoring(
        audit_binary, key_json, scale_overrides_json, [layer_idx],
    )
    diags = json.loads(diag_json)
    if not diags:
        print("  DIAG: no diagnostic data returned")
        return
    d = diags[0]
    seq_len = d["seq_len"]

    # Verifier Q pre-RoPE (from Rust).
    q_ver_pre = np.array(d["q_verifier_pre_rope"], dtype=np.float64)

    # ── Element-wise Q comparison (pre-RoPE) ──
    q_gpu_f64 = q_gpu_pre.astype(np.float64)
    q_diff = np.abs(q_gpu_f64 - q_ver_pre)
    print(f"  DIAG Q pre-RoPE: max_diff={q_diff.max():.6f}  mean_diff={q_diff.mean():.6f}  "
          f"q_gpu_mag={np.abs(q_gpu_f64).max():.2f}  q_ver_mag={np.abs(q_ver_pre).max():.2f}")

    # ── RoPE constants (used for both K pre-RoPE and post-RoPE comparisons) ──
    inv_freq = buf._sw_inv_freq.cpu().numpy().astype(np.float64)

    # ── Element-wise K comparison: pre-RoPE (GPU vs committed un-RoPE'd) ──
    # Undo RoPE on committed K at gen_pos to get prover's pre-RoPE K.
    # committed K = RoPE(dequant(W_k @ x_int8, scales) + bias)
    # un-RoPE: rotate by -angle to recover pre-RoPE values.
    k_com_gen_roped = np.array(d["k_committed_roped"][gen_pos], dtype=np.float64)
    # Reshape to (n_kv_heads, d_head), un-RoPE, flatten back.
    k_com_heads = k_com_gen_roped.reshape(n_kv_heads, d_head)
    gen_angles = float(gen_pos) * inv_freq  # (half,)
    gen_cos = np.cos(gen_angles)
    gen_sin = np.sin(gen_angles)
    # RoPE forward: r0 = x0*cos - x1*sin, r1 = x1*cos + x0*sin
    # RoPE inverse: x0 = r0*cos + r1*sin, x1 = r1*cos - r0*sin
    k_com_pre_heads = np.empty_like(k_com_heads)
    k_com_pre_heads[:, :half] = k_com_heads[:, :half] * gen_cos + k_com_heads[:, half:] * gen_sin
    k_com_pre_heads[:, half:] = k_com_heads[:, half:] * gen_cos - k_com_heads[:, :half] * gen_sin
    k_com_pre = k_com_pre_heads.flatten()

    k_gpu_gen_f64 = k_gpu_pre[gen_pos].astype(np.float64) if gen_pos < len(k_gpu_pre) else k_gpu_pre[-1].astype(np.float64)

    k_pre_diff = np.abs(k_gpu_gen_f64 - k_com_pre)
    print(f"  DIAG K pre-RoPE (gpu vs committed-unroped): max_diff={k_pre_diff.max():.6f}  "
          f"mean_diff={k_pre_diff.mean():.6f}  "
          f"k_gpu_mag={np.abs(k_gpu_gen_f64).max():.2f}  k_com_mag={np.abs(k_com_pre).max():.2f}")
    worst_k_idx = np.argmax(k_pre_diff)
    print(f"    worst idx={worst_k_idx}: gpu={k_gpu_gen_f64[worst_k_idx]:.6f}  "
          f"com={k_com_pre[worst_k_idx]:.6f}  diff={k_pre_diff[worst_k_idx]:.6f}")
    # If pre-RoPE diff is already large (~1.0), fault is before RoPE: dequant/bias/accumulator
    # If pre-RoPE diff is small (~0) but post-RoPE diff is ~1.0, fault is in RoPE
    if k_pre_diff.max() > 0.1:
        print(f"  → K fault is BEFORE RoPE (dequant/bias/accumulator mismatch)")
    else:
        print(f"  → K pre-RoPE matches well; fault is likely in RoPE application")

    # ── Element-wise K comparison (committed vs GPU, both post-RoPE) ──
    k_gpu_f64 = k_gpu_pre[:seq_len].astype(np.float64)  # (seq_len, kv_dim)

    # RoPE on GPU K: reshape to (n_kv_heads, seq_len, d_head), apply per-position.
    k_gpu_heads = k_gpu_f64.reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)
    positions = np.arange(seq_len, dtype=np.float64)
    angles = positions[:, None] * inv_freq[None, :]  # (seq_len, half)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    k_gpu_roped = np.empty_like(k_gpu_heads)
    k_gpu_roped[:, :, :half] = k_gpu_heads[:, :, :half] * cos_a - k_gpu_heads[:, :, half:] * sin_a
    k_gpu_roped[:, :, half:] = k_gpu_heads[:, :, half:] * cos_a + k_gpu_heads[:, :, :half] * sin_a
    # Flatten back to (seq_len, kv_dim).
    k_gpu_roped_flat = k_gpu_roped.transpose(1, 0, 2).reshape(seq_len, n_kv_heads * d_head)

    # Committed K_roped from verifier.
    k_com_roped = np.array(d["k_committed_roped"], dtype=np.float64)  # (seq_len, kv_dim)

    k_diff = np.abs(k_gpu_roped_flat - k_com_roped)
    print(f"  DIAG K roped gpu-vs-com: max_diff={k_diff.max():.6f}  mean_diff={k_diff.mean():.6f}  "
          f"k_gpu_mag={np.abs(k_gpu_roped_flat).max():.2f}  k_com_mag={np.abs(k_com_roped).max():.2f}")

    # ── Three score variants ──
    # Apply RoPE to verifier Q and GPU Q.
    q_ver_roped = np.array(d["q_verifier_roped"], dtype=np.float64)

    # RoPE on GPU Q at gen_pos.
    q_gpu_heads = q_gpu_f64.reshape(n_q_heads, d_head)
    q_angles = float(gen_pos) * inv_freq
    q_cos = np.cos(q_angles)
    q_sin = np.sin(q_angles)
    q_gpu_roped = np.empty_like(q_gpu_heads)
    q_gpu_roped[:, :half] = q_gpu_heads[:, :half] * q_cos - q_gpu_heads[:, half:] * q_sin
    q_gpu_roped[:, half:] = q_gpu_heads[:, half:] * q_cos + q_gpu_heads[:, :half] * q_sin

    # Compute scores for all three combos.
    def compute_scores(q_roped_2d, k_roped_2d):
        """q: (n_q_heads, d_head), k: (seq_len, kv_dim) → (n_q_heads, seq_len)"""
        scores = np.zeros((n_q_heads, seq_len), dtype=np.float64)
        k_heads = k_roped_2d.reshape(seq_len, n_kv_heads, d_head)
        for qh in range(n_q_heads):
            kv_h = qh // heads_per_kv
            for t in range(seq_len):
                scores[qh, t] = np.dot(q_roped_2d[qh], k_heads[t, kv_h]) * scale
        return scores.ravel()

    # A: Q_gpu × K_gpu (should ≈ witnessed scores)
    scores_a = compute_scores(q_gpu_roped, k_gpu_roped_flat)
    # B: Q_verifier × K_gpu (cross)
    q_ver_roped_2d = q_ver_roped.reshape(n_q_heads, d_head)
    scores_b = compute_scores(q_ver_roped_2d, k_gpu_roped_flat)
    # C: Q_verifier × K_committed (canonical)
    scores_c = compute_scores(q_ver_roped_2d, k_com_roped)

    gap_ab = np.abs(scores_a - scores_b)
    gap_bc = np.abs(scores_b - scores_c)
    gap_ac = np.abs(scores_a - scores_c)

    print(f"  DIAG scores (layer {layer_idx}, pos={gen_pos}, seq_len={seq_len}):")
    print(f"    A=Q_gpu×K_gpu  B=Q_ver×K_gpu  C=Q_ver×K_com")
    print(f"    |A-B| max={gap_ab.max():.6f}  mean={gap_ab.mean():.6f}   (isolates Q mismatch)")
    print(f"    |B-C| max={gap_bc.max():.6f}  mean={gap_bc.mean():.6f}   (isolates K mismatch)")
    print(f"    |A-C| max={gap_ac.max():.6f}  mean={gap_ac.mean():.6f}   (total anchor gap)")

    # Also compare A against actual witnessed scores (sanity check).
    ws_json = verilm_rs.verify_witnessed_score_anchoring(audit_binary, key_json, scale_overrides_json)
    ws_data = json.loads(ws_json)
    ws_layer = [l for l in ws_data["layers"] if l["layer"] == layer_idx]
    if ws_layer:
        print(f"    Rust anchor gap for this layer: {ws_layer[0]['max_gap']:.6f}")

    # Interpretation.
    if gap_ab.max() > 10 * gap_bc.max():
        print("  → Q is the dominant mismatch source")
    elif gap_bc.max() > 10 * gap_ab.max():
        print("  → K is the dominant mismatch source")
    elif gap_ab.max() > 1.0 and gap_bc.max() > 1.0:
        print("  → Both Q and K contribute significantly")
    else:
        print("  → Gap is small (<1.0) on both legs")


def _diagnose_fused_qkv(llm, server, buf, full_layers, key_json, scale_overrides_json, cfg):
    """Diagnose fused QKV mismatch for score anchoring.

    Compares:
    1. Keygen K weight_scale (from safetensors) vs vLLM fused qkv_proj K scale slice
    2. i32 K accumulators: GPU fused path vs prover standalone path
    3. Dequantized K values: GPU fp16 vs prover f64
    """
    import json

    import numpy as np
    import torch
    import verilm_rs

    n_q_heads = cfg["n_q_heads"]
    n_kv_heads = cfg["n_kv_heads"]
    d_head = cfg["d_head"]
    hidden_dim = cfg["hidden_dim"]
    kv_dim = n_kv_heads * d_head
    q_dim = n_q_heads * d_head

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer_idx = 27  # worst layer from diagnostics

    print(f"\n  FUSED-QKV DIAGNOSTIC: layer={layer_idx}")

    # ── 1. Compare weight scales ──
    # vLLM fused scale
    layer = model.model.layers[layer_idx]
    fused_ws = layer.self_attn.qkv_proj.weight_scale.detach().cpu().float().numpy().flatten()
    fused_k_scale = fused_ws[q_dim:q_dim + kv_dim]

    # Keygen scale (from the verifier key)
    key = json.loads(key_json)
    keygen_k_scale = None
    if key.get("per_channel_weight_scales") and layer_idx < len(key["per_channel_weight_scales"]):
        # MatrixType::PER_LAYER order: Wq=0, Wk=1, Wv=2, Wo=3, Wg=4, Wu=5, Wd=6
        layer_scales = key["per_channel_weight_scales"][layer_idx]
        if len(layer_scales) > 1 and len(layer_scales[1]) > 0:
            keygen_k_scale = np.array(layer_scales[1], dtype=np.float32)

    if keygen_k_scale is not None:
        scale_diff = np.abs(fused_k_scale - keygen_k_scale)
        print(f"  K weight_scale: fused_len={len(fused_k_scale)}  keygen_len={len(keygen_k_scale)}")
        print(f"  K scale diff: max={scale_diff.max():.10f}  mean={scale_diff.mean():.10f}")
        if scale_diff.max() > 0:
            # Find where they differ
            worst_idx = np.argmax(scale_diff)
            print(f"    worst idx={worst_idx}: fused={fused_k_scale[worst_idx]:.10f}  "
                  f"keygen={keygen_k_scale[worst_idx]:.10f}")
        else:
            print("  K scales match exactly!")
    else:
        print("  WARNING: keygen has no per-channel K scales")

    # Also check slicing: is fused scale layout [Q; K; V]?
    print(f"  fused_ws shape: {fused_ws.shape}  expected: {q_dim + 2*kv_dim}")
    print(f"  Q slice: [{0}:{q_dim}]  K slice: [{q_dim}:{q_dim+kv_dim}]  "
          f"V slice: [{q_dim+kv_dim}:{q_dim+2*kv_dim}]")

    # ── 2. Compare i32 K accumulators ──
    # Run a short inference to get x_attn.
    chat_r = server.chat(prompt="Explain TCP.", max_tokens=8, temperature=0.0)
    n_tokens = chat_r["n_tokens"]
    request_id = chat_r["request_id"]
    gen_pos = n_tokens - 1

    # Get fused QKV weight (INT8) for this layer.
    fused_w_i8 = layer.self_attn.qkv_proj.weight.detach().cpu().to(torch.int8)
    # Slice K rows from fused weight.
    w_k_fused_slice = fused_w_i8[q_dim:q_dim + kv_dim, :]  # (kv_dim, hidden_dim) i8

    # Get x_attn from capture buffer snapshot.
    # x_attn_inputs are stored per qkv_proj call. For the last decode step,
    # layer_idx's x_attn is at index (n_decode_tokens - 1) * n_layers + layer_idx.
    # But after drain, we'd need to save it. Let me get it from the audit instead.
    audit_binary = server.audit(
        request_id=request_id,
        token_index=gen_pos,
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )

    # Get shell K i32 accumulator from audit.
    diag_json = verilm_rs.diagnose_score_anchoring(
        audit_binary, key_json, scale_overrides_json, [layer_idx],
    )
    diags = json.loads(diag_json)
    if not diags:
        print("  FUSED-QKV DIAG: no diagnostic data")
        return

    d = diags[0]
    # The shell opening has q and k i32 accumulators. But diagnose_score_anchoring
    # returns dequanted values, not raw i32. Let me check if we can get raw i32.
    # For now, compare the dequanted+biased Q/K values.

    q_ver_pre = np.array(d["q_verifier_pre_rope"], dtype=np.float64)  # dequant+bias Q
    # K from committed KV (at gen_pos) is in k_committed_roped[gen_pos]
    # But we need pre-RoPE K. The committed K already has RoPE.

    # ── 3. Compare GPU K output vs standalone K reconstruction ──
    # GPU K (fp16 from fused cutlass output, stored in diagnostic snapshot)
    try:
        _, k_gpu_pre = buf.get_diagnostic_qk(layer_idx)
        # k_gpu_pre is (seq_len, kv_dim) f32, pre-RoPE
        # Take the last position (generated token)
        k_gpu_gen = k_gpu_pre[-1]  # (kv_dim,) - GPU K for generated token

        # Compare GPU K scale against keygen K scale
        fused_input_scale = layer.self_attn.qkv_proj.input_scale
        if fused_input_scale is not None:
            fused_input_scale = fused_input_scale.item()
            print(f"  fused input_scale (scale_a): {fused_input_scale:.10f}")

        # Now let's also check if input_scale matches what the shell reports.
        print(f"  GPU K gen-token sample (first 5): {k_gpu_gen[:5]}")
        print(f"  GPU K gen-token magnitude: max={np.abs(k_gpu_gen).max():.4f}")

    except Exception as e:
        print(f"  WARNING: could not get GPU K: {e}")

    # ── 4. Check fused weight layout empirically ──
    # Compare first few elements of fused K scale slice with different offsets
    # to verify the [Q;K;V] layout assumption.
    q_scale_slice = fused_ws[:q_dim]
    k_scale_slice = fused_ws[q_dim:q_dim + kv_dim]
    v_scale_slice = fused_ws[q_dim + kv_dim:]
    print(f"  Q scale range: [{q_scale_slice.min():.6f}, {q_scale_slice.max():.6f}]")
    print(f"  K scale range: [{k_scale_slice.min():.6f}, {k_scale_slice.max():.6f}]")
    print(f"  V scale range: [{v_scale_slice.min():.6f}, {v_scale_slice.max():.6f}]")

    # ── 5. Check if separate k_proj exists vs fused ──
    # Try to access separate k_proj to see if vLLM exposes it
    has_separate = hasattr(layer.self_attn, 'k_proj')
    print(f"  vLLM has separate k_proj: {has_separate}")
    if has_separate:
        sep_k_ws = layer.self_attn.k_proj.weight_scale.detach().cpu().float().numpy().flatten()
        sep_diff = np.abs(sep_k_ws - fused_k_scale)
        print(f"  separate k_proj scale diff vs fused slice: max={sep_diff.max():.10f}")


def _diagnose_dequant_cast_order(
    server, buf, llm, full_layers, key_json, scale_overrides_json, cfg
):
    """Diagnose which dequant+bias cast order matches the GPU's actual K output.

    Runs 5 different bf16 cast-order strategies on the K accumulator and compares
    each against the GPU's pre-RoPE K. The strategy with the lowest max_diff tells
    us what CUTLASS actually does internally.

    Strategies:
      - bf16_final:          bf16(f32_dequant + f32_bias)
      - bf16_before_bias:    bf16(f32_dequant) + bf16(bias) in bf16
      - bf16_dequant_f32_bias: bf16(bf16(f32_dequant) + f32_bias)
      - f32_all:             f32 only (no bf16 narrowing)
      - bf16_per_mult:       bf16(bf16(a*sw) * sx) + bf16(bias)
    """
    import json

    import numpy as np
    import verilm_rs

    print(f"\n--- Dequant cast-order diagnostic ---")

    # Run a short inference to get fresh data.
    chat_r = server.chat(
        prompt="Explain TCP three-way handshake.", max_tokens=32, temperature=0.0
    )
    n_tokens = chat_r["n_tokens"]
    request_id = chat_r["request_id"]
    gen_pos = n_tokens - 1
    if gen_pos < 0:
        print("  CAST-ORDER: no tokens generated, skipping")
        return

    audit_binary = server.audit(
        request_id=request_id,
        token_index=gen_pos,
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )

    # Find worst anchoring layer.
    anchor_json = verilm_rs.verify_witnessed_score_anchoring(
        audit_binary, key_json, scale_overrides_json,
    )
    anchor = json.loads(anchor_json)
    if not anchor["layers"]:
        print("  CAST-ORDER: no anchoring layers, skipping")
        return

    worst_layer = max(anchor["layers"], key=lambda l: l["max_gap"])
    layer_idx = worst_layer["layer"]
    print(f"  Worst anchor layer={layer_idx}  gap={worst_layer['max_gap']:.4f}")

    # Get GPU K pre-RoPE from diagnostic snapshot.
    try:
        _, k_gpu_pre = buf.get_diagnostic_qk(layer_idx)
    except (RuntimeError, IndexError) as e:
        print(f"  CAST-ORDER: no GPU K for layer {layer_idx}: {e}")
        return

    k_gpu_gen = k_gpu_pre[-1].astype(np.float64)  # generated token, pre-RoPE

    # Run all cast-order variants via Rust (extracts K acc from shell internally).
    diag_json = verilm_rs.diagnose_cast_order(
        audit_binary, key_json, scale_overrides_json, [layer_idx],
    )
    diags = json.loads(diag_json)
    if not diags:
        print("  CAST-ORDER: no diagnostic data for layer")
        return

    d = diags[0]
    print(f"  K acc len={len(d['k_acc'])}  scales len={len(d['k_scales'])}  "
          f"scale_x={d['scale_x']:.8f}  has_bias={len(d['k_bias']) > 0}")

    # Compare each variant against GPU K.
    print(f"\n  {'Strategy':<25s}  {'max_diff':>10s}  {'mean_diff':>10s}  "
          f"{'worst_idx':>10s}  {'gpu_val':>12s}  {'ver_val':>12s}")
    print("  " + "-" * 85)

    best_name = None
    best_max = float("inf")
    for name, vals in d["variants"]:
        ver_k = np.array(vals, dtype=np.float64)
        diff = np.abs(k_gpu_gen - ver_k)
        max_d = diff.max()
        mean_d = diff.mean()
        worst_idx = int(np.argmax(diff))
        print(f"  {name:<25s}  {max_d:10.6f}  {mean_d:10.6f}  "
              f"{worst_idx:10d}  {k_gpu_gen[worst_idx]:12.6f}  {ver_k[worst_idx]:12.6f}")
        if max_d < best_max:
            best_max = max_d
            best_name = name

    print(f"\n  → Best match: {best_name} (max_diff={best_max:.6f})")
    if best_max < 0.01:
        print(f"  → MATCH FOUND: CUTLASS uses '{best_name}' cast order")
    elif best_max < 0.1:
        print(f"  → CLOSE: '{best_name}' is close but not exact (may need finer granularity)")
    else:
        print(f"  → NO MATCH: best strategy still has max_diff={best_max:.6f}")
        print(f"  → The dequant path may differ in ways not covered by these 5 strategies")


def _run_model(model_id: str, max_model_len: int, workloads: list):
    """Run A/B corridor measurement for one model at one context length."""
    import hashlib
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VERILM_SCORE_WITNESS"] = "1"

    import verilm_rs
    import torch
    import vllm
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    buf = get_capture_buffer()

    print(f"\n{'='*70}")
    print(f"Model: {model_id}  max_model_len={max_model_len}")
    print(f"{'='*70}")

    llm = LLM(
        model=model_id, dtype="auto", max_model_len=max_model_len,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    print(
        f"Runtime: vllm={vllm.__version__} torch={torch.__version__} "
        f"backend={server._attn_backend} dtype={server._attn_dtype}"
    )
    print(f"Score witness: enabled={buf._sw_enabled}")

    # Warmup.
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=8)

    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, key_seed)
    full_layers = list(range(n_layers))

    cfg = json.loads(key_json)["config"]
    cfg_dict = {
        "hidden_dim": cfg["hidden_dim"],
        "kv_dim": cfg["n_kv_heads"] * cfg["d_head"],
    }
    weight_scales = _extract_weight_scales(llm, n_layers, cfg_dict)
    scale_overrides_json = json.dumps(weight_scales)

    all_results = []
    print(f"\n--- A/B corridor: standard replay vs witnessed scores ---")
    for wl in workloads:
        result = _measure_one_ab(
            server, wl, full_layers, key_json, model_id, buf, scale_overrides_json,
        )
        if result:
            all_results.append(result)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_id} (ctx={max_model_len})")
    print(f"{'='*70}")
    if all_results:
        max_std = max(r["standard"]["global_linf"] for r in all_results)
        max_ws = max(r["witnessed"]["global_linf"] for r in all_results)
        avg_std_ms = sum(r["standard"]["verifier_ms"] for r in all_results) / len(all_results)
        avg_ws_ms = sum(r["witnessed"]["verifier_ms"] for r in all_results) / len(all_results)
        avg_audit_mb = sum(r["audit_size_mb"] for r in all_results) / len(all_results)
        avg_score_mb = sum(r["score_payload_mb"] for r in all_results) / len(all_results)
        max_anchor_f64 = max(r["anchoring_f64"]["global_max_gap"] for r in all_results)
        max_anchor_gpu = max(r["anchoring_gpu_like"]["global_max_gap"] for r in all_results)
        max_anchor_bf16 = max(r["anchoring_bf16"]["global_max_gap"] for r in all_results)
        print(f"  Standard replay:  max L-inf = {max_std}")
        print(f"  Witnessed scores: max L-inf = {max_ws}")
        print(f"  Improvement: L-inf {max_std} → {max_ws}")
        print(f"  Avg verifier time: std={avg_std_ms:.1f}ms  ws={avg_ws_ms:.1f}ms")
        print(f"  Avg audit payload: {avg_audit_mb:.2f} MB (scores: {avg_score_mb:.2f} MB)")
        print(f"  Anchoring f64:      max_gap={max_anchor_f64:.6f}")
        print(f"  Anchoring GPU-like: max_gap={max_anchor_gpu:.6f}")
        print(f"  Anchoring bf16:     max_gap={max_anchor_bf16:.6f}")

    # ── Diagnostic: isolate Q vs K mismatch on worst layer ──
    if max_anchor_gpu > 0.5:
        print(f"\n--- Score anchoring diagnostic (gap > 0.5) ---")
        try:
            _diagnose_anchor_mismatch(
                server, buf, full_layers, key_json, model_id,
                scale_overrides_json, cfg,
            )
        except Exception as e:
            print(f"  DIAG ERROR: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n--- Fused QKV diagnostic ---")
        try:
            _diagnose_fused_qkv(
                llm, server, buf, full_layers, key_json,
                scale_overrides_json, cfg,
            )
        except Exception as e:
            print(f"  FUSED-QKV ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Cast-order diagnostic: try different dequant+bias orderings.
        try:
            _diagnose_dequant_cast_order(
                server, buf, llm, full_layers, key_json,
                scale_overrides_json, cfg,
            )
        except Exception as e:
            print(f"  CAST-ORDER ERROR: {e}")
            import traceback
            traceback.print_exc()

    return all_results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/tmp/corridor": modal.Volume.from_name("corridor-results", create_if_missing=True)},
)
def measure_qwen_4k():
    import json
    results = _run_model(
        "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
        max_model_len=4096,
        workloads=WORKLOADS_4K,
    )
    with open("/tmp/corridor/qwen_witnessed_4k.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    memory=32768,
    volumes={"/tmp/corridor": modal.Volume.from_name("corridor-results", create_if_missing=True)},
)
def measure_qwen_32k():
    import json
    results = _run_model(
        "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
        max_model_len=32768,
        workloads=WORKLOADS_32K,
    )
    with open("/tmp/corridor/qwen_witnessed_32k.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/tmp/corridor": modal.Volume.from_name("corridor-results", create_if_missing=True)},
)
def measure_llama_4k():
    import json
    results = _run_model(
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        max_model_len=4096,
        workloads=WORKLOADS_4K,
    )
    with open("/tmp/corridor/llama_witnessed_4k.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    memory=32768,
    volumes={"/tmp/corridor": modal.Volume.from_name("corridor-results", create_if_missing=True)},
)
def measure_llama_32k():
    import json
    results = _run_model(
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        max_model_len=32768,
        workloads=WORKLOADS_32K,
    )
    with open("/tmp/corridor/llama_witnessed_32k.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.local_entrypoint()
def main():
    import json
    import os

    models_env = os.environ.get("CORRIDOR_MODELS", "qwen,llama")
    ctx_env = os.environ.get("CORRIDOR_CONTEXTS", "4k")
    run_qwen = "qwen" in models_env
    run_llama = "llama" in models_env
    run_4k = "4k" in ctx_env
    run_32k = "32k" in ctx_env

    print("Launching witnessed-score corridor A/B measurement...")
    print(f"  Models: qwen={run_qwen}, llama={run_llama}")
    print(f"  Contexts: 4k={run_4k}, 32k={run_32k}")
    print("Results saved to Modal volume 'corridor-results'.\n")

    futures = []
    labels = []

    if run_qwen and run_4k:
        futures.append(measure_qwen_4k.spawn())
        labels.append("Qwen 7B @ 4K")
    if run_qwen and run_32k:
        futures.append(measure_qwen_32k.spawn())
        labels.append("Qwen 7B @ 32K")
    if run_llama and run_4k:
        futures.append(measure_llama_4k.spawn())
        labels.append("Llama 8B @ 4K")
    if run_llama and run_32k:
        futures.append(measure_llama_32k.spawn())
        labels.append("Llama 8B @ 32K")

    all_results = []
    for label, future in zip(labels, futures):
        results = future.get()
        all_results.extend(results)
        print(f"\n{'='*70}")
        print(f"{label}: completed ({len(results)} measurements)")

    # Combined summary.
    print(f"\n{'='*70}")
    print("COMBINED A/B SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':>50s}  {'Workload':>12s}  {'Ctx':>4s}  {'Std L∞':>7s}  {'WS L∞':>7s}  {'Score MB':>9s}  {'Anch f64':>10s}  {'Anch GPU':>10s}  {'Anch bf16':>10s}")
    print("-" * 145)
    for r in all_results:
        ctx = "32k" if r["n_tokens"] > 2048 else "4k"
        print(
            f"{r['model']:>50s}  {r['workload']:>12s}  {ctx:>4s}  "
            f"{r['standard']['global_linf']:>7d}  {r['witnessed']['global_linf']:>7d}  "
            f"{r['score_payload_mb']:>9.2f}  "
            f"{r['anchoring_f64']['global_max_gap']:>10.4f}  "
            f"{r['anchoring_gpu_like']['global_max_gap']:>10.4f}  "
            f"{r['anchoring_bf16']['global_max_gap']:>10.4f}"
        )
