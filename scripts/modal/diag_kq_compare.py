"""
Direct K/Q comparison diagnostic: GPU actual vs committed/reconstructed.

For one failing token/layer:
1. Capture GPU's actual K (after RoPE, what enters KV cache) and Q (after RoPE)
2. Get committed k_roped from audit KV entries
3. Get verifier-reconstructed q_roped
4. Compare element-wise
5. Test position +1 offset hypothesis

Usage:
    modal run --detach scripts/modal/diag_kq_compare.py
"""

import modal

app = modal.App("verilm-kq-compare")

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
    })
    .pip_install("vllm>=0.8", "torch", "numpy", "fastapi", "maturin")
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
)
def compare_kq():
    import hashlib
    import json
    import os

    import numpy as np
    import torch

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    model_id = os.environ.get(
        "DIAG_MODEL_ID",
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
    )
    buf = get_capture_buffer()

    print(f"\n{'='*70}")
    print(f"K/Q Comparison Diagnostic")
    print(f"Model: {model_id}")
    print(f"{'='*70}")

    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers

    # Read model dimensions from verifier key (works for any model family)
    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(server._model_dir, key_seed)
    key_cfg = json.loads(key_json)["config"]
    d_head = key_cfg["d_head"]
    n_q_heads = key_cfg["n_q_heads"]
    n_kv_heads = key_cfg["n_kv_heads"]
    hidden_dim = key_cfg["hidden_dim"]
    kv_dim = n_kv_heads * d_head

    # ── RoPE helpers (used for both prefill Q comparison and decode diagnostics) ──
    def compute_scaled_inv_freq(theta, d_head_local, rope_scaling_cfg):
        """Compute (possibly scaled) inverse frequencies."""
        half = d_head_local // 2
        base_inv_freq = np.array([
            1.0 / (theta ** (2.0 * k / d_head_local))
            for k in range(half)
        ])
        if rope_scaling_cfg is None:
            return base_inv_freq

        rope_type = rope_scaling_cfg.get("rope_type", "")
        if rope_type == "llama3":
            factor = rope_scaling_cfg["factor"]
            low_freq_factor = rope_scaling_cfg.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling_cfg.get("high_freq_factor", 4.0)
            old_ctx = rope_scaling_cfg["original_max_position_embeddings"]

            low_freq_wavelen = old_ctx / low_freq_factor
            high_freq_wavelen = old_ctx / high_freq_factor

            scaled = np.zeros_like(base_inv_freq)
            for k in range(half):
                wavelen = 2.0 * np.pi / base_inv_freq[k]
                if wavelen < high_freq_wavelen:
                    scaled[k] = base_inv_freq[k]
                elif wavelen > low_freq_wavelen:
                    scaled[k] = base_inv_freq[k] / factor
                else:
                    smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    scaled[k] = base_inv_freq[k] * ((1.0 - smooth) / factor + smooth)
            return scaled
        elif rope_type == "linear":
            return base_inv_freq / rope_scaling_cfg["factor"]
        else:
            return base_inv_freq

    def apply_rope_half(vec, position, inv_freq_arr, d_head_local, n_heads):
        out = np.zeros_like(vec)
        half = d_head_local // 2
        for h in range(n_heads):
            start = h * d_head_local
            for i in range(half):
                angle = position * inv_freq_arr[i]
                cos_f = np.cos(angle)
                sin_f = np.sin(angle)
                out[start + i] = vec[start + i] * cos_f - vec[start + half + i] * sin_f
                out[start + half + i] = vec[start + half + i] * cos_f + vec[start + i] * sin_f
        return out

    rope_theta = key_cfg.get("rope_theta", 10000.0)
    rope_scaling_cfg = key_cfg.get("rope_scaling")
    inv_freq = compute_scaled_inv_freq(rope_theta, d_head, rope_scaling_cfg)
    print(f"  rope_theta from key: {rope_theta}")
    print(f"  rope_scaling from key: {rope_scaling_cfg}")

    # Warmup
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=8)

    # ── Hook rotary_emb to capture Q and K after RoPE ──
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    TARGET_LAYER = 0
    layer0 = model.model.layers[TARGET_LAYER]
    attn = layer0.self_attn

    captured_qk = {}

    # Hook the attention forward to capture Q, K, V and positions.
    # In vLLM Qwen2, the attention forward does:
    #   qkv, _ = self.qkv_proj(hidden_states)
    #   q, k, v = qkv.split(...)
    #   q, k = self.rotary_emb(positions, q, k)
    #   attn_output = self.attn(q, k, v, ...)
    # We wrap rotary_emb to capture its output.
    orig_rotary = attn.rotary_emb

    class RotaryCapture(torch.nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.orig = orig
            self.enabled = False

        def forward(self, positions, q, k, *args, **kwargs):
            q_out, k_out = self.orig(positions, q, k, *args, **kwargs)
            if self.enabled:
                captured_qk["positions"] = positions.detach().cpu().clone()
                captured_qk["q_roped"] = q_out.detach().cpu().float().clone()
                captured_qk["k_roped"] = k_out.detach().cpu().float().clone()
                captured_qk["q_pre"] = q.detach().cpu().float().clone()
                captured_qk["k_pre"] = k.detach().cpu().float().clone()
            return q_out, k_out

    rotary_wrapper = RotaryCapture(orig_rotary)
    attn.rotary_emb = rotary_wrapper

    # Also capture pre-RoPE K and V by hooking qkv_proj output
    # Not needed — we get pre-RoPE Q/K from the wrapper above

    # ── Workload selection via env ──
    WORKLOAD = os.environ.get("DIAG_WORKLOAD", "long_context")
    WORKLOADS = {
        "short": {
            "prompt": "What is 2+2?",
            "max_tokens": 32,
        },
        "long_context": {
            "prompt": (
                "You are a senior software architect writing a comprehensive design "
                "document. Cover the following topics in depth with concrete examples: "
                "(1) microservices vs monolithic architecture trade-offs, including "
                "failure modes, deployment complexity, and data consistency; "
                "(2) event-driven systems with Kafka or similar message brokers, "
                "covering exactly-once semantics, consumer group rebalancing, and "
                "schema evolution; (3) database sharding strategies including "
                "consistent hashing, range partitioning, and cross-shard transactions; "
                "(4) observability infrastructure including structured logging, "
                "distributed tracing with OpenTelemetry, and SLO-based alerting; "
                "(5) CI/CD pipeline design for zero-downtime deployments with "
                "canary releases and automated rollbacks."
            ),
            "max_tokens": 1024,
        },
    }
    wl = WORKLOADS[WORKLOAD]
    print(f"\n--- Workload: {WORKLOAD} (max_tokens={wl['max_tokens']}) ---")

    # ── Run inference with capture ──
    print(f"\n--- Running inference ---")
    rotary_wrapper.enabled = True
    chat_r = server.chat(prompt=wl["prompt"], max_tokens=wl["max_tokens"], temperature=0.0)
    rotary_wrapper.enabled = False

    n_tokens = chat_r["n_tokens"]
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]
    request_id = chat_r["request_id"]

    print(f"  n_tokens={n_tokens}, n_prompt_tokens={n_prompt}")
    print(f"  Captured rotary positions: {captured_qk.get('positions', 'NONE')}")

    if "positions" not in captured_qk:
        print("  ERROR: rotary_emb hook did not fire!")
        return

    # The last forward pass is a single-token decode. captured_qk has
    # Q/K from the LAST forward pass (overwritten each time).
    # For prefill comparison, we need to run a SEPARATE inference and capture
    # only the prefill forward pass.

    # Strategy: run inference again but capture only the FIRST forward (prefill).
    captured_prefill = {}
    call_counter = [0]

    class RotaryCapturePrefill(torch.nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.orig = orig
            self.enabled = False

        def forward(self, positions, q, k, *args, **kwargs):
            q_out, k_out = self.orig(positions, q, k, *args, **kwargs)
            if self.enabled and call_counter[0] == 0:
                captured_prefill["positions"] = positions.detach().cpu().clone()
                captured_prefill["q_roped"] = q_out.detach().cpu().float().clone()
                captured_prefill["k_roped"] = k_out.detach().cpu().float().clone()
                call_counter[0] += 1
            elif self.enabled:
                call_counter[0] += 1
            return q_out, k_out

    rotary_prefill = RotaryCapturePrefill(orig_rotary)
    attn.rotary_emb = rotary_prefill

    print(f"\n--- Running second inference (prefill capture) ---")
    rotary_prefill.enabled = True
    call_counter[0] = 0
    chat_r2 = server.chat(prompt=wl["prompt"], max_tokens=wl["max_tokens"], temperature=0.0)
    rotary_prefill.enabled = False
    request_id2 = chat_r2["request_id"]
    n_tokens2 = chat_r2["n_tokens"]
    n_prompt2 = chat_r2["commitment"]["n_prompt_tokens"]

    # Restore original
    attn.rotary_emb = orig_rotary

    print(f"  n_tokens={n_tokens2}, n_prompt={n_prompt2}")
    print(f"  Prefill positions: {captured_prefill.get('positions', 'NONE')}")

    if "positions" not in captured_prefill:
        print("  ERROR: prefill capture failed!")
        return

    gpu_positions = captured_prefill["positions"].numpy()
    gpu_q = captured_prefill["q_roped"].numpy()  # shape: (n_prompt_tokens, n_q_heads, d_head) or (total_tokens, ...)
    gpu_k = captured_prefill["k_roped"].numpy()

    print(f"\n{'='*70}")
    print(f"GPU PREFILL CAPTURE (layer {TARGET_LAYER})")
    print(f"{'='*70}")
    print(f"  positions shape: {gpu_positions.shape}, values: {gpu_positions.flatten().tolist()}")
    print(f"  q_roped shape: {gpu_q.shape}")
    print(f"  k_roped shape: {gpu_k.shape}")

    # Reshape Q and K to flat layout matching the verifier's convention
    # vLLM might have shape (n_tokens, n_heads * d_head) or (n_tokens, n_heads, d_head)
    n_prefill = gpu_q.shape[0]
    if gpu_q.ndim == 3:
        gpu_q_flat = gpu_q.reshape(n_prefill, -1)
        gpu_k_flat = gpu_k.reshape(n_prefill, -1)
    else:
        gpu_q_flat = gpu_q
        gpu_k_flat = gpu_k

    print(f"  q_flat shape: {gpu_q_flat.shape} (expect n_tokens × {n_q_heads * d_head})")
    print(f"  k_flat shape: {gpu_k_flat.shape} (expect n_tokens × {n_kv_heads * d_head})")

    # ── Get committed k_roped from audit ──
    print(f"\n{'='*70}")
    print(f"COMMITTED KV ENTRIES")
    print(f"{'='*70}")

    full_layers = list(range(n_layers))

    # Audit at pos=6 (n_prompt-1) to get committed KV + shell opening
    audit_pos = n_prompt2 - 1
    audit_binary = server.audit(
        request_id=request_id2,
        token_index=audit_pos,
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )

    # Parse the audit to get committed k_roped (JSON mode for introspection)
    audit_json_str = server.audit(
        request_id=request_id2,
        token_index=audit_pos,
        layer_indices=full_layers,
        tier="full",
        binary=False,
        use_captured_x_attn=True,
    )
    audit_json = json.loads(audit_json_str)

    kv_entries = audit_json.get("kv_entries", [])
    if kv_entries and len(kv_entries) > TARGET_LAYER:
        layer_kv = kv_entries[TARGET_LAYER]
        print(f"  KV entries for layer {TARGET_LAYER}: {len(layer_kv)} entries")
        for i, entry in enumerate(layer_kv[:8]):
            k_first8 = [f"{v:.4f}" for v in entry["k_roped"][:8]]
            print(f"    entry[{i}] k_roped[:8] = {k_first8}")
    else:
        print("  ERROR: no kv_entries in audit!")
        return

    # ── Compare GPU K vs committed K for each position ──
    print(f"\n{'='*70}")
    print(f"K COMPARISON: GPU vs Committed (layer {TARGET_LAYER})")
    print(f"{'='*70}")

    for pos_idx in range(min(n_prefill, len(layer_kv))):
        committed_k = np.array(layer_kv[pos_idx]["k_roped"])
        gpu_k_pos = gpu_k_flat[pos_idx, :kv_dim]

        if committed_k.shape[0] != gpu_k_pos.shape[0]:
            print(f"  pos={pos_idx}: shape mismatch committed={committed_k.shape} gpu={gpu_k_pos.shape}")
            continue

        diff = np.abs(committed_k - gpu_k_pos.astype(np.float64))
        linf = diff.max()
        mean_diff = diff.mean()
        # Also check +1 offset: committed_k[pos] vs gpu_k[pos+1]
        if pos_idx + 1 < n_prefill:
            gpu_k_plus1 = gpu_k_flat[pos_idx + 1, :kv_dim]
            diff_plus1 = np.abs(committed_k - gpu_k_plus1.astype(np.float64))
            linf_plus1 = diff_plus1.max()
        else:
            linf_plus1 = -1

        # And check -1 offset
        if pos_idx > 0:
            gpu_k_minus1 = gpu_k_flat[pos_idx - 1, :kv_dim]
            diff_minus1 = np.abs(committed_k - gpu_k_minus1.astype(np.float64))
            linf_minus1 = diff_minus1.max()
        else:
            linf_minus1 = -1

        gpu_pos = gpu_positions.flatten()[pos_idx] if pos_idx < len(gpu_positions.flatten()) else "?"
        print(
            f"  committed[{pos_idx}] vs gpu_k[{pos_idx}] (gpu_rope_pos={gpu_pos}): "
            f"L-inf={linf:.4f} mean={mean_diff:.6f}"
            f"  |vs gpu[{pos_idx+1}]: {linf_plus1:.4f}"
            f"  |vs gpu[{pos_idx-1}]: {linf_minus1:.4f}" if pos_idx > 0 else ""
        )

    # ── Detailed comparison for position n_prompt-1 ──
    DIAG_POS = audit_pos
    print(f"\n{'='*70}")
    print(f"DETAILED K COMPARISON at pos={DIAG_POS} (layer {TARGET_LAYER})")
    print(f"{'='*70}")

    committed_k = np.array(layer_kv[DIAG_POS]["k_roped"])
    if DIAG_POS < n_prefill:
        gpu_k_diag = gpu_k_flat[DIAG_POS, :kv_dim].astype(np.float64)
        n_show = 16
        print(f"  committed_k[:16] = {[f'{v:.6f}' for v in committed_k[:n_show]]}")
        print(f"  gpu_k[{DIAG_POS}][:16] = {[f'{v:.6f}' for v in gpu_k_diag[:n_show]]}")
        diff = np.abs(committed_k - gpu_k_diag)
        print(f"  |diff|[:16]       = {[f'{v:.6f}' for v in diff[:n_show]]}")
        print(f"  L-inf={diff.max():.6f} mean={diff.mean():.6f}")

        # Test +1 offset
        if DIAG_POS + 1 < n_prefill:
            gpu_k_p1 = gpu_k_flat[DIAG_POS + 1, :kv_dim].astype(np.float64)
            diff_p1 = np.abs(committed_k - gpu_k_p1)
            print(f"\n  With +1 offset: committed[{DIAG_POS}] vs gpu_k[{DIAG_POS+1}]:")
            print(f"  gpu_k[{DIAG_POS+1}][:16] = {[f'{v:.6f}' for v in gpu_k_p1[:n_show]]}")
            print(f"  |diff|[:16]       = {[f'{v:.6f}' for v in diff_p1[:n_show]]}")
            print(f"  L-inf={diff_p1.max():.6f} mean={diff_p1.mean():.6f}")

    # ── Q comparison: compute committed Q from shell opening ──
    print(f"\n{'='*70}")
    print(f"Q COMPARISON at pos={DIAG_POS} (layer {TARGET_LAYER})")
    print(f"{'='*70}")

    shell = audit_json.get("shell_opening", {})
    shell_layers = shell.get("layers", [])
    committed_q_roped = None

    if TARGET_LAYER < len(shell_layers):
        sl = shell_layers[TARGET_LAYER]
        q_acc_raw = sl.get("q")
        scale_x_attn = sl.get("scale_x_attn")
        if q_acc_raw and scale_x_attn is not None:
            q_acc = np.array(q_acc_raw, dtype=np.int64)
            print(f"  Shell Q accumulator: {len(q_acc)} elements, scale_x_attn={scale_x_attn}")

            # Extract per-channel Wq scales from the model
            model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
            ws = model_obj.model.layers[TARGET_LAYER].self_attn.qkv_proj.weight_scale
            ws_np = ws.detach().cpu().float().numpy().flatten()
            wq_scale = ws_np[:hidden_dim].astype(np.float64)

            # Dequant: q_f64 = q_acc * wq_scale * scale_x
            q_f64 = q_acc.astype(np.float64) * wq_scale * float(scale_x_attn)

            # Add Q bias
            bias_tensor = model_obj.model.layers[TARGET_LAYER].self_attn.qkv_proj.bias
            if bias_tensor is not None:
                bias_np = bias_tensor.detach().cpu().float().numpy().flatten()
                q_bias = bias_np[:hidden_dim].astype(np.float64)
                q_f64 = q_f64 + q_bias

            # Apply RoPE using outer-scope helpers
            committed_q_roped = apply_rope_half(q_f64, DIAG_POS, inv_freq, d_head, n_q_heads)

            if DIAG_POS < n_prefill:
                gpu_q_diag = gpu_q_flat[DIAG_POS, :hidden_dim].astype(np.float64)
                q_diff = np.abs(committed_q_roped - gpu_q_diag)
                print(f"  committed_q[:16] = {[f'{v:.6f}' for v in committed_q_roped[:16]]}")
                print(f"  gpu_q[{DIAG_POS}][:16]   = {[f'{v:.6f}' for v in gpu_q_diag[:16]]}")
                print(f"  |diff|[:16]       = {[f'{v:.6f}' for v in q_diff[:16]]}")
                print(f"  Q L-inf={q_diff.max():.6f} mean={q_diff.mean():.6f}")

                # Per-head Q L-inf
                q_linfs_per_head = []
                for h in range(n_q_heads):
                    h_diff = q_diff[h*d_head:(h+1)*d_head]
                    q_linfs_per_head.append(h_diff.max())
                print(f"  Per-head Q L-inf: {[f'{v:.4f}' for v in q_linfs_per_head]}")
        else:
            print("  No Q accumulator or scale_x_attn in shell opening")
    else:
        print("  No shell layer data")

    # ── Score vector comparison ──
    print(f"\n{'='*70}")
    print(f"SCORE VECTORS Q·K/sqrt(d) at pos={DIAG_POS}, head 0 (layer {TARGET_LAYER})")
    print(f"{'='*70}")

    if DIAG_POS < n_prefill:
        inv_sqrt_d = 1.0 / np.sqrt(d_head)

        # GPU scores using GPU Q and K
        gpu_q_diag = gpu_q_flat[DIAG_POS, :hidden_dim].astype(np.float64)
        q_head0_gpu = gpu_q_diag[:d_head]
        gpu_scores = []
        for t in range(DIAG_POS + 1):
            k_t = gpu_k_flat[t, :d_head].astype(np.float64)
            score = np.dot(q_head0_gpu, k_t) * inv_sqrt_d
            gpu_scores.append(score)

        # Committed scores: committed_q · committed_k (pure verifier path)
        if committed_q_roped is not None:
            q_head0_committed = committed_q_roped[:d_head]
            committed_scores = []
            for t in range(min(DIAG_POS + 1, len(layer_kv))):
                k_t = np.array(layer_kv[t]["k_roped"][:d_head])
                score = np.dot(q_head0_committed, k_t) * inv_sqrt_d
                committed_scores.append(score)

            print(f"  GPU scores (gpu_q · gpu_k):           {[f'{s:.4f}' for s in gpu_scores]}")
            print(f"  Committed scores (comm_q · comm_k):   {[f'{s:.4f}' for s in committed_scores]}")
            score_diff = [abs(a - b) for a, b in zip(gpu_scores, committed_scores)]
            print(f"  Score diff:                            {[f'{d:.4f}' for d in score_diff]}")
            print(f"  Max score diff: {max(score_diff):.6f}")

            # Show softmax weights for both
            def softmax(scores):
                s = np.array(scores)
                s = s - s.max()
                e = np.exp(s)
                return e / e.sum()

            gpu_weights = softmax(gpu_scores)
            committed_weights = softmax(committed_scores)
            print(f"  GPU softmax:      {[f'{w:.6f}' for w in gpu_weights]}")
            print(f"  Committed softmax: {[f'{w:.6f}' for w in committed_weights]}")
            weight_diff = np.abs(gpu_weights - committed_weights)
            print(f"  Weight diff:       {[f'{d:.6f}' for d in weight_diff]}")
            print(f"  Max weight diff: {weight_diff.max():.6f}")

    # ── Final summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  GPU RoPE positions during prefill: {gpu_positions.flatten().tolist()}")
    print(f"  Prover KV entry count: {len(layer_kv)}")
    print(f"  n_prompt_tokens (from commitment): {n_prompt2}")
    print(f"  Audit token_index: {audit_pos}")

    # Overall K match quality
    k_linfs = []
    for i in range(min(n_prefill, len(layer_kv))):
        ck = np.array(layer_kv[i]["k_roped"])
        gk = gpu_k_flat[i, :kv_dim].astype(np.float64)
        k_linfs.append(np.abs(ck - gk).max())
    print(f"  Per-position K L-inf: {[f'{v:.4f}' for v in k_linfs]}")

    if any(v > 1.0 for v in k_linfs):
        print("  → K MISMATCH DETECTED — committed K does NOT match GPU K")
        # Check if +1 offset fixes it
        k_linfs_p1 = []
        for i in range(min(n_prefill - 1, len(layer_kv))):
            ck = np.array(layer_kv[i]["k_roped"])
            gk = gpu_k_flat[i + 1, :kv_dim].astype(np.float64)
            k_linfs_p1.append(np.abs(ck - gk).max())
        print(f"  With +1 offset: {[f'{v:.4f}' for v in k_linfs_p1]}")
        if k_linfs_p1 and max(k_linfs_p1) < 1.0:
            print("  → +1 OFFSET FIXES K MATCH — RoPE position bug confirmed!")
        elif k_linfs_p1 and max(k_linfs_p1) < max(k_linfs) * 0.1:
            print("  → +1 offset significantly improves match")
        else:
            print("  → +1 offset does NOT fix — problem is elsewhere")
    else:
        print("  → K match is good — no RoPE offset detected")

    # ── Multi-position decode diagnostics (growth vs context length) ──
    if n_tokens2 > n_prompt2 + 4:
        print(f"\n{'='*70}")
        print(f"DECODE POSITION K/Q DIAGNOSTICS (growth vs context length)")
        print(f"{'='*70}")

        # Pick ~5 decode positions spread across the generation
        n_decode = n_tokens2 - n_prompt2
        if n_decode <= 6:
            decode_positions = list(range(n_prompt2, n_tokens2))
        else:
            step = max(1, n_decode // 5)
            decode_positions = list(range(n_prompt2, n_tokens2, step))
            if n_tokens2 - 1 not in decode_positions:
                decode_positions.append(n_tokens2 - 1)

        print(f"  Total tokens: {n_tokens2}, prompt: {n_prompt2}, decode: {n_decode}")
        print(f"  Sampling decode positions: {decode_positions}")
        print(f"  {'pos':>6s}  {'K L-inf':>8s}  {'K mean':>8s}  {'K frac_eq':>10s}  {'K frac<=1':>10s}")

        for dec_pos in decode_positions:
            try:
                dec_audit_json_str = server.audit(
                    request_id=request_id2,
                    token_index=dec_pos,
                    layer_indices=[TARGET_LAYER],
                    tier="full",
                    binary=False,
                    use_captured_x_attn=True,
                )
                dec_audit = json.loads(dec_audit_json_str)
                dec_kv = dec_audit.get("kv_entries", [])
                if not dec_kv or len(dec_kv) == 0:
                    print(f"  {dec_pos:6d}  (no kv_entries)")
                    continue

                # Compare committed K at the opened position vs what the verifier would produce
                # For decode tokens the committed K IS the GPU K (captured during decode)
                # What we want: the committed K at position dec_pos should be self-consistent
                # with the RoPE scaling applied at that position
                dec_layer_kv = dec_kv[0]  # layer TARGET_LAYER
                n_kv_entries = len(dec_layer_kv)

                # The last KV entry is the opened token's K
                opened_k = np.array(dec_layer_kv[-1]["k_roped"])

                # Reconstruct verifier K from shell opening Q accumulator
                dec_shell = dec_audit.get("shell_opening", {})
                dec_shell_layers = dec_shell.get("layers", [])

                if len(dec_shell_layers) > 0:
                    dsl = dec_shell_layers[0]
                    k_acc_raw = dsl.get("k")
                    if k_acc_raw is not None:
                        k_acc = np.array(k_acc_raw, dtype=np.int64)
                        # Get K weight scales
                        model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
                        ws = model_obj.model.layers[TARGET_LAYER].self_attn.qkv_proj.weight_scale
                        ws_np = ws.detach().cpu().float().numpy().flatten()
                        wk_scale = ws_np[hidden_dim:hidden_dim + kv_dim].astype(np.float64)

                        scale_x_attn_dec = dsl.get("scale_x_attn", 1.0)
                        k_f64 = k_acc.astype(np.float64) * wk_scale * float(scale_x_attn_dec)

                        # Apply RoPE at position dec_pos
                        k_roped_verifier = apply_rope_half(k_f64, dec_pos, inv_freq, d_head, n_kv_heads)

                        diff = np.abs(opened_k - k_roped_verifier)
                        k_linf = diff.max()
                        k_mean = diff.mean()
                        n_el = len(diff)
                        frac_eq = np.sum(diff == 0.0) / n_el
                        frac_le1 = np.sum(diff <= 1.0) / n_el
                        print(f"  {dec_pos:6d}  {k_linf:8.4f}  {k_mean:8.6f}  {frac_eq:10.4f}  {frac_le1:10.4f}")
                    else:
                        print(f"  {dec_pos:6d}  (no k accumulator in shell)")
                else:
                    print(f"  {dec_pos:6d}  (no shell layers)")
            except Exception as e:
                print(f"  {dec_pos:6d}  ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"rope_scaling: {rope_scaling_cfg}")
    print(f"{'='*70}")


@app.local_entrypoint()
def main():
    compare_kq.remote()
    print("\nDone.")
