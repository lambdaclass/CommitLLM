"""
Qwen K divergence diagnostic — walks through the decision tree:

1. Does cutlass_epilogue_bf16(shell_k_acc) == GPU pre-RoPE K?
   → If yes: accumulators and epilogue are correct
   → If no: shell k_acc or scales differ from GPU

2. Does committed K (from KV transcript) match GPU post-RoPE K?
   → If mismatch: the f64 dequant path in compute_kv_transcript diverges

3. Does opened token match but historical tokens diverge?
   → If so: prefix KV commit semantics bug

Usage:
    modal run --detach scripts/modal/diag_qwen_k_divergence.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-qwen-k-diag")

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
        "VERILM_CAPTURE_X_ATTN": "1",
    })
    .pip_install(*VERIFICATION)
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

vol = modal.Volume.from_name("verilm-diag-output", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/output": vol},
)
def diagnose():
    import glob
    import hashlib
    import json
    import os
    import sys
    import traceback

    # Tee stdout to a file in the volume
    class Tee:
        def __init__(self, orig, log):
            self._orig = orig
            self._log = log
        def write(self, data):
            self._orig.write(data)
            self._orig.flush()
            self._log.write(data)
            self._log.flush()
        def flush(self):
            self._orig.flush()
            self._log.flush()
        def fileno(self):
            return self._orig.fileno()
        def isatty(self):
            return self._orig.isatty()

    log_file = open("/output/diag_qwen_k.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    try:
        _diagnose_inner()
    except Exception:
        traceback.print_exc()
    finally:
        log_file.close()
        vol.commit()


def _diagnose_inner():
    import glob
    import hashlib
    import json
    import os

    import numpy as np
    import torch

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from safetensors import safe_open
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    model_id = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
    TARGET_LAYER = 0

    print(f"\n{'='*70}")
    print(f"QWEN K DIVERGENCE DIAGNOSTIC")
    print(f"Model: {model_id}")
    print(f"{'='*70}")

    # ── Load model ──
    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    buf = get_capture_buffer()
    n_layers = cap._n_layers

    # ── Key/config ──
    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(server._model_dir, key_seed)
    key_cfg = json.loads(key_json)["config"]
    d_head = key_cfg["d_head"]
    n_q_heads = key_cfg["n_q_heads"]
    n_kv_heads = key_cfg["n_kv_heads"]
    hidden_dim = key_cfg["hidden_dim"]
    kv_dim = n_kv_heads * d_head
    q_dim = n_q_heads * d_head

    print(f"  hidden_dim={hidden_dim}, q_dim={q_dim}, kv_dim={kv_dim}")
    print(f"  n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}, d_head={d_head}")
    print(f"  n_layers={n_layers}")

    # ── Extract model weights and scales ──
    model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer0_attn = model_obj.model.layers[TARGET_LAYER].self_attn

    # Fused QKV weight: vLLM stores as [hidden_dim, q_dim + 2*kv_dim] INT8
    # (i.e., _scaled_mm(x, weight) computes x @ weight)
    qkv_weight = layer0_attn.qkv_proj.weight.detach().cpu()
    print(f"  qkv_weight shape: {qkv_weight.shape}, dtype={qkv_weight.dtype}")
    # Determine layout: if shape[0] == hidden_dim, columns are [Q|K|V]
    # If shape[0] == q_dim + 2*kv_dim, rows are [Q|K|V]
    total_out = q_dim + 2 * kv_dim
    if qkv_weight.shape[0] == hidden_dim and qkv_weight.shape[1] == total_out:
        # [hidden_dim, q+2k] — K is in columns
        wk_gpu = qkv_weight[:, q_dim: q_dim + kv_dim].contiguous().t()  # → [kv_dim, hidden_dim]
        weight_layout = "col"
    elif qkv_weight.shape[0] == total_out and qkv_weight.shape[1] == hidden_dim:
        # [q+2k, hidden_dim] — K is in rows
        wk_gpu = qkv_weight[q_dim: q_dim + kv_dim, :].contiguous()
        weight_layout = "row"
    else:
        print(f"  UNEXPECTED weight shape: {qkv_weight.shape} (expected [{hidden_dim},{total_out}] or [{total_out},{hidden_dim}])")
        wk_gpu = qkv_weight[q_dim: q_dim + kv_dim, :].contiguous()
        weight_layout = "unknown"
    print(f"  Wk from GPU model: shape={wk_gpu.shape}, layout={weight_layout}")

    # Per-channel weight scales
    weight_scale_all = layer0_attn.qkv_proj.weight_scale.detach().cpu().float().numpy().flatten()
    print(f"  weight_scale_all: {len(weight_scale_all)} elements")
    # Weight scales are per-output-channel, so indexed by output dimension
    k_weight_scale = weight_scale_all[q_dim: q_dim + kv_dim].copy()
    print(f"  K weight scale[:4]: {k_weight_scale[:4].tolist()}")

    # K bias
    bias_tensor = layer0_attn.qkv_proj.bias
    k_bias_np = None
    if bias_tensor is not None:
        bias_np = bias_tensor.detach().cpu().float().numpy().flatten()
        k_bias_np = bias_np[q_dim: q_dim + kv_dim].copy()
        print(f"  K bias present, first 4 = {k_bias_np[:4].tolist()}")
    else:
        print(f"  K bias: None")

    # ── Verify weights from safetensors ──
    print(f"\n{'='*70}")
    print("WEIGHT VERIFICATION (safetensors vs GPU model)")
    print(f"{'='*70}")

    model_dir = server._model_dir
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    # Try separate k_proj first, then fused qkv_proj
    wk_st = None
    k_scale_st = None
    k_bias_st = None
    for f in st_files:
        with safe_open(f, framework="pt") as sf:
            keys = sf.keys()
            # Separate weights
            k_name = f"model.layers.{TARGET_LAYER}.self_attn.k_proj.weight"
            if k_name in keys:
                wk_st = sf.get_tensor(k_name)
                print(f"  Found separate {k_name}: shape={wk_st.shape}")
            # Scale
            k_scale_name = f"model.layers.{TARGET_LAYER}.self_attn.k_proj.weight_scale"
            if k_scale_name in keys:
                k_scale_st = sf.get_tensor(k_scale_name)
            # Bias
            k_bias_name = f"model.layers.{TARGET_LAYER}.self_attn.k_proj.bias"
            if k_bias_name in keys:
                k_bias_st = sf.get_tensor(k_bias_name)

    if wk_st is None:
        # Try fused
        for f in st_files:
            with safe_open(f, framework="pt") as sf:
                qkv_name = f"model.layers.{TARGET_LAYER}.self_attn.qkv_proj.weight"
                if qkv_name in sf.keys():
                    qkv_w = sf.get_tensor(qkv_name)
                    wk_st = qkv_w[q_dim: q_dim + kv_dim, :]
                    print(f"  Found fused {qkv_name}, extracted K: shape={wk_st.shape}")
                    break

    if wk_st is not None:
        # Safetensors Wk is [kv_dim, hidden_dim] (row-major)
        # wk_gpu is already transposed to [kv_dim, hidden_dim]
        print(f"  Safetensors Wk shape: {wk_st.shape}, GPU Wk shape: {wk_gpu.shape}")
        if wk_st.shape == wk_gpu.shape:
            w_match = torch.equal(wk_st, wk_gpu)
            print(f"  Safetensors Wk == GPU model Wk: {w_match}")
            if not w_match:
                d = (wk_st.to(torch.int16) - wk_gpu.to(torch.int16)).abs()
                print(f"    Max diff: {d.max().item()}, nonzero: {(d > 0).sum().item()}")
        else:
            print(f"  Shape mismatch! Trying transpose...")
            wk_st_t = wk_st.t().contiguous()
            if wk_st_t.shape == wk_gpu.shape:
                w_match = torch.equal(wk_st_t, wk_gpu)
                print(f"  Safetensors Wk.T == GPU model Wk: {w_match}")
            else:
                print(f"  Cannot compare: ST={wk_st.shape}, GPU={wk_gpu.shape}")

    if k_scale_st is not None:
        k_sc_st_np = k_scale_st.float().numpy().flatten()
        sc_match = np.allclose(k_sc_st_np, k_weight_scale)
        print(f"  Safetensors K scale == GPU K scale: {sc_match}")

    if k_bias_st is not None and k_bias_np is not None:
        k_bi_st_np = k_bias_st.float().numpy().flatten()
        bi_match = np.allclose(k_bi_st_np, k_bias_np)
        print(f"  Safetensors K bias == GPU K bias: {bi_match}")

    # ── Hook rotary_emb to capture pre/post-RoPE Q/K ──
    layer0 = model_obj.model.layers[TARGET_LAYER]
    attn = layer0.self_attn
    orig_rotary = attn.rotary_emb
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
                captured_prefill["q_pre"] = q.detach().cpu().float().clone()
                captured_prefill["k_pre"] = k.detach().cpu().float().clone()
                captured_prefill["q_roped"] = q_out.detach().cpu().float().clone()
                captured_prefill["k_roped"] = k_out.detach().cpu().float().clone()
                call_counter[0] += 1
            elif self.enabled:
                call_counter[0] += 1
            return q_out, k_out

    wrapper = RotaryCapturePrefill(orig_rotary)
    attn.rotary_emb = wrapper

    # ── Warmup ──
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    print(f"\n--- Warmup ---")
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=4)

    # ── Run inference (prefill capture) ──
    print(f"\n--- Running inference with prefill capture ---")
    wrapper.enabled = True
    call_counter[0] = 0
    chat_r = server.chat(prompt="What is 2+2?", max_tokens=8, temperature=0.0)
    wrapper.enabled = False
    attn.rotary_emb = orig_rotary

    request_id = chat_r["request_id"]
    n_tokens = chat_r["n_tokens"]
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]
    print(f"  n_tokens={n_tokens}, n_prompt={n_prompt}")
    print(f"  Prefill positions: {captured_prefill.get('positions', 'NONE')}")

    if "positions" not in captured_prefill:
        print("  ERROR: prefill capture failed!")
        return

    gpu_k_pre = captured_prefill["k_pre"].numpy()  # pre-RoPE K from GPU
    gpu_k_post = captured_prefill["k_roped"].numpy()  # post-RoPE K from GPU
    gpu_positions = captured_prefill["positions"].numpy().flatten()
    n_captured = gpu_k_pre.shape[0]

    # Reshape to (n_tokens, kv_dim)
    if gpu_k_pre.ndim == 3:
        gpu_k_pre_flat = gpu_k_pre.reshape(n_captured, -1)
        gpu_k_post_flat = gpu_k_post.reshape(n_captured, -1)
    else:
        gpu_k_pre_flat = gpu_k_pre
        gpu_k_post_flat = gpu_k_post

    print(f"  GPU K pre-RoPE: shape={gpu_k_pre_flat.shape}")
    print(f"  GPU K post-RoPE: shape={gpu_k_post_flat.shape}")
    print(f"  GPU positions: {gpu_positions.tolist()}")

    # ── Get audit data (with deep prefix) ──
    print(f"\n--- Getting audit data ---")
    audit_pos = n_prompt - 1
    audit_json_str = server.audit(
        request_id=request_id,
        token_index=audit_pos,
        layer_indices=[TARGET_LAYER],
        tier="full",
        binary=False,
        use_captured_x_attn=True,
        deep_prefix=True,
    )
    audit_json = json.loads(audit_json_str)

    shell = audit_json.get("shell_opening", {})
    shell_layers = shell.get("layers", [])
    prefix_shells = audit_json.get("prefix_shell_openings", [])
    kv_entries = audit_json.get("kv_entries", [])

    print(f"  Shell layers: {len(shell_layers)}")
    print(f"  Prefix shells: {len(prefix_shells)}")
    print(f"  KV entries: {len(kv_entries)} layers")
    if kv_entries:
        print(f"  KV entries[0]: {len(kv_entries[0])} positions")

    # ── STEP 1: Accumulator & epilogue comparison for each position ──
    print(f"\n{'='*70}")
    print("STEP 1: SHELL k_acc → CUTLASS EPILOGUE → compare with GPU pre-RoPE K")
    print(f"{'='*70}")

    k_bias_list = k_bias_np.tolist() if k_bias_np is not None else None
    k_scale_list = k_weight_scale.tolist()

    for t in range(min(n_captured, n_prompt)):
        # Get shell k_acc for this position
        k_acc = None
        scale_x = None
        if t < len(prefix_shells):
            ps_layers = prefix_shells[t].get("layers", [])
            if ps_layers:
                k_acc_raw = ps_layers[0].get("k")
                scale_x = ps_layers[0].get("scale_x_attn")
                if k_acc_raw is not None:
                    k_acc = k_acc_raw
        elif t == audit_pos and shell_layers:
            k_acc_raw = shell_layers[0].get("k")
            scale_x = shell_layers[0].get("scale_x_attn")
            if k_acc_raw is not None:
                k_acc = k_acc_raw

        if k_acc is None:
            print(f"  pos={t}: NO k_acc in shell")
            continue

        # Apply our epilogue
        epilogue_k = np.array(
            verilm_rs.cutlass_epilogue_bf16(k_acc, k_scale_list, float(scale_x), k_bias_list),
            dtype=np.float32,
        )

        # GPU pre-RoPE K for this position (bf16 from qkv_proj output)
        gpu_k_t = gpu_k_pre_flat[t, :kv_dim].astype(np.float32)

        diff = np.abs(epilogue_k - gpu_k_t)
        linf = diff.max()
        mean_d = diff.mean()
        frac_eq = np.sum(diff == 0.0) / len(diff)

        print(f"  pos={t}: epilogue_vs_gpu L-inf={linf:.6f} mean={mean_d:.8f} "
              f"frac_eq={frac_eq:.4f} scale_x={scale_x}")

        if t == 0 or t == audit_pos:
            # Show details for first and last
            print(f"    epilogue[:8] = {[f'{v:.4f}' for v in epilogue_k[:8]]}")
            print(f"    gpu_k[:8]    = {[f'{v:.4f}' for v in gpu_k_t[:8]]}")
            print(f"    diff[:8]     = {[f'{v:.6f}' for v in diff[:8]]}")

    # ── STEP 2: Committed K vs GPU post-RoPE K ──
    print(f"\n{'='*70}")
    print("STEP 2: COMMITTED K (KV transcript, f64) vs GPU post-RoPE K")
    print(f"{'='*70}")

    if kv_entries and len(kv_entries) > 0:
        layer_kv = kv_entries[0]
        for t in range(min(len(layer_kv), n_captured)):
            committed_k = np.array(layer_kv[t]["k_roped"], dtype=np.float64)
            gpu_k_t = gpu_k_post_flat[t, :kv_dim].astype(np.float64)

            diff = np.abs(committed_k - gpu_k_t)
            linf = diff.max()
            mean_d = diff.mean()
            frac_eq = np.sum(diff == 0.0) / len(diff)

            print(f"  pos={t}: committed_vs_gpu L-inf={linf:.6f} mean={mean_d:.8f} "
                  f"frac_eq={frac_eq:.4f}")

            if t == 0:
                print(f"    committed[:8] = {[f'{v:.4f}' for v in committed_k[:8]]}")
                print(f"    gpu_roped[:8] = {[f'{v:.4f}' for v in gpu_k_t[:8]]}")

    # ── STEP 3: Independent accumulator verification ──
    # Compute torch._int_mm(x_attn_i8, Wk.T) on GPU and compare with shell k_acc
    print(f"\n{'='*70}")
    print("STEP 3: INDEPENDENT ACCUMULATOR VERIFICATION")
    print("Compute _int_mm(Wk, x_attn) on GPU and compare with shell k_acc")
    print(f"{'='*70}")

    # We need x_attn_i8 for each position. The shell k_acc was computed from
    # captured x_attn. We can get x_attn from the capture buffer indirectly
    # by checking if the shell k_acc matches our independent _int_mm.
    #
    # But we don't have x_attn_i8 directly. What we DO know:
    # - shell k_acc = matmul_i32(Wk_rust, x_attn_captured)
    # - If shell k_acc == Wk_gpu @ x_attn_captured, then Wk_rust == Wk_gpu
    #
    # We can verify Wk match via safetensors comparison (done in weight verification).
    # If weights match, then we need to check if x_attn_captured == GPU's actual x_attn.
    #
    # The x_attn_captured IS the GPU's x_attn (captured via _wrapped_cutlass_scaled_mm
    # hook on the QKV projection input). So if weights are correct, k_acc must be correct.
    #
    # If epilogue(k_acc) still doesn't match GPU K, the problem is in the epilogue
    # parameters (scales, bias), not the accumulator.

    # Sanity check: compute k_acc from a known x_attn
    # We can fabricate a test by extracting the GPU's qkv_proj output and working backwards
    #
    # GPU qkv_proj output = _scaled_mm(x_attn_i8, Wqkv.T, scale_a, scale_b, bias)
    # GPU K output = qkv_proj_output[:, q_dim:q_dim+kv_dim]
    #
    # The pre-RoPE K we captured IS the GPU K output (bf16 from _scaled_mm epilogue)
    #
    # If cutlass_epilogue_bf16(shell_k_acc, scales, scale_x, bias) == gpu_k_pre
    # then the accumulators are correct (given correct scales/bias).
    #
    # If they DON'T match, we need to determine if it's the accumulator or scales.

    # Check: can we reproduce GPU K from shell_k_acc using GPU's own scales?
    # The scales used in the shell come from verilm_rs.generate_key (keygen).
    # The GPU uses the model's actual scales.
    # These should be the same, but let's verify explicitly.

    print(f"  K weight scales (from GPU model)[:4]: {k_weight_scale[:4].tolist()}")

    # ── STEP 4: Diagnose per-element epilogue divergence ──
    print(f"\n{'='*70}")
    print("STEP 4: DETAILED EPILOGUE ELEMENT ANALYSIS (pos=0)")
    print(f"{'='*70}")

    # For position 0, show element-by-element epilogue computation
    k_acc_0 = None
    scale_x_0 = None
    if prefix_shells and prefix_shells[0].get("layers"):
        k_acc_raw = prefix_shells[0]["layers"][0].get("k")
        scale_x_0 = prefix_shells[0]["layers"][0].get("scale_x_attn")
        if k_acc_raw:
            k_acc_0 = np.array(k_acc_raw, dtype=np.int64)

    if k_acc_0 is not None:
        gpu_k_0 = gpu_k_pre_flat[0, :kv_dim].astype(np.float64)
        print(f"  scale_x_attn = {scale_x_0}")
        print(f"  K has bias: {k_bias_np is not None}")

        # Manual epilogue for first 8 elements
        for i in range(min(8, len(k_acc_0))):
            acc = int(k_acc_0[i])
            sw = float(k_weight_scale[i])
            sx = float(scale_x_0)
            temp = acc * sw  # f32(acc) * scale_b
            if k_bias_np is not None:
                bv = float(k_bias_np[i])
                # FMA: bf16(scale_a * temp + bias)
                result_f32 = sx * temp + bv  # fma
                result_bf16 = float(np.float16(np.float32(result_f32)))
                # Also try: bf16 with proper rounding
                import struct
                f32_bytes = struct.pack('f', result_f32)
                f32_bits = struct.unpack('I', f32_bytes)[0]
                bf16_bits = (f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) >> 16
                bf16_from_bits = struct.unpack('e', struct.pack('H', bf16_bits))[0] if bf16_bits < 0x7FFF else float(np.float32(struct.pack('H', bf16_bits)))
                # Use half crate equivalent: bf16 = truncate f32 to bfloat16
                import ctypes
                bf16_val = float(torch.tensor(result_f32, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).item())
            else:
                result_f32 = temp * sx
                bf16_val = float(torch.tensor(result_f32, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).item())

            gpu_val = float(gpu_k_0[i])
            diff = abs(bf16_val - gpu_val)
            print(f"  [{i:3d}] acc={acc:8d} sw={sw:.6f} "
                  f"{'bias='+f'{k_bias_np[i]:.4f}' if k_bias_np is not None else ''} "
                  f"→ bf16={bf16_val:.4f} gpu={gpu_val:.4f} diff={diff:.6f}")

    # ── STEP 5: Check scale_x_attn alignment across positions ──
    print(f"\n{'='*70}")
    print("STEP 5: scale_x_attn CONSISTENCY")
    print(f"{'='*70}")

    all_scales = []
    for t in range(min(len(prefix_shells), n_prompt)):
        ps_layers = prefix_shells[t].get("layers", [])
        if ps_layers:
            sx = ps_layers[0].get("scale_x_attn")
            all_scales.append((t, sx))

    if shell_layers:
        sx = shell_layers[0].get("scale_x_attn")
        all_scales.append((audit_pos, sx))

    for t, sx in all_scales:
        print(f"  pos={t}: scale_x_attn={sx}")

    # Note: in a fused _scaled_mm call with batch, scale_a is per-row.
    # Each token gets its own scale_a. The shell should store per-token scale.

    # ── Summary ──
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")

    # Compute overall stats
    epi_linfs = []
    committed_linfs = []
    for t in range(min(n_captured, n_prompt)):
        k_acc = None
        scale_x = None
        if t < len(prefix_shells):
            ps_layers = prefix_shells[t].get("layers", [])
            if ps_layers:
                k_acc = ps_layers[0].get("k")
                scale_x = ps_layers[0].get("scale_x_attn")
        elif t == audit_pos and shell_layers:
            k_acc = shell_layers[0].get("k")
            scale_x = shell_layers[0].get("scale_x_attn")

        if k_acc is not None:
            epi_k = np.array(
                verilm_rs.cutlass_epilogue_bf16(k_acc, k_scale_list, float(scale_x), k_bias_list),
                dtype=np.float32,
            )
            gpu_k_t = gpu_k_pre_flat[t, :kv_dim].astype(np.float32)
            epi_linfs.append(np.abs(epi_k - gpu_k_t).max())

        if kv_entries and len(kv_entries) > 0 and t < len(kv_entries[0]):
            ck = np.array(kv_entries[0][t]["k_roped"], dtype=np.float64)
            gk = gpu_k_post_flat[t, :kv_dim].astype(np.float64)
            committed_linfs.append(np.abs(ck - gk).max())

    if epi_linfs:
        print(f"  Epilogue(shell_k_acc) vs GPU pre-RoPE K:")
        print(f"    Max L-inf: {max(epi_linfs):.6f}")
        print(f"    Mean L-inf: {np.mean(epi_linfs):.6f}")
        if max(epi_linfs) < 0.01:
            print("    → MATCH: shell accumulators + epilogue reproduce GPU K exactly")
        else:
            print("    → MISMATCH: divergence in accumulators, scales, or epilogue")

    if committed_linfs:
        print(f"  Committed K (KV transcript) vs GPU post-RoPE K:")
        print(f"    Max L-inf: {max(committed_linfs):.6f}")
        print(f"    Mean L-inf: {np.mean(committed_linfs):.6f}")
        if max(committed_linfs) < 1.0:
            print("    → MATCH: KV transcript agrees with GPU")
        else:
            print("    → MISMATCH: KV transcript f64 dequant diverges from GPU bf16")

    if epi_linfs and committed_linfs:
        if max(epi_linfs) < 0.01 and max(committed_linfs) > 1.0:
            print("\n  CONCLUSION: Shell k_acc is CORRECT but committed K (KV transcript)")
            print("  uses f64 dequant that diverges from GPU bf16. The anchor_bf16 path")
            print("  (which uses cutlass_epilogue_bf16 on shell k_acc) should work IF")
            print("  all prefix positions have shell k_acc available.")
            print("  The committed K in KV entries is the wrong comparison target.")
        elif max(epi_linfs) >= 0.01:
            print("\n  CONCLUSION: Shell k_acc or epilogue DIVERGES from GPU K.")
            print("  Investigate: scale_x_attn capture, weight scale extraction,")
            print("  bias extraction, or _scaled_mm calling convention.")


@app.local_entrypoint()
def main():
    diagnose.remote()
    print("\nFunction completed. Retrieve log with:")
    print("  modal volume get verilm-diag-output diag_qwen_k.log /tmp/diag_qwen_k.log")
    print("  cat /tmp/diag_qwen_k.log")
