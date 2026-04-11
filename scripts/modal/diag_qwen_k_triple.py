"""
Qwen K triple-comparison diagnostic.

For each position in the prefill, compare three representations of pre-RoPE K:

  1. gpu_k_pre_rope       — straight from the fused qkv_proj output slice
  2. shell_k_bf16         — matmul_i32(Wk, x_attn_i8) → cutlass_epilogue_bf16(...)
  3. transcript_k_f64     — matmul_i32(Wk, x_attn_i8) → f64 dequant (scale_w * scale_x * acc + bias)

Branch-point logic:
  - GPU ≈ shell-bf16 for all positions → capture is fine, committed historical K
    uses wrong precision/semantics for strong anchoring
  - GPU ≠ shell-bf16 for positions 1+ → wrong x_attn_i8, scale, weight, or binding

Usage:
    modal run --detach scripts/modal/diag_qwen_k_triple.py
    modal volume get verilm-diag-output diag_qwen_k_triple.log /tmp/diag_qwen_k_triple.log
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pins import VERIFICATION

import modal

app = modal.App("verilm-qwen-k-triple")

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
    import sys
    import traceback

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

    log_file = open("/output/diag_qwen_k_triple.log", "w")
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
    import struct

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
    print(f"QWEN K TRIPLE-COMPARISON DIAGNOSTIC")
    print(f"Model: {model_id}")
    print(f"Layer: {TARGET_LAYER}")
    print(f"{'='*70}")

    # ── Load model ──
    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    buf = get_capture_buffer()
    n_layers = cap._n_layers

    # ── Dimensions ──
    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(server._model_dir, key_seed)
    key_cfg = json.loads(key_json)["config"]
    d_head = key_cfg["d_head"]
    n_q_heads = key_cfg["n_q_heads"]
    n_kv_heads = key_cfg["n_kv_heads"]
    hidden_dim = key_cfg["hidden_dim"]
    kv_dim = n_kv_heads * d_head
    q_dim = n_q_heads * d_head
    total_out = q_dim + 2 * kv_dim

    print(f"  hidden_dim={hidden_dim}, q_dim={q_dim}, kv_dim={kv_dim}")
    print(f"  n_kv_heads={n_kv_heads}, d_head={d_head}, n_layers={n_layers}")

    # ── Extract GPU weights & scales ──
    model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer0_attn = model_obj.model.layers[TARGET_LAYER].self_attn

    qkv_weight = layer0_attn.qkv_proj.weight.detach().cpu()
    if qkv_weight.shape[0] == hidden_dim and qkv_weight.shape[1] == total_out:
        wk_gpu = qkv_weight[:, q_dim: q_dim + kv_dim].contiguous().t()  # [kv_dim, hidden_dim]
        layout = "col"
    elif qkv_weight.shape[0] == total_out and qkv_weight.shape[1] == hidden_dim:
        wk_gpu = qkv_weight[q_dim: q_dim + kv_dim, :].contiguous()
        layout = "row"
    else:
        raise RuntimeError(f"Unexpected qkv_weight shape: {qkv_weight.shape}")
    print(f"  Wk from GPU: shape={wk_gpu.shape}, layout={layout}")

    weight_scale_all = layer0_attn.qkv_proj.weight_scale.detach().cpu().float().numpy().flatten()
    k_weight_scale = weight_scale_all[q_dim: q_dim + kv_dim].copy()

    bias_tensor = layer0_attn.qkv_proj.bias
    k_bias_np = None
    if bias_tensor is not None:
        bias_np = bias_tensor.detach().cpu().float().numpy().flatten()
        k_bias_np = bias_np[q_dim: q_dim + kv_dim].copy()
        print(f"  K bias present, first 4 = {k_bias_np[:4].tolist()}")
    else:
        print(f"  K bias: None")

    # ── Verify Rust Wk matches GPU Wk ──
    model_dir = server._model_dir
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    wk_st = None
    for f in st_files:
        with safe_open(f, framework="pt") as sf:
            k_name = f"model.layers.{TARGET_LAYER}.self_attn.k_proj.weight"
            if k_name in sf.keys():
                wk_st = sf.get_tensor(k_name)
                break
    if wk_st is not None:
        if wk_st.shape == wk_gpu.shape:
            w_match = torch.equal(wk_st, wk_gpu)
            print(f"  Safetensors Wk == GPU Wk: {w_match}")
        else:
            print(f"  Wk shape mismatch: ST={wk_st.shape}, GPU={wk_gpu.shape}")

    # ── Hook rotary_emb to capture GPU pre-RoPE K ──
    layer0 = model_obj.model.layers[TARGET_LAYER]
    attn = layer0.self_attn
    orig_rotary = attn.rotary_emb
    captured_prefill = {}
    call_counter = [0]

    class RotaryCapture(torch.nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.orig = orig
            self.enabled = False

        def forward(self, positions, q, k, *args, **kwargs):
            q_out, k_out = self.orig(positions, q, k, *args, **kwargs)
            if self.enabled and call_counter[0] == 0:
                captured_prefill["positions"] = positions.detach().cpu().clone()
                captured_prefill["k_pre"] = k.detach().cpu().float().clone()
                call_counter[0] += 1
            elif self.enabled:
                call_counter[0] += 1
            return q_out, k_out

    wrapper = RotaryCapture(orig_rotary)
    attn.rotary_emb = wrapper

    # ── Hook _real_kernel to capture x_attn_i8 for layer 0 QKV ──
    # Wrap the REAL kernel so our code runs when _wrapped_cutlass_scaled_mm
    # calls _real_kernel[0](a, b, ...).
    import verilm.capture as cap_mod
    raw_x_attn_sync = []
    _hook_enabled = [False]
    _smm_count = [0]

    _orig_real_kernel = cap_mod._real_kernel[0]
    print(f"  _real_kernel type: {type(_orig_real_kernel)}")
    print(f"  _real_kernel value: {_orig_real_kernel}")

    raw_scale_a_sync = []  # per-row activation scales from GPU
    raw_qkv_output_sync = []  # full QKV output from _scaled_mm

    def _diag_real_kernel(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
        _smm_count[0] += 1
        result = _orig_real_kernel(a, b, scale_a, scale_b, out_dtype, bias)
        if _hook_enabled[0]:
            calls_per_fwd = n_layers * 4
            idx = (_smm_count[0] - 1)
            proj_idx = idx % 4
            layer = (idx % calls_per_fwd) // 4
            if layer == TARGET_LAYER and proj_idx == 0:
                # QKV proj layer 0 — synchronous copy of input AND output
                torch.cuda.synchronize()
                raw_x_attn_sync.append(a.detach().cpu().clone())
                raw_scale_a_sync.append(scale_a.detach().cpu().float().clone())
                raw_qkv_output_sync.append(result.detach().cpu().float().clone())
                print(f"  [HOOK] call {idx}: a={a.shape} scale_a={scale_a.shape} "
                      f"out={result.shape} "
                      f"scale_a={scale_a.detach().cpu().float().numpy().flatten().tolist()}")
        return result

    cap_mod._real_kernel[0] = _diag_real_kernel
    print(f"  Patched _real_kernel[0] = {cap_mod._real_kernel[0]}")

    # Also verify the function reference chain:
    print(f"  torch._scaled_mm = {torch._scaled_mm}")
    print(f"  cap._wrapped = {cap_mod._wrapped_cutlass_scaled_mm}")

    # ── Warmup ──
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    print(f"\n--- Warmup ---")
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=4)

    # ── Run inference with capture ──
    print(f"\n--- Running inference ---")
    wrapper.enabled = True
    _hook_enabled[0] = True
    _smm_count[0] = 0
    call_counter[0] = 0
    chat_r = server.chat(prompt="What is 2+2?", max_tokens=8, temperature=0.0)
    wrapper.enabled = False
    _hook_enabled[0] = False
    attn.rotary_emb = orig_rotary
    cap_mod._real_kernel[0] = _orig_real_kernel  # restore

    request_id = chat_r["request_id"]
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]
    print(f"  n_prompt={n_prompt}")

    if "k_pre" not in captured_prefill:
        print("  ERROR: prefill capture failed!")
        return

    gpu_k_pre = captured_prefill["k_pre"].numpy()
    n_captured = gpu_k_pre.shape[0]
    if gpu_k_pre.ndim == 3:
        gpu_k_pre_flat = gpu_k_pre.reshape(n_captured, -1)
    else:
        gpu_k_pre_flat = gpu_k_pre
    print(f"  GPU K pre-RoPE: shape={gpu_k_pre_flat.shape}")
    print(f"  Positions: {captured_prefill['positions'].numpy().flatten().tolist()}")

    # ── Verify captured_x_attn state and compare bridge vs captured ──
    print(f"\n{'='*70}")
    print("BRIDGE vs CAPTURED x_attn COMPARISON")
    print(f"{'='*70}")

    audit_pos = n_prompt - 1
    entry = server._audit_store.get(request_id)
    if entry:
        state = entry["state"]
        print(f"  has_captured_x_attn: {state.has_captured_x_attn()}")
        print(f"  use_captured_x_attn (default): {state.use_captured_x_attn}")

        # Run audit with bridge-only (use_captured_x_attn=False)
        audit_no_cap = json.loads(server.audit(
            request_id=request_id,
            token_index=audit_pos,
            layer_indices=[TARGET_LAYER],
            binary=False,
            use_captured_x_attn=False,
            deep_prefix=True,
        ))
        ps_no_cap = audit_no_cap.get("prefix_shell_openings", [])

        # Run audit with captured x_attn (use_captured_x_attn=True)
        audit_with_cap = json.loads(server.audit(
            request_id=request_id,
            token_index=audit_pos,
            layer_indices=[TARGET_LAYER],
            binary=False,
            use_captured_x_attn=True,
            deep_prefix=True,
        ))
        ps_with_cap = audit_with_cap.get("prefix_shell_openings", [])

        print(f"  Bridge-only prefix shells: {len(ps_no_cap)}")
        print(f"  Captured prefix shells: {len(ps_with_cap)}")

        for t in range(min(len(ps_no_cap), len(ps_with_cap), n_prompt)):
            k_bridge = ps_no_cap[t]["layers"][0].get("k", [])[:4] if ps_no_cap[t].get("layers") else None
            k_captured = ps_with_cap[t]["layers"][0].get("k", [])[:4] if ps_with_cap[t].get("layers") else None
            same = k_bridge == k_captured
            tag = "SAME" if same else "DIFF"
            print(f"  pos={t}: bridge[:4]={k_bridge}  captured[:4]={k_captured}  [{tag}]")

    # ── Use the captured audit for the triple comparison ──
    audit_json = audit_with_cap

    shell = audit_json.get("shell_opening", {})
    shell_layers = shell.get("layers", [])
    prefix_shells = audit_json.get("prefix_shell_openings", [])
    kv_entries = audit_json.get("kv_entries", [])

    print(f"  Prefix shells: {len(prefix_shells)}")
    print(f"  KV entries: {len(kv_entries)} layers, {len(kv_entries[0]) if kv_entries else 0} positions")

    # ── Helper: f64 dequant (same as compute_kv_transcript) ──
    def dequant_f64_per_channel(k_acc_i32, per_ch_scale, scale_x, bias):
        """Replicate the f64 dequant path from compute_kv_transcript."""
        k = np.zeros(len(k_acc_i32), dtype=np.float64)
        for i, acc_val in enumerate(k_acc_i32):
            k[i] = float(acc_val) * float(per_ch_scale[i]) * float(scale_x)
            if bias is not None:
                k[i] += float(bias[i])
        return k

    # ── Helper: bf16 conversion ──
    def f32_to_bf16(x):
        """Truncate f32 to bf16 (same as hardware truncation)."""
        x32 = np.float32(x)
        bs = struct.pack('f', x32)
        # bf16 = top 16 bits of f32
        bf16_bytes = bs[2:4]
        # Reconstruct as f32 with bottom 16 bits zeroed
        return struct.unpack('f', b'\x00\x00' + bf16_bytes)[0]

    # ── COMPARISON TABLE ──
    print(f"\n{'='*70}")
    print("TRIPLE COMPARISON: GPU vs shell-bf16 vs transcript-f64 (pre-RoPE K)")
    print(f"{'='*70}")
    print(f"{'pos':>4s}  {'GPU_vs_ShellBF16':>16s}  {'ShellBF16_vs_TxF64':>18s}  {'GPU_vs_TxF64':>14s}")
    print(f"{'':>4s}  {'(L-inf)':>16s}  {'(L-inf)':>18s}  {'(L-inf)':>14s}")
    print(f"{'-'*60}")

    gpu_vs_shell = []
    shell_vs_tx = []
    gpu_vs_tx = []

    k_scale_list = k_weight_scale.tolist()
    k_bias_list = k_bias_np.tolist() if k_bias_np is not None else None

    for t in range(min(n_captured, n_prompt)):
        # Get shell k_acc for this position
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

        if k_acc is None:
            print(f"  {t:4d}  NO k_acc available")
            continue

        # (1) GPU pre-RoPE K
        gpu_k_t = gpu_k_pre_flat[t, :kv_dim].astype(np.float32)

        # (2) Shell-bf16: cutlass_epilogue_bf16(k_acc, scales, scale_x, bias)
        shell_bf16 = np.array(
            verilm_rs.cutlass_epilogue_bf16(k_acc, k_scale_list, float(scale_x), k_bias_list),
            dtype=np.float32,
        )

        # (3) Transcript-f64: same k_acc, but f64 dequant path
        tx_f64 = dequant_f64_per_channel(k_acc, k_weight_scale, float(scale_x), k_bias_np)

        # Diffs
        d_gpu_shell = np.abs(gpu_k_t - shell_bf16).max()
        d_shell_tx = np.abs(shell_bf16.astype(np.float64) - tx_f64).max()
        d_gpu_tx = np.abs(gpu_k_t.astype(np.float64) - tx_f64).max()

        gpu_vs_shell.append(d_gpu_shell)
        shell_vs_tx.append(d_shell_tx)
        gpu_vs_tx.append(d_gpu_tx)

        print(f"  {t:4d}  {d_gpu_shell:16.6f}  {d_shell_tx:18.6f}  {d_gpu_tx:14.6f}")

        # Show detail for pos 0 and 1
        if t <= 1:
            print(f"        gpu[:4]       = {gpu_k_t[:4].tolist()}")
            print(f"        shell_bf16[:4]= {shell_bf16[:4].tolist()}")
            print(f"        tx_f64[:4]    = {[f'{v:.6f}' for v in tx_f64[:4]]}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if gpu_vs_shell:
        print(f"  GPU vs shell-bf16:    max={max(gpu_vs_shell):.6f}  mean={np.mean(gpu_vs_shell):.6f}")
    if shell_vs_tx:
        print(f"  shell-bf16 vs tx-f64: max={max(shell_vs_tx):.6f}  mean={np.mean(shell_vs_tx):.6f}")
    if gpu_vs_tx:
        print(f"  GPU vs tx-f64:        max={max(gpu_vs_tx):.6f}  mean={np.mean(gpu_vs_tx):.6f}")

    # ── STEP 2: Independent x_attn verification ──
    # Use the raw x_attn tensor captured at _scaled_mm time (ground truth)
    # and compute Wk @ x_attn per-token to verify against shell k_acc.
    print(f"\n{'='*70}")
    print("INDEPENDENT x_attn VERIFICATION (ground truth from _scaled_mm hook)")
    print(f"{'='*70}")

    # Use raw_x_attn_sync (synchronous copy from _scaled_mm hook)
    print(f"  raw_x_attn_sync captures: {len(raw_x_attn_sync)}")
    print(f"  raw_scale_a_sync captures: {len(raw_scale_a_sync)}")

    # Compare GPU scale_a with shell scale_x_attn for each position
    if raw_scale_a_sync:
        gpu_scales = raw_scale_a_sync[0].numpy().flatten()  # [7] for prefill
        print(f"  GPU scale_a (per-row): {gpu_scales.tolist()}")
        for t in range(min(len(gpu_scales), len(prefix_shells))):
            shell_sx = prefix_shells[t]["layers"][0].get("scale_x_attn") if prefix_shells[t].get("layers") else None
            if shell_sx is not None:
                match = abs(float(shell_sx) - float(gpu_scales[t])) < 1e-7
                tag = "MATCH" if match else f"MISMATCH (gpu={gpu_scales[t]}, shell={shell_sx})"
                print(f"    pos={t}: gpu_scale={gpu_scales[t]:.8f} shell_scale={float(shell_sx):.8f} [{tag}]")
    if raw_x_attn_sync:
        xa_tensor = raw_x_attn_sync[0]  # first QKV call for layer 0 = prefill
        print(f"  Sync x_attn[0]: shape={xa_tensor.shape}, dtype={xa_tensor.dtype}")
        xa_layer0 = xa_tensor.numpy()
        print(f"  As numpy: shape={xa_layer0.shape}, dtype={xa_layer0.dtype}")

        # Compute Wk @ x_attn for each row using Python int arithmetic
        wk_np = wk_gpu.numpy().astype(np.int16)  # [kv_dim, hidden_dim]
        for t in range(min(xa_layer0.shape[0], n_prompt)):
            x_row = xa_layer0[t].astype(np.int16)  # [hidden_dim]
            # matmul: [kv_dim, hidden_dim] @ [hidden_dim] → [kv_dim]
            k_acc_py = (wk_np.astype(np.int64) @ x_row.astype(np.int64)).astype(np.int64)

            # Compare with shell k_acc
            shell_k_acc = None
            if t < len(prefix_shells):
                ps_layers = prefix_shells[t].get("layers", [])
                if ps_layers:
                    shell_k_acc = ps_layers[0].get("k")
            elif t == audit_pos and shell_layers:
                shell_k_acc = shell_layers[0].get("k")

            if shell_k_acc is not None:
                shell_arr = np.array(shell_k_acc, dtype=np.int64)
                diff = np.abs(k_acc_py - shell_arr)
                linf = diff.max()
                n_diff = (diff > 0).sum()
                # Also compute epilogue from our independent k_acc and compare with GPU
                epi_indep = np.array(
                    verilm_rs.cutlass_epilogue_bf16(k_acc_py.tolist(), k_scale_list,
                        float(prefix_shells[t]["layers"][0]["scale_x_attn"])
                        if t < len(prefix_shells) else 0.0,
                        k_bias_list),
                    dtype=np.float32,
                )
                gpu_k_t = gpu_k_pre_flat[t, :kv_dim].astype(np.float32)
                epi_gpu_diff = np.abs(epi_indep - gpu_k_t).max()

                print(f"  pos={t}: py_matmul_vs_shell L-inf={linf}  "
                      f"epi(py_matmul)_vs_gpu L-inf={epi_gpu_diff:.6f}  "
                      f"nonzero_diffs={n_diff}/{len(diff)}")
                if t <= 1:
                    print(f"    py_k_acc[:4]    = {k_acc_py[:4].tolist()}")
                    print(f"    shell_k_acc[:4] = {shell_arr[:4].tolist()}")
                    print(f"    x_attn[:8]      = {xa_layer0[t, :8].tolist()}")
            else:
                print(f"  pos={t}: no shell k_acc")
                print(f"    x_attn[:8] = {xa_layer0[t, :8].tolist()}")
    else:
        print(f"  No sync x_attn captures!")

    # ── QKV output K-slice vs rotary hook K comparison ──
    print(f"\n{'='*70}")
    print("QKV OUTPUT vs ROTARY HOOK K COMPARISON")
    print(f"{'='*70}")
    if raw_qkv_output_sync:
        qkv_out = raw_qkv_output_sync[0].numpy()  # [7, 4608] f32
        print(f"  QKV output shape: {qkv_out.shape}")
        # K slice from QKV output
        k_from_qkv = qkv_out[:, q_dim: q_dim + kv_dim]  # [7, 512]
        print(f"  K from QKV output shape: {k_from_qkv.shape}")
        for t in range(min(k_from_qkv.shape[0], n_prompt)):
            gpu_k_t = gpu_k_pre_flat[t, :kv_dim].astype(np.float32)
            qkv_k_t = k_from_qkv[t].astype(np.float32)
            diff = np.abs(gpu_k_t - qkv_k_t).max()
            tag = "MATCH" if diff < 1e-6 else f"DIFF={diff:.6f}"
            if t <= 2:
                print(f"  pos={t}: qkv_k[:4]={qkv_k_t[:4].tolist()}  "
                      f"rotary_k[:4]={gpu_k_t[:4].tolist()}  [{tag}]")
            else:
                print(f"  pos={t}: [{tag}]")
    else:
        print("  No QKV output captures!")

    # ── Branch-point diagnosis ──
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")
    if gpu_vs_shell:
        max_gpu_shell = max(gpu_vs_shell)
        max_shell_tx = max(shell_vs_tx) if shell_vs_tx else 0
        max_gpu_tx = max(gpu_vs_tx) if gpu_vs_tx else 0

        if max_gpu_shell < 0.01:
            print("  GPU ≈ shell-bf16 for ALL positions.")
            print("  → Shell capture + epilogue reproduce GPU K exactly.")
            if max_shell_tx > 0.1:
                print(f"  → shell-bf16 vs tx-f64 diverges ({max_shell_tx:.4f}).")
                print("  → CONCLUSION: Qwen strong tier needs a different historical-K")
                print("    commitment object. The f64 transcript path produces a K that")
                print("    doesn't match the GPU's bf16 K. This is a protocol design")
                print("    issue, not a capture/binding bug.")
                print("  → FIX: Commit cutlass_epilogue_bf16(k_acc) as the K object,")
                print("    not the f64 dequant K.")
            else:
                print(f"  → All three agree ({max_shell_tx:.4f}). No divergence.")
        else:
            # Check per-position pattern
            if len(gpu_vs_shell) > 1 and gpu_vs_shell[0] < 0.01 and max(gpu_vs_shell[1:]) > 0.1:
                print(f"  GPU ≈ shell-bf16 ONLY at pos=0 ({gpu_vs_shell[0]:.6f}).")
                print(f"  pos=1+ diverges (max={max(gpu_vs_shell[1:]):.6f}).")
                print("  → CONCLUSION: x_attn_i8 capture/binding bug for prefix positions.")
                print("    Position 0 uses different x_attn source than 1+.")
                print("  → Investigate: captured_x_attn indexing in build_retained_from_captures,")
                print("    or whether bridge-derived x_attn is being used instead of captured.")
            else:
                print(f"  GPU ≠ shell-bf16 (max={max_gpu_shell:.6f}).")
                print("  → CONCLUSION: wrong x_attn_i8, scale, weight, or bias in shell path.")


@app.local_entrypoint()
def main():
    diagnose.remote()
    print("\nFunction completed. Retrieve log with:")
    print("  modal volume get verilm-diag-output diag_qwen_k_triple.log /tmp/diag_qwen_k_triple.log")
    print("  cat /tmp/diag_qwen_k_triple.log")
