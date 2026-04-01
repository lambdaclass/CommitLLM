"""
Diagnostic: GPU V vs software V reconstruction.

Captures the raw GPU-side qkv_proj output (BF16) for layer 0 and compares
the V slice against the prover's software reconstruction using SafeTensors
weights. This separates two failure modes:

  1. Weight mismatch — SafeTensors W_v ≠ vLLM's fused qkv_proj weight (V portion)
  2. Scale mismatch — INT8 matmul identical, but dequant scales differ

For layer 0, prefill token 0 (first committed token):
  - Hook cutlass_scaled_mm to capture: a (x_attn INT8), b (fused weights INT8),
    scale_a, scale_b, and the BF16 output
  - Extract V slice from fused output and weights
  - Load separate v_proj.weight + weight_scale from SafeTensors
  - Compare byte-for-byte and numerically

Usage:
    modal run --detach scripts/modal/diag_v_weights.py
"""

import modal

app = modal.App("verilm-diag-v-weights")

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
    .pip_install("vllm>=0.8", "torch", "numpy", "fastapi", "maturin", "safetensors")
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
    timeout=600,
)
def diag():
    import hashlib
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import numpy as np
    import torch
    from safetensors import safe_open
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer, get_model_from_llm
    from verilm.server import VerifiedInferenceServer

    model_id = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
    buf = get_capture_buffer()

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: GPU V vs software V — {model_id}")
    print(f"{'='*70}")

    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers

    # Warmup
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=8)

    # ── Phase 1: Hook qkv_proj layer 0 to capture everything ──
    print(f"\n{'='*70}")
    print("Phase 1: Capture qkv_proj layer 0 raw tensors")
    print(f"{'='*70}")

    calls_per_fwd = n_layers * cap.PROJS_PER_LAYER
    qkv_captures = []  # One entry per qkv_proj layer 0 call
    real_kernel = cap._real_kernel[0]

    def logging_kernel(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
        hook = cap._capture_hook
        if hook is not None:
            idx = hook.call_counter % calls_per_fwd
            layer = idx // cap.PROJS_PER_LAYER
            proj_idx = idx % cap.PROJS_PER_LAYER
            proj_name = cap.PROJ_SEQUENCE[proj_idx]
            if layer == 0 and proj_name == "qkv_proj":
                # Compute output FIRST so we can capture it
                output = real_kernel(a, b, scale_a, scale_b, out_dtype, bias)
                entry = {
                    "counter": hook.call_counter,
                    "a_shape": tuple(a.shape),
                    "b_shape": tuple(b.shape),
                    "scale_a_shape": tuple(scale_a.shape),
                    "scale_b_shape": tuple(scale_b.shape),
                    "output_shape": tuple(output.shape),
                    # Copy tensors to CPU for analysis
                    "a_i8": a.detach().cpu().to(torch.int8).numpy(),
                    "b_i8": b.detach().cpu().to(torch.int8).numpy(),
                    "scale_a": scale_a.detach().cpu().float().numpy().flatten(),
                    "scale_b": scale_b.detach().cpu().float().numpy().flatten(),
                    "output_bf16": output.detach().cpu().float().numpy(),
                }
                qkv_captures.append(entry)
                return output
        return real_kernel(a, b, scale_a, scale_b, out_dtype, bias)

    cap._real_kernel[0] = logging_kernel

    print(f"\n--- Generating 'What is 2+2?' (max_tokens=8) ---")
    chat_r = server.chat(prompt="What is 2+2?", max_tokens=8, temperature=0.0)
    n_tokens = chat_r["n_tokens"]
    request_id = chat_r["request_id"]
    print(f"  Generated {n_tokens} tokens, request_id={request_id}")

    cap._real_kernel[0] = real_kernel

    print(f"\n  Captured {len(qkv_captures)} qkv_proj layer 0 calls")
    if not qkv_captures:
        print("  ERROR: No captures! Aborting.")
        return

    # The first capture is the prefill call (multiple rows)
    prefill = qkv_captures[0]
    print(f"\n  Prefill shapes:")
    print(f"    a (x_attn):  {prefill['a_shape']}")
    print(f"    b (weights): {prefill['b_shape']}")
    print(f"    scale_a:     {prefill['scale_a_shape']}")
    print(f"    scale_b:     {prefill['scale_b_shape']}")
    print(f"    output:      {prefill['output_shape']}")

    # ── Phase 2: Load SafeTensors weights ──
    print(f"\n{'='*70}")
    print("Phase 2: Load V weights from SafeTensors")
    print(f"{'='*70}")

    # Find model files
    model = get_model_from_llm(llm)
    model_dir = server._model_dir
    print(f"  model_dir: {model_dir}")

    import glob
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    print(f"  SafeTensors files: {len(st_files)}")

    # Load v_proj weight and scale for layer 0
    v_weight_name = "model.layers.0.self_attn.v_proj.weight"
    v_scale_name = "model.layers.0.self_attn.v_proj.weight_scale"
    # Also load the full qkv weight for comparison
    qkv_weight_name = "model.layers.0.self_attn.qkv_proj.weight"
    qkv_scale_name = "model.layers.0.self_attn.qkv_proj.weight_scale"

    st_v_weight = None
    st_v_scale = None
    st_qkv_weight = None
    st_qkv_scale = None

    # Use PyTorch framework to handle bfloat16 scales
    for f in st_files:
        with safe_open(f, framework="pt") as sf:
            keys = sf.keys()
            if v_weight_name in keys:
                st_v_weight = sf.get_tensor(v_weight_name)
                print(f"  Found {v_weight_name}: shape={st_v_weight.shape} dtype={st_v_weight.dtype}")
            if v_scale_name in keys:
                st_v_scale = sf.get_tensor(v_scale_name)
                print(f"  Found {v_scale_name}: shape={st_v_scale.shape} dtype={st_v_scale.dtype}")
            if qkv_weight_name in keys:
                st_qkv_weight = sf.get_tensor(qkv_weight_name)
                print(f"  Found {qkv_weight_name}: shape={st_qkv_weight.shape} dtype={st_qkv_weight.dtype}")
            if qkv_scale_name in keys:
                st_qkv_scale = sf.get_tensor(qkv_scale_name)
                print(f"  Found {qkv_scale_name}: shape={st_qkv_scale.shape} dtype={st_qkv_scale.dtype}")

    # Also check what tensor names exist for layer 0 self_attn
    all_attn_keys = []
    for f in st_files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if "layers.0.self_attn" in k:
                    t = sf.get_tensor(k)
                    all_attn_keys.append((k, tuple(t.shape), str(t.dtype)))
    print(f"\n  All layer 0 self_attn tensors in SafeTensors:")
    for k, shape, dtype in sorted(all_attn_keys):
        print(f"    {k}: {shape} {dtype}")

    # ── Phase 3: Compare fused GPU weights vs SafeTensors ──
    print(f"\n{'='*70}")
    print("Phase 3: Compare weights (byte-for-byte)")
    print(f"{'='*70}")

    gpu_b = prefill["b_i8"]  # shape from CUTLASS: could be [hidden, out] or [out, hidden]
    gpu_scale_b = prefill["scale_b"]

    # Determine dimensions from config
    import verilm_rs
    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, key_seed)
    cfg = json.loads(key_json)["config"]
    hidden_dim = cfg["hidden_dim"]
    n_q = cfg["n_q_heads"]
    n_kv = cfg["n_kv_heads"]
    d_head = cfg["d_head"]
    q_dim = n_q * d_head
    kv_dim = n_kv * d_head
    out_dim = q_dim + 2 * kv_dim

    print(f"\n  Config: hidden={hidden_dim} q_dim={q_dim} kv_dim={kv_dim} d_head={d_head}")
    print(f"  GPU b shape: {gpu_b.shape}")
    print(f"  GPU scale_b shape: {gpu_scale_b.shape}")

    # Detect weight layout from shapes
    # CUTLASS `cutlass_scaled_mm(a, b, ...)`:
    #   If b shape = [hidden, out]: output = a @ b  (column-major / transposed storage)
    #   If b shape = [out, hidden]: output = a @ b^T (row-major / standard storage)
    if gpu_b.shape == (hidden_dim, out_dim):
        print(f"  Layout: b is [hidden, out] — GPU stores weight TRANSPOSED")
        print(f"  V is in COLUMNS {q_dim+kv_dim}:{out_dim}")
        v_start = q_dim + kv_dim
        v_end = out_dim
        # Extract V as columns then transpose to get [kv_dim, hidden_dim]
        gpu_v_weight = gpu_b[:, v_start:v_end].T  # [kv_dim, hidden_dim]
        gpu_k_weight_raw = gpu_b[:, q_dim:q_dim+kv_dim].T  # [kv_dim, hidden_dim]
        b_transposed = True
    elif gpu_b.shape == (out_dim, hidden_dim):
        print(f"  Layout: b is [out, hidden] — standard row-major")
        v_start = q_dim + kv_dim
        v_end = out_dim
        gpu_v_weight = gpu_b[v_start:v_end, :]  # [kv_dim, hidden_dim]
        gpu_k_weight_raw = gpu_b[q_dim:q_dim+kv_dim, :]
        b_transposed = False
    else:
        print(f"  UNEXPECTED b shape: {gpu_b.shape}")
        print(f"  Expected either [{hidden_dim}, {out_dim}] or [{out_dim}, {hidden_dim}]")
        return

    gpu_v_scale = gpu_scale_b[v_start:v_end] if len(gpu_scale_b) == out_dim else gpu_scale_b

    print(f"\n  GPU V weight (extracted): shape={gpu_v_weight.shape}")
    print(f"  GPU V scale: shape={gpu_v_scale.shape}")

    if st_v_weight is not None:
        # Convert torch tensor to numpy int8
        st_v_i8 = st_v_weight.to(torch.int8).numpy() if st_v_weight.dtype != torch.int8 else st_v_weight.numpy()
        print(f"\n  SafeTensors V weight: shape={st_v_i8.shape}")

        # Byte-for-byte comparison
        if st_v_i8.shape == gpu_v_weight.shape:
            diff = (gpu_v_weight.astype(np.int16) - st_v_i8.astype(np.int16))
            exact_match = np.all(diff == 0)
            n_differ = np.count_nonzero(diff)
            n_total = diff.size
            print(f"\n  Weight byte-for-byte match: {exact_match}")
            print(f"  Differing elements: {n_differ}/{n_total} ({100*n_differ/n_total:.2f}%)")
            if not exact_match:
                abs_diff = np.abs(diff)
                print(f"  Max abs diff: {abs_diff.max()}")
                print(f"  Mean abs diff: {abs_diff.mean():.4f}")
                # Show first few mismatches
                rows, cols = np.where(diff != 0)
                for idx in range(min(10, len(rows))):
                    r, c = rows[idx], cols[idx]
                    print(f"    [{r},{c}]: GPU={gpu_v_weight[r,c]}  ST={st_v_i8[r,c]}  diff={diff[r,c]}")
        else:
            print(f"  Shape mismatch! GPU={gpu_v_weight.shape} ST={st_v_i8.shape}")

        # Also check first 8 bytes of each for sanity
        print(f"\n  GPU V weight row 0 [:16]: {gpu_v_weight[0,:16].tolist()}")
        print(f"  ST  V weight row 0 [:16]: {st_v_i8[0,:16].tolist()}")
        print(f"  GPU V weight row -1 [:16]: {gpu_v_weight[-1,:16].tolist()}")
        print(f"  ST  V weight row -1 [:16]: {st_v_i8[-1,:16].tolist()}")
    else:
        print("\n  SafeTensors v_proj.weight NOT FOUND — may be fused as qkv_proj")

    # ── Phase 3b: Scale comparison ──
    print(f"\n  --- Scale comparison ---")
    if st_v_scale is not None:
        st_v_scale_f = st_v_scale.float().flatten().numpy()
        print(f"  SafeTensors V scale: shape={st_v_scale.shape} dtype={st_v_scale.dtype}")
        print(f"  GPU V scale shape: {gpu_v_scale.shape}")
        if len(st_v_scale_f) == len(gpu_v_scale):
            scale_diff = np.abs(gpu_v_scale - st_v_scale_f)
            exact = np.all(scale_diff == 0)
            print(f"  Scale exact match: {exact}")
            print(f"  Max scale diff: {scale_diff.max():.10f}")
            print(f"  Mean scale diff: {scale_diff.mean():.10f}")
            print(f"  GPU  V scale [:8]: {gpu_v_scale[:8].tolist()}")
            print(f"  ST   V scale [:8]: {st_v_scale_f[:8].tolist()}")
        else:
            print(f"  Scale length mismatch! GPU={len(gpu_v_scale)} ST={len(st_v_scale_f)}")
    else:
        print("  SafeTensors v_proj.weight_scale NOT FOUND")

    # If SafeTensors has fused qkv_proj, compare that too
    if st_qkv_weight is not None:
        st_qkv_i8 = st_qkv_weight.to(torch.int8).numpy() if st_qkv_weight.dtype != torch.int8 else st_qkv_weight.numpy()
        print(f"\n  SafeTensors has fused qkv_proj: shape={st_qkv_i8.shape}")
        if st_qkv_i8.shape == gpu_b.shape:
            diff = (gpu_b.astype(np.int16) - st_qkv_i8.astype(np.int16))
            exact = np.all(diff == 0)
            print(f"  Full QKV weight match: {exact}")
            if not exact:
                print(f"  Differing: {np.count_nonzero(diff)}/{diff.size}")

    # ── Phase 4: Compare GPU output vs software reconstruction ──
    print(f"\n{'='*70}")
    print("Phase 4: GPU BF16 output vs software INT8 matmul reconstruction")
    print(f"{'='*70}")

    # Pick row 0 of the prefill (= first committed token, prompt position 1)
    # Row 0 in prefill corresponds to token_ids[0] after BOS removal
    gpu_output = prefill["output_bf16"]  # [tokens, q+2kv] as float32
    gpu_scale_a = prefill["scale_a"]
    x_attn = prefill["a_i8"]  # [tokens, hidden] as int8

    row = 0  # First committed token
    x_row = x_attn[row]  # [hidden_dim] int8
    sa_row = gpu_scale_a[row] if len(gpu_scale_a) > 1 else gpu_scale_a[0]

    # GPU V output for this row — v_start/v_end refer to output columns
    gpu_v_out = gpu_output[row, v_start:v_end]  # [kv_dim] float32 (from bf16)

    print(f"\n  Row {row}: x_attn shape={x_row.shape}, scale_a={sa_row:.8f}")
    print(f"  GPU V output [{v_start}:{v_end}]: first 8 = {gpu_v_out[:8].tolist()}")

    # Software reconstruction using GPU's own fused weights
    v_acc_gpu_w = np.zeros(kv_dim, dtype=np.int64)
    for r in range(kv_dim):
        for c in range(hidden_dim):
            v_acc_gpu_w[r] += int(gpu_v_weight[r, c]) * int(x_row[c])

    # Dequantize with GPU's own scales
    v_deq_gpu_w = np.array([
        float(v_acc_gpu_w[r]) * float(sa_row) * float(gpu_v_scale[r])
        for r in range(kv_dim)
    ])

    print(f"  Software V (GPU weights): first 8 = {v_deq_gpu_w[:8].tolist()}")
    diff_gpu_w = np.abs(gpu_v_out - v_deq_gpu_w)
    print(f"  Diff (GPU out vs software w/ GPU weights): max={diff_gpu_w.max():.6f} mean={diff_gpu_w.mean():.6f}")

    # Software reconstruction using SafeTensors weights (if available)
    if st_v_weight is not None and st_v_scale is not None:
        st_v_i8 = st_v_weight.to(torch.int8).numpy() if st_v_weight.dtype != torch.int8 else st_v_weight.numpy()
        st_v_scale_f = st_v_scale.float().flatten().numpy()

        v_acc_st_w = np.zeros(kv_dim, dtype=np.int64)
        for r in range(kv_dim):
            for c in range(hidden_dim):
                v_acc_st_w[r] += int(st_v_i8[r, c]) * int(x_row[c])

        v_deq_st_w = np.array([
            float(v_acc_st_w[r]) * float(sa_row) * float(st_v_scale_f[r])
            for r in range(kv_dim)
        ])

        print(f"\n  Software V (ST weights):  first 8 = {v_deq_st_w[:8].tolist()}")
        diff_st_w = np.abs(gpu_v_out - v_deq_st_w)
        print(f"  Diff (GPU out vs software w/ ST weights): max={diff_st_w.max():.6f} mean={diff_st_w.mean():.6f}")

        # Cross-compare: are the i32 accumulators the same?
        acc_diff = np.abs(v_acc_gpu_w - v_acc_st_w)
        acc_match = np.all(acc_diff == 0)
        print(f"\n  INT32 accumulator match (GPU_w vs ST_w): {acc_match}")
        if not acc_match:
            print(f"  Accumulator diffs: max={acc_diff.max()} mean={acc_diff.mean():.1f}")
            n_diff = np.count_nonzero(acc_diff)
            print(f"  Differing accumulators: {n_diff}/{kv_dim}")

    # ── Phase 5: Compare captured x_attn vs what prover would see ──
    print(f"\n{'='*70}")
    print("Phase 5: x_attn consistency")
    print(f"{'='*70}")

    # The capture buffer should have captured x_attn for qkv_proj
    # Check if it matches what we captured in the hook
    print(f"  Hook x_attn row 0 [:16]: {x_row[:16].tolist()}")
    print(f"  Hook x_attn row 0 dtype: {x_row.dtype}")
    print(f"  scale_a for row 0: {sa_row:.10f}")

    # Also check multiple rows to see if scale_a varies
    n_rows = min(prefill["a_shape"][0], 8)
    print(f"\n  scale_a per row (first {n_rows} rows):")
    for r in range(n_rows):
        sa = gpu_scale_a[r] if len(gpu_scale_a) > 1 else gpu_scale_a[0]
        print(f"    row {r}: scale_a = {sa:.10f}")

    # ── Phase 6: Also compare K weights (to check if pattern is V-specific) ──
    print(f"\n{'='*70}")
    print("Phase 6: K weight comparison (control check)")
    print(f"{'='*70}")

    # Use pre-extracted K weight from GPU b (computed during layout detection)
    gpu_k_weight = gpu_k_weight_raw  # [kv_dim, hidden_dim]
    k_start_scale = q_dim
    k_end_scale = q_dim + kv_dim
    gpu_k_scale = gpu_scale_b[k_start_scale:k_end_scale] if len(gpu_scale_b) == out_dim else gpu_scale_b

    k_weight_name = "model.layers.0.self_attn.k_proj.weight"
    k_scale_name = "model.layers.0.self_attn.k_proj.weight_scale"
    st_k_weight = None
    st_k_scale = None
    for f_path in st_files:
        with safe_open(f_path, framework="pt") as sf:
            if k_weight_name in sf.keys():
                st_k_weight = sf.get_tensor(k_weight_name)
            if k_scale_name in sf.keys():
                st_k_scale = sf.get_tensor(k_scale_name)

    if st_k_weight is not None:
        st_k_i8 = st_k_weight.to(torch.int8).numpy() if st_k_weight.dtype != torch.int8 else st_k_weight.numpy()
        print(f"  ST K weight: {st_k_i8.shape}")
        if st_k_i8.shape == gpu_k_weight.shape:
            diff = (gpu_k_weight.astype(np.int16) - st_k_i8.astype(np.int16))
            exact = np.all(diff == 0)
            n_differ = np.count_nonzero(diff)
            print(f"  K weight byte-for-byte match: {exact}")
            print(f"  Differing elements: {n_differ}/{diff.size} ({100*n_differ/diff.size:.2f}%)")
            if not exact and n_differ > 0:
                abs_diff = np.abs(diff)
                print(f"  Max abs diff: {abs_diff.max()}")
    else:
        print("  ST k_proj.weight NOT FOUND")

    if st_k_scale is not None:
        st_k_scale_f = st_k_scale.float().flatten().numpy()
        scale_diff = np.abs(gpu_k_scale - st_k_scale_f)
        print(f"  K scale exact match: {np.all(scale_diff == 0)}")
        print(f"  K scale max diff: {scale_diff.max():.10f}")

    # ── Phase 7: Check vLLM's actual weight tensor vs SafeTensors ──
    print(f"\n{'='*70}")
    print("Phase 7: vLLM loaded weight tensor inspection")
    print(f"{'='*70}")

    layer0 = model.model.layers[0]
    qkv = layer0.self_attn.qkv_proj

    # The actual weight stored in the model
    w = qkv.weight.detach().cpu()
    print(f"  qkv_proj.weight: shape={w.shape} dtype={w.dtype}")

    # Weight scale
    ws = qkv.weight_scale.detach().cpu().float()
    print(f"  qkv_proj.weight_scale: shape={ws.shape}")

    # Input scale (if static quantization)
    if hasattr(qkv, 'input_scale') and qkv.input_scale is not None:
        inp_s = qkv.input_scale.detach().cpu().float()
        print(f"  qkv_proj.input_scale: {inp_s}")
    else:
        print("  qkv_proj.input_scale: None (dynamic quantization)")

    # Compare vLLM weight vs what CUTLASS received
    w_np = w.numpy() if w.dtype == torch.int8 else w.to(torch.int8).numpy()
    print(f"\n  vLLM weight shape: {w_np.shape}, CUTLASS b shape: {gpu_b.shape}")
    if w_np.shape == gpu_b.shape:
        diff = (gpu_b.astype(np.int16) - w_np.astype(np.int16))
        print(f"  vLLM weight == CUTLASS b: {np.all(diff == 0)}")
    elif b_transposed and w_np.T.shape == gpu_b.shape:
        diff = (gpu_b.astype(np.int16) - w_np.T.astype(np.int16))
        print(f"  vLLM weight.T == CUTLASS b: {np.all(diff == 0)}")
    else:
        print(f"  Shape mismatch and not transposable: vLLM={w_np.shape} CUTLASS={gpu_b.shape}")

    # Compare vLLM weight_scale vs CUTLASS scale_b
    ws_np = ws.numpy().flatten()
    print(f"  vLLM weight_scale len: {len(ws_np)}, CUTLASS scale_b len: {len(gpu_scale_b)}")
    if len(ws_np) == len(gpu_scale_b):
        scale_diff = np.abs(ws_np - gpu_scale_b)
        print(f"  vLLM weight_scale == CUTLASS scale_b: {np.all(scale_diff == 0)}")
        if not np.all(scale_diff == 0):
            print(f"  Max scale diff: {scale_diff.max():.10f}")
    else:
        print(f"  Scale length mismatch: vLLM={len(ws_np)} CUTLASS={len(gpu_scale_b)}")

    # The key question: does vLLM's V portion match SafeTensors v_proj.weight?
    if st_v_weight is not None:
        st_v_i8_np = st_v_weight.to(torch.int8).numpy() if st_v_weight.dtype != torch.int8 else st_v_weight.numpy()
        # vLLM stores as [out, hidden] = [4608, 3584]; V portion is rows v_start:v_end
        vllm_v = w_np[v_start:v_end, :]
        print(f"\n  vLLM V portion shape: {vllm_v.shape}, ST v_proj shape: {st_v_i8_np.shape}")
        if vllm_v.shape == st_v_i8_np.shape:
            diff = (vllm_v.astype(np.int16) - st_v_i8_np.astype(np.int16))
            match = np.all(diff == 0)
            print(f"  vLLM fused V portion == ST v_proj.weight: {match}")
            if not match:
                n_differ = np.count_nonzero(diff)
                print(f"  Differing: {n_differ}/{diff.size} ({100*n_differ/diff.size:.2f}%)")
        else:
            print(f"  Shape mismatch: vLLM V={vllm_v.shape} ST={st_v_i8_np.shape}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


@app.local_entrypoint()
def main():
    diag.remote()
