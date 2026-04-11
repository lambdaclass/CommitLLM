"""
Diagnostic: token 0, layer 0 — trace scale_a mismatch.

The first run showed:
  - v_deq and a_f64 match perfectly (replay math correct)
  - replayed_i8 = round(a_f64 / scale_a) ≈ 1.65x too large vs claimed_i8
  - So captured scale_a (0.0228) is too small vs actual GPU scale (~0.0375)

This run hooks o_proj layer 0 to capture the actual per-row scale_a
from dynamic quantization and compare with the committed value.

Usage:
    modal run --detach scripts/modal/diag_token0.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pins import VERIFICATION

import modal

app = modal.App("verilm-diag-token0")

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


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=600,
)
def diag():
    import hashlib
    import json
    import os
    import sys

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    import verilm_rs
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    model_id = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
    buf = get_capture_buffer()

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: scale_a trace — {model_id}")
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

    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(server._model_dir, key_seed)
    cfg = json.loads(key_json)["config"]

    # Extract per-channel weight scales
    import numpy as np
    hidden_dim = cfg["hidden_dim"]
    kv_dim = cfg["n_kv_heads"] * cfg["d_head"]
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    scales_dict = {"wq": [], "wk": [], "wv": []}
    for li in range(n_layers):
        layer = model.model.layers[li]
        ws = layer.self_attn.qkv_proj.weight_scale.detach().cpu().float().numpy().flatten()
        scales_dict["wq"].append(ws[:hidden_dim].tolist())
        scales_dict["wk"].append(ws[hidden_dim:hidden_dim + kv_dim].tolist())
        scales_dict["wv"].append(ws[hidden_dim + kv_dim:].tolist())
    scale_overrides_json = json.dumps(scales_dict)

    # ── Phase 1: Hook cutlass_scaled_mm to see raw scale_a at o_proj ──
    print(f"\n{'='*70}")
    print("Phase 1: Intercept cutlass_scaled_mm to log scale_a at o_proj layer 0")
    print(f"{'='*70}")

    calls_per_fwd = n_layers * cap.PROJS_PER_LAYER
    scale_log = []  # (global_idx, layer, proj_name, scale_numel, scale_vals)
    real_kernel = cap._real_kernel[0]

    # We must NOT break the existing capture hook's counter. So we wrap
    # the real kernel with a logging layer that fires BEFORE the hook.
    def logging_kernel(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
        # The hook's record() hasn't fired yet for this call.
        # Use the hook's internal counter to identify the projection.
        hook = cap._capture_hook
        if hook is not None:
            idx = hook.call_counter % calls_per_fwd
            layer = idx // cap.PROJS_PER_LAYER
            proj_idx = idx % cap.PROJS_PER_LAYER
            proj_name = cap.PROJ_SEQUENCE[proj_idx]
            if layer == 0:
                s = scale_a.detach().cpu().float().flatten()
                entry = {
                    "counter": hook.call_counter,
                    "layer": layer,
                    "proj": proj_name,
                    "a_shape": tuple(a.shape),
                    "scale_a_shape": tuple(scale_a.shape),
                    "scale_a_numel": scale_a.numel(),
                    "scale_a_vals": s.tolist()[:16],
                }
                # Save raw a_i8 for o_proj prefill (first call) for head mapping
                if proj_name == "o_proj" and a.shape[0] > 1:
                    entry["a_i8_rows"] = a.detach().cpu().to(torch.int8).numpy().tolist()
                scale_log.append(entry)
        # Call the REAL kernel (not the wrapper)
        return real_kernel(a, b, scale_a, scale_b, out_dtype, bias)

    # Replace the real kernel pointer so the wrapper calls our logging version
    cap._real_kernel[0] = logging_kernel

    # Generate with capture
    print(f"\n--- Generating 'What is 2+2?' (max_tokens=8) ---")
    chat_r = server.chat(prompt="What is 2+2?", max_tokens=8, temperature=0.0)
    n_tokens = chat_r["n_tokens"]
    request_id = chat_r["request_id"]
    print(f"  Generated {n_tokens} tokens, request_id={request_id}")

    # Restore real kernel
    cap._real_kernel[0] = real_kernel

    # Print scale_a log for o_proj layer 0
    o_proj_calls = [s for s in scale_log if s["proj"] == "o_proj"]
    print(f"\n--- o_proj layer 0: {len(o_proj_calls)} calls ---")
    for i, info in enumerate(o_proj_calls):
        print(f"\n  Call {i} (counter={info['counter']}):")
        print(f"    a_shape={info['a_shape']}")
        print(f"    scale_a_shape={info['scale_a_shape']} numel={info['scale_a_numel']}")
        print(f"    scale_a values = {[f'{v:.8f}' for v in info['scale_a_vals']]}")
        if info['scale_a_numel'] > 1:
            vals = info['scale_a_vals']
            print(f"    max = {max(vals):.8f}  min = {min(vals):.8f}")

    # Also print qkv_proj layer 0 (for reference)
    qkv_calls = [s for s in scale_log if s["proj"] == "qkv_proj"]
    print(f"\n--- qkv_proj layer 0: {len(qkv_calls)} calls ---")
    for i, info in enumerate(qkv_calls):
        print(f"  Call {i}: a_shape={info['a_shape']} scale_a_shape={info['scale_a_shape']} "
              f"numel={info['scale_a_numel']} vals[:4]={[f'{v:.8f}' for v in info['scale_a_vals'][:4]]}")

    # ── Phase 2: Corridor measurement at 3 positions ──
    #
    # Indexing:
    #   committed tokens = all_token_ids[1:]
    #   token_index 0                  = first retained prefill row (prompt pos 1)
    #   token_index n_prompt_tokens-1  = last prefill row = first generated token
    #   token_index n_prompt_tokens    = first decode token
    #
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]
    positions = {
        "prefill_row0": 0,
        "first_gen (last prefill)": n_prompt - 1,
        "first_decode": n_prompt,
    }

    print(f"\n{'='*70}")
    print(f"Phase 2: Corridor measurement (n_prompt={n_prompt})")
    print(f"{'='*70}")

    full_layers = list(range(n_layers))
    for label, tidx in positions.items():
        print(f"\n  --- {label} (token_index={tidx}) ---")
        audit_binary = server.audit(
            request_id=request_id,
            token_index=tidx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        print(f"  audit binary size: {len(audit_binary)} bytes")

        sys.stderr.flush()
        report_json = verilm_rs.measure_corridor_committed_kv(
            audit_binary, key_json, scale_overrides_json,
        )
        sys.stderr.flush()
        report = json.loads(report_json)

        print(f"  global_linf = {report['global_linf']}")
        if report["measurements"]:
            m0 = report["measurements"][0]
            print(f"  Layer 0: linf={m0['linf']}  frac_eq={m0['frac_eq']:.4f}")

    # ── Phase 3: Cross-reference ──
    print(f"\n{'='*70}")
    print("Phase 3: Scale cross-reference")
    print(f"{'='*70}")

    # The committed scale_a is from the Rust DIAG-CKV output.
    # Cross-reference with the actual o_proj scale_a we intercepted.
    if o_proj_calls:
        prefill_call = o_proj_calls[0]
        print(f"\n  Prefill o_proj call:")
        print(f"    a_shape = {prefill_call['a_shape']} (batch_size = {prefill_call['a_shape'][0]})")
        print(f"    scale_a values = {[f'{v:.8f}' for v in prefill_call['scale_a_vals']]}")
        if prefill_call['scale_a_numel'] > 1:
            vals = prefill_call['scale_a_vals']
            print(f"    max(scale_a) = {max(vals):.8f}")
            print(f"    min(scale_a) = {min(vals):.8f}")
            # First generated token = last prefill row (token_index = n_prompt-1)
            # First committed token = first prefill row (token_index = 0)
            last_idx = min(prefill_call['a_shape'][0] - 1, len(vals) - 1)
            print(f"    scale_a[last_row={last_idx}] = {vals[last_idx]:.8f}  (first gen token, idx={last_idx})")
            print(f"    scale_a[0] = {vals[0]:.8f}  (first committed token, prompt pos 1)")
        if len(o_proj_calls) > 1:
            decode_call = o_proj_calls[1]
            print(f"\n  First decode o_proj call:")
            print(f"    a_shape = {decode_call['a_shape']}")
            print(f"    scale_a values = {[f'{v:.8f}' for v in decode_call['scale_a_vals']]}")

    # ── Phase 4: Python-side per-head check using raw GPU a_i8 ──
    print(f"\n{'='*70}")
    print("Phase 4: Per-head comparison from raw GPU capture")
    print(f"{'='*70}")

    if o_proj_calls and "a_i8_rows" in o_proj_calls[0]:
        a_i8_gpu = o_proj_calls[0]["a_i8_rows"]  # list of lists
        row0 = np.array(a_i8_gpu[0], dtype=np.int8)  # batch row 0 = token 0
        hidden_dim = cfg["hidden_dim"]
        d_head = cfg["d_head"]
        n_q = cfg["n_q_heads"]
        n_kv = cfg["n_kv_heads"]
        heads_per_kv = n_q // n_kv

        print(f"\n  GPU a_i8 row 0: shape={row0.shape}, first 16 = {row0[:16].tolist()}")

        # Check: are Q heads within the same GQA group identical?
        # For n_kv=1, all Q heads sharing a KV head should produce the same V.
        print(f"\n  Intra-GQA-group consistency (n_kv=1 → should be identical):")
        for kvg in range(n_kv):
            ref_qh = kvg * heads_per_kv
            ref_slice = row0[ref_qh * d_head : (ref_qh + 1) * d_head]
            max_diff = 0
            for offset in range(1, heads_per_kv):
                qh = ref_qh + offset
                s = row0[qh * d_head : (qh + 1) * d_head]
                diff = int(np.max(np.abs(s.astype(np.int16) - ref_slice.astype(np.int16))))
                if diff > max_diff:
                    max_diff = diff
            print(f"    KV group {kvg} (Q heads {ref_qh}-{ref_qh + heads_per_kv - 1}): "
                  f"max intra-group diff = {max_diff}")

        # Show first 16 vals of a few Q heads
        print(f"\n  Head slices (first 8 elements):")
        for qh in [0, 1, 7, 14, 21]:
            if (qh + 1) * d_head <= len(row0):
                s = row0[qh * d_head : qh * d_head + 8]
                print(f"    qh={qh:>2} (kvg={qh // heads_per_kv}): {s.tolist()}")
    else:
        print("  (no raw a_i8 captured)")

    print(f"\n{'='*70}")
    print("DONE — compare DIAG-HEAD-MAP, GQA group consistency, and raw head slices")
    print(f"{'='*70}")


@app.local_entrypoint()
def main():
    diag.remote()
