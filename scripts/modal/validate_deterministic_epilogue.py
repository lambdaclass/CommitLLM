"""
Validate deterministic epilogue against real CUTLASS output on Qwen 7B.

This script tests whether our Rust `cutlass_epilogue_bf16()` exactly matches
the real GPU's CUTLASS `_scaled_mm` output for the QKV projection.

Approach:
1. Run Qwen 7B inference with capture hooks enabled
2. For each QKV projection call, also compute `torch._int_mm` to get raw INT32 accumulators
3. Apply our deterministic epilogue (matching CUTLASS's exact computation order)
4. Compare: does our epilogue output match CUTLASS's output bit-for-bit?

If the max diff is 0, we have an exact match — the deterministic epilogue
specification is correct and the verifier can reproduce it perfectly.

Usage:
    modal run --detach scripts/modal/validate_deterministic_epilogue.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pins import VERIFICATION

import modal

app = modal.App("verilm-validate-epilogue")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "ca-certificates")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    })
    .pip_install(*VERIFICATION)
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

MODEL = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
PROMPT = (
    "Explain the theory of general relativity in detail, including its key principles, "
    "how it differs from special relativity, and its most important experimental confirmations."
)
MAX_TOKENS = 16


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "huggingface-cache", create_if_missing=True
        ),
    },
)
def validate_epilogue():
    """Compare deterministic epilogue against CUTLASS on Qwen 7B QKV projection."""
    import torch
    import numpy as np

    print("=" * 70)
    print("DETERMINISTIC EPILOGUE VALIDATION")
    print(f"Model: {MODEL}")
    print("=" * 70)

    # ── Step 1: Load model and get weight references ──
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    # Get model weights for QKV projection
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layers = model.model.layers

    # Extract layer 0 QKV weights and scales
    layer0 = layers[0]
    qkv_proj = layer0.self_attn.qkv_proj

    # W8A8 quantized: weight is int8, has weight_scale and input_scale
    w_int8 = qkv_proj.weight.data  # [out_features, in_features] int8
    scale_b = qkv_proj.weight_scale.data  # per-channel weight scale
    # input_scale may be per-tensor or per-token depending on scheme
    # We'll capture it from the actual _scaled_mm call

    # Check for bias
    has_bias = qkv_proj.bias is not None
    if has_bias:
        bias_tensor = qkv_proj.bias.data  # bf16
        bias_f32 = bias_tensor.float()
        print(f"  QKV bias: shape={bias_tensor.shape}, dtype={bias_tensor.dtype}")
    else:
        bias_f32 = None
        print("  QKV bias: None")

    print(f"  Weight: shape={w_int8.shape}, dtype={w_int8.dtype}, strides={w_int8.stride()}")
    print(f"  scale_b: shape={scale_b.shape}, dtype={scale_b.dtype}")
    print(f"  Weight is_contiguous: {w_int8.is_contiguous()}")
    print()

    # ── Step 2: Intercept _scaled_mm to capture inputs and INT32 accumulators ──
    captured_calls = []

    import vllm._custom_ops as ops
    real_kernel = ops.cutlass_scaled_mm

    def capturing_wrapper(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
        # Call real kernel
        output = real_kernel(a, b, scale_a, scale_b, out_dtype, bias)

        # Only capture QKV projections (first of every 4 calls per layer)
        call_idx = len(captured_calls) % 4
        if call_idx == 0 and len(captured_calls) < 4 * len(layers) and a.size(0) > 16:
            # Compute raw INT32 accumulators via torch._int_mm
            # _int_mm requires M > 16.
            # _int_mm expects: [M, K] @ [K, N] -> [M, N] in int32
            # CUTLASS cutlass_scaled_mm(a, b) computes a @ b^T or a @ b depending on layout.
            # We need to match a.size(1) == mat2.size(0) for _int_mm.
            with torch.no_grad():
                K = a.size(1)
                if b.size(0) == K:
                    # b is [K, N] — already correct layout for _int_mm
                    mat2 = b.contiguous()
                elif b.size(1) == K:
                    # b is [N, K] — need transpose to [K, N]
                    mat2 = b.T.contiguous()
                else:
                    print(f"    WARNING: shape mismatch a={a.shape}, b={b.shape}, skipping _int_mm")
                    captured_calls.append(None)
                    return output
                acc_i32 = torch._int_mm(a, mat2)

            captured_calls.append({
                "layer": len(captured_calls) // 4,
                "proj": "qkv",
                "a_int8": a.cpu().clone(),
                "scale_a": scale_a.cpu().clone(),
                "scale_b": scale_b.cpu().clone(),
                "bias": bias.cpu().clone() if bias is not None else None,
                "acc_i32": acc_i32.cpu().clone(),
                "cutlass_output": output.cpu().clone(),
                "out_dtype": out_dtype,
            })
        else:
            captured_calls.append(None)  # placeholder for non-QKV

        return output

    ops.cutlass_scaled_mm = capturing_wrapper

    # ── Step 3: Run inference ──
    print("Running inference...")
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    outputs = llm.generate([PROMPT], sp)
    print(f"  Generated: {outputs[0].outputs[0].text[:80]}...")
    print(f"  Total _scaled_mm calls captured: {len(captured_calls)}")
    print(f"  QKV captures: {sum(1 for c in captured_calls if c is not None)}")
    print()

    # Restore
    ops.cutlass_scaled_mm = real_kernel

    # ── Step 4: Validate deterministic epilogue ──
    print("=" * 70)
    print("EPILOGUE VALIDATION RESULTS")
    print("=" * 70)

    qkv_captures = [c for c in captured_calls if c is not None]
    if not qkv_captures:
        print("ERROR: No QKV captures!")
        return

    for cap in qkv_captures[:5]:  # First 5 layers
        layer_idx = cap["layer"]
        acc = cap["acc_i32"]            # [M, N] int32
        sa = cap["scale_a"]             # per-token: [M, 1] or scalar
        sb = cap["scale_b"]             # per-channel: [1, N]
        bias = cap["bias"]              # [N] bf16 or None
        cutlass_out = cap["cutlass_output"]  # [M, N] bf16

        M, N = acc.shape
        print(f"\n  Layer {layer_idx}: acc shape={acc.shape}")
        print(f"    scale_a: shape={sa.shape}, dtype={sa.dtype}")
        print(f"    scale_b: shape={sb.shape}, dtype={sb.dtype}")
        if bias is not None:
            print(f"    bias: shape={bias.shape}, dtype={bias.dtype}")

        # ── Method 1: Our deterministic epilogue in PyTorch ──
        # Matches CUTLASS's ScaledEpilogueBias EVT exactly:
        #   temp = f32(acc) * scale_b          (scale_b first, per-channel)
        #   out = bf16(fma(scale_a, temp, bias))  (or bf16(scale_a * temp) if no bias)
        acc_f32 = acc.float()

        # scale_b is per-channel weight scale. May be [N], [N,1], or [1,N].
        # We need it as [1, N] for broadcasting with [M, N].
        sb_flat = sb.float().squeeze()  # → [N]
        sb_broad = sb_flat.unsqueeze(0)  # → [1, N]

        # scale_a is per-token activation scale. May be scalar, [M], [M,1], or [1,1].
        # We need it as [M, 1] for broadcasting with [M, N].
        sa_flat = sa.float().squeeze()
        if sa_flat.dim() == 0:
            sa_broad = sa_flat.unsqueeze(0).unsqueeze(1)  # → [1, 1]
        else:
            sa_broad = sa_flat.unsqueeze(1)  # → [M, 1]

        # Step 1: f32(acc) * scale_b
        temp = acc_f32 * sb_broad.float()

        # Step 2: scale_a * temp (+ bias if present)
        if bias is not None:
            # CUTLASS uses homogeneous_multiply_add → compiler FMA
            # PyTorch equivalent: torch.addcmul or manual fma
            # torch doesn't expose scalar fma, so we use: scale_a * temp + bias
            # On GPU this should match if both use FMA...
            # Test both: with and without manual FMA
            bias_f32 = bias.float()  # bf16 → f32

            # Method A: separate multiply + add (two roundings)
            result_separate = (sa_broad * temp + bias_f32).to(torch.bfloat16)

            # Method B: use torch.addcmul for FMA-like behavior
            # Actually torch.addcmul is alpha*t1*t2 + t3, not fma.
            # Let's try: result = torch.fma(sa_broad, temp, bias_f32) if available
            # PyTorch doesn't have torch.fma for tensors... use element-wise
            # On CUDA, a*b+c may or may not get FMA'd by the compiler.
            result_fma_attempt = (sa_broad * temp + bias_f32).to(torch.bfloat16)

            # Compare both against CUTLASS
            cutlass_bf16 = cutlass_out.to(torch.bfloat16)

            diff_separate = (result_separate.float() - cutlass_bf16.float()).abs()
            max_diff_sep = diff_separate.max().item()
            n_mismatch_sep = (diff_separate > 0).sum().item()

            print(f"    Separate mul+add vs CUTLASS: max_diff={max_diff_sep:.6f}, mismatches={n_mismatch_sep}/{M*N}")

            # ── Method 2: Replicate exact CUTLASS order with per-element loop ──
            # This tests whether the issue is FMA or something else
            result_exact = torch.empty_like(cutlass_out)
            for row in range(min(M, 4)):  # Only check first few rows (slow loop)
                for col in range(N):
                    a_val = acc[row, col].item()
                    sw = sb_broad[0, col].item()
                    sx = sa_broad[row, 0].item() if sa_broad.shape[0] > 1 else sa_broad[0, 0].item()
                    bv = bias[col].float().item()

                    t = float(np.float32(a_val) * np.float32(sw))
                    # FMA: sx * t + bv with single rounding
                    r = float(np.float32(np.float32(sx) * np.float32(t) + np.float32(bv)))
                    result_exact[row, col] = torch.tensor(r, dtype=torch.float32).to(torch.bfloat16)

            for row in range(min(M, 4)):
                row_diff = (result_exact[row].float() - cutlass_bf16[row].float()).abs()
                max_row_diff = row_diff.max().item()
                n_row_mismatch = (row_diff > 0).sum().item()
                print(f"    Row {row} per-element (no-FMA) vs CUTLASS: max_diff={max_row_diff:.6f}, mismatches={n_row_mismatch}/{N}")

            # ── Method 3: Numpy with explicit FMA ──
            for row in range(min(M, 2)):
                acc_row = acc[row].numpy().astype(np.float32)
                sb_row = sb_broad[0].numpy().astype(np.float32)
                sx_val = np.float32(sa_broad[row, 0].item() if sa_broad.shape[0] > 1 else sa_broad[0, 0].item())
                bias_row = bias.float().numpy().astype(np.float32)

                # Step 1: acc * scale_b
                temp_np = acc_row * sb_row

                # Step 2a: separate mul + add
                result_sep = np.float32(sx_val) * temp_np + bias_row

                # Step 2b: numpy doesn't have FMA... use math.fma if available (Python 3.12+)
                try:
                    import math
                    result_fma_np = np.array([
                        math.fma(float(sx_val), float(t), float(b))
                        for t, b in zip(temp_np, bias_row)
                    ], dtype=np.float32)
                    has_fma = True
                except AttributeError:
                    result_fma_np = result_sep
                    has_fma = False

                cutlass_row = cutlass_bf16[row].float().numpy()

                # Convert to bf16 for comparison
                from numpy import frombuffer
                def to_bf16_rne(arr):
                    """Convert f32 array to bf16 (via torch) and back to f32."""
                    t = torch.from_numpy(arr).to(torch.bfloat16).float().numpy()
                    return t

                sep_bf16 = to_bf16_rne(result_sep)
                diff_sep_np = np.abs(sep_bf16 - cutlass_row)

                if has_fma:
                    fma_bf16 = to_bf16_rne(result_fma_np)
                    diff_fma_np = np.abs(fma_bf16 - cutlass_row)
                    print(f"    Row {row} numpy separate: max_diff={diff_sep_np.max():.8f}, mismatches={(diff_sep_np > 0).sum()}/{N}")
                    print(f"    Row {row} numpy FMA:      max_diff={diff_fma_np.max():.8f}, mismatches={(diff_fma_np > 0).sum()}/{N}")
                else:
                    print(f"    Row {row} numpy separate: max_diff={diff_sep_np.max():.8f}, mismatches={(diff_sep_np > 0).sum()}/{N}")
                    print(f"    (math.fma not available — Python < 3.12)")

        else:
            # No bias: bf16(scale_a * (f32(acc) * scale_b))
            result = (sa_broad * temp).to(torch.bfloat16)
            cutlass_bf16 = cutlass_out.to(torch.bfloat16)

            diff = (result.float() - cutlass_bf16.float()).abs()
            max_diff = diff.max().item()
            n_mismatch = (diff > 0).sum().item()
            print(f"    No-bias epilogue vs CUTLASS: max_diff={max_diff:.6f}, mismatches={n_mismatch}/{M*N}")

    # ── Step 5: Now test the Rust epilogue via verilm_rs ──
    print("\n" + "=" * 70)
    print("RUST EPILOGUE VALIDATION (verilm_rs.dequant_bias_cast_variants)")
    print("=" * 70)

    import verilm_rs

    for cap in qkv_captures[:3]:
        layer_idx = cap["layer"]
        acc = cap["acc_i32"]         # [M, N] int32
        sa = cap["scale_a"]          # per-token
        sb = cap["scale_b"]          # per-channel
        bias = cap["bias"]
        cutlass_out = cap["cutlass_output"]

        M, N = acc.shape

        # Extract scale_a as scalar (per-token, take first token's scale)
        sa_flat = sa.float().squeeze()
        sx = sa_flat[0].item() if sa_flat.dim() > 0 else sa_flat.item()

        # For row 0: test Rust epilogue
        acc_row = acc[0].numpy().tolist()
        sb_list = sb.float().squeeze().numpy().tolist()
        bias_list = bias.float().numpy().tolist() if bias is not None else None

        # Use the cast_variants diagnostic to compare
        import json
        variants_json = verilm_rs.dequant_bias_cast_variants(
            acc_row, sb_list, sx, bias_list
        )
        variants = json.loads(variants_json)

        cutlass_row = cutlass_out[0].float().numpy()

        print(f"\n  Layer {layer_idx}, row 0 ({N} elements):")
        for name, values in variants.items():
            values_np = np.array(values, dtype=np.float32)
            diff = np.abs(values_np - cutlass_row)
            max_d = diff.max()
            n_mis = (diff > 0).sum()
            print(f"    {name:25s}: max_diff={max_d:.8f}, mismatches={n_mis}/{N}")

        # ── Test the new cutlass_epilogue_bf16 Rust function ──
        rust_result = verilm_rs.cutlass_epilogue_bf16(
            acc_row, sb_list, sx, bias_list
        )
        rust_np = np.array(rust_result, dtype=np.float32)
        diff_rust = np.abs(rust_np - cutlass_row)
        max_d_rust = diff_rust.max()
        n_mis_rust = (diff_rust > 0).sum()
        print(f"    {'cutlass_epilogue_bf16':25s}: max_diff={max_d_rust:.8f}, mismatches={n_mis_rust}/{N}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


@app.local_entrypoint()
def main():
    validate_epilogue.remote()
