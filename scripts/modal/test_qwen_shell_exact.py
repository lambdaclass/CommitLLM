"""
Qwen shell-K exactness regression test.

Asserts that shell K (from captured x_attn + bf16 epilogue) exactly matches
the GPU's QKV output K slice for ALL prefix positions, not just pos=0.

Background: The Qwen anchor gap (L-inf 6-10) was caused by the bridge-derived
x_attn diverging from vLLM's fused norm+quant kernel. With captured x_attn,
the shell K is exact. This test prevents regressions on that path.

Usage:
    modal run scripts/modal/test_qwen_shell_exact.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pins import VERIFICATION

import modal

app = modal.App("verilm-qwen-shell-exact")

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


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def check_qwen_shell_exact(model_id: str = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"):
    """Assert shell K from captured x_attn == GPU QKV output K for all prefix positions."""
    import hashlib
    import json

    import numpy as np
    import torch
    import verilm_rs
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    print(f"\n{'=' * 70}")
    print(f"Qwen shell-K exactness regression: {model_id}")
    print(f"{'=' * 70}")

    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    buf = get_capture_buffer()
    n_layers = cap._n_layers
    import verilm.capture as cap_mod

    # ── Dimensions ──
    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(server._model_dir, key_seed)
    key_cfg = json.loads(key_json)["config"]
    d_head = key_cfg["d_head"]
    n_kv_heads = key_cfg["n_kv_heads"]
    n_q_heads = key_cfg["n_q_heads"]
    hidden_dim = key_cfg["hidden_dim"]
    kv_dim = n_kv_heads * d_head
    q_dim = n_q_heads * d_head

    # ── Extract weight scales and bias for bf16 epilogue ──
    model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer0_attn = model_obj.model.layers[0].self_attn
    weight_scale_all = layer0_attn.qkv_proj.weight_scale.detach().cpu().float().numpy().flatten()
    k_weight_scale = weight_scale_all[q_dim: q_dim + kv_dim].tolist()

    bias_tensor = layer0_attn.qkv_proj.bias
    k_bias_list = None
    if bias_tensor is not None:
        bias_np = bias_tensor.detach().cpu().float().numpy().flatten()
        k_bias_list = bias_np[q_dim: q_dim + kv_dim].tolist()

    # ── Hook _real_kernel[0] to capture QKV output ──
    raw_qkv_outputs = []
    _hook_enabled = [False]
    _smm_count = [0]
    _orig_real_kernel = cap_mod._real_kernel[0]

    def _hook_kernel(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
        result = _orig_real_kernel(a, b, scale_a, scale_b, out_dtype, bias)
        _smm_count[0] += 1
        if _hook_enabled[0]:
            calls_per_fwd = n_layers * 4
            idx = (_smm_count[0] - 1)
            proj_idx = idx % 4
            layer = (idx % calls_per_fwd) // 4
            if layer == 0 and proj_idx == 0:
                torch.cuda.synchronize()
                raw_qkv_outputs.append(result.detach().cpu().float().clone())
        return result

    cap_mod._real_kernel[0] = _hook_kernel

    # ── Warmup ──
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=4)

    # ── Run inference ──
    _hook_enabled[0] = True
    _smm_count[0] = 0
    chat_r = server.chat(prompt="What is 2+2?", max_tokens=8, temperature=0.0)
    _hook_enabled[0] = False
    cap_mod._real_kernel[0] = _orig_real_kernel

    request_id = chat_r["request_id"]
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]
    audit_pos = n_prompt - 1

    # ── Get QKV output K slice (ground truth) ──
    assert raw_qkv_outputs, "No QKV output captured"
    qkv_out = raw_qkv_outputs[0].numpy()  # [n_prompt, total_out]
    k_from_qkv = qkv_out[:, q_dim: q_dim + kv_dim]  # [n_prompt, kv_dim]

    # ── Get shell K from audit with captured x_attn ──
    audit_json = json.loads(server.audit(
        request_id=request_id,
        token_index=audit_pos,
        layer_indices=[0],
        binary=False,
        use_captured_x_attn=True,
        deep_prefix=True,
    ))
    prefix_shells = audit_json.get("prefix_shell_openings", [])
    shell_opening = audit_json.get("shell_opening", {})
    shell_layers = shell_opening.get("layers", [])

    # ── Compare for every prefix position ──
    print(f"  n_prompt={n_prompt}, prefix_shells={len(prefix_shells)}")
    max_linf = 0.0
    failures = []

    for t in range(n_prompt):
        # Get shell k_acc and scale for this position
        if t < len(prefix_shells):
            ps_layers = prefix_shells[t].get("layers", [])
            if not ps_layers:
                continue
            k_acc = ps_layers[0].get("k")
            scale_x = ps_layers[0].get("scale_x_attn")
        elif t == audit_pos and shell_layers:
            k_acc = shell_layers[0].get("k")
            scale_x = shell_layers[0].get("scale_x_attn")
        else:
            continue

        if k_acc is None or scale_x is None:
            continue

        # Compute shell bf16 K
        shell_k = np.array(
            verilm_rs.cutlass_epilogue_bf16(k_acc, k_weight_scale, float(scale_x), k_bias_list),
            dtype=np.float32,
        )

        # GPU K from QKV output
        gpu_k = k_from_qkv[t].astype(np.float32)

        linf = np.abs(gpu_k - shell_k).max()
        max_linf = max(max_linf, linf)

        status = "PASS" if linf == 0.0 else f"FAIL L-inf={linf:.6f}"
        print(f"  pos={t}: {status}")

        if linf > 0.0:
            failures.append((t, linf))

    print(f"\n  max L-inf across all positions: {max_linf:.6f}")

    if failures:
        raise AssertionError(
            f"{model_id}: shell K from captured x_attn does NOT match GPU QKV output K. "
            f"Failing positions: {[(t, f'{d:.6f}') for t, d in failures]}. "
            f"Max L-inf: {max_linf:.6f}"
        )

    print(f"\n  PASS: shell K == GPU QKV output K for all {n_prompt} positions")
    return {"model": model_id, "n_prompt": n_prompt, "max_linf": max_linf}


@app.local_entrypoint()
def main():
    result = check_qwen_shell_exact.remote()
    print(f"\nResult: {result}")
