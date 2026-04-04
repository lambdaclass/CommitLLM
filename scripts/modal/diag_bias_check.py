"""
Quick spot-check: does the QKV bias fix collapse L-inf from ~235-255 to
something plausible?

Tests 3 positions on the short workload only (Qwen W8A8):
  - 0                  (first committed prefill token)
  - n_prompt_tokens-1  (last prefill / first generated token)
  - n_prompt_tokens    (first decode token)

Usage:
    modal run --detach scripts/modal/diag_bias_check.py
"""

import os

import modal

app = modal.App("verilm-bias-check")

VLLM_SPEC = os.environ.get("VERILM_VLLM_SPEC", "vllm==0.18.0")
TORCH_SPEC = os.environ.get("VERILM_TORCH_SPEC", "torch")
TRANSFORMERS_SPEC = os.environ.get("VERILM_TRANSFORMERS_SPEC", "transformers<5")
COMPRESSED_TENSORS_SPEC = os.environ.get(
    "VERILM_COMPRESSED_TENSORS_SPEC", "compressed-tensors==0.9.3"
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
        ".git", "target", "scripts/__pycache__", "*.pdf",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)


def _extract_weight_scales(llm, n_layers, cfg):
    """Extract per-channel weight scales from the vLLM model."""
    import numpy as np

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


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
)
def check_bias():
    import hashlib
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    import torch
    import vllm
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    model_id = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
    buf = get_capture_buffer()

    print(f"\n{'='*70}")
    print(f"QKV Bias Fix Spot-Check")
    print(f"Model: {model_id}")
    print(f"{'='*70}")
    print(
        "Pinned runtime specs: "
        f"vllm={VLLM_SPEC}, transformers={TRANSFORMERS_SPEC}, "
        f"compressed_tensors={COMPRESSED_TENSORS_SPEC}, torch={TORCH_SPEC}"
    )

    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    print(
        f"Runtime: vllm={vllm.__version__} torch={torch.__version__} "
        f"backend={server._attn_backend} dtype={server._attn_dtype}"
    )

    # Warmup
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=8)

    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(server._model_dir, key_seed)
    full_layers = list(range(n_layers))

    cfg = json.loads(key_json)["config"]
    cfg_dict = {
        "hidden_dim": cfg["hidden_dim"],
        "kv_dim": cfg["n_kv_heads"] * cfg["d_head"],
    }
    weight_scales = _extract_weight_scales(llm, n_layers, cfg_dict)
    scale_overrides_json = json.dumps(weight_scales)

    # ── Short workload: 3 targeted positions ──
    print(f"\n--- Short workload: 'What is 2+2?' max_tokens=32 ---")
    chat_r = server.chat(prompt="What is 2+2?", max_tokens=32, temperature=0.0)
    n_tokens = chat_r["n_tokens"]
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]
    request_id = chat_r["request_id"]

    print(f"  n_tokens={n_tokens}, n_prompt_tokens={n_prompt}")

    positions = [0, n_prompt - 1, min(n_prompt, n_tokens - 1)]
    # Deduplicate while preserving order
    seen = set()
    positions = [p for p in positions if p < n_tokens and not (p in seen or seen.add(p))]

    print(f"  Testing positions: {positions}")
    print(f"  (0=first prefill, {n_prompt-1}=first gen, {n_prompt}=first decode)")
    print()

    results = []
    for pos in positions:
        audit_binary = server.audit(
            request_id=request_id,
            token_index=pos,
            layer_indices=full_layers,
            tier="full",
            binary=True,
            use_captured_x_attn=True,
        )

        report_json = verilm_rs.measure_corridor_committed_kv(
            audit_binary, key_json, scale_overrides_json,
        )
        report = json.loads(report_json)

        # Per-layer detail
        pl = report["per_layer_max_linf"]
        worst_layer = max(range(len(pl)), key=lambda i: pl[i]) if pl else -1

        agg_eq = [m["frac_eq"] for m in report["measurements"]]
        agg_le1 = [m["frac_le_1"] for m in report["measurements"]]
        avg_eq = sum(agg_eq) / len(agg_eq) if agg_eq else 0
        avg_le1 = sum(agg_le1) / len(agg_le1) if agg_le1 else 0

        label = {0: "first_prefill", n_prompt - 1: "first_gen"}.get(pos, f"decode_{pos}")
        print(
            f"  pos={pos:3d} ({label:>14s})  "
            f"L-inf={report['global_linf']:3d}  "
            f"frac_eq={avg_eq:.4f}  frac≤1={avg_le1:.4f}  "
            f"worst_layer={worst_layer}"
        )

        # Show per-layer histogram for worst 3 layers
        layer_items = [(i, pl[i]) for i in range(len(pl))]
        layer_items.sort(key=lambda x: -x[1])
        for li, linf in layer_items[:3]:
            m = [x for x in report["measurements"] if x["layer"] == li]
            if m:
                m = m[0]
                print(
                    f"      layer={li:2d}  linf={linf:3d}  "
                    f"hist={m['histogram']}  "
                    f"mean_abs={m['mean_abs']:.3f}"
                )

        results.append({
            "position": pos,
            "label": label,
            "global_linf": report["global_linf"],
            "avg_frac_eq": avg_eq,
            "avg_frac_le1": avg_le1,
            "per_layer_max_linf": pl,
            "measurements": report["measurements"],
        })

    # ── Verdict ──
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    max_linf = max(r["global_linf"] for r in results)
    print(f"  Max L-inf across all positions: {max_linf}")
    if max_linf <= 10:
        print("  ✓ BIAS FIX CONFIRMED — corridor is in plausible BF16-vs-f64 range")
        print("  → Proceed with committed-KV corridor measurement")
    elif max_linf <= 50:
        print("  ~ PARTIAL — corridor reduced but still elevated")
        print("  → Investigate remaining discrepancy")
    else:
        print("  ✗ STILL BROKEN — L-inf still catastrophic")
        print("  → Bias fix alone is not sufficient")

    return results


@app.local_entrypoint()
def main():
    results = check_bias.remote()
    print("\nDone.")
