"""
Precision ladder: f64 vs f32 vs fp16+f32 vs bf16+f32 attention replay.

Runs the same workloads through the vLLM sidecar, then measures the
corridor gap at each precision level:

  f64       — current verifier default (over-precise vs GPU)
  f32       — tests whether f32 accumulation alone closes the gap
  fp16_f32  — fp16 input truncation + f32 accum (closest to GPU SDPA)
  bf16_f32  — bf16 input truncation + f32 accum (for bf16 models)

Interpretation:
  If f32 alone closes the gap → verifier is over-precise, switch to f32.
  If fp16_f32 closes it further → input truncation is the main source.
  If neither helps → gap is from kernel semantics (fused ops, etc.).

Usage:
    modal run --detach redteam/modal/measure_corridor_precision.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/modal"))
from _pins import VERIFICATION, VLLM_SPEC, TORCH_SPEC, TRANSFORMERS_SPEC, COMPRESSED_TENSORS_SPEC

import modal

app = modal.App("verilm-corridor-precision-ab")

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
        "lean", "article", "docs", "paper", "research", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

WORKLOADS = [
    {"name": "short", "prompt": "What is 2+2?", "max_tokens": 32},
    {"name": "medium", "prompt": "Explain the theory of relativity in one paragraph.", "max_tokens": 64},
    {
        "name": "long",
        "prompt": (
            "You are a historian specializing in the Industrial Revolution. "
            "Write a detailed analysis of how steam power transformed manufacturing "
            "in 18th-century Britain."
        ),
        "max_tokens": 128,
    },
    {
        "name": "extended",
        "prompt": (
            "You are a computer science professor. Explain in detail how modern "
            "transformer architectures work, starting from the attention mechanism, "
            "covering multi-head attention, positional encodings, layer normalization, "
            "feed-forward networks, and the training process."
        ),
        "max_tokens": 512,
    },
]


def _extract_weight_scales(llm, n_layers, cfg):
    """Extract per-channel weight scales for faithful corridor measurement."""
    import numpy as np

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    hidden_dim = cfg["hidden_dim"]
    kv_dim = cfg["kv_dim"]
    scales = {"wq": [], "wk": [], "wv": []}

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        qkv_proj = layer.self_attn.qkv_proj
        ws = qkv_proj.weight_scale
        if ws is None:
            raise RuntimeError(f"layer {layer_idx}: no weight_scale on qkv_proj")

        ws_np = ws.detach().cpu().float().numpy().flatten()
        expected_len = hidden_dim + 2 * kv_dim
        if ws_np.shape[0] != expected_len:
            raise RuntimeError(
                f"layer {layer_idx}: weight_scale shape {ws_np.shape} != expected {expected_len}"
            )

        scales["wq"].append(ws_np[:hidden_dim].tolist())
        scales["wk"].append(ws_np[hidden_dim:hidden_dim + kv_dim].tolist())
        scales["wv"].append(ws_np[hidden_dim + kv_dim:].tolist())

    return scales


PRECISIONS = ["f64", "f32", "fp16_f32", "bf16_f32"]


def _measure_ab(server, wl, full_layers, key_json, model_id, buf, scale_overrides_json):
    """Run one workload, measure corridor across all replay precisions."""
    import json
    import verilm_rs

    chat_r = server.chat(prompt=wl["prompt"], max_tokens=wl["max_tokens"], temperature=0.0)
    n_tokens = chat_r["n_tokens"]
    request_id = chat_r["request_id"]
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]

    # Sample positions: first-gen, first-decode, mid, late
    positions = sorted(set([
        n_prompt - 1,
        n_prompt,
        max(n_prompt, n_tokens // 2),
        max(n_prompt, n_tokens - 1),
    ]))

    results = []
    for pos in positions:
        if pos >= n_tokens:
            continue
        try:
            audit_binary = server.audit(
                request_id=request_id,
                token_index=pos,
                layer_indices=full_layers,
                tier="full",
                binary=True,
                use_captured_x_attn=True,
            )
        except KeyError:
            chat_r2 = server.chat(prompt=wl["prompt"], max_tokens=wl["max_tokens"], temperature=0.0)
            audit_binary = server.audit(
                request_id=chat_r2["request_id"],
                token_index=pos,
                layer_indices=full_layers,
                tier="full",
                binary=True,
                use_captured_x_attn=True,
            )

        # Measure all precision levels
        reports = {}
        for prec in PRECISIONS:
            report_json = verilm_rs.measure_corridor_precision(
                audit_binary, key_json, prec, scale_overrides_json,
            )
            reports[prec] = json.loads(report_json)

        legacy_json = verilm_rs.measure_corridor_committed_kv(
            audit_binary, key_json, scale_overrides_json,
        )
        legacy = json.loads(legacy_json)
        if legacy != reports["f64"]:
            raise RuntimeError(
                "precision dispatch mismatch on real audit: "
                f"workload={wl['name']} pos={pos} "
                f"legacy_linf={legacy['global_linf']} "
                f"f64_linf={reports['f64']['global_linf']}"
            )

        result = {
            "model": model_id,
            "workload": wl["name"],
            "n_tokens": n_tokens,
            "n_prompt_tokens": n_prompt,
            "token_position": pos,
        }
        for prec in PRECISIONS:
            r = reports[prec]
            result[prec] = {
                "global_linf": r["global_linf"],
                "per_layer_max_linf": r["per_layer_max_linf"],
                "measurements": r["measurements"],
            }
        results.append(result)

        # Summary line
        parts = []
        for prec in PRECISIONS:
            r = reports[prec]
            eq = sum(m["frac_eq"] for m in r["measurements"]) / max(len(r["measurements"]), 1)
            parts.append(f"{prec}: L-inf={r['global_linf']:2d} eq={eq:.4f}")
        print(f"  pos={pos:4d}  " + "  ".join(parts))

    return results


def _run_model(model_id: str):
    import hashlib
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    import verilm_rs
    import vllm
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    buf = get_capture_buffer()

    print(f"\n{'='*70}")
    print(f"Model: {model_id}")
    print(f"{'='*70}")
    print(
        "Pinned runtime specs: "
        f"vllm={VLLM_SPEC}, transformers={TRANSFORMERS_SPEC}, "
        f"compressed_tensors={COMPRESSED_TENSORS_SPEC}, torch={TORCH_SPEC}"
    )
    print(f"Runtime: vllm={vllm.__version__} torch={torch.__version__}")

    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    print(
        f"Resolved attention runtime: backend={server._attn_backend} "
        f"dtype={server._attn_dtype}"
    )

    # Warmup
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
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

    print(f"\n{'='*70}")
    print(f"Precision ladder: {', '.join(PRECISIONS)}")
    print(f"{'='*70}")

    for wl in WORKLOADS:
        print(f"\n--- {wl['name']} (max_tokens={wl['max_tokens']}) ---")
        results = _measure_ab(
            server, wl, full_layers, key_json, model_id,
            buf, scale_overrides_json,
        )
        all_results.extend(results)

    # Summary
    print(f"\n{'='*70}")
    print("PRECISION LADDER SUMMARY")
    print(f"{'='*70}")

    if all_results:
        print(f"\n  Global max L-inf per precision:")
        for prec in PRECISIONS:
            max_linf = max(r[prec]["global_linf"] for r in all_results)
            print(f"    {prec:>10s}: {max_linf}")

        # Per-workload comparison
        print(f"\n  Per-workload/position:")
        for r in all_results:
            parts = [f"{prec}={r[prec]['global_linf']:2d}" for prec in PRECISIONS]
            print(
                f"    {r['workload']:>10s} pos={r['token_position']:4d}  "
                + "  ".join(parts)
            )

        # Per-layer comparison for worst f64 case
        worst = max(all_results, key=lambda r: r["f64"]["global_linf"])
        print(f"\n  Worst case ({worst['workload']} pos={worst['token_position']}):")
        for i in range(len(worst["f64"]["per_layer_max_linf"])):
            parts = [f"{prec}={worst[prec]['per_layer_max_linf'][i]:2d}" for prec in PRECISIONS]
            print(f"    layer {i:2d}: " + "  ".join(parts))

    return all_results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/tmp/corridor": modal.Volume.from_name("corridor-results", create_if_missing=True)},
)
def measure_qwen():
    import json
    results = _run_model("neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8")
    with open("/tmp/corridor/qwen_precision_ab.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/tmp/corridor": modal.Volume.from_name("corridor-results", create_if_missing=True)},
)
def measure_llama():
    import json
    results = _run_model("neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8")
    with open("/tmp/corridor/llama_precision_ab.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.local_entrypoint()
def main():
    import json
    import os

    models_env = os.environ.get("CORRIDOR_MODELS", "qwen,llama")
    run_qwen = "qwen" in models_env
    run_llama = "llama" in models_env

    print("Launching precision A/B measurement...")
    print(f"  Models: qwen={run_qwen}, llama={run_llama}")

    qwen_results = measure_qwen.remote() if run_qwen else []
    llama_results = measure_llama.remote() if run_llama else []

    print("\n" + "=" * 70)
    print("COMBINED PRECISION LADDER SUMMARY")
    print("=" * 70)

    for results in [qwen_results, llama_results]:
        if not results:
            continue
        model = results[0]["model"]
        print(f"\n{model}")
        for prec in PRECISIONS:
            max_linf = max(r[prec]["global_linf"] for r in results)
            print(f"  {prec:>10s}: max L-inf = {max_linf}")
        for r in results:
            parts = [f"{prec}={r[prec]['global_linf']:2d}" for prec in PRECISIONS]
            print(
                f"  {r['workload']:>10s} pos={r['token_position']:4d}  "
                + "  ".join(parts)
            )
