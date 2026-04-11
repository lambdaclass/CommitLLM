"""
INT16 boundary simulation: measure how INT16 retained `a` would change
the honest corridor in float space relative to the current INT8 boundary.

Does NOT change the protocol — uses existing audit payloads and simulates
what the honest mismatch would look like if the same tensor were retained
as INT16 with a tensor-wide scale.

Reports:
  - Current INT8 corridor (L-inf in retained INT8 units)
  - Simulated INT16 corridor (L-inf in retained INT16 units)
  - Honest float-space gap under each boundary
  - Ratio of INT16 / INT8 float-space gap (< 1 means improvement)

Usage:
    modal run --detach redteam/modal/simulate_int16_boundary.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/modal"))
try:
    from _pins import VERIFICATION, VLLM_SPEC, TORCH_SPEC, TRANSFORMERS_SPEC, COMPRESSED_TENSORS_SPEC
except ImportError:
    VERIFICATION = []
    VLLM_SPEC = TORCH_SPEC = TRANSFORMERS_SPEC = COMPRESSED_TENSORS_SPEC = ""

import modal

app = modal.App("verilm-int16-boundary-sim")

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
]


def _extract_weight_scales(llm, n_layers, cfg):
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
        scales["wq"].append(ws_np[:hidden_dim].tolist())
        scales["wk"].append(ws_np[hidden_dim:hidden_dim + kv_dim].tolist())
        scales["wv"].append(ws_np[hidden_dim + kv_dim:].tolist())

    return scales


def _run_simulation(model_id: str):
    import hashlib
    import json

    import numpy as np
    import torch
    import verilm_rs
    import vllm
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    buf = get_capture_buffer()

    print(f"\n{'=' * 70}")
    print(f"INT16 boundary simulation: {model_id}")
    print(f"{'=' * 70}")
    print(
        "Pinned runtime specs: "
        f"vllm={VLLM_SPEC}, transformers={TRANSFORMERS_SPEC}, "
        f"compressed_tensors={COMPRESSED_TENSORS_SPEC}, torch={TORCH_SPEC}"
    )

    llm = LLM(
        model=model_id,
        dtype="auto",
        max_model_len=4096,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    print(
        f"Runtime: vllm={vllm.__version__} torch={torch.__version__} "
        f"backend={server._attn_backend} dtype={server._attn_dtype}"
    )

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
    scale_overrides_json = json.dumps(_extract_weight_scales(llm, n_layers, cfg_dict))

    all_results = []

    for wl in WORKLOADS:
        print(f"\n--- {wl['name']} (max_tokens={wl['max_tokens']}) ---")
        chat_r = server.chat(prompt=wl["prompt"], max_tokens=wl["max_tokens"], temperature=0.0)
        n_tokens = chat_r["n_tokens"]
        request_id = chat_r["request_id"]
        n_prompt = chat_r["commitment"]["n_prompt_tokens"]

        positions = sorted(set([
            n_prompt - 1,
            n_prompt,
            max(n_prompt, n_tokens // 2),
            max(n_prompt, n_tokens - 1),
        ]))

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

            report = json.loads(
                verilm_rs.simulate_int16_boundary(
                    audit_binary,
                    key_json,
                    scale_overrides_json,
                )
            )

            print(f"\n  pos={pos:4d} (n_kv={pos + 1})")
            print(
                f"    L-inf  int8={report['global_linf_int8']:4d}  "
                f"int16={report['global_linf_int16']:6d}"
            )
            print(
                f"    Float honest gap  int8={report['global_float_linf_int8']:.6f}  "
                f"int16={report['global_float_linf_int16']:.6f}"
            )
            print(
                f"    Ratios: float_gap={report['worst_float_linf_ratio']:.4f}  "
                f"scale={report['worst_scale_ratio']:.4f}"
            )

            all_results.append({
                "model": model_id,
                "workload": wl["name"],
                "token_position": pos,
                "n_tokens": n_tokens,
                "report": report,
            })

    print(f"\n{'=' * 70}")
    print(f"INT16 BOUNDARY SUMMARY: {model_id}")
    print(f"{'=' * 70}")
    if all_results:
        all_int8 = [r["report"]["global_float_linf_int8"] for r in all_results]
        all_int16 = [r["report"]["global_float_linf_int16"] for r in all_results]
        all_ratio = [r["report"]["worst_float_linf_ratio"] for r in all_results]
        all_scale = [r["report"]["worst_scale_ratio"] for r in all_results]
        print(f"  Global float-gap int8:  {max(all_int8):.6f}")
        print(f"  Global float-gap int16: {max(all_int16):.6f}")
        print(f"  Worst float-gap ratio:  {max(all_ratio):.4f}")
        print(f"  Worst scale ratio:      {max(all_scale):.4f}")
        print(f"  Mean float-gap ratio:   {np.mean(all_ratio):.4f}")

    return all_results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/tmp/boundary": modal.Volume.from_name("boundary-sim-results", create_if_missing=True)},
)
def simulate_qwen():
    import json
    results = _run_simulation("neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8")
    with open("/tmp/boundary/qwen_int16_boundary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/tmp/boundary": modal.Volume.from_name("boundary-sim-results", create_if_missing=True)},
)
def simulate_llama():
    import json
    results = _run_simulation("neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8")
    with open("/tmp/boundary/llama_int16_boundary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.local_entrypoint()
def main():
    models_env = os.environ.get("BOUNDARY_MODELS", "qwen,llama")
    run_qwen = "qwen" in models_env
    run_llama = "llama" in models_env

    print("Launching INT16 boundary simulation...")
    print(f"  Models: qwen={run_qwen}, llama={run_llama}")

    qwen_results = simulate_qwen.remote() if run_qwen else []
    llama_results = simulate_llama.remote() if run_llama else []

    print("\n" + "=" * 70)
    print("COMBINED INT16 BOUNDARY SUMMARY")
    print("=" * 70)

    for results in [qwen_results, llama_results]:
        if not results:
            continue
        model = results[0]["model"]
        all_int8 = [r["report"]["global_float_linf_int8"] for r in results]
        all_int16 = [r["report"]["global_float_linf_int16"] for r in results]
        all_ratio = [r["report"]["worst_float_linf_ratio"] for r in results]
        print(f"\n{model}")
        print(f"  Global float-gap int8:  {max(all_int8):.6f}")
        print(f"  Global float-gap int16: {max(all_int16):.6f}")
        print(f"  Worst float-gap ratio:  {max(all_ratio):.4f}")
