"""
Per-head boundary simulation: measure how per-head scale_a would change
the honest corridor and adversarial float-space room.

Does NOT change the protocol — uses existing audit payloads and simulates
what the corridor would look like under per-head requantization.

Reports:
  - Current per-tensor corridor (L-inf in INT8)
  - Simulated per-head corridor (L-inf in INT8, may be worse)
  - Float-space adversarial room under both strategies (should be better)
  - Per-head scale distribution (how much do heads vary?)

Usage:
    modal run --detach redteam/modal/simulate_perhead_boundary.py
"""

import os

import modal

app = modal.App("verilm-perhead-boundary-sim")

VLLM_SPEC = os.environ.get("VERILM_VLLM_SPEC", "vllm==0.18.0")
TORCH_SPEC = os.environ.get("VERILM_TORCH_SPEC", "torch")
TRANSFORMERS_SPEC = os.environ.get("VERILM_TRANSFORMERS_SPEC", "transformers<5")
COMPRESSED_TENSORS_SPEC = os.environ.get(
    "VERILM_COMPRESSED_TENSORS_SPEC", "compressed-tensors==0.13.0"
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
    """Extract per-channel weight scales for faithful corridor measurement."""
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


def _run_simulation(model_id: str):
    import hashlib
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import numpy as np
    import verilm_rs
    import vllm
    import torch
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    buf = get_capture_buffer()

    print(f"\n{'='*70}")
    print(f"Per-head boundary simulation: {model_id}")
    print(f"{'='*70}")
    print(f"Runtime: vllm={vllm.__version__} torch={torch.__version__}")

    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    print(f"Attention: backend={server._attn_backend} dtype={server._attn_dtype}")

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

    n_q_heads = cfg["n_q_heads"]
    d_head = cfg["d_head"]

    all_results = []

    for wl in WORKLOADS:
        print(f"\n--- {wl['name']} (max_tokens={wl['max_tokens']}) ---")

        chat_r = server.chat(prompt=wl["prompt"], max_tokens=wl["max_tokens"], temperature=0.0)
        n_tokens = chat_r["n_tokens"]
        request_id = chat_r["request_id"]
        n_prompt = chat_r["commitment"]["n_prompt_tokens"]

        # Sample positions
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

            report_json = verilm_rs.simulate_boundary_strategies(
                audit_binary, key_json, scale_overrides_json,
            )
            report = json.loads(report_json)

            # Print summary for this position
            print(f"\n  pos={pos:4d} (n_kv={pos+1})")
            print(f"    L-inf  tensor={report['global_linf_tensor']:2d}  per_head={report['global_linf_per_head']:2d}")
            print(f"    Float room ratio: worst={report['worst_float_room_ratio']:.4f}  mean={report['mean_float_room_ratio']:.4f}")

            # Per-layer detail: show worst 5 layers by float_room_ratio
            entries = report["entries"]
            if entries:
                # Scale distribution across heads for worst layer
                worst_entry = max(entries, key=lambda e: e["linf_tensor"])
                scales = worst_entry["scale_a_per_head"]
                tensor_scale = worst_entry["scale_a_tensor"]
                ratios = [s / tensor_scale if tensor_scale > 1e-30 else 1.0 for s in scales]
                print(f"    Worst layer {worst_entry['layer']}: tensor_scale={tensor_scale:.6f}")
                print(f"      head scale/tensor ratios: min={min(ratios):.4f} max={max(ratios):.4f} mean={np.mean(ratios):.4f}")
                print(f"      per-head L-inf: {worst_entry['linf_per_head_detail']}")

            all_results.append({
                "model": model_id,
                "workload": wl["name"],
                "token_position": pos,
                "n_tokens": n_tokens,
                "report": report,
            })

    # Final summary
    print(f"\n{'='*70}")
    print(f"BOUNDARY SIMULATION SUMMARY: {model_id}")
    print(f"{'='*70}")

    if all_results:
        all_tensor_linf = [r["report"]["global_linf_tensor"] for r in all_results]
        all_perhead_linf = [r["report"]["global_linf_per_head"] for r in all_results]
        all_worst_ratio = [r["report"]["worst_float_room_ratio"] for r in all_results]
        all_mean_ratio = [r["report"]["mean_float_room_ratio"] for r in all_results]

        print(f"  Global max L-inf  tensor: {max(all_tensor_linf)}")
        print(f"  Global max L-inf  per-head: {max(all_perhead_linf)}")
        print(f"  Float room ratio  worst: {max(all_worst_ratio):.4f}")
        print(f"  Float room ratio  mean (across all): {np.mean(all_mean_ratio):.4f}")
        print()

        # Collect all per-head scale ratios across all layers/positions
        all_head_ratios = []
        for r in all_results:
            for entry in r["report"]["entries"]:
                ts = entry["scale_a_tensor"]
                if ts > 1e-30:
                    for hs in entry["scale_a_per_head"]:
                        all_head_ratios.append(hs / ts)

        if all_head_ratios:
            arr = np.array(all_head_ratios)
            print(f"  Head scale / tensor scale distribution:")
            print(f"    min={arr.min():.4f}  p25={np.percentile(arr, 25):.4f}  "
                  f"median={np.median(arr):.4f}  p75={np.percentile(arr, 75):.4f}  "
                  f"max={arr.max():.4f}")
            print(f"    If max ratio ≈ 1.0 → per-head won't help (one head dominates)")
            print(f"    If max ratio < 0.5 → per-head could halve adversarial room")

        # Per-workload/position table
        print(f"\n  Per-position detail:")
        for r in all_results:
            rp = r["report"]
            print(
                f"    {r['workload']:>10s} pos={r['token_position']:4d}  "
                f"tensor={rp['global_linf_tensor']:2d}  "
                f"perhead={rp['global_linf_per_head']:2d}  "
                f"room_ratio={rp['worst_float_room_ratio']:.4f}"
            )

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
    with open("/tmp/boundary/qwen_boundary_sim.json", "w") as f:
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
    with open("/tmp/boundary/llama_boundary_sim.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.local_entrypoint()
def main():
    import json
    import os

    models_env = os.environ.get("BOUNDARY_MODELS", "qwen,llama")
    run_qwen = "qwen" in models_env
    run_llama = "llama" in models_env

    print("Launching per-head boundary simulation...")
    print(f"  Models: qwen={run_qwen}, llama={run_llama}")

    qwen_results = simulate_qwen.remote() if run_qwen else []
    llama_results = simulate_llama.remote() if run_llama else []

    print("\n" + "=" * 70)
    print("COMBINED BOUNDARY SIMULATION SUMMARY")
    print("=" * 70)

    for results in [qwen_results, llama_results]:
        if not results:
            continue
        model = results[0]["model"]
        print(f"\n{model}")
        all_tensor = [r["report"]["global_linf_tensor"] for r in results]
        all_perhead = [r["report"]["global_linf_per_head"] for r in results]
        all_ratio = [r["report"]["worst_float_room_ratio"] for r in results]
        print(f"  L-inf tensor: {max(all_tensor)}  per-head: {max(all_perhead)}")
        print(f"  Float room ratio worst: {max(all_ratio):.4f}")
