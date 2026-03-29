"""
Measure the attention corridor: L-inf and agreement rates between GPU FP16
attention output and verifier f64 replay.

Runs two models, multiple workloads, multiple seeds. Saves structured JSON
artifacts alongside a human-readable summary.

Models:
  - neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8
  - neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8

Usage:
    modal run --detach scripts/modal/measure_corridor.py
"""

import modal

app = modal.App("verilm-measure-corridor")

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
        "VERILM_PACKED_COMMIT": "0",  # Use unpacked path for x_attn capture
    })
    .pip_install("vllm>=0.8", "torch", "numpy", "fastapi", "maturin")
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

WORKLOADS = [
    {"name": "short", "prompt": "What is 2+2?", "max_tokens": 32},
    {"name": "medium", "prompt": "Explain the theory of relativity in one paragraph.", "max_tokens": 64},
    {
        "name": "long",
        "prompt": (
            "You are a historian specializing in the Industrial Revolution. "
            "Write a detailed analysis of how steam power transformed manufacturing "
            "in 18th-century Britain, covering the transition from cottage industry "
            "to factory systems, the social consequences for workers, and the "
            "environmental impact of early coal-powered machinery. Include specific "
            "examples of key inventions and their inventors."
        ),
        "max_tokens": 128,
    },
    {
        "name": "extended",
        "prompt": (
            "You are a computer science professor. Explain in detail how modern "
            "transformer architectures work, starting from the attention mechanism, "
            "covering multi-head attention, positional encodings, layer normalization, "
            "feed-forward networks, and the training process. Compare the original "
            "Vaswani et al. architecture with recent developments like rotary "
            "position embeddings, grouped query attention, and KV caching for "
            "efficient inference. Discuss the computational complexity of each "
            "component and how hardware accelerators like GPUs and TPUs handle "
            "the matrix operations involved."
        ),
        "max_tokens": 512,
    },
    {
        "name": "long_context",
        "prompt": (
            "You are a senior software architect writing a comprehensive design "
            "document. Cover the following topics in depth with concrete examples: "
            "(1) microservices vs monolithic architecture trade-offs, including "
            "failure modes, deployment complexity, and data consistency; "
            "(2) event-driven systems with Kafka or similar message brokers, "
            "covering exactly-once semantics, consumer group rebalancing, and "
            "schema evolution; (3) database sharding strategies including "
            "consistent hashing, range partitioning, and cross-shard transactions; "
            "(4) observability infrastructure including structured logging, "
            "distributed tracing with OpenTelemetry, and SLO-based alerting; "
            "(5) CI/CD pipeline design for zero-downtime deployments with "
            "canary releases and automated rollbacks."
        ),
        "max_tokens": 1024,
    },
]

# Vary decode paths: different temperature/top_k combos exercise different
# softmax distributions and thus different attention corridors.
# The batch seed is generated fresh per request inside server.chat().
DECODE_CONFIGS = [
    {"name": "greedy", "temperature": 0.0},
    {"name": "warm", "temperature": 0.8},
    {"name": "top_k50", "temperature": 1.0, "top_k": 50},
]


def _extract_weight_scales(llm, n_layers, cfg):
    """Extract per-channel weight scales from the vLLM model.

    W8A8 models have per-channel (per output feature) weight scales stored
    as `weight_scale` tensors on each linear layer. The fused QKV projection
    has shape [hidden_dim + 2*kv_dim], which we split into Q, K, V portions.

    Returns a dict suitable for JSON serialization as CorridorScaleOverrides.
    """
    import numpy as np

    # Access the inner model through vLLM's execution pipeline.
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    hidden_dim = cfg["hidden_dim"]
    kv_dim = cfg["kv_dim"]

    scales = {"wq": [], "wk": [], "wv": []}

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        qkv_proj = layer.self_attn.qkv_proj

        # weight_scale: per-channel scales, shape [out_features]
        # For fused QKV: out_features = hidden_dim + 2 * kv_dim
        ws = qkv_proj.weight_scale
        if ws is None:
            raise RuntimeError(f"layer {layer_idx}: no weight_scale on qkv_proj")

        ws_np = ws.detach().cpu().float().numpy().flatten()
        expected_len = hidden_dim + 2 * kv_dim
        if ws_np.shape[0] != expected_len:
            raise RuntimeError(
                f"layer {layer_idx}: weight_scale shape {ws_np.shape} != expected {expected_len}"
            )

        # Split: Q first, then K, then V (vLLM QKVParallelLinear layout)
        wq = ws_np[:hidden_dim]
        wk = ws_np[hidden_dim:hidden_dim + kv_dim]
        wv = ws_np[hidden_dim + kv_dim:]

        scales["wq"].append(wq.tolist())
        scales["wk"].append(wk.tolist())
        scales["wv"].append(wv.tolist())

    print(f"  Extracted per-channel weight scales: {n_layers} layers, "
          f"wq[{hidden_dim}] wk[{kv_dim}] wv[{kv_dim}]")

    # Spot-check: print scale range for layer 0
    wq0 = np.array(scales["wq"][0])
    print(f"  Layer 0 wq scale range: [{wq0.min():.6f}, {wq0.max():.6f}]")

    return scales


def _diagnose_model_scales(llm, n_layers):
    """Targeted diagnostic: verify scale formula and QKV layout.

    Prints the exact values needed to identify whether the dequantization
    is acc*sx*sw, acc*sx/sw, or something else, and whether the fused
    QKV row layout is [Q|K|V] as assumed.
    """
    import torch
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer0 = model.model.layers[0]
    qkv = layer0.self_attn.qkv_proj

    print(f"\n{'='*70}")
    print("DIAGNOSTIC: Scale formula + QKV layout verification")
    print(f"{'='*70}")

    # 1. All scale-related attributes on qkv_proj
    scale_attrs = [a for a in dir(qkv) if "scale" in a.lower()]
    print(f"\n  qkv_proj scale attrs: {scale_attrs}")
    for attr in scale_attrs:
        val = getattr(qkv, attr, None)
        if val is not None and hasattr(val, "shape"):
            flat = val.detach().cpu().float().flatten()
            print(f"    {attr}: shape={tuple(val.shape)}, dtype={val.dtype}, "
                  f"[:8]={flat[:8].tolist()}")
        elif val is not None:
            print(f"    {attr}: {val}")

    # 2. Weight shape and dtype
    w = qkv.weight
    ws = qkv.weight_scale.detach().cpu().float().flatten()
    print(f"\n  weight: shape={tuple(w.shape)}, dtype={w.dtype}")
    print(f"  weight_scale: shape={tuple(qkv.weight_scale.shape)}, len={len(ws)}")

    # 3. Verify QKV layout and dequant formula
    print(f"\n  --- QKV layout + dequant formula ---")
    print(f"  weight shape: {tuple(w.shape)} (rows x cols)")
    print(f"  weight_scale shape: {tuple(qkv.weight_scale.shape)}")

    # Determine out_features: for linear y = x @ W.T, W is [out, in]
    # weight_scale should be [out_features]
    out_features = w.shape[0]
    in_features = w.shape[1]
    from verilm import capture as cap
    hidden_dim = cap._hidden_dim if hasattr(cap, '_hidden_dim') else 3584
    kv_dim = cap._kv_dim if hasattr(cap, '_kv_dim') else 512
    print(f"  out_features={out_features}, in_features={in_features}")
    print(f"  expected: out={hidden_dim}+{kv_dim}+{kv_dim}={hidden_dim+2*kv_dim}")

    leftover = out_features - hidden_dim - 2*kv_dim
    if leftover != 0:
        print(f"  WARNING: out_features mismatch by {leftover}!")

    # Oracle: x_probe(ones) @ W.T gives raw INT-like accumulator [1, out_features]
    w_f32 = w.detach().float()
    x_probe = torch.ones(1, in_features, dtype=torch.float32, device=w.device)
    raw = (x_probe @ w_f32.T).squeeze(0)  # [out_features]
    print(f"  raw oracle shape: {tuple(raw.shape)}")

    # Check section norms to verify [Q|K|V] layout
    sec_q = raw[:hidden_dim]
    sec_k = raw[hidden_dim:hidden_dim+kv_dim]
    sec_v = raw[hidden_dim+kv_dim:hidden_dim+2*kv_dim]
    print(f"  section norms: Q={sec_q.norm():.2f}, K={sec_k.norm():.2f}, V={sec_v.norm():.2f}")

    # Test dequant formulas on first 8 channels
    ws8 = ws[:8].to(raw.device)
    raw8 = raw[:8]
    print(f"\n  --- Dequant formula check (first 8 Q channels, scale_x=1.0) ---")
    print(f"  raw[:8]:    {raw8.tolist()}")
    print(f"  ws[:8]:     {ws8.tolist()}")
    print(f"  raw*ws:     {(raw8 * ws8).tolist()}")
    print(f"  raw/ws:     {(raw8 / ws8).tolist()}")

    # 5. Check if there's an input_scale (static quantization)
    if hasattr(qkv, "input_scale") and qkv.input_scale is not None:
        print(f"\n  input_scale: {qkv.input_scale.detach().cpu().item()}")
    else:
        print(f"\n  input_scale: None (dynamic per-tensor)")

    print(f"\n{'='*70}\n")


def _measure_one(server, wl, dc, full_layers, key_json, model_id, corridor_mode, buf, scale_overrides_json):
    """Run one workload/decode_config and return results for all token positions."""
    import json

    params_kw = {"max_tokens": wl["max_tokens"], "temperature": dc["temperature"]}
    if "top_k" in dc:
        params_kw["top_k"] = dc["top_k"]

    results = []
    chat_r = server.chat(prompt=wl["prompt"], **params_kw)
    n_tokens = chat_r["n_tokens"]
    request_id = chat_r["request_id"]

    entry = server._audit_store.get(request_id)
    if entry is None:
        print(f"  WARN: no audit entry for {request_id}, skipping")
        return results
    entry["state"].deep_prefix = True

    positions = sorted(set([
        1,
        max(1, n_tokens // 2),
        max(1, n_tokens - 1),
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
            )
        except KeyError:
            chat_r2 = server.chat(prompt=wl["prompt"], **params_kw)
            entry2 = server._audit_store.get(chat_r2["request_id"])
            if entry2:
                entry2["state"].deep_prefix = True
            audit_binary = server.audit(
                request_id=chat_r2["request_id"],
                token_index=pos,
                layer_indices=full_layers,
                tier="full",
                binary=True,
            )

        import verilm_rs
        report_json = verilm_rs.measure_corridor(
            audit_binary, key_json, scale_overrides_json,
        )
        report = json.loads(report_json)

        result_entry = {
            "model": model_id,
            "corridor_mode": corridor_mode,
            "workload": wl["name"],
            "decode_config": dc["name"],
            "max_tokens": wl["max_tokens"],
            "n_tokens": n_tokens,
            "token_position": pos,
            "global_linf": report["global_linf"],
            "n_measurements": len(report["measurements"]),
            "per_layer_max_linf": report["per_layer_max_linf"],
            "measurements": report["measurements"],
        }
        results.append(result_entry)

        agg_eq = [m["frac_eq"] for m in report["measurements"]]
        agg_le1 = [m["frac_le_1"] for m in report["measurements"]]
        avg_eq = sum(agg_eq) / len(agg_eq) if agg_eq else 0
        avg_le1 = sum(agg_le1) / len(agg_le1) if agg_le1 else 0
        print(
            f"  [{corridor_mode:>8}] pos={pos:4d}  L-inf={report['global_linf']:3d}  "
            f"frac_eq={avg_eq:.4f}  frac≤1={avg_le1:.4f}"
        )

    return results


def _run_model(model_id: str):
    import hashlib
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    buf = get_capture_buffer()

    print(f"\n{'='*70}")
    print(f"Model: {model_id}")
    print(f"{'='*70}")

    llm = LLM(
        model=model_id, dtype="auto", max_model_len=4096,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir

    # Warmup
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=8)

    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, key_seed)
    full_layers = list(range(n_layers))

    # Extract per-channel weight scales from the model for faithful replay.
    cfg = json.loads(key_json)["config"]
    cfg_dict = {
        "hidden_dim": cfg["hidden_dim"],
        "kv_dim": cfg["n_kv_heads"] * cfg["d_head"],
    }
    weight_scales = _extract_weight_scales(llm, n_layers, cfg_dict)
    scale_overrides_json = json.dumps(weight_scales)

    # ── Diagnostic: inspect model scale attributes ──
    _diagnose_model_scales(llm, n_layers)

    all_results = []

    # ── Phase 1: Precision corridor (GPU x_attn → accurate QKV accumulators) ──
    print(f"\n{'='*70}")
    print("PHASE 1: Precision corridor (captured x_attn + per-channel scales)")
    print(f"{'='*70}")
    buf._capture_x_attn = True
    for wl in WORKLOADS[:3]:  # short, medium, long only
        for dc in DECODE_CONFIGS[:1]:  # greedy only for precision
            label = f"{wl['name']}/{dc['name']}"
            print(f"\n--- {label} (max_tokens={wl['max_tokens']}) ---")
            has_xa = hasattr(buf, '_capture_x_attn') and buf._capture_x_attn
            print(f"  x_attn capture: {has_xa}")
            results = _measure_one(
                server, wl, dc, full_layers, key_json, model_id,
                "precision", buf, scale_overrides_json,
            )
            all_results.extend(results)

    # ── Phase 2: Verifier replay corridor (bridge-derived QKV accumulators) ──
    print(f"\n{'='*70}")
    print("PHASE 2: Verifier replay corridor (bridge-derived + per-channel scales)")
    print(f"{'='*70}")
    buf._capture_x_attn = False
    for wl in WORKLOADS[:3]:  # short, medium, long
        for dc in DECODE_CONFIGS[:1]:  # greedy only
            label = f"{wl['name']}/{dc['name']}"
            print(f"\n--- {label} (max_tokens={wl['max_tokens']}) ---")
            results = _measure_one(
                server, wl, dc, full_layers, key_json, model_id,
                "replay", buf, scale_overrides_json,
            )
            all_results.extend(results)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    for mode in ["precision", "replay"]:
        mode_results = [r for r in all_results if r["corridor_mode"] == mode]
        if mode_results:
            max_linf = max(r["global_linf"] for r in mode_results)
            print(f"\n  {mode:>10} corridor: max L-inf = {max_linf}")
            for r in mode_results:
                print(
                    f"    {r['workload']:>8s} pos={r['token_position']:4d}  "
                    f"L-inf={r['global_linf']:3d}"
                )

    # ── Save structured artifacts ──
    artifact_path = "/tmp/corridor_results.json"
    with open(artifact_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nStructured results saved to {artifact_path}")
    print(f"Total measurements: {sum(r['n_measurements'] for r in all_results)}")

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
    with open("/tmp/corridor/qwen_corridor.json", "w") as f:
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
    with open("/tmp/corridor/llama_corridor.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.local_entrypoint()
def main():
    import json

    print("Launching corridor measurement on two models...")
    print("Results will be saved to Modal volume 'corridor-results'.\n")

    qwen_results = measure_qwen.remote()
    llama_results = measure_llama.remote()

    # Combined summary
    print("\n" + "=" * 70)
    print("COMBINED SUMMARY")
    print("=" * 70)

    for results in [qwen_results, llama_results]:
        if not results:
            continue
        model = results[0]["model"]
        max_linf = max(r["global_linf"] for r in results)
        print(f"\n{model}")
        print(f"  Global max L-inf: {max_linf}")
        for r in results:
            print(
                f"  {r['workload']:>10s}/{r['decode_config']:<8s} pos={r['token_position']:4d}  "
                f"L-inf={r['global_linf']:2d}"
            )
