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


def _measure_one(server, wl, dc, full_layers, key_json, model_id, corridor_mode, buf):
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
        report_json = verilm_rs.measure_corridor(audit_binary, key_json)
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

    all_results = []

    # ── Phase 1: Precision corridor (GPU x_attn → accurate QKV accumulators) ──
    print(f"\n{'='*70}")
    print("PHASE 1: Precision corridor (captured x_attn)")
    print(f"{'='*70}")
    buf._capture_x_attn = True
    for wl in WORKLOADS[:3]:  # short, medium, long only
        for dc in DECODE_CONFIGS[:1]:  # greedy only for precision
            label = f"{wl['name']}/{dc['name']}"
            print(f"\n--- {label} (max_tokens={wl['max_tokens']}) ---")
            has_xa = hasattr(buf, '_capture_x_attn') and buf._capture_x_attn
            print(f"  x_attn capture: {has_xa}")
            results = _measure_one(
                server, wl, dc, full_layers, key_json, model_id, "precision", buf,
            )
            all_results.extend(results)

    # ── Phase 2: Verifier replay corridor (bridge-derived QKV accumulators) ──
    print(f"\n{'='*70}")
    print("PHASE 2: Verifier replay corridor (bridge-derived)")
    print(f"{'='*70}")
    buf._capture_x_attn = False
    for wl in WORKLOADS[:3]:  # short, medium, long
        for dc in DECODE_CONFIGS[:1]:  # greedy only
            label = f"{wl['name']}/{dc['name']}"
            print(f"\n--- {label} (max_tokens={wl['max_tokens']}) ---")
            results = _measure_one(
                server, wl, dc, full_layers, key_json, model_id, "replay", buf,
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
