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

SEEDS = [42, 137, 2024]


def _run_model(model_id: str):
    import hashlib
    import json
    import os
    import time

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

    # Generate key
    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, key_seed)
    full_layers = list(range(n_layers))

    all_results = []

    for wl in WORKLOADS:
        for seed_val in SEEDS:
            label = f"{wl['name']}/seed={seed_val}"
            print(f"\n--- {label} (max_tokens={wl['max_tokens']}) ---")

            params_kw = {"max_tokens": wl["max_tokens"]}
            if seed_val != 42:
                params_kw["temperature"] = 0.8
                params_kw["seed"] = seed_val
            else:
                params_kw["temperature"] = 0.0

            # Generate with capture
            chat_r = server.chat(prompt=wl["prompt"], **params_kw)
            n_tokens = chat_r["n_tokens"]
            request_id = chat_r["request_id"]
            print(f"  Generated {n_tokens} tokens")

            # Enable deep_prefix on the audit state
            entry = server._audit_store.get(request_id)
            if entry is None:
                print(f"  WARN: no audit entry for {request_id}, skipping")
                continue
            entry["state"].deep_prefix = True

            # Pick token positions: token 1, mid, last
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
                    # Re-generate for this position (audit store is single-use per token)
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

                report_json = verilm_rs.measure_corridor(audit_binary, key_json)
                report = json.loads(report_json)

                result_entry = {
                    "model": model_id,
                    "workload": wl["name"],
                    "seed": seed_val,
                    "max_tokens": wl["max_tokens"],
                    "n_tokens": n_tokens,
                    "token_position": pos,
                    "global_linf": report["global_linf"],
                    "n_measurements": len(report["measurements"]),
                    "per_layer_max_linf": report["per_layer_max_linf"],
                    "measurements": report["measurements"],
                }
                all_results.append(result_entry)

                # Summary line
                agg_frac_eq = []
                agg_frac_le_1 = []
                agg_frac_le_2 = []
                for m in report["measurements"]:
                    agg_frac_eq.append(m["frac_eq"])
                    agg_frac_le_1.append(m["frac_le_1"])
                    agg_frac_le_2.append(m["frac_le_2"])

                avg_eq = sum(agg_frac_eq) / len(agg_frac_eq) if agg_frac_eq else 0
                avg_le1 = sum(agg_frac_le_1) / len(agg_frac_le_1) if agg_frac_le_1 else 0
                avg_le2 = sum(agg_frac_le_2) / len(agg_frac_le_2) if agg_frac_le_2 else 0

                print(
                    f"  pos={pos:4d}  L-inf={report['global_linf']:2d}  "
                    f"frac_eq={avg_eq:.4f}  frac≤1={avg_le1:.4f}  frac≤2={avg_le2:.4f}"
                )

    # ── Per-layer detail table ──
    print(f"\n{'='*70}")
    print("Per-layer detail (last workload, last seed, last position):")
    print(f"{'Layer':>6}  {'L-inf':>6}  {'mean':>7}  {'p95':>5}  {'frac_eq':>8}  {'frac≤1':>8}  {'frac≤2':>8}  Hist [0,1,2,3,4,5+]")
    if all_results:
        last = all_results[-1]
        for m in last["measurements"]:
            h = m["histogram"]
            print(
                f"  {m['layer']:4d}  {m['linf']:6d}  {m['mean_abs']:7.3f}  {m['p95_abs_diff']:5d}  "
                f"{m['frac_eq']:8.4f}  {m['frac_le_1']:8.4f}  {m['frac_le_2']:8.4f}  "
                f"{h}"
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
                f"  {r['workload']:>10s}/seed={r['seed']:4d} pos={r['token_position']:4d}  "
                f"L-inf={r['global_linf']:2d}"
            )
