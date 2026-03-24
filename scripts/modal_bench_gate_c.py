"""
Benchmark Gate C: sync + buffer-protocol optimization checkpoint.

Measures the impact of:
  - Event-based CUDA sync (dedicated copy stream) vs global sync
  - Buffer-protocol tensor ingress (no .tobytes() copy)

Compares both sync modes (VERILM_SYNC_MODE=global and =event) with
the same model, prompt, and token count.

  1. Baseline latency (inference only, no capture)
  2. Online path: global sync mode
  3. Online path: event sync mode
  4. Routine audit open time
  5. Verify time
  6. Binary payload size

Usage:
    modal run --detach scripts/modal_bench_gate_c.py
"""

import modal

app = modal.App("verilm-bench-gate-c")

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"

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

PROMPT = "Explain the theory of relativity in one paragraph."
N_WARMUP = 3
N_ITERS = 10
MAX_TOKENS = 32


def _run_bench():
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

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    server = VerifiedInferenceServer(llm)

    n_layers = cap._n_layers
    model_dir = server._model_dir
    seed = hashlib.sha256(PROMPT.encode()).digest()

    # ── Warmup ──
    print(f"\nWarmup: {N_WARMUP} iterations...")
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    for _ in range(N_WARMUP):
        server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
    print("Warmup done.\n")

    def stats(times):
        times = sorted(times)
        n = len(times)
        return sum(times) / n, times[n // 2]

    # ── 1. Baseline: capture OFF ──
    print(f"1. Baseline (capture OFF): {N_ITERS} iterations...")
    buf.enabled = False
    buf.drain()
    baseline_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        llm.generate([PROMPT], params)
        baseline_times.append(time.monotonic() - t0)

    # ── 2. Global sync mode ──
    print(f"2. Online path (SYNC_MODE=global): {N_ITERS} iterations...")
    cap._capture_mode = "minimal"
    buf.enabled = True
    buf.set_sync_mode("global")
    global_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        global_times.append(time.monotonic() - t0)

    # ── 3. Event sync mode ──
    print(f"3. Online path (SYNC_MODE=event): {N_ITERS} iterations...")
    buf.set_sync_mode("event")
    event_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        event_times.append(time.monotonic() - t0)

    # ── 4. Audit open ──
    print(f"4. Full audit open (binary): {N_ITERS} iterations...")
    full_layers = list(range(n_layers))
    audit_open_times = []
    binary_data = None
    for i in range(N_ITERS):
        chat_r = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        t0 = time.monotonic()
        result = server.audit(
            request_id=chat_r["request_id"],
            token_index=0,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        audit_open_times.append(time.monotonic() - t0)
        binary_data = result

    n_tokens = chat_r["n_tokens"]

    # Also get JSON for verify
    chat_for_json = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
    audit_json = server.audit(
        request_id=chat_for_json["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=False,
    )

    # ── 5. Key generation + verify ──
    print(f"5. Verify (key-only): {N_ITERS} iterations...")
    key_t0 = time.monotonic()
    key_json = verilm_rs.generate_key(model_dir, seed)
    key_gen_time = time.monotonic() - key_t0

    verify_json_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        verilm_rs.verify_v4(audit_json, key_json)
        verify_json_times.append(time.monotonic() - t0)

    verify_binary_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        verilm_rs.verify_v4_binary(bytes(binary_data), key_json)
        verify_binary_times.append(time.monotonic() - t0)

    # ── Payload ──
    json_bytes = len(audit_json.encode("utf-8")) if isinstance(audit_json, str) else len(audit_json)
    binary_bytes = len(binary_data)

    # ── Report ──
    b_mean, b_med = stats(baseline_times)
    g_mean, g_med = stats(global_times)
    e_mean, e_med = stats(event_times)
    ao_mean, ao_med = stats(audit_open_times)
    vj_mean, vj_med = stats(verify_json_times)
    vb_mean, vb_med = stats(verify_binary_times)

    global_oh = ((g_mean - b_mean) / b_mean) * 100
    event_oh = ((e_mean - b_mean) / b_mean) * 100
    event_vs_global = ((e_mean - g_mean) / g_mean) * 100

    print(f"\n{'='*80}")
    print("BENCHMARK GATE C  —  Sync + Buffer-Protocol Optimization")
    print(f"{'='*80}")
    print(f"Model:        {MODEL_ID}")
    print(f"GPU:          A100-80GB")
    print(f"Tokens:       {n_tokens} generated ({MAX_TOKENS} max)")
    print(f"Layers:       {n_layers}")
    print(f"Iters:        {N_ITERS} (after {N_WARMUP} warmup)")

    print(f"\n--- Online Path Latency (ms) ---")
    print(f"{'Metric':<50} {'Mean':>10} {'Median':>10} {'OH%':>8}")
    print(f"{'-'*78}")
    print(f"{'1. Baseline (no capture)':<50} {b_mean*1000:>10.1f} {b_med*1000:>10.1f} {'—':>8}")
    print(f"{'2. Global sync (torch.cuda.synchronize)':<50} {g_mean*1000:>10.1f} {g_med*1000:>10.1f} {global_oh:>+7.1f}%")
    print(f"{'3. Event sync (dedicated copy stream)':<50} {e_mean*1000:>10.1f} {e_med*1000:>10.1f} {event_oh:>+7.1f}%")
    print(f"{'   Event vs global':<50} {'':>10} {'':>10} {event_vs_global:>+7.1f}%")

    print(f"\n--- Offline Latency (ms) ---")
    print(f"{'Metric':<50} {'Mean':>10} {'Median':>10}")
    print(f"{'-'*70}")
    print(f"{'4. Full audit open (binary)':<50} {ao_mean*1000:>10.1f} {ao_med*1000:>10.1f}")
    print(f"{'5a. Verify full (JSON)':<50} {vj_mean*1000:>10.1f} {vj_med*1000:>10.1f}")
    print(f"{'5b. Verify full (binary)':<50} {vb_mean*1000:>10.1f} {vb_med*1000:>10.1f}")
    print(f"{'   Key generation (one-time)':<50} {key_gen_time*1000:>10.1f}")

    print(f"\n--- Payload ---")
    print(f"{'Full audit (JSON)':<50} {json_bytes:>10,} bytes")
    print(f"{'Full audit (binary)':<50} {binary_bytes:>10,} bytes")
    print(f"{'Compression ratio':<50} {json_bytes / max(binary_bytes, 1):>10.1f}x")
    print(f"{'='*80}")

    return {
        "model": MODEL_ID,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "baseline_med_ms": b_med * 1000,
        "global_sync_med_ms": g_med * 1000,
        "event_sync_med_ms": e_med * 1000,
        "global_oh_pct": global_oh,
        "event_oh_pct": event_oh,
        "event_vs_global_pct": event_vs_global,
        "audit_open_med_ms": ao_med * 1000,
        "verify_json_med_ms": vj_med * 1000,
        "verify_binary_med_ms": vb_med * 1000,
        "key_gen_ms": key_gen_time * 1000,
        "json_bytes": json_bytes,
        "binary_bytes": binary_bytes,
    }


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_bench():
    return _run_bench()


@app.local_entrypoint()
def main():
    print("VeriLM Benchmark Gate C — Sync + Buffer-Protocol Optimization")
    print("=" * 72)
    r = run_bench.remote()
    print(f"\n--- Summary ---")
    print(f"Online path overhead (global sync): {r['global_oh_pct']:+.1f}%")
    print(f"Online path overhead (event sync):  {r['event_oh_pct']:+.1f}%")
    print(f"Event vs global:                    {r['event_vs_global_pct']:+.1f}%")
    print(f"Full audit open:                    {r['audit_open_med_ms']:.1f} ms")
    print(f"Verify full (binary):               {r['verify_binary_med_ms']:.1f} ms")
    print(f"Binary payload:                     {r['binary_bytes']:,} bytes")
