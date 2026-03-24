"""
Benchmark Gate B: post-optimization regression checkpoint.

Measures the impact of cached WeightProvider, stratified audit tiers,
and binary wire format against Gate A.5 baselines.

  1. Baseline latency (inference only, no capture)
  2. Minimal online path latency (capture + V4 commit)
  3. Routine audit open time (contiguous prefix, cached provider)
  4. Full audit open time (all layers, cached provider)
  5. Verify time (key-only Freivalds)
  6. Binary payload size vs JSON
  7. Binary audit open + verify round-trip

Usage:
    modal run --detach scripts/modal_bench_gate_b.py
"""

import modal

app = modal.App("verilm-bench-gate-b")

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
        "python -c \""
        "import site, os; "
        "d = site.getsitepackages()[0]; "
        "open(os.path.join(d, 'verilm_capture.pth'), 'w')"
        ".write('import verilm._startup\\n')\"",
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

    # ── 2. Minimal online path (capture + V4 commit) ──
    print(f"2. Minimal pipeline (V4): {N_ITERS} iterations...")
    cap._capture_mode = "minimal"
    buf.enabled = True
    min_pipe_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        min_pipe_times.append(time.monotonic() - t0)

    # ── 3. Routine audit open (contiguous prefix, cached provider) ──
    print(f"3. Routine audit open: {N_ITERS} iterations...")
    # Derive a routine challenge to get the prefix layer set
    challenge_seed = hashlib.sha256(b"gate-b-bench-verifier").digest()

    routine_open_times = []
    routine_json = None
    for i in range(N_ITERS):
        chat_r = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        challenge = verilm_rs.build_audit_challenge(
            list(challenge_seed), chat_r["n_tokens"], n_layers, "routine"
        )
        t0 = time.monotonic()
        result = server.audit(
            request_id=chat_r["request_id"],
            token_index=challenge["token_index"],
            layer_indices=challenge["layer_indices"],
            tier="routine",
        )
        routine_open_times.append(time.monotonic() - t0)
        routine_json = result

    routine_layers = challenge["layer_indices"]
    n_routine_layers = len(routine_layers)

    # ── 4. Full audit open (all layers, cached provider) ──
    print(f"4. Full audit open: {N_ITERS} iterations...")
    full_layers = list(range(n_layers))
    full_open_times = []
    full_json = None
    for i in range(N_ITERS):
        chat_f = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        t0 = time.monotonic()
        result = server.audit(
            request_id=chat_f["request_id"],
            token_index=0,
            layer_indices=full_layers,
            tier="full",
        )
        full_open_times.append(time.monotonic() - t0)
        full_json = result

    n_tokens = chat_f["n_tokens"]

    # ── 5. Generate verifier key ──
    print("5. Key generation...")
    key_t0 = time.monotonic()
    key_json = verilm_rs.generate_key(model_dir, seed)
    key_gen_time = time.monotonic() - key_t0

    # ── 6. Verify time (JSON) ──
    print(f"6. Verify V4 (key-only, JSON): {N_ITERS} iterations...")
    verify_json_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        verilm_rs.verify_v4(full_json, key_json)
        verify_json_times.append(time.monotonic() - t0)

    # Verify routine too
    verify_routine_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        verilm_rs.verify_v4(routine_json, key_json)
        verify_routine_times.append(time.monotonic() - t0)

    # ── 7. Binary audit open + verify ──
    print(f"7. Binary audit (full): {N_ITERS} iterations...")
    binary_open_times = []
    binary_data = None
    for i in range(N_ITERS):
        chat_b = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        t0 = time.monotonic()
        result = server.audit(
            request_id=chat_b["request_id"],
            token_index=0,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        binary_open_times.append(time.monotonic() - t0)
        binary_data = result

    print(f"7b. Verify V4 (binary): {N_ITERS} iterations...")
    verify_binary_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        verilm_rs.verify_v4_binary(bytes(binary_data), key_json)
        verify_binary_times.append(time.monotonic() - t0)

    # ── Payload sizes ──
    json_full_bytes = len(full_json.encode("utf-8")) if isinstance(full_json, str) else len(full_json)
    json_routine_bytes = len(routine_json.encode("utf-8")) if isinstance(routine_json, str) else len(routine_json)
    binary_full_bytes = len(binary_data)

    # Check bridge state
    parsed = json.loads(full_json) if isinstance(full_json, str) else full_json
    shell = parsed.get("shell_opening")
    has_initial_residual = shell is not None and shell.get("initial_residual") is not None
    has_embedding_proof = shell is not None and shell.get("embedding_proof") is not None

    # ── Report ──
    b_mean, b_med = stats(baseline_times)
    mp_mean, mp_med = stats(min_pipe_times)
    ro_mean, ro_med = stats(routine_open_times)
    fo_mean, fo_med = stats(full_open_times)
    vj_mean, vj_med = stats(verify_json_times)
    vr_mean, vr_med = stats(verify_routine_times)
    bo_mean, bo_med = stats(binary_open_times)
    vb_mean, vb_med = stats(verify_binary_times)

    online_oh = ((mp_mean - b_mean) / b_mean) * 100
    compression = json_full_bytes / max(binary_full_bytes, 1)

    print(f"\n{'='*80}")
    print("BENCHMARK GATE B  —  Post-Optimization Regression Checkpoint")
    print(f"{'='*80}")
    print(f"Model:        {MODEL_ID}")
    print(f"GPU:          A100-80GB")
    print(f"Tokens:       {n_tokens} generated ({MAX_TOKENS} max)")
    print(f"Layers:       {n_layers}")
    print(f"Iters:        {N_ITERS} (after {N_WARMUP} warmup)")
    print(f"Full bridge:  initial_residual={'yes' if has_initial_residual else 'no'}, "
          f"embedding_proof={'yes' if has_embedding_proof else 'no'}")
    print(f"Routine:      {n_routine_layers}/{n_layers} layers (prefix 0..={max(routine_layers)})")

    print(f"\n--- Latency (ms) ---")
    print(f"{'Metric':<45} {'Mean':>10} {'Median':>10} {'OH%':>8}")
    print(f"{'-'*73}")
    print(f"{'1. Baseline (no capture)':<45} {b_mean*1000:>10.1f} {b_med*1000:>10.1f} {'—':>8}")
    print(f"{'2. Minimal online path (V4 commit)':<45} {mp_mean*1000:>10.1f} {mp_med*1000:>10.1f} {online_oh:>+7.1f}%")
    print(f"{'3. Routine audit open (prefix, cached)':<45} {ro_mean*1000:>10.1f} {ro_med*1000:>10.1f}")
    print(f"{'4. Full audit open (all layers, cached)':<45} {fo_mean*1000:>10.1f} {fo_med*1000:>10.1f}")
    print(f"{'5. Key generation (one-time)':<45} {key_gen_time*1000:>10.1f}")
    print(f"{'6a. Verify full (JSON deser + Freivalds)':<45} {vj_mean*1000:>10.1f} {vj_med*1000:>10.1f}")
    print(f"{'6b. Verify routine (JSON deser + Freivalds)':<45} {vr_mean*1000:>10.1f} {vr_med*1000:>10.1f}")
    print(f"{'7a. Full audit open (binary)':<45} {bo_mean*1000:>10.1f} {bo_med*1000:>10.1f}")
    print(f"{'7b. Verify full (binary deser + Freivalds)':<45} {vb_mean*1000:>10.1f} {vb_med*1000:>10.1f}")

    print(f"\n--- Payload ---")
    print(f"{'Metric':<45} {'Value':>12} {'Unit':>8}")
    print(f"{'-'*65}")
    print(f"{'Full audit (JSON)':<45} {json_full_bytes:>12,} {'bytes':>8}")
    print(f"{'Routine audit (JSON)':<45} {json_routine_bytes:>12,} {'bytes':>8}")
    print(f"{'Full audit (binary, bincode+zstd)':<45} {binary_full_bytes:>12,} {'bytes':>8}")
    print(f"{'Compression ratio (JSON/binary)':<45} {compression:>12.1f}x")
    print(f"{'Tokens generated':<45} {n_tokens:>12}")
    print(f"{'='*80}")

    # Gate A.5 reference: audit open was ~X ms with reload each time.
    # Gate B should show significant improvement from caching.

    return {
        "model": MODEL_ID,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "n_routine_layers": n_routine_layers,
        "baseline_med_ms": b_med * 1000,
        "online_path_med_ms": mp_med * 1000,
        "online_oh_pct": online_oh,
        "routine_open_med_ms": ro_med * 1000,
        "full_open_med_ms": fo_med * 1000,
        "key_gen_ms": key_gen_time * 1000,
        "verify_full_json_med_ms": vj_med * 1000,
        "verify_routine_json_med_ms": vr_med * 1000,
        "binary_open_med_ms": bo_med * 1000,
        "verify_full_binary_med_ms": vb_med * 1000,
        "json_full_bytes": json_full_bytes,
        "json_routine_bytes": json_routine_bytes,
        "binary_full_bytes": binary_full_bytes,
        "compression_ratio": compression,
        "has_full_bridge": has_initial_residual,
        "has_embedding_proof": has_embedding_proof,
    }


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_bench():
    return _run_bench()


@app.local_entrypoint()
def main():
    print("VeriLM Benchmark Gate B — Post-Optimization Regression")
    print("=" * 72)
    r = run_bench.remote()
    print(f"\n--- Summary ---")
    print(f"Online path overhead:       {r['online_oh_pct']:+.1f}%")
    print(f"Routine audit open:         {r['routine_open_med_ms']:.1f} ms ({r['n_routine_layers']}/{r['n_layers']} layers)")
    print(f"Full audit open:            {r['full_open_med_ms']:.1f} ms")
    print(f"Verify full (JSON):         {r['verify_full_json_med_ms']:.1f} ms")
    print(f"Verify routine (JSON):      {r['verify_routine_json_med_ms']:.1f} ms")
    print(f"Verify full (binary):       {r['verify_full_binary_med_ms']:.1f} ms")
    print(f"JSON full payload:          {r['json_full_bytes']:,} bytes")
    print(f"Binary full payload:        {r['binary_full_bytes']:,} bytes ({r['compression_ratio']:.1f}x)")
    print(f"Full bridge:                {r['has_full_bridge']}")
