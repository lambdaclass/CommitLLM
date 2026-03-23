"""
Capture overhead benchmark: inference with vs without capture.

Four measurements:
  1. Baseline — capture hook installed but disabled (wrapper skips clone)
  2. Capture only — capture enabled, buffers drained but not processed
  3. Full pipeline — capture + trace build + bytes serialization + Rust commit
  4. Phase breakdown — generate / drain / trace build / serialize / commit

Usage:
    modal run scripts/modal_bench_capture.py
"""

import modal

app = modal.App("verilm-bench-capture")

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
    import os
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    from verilm.capture import get_capture_buffer

    buf = get_capture_buffer()

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    # Warmup.
    print(f"\nWarmup: {N_WARMUP} iterations...")
    buf.enabled = True
    for _ in range(N_WARMUP):
        buf.drain()
        llm.generate([PROMPT], params)
        buf.drain()
    print("Warmup done.\n")

    # ── 1. Baseline: capture OFF ──
    print(f"Baseline (capture OFF): {N_ITERS} iterations...")
    buf.enabled = False
    buf.drain()

    baseline_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        llm.generate([PROMPT], params)
        t1 = time.monotonic()
        baseline_times.append(t1 - t0)

    # ── 2. Capture only ──
    print(f"Capture ON (generate only): {N_ITERS} iterations...")
    buf.enabled = True

    capture_times = []
    for _ in range(N_ITERS):
        buf.drain()
        t0 = time.monotonic()
        llm.generate([PROMPT], params)
        t1 = time.monotonic()
        capture_times.append(t1 - t0)
        buf.drain()

    # ── 3. Full pipeline ──
    print(f"Full pipeline (capture + commit): {N_ITERS} iterations...")
    from verilm.server import VerifiedInferenceServer

    server = VerifiedInferenceServer(llm)

    pipeline_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        t1 = time.monotonic()
        pipeline_times.append(t1 - t0)

    # ── 4. Phase breakdown ──
    # Instrument each phase: generate, drain, bytes conversion, Rust trace+commit.
    import hashlib
    import verilm_rs
    from verilm import capture as cap

    print(f"Phase breakdown: {N_ITERS} iterations...")

    phase_generate = []
    phase_drain = []
    phase_to_bytes = []
    phase_rust = []

    el_capture = server.el_capture
    n_layers = cap._n_layers
    calls_per_fwd = n_layers * cap.PROJS_PER_LAYER

    for _ in range(N_ITERS):
        buf.drain()
        el_capture.drain()

        # Phase: generate
        t0 = time.monotonic()
        outputs = llm.generate([PROMPT], params)
        t1 = time.monotonic()
        phase_generate.append(t1 - t0)

        output = outputs[0]
        gen_token_ids = list(output.outputs[0].token_ids)
        prompt_token_ids = list(output.prompt_token_ids)
        all_token_ids = prompt_token_ids + gen_token_ids

        # Phase: sync + drain buffers
        import torch
        t2 = time.monotonic()
        torch.cuda.synchronize()
        captures = buf.drain()
        el_data = el_capture.drain()
        t3 = time.monotonic()
        phase_drain.append(t3 - t2)

        # Phase: convert captures to bytes
        t4 = time.monotonic()
        capture_inputs = [c[2].numpy().tobytes() for c in captures]
        capture_accs = [c[3].numpy().tobytes() for c in captures]
        capture_scales = []
        for c in captures:
            s = c[4]
            if hasattr(s, 'numel') and s.numel() > 1:
                capture_scales.append(float(s.max().item()))
            elif hasattr(s, 'item'):
                capture_scales.append(float(s.item()))
            else:
                capture_scales.append(float(s))

        residuals = el_data.get("residuals")
        residual_bytes = None
        if residuals:
            residual_bytes = [r.float().numpy().tobytes() for r in residuals]

        n_fwd = len(captures) // calls_per_fwd
        fwd_batch_sizes = [captures[i * calls_per_fwd][2].shape[0] for i in range(n_fwd)]
        t5 = time.monotonic()
        phase_to_bytes.append(t5 - t4)

        # Phase: Rust trace build + commit (single call)
        seed = hashlib.sha256(PROMPT.encode()).digest()
        manifest = {
            "tokenizer_hash": server._tokenizer_hash,
            "temperature": 0.0, "top_k": 0, "top_p": 1.0,
            "eos_policy": "stop",
            "weight_hash": server._weight_hash,
            "quant_hash": server._quant_hash,
            "system_prompt_hash": server._system_prompt_hash,
        }
        t6 = time.monotonic()
        state = verilm_rs.commit_from_captures(
            capture_inputs=capture_inputs,
            capture_accumulators=capture_accs,
            capture_scales=capture_scales,
            fwd_batch_sizes=fwd_batch_sizes,
            n_layers=n_layers,
            q_dim=cap._q_dim,
            kv_dim=cap._kv_dim,
            intermediate_size=cap._gate_up_half,
            level_c=True,
            token_ids=[int(t) for t in all_token_ids[1:]],
            prompt=PROMPT.encode(),
            sampling_seed=seed,
            manifest=manifest,
            residuals=residual_bytes,
        )
        t7 = time.monotonic()
        phase_rust.append(t7 - t6)

    # ── Results ──
    def stats(times):
        times = sorted(times)
        n = len(times)
        return sum(times) / n, times[n // 2]

    b_mean, b_med = stats(baseline_times)
    c_mean, c_med = stats(capture_times)
    p_mean, p_med = stats(pipeline_times)

    gen_mean, _ = stats(phase_generate)
    drain_mean, _ = stats(phase_drain)
    bytes_mean, _ = stats(phase_to_bytes)
    rust_mean, _ = stats(phase_rust)

    cap_overhead = ((c_mean - b_mean) / b_mean) * 100
    pipe_overhead = ((p_mean - b_mean) / b_mean) * 100

    print(f"\n{'='*65}")
    print("CAPTURE OVERHEAD BENCHMARK")
    print(f"{'='*65}")
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {len(prompt_token_ids)} tokens, max_tokens: {MAX_TOKENS}")
    print(f"Generated: {len(gen_token_ids)} tokens")
    print(f"Iterations: {N_ITERS} (after {N_WARMUP} warmup)")

    print(f"\n--- Wall clock ---")
    print(f"{'Metric':<25} {'Mean (ms)':>10} {'Median (ms)':>12}")
    print(f"{'-'*47}")
    print(f"{'Baseline (no capture)':<25} {b_mean*1000:>10.1f} {b_med*1000:>12.1f}")
    print(f"{'Capture only':<25} {c_mean*1000:>10.1f} {c_med*1000:>12.1f}")
    print(f"{'Full pipeline':<25} {p_mean*1000:>10.1f} {p_med*1000:>12.1f}")
    print(f"\nCapture overhead:  {cap_overhead:+.1f}%")
    print(f"Pipeline overhead: {pipe_overhead:+.1f}%")

    print(f"\n--- Phase breakdown (mean, ms) ---")
    total_phase = gen_mean + drain_mean + bytes_mean + rust_mean
    print(f"{'Generate':<25} {gen_mean*1000:>8.1f}  ({gen_mean/total_phase*100:>5.1f}%)")
    print(f"{'Drain buffers':<25} {drain_mean*1000:>8.1f}  ({drain_mean/total_phase*100:>5.1f}%)")
    print(f"{'Convert to bytes':<25} {bytes_mean*1000:>8.1f}  ({bytes_mean/total_phase*100:>5.1f}%)")
    print(f"{'Rust trace+commit':<25} {rust_mean*1000:>8.1f}  ({rust_mean/total_phase*100:>5.1f}%)")
    print(f"{'Total':<25} {total_phase*1000:>8.1f}")
    print(f"{'='*65}")

    return {
        "baseline_mean_ms": b_mean * 1000,
        "capture_mean_ms": c_mean * 1000,
        "pipeline_mean_ms": p_mean * 1000,
        "capture_overhead_pct": cap_overhead,
        "pipeline_overhead_pct": pipe_overhead,
        "phase_generate_ms": gen_mean * 1000,
        "phase_drain_ms": drain_mean * 1000,
        "phase_to_bytes_ms": bytes_mean * 1000,
        "phase_rust_ms": rust_mean * 1000,
    }


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_bench():
    return _run_bench()


@app.local_entrypoint()
def main():
    print("VeriLM Capture Overhead Benchmark")
    print("=" * 65)
    result = run_bench.remote()
    print(f"\nCapture overhead: {result['capture_overhead_pct']:+.1f}%")
    print(f"Pipeline overhead: {result['pipeline_overhead_pct']:+.1f}%")
