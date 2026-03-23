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
    # Instrument one iteration to measure each phase separately.
    import hashlib
    import verilm_rs
    from verilm import capture as cap
    from verilm.hooks import EmbeddingLogitCapture
    from verilm.trace import build_layer_traces

    print(f"Phase breakdown: {N_ITERS} iterations...")

    phase_generate = []
    phase_drain = []
    phase_traces = []
    phase_serialize = []
    phase_commit = []

    el_capture = server.el_capture
    n_layers = cap._n_layers

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

        # Phase: build traces
        residuals = el_data.get("residuals")
        t4 = time.monotonic()
        traces = build_layer_traces(
            captures, n_layers=n_layers, level_c=True,
            residuals=residuals if residuals else None,
        )
        t5 = time.monotonic()
        phase_traces.append(t5 - t4)

        # Phase: serialize to bytes (tensors already on CPU from capture hook)
        t6 = time.monotonic()
        trace_dicts = []
        for token_layers in traces:
            layer_dicts = []
            for lt in token_layers:
                d = {
                    "x_attn": lt["x_attn"].numpy().tobytes(),
                    "q": lt["q"].numpy().tobytes(),
                    "k": lt["k"].numpy().tobytes(),
                    "v": lt["v"].numpy().tobytes(),
                    "a": lt["a"].numpy().tobytes(),
                    "attn_out": lt["attn_out"].numpy().tobytes(),
                    "x_ffn": lt["x_ffn"].numpy().tobytes(),
                    "g": lt["g"].numpy().tobytes(),
                    "u": lt["u"].numpy().tobytes(),
                    "h": lt["h"].numpy().tobytes(),
                    "ffn_out": lt["ffn_out"].numpy().tobytes(),
                    "kv_cache_k": [t.numpy().tobytes() for t in lt.get("kv_cache_k", [])],
                    "kv_cache_v": [t.numpy().tobytes() for t in lt.get("kv_cache_v", [])],
                }
                if "residual" in lt:
                    d["residual"] = lt["residual"].numpy().tobytes()
                if "qkv_scale" in lt:
                    d["scale_x_attn"] = float(lt["qkv_scale"].item()) if lt["qkv_scale"].numel() == 1 else float(lt["qkv_scale"].max().item())
                if "o_scale" in lt:
                    d["scale_a"] = float(lt["o_scale"].item()) if lt["o_scale"].numel() == 1 else float(lt["o_scale"].max().item())
                if "gu_scale" in lt:
                    d["scale_x_ffn"] = float(lt["gu_scale"].item()) if lt["gu_scale"].numel() == 1 else float(lt["gu_scale"].max().item())
                if "d_scale" in lt:
                    d["scale_h"] = float(lt["d_scale"].item()) if lt["d_scale"].numel() == 1 else float(lt["d_scale"].max().item())
                layer_dicts.append(d)
            trace_dicts.append(layer_dicts)
        t7 = time.monotonic()
        phase_serialize.append(t7 - t6)

        # Phase: Rust commit
        seed = hashlib.sha256(PROMPT.encode()).digest()
        manifest = {
            "tokenizer_hash": server._tokenizer_hash,
            "temperature": 0.0, "top_k": 0, "top_p": 1.0,
            "eos_policy": "stop",
            "weight_hash": server._weight_hash,
            "quant_hash": server._quant_hash,
            "system_prompt_hash": server._system_prompt_hash,
        }
        t8 = time.monotonic()
        state = verilm_rs.commit(
            traces=trace_dicts,
            token_ids=[int(t) for t in all_token_ids[1:]],
            prompt=PROMPT.encode(),
            sampling_seed=seed,
            manifest=manifest,
        )
        t9 = time.monotonic()
        phase_commit.append(t9 - t8)

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
    trace_mean, _ = stats(phase_traces)
    ser_mean, _ = stats(phase_serialize)
    com_mean, _ = stats(phase_commit)

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
    total_phase = gen_mean + drain_mean + trace_mean + ser_mean + com_mean
    print(f"{'Generate':<25} {gen_mean*1000:>8.1f}  ({gen_mean/total_phase*100:>5.1f}%)")
    print(f"{'Drain buffers':<25} {drain_mean*1000:>8.1f}  ({drain_mean/total_phase*100:>5.1f}%)")
    print(f"{'Build traces':<25} {trace_mean*1000:>8.1f}  ({trace_mean/total_phase*100:>5.1f}%)")
    print(f"{'Serialize (tobytes)':<25} {ser_mean*1000:>8.1f}  ({ser_mean/total_phase*100:>5.1f}%)")
    print(f"{'Rust commit':<25} {com_mean*1000:>8.1f}  ({com_mean/total_phase*100:>5.1f}%)")
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
        "phase_traces_ms": trace_mean * 1000,
        "phase_serialize_ms": ser_mean * 1000,
        "phase_commit_ms": com_mean * 1000,
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
