"""
Capture overhead benchmark: inference with vs without capture.

Five measurements:
  1. Baseline — capture hook installed but disabled
  2. Full capture only — capture enabled (with _int_mm), drained but not committed
  3. Minimal capture only — capture enabled (no _int_mm), drained but not committed
  4. Full pipeline — capture + full trace build + commit (V1-V3)
  5. Minimal pipeline — capture + retained-state commit (V4)

Usage:
    modal run --detach scripts/modal_bench_capture.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pins import VERIFICATION

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
    .pip_install(*VERIFICATION)
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


def _extract_scale(s):
    """Extract a float from a scale tensor or scalar."""
    if hasattr(s, 'numel') and s.numel() > 1:
        return float(s.max().item())
    elif hasattr(s, 'item'):
        return float(s.item())
    return float(s)


def _run_bench():
    import os
    import time
    import hashlib

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    import verilm_rs
    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer

    buf = get_capture_buffer()

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True, enable_prefix_caching=False)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    # Warmup (full mode to cover the widest path).
    print(f"\nWarmup: {N_WARMUP} iterations...")
    cap._capture_mode = "full"
    buf.enabled = True
    for _ in range(N_WARMUP):
        buf.drain()
        llm.generate([PROMPT], params)
        buf.drain()
    print("Warmup done.\n")

    # Helper: shared manifest and seed.
    from verilm.server import VerifiedInferenceServer
    server = VerifiedInferenceServer(llm)
    seed = hashlib.sha256(PROMPT.encode()).digest()
    manifest = {
        "tokenizer_hash": server._tokenizer_hash,
        "temperature": 0.0, "top_k": 0, "top_p": 1.0,
        "eos_policy": "stop",
        "weight_hash": server._weight_hash,
        "quant_hash": server._quant_hash,
        "system_prompt_hash": server._system_prompt_hash,
    }
    n_layers = cap._n_layers
    calls_per_fwd = n_layers * cap.PROJS_PER_LAYER

    def stats(times):
        times = sorted(times)
        n = len(times)
        return sum(times) / n, times[n // 2]

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

    # ── 2. Full capture only (with _int_mm) ──
    print(f"Full capture ON: {N_ITERS} iterations...")
    buf.enabled = True
    cap._capture_mode = "full"

    full_cap_times = []
    for _ in range(N_ITERS):
        buf.drain()
        t0 = time.monotonic()
        llm.generate([PROMPT], params)
        t1 = time.monotonic()
        full_cap_times.append(t1 - t0)
        buf.drain()

    # ── 3. Minimal capture only (no _int_mm) ──
    print(f"Minimal capture ON: {N_ITERS} iterations...")
    cap._capture_mode = "minimal"

    min_cap_times = []
    for _ in range(N_ITERS):
        buf.drain()
        t0 = time.monotonic()
        llm.generate([PROMPT], params)
        t1 = time.monotonic()
        min_cap_times.append(t1 - t0)
        buf.drain()

    # ── 4. Full pipeline (capture + full commit) ──
    print(f"Full pipeline: {N_ITERS} iterations...")
    cap._capture_mode = "full"

    full_pipe_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        t1 = time.monotonic()
        full_pipe_times.append(t1 - t0)

    # ── 5. Minimal pipeline (capture + V4 commit) ──
    print(f"Minimal pipeline: {N_ITERS} iterations...")
    cap._capture_mode = "minimal"

    min_pipe_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        t1 = time.monotonic()
        min_pipe_times.append(t1 - t0)

    # ── 6. Phase breakdown: V4 minimal path ──
    print(f"V4 phase breakdown: {N_ITERS} iterations...")
    cap._capture_mode = "minimal"

    v4_gen = []
    v4_drain = []
    v4_extract = []
    v4_commit = []

    el_capture = server.el_capture

    for _ in range(N_ITERS):
        buf.drain()
        el_capture.drain()

        # Phase: generate
        t0 = time.monotonic()
        outputs = llm.generate([PROMPT], params)
        t1 = time.monotonic()
        v4_gen.append(t1 - t0)

        output = outputs[0]
        gen_token_ids = list(output.outputs[0].token_ids)
        prompt_token_ids = list(output.prompt_token_ids)
        all_token_ids = prompt_token_ids + gen_token_ids

        # Phase: sync + drain
        t2 = time.monotonic()
        torch.cuda.synchronize()
        captures = buf.drain()
        el_capture.drain()
        t3 = time.monotonic()
        v4_drain.append(t3 - t2)

        # Phase: extract o_proj inputs + scales
        t4 = time.monotonic()
        n_fwd = len(captures) // calls_per_fwd
        fwd_batch_sizes = [
            captures[i * calls_per_fwd + 1][2].shape[0] for i in range(n_fwd)
        ]

        o_proj_inputs = []
        minimal_scales = []
        for fwd_i in range(n_fwd):
            for l_i in range(n_layers):
                base = fwd_i * calls_per_fwd + l_i * 4
                o_proj_inputs.append(captures[base + 1][2].numpy().tobytes())
                for proj_off in range(4):
                    minimal_scales.append(_extract_scale(captures[base + proj_off][4]))
        t5 = time.monotonic()
        v4_extract.append(t5 - t4)

        # Phase: Rust V4 commit
        t6 = time.monotonic()
        verilm_rs.commit_minimal_from_captures(
            o_proj_inputs=o_proj_inputs,
            scales=minimal_scales,
            n_layers=n_layers,
            fwd_batch_sizes=fwd_batch_sizes,
            token_ids=[int(t) for t in all_token_ids[1:]],
            prompt=PROMPT.encode(),
            sampling_seed=seed,
            manifest=manifest,
        )
        t7 = time.monotonic()
        v4_commit.append(t7 - t6)

    # ── Results ──
    b_mean, b_med = stats(baseline_times)
    fc_mean, fc_med = stats(full_cap_times)
    mc_mean, mc_med = stats(min_cap_times)
    fp_mean, fp_med = stats(full_pipe_times)
    mp_mean, mp_med = stats(min_pipe_times)

    full_cap_oh = ((fc_mean - b_mean) / b_mean) * 100
    min_cap_oh = ((mc_mean - b_mean) / b_mean) * 100
    full_pipe_oh = ((fp_mean - b_mean) / b_mean) * 100
    min_pipe_oh = ((mp_mean - b_mean) / b_mean) * 100

    gen_mean, _ = stats(v4_gen)
    drain_mean, _ = stats(v4_drain)
    extract_mean, _ = stats(v4_extract)
    commit_mean, _ = stats(v4_commit)

    print(f"\n{'='*65}")
    print("CAPTURE OVERHEAD BENCHMARK — FULL vs MINIMAL (V4)")
    print(f"{'='*65}")
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {len(prompt_token_ids)} tokens, max_tokens: {MAX_TOKENS}")
    print(f"Generated: {len(gen_token_ids)} tokens")
    print(f"Iterations: {N_ITERS} (after {N_WARMUP} warmup)")

    print(f"\n--- Wall clock ---")
    print(f"{'Metric':<30} {'Mean (ms)':>10} {'Median (ms)':>12} {'OH%':>8}")
    print(f"{'-'*60}")
    print(f"{'Baseline (no capture)':<30} {b_mean*1000:>10.1f} {b_med*1000:>12.1f} {'—':>8}")
    print(f"{'Full capture only':<30} {fc_mean*1000:>10.1f} {fc_med*1000:>12.1f} {full_cap_oh:>+7.1f}%")
    print(f"{'Minimal capture only':<30} {mc_mean*1000:>10.1f} {mc_med*1000:>12.1f} {min_cap_oh:>+7.1f}%")
    print(f"{'Full pipeline (V1-V3)':<30} {fp_mean*1000:>10.1f} {fp_med*1000:>12.1f} {full_pipe_oh:>+7.1f}%")
    print(f"{'Minimal pipeline (V4)':<30} {mp_mean*1000:>10.1f} {mp_med*1000:>12.1f} {min_pipe_oh:>+7.1f}%")

    print(f"\n--- V4 phase breakdown (mean, ms) ---")
    total_v4 = gen_mean + drain_mean + extract_mean + commit_mean
    print(f"{'Generate':<25} {gen_mean*1000:>8.1f}  ({gen_mean/total_v4*100:>5.1f}%)")
    print(f"{'Sync+drain':<25} {drain_mean*1000:>8.1f}  ({drain_mean/total_v4*100:>5.1f}%)")
    print(f"{'Extract o_proj+scales':<25} {extract_mean*1000:>8.1f}  ({extract_mean/total_v4*100:>5.1f}%)")
    print(f"{'Rust V4 commit':<25} {commit_mean*1000:>8.1f}  ({commit_mean/total_v4*100:>5.1f}%)")
    print(f"{'Total':<25} {total_v4*1000:>8.1f}")
    print(f"{'='*65}")

    print(f"\nCapture overhead:  full={full_cap_oh:+.1f}%  minimal={min_cap_oh:+.1f}%")
    print(f"Pipeline overhead: full={full_pipe_oh:+.1f}%  minimal={min_pipe_oh:+.1f}%")

    return {
        "baseline_mean_ms": b_mean * 1000,
        "full_capture_mean_ms": fc_mean * 1000,
        "minimal_capture_mean_ms": mc_mean * 1000,
        "full_pipeline_mean_ms": fp_mean * 1000,
        "minimal_pipeline_mean_ms": mp_mean * 1000,
        "full_capture_oh_pct": full_cap_oh,
        "minimal_capture_oh_pct": min_cap_oh,
        "full_pipeline_oh_pct": full_pipe_oh,
        "minimal_pipeline_oh_pct": min_pipe_oh,
        "v4_generate_ms": gen_mean * 1000,
        "v4_drain_ms": drain_mean * 1000,
        "v4_extract_ms": extract_mean * 1000,
        "v4_commit_ms": commit_mean * 1000,
    }


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_bench():
    return _run_bench()


@app.local_entrypoint()
def main():
    print("VeriLM Capture Overhead Benchmark — Full vs Minimal (V4)")
    print("=" * 65)
    result = run_bench.remote()
    print(f"\nCapture overhead:  full={result['full_capture_oh_pct']:+.1f}%  minimal={result['minimal_capture_oh_pct']:+.1f}%")
    print(f"Pipeline overhead: full={result['full_pipeline_oh_pct']:+.1f}%  minimal={result['minimal_pipeline_oh_pct']:+.1f}%")
