"""
Fast benchmark: commit-phase subphase timers.

Runs baseline vs full server.chat at multiple generation lengths,
printing results immediately after each length completes.

Enables VERILM_COMMIT_TIMERS to get per-phase breakdown:
  generate → sync → drain → commit (pack_a + pack_fr + rust)

Usage:
    # Via RunPod:
    python scripts/runpod/test.py --script scripts/modal/bench_commit_phases.py

    # Via Modal:
    modal run --detach scripts/modal/bench_commit_phases.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pins import VERIFICATION

import modal

app = modal.App("verilm-bench-commit-phases")

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
N_ITERS = 8


def _stats(times):
    times = sorted(times)
    n = len(times)
    mean = sum(times) / n
    median = times[n // 2]
    p5 = times[max(0, int(n * 0.05))]
    p95 = times[min(n - 1, int(n * 0.95))]
    return mean, median, p5, p95


def _run_bench():
    import logging
    import os
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VERILM_COMMIT_TIMERS"] = "1"

    # Set up logging to capture commit timers on stdout.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID, dtype="auto", max_model_len=8192,
        enforce_eager=True, enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)

    # Warmup.
    print(f"\nWarmup: {N_WARMUP} iterations...")
    cap._capture_mode = "minimal"
    for _ in range(N_WARMUP):
        server.chat(prompt=PROMPT, max_tokens=32)
    print("Warmup done.\n")

    results = {}

    for max_tokens in [16, 64, 128, 256, 1024]:
        n_iters = N_ITERS if max_tokens <= 128 else max(3, N_ITERS // 2)
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        print(f"\n{'='*70}")
        print(f"  max_tokens = {max_tokens}  (n_iters = {n_iters})")
        print(f"{'='*70}")

        # ── Baseline: no capture ──
        from verilm.capture import get_capture_buffer
        from verilm.hooks import FinalResidualCapture
        buf = get_capture_buffer()
        buf.enabled = False
        if server.final_res_capture:
            server.final_res_capture.enabled = False

        times_baseline = []
        gen_tokens = 0
        for _ in range(n_iters):
            buf.drain()
            if server.final_res_capture:
                server.final_res_capture.drain()
            t0 = time.monotonic()
            out = llm.generate([PROMPT], params)
            t1 = time.monotonic()
            times_baseline.append(t1 - t0)
            gen_tokens = len(out[0].outputs[0].token_ids)

        # ── Full server.chat (packed commit) ──
        buf.enabled = True
        cap._capture_mode = "minimal"
        if server.final_res_capture:
            server.final_res_capture.enabled = True

        times_full = []
        for _ in range(n_iters):
            t0 = time.monotonic()
            server.chat(prompt=PROMPT, max_tokens=max_tokens)
            t1 = time.monotonic()
            times_full.append(t1 - t0)

        # Reset.
        buf.enabled = False
        if server.final_res_capture:
            server.final_res_capture.enabled = False

        # ── Report immediately ──
        b = _stats(times_baseline)
        f = _stats(times_full)
        oh = ((f[0] - b[0]) / b[0]) * 100 if b[0] > 0 else 0

        print(f"\nGenerated {gen_tokens} tokens (max_tokens={max_tokens})")
        print(f"  Baseline mean: {b[0]*1000:.1f}ms  (med {b[1]*1000:.1f}ms)")
        print(f"  Full mean:     {f[0]*1000:.1f}ms  (med {f[1]*1000:.1f}ms)")
        print(f"  Overhead:      {oh:+.1f}%  ({(f[0]-b[0])*1000:+.1f}ms)")
        print(f"  Per-token OH:  {(f[0]-b[0])*1000/max(gen_tokens,1):.2f}ms/tok")

        results[max_tokens] = {
            "gen_tokens": gen_tokens,
            "baseline_mean_ms": b[0] * 1000,
            "full_mean_ms": f[0] * 1000,
            "overhead_pct": oh,
            "overhead_ms": (f[0] - b[0]) * 1000,
            "per_token_oh_ms": (f[0] - b[0]) * 1000 / max(gen_tokens, 1),
        }

    # Final summary.
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'max_tok':<10} {'gen':<6} {'baseline':>10} {'full':>10} {'OH%':>8} {'OH/tok':>10}")
    print(f"{'-'*54}")
    for mt, r in sorted(results.items()):
        print(f"{mt:<10} {r['gen_tokens']:<6} {r['baseline_mean_ms']:>9.1f}ms {r['full_mean_ms']:>9.1f}ms {r['overhead_pct']:>+7.1f}% {r['per_token_oh_ms']:>9.2f}ms")

    return results


def _run_test():
    results = _run_bench()
    return {"passed": True, "results": results}


@app.function(image=image, gpu="A100-80GB", timeout=600)
def run_bench():
    return _run_bench()


@app.local_entrypoint()
def main():
    print("VeriLM Commit Phase Benchmark")
    print("=" * 50)
    results = run_bench.remote()
    print("\n--- Done ---")
