"""
Benchmark: inference overhead from verification hooks.

Isolates the cost of each capture component added during inference:
  1. Baseline — vLLM inference, no capture at all
  2. Minimal capture — cutlass_scaled_mm wrapper only (matmul input/scale capture)
  3. + final_residual hook — adds model.norm forward hook (float32 D2H copy)
  4. + cuda.synchronize — device-wide sync before draining hooks
  5. Full server.chat — entire online path (capture + hooks + sync + commit)

Runs N_ITERS iterations at multiple generation lengths to show scaling.

Usage:
    modal run --detach scripts/modal/bench_hooks_overhead.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pins import VERIFICATION

import modal

app = modal.App("verilm-bench-hooks-overhead")

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
N_WARMUP = 5
N_ITERS = 20


def _stats(times):
    times = sorted(times)
    n = len(times)
    mean = sum(times) / n
    median = times[n // 2]
    p5 = times[max(0, int(n * 0.05))]
    p95 = times[min(n - 1, int(n * 0.95))]
    return mean, median, p5, p95


def _run_bench():
    import os
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.hooks import FinalResidualCapture
    from verilm.server import VerifiedInferenceServer

    buf = get_capture_buffer()

    print(f"Loading {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID, dtype="auto", max_model_len=8192,
        enforce_eager=True, enable_prefix_caching=False,
    )

    # Install final_residual hook separately so we can toggle it.
    fr_capture = FinalResidualCapture()
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    fr_installed = fr_capture.install(model)
    print(f"FinalResidualCapture installed: {fr_installed}")

    # Also create a full server for the end-to-end measurement.
    server = VerifiedInferenceServer(llm)

    # Warmup with everything enabled.
    print(f"\nWarmup: {N_WARMUP} iterations...")
    cap._capture_mode = "minimal"
    buf.enabled = True
    fr_capture.enabled = True
    for _ in range(N_WARMUP):
        buf.drain()
        fr_capture.drain()
        llm.generate([PROMPT], SamplingParams(max_tokens=64, temperature=0.0))
        torch.cuda.synchronize()
        buf.drain()
        fr_capture.drain()
    print("Warmup done.\n")

    results = {}

    for max_tokens in [16, 64, 128, 1024, 5120]:
        # Fewer iterations for long generations to keep total runtime reasonable.
        n_iters = N_ITERS if max_tokens <= 128 else (10 if max_tokens <= 1024 else 5)
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        print(f"\n{'='*70}")
        print(f"  max_tokens = {max_tokens}  (n_iters = {n_iters})")
        print(f"{'='*70}")

        # ── 1. Baseline: capture OFF, hook OFF ──
        buf.enabled = False
        fr_capture.enabled = False
        buf.drain()
        fr_capture.drain()

        times_baseline = []
        gen_tokens = 0
        for _ in range(n_iters):
            t0 = time.monotonic()
            out = llm.generate([PROMPT], params)
            t1 = time.monotonic()
            times_baseline.append(t1 - t0)
            gen_tokens = len(out[0].outputs[0].token_ids)

        # ── 2. Minimal capture ON, hook OFF ──
        buf.enabled = True
        cap._capture_mode = "minimal"
        fr_capture.enabled = False

        times_capture = []
        for _ in range(n_iters):
            buf.drain()
            t0 = time.monotonic()
            llm.generate([PROMPT], params)
            t1 = time.monotonic()
            times_capture.append(t1 - t0)
            buf.drain()

        # ── 3. Minimal capture ON + final_residual hook ON (no sync) ──
        fr_capture.enabled = True

        times_hook = []
        for _ in range(n_iters):
            buf.drain()
            fr_capture.drain()
            t0 = time.monotonic()
            llm.generate([PROMPT], params)
            t1 = time.monotonic()
            times_hook.append(t1 - t0)
            buf.drain()
            fr_capture.drain()

        # ── 4. Minimal capture ON + hook ON + cuda.synchronize ──
        times_sync = []
        for _ in range(n_iters):
            buf.drain()
            fr_capture.drain()
            t0 = time.monotonic()
            llm.generate([PROMPT], params)
            torch.cuda.synchronize()
            t1 = time.monotonic()
            times_sync.append(t1 - t0)
            buf.drain()
            fr_capture.drain()

        # ── 5. Full server.chat (everything) ──
        # Re-enable everything. server.chat does its own drain + sync + commit.
        cap._capture_mode = "minimal"
        buf.enabled = True
        fr_capture.enabled = True

        times_full = []
        for _ in range(n_iters):
            t0 = time.monotonic()
            server.chat(prompt=PROMPT, max_tokens=max_tokens)
            t1 = time.monotonic()
            times_full.append(t1 - t0)

        # Reset for next round.
        buf.enabled = False
        fr_capture.enabled = False

        # ── Report ──
        b = _stats(times_baseline)
        c = _stats(times_capture)
        h = _stats(times_hook)
        s = _stats(times_sync)
        f = _stats(times_full)

        def oh(x):
            return ((x[0] - b[0]) / b[0]) * 100 if b[0] > 0 else 0

        print(f"\nGenerated {gen_tokens} tokens (max_tokens={max_tokens})")
        print(f"{'Scenario':<45} {'Mean':>8} {'Med':>8} {'p5':>8} {'p95':>8} {'OH%':>7}")
        print(f"{'-'*84}")
        print(f"{'1. Baseline (no capture)':<45} {b[0]*1000:>8.1f} {b[1]*1000:>8.1f} {b[2]*1000:>8.1f} {b[3]*1000:>8.1f} {'—':>7}")
        print(f"{'2. + minimal capture (matmul wrapper)':<45} {c[0]*1000:>8.1f} {c[1]*1000:>8.1f} {c[2]*1000:>8.1f} {c[3]*1000:>8.1f} {oh(c):>+6.1f}%")
        print(f"{'3. + final_residual hook (no sync)':<45} {h[0]*1000:>8.1f} {h[1]*1000:>8.1f} {h[2]*1000:>8.1f} {h[3]*1000:>8.1f} {oh(h):>+6.1f}%")
        print(f"{'4. + cuda.synchronize()':<45} {s[0]*1000:>8.1f} {s[1]*1000:>8.1f} {s[2]*1000:>8.1f} {s[3]*1000:>8.1f} {oh(s):>+6.1f}%")
        print(f"{'5. Full server.chat (all + commit)':<45} {f[0]*1000:>8.1f} {f[1]*1000:>8.1f} {f[2]*1000:>8.1f} {f[3]*1000:>8.1f} {oh(f):>+6.1f}%")

        # Marginal costs
        cap_delta = (c[0] - b[0]) * 1000
        hook_delta = (h[0] - c[0]) * 1000
        sync_delta = (s[0] - h[0]) * 1000
        commit_delta = (f[0] - s[0]) * 1000

        print(f"\n  Marginal cost breakdown (mean, ms):")
        print(f"    Capture wrapper:       {cap_delta:>+7.2f} ms")
        print(f"    final_residual hook:   {hook_delta:>+7.2f} ms")
        print(f"    cuda.synchronize():    {sync_delta:>+7.2f} ms")
        print(f"    Commit + rest:         {commit_delta:>+7.2f} ms")
        print(f"    Total overhead:        {(f[0] - b[0]) * 1000:>+7.2f} ms ({oh(f):+.1f}%)")

        results[max_tokens] = {
            "gen_tokens": gen_tokens,
            "baseline_mean_ms": b[0] * 1000,
            "baseline_med_ms": b[1] * 1000,
            "capture_oh_pct": oh(c),
            "hook_oh_pct": oh(h),
            "sync_oh_pct": oh(s),
            "full_oh_pct": oh(f),
            "marginal_capture_ms": cap_delta,
            "marginal_hook_ms": hook_delta,
            "marginal_sync_ms": sync_delta,
            "marginal_commit_ms": commit_delta,
        }

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY — Overhead % by generation length")
    print(f"{'='*70}")
    print(f"{'max_tokens':<12} {'gen':<6} {'baseline':>10} {'capture':>10} {'+ hook':>10} {'+ sync':>10} {'full':>10}")
    print(f"{'-'*70}")
    for mt, r in sorted(results.items()):
        print(f"{mt:<12} {r['gen_tokens']:<6} {r['baseline_mean_ms']:>9.1f}ms {r['capture_oh_pct']:>+9.1f}% {r['hook_oh_pct']:>+9.1f}% {r['sync_oh_pct']:>+9.1f}% {r['full_oh_pct']:>+9.1f}%")

    return results


# Alias for RunPod test runner (expects _run_test or _run_e2e).
def _run_test():
    results = _run_bench()
    return {"passed": True, "results": results}


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_bench():
    return _run_bench()


@app.local_entrypoint()
def main():
    print("VeriLM Hook Overhead Benchmark")
    print("=" * 50)
    results = run_bench.remote()

    print(f"\n--- Summary ---")
    for mt, r in sorted(results.items()):
        print(f"max_tokens={mt}: baseline={r['baseline_mean_ms']:.1f}ms, "
              f"full overhead={r['full_oh_pct']:+.1f}%, "
              f"hook marginal={r['marginal_hook_ms']:+.2f}ms, "
              f"sync marginal={r['marginal_sync_ms']:+.2f}ms")
