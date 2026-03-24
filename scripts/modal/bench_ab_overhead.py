"""
A/B overhead benchmark: 5 scenarios, interleaved, two prompts.

Each config runs on its own A100 in parallel via Modal.
RunPod path runs sequentially on a single pod.

Scenarios (all on same GPU per config, interleaved per round):
  1. Baseline — llm.generate, no capture, no hooks
  2. + capture wrapper — matmul wrapper active, no hooks
  3. + hooks + sync — capture + final_residual hook + cuda.synchronize, no commit
  4. Full path (unpacked) — server.chat with VERILM_PACKED_COMMIT=0
  5. Full path (packed) — server.chat with VERILM_PACKED_COMMIT=1

Prompts:
  - LONG_PROMPT: reliably generates max_tokens without EOS
  - SHORT_PROMPT: triggers early EOS, exercises trim path

Reports:
  - actual generated tokens
  - absolute ms (median)
  - delta vs baseline (ms)
  - extra ms/token vs baseline
  - per-phase subphase timers (when VERILM_COMMIT_TIMERS=1)

Usage:
    # Parallel on Modal (one A100 per config, ~6 min total):
    modal run --detach scripts/modal/bench_ab_overhead.py

    # Sequential on RunPod (single pod, prints per-config):
    python scripts/runpod/test.py --script scripts/modal/bench_ab_overhead.py
"""

import modal

app = modal.App("verilm-bench-ab-overhead")

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

# Long prompt: enumerated list that reliably generates max_tokens without EOS.
LONG_PROMPT = (
    "Write a numbered list of exactly 100 different animals, one per line. "
    "For each animal, include its common name, scientific name, and typical habitat. "
    "Start with number 1 and continue sequentially."
)

# Short prompt: triggers early EOS for trim path coverage.
SHORT_PROMPT = "What is 2+2? Answer with just the number."

N_WARMUP = 3
N_ROUNDS = 8  # interleaved rounds

# Each config: (key, prompt_label, prompt, max_tokens)
CONFIGS = [
    ("long_64", "long", LONG_PROMPT, 64),
    ("long_128", "long", LONG_PROMPT, 128),
    ("long_256", "long", LONG_PROMPT, 256),
    ("short_eos_256", "short_eos", SHORT_PROMPT, 256),
]


def _median(times):
    s = sorted(times)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _stats(times):
    times = sorted(times)
    n = len(times)
    mean = sum(times) / n
    med = _median(times)
    p5 = times[max(0, int(n * 0.05))]
    p95 = times[min(n - 1, int(n * 0.95))]
    return {"mean": mean, "med": med, "p5": p5, "p95": p95}


def _run_single_config(config_key, prompt_label, prompt, max_tokens):
    """Run one config's 5-scenario interleaved benchmark. One GPU."""
    import logging
    import os
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VERILM_COMMIT_TIMERS"] = "1"

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.hooks import FinalResidualCapture
    from verilm.server import VerifiedInferenceServer

    print(f"[{config_key}] Loading {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID, dtype="auto", max_model_len=8192,
        enforce_eager=True, enable_prefix_caching=False,
    )

    buf = get_capture_buffer()
    fr_capture = FinalResidualCapture()
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    fr_capture.install(model)

    server = VerifiedInferenceServer(llm)

    # Warmup.
    print(f"[{config_key}] Warmup: {N_WARMUP} iterations...")
    cap._capture_mode = "minimal"
    buf.enabled = True
    fr_capture.enabled = True
    for _ in range(N_WARMUP):
        server.chat(prompt=prompt, max_tokens=min(max_tokens, 64))
    print(f"[{config_key}] Warmup done.\n")

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Collect times per scenario across interleaved rounds.
    times = {s: [] for s in ["baseline", "capture", "hooks_sync", "full_unpacked", "full_packed"]}
    gen_tokens = {}

    for round_i in range(N_ROUNDS):
        # ── 1. Baseline ──
        buf.enabled = False
        fr_capture.enabled = False
        buf.drain()
        fr_capture.drain()

        t0 = time.monotonic()
        out = llm.generate([prompt], params)
        t1 = time.monotonic()
        times["baseline"].append(t1 - t0)
        gen_tokens["baseline"] = len(out[0].outputs[0].token_ids)

        # ── 2. + capture wrapper ──
        buf.enabled = True
        cap._capture_mode = "minimal"
        fr_capture.enabled = False
        buf.drain()

        t0 = time.monotonic()
        out = llm.generate([prompt], params)
        t1 = time.monotonic()
        times["capture"].append(t1 - t0)
        gen_tokens["capture"] = len(out[0].outputs[0].token_ids)
        buf.drain()

        # ── 3. + hooks + sync, no commit ──
        buf.enabled = True
        cap._capture_mode = "minimal"
        fr_capture.enabled = True
        buf.drain()
        fr_capture.drain()

        t0 = time.monotonic()
        out = llm.generate([prompt], params)
        torch.cuda.synchronize()
        buf.drain()
        fr_capture.drain()
        t1 = time.monotonic()
        times["hooks_sync"].append(t1 - t0)
        gen_tokens["hooks_sync"] = len(out[0].outputs[0].token_ids)

        # ── 4. Full path (unpacked) ──
        buf.enabled = True
        cap._capture_mode = "minimal"
        fr_capture.enabled = True
        os.environ["VERILM_PACKED_COMMIT"] = "0"

        t0 = time.monotonic()
        result = server.chat(prompt=prompt, max_tokens=max_tokens)
        t1 = time.monotonic()
        times["full_unpacked"].append(t1 - t0)
        gen_tokens["full_unpacked"] = result["n_tokens"]

        # ── 5. Full path (packed) ──
        os.environ["VERILM_PACKED_COMMIT"] = "1"

        t0 = time.monotonic()
        result = server.chat(prompt=prompt, max_tokens=max_tokens)
        t1 = time.monotonic()
        times["full_packed"].append(t1 - t0)
        gen_tokens["full_packed"] = result["n_tokens"]

    # Reset.
    buf.enabled = False
    fr_capture.enabled = False
    os.environ["VERILM_PACKED_COMMIT"] = "1"

    # ── Report ──
    stats = {s: _stats(t) for s, t in times.items()}
    baseline_med = stats["baseline"]["med"]

    print(f"\n[{config_key}] prompt={prompt_label}, max_tokens={max_tokens}, rounds={N_ROUNDS}")
    print(f"{'Scenario':<30} {'Tokens':>6} {'Med ms':>9} {'Delta ms':>10} {'ms/tok':>8}")
    print(f"{'-'*68}")
    for scenario in ["baseline", "capture", "hooks_sync", "full_unpacked", "full_packed"]:
        s = stats[scenario]
        tok = gen_tokens.get(scenario, 0)
        delta = (s["med"] - baseline_med) * 1000
        ms_tok = delta / max(tok, 1)
        label = {
            "baseline": "1. Baseline (no capture)",
            "capture": "2. + capture wrapper",
            "hooks_sync": "3. + hooks + sync (no commit)",
            "full_unpacked": "4. Full (unpacked commit)",
            "full_packed": "5. Full (packed commit)",
        }[scenario]
        print(f"{label:<30} {tok:>6} {s['med']*1000:>8.1f}ms {delta:>+9.1f}ms {ms_tok:>+7.2f}")

    # Marginal cost breakdown.
    b = stats["baseline"]["med"]
    c = stats["capture"]["med"]
    h = stats["hooks_sync"]["med"]
    fu = stats["full_unpacked"]["med"]
    fp = stats["full_packed"]["med"]
    tok = gen_tokens.get("full_packed", gen_tokens.get("baseline", 1))

    print(f"\n  Marginal breakdown (median, ms):")
    print(f"    Capture wrapper:         {(c - b)*1000:>+8.1f}ms  ({(c - b)*1000/max(tok,1):>+6.2f}ms/tok)")
    print(f"    Hooks + sync:            {(h - c)*1000:>+8.1f}ms  ({(h - c)*1000/max(tok,1):>+6.2f}ms/tok)")
    print(f"    Commit (unpacked):       {(fu - h)*1000:>+8.1f}ms  ({(fu - h)*1000/max(tok,1):>+6.2f}ms/tok)")
    print(f"    Commit (packed):         {(fp - h)*1000:>+8.1f}ms  ({(fp - h)*1000/max(tok,1):>+6.2f}ms/tok)")
    print(f"    Packed vs unpacked:      {(fp - fu)*1000:>+8.1f}ms")
    print(f"    Total overhead (packed):  {(fp - b)*1000:>+8.1f}ms  ({(fp - b)*1000/max(tok,1):>+6.2f}ms/tok)")

    return {
        "config_key": config_key,
        "prompt": prompt_label,
        "max_tokens": max_tokens,
        "gen_tokens": gen_tokens,
        "stats_ms": {s: {k: v * 1000 for k, v in st.items()} for s, st in stats.items()},
        "marginal_capture_ms": (c - b) * 1000,
        "marginal_hooks_sync_ms": (h - c) * 1000,
        "marginal_commit_unpacked_ms": (fu - h) * 1000,
        "marginal_commit_packed_ms": (fp - h) * 1000,
        "packed_vs_unpacked_ms": (fp - fu) * 1000,
        "total_overhead_packed_ms": (fp - b) * 1000,
    }


# ── RunPod path: sequential (single pod) ──

def _run_bench():
    """Sequential fallback for RunPod (single GPU)."""
    all_results = {}
    for config_key, prompt_label, prompt, max_tokens in CONFIGS:
        result = _run_single_config(config_key, prompt_label, prompt, max_tokens)
        all_results[config_key] = result
    _print_summary(all_results)
    return all_results


def _run_test():
    results = _run_bench()
    return {"passed": True, "results": results}


def _print_summary(all_results):
    print(f"\n{'='*78}")
    print("SUMMARY — Marginal overhead (median ms)")
    print(f"{'='*78}")
    print(f"{'Config':<20} {'Tok':>5} {'Capture':>9} {'Hook+Sync':>10} {'Commit_U':>9} {'Commit_P':>9} {'P-U':>7} {'Total_P':>9}")
    print(f"{'-'*78}")
    for key in sorted(all_results):
        r = all_results[key]
        tok = r["gen_tokens"].get("full_packed", r["gen_tokens"].get("baseline", 0))
        print(
            f"{key:<20} {tok:>5} "
            f"{r['marginal_capture_ms']:>+8.1f} "
            f"{r['marginal_hooks_sync_ms']:>+9.1f} "
            f"{r['marginal_commit_unpacked_ms']:>+8.1f} "
            f"{r['marginal_commit_packed_ms']:>+8.1f} "
            f"{r['packed_vs_unpacked_ms']:>+6.1f} "
            f"{r['total_overhead_packed_ms']:>+8.1f}"
        )


# ── Modal path: parallel (one A100 per config) ──

@app.function(image=image, gpu="A100-80GB", timeout=600)
def run_config(config_key: str, prompt_label: str, prompt: str, max_tokens: int):
    return _run_single_config(config_key, prompt_label, prompt, max_tokens)


@app.local_entrypoint()
def main():
    print("VeriLM A/B Overhead Benchmark (parallel)")
    print("=" * 60)
    print(f"Launching {len(CONFIGS)} configs on separate A100s...\n")

    # Parallel dispatch: each config gets its own GPU container.
    handles = []
    for config_key, prompt_label, prompt, max_tokens in CONFIGS:
        print(f"  Spawning {config_key} (max_tokens={max_tokens})...")
        h = run_config.spawn(config_key, prompt_label, prompt, max_tokens)
        handles.append((config_key, h))

    # Collect results as they finish.
    all_results = {}
    for config_key, h in handles:
        result = h.get()
        all_results[config_key] = result
        print(f"\n  {config_key} done.")

    _print_summary(all_results)
    print("\n--- All configs complete ---")
