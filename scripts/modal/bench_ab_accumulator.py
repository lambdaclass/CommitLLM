"""Quick A/B: Rust hook vs native C++ accumulator, same GPU, same run.

Only tests the capture wrapper (scenario 2) — no commit.
Interleaves Rust hook and native accumulator rounds for fair comparison.

Usage:
    python scripts/runpod/test.py --script scripts/modal/bench_ab_accumulator.py
"""

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VERILM_CAPTURE_MODE"] = "minimal"

import time
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
PROMPT = (
    "Write a numbered list of exactly 100 different animals, one per line. "
    "For each animal, include its common name, scientific name, and typical habitat. "
    "Start with number 1 and continue sequentially."
)
MAX_TOKENS = 256
N_WARMUP = 3
N_ROUNDS = 8


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def _run_test():
    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.hooks import FinalResidualCapture
    from verilm.server import VerifiedInferenceServer

    cap._capture_mode = "minimal"

    print(f"Loading {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID, dtype="auto", max_model_len=8192,
        enforce_eager=True, enable_prefix_caching=False,
    )

    buf = get_capture_buffer()
    fr_capture = FinalResidualCapture()
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    fr_capture.install(model)
    server = VerifiedInferenceServer(llm)

    native = cap._native_capture
    rust_hook = cap._capture_hook

    if native is not None:
        print(f">>> Native C++ accumulator loaded")
    else:
        print(f">>> Native accumulator NOT available — can only test Rust hook")
    if rust_hook is not None:
        print(f">>> Rust CaptureHook loaded")

    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    # Warmup
    print(f"Warmup: {N_WARMUP} iterations...")
    buf.enabled = True
    fr_capture.enabled = False
    for _ in range(N_WARMUP):
        llm.generate([PROMPT], params)
        buf.wait_for_transfers()
        buf.drain()
        if native:
            native.drain_discard()
    print("Warmup done.\n")

    times_baseline = []
    times_rust = []
    times_native = []

    for round_i in range(N_ROUNDS):
        # ── 1. Baseline (no capture) ──
        buf.enabled = False
        fr_capture.enabled = False
        buf.drain()
        if native:
            native.drain_discard()

        t0 = time.monotonic()
        out = llm.generate([PROMPT], params)
        t1 = time.monotonic()
        times_baseline.append(t1 - t0)
        gen_tokens = len(out[0].outputs[0].token_ids)

        # ── 2. Rust hook path ──
        # Temporarily disable native accumulator so Python wrapper uses Rust hook.
        cap._native_capture = None
        buf.enabled = True
        buf.reset_counter()
        buf.drain()

        t0 = time.monotonic()
        out = llm.generate([PROMPT], params)
        buf.wait_for_transfers()
        t1 = time.monotonic()
        times_rust.append(t1 - t0)
        # Drain Rust hook scales
        buf.drain_minimal()

        # ── 3. Native accumulator path ──
        cap._native_capture = native  # Restore
        if native:
            buf.enabled = True
            buf.reset_counter()
            buf.drain()
            if native:
                native.drain_discard()

            t0 = time.monotonic()
            out = llm.generate([PROMPT], params)
            buf.wait_for_transfers()
            t1 = time.monotonic()
            times_native.append(t1 - t0)
            # Drain native scales
            buf.drain_minimal()

    # Reset
    buf.enabled = False

    # Report
    med_base = _median(times_baseline)
    med_rust = _median(times_rust)

    print(f"\nlong_256 A/B — {gen_tokens} tokens, {N_ROUNDS} rounds")
    print(f"{'='*68}")
    print(f"{'Path':<30} {'Med ms':>9} {'Delta ms':>10} {'ms/tok':>8}")
    print(f"{'-'*68}")
    print(f"{'Baseline (no capture)':<30} {med_base*1000:>8.1f}ms {'':>10} {'':>8}")

    delta_rust = (med_rust - med_base) * 1000
    print(f"{'Rust CaptureHook':<30} {med_rust*1000:>8.1f}ms {delta_rust:>+9.1f}ms {delta_rust/gen_tokens:>+7.2f}")

    if times_native:
        med_native = _median(times_native)
        delta_native = (med_native - med_base) * 1000
        print(f"{'C++ native accumulator':<30} {med_native*1000:>8.1f}ms {delta_native:>+9.1f}ms {delta_native/gen_tokens:>+7.2f}")
        print(f"\n  Native vs Rust: {delta_native - delta_rust:>+.1f}ms ({(delta_native - delta_rust)/gen_tokens:>+.2f}ms/tok)")

    print(f"\n  Baseline: {med_base*1000:.1f}ms = {gen_tokens/med_base:.1f} tok/s")

    return {"passed": True}
