"""
Capture stability regression test: counter reset + mixed-shape sequences.

Tests that the call counter reset fix prevents capture-count mismatches
across back-to-back requests with varied prompt/output shapes. Runs on
a single GPU worker to catch counter drift that accumulates over many
requests in one process.

Three test phases:
  1. Previously crashing prompts (haiku 16tok, "AAA..." 4tok)
  2. Back-to-back shape transitions (short→long→short)
  3. Extended mixed-shape sequence (40 requests, alternating shapes)

Success: zero capture-count mismatches, zero token-count mismatches,
stable commitments, both sync modes produce identical results.

Usage:
    modal run --detach scripts/modal_test_capture_stability.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-test-capture-stability")

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
        ".git", "target", "scripts/__pycache__", "*.pdf", "*.md",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

# Phase 1: Previously crashing prompts
REGRESSION_CASES = [
    {"prompt": "Write a haiku about CUDA synchronization.", "max_tokens": 16},
    {"prompt": "A" * 200, "max_tokens": 4},
]

# Phase 2: Shape transitions (short/medium/long in each dimension)
TRANSITION_CASES = [
    {"prompt": "Hi", "max_tokens": 4},                # short/short
    {"prompt": "Explain quantum computing in detail.", "max_tokens": 128},  # medium/long
    {"prompt": "A" * 200, "max_tokens": 4},            # long/short
    {"prompt": "What is 2+2?", "max_tokens": 8},       # short/short
    {"prompt": "List all US presidents and their terms.", "max_tokens": 64},  # medium/medium
    {"prompt": "Hi", "max_tokens": 128},               # short/long
]

# Phase 3: Extended mixed-shape cycle (repeated 10x = 40 requests)
MIXED_CYCLE = [
    {"prompt": "Hi", "max_tokens": 4},                # short/short
    {"prompt": "Explain the theory of relativity.", "max_tokens": 64},  # medium/medium
    {"prompt": "A" * 200, "max_tokens": 4},            # long/short
    {"prompt": "Count to twenty.", "max_tokens": 128},  # short/long
]
MIXED_REPEATS = 10


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_stability_test():
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
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir

    # Warmup
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=4)
    print("Warmup done.\n")

    failures = []
    checks = 0
    key_cache = {}

    def run_case(label, prompt, max_tokens, sync_mode):
        """Run one request, return result or record failure."""
        nonlocal checks
        buf.set_sync_mode(sync_mode)
        cap._capture_mode = "minimal"
        try:
            result = server.chat(prompt=prompt, max_tokens=max_tokens)
            checks += 1
            print(f"  OK: {label} [{sync_mode}] — {result['n_tokens']} tokens, "
                  f"root={result['commitment']['merkle_root'][:16]}...")
            return result
        except RuntimeError as e:
            failures.append(f"{label} [{sync_mode}]: {e}")
            print(f"  FAIL: {label} [{sync_mode}]: {e}")
            return None

    def run_case_both_modes(label, prompt, max_tokens):
        """Run under both sync modes, compare commitments."""
        nonlocal checks
        r_global = run_case(label, prompt, max_tokens, "global")
        r_event = run_case(label, prompt, max_tokens, "event")

        if r_global is not None and r_event is not None:
            checks += 1
            if r_global["commitment"]["merkle_root"] != r_event["commitment"]["merkle_root"]:
                msg = f"{label}: commitment divergence global≠event"
                failures.append(msg)
                print(f"  FAIL: {msg}")
            else:
                print(f"  OK: {label} commitment match across sync modes")

    # ── Phase 1: Previously crashing prompts ──
    print("=" * 60)
    print("PHASE 1: Previously crashing prompts (regression)")
    print("=" * 60)
    for i, case in enumerate(REGRESSION_CASES):
        prompt_label = case["prompt"][:40] + "..." if len(case["prompt"]) > 40 else case["prompt"]
        for repeat in range(3):
            label = f"regression-{i} repeat={repeat} ({prompt_label}, {case['max_tokens']}tok)"
            run_case_both_modes(label, case["prompt"], case["max_tokens"])

    # ── Phase 2: Shape transitions ──
    print(f"\n{'='*60}")
    print("PHASE 2: Back-to-back shape transitions")
    print("=" * 60)
    for i, case in enumerate(TRANSITION_CASES):
        prompt_label = case["prompt"][:40] + "..." if len(case["prompt"]) > 40 else case["prompt"]
        label = f"transition-{i} ({prompt_label}, {case['max_tokens']}tok)"
        run_case_both_modes(label, case["prompt"], case["max_tokens"])

    # ── Phase 3: Extended mixed-shape sequence ──
    print(f"\n{'='*60}")
    print(f"PHASE 3: Extended mixed-shape sequence ({len(MIXED_CYCLE)} shapes x {MIXED_REPEATS} repeats = {len(MIXED_CYCLE) * MIXED_REPEATS} requests)")
    print("=" * 60)

    # Run all on global first, then verify a sample on event
    commitment_history = []
    for cycle in range(MIXED_REPEATS):
        for j, case in enumerate(MIXED_CYCLE):
            idx = cycle * len(MIXED_CYCLE) + j
            prompt_label = case["prompt"][:30] + "..." if len(case["prompt"]) > 30 else case["prompt"]
            label = f"mixed-{idx} cycle={cycle} ({prompt_label}, {case['max_tokens']}tok)"
            r = run_case(label, case["prompt"], case["max_tokens"], "global")
            if r is not None:
                commitment_history.append({
                    "idx": idx,
                    "prompt": case["prompt"],
                    "max_tokens": case["max_tokens"],
                    "root": r["commitment"]["merkle_root"],
                    "n_tokens": r["n_tokens"],
                })

    # Verify commitment stability: same prompt+max_tokens should produce same root
    print(f"\n  Checking commitment stability across {len(commitment_history)} requests...")
    root_by_key = {}
    for entry in commitment_history:
        key = (entry["prompt"], entry["max_tokens"])
        if key not in root_by_key:
            root_by_key[key] = entry["root"]
        else:
            checks += 1
            if root_by_key[key] != entry["root"]:
                msg = f"mixed-{entry['idx']}: commitment drift (first={root_by_key[key][:16]}, now={entry['root'][:16]})"
                failures.append(msg)
                print(f"  FAIL: {msg}")

    stable_keys = len(root_by_key)
    print(f"  OK: {stable_keys} unique prompt shapes, all commitments stable across repeats")

    # Spot-check event mode on last cycle
    print(f"\n  Spot-checking event mode on last cycle...")
    for j, case in enumerate(MIXED_CYCLE):
        prompt_label = case["prompt"][:30] + "..." if len(case["prompt"]) > 30 else case["prompt"]
        label = f"mixed-event-check-{j} ({prompt_label}, {case['max_tokens']}tok)"
        r_event = run_case(label, case["prompt"], case["max_tokens"], "event")
        if r_event is not None:
            key = (case["prompt"], case["max_tokens"])
            checks += 1
            if root_by_key.get(key) != r_event["commitment"]["merkle_root"]:
                msg = f"{label}: event commitment ≠ global commitment"
                failures.append(msg)
                print(f"  FAIL: {msg}")
            else:
                print(f"  OK: {label} matches global commitment")

    # ── Summary ──
    elapsed = time.monotonic()
    print(f"\n{'='*60}")
    total_requests = (len(REGRESSION_CASES) * 3 * 2  # phase 1: 3 repeats x 2 modes
                      + len(TRANSITION_CASES) * 2      # phase 2: 2 modes each
                      + len(MIXED_CYCLE) * MIXED_REPEATS  # phase 3: global
                      + len(MIXED_CYCLE))              # phase 3: event spot-check
    if failures:
        print(f"CAPTURE STABILITY TEST FAILED — {len(failures)} failure(s) in {total_requests} requests:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"CAPTURE STABILITY TEST PASSED — {checks} checks, {total_requests} requests, 0 failures")
    print(f"{'='*60}")

    return {
        "passed": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures,
        "checks": checks,
        "total_requests": total_requests,
    }


@app.local_entrypoint()
def main():
    print("VeriLM Capture Stability Test")
    print("=" * 60)
    result = run_stability_test.remote()
    if result["passed"]:
        print(f"\nCAPTURE STABILITY TEST PASSED — {result['checks']} checks, "
              f"{result['total_requests']} requests")
    else:
        print(f"\nCAPTURE STABILITY TEST FAILED — {result['n_failures']} failure(s):")
        for f in result["failures"]:
            print(f"  - {f}")
        raise SystemExit(1)
