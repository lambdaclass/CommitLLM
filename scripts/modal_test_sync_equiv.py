"""
Sync-mode equivalence test: global vs event produce identical protocol output.

Shards test cases across parallel GPU workers. Each worker loads the model
once, generates the verifier key once, and runs all repeats for its case.

Usage:
    modal run --detach scripts/modal_test_sync_equiv.py
"""

import modal

app = modal.App("verilm-test-sync-equiv")

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

TEST_CASES = [
    {"prompt": "Explain the theory of relativity in one paragraph.", "max_tokens": 32},
    {"prompt": "What is 2+2?", "max_tokens": 8},
    {"prompt": "Write a haiku about CUDA synchronization.", "max_tokens": 16},
    {"prompt": "List the first 10 prime numbers.", "max_tokens": 64},
    {"prompt": "A" * 200, "max_tokens": 4},   # Long prompt, short output
    {"prompt": "Hi", "max_tokens": 128},       # Short prompt, long output
]

N_REPEATS = 3


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_case(case_idx: int, prompt: str, max_tokens: int, n_repeats: int):
    """Run one test case on a dedicated GPU worker."""
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

    t_start = time.monotonic()
    buf = get_capture_buffer()

    print(f"[case={case_idx}] Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    print(f"[case={case_idx}] Model loaded in {time.monotonic() - t_start:.1f}s")

    # Warmup
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=4)

    # Cache keygen once (expensive, doesn't depend on sync mode)
    t_key = time.monotonic()
    seed = hashlib.sha256(prompt.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)
    print(f"[case={case_idx}] Key generated in {time.monotonic() - t_key:.1f}s")

    failures = []
    checks = 0
    full_layers = list(range(n_layers))
    prompt_label = prompt[:40] + "..." if len(prompt) > 40 else prompt

    def assert_eq(a, b, msg):
        nonlocal checks
        checks += 1
        if a != b:
            failures.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK: {msg}")

    MAX_RETRIES = 3  # Retry on intermittent capture-count mismatch

    def chat_with_retry(mode_label, prompt, max_tokens):
        """Retry server.chat on capture mismatch (known intermittent issue)."""
        for attempt in range(MAX_RETRIES):
            try:
                return server.chat(prompt=prompt, max_tokens=max_tokens)
            except RuntimeError as e:
                if "Token count" in str(e) and "does not match expected" in str(e):
                    print(f"  WARN: capture mismatch ({mode_label}, attempt {attempt+1}) — retrying")
                    buf.drain()  # clear stale captures
                    continue
                raise
        raise RuntimeError(f"Capture mismatch persisted after {MAX_RETRIES} retries ({mode_label})")

    for repeat in range(n_repeats):
        tag = f"case={case_idx} repeat={repeat} prompt={prompt_label!r} max_tokens={max_tokens}"
        print(f"\n--- {tag} ---")

        # ── Global sync ──
        buf.set_sync_mode("global")
        cap._capture_mode = "minimal"
        result_global = chat_with_retry("global", prompt, max_tokens)

        commit_global = result_global["commitment"]["merkle_root"]
        audit_binary_global = server.audit(
            request_id=result_global["request_id"],
            token_index=0, layer_indices=full_layers, tier="full", binary=True,
        )

        result_global_2 = chat_with_retry("global-json", prompt, max_tokens)
        audit_json_global = server.audit(
            request_id=result_global_2["request_id"],
            token_index=0, layer_indices=full_layers, tier="full", binary=False,
        )
        report_global = verilm_rs.verify_v4(audit_json_global, key_json)

        # ── Event sync ──
        buf.set_sync_mode("event")
        cap._capture_mode = "minimal"
        result_event = chat_with_retry("event", prompt, max_tokens)

        commit_event = result_event["commitment"]["merkle_root"]
        audit_binary_event = server.audit(
            request_id=result_event["request_id"],
            token_index=0, layer_indices=full_layers, tier="full", binary=True,
        )

        result_event_2 = chat_with_retry("event-json", prompt, max_tokens)
        audit_json_event = server.audit(
            request_id=result_event_2["request_id"],
            token_index=0, layer_indices=full_layers, tier="full", binary=False,
        )
        report_event = verilm_rs.verify_v4(audit_json_event, key_json)

        # ── Compare ──
        assert_eq(result_global["n_tokens"], result_event["n_tokens"],
                  f"[{tag}] n_tokens match")
        assert_eq(result_global["generated_text"], result_event["generated_text"],
                  f"[{tag}] generated text match")
        assert_eq(commit_global, commit_event,
                  f"[{tag}] commitment merkle_root match")
        assert_eq(bytes(audit_binary_global), bytes(audit_binary_event),
                  f"[{tag}] binary audit payload match ({len(audit_binary_global)} bytes)")

        audit_g = json.loads(audit_json_global) if isinstance(audit_json_global, str) else audit_json_global
        audit_e = json.loads(audit_json_event) if isinstance(audit_json_event, str) else audit_json_event
        assert_eq(json.dumps(audit_g, sort_keys=True), json.dumps(audit_e, sort_keys=True),
                  f"[{tag}] JSON audit match")

        assert_eq(report_global["passed"], report_event["passed"],
                  f"[{tag}] verify passed match (global={report_global['passed']}, event={report_event['passed']})")
        assert_eq(report_global["checks_run"], report_event["checks_run"],
                  f"[{tag}] verify checks_run match")

    elapsed = time.monotonic() - t_start
    print(f"\n[case={case_idx}] Done in {elapsed:.1f}s — {checks} checks, {len(failures)} failures")

    return {
        "case_idx": case_idx,
        "prompt": prompt_label,
        "failures": failures,
        "checks": checks,
        "elapsed_s": elapsed,
    }


@app.local_entrypoint()
def main():
    print("VeriLM Sync Equivalence Test: global vs event")
    print(f"Launching {len(TEST_CASES)} cases across parallel GPU workers...")
    print("=" * 60)

    # Fan out all cases in parallel — one GPU worker per case
    results = list(run_case.starmap(
        [(i, c["prompt"], c["max_tokens"], N_REPEATS) for i, c in enumerate(TEST_CASES)],
        return_exceptions=True,
    ))

    # Aggregate — handle worker crashes gracefully
    total_checks = 0
    all_failures = []
    worker_errors = []
    good_results = []

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            worker_errors.append(f"case={i} CRASHED: {r}")
        else:
            good_results.append(r)
            total_checks += r["checks"]
            all_failures.extend(r["failures"])

    print(f"\n{'='*60}")
    for r in sorted(good_results, key=lambda x: x["case_idx"]):
        status = "PASS" if not r["failures"] else f"FAIL ({len(r['failures'])})"
        print(f"  case={r['case_idx']} [{status}] {r['prompt']} — {r['elapsed_s']:.1f}s")
    for err in worker_errors:
        print(f"  {err}")

    if all_failures or worker_errors:
        print(f"\nSYNC EQUIVALENCE TEST FAILED — {len(all_failures)} comparison failure(s), "
              f"{len(worker_errors)} worker crash(es):")
        for f in all_failures:
            print(f"  - {f}")
        for e in worker_errors:
            print(f"  - {e}")
        raise SystemExit(1)
    else:
        print(f"\nSYNC EQUIVALENCE TEST PASSED — {total_checks} checks across "
              f"{len(TEST_CASES)} cases x {N_REPEATS} repeats")
    print(f"{'='*60}")
