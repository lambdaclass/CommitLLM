"""
Benchmark Gate A.5: real-model regression checkpoint.

Measures the full-bridge + embedding-proof work against the last minimal-path numbers.
Same model/GPU/prompt as modal_bench_capture.py, but focused on:

  1. Baseline latency (inference only, no capture)
  2. Minimal online path latency (capture + V4 commit)
  3. Audit open time (open_v4 with full bridge + embedding proof)
  4. Verify time (verify_v4, key-only)
  5. Audit payload size (JSON bytes)
  6. Retained bytes/token (bincode)

Usage:
    modal run --detach scripts/modal_bench_gate_a5.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pins import VERIFICATION

import modal

app = modal.App("verilm-bench-gate-a5")

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


def _run_bench():
    import json
    import os
    import time
    import hashlib

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    import verilm_rs
    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    buf = get_capture_buffer()

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True, enable_prefix_caching=False)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    server = VerifiedInferenceServer(llm)

    n_layers = cap._n_layers
    calls_per_fwd = n_layers * cap.PROJS_PER_LAYER
    model_dir = server._model_dir

    seed = hashlib.sha256(PROMPT.encode()).digest()
    manifest = {
        "tokenizer_hash": server._tokenizer_hash,
        "temperature": 0.0, "top_k": 0, "top_p": 1.0,
        "eos_policy": "stop",
        "weight_hash": server._weight_hash,
        "quant_hash": server._quant_hash,
        "system_prompt_hash": server._system_prompt_hash,
    }

    # Warmup
    print(f"\nWarmup: {N_WARMUP} iterations...")
    cap._capture_mode = "minimal"
    buf.enabled = True
    for _ in range(N_WARMUP):
        buf.drain()
        server.el_capture.drain()
        llm.generate([PROMPT], params)
        buf.drain()
        server.el_capture.drain()
    print("Warmup done.\n")

    def stats(times):
        times = sorted(times)
        n = len(times)
        return sum(times) / n, times[n // 2]

    def _extract_scale(s):
        if hasattr(s, 'numel') and s.numel() > 1:
            return float(s.max().item())
        elif hasattr(s, 'item'):
            return float(s.item())
        return float(s)

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
    last_chat_result = None
    last_request_id = None
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        result = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
        min_pipe_times.append(time.monotonic() - t0)
        last_chat_result = result
        last_request_id = result["request_id"]

    n_tokens = last_chat_result["n_tokens"]
    token_ids = last_chat_result["token_ids"]

    # ── 3. Audit open time (V4, includes SafetensorsWeightProvider load + full bridge) ──
    print(f"3. Audit open (V4 full bridge): {N_ITERS} iterations (token 0)...")
    # Perform one chat to get fresh state for audit
    cap._capture_mode = "minimal"
    chat_for_audit = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
    audit_request_id = chat_for_audit["request_id"]
    audit_n_tokens = chat_for_audit["n_tokens"]

    # Time the audit open (this calls open_v4 with full bridge)
    audit_open_times = []
    audit_json = None
    for i in range(N_ITERS):
        # Re-chat each time since audit consumes state
        if i > 0:
            chat_for_audit = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS)
            audit_request_id = chat_for_audit["request_id"]

        t0 = time.monotonic()
        audit_result = server.audit(
            request_id=audit_request_id,
            token_index=0,
            layer_indices=list(range(n_layers)),
            tier="routine",
            binary=False,
        )
        audit_open_times.append(time.monotonic() - t0)
        audit_json = audit_result  # JSON string from V4 path

    # ── 4. Verify time (key-only, no weights) ──
    # We need to run the Rust verifier. Use verilm_rs.
    print(f"4. Verify (key-only): measuring...")

    # Generate a verifier key
    key_gen_t0 = time.monotonic()
    key_json = verilm_rs.generate_key(model_dir, seed)
    key_gen_time = time.monotonic() - key_gen_t0
    print(f"   Key generation: {key_gen_time*1000:.1f} ms")

    # Verify the audit response
    verify_times = []
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        report = verilm_rs.verify_v4(audit_json, key_json)
        verify_times.append(time.monotonic() - t0)

    # ── 5. Payload sizes ──
    payload_bytes = len(audit_json.encode("utf-8")) if isinstance(audit_json, str) else len(audit_json)

    # Parse to get retained state size
    audit_parsed = json.loads(audit_json) if isinstance(audit_json, str) else audit_json
    retained_json_bytes = len(json.dumps(audit_parsed.get("retained", {})).encode())

    # Check if full bridge was used
    shell = audit_parsed.get("shell_opening")
    has_initial_residual = shell is not None and shell.get("initial_residual") is not None
    has_embedding_proof = shell is not None and shell.get("embedding_proof") is not None

    # ── Report ──
    b_mean, b_med = stats(baseline_times)
    mp_mean, mp_med = stats(min_pipe_times)
    ao_mean, ao_med = stats(audit_open_times)
    vf_mean, vf_med = stats(verify_times)

    min_oh = ((mp_mean - b_mean) / b_mean) * 100

    print(f"\n{'='*72}")
    print("BENCHMARK GATE A.5  —  Real-Model Regression Checkpoint")
    print(f"{'='*72}")
    print(f"Model:     {MODEL_ID}")
    print(f"GPU:       A100-80GB")
    print(f"Tokens:    {n_tokens} generated ({MAX_TOKENS} max)")
    print(f"Iters:     {N_ITERS} (after {N_WARMUP} warmup)")
    print(f"Full bridge: initial_residual={'yes' if has_initial_residual else 'no'}, "
          f"embedding_proof={'yes' if has_embedding_proof else 'no'}")

    print(f"\n--- Latency (ms) ---")
    print(f"{'Metric':<40} {'Mean':>10} {'Median':>10} {'OH%':>8}")
    print(f"{'-'*68}")
    print(f"{'1. Baseline (no capture)':<40} {b_mean*1000:>10.1f} {b_med*1000:>10.1f} {'—':>8}")
    print(f"{'2. Minimal pipeline (V4 commit)':<40} {mp_mean*1000:>10.1f} {mp_med*1000:>10.1f} {min_oh:>+7.1f}%")
    print(f"{'3. Audit open (full bridge + emb proof)':<40} {ao_mean*1000:>10.1f} {ao_med*1000:>10.1f}")
    print(f"{'4. Verify V4 (key-only)':<40} {vf_mean*1000:>10.1f} {vf_med*1000:>10.1f}")
    print(f"{'   Key generation (one-time)':<40} {key_gen_time*1000:>10.1f}")

    print(f"\n--- Payload ---")
    print(f"{'Metric':<40} {'Value':>10} {'Unit':>10}")
    print(f"{'-'*60}")
    print(f"{'5. Audit payload (JSON)':<40} {payload_bytes:>10} {'bytes':>10}")
    print(f"{'   Retained state (JSON)':<40} {retained_json_bytes:>10} {'bytes':>10}")
    print(f"{'   Retained/token (JSON, est.)':<40} {retained_json_bytes // max(n_layers, 1):>10} {'bytes':>10}")
    print(f"{'6. Tokens generated':<40} {n_tokens:>10}")
    print(f"{'   Layers':<40} {n_layers:>10}")
    print(f"{'='*72}")

    return {
        "model": MODEL_ID,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "baseline_med_ms": b_med * 1000,
        "minimal_pipeline_med_ms": mp_med * 1000,
        "minimal_oh_pct": min_oh,
        "audit_open_med_ms": ao_med * 1000,
        "verify_med_ms": vf_med * 1000,
        "key_gen_ms": key_gen_time * 1000,
        "payload_bytes": payload_bytes,
        "retained_json_bytes": retained_json_bytes,
        "has_full_bridge": has_initial_residual,
        "has_embedding_proof": has_embedding_proof,
    }


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_bench():
    return _run_bench()


@app.local_entrypoint()
def main():
    print("VeriLM Benchmark Gate A.5 — Real-Model Regression Checkpoint")
    print("=" * 72)
    result = run_bench.remote()
    print(f"\nMinimal pipeline overhead: {result['minimal_oh_pct']:+.1f}%")
    print(f"Audit open (full bridge):  {result['audit_open_med_ms']:.1f} ms")
    print(f"Verify V4 (key-only):      {result['verify_med_ms']:.1f} ms")
    print(f"Full bridge active:        {result['has_full_bridge']}")
    print(f"Embedding proof present:   {result['has_embedding_proof']}")
