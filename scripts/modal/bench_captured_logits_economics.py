"""
Benchmark: CapturedLogits steady-state economics.

Measures the four cost numbers for CapturedLogits decode verification:

  1. Capture overhead: ms/token (compare sampler hook with vs without logit capture)
  2. Retained state: bytes/token committed, bytes total for the answer
  3. Open payload: bytes per challenged (opened) token
  4. Verifier time: ms per opened token (from verify report duration_us)

Keys/artifacts are generated once (offline, not timed) then reused.
Both Llama and Qwen. Greedy and sampled. Multiple gen lengths.

Usage:
    modal run --detach scripts/modal/bench_captured_logits_economics.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-bench-captured-logits-economics")

MODELS = {
    "qwen": "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
    "llama": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
}

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
        "VERILM_CAPTURE_X_ATTN": "1",
    })
    .pip_install(*VERIFICATION)
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf", "*.md", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

# Different gen lengths to measure scaling.
GEN_LENGTHS = [16, 32, 64, 128]
N_WARMUP = 2
N_ITERS = 5  # per gen length

PROMPTS = [
    "Explain how neural networks learn through backpropagation.",
    "Describe the process of photosynthesis in detail.",
    "What are the key principles of distributed systems?",
    "How does the immune system fight viral infections?",
    "Explain the concept of entropy in information theory.",
]


def _run_bench(model_name, model_id):
    import hashlib
    import random
    import statistics
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"\n{'='*70}")
    print(f"Loading {model_id}...")
    llm = LLM(model=model_id, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))

    # ── Step 1: Offline keygen (timed but not counted in per-token cost) ──
    profile_name = f"{model_name}-w8a8-captured-logits"
    print(f"\n1. Offline keygen (profile={profile_name})...")
    seed = hashlib.sha256(f"bench-economics-{model_name}".encode()).digest()
    t0 = time.time()
    key_binary, artifact_binary = verilm_rs.generate_key_binary_with_profile(
        model_dir, list(seed), profile_name,
    )
    keygen_s = time.time() - t0
    print(f"  keygen: {keygen_s:.1f}s")
    print(f"  key: {len(key_binary)} bytes ({len(key_binary)/1024/1024:.1f} MiB)")
    if artifact_binary:
        print(f"  artifact: {len(artifact_binary)} bytes ({len(artifact_binary)/1024/1024:.1f} MiB)")

    # ── Step 2: Warmup ──
    print(f"\n2. Warmup ({N_WARMUP} iters)...")
    for i in range(N_WARMUP):
        r = server.chat(prompt=PROMPTS[0], max_tokens=32, temperature=0.8, top_k=50, top_p=0.9)
        n_prompt = r["commitment"].get("n_prompt_tokens", 0)
        server.audit(request_id=r["request_id"], token_index=n_prompt - 1,
                     layer_indices=full_layers, tier="full", binary=True)

    # ── Step 3: Benchmark per gen length ──
    rng = random.Random(12345)
    all_results = []

    for max_tokens in GEN_LENGTHS:
        print(f"\n{'='*70}")
        print(f"3. [{model_name}] max_tokens={max_tokens}, {N_ITERS} iterations")
        print(f"{'='*70}")

        samples = []
        for i in range(N_ITERS):
            prompt = PROMPTS[i % len(PROMPTS)]

            # ── 3a. Chat (includes capture overhead) ──
            t_chat_start = time.perf_counter()
            result = server.chat(
                prompt=prompt, max_tokens=max_tokens,
                temperature=0.8, top_k=50, top_p=0.9,
            )
            chat_ms = (time.perf_counter() - t_chat_start) * 1000

            request_id = result["request_id"]
            n_tokens = result["n_tokens"]
            commitment = result["commitment"]
            n_prompt = commitment.get("n_prompt_tokens", 0)
            n_gen = n_tokens - n_prompt
            token_ids = result["token_ids"]

            if n_gen <= 0:
                print(f"  skip: n_gen={n_gen}")
                continue

            # ── 3b. Measure retained state size ──
            # The captured_logits list has n_gen entries, each vocab_size f32s.
            captured = server.sampler_hook._captured_logits
            n_captured = len(captured)
            if n_captured > 0:
                logit_bytes_per_token = captured[0].nbytes  # vocab_size * 4
            else:
                logit_bytes_per_token = 0
            total_logit_bytes = sum(c.nbytes for c in captured)

            # ── 3c. Audit: open a random gen token ──
            gen_start = n_prompt - 1
            gen_offset = rng.randint(0, n_gen - 1)
            challenge_idx = gen_start + gen_offset

            t_audit_start = time.perf_counter()
            audit_binary = server.audit(
                request_id=request_id,
                token_index=challenge_idx,
                layer_indices=full_layers,
                tier="full",
                binary=True,
            )
            audit_ms = (time.perf_counter() - t_audit_start) * 1000
            payload_bytes = len(audit_binary)

            # ── 3d. Verify ──
            t_verify_start = time.perf_counter()
            report = verilm_rs.verify_v4_full_binary(
                bytes(audit_binary), key_binary, artifact_binary
            )
            verify_ms = (time.perf_counter() - t_verify_start) * 1000
            verify_us = report["duration_us"]

            # Classify failures.
            attn_failures = [f for f in report.get("failures", []) if "attention mismatch" in f]
            decode_failures = [f for f in report.get("failures", []) if "attention mismatch" not in f]

            sample = {
                "iter": i,
                "max_tokens": max_tokens,
                "n_prompt": n_prompt,
                "n_gen": n_gen,
                "n_captured": n_captured,
                "chat_ms": round(chat_ms, 2),
                "chat_ms_per_token": round(chat_ms / n_gen, 2),
                "audit_ms": round(audit_ms, 2),
                "payload_bytes": payload_bytes,
                "payload_kib": round(payload_bytes / 1024, 1),
                "logit_bytes_per_token": logit_bytes_per_token,
                "total_logit_bytes": total_logit_bytes,
                "total_logit_kib": round(total_logit_bytes / 1024, 1),
                "verify_ms": round(verify_ms, 2),
                "verify_us": verify_us,
                "checks_run": report["checks_run"],
                "checks_passed": report["checks_passed"],
                "decode_passed": len(decode_failures) == 0,
                "n_attn_failures": len(attn_failures),
                "n_decode_failures": len(decode_failures),
                "challenge_gen_offset": gen_offset,
            }
            samples.append(sample)

            print(f"  [{i}] n_gen={n_gen} chat={chat_ms:.0f}ms "
                  f"audit={audit_ms:.0f}ms payload={payload_bytes/1024:.0f}KiB "
                  f"verify={verify_ms:.1f}ms "
                  f"logits_retained={total_logit_bytes/1024:.0f}KiB "
                  f"decode={'OK' if len(decode_failures) == 0 else 'FAIL'}")

        if not samples:
            continue

        # ── Aggregate stats per gen length ──
        def stat(vals):
            if not vals:
                return {"mean": 0, "p50": 0, "p99": 0}
            s = sorted(vals)
            return {
                "mean": round(statistics.mean(s), 2),
                "p50": round(s[len(s) // 2], 2),
                "p99": round(s[int(len(s) * 0.99)], 2),
            }

        n_gens = [s["n_gen"] for s in samples]
        agg = {
            "model": model_name,
            "max_tokens": max_tokens,
            "n_gen": stat(n_gens),
            "chat_ms_per_token": stat([s["chat_ms_per_token"] for s in samples]),
            "payload_kib": stat([s["payload_kib"] for s in samples]),
            "logit_kib_per_token": stat([round(s["logit_bytes_per_token"] / 1024, 1) for s in samples]),
            "total_logit_kib": stat([s["total_logit_kib"] for s in samples]),
            "verify_ms": stat([s["verify_ms"] for s in samples]),
            "audit_ms": stat([s["audit_ms"] for s in samples]),
            "decode_pass_rate": f"{sum(1 for s in samples if s['decode_passed'])}/{len(samples)}",
        }
        all_results.append(agg)

        print(f"\n  --- max_tokens={max_tokens} summary ---")
        print(f"  n_gen:              {agg['n_gen']}")
        print(f"  chat ms/token:      {agg['chat_ms_per_token']}")
        print(f"  payload (KiB):      {agg['payload_kib']}")
        print(f"  logits/token (KiB): {agg['logit_kib_per_token']}")
        print(f"  total logits (KiB): {agg['total_logit_kib']}")
        print(f"  verify (ms):        {agg['verify_ms']}")
        print(f"  audit (ms):         {agg['audit_ms']}")
        print(f"  decode pass rate:   {agg['decode_pass_rate']}")

    # ── Final summary table ──
    print(f"\n{'='*70}")
    print(f"[{model_name}] CapturedLogits Economics Summary")
    print(f"{'='*70}")
    print(f"  Keygen (offline, one-time): {keygen_s:.1f}s")
    print(f"  Key size: {len(key_binary)/1024/1024:.1f} MiB")
    if artifact_binary:
        print(f"  Artifact size: {len(artifact_binary)/1024/1024:.1f} MiB")
    print()
    print(f"  {'max_tok':>8} {'n_gen':>6} {'chat/tok':>10} {'payload':>10} "
          f"{'logits/tok':>12} {'verify':>10} {'audit':>10} {'decode':>8}")
    print(f"  {'':>8} {'':>6} {'ms':>10} {'KiB':>10} {'KiB':>12} {'ms':>10} {'ms':>10} {'':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    for r in all_results:
        print(f"  {r['max_tokens']:>8} "
              f"{r['n_gen']['mean']:>6.0f} "
              f"{r['chat_ms_per_token']['mean']:>10.1f} "
              f"{r['payload_kib']['mean']:>10.0f} "
              f"{r['logit_kib_per_token']['mean']:>12.0f} "
              f"{r['verify_ms']['mean']:>10.1f} "
              f"{r['audit_ms']['mean']:>10.1f} "
              f"{r['decode_pass_rate']:>8}")
    print(f"{'='*70}")

    return {
        "model": model_name,
        "keygen_s": keygen_s,
        "key_bytes": len(key_binary),
        "artifact_bytes": len(artifact_binary) if artifact_binary else 0,
        "results": all_results,
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_qwen():
    return _run_bench("qwen", MODELS["qwen"])


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_llama():
    return _run_bench("llama", MODELS["llama"])


@app.local_entrypoint()
def main():
    print("CapturedLogits Economics Benchmark")
    print("=" * 70)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Gen lengths: {GEN_LENGTHS}")
    print(f"Iterations per length: {N_ITERS}")
    print()

    qwen_future = run_qwen.spawn()
    llama_future = run_llama.spawn()

    qwen_result = qwen_future.get()
    llama_result = llama_future.get()

    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    for result in [qwen_result, llama_result]:
        model = result["model"]
        print(f"\n{model}:")
        print(f"  keygen: {result['keygen_s']:.1f}s (offline, one-time)")
        print(f"  key: {result['key_bytes']/1024/1024:.1f} MiB, "
              f"artifact: {result['artifact_bytes']/1024/1024:.1f} MiB")
        for r in result["results"]:
            n = r["n_gen"]["mean"]
            print(f"  gen={n:.0f}: "
                  f"chat={r['chat_ms_per_token']['mean']:.1f}ms/tok, "
                  f"payload={r['payload_kib']['mean']:.0f}KiB/opened, "
                  f"logits={r['logit_kib_per_token']['mean']:.0f}KiB/tok retained, "
                  f"verify={r['verify_ms']['mean']:.1f}ms, "
                  f"decode={r['decode_pass_rate']}")
