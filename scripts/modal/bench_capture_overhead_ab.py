"""
A/B Benchmark: CapturedLogits capture overhead isolation.

Measures the delta from capture hooks on vs off:

  1. Generation time delta  (ms/token, capture off vs on)
  2. Commit/finalize delta  (ms, from server.chat − raw generate)
  3. Retained host memory   (bytes/token and bytes/answer)
  4. Open payload by tier   (routine 10-layer vs full all-layer)
  5. Verify time by tier    (routine vs full)

Usage:
    modal run --detach scripts/modal/bench_capture_overhead_ab.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-bench-capture-overhead-ab")

MODELS = {
    "qwen": "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
    "llama": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
}

# ── Images ──────────────────────────────────────────

baseline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*VERIFICATION)
    .env({"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)

capture_image = (
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

# ── Parameters ──────────────────────────────────────

GEN_LENGTHS = [128, 512, 1024]
N_WARMUP = 2
N_ITERS = 5
N_ROUTINE_LAYERS = 10

PROMPTS = [
    "Explain how neural networks learn through backpropagation.",
    "Describe the process of photosynthesis in detail.",
    "What are the key principles of distributed systems?",
    "How does the immune system fight viral infections?",
    "Explain the concept of entropy in information theory.",
    "Describe the history of cryptography from ancient ciphers to modern public-key systems.",
    "What are the main challenges in quantum computing?",
    "Explain how compilers transform source code to machine code.",
]

SAMPLING_PARAMS = {"temperature": 0.8, "top_k": 50, "top_p": 0.9}


# ── Baseline: plain vLLM, no hooks ──────────────────

def _run_baseline(model_name, model_id):
    import statistics
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    print(f"\n{'='*70}")
    print(f"BASELINE [{model_name}] Loading {model_id}...")
    llm = LLM(model=model_id, dtype="auto", max_model_len=4096,
              enforce_eager=True, enable_prefix_caching=False)
    tokenizer = llm.get_tokenizer()

    all_results = []

    for max_tokens in GEN_LENGTHS:
        print(f"\n{'='*70}")
        print(f"BASELINE [{model_name}] max_tokens={max_tokens}")
        print(f"{'='*70}")

        params = SamplingParams(max_tokens=max_tokens, **SAMPLING_PARAMS)

        # Warmup
        for w in range(N_WARMUP):
            prompt = PROMPTS[w % len(PROMPTS)]
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            llm.generate([text], params)
        print(f"  warmup done")

        samples = []
        for i in range(N_ITERS):
            prompt = PROMPTS[i % len(PROMPTS)]
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            t0 = time.perf_counter()
            outputs = llm.generate([text], params)
            gen_ms = (time.perf_counter() - t0) * 1000

            n_gen = len(outputs[0].outputs[0].token_ids)
            ms_per_tok = gen_ms / n_gen if n_gen > 0 else 0

            samples.append({
                "n_gen": n_gen,
                "gen_ms": round(gen_ms, 2),
                "ms_per_tok": round(ms_per_tok, 2),
            })
            print(f"  [{i}] n_gen={n_gen} gen={gen_ms:.0f}ms "
                  f"({ms_per_tok:.1f}ms/tok)")

        ms_per_toks = [s["ms_per_tok"] for s in samples]
        agg = {
            "max_tokens": max_tokens,
            "n_gen_mean": round(statistics.mean(
                [s["n_gen"] for s in samples]), 1),
            "ms_per_tok_mean": round(statistics.mean(ms_per_toks), 2),
            "ms_per_tok_p50": round(
                sorted(ms_per_toks)[len(ms_per_toks) // 2], 2),
        }
        all_results.append(agg)
        print(f"\n  mean: {agg['ms_per_tok_mean']} ms/tok "
              f"(n_gen~{agg['n_gen_mean']})")

    return {"model": model_name, "type": "baseline", "results": all_results}


# ── Capture: full stack with granular timing ────────

def _run_capture(model_name, model_id):
    import hashlib
    import random
    import statistics
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM, SamplingParams
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"\n{'='*70}")
    print(f"CAPTURE [{model_name}] Loading {model_id}...")
    llm = LLM(model=model_id, dtype="auto", max_model_len=4096,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    tokenizer = llm.get_tokenizer()
    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))

    # ── Keygen (offline, timed but not per-token) ──
    profile_name = f"{model_name}-w8a8-captured-logits"
    print(f"\nOffline keygen (profile={profile_name})...")
    seed = hashlib.sha256(
        f"bench-ab-{model_name}".encode()).digest()
    t0 = time.time()
    key_binary, artifact_binary = verilm_rs.generate_key_binary_with_profile(
        model_dir, list(seed), profile_name,
    )
    keygen_s = time.time() - t0
    print(f"  keygen: {keygen_s:.1f}s")

    rng = random.Random(42)
    all_results = []

    for max_tokens in GEN_LENGTHS:
        print(f"\n{'='*70}")
        print(f"CAPTURE [{model_name}] max_tokens={max_tokens}")
        print(f"{'='*70}")

        params = SamplingParams(max_tokens=max_tokens, **SAMPLING_PARAMS)

        # Warmup
        for w in range(N_WARMUP):
            server.chat(prompt=PROMPTS[w % len(PROMPTS)],
                        max_tokens=min(32, max_tokens),
                        **SAMPLING_PARAMS)
        print(f"  warmup done")

        samples = []
        for i in range(N_ITERS):
            prompt = PROMPTS[i % len(PROMPTS)]

            # ── A) llm.generate() with hooks (gen + capture, no commit) ──
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            t0 = time.perf_counter()
            outputs = llm.generate([text], params)
            gen_hooks_ms = (time.perf_counter() - t0) * 1000
            n_gen_raw = len(outputs[0].outputs[0].token_ids)

            # ── B) server.chat() (gen + capture + commit) ──
            t0 = time.perf_counter()
            result = server.chat(prompt=prompt, max_tokens=max_tokens,
                                 **SAMPLING_PARAMS)
            chat_ms = (time.perf_counter() - t0) * 1000

            request_id = result["request_id"]
            n_tokens = result["n_tokens"]
            commitment = result["commitment"]
            n_prompt = commitment.get("n_prompt_tokens", 0)
            n_gen = n_tokens - n_prompt

            if n_gen <= 0:
                print(f"  skip: n_gen={n_gen}")
                continue

            gen_hooks_ms_per_tok = (gen_hooks_ms / n_gen_raw
                                    if n_gen_raw > 0 else 0)
            chat_ms_per_tok = chat_ms / n_gen
            # Approximate commit overhead
            commit_ms = max(0, chat_ms - gen_hooks_ms * (n_gen / n_gen_raw)
                           if n_gen_raw > 0 else 0)

            # ── C) Retained state ──
            captured = server.sampler_hook._captured_logits
            n_captured = len(captured)
            bytes_per_token = captured[0].nbytes if n_captured > 0 else 0
            total_retained = sum(c.nbytes for c in captured)

            # ── D) Audit — routine (10 random layers) ──
            gen_start = n_prompt - 1
            challenge_idx = gen_start + rng.randint(0, n_gen - 1)
            routine_layers = sorted(
                rng.sample(full_layers,
                           min(N_ROUTINE_LAYERS, n_layers)))

            t0 = time.perf_counter()
            audit_routine = server.audit(
                request_id=request_id, token_index=challenge_idx,
                layer_indices=routine_layers, tier="full", binary=True)
            audit_routine_ms = (time.perf_counter() - t0) * 1000

            # ── E) Audit — full (all layers) ──
            t0 = time.perf_counter()
            audit_full = server.audit(
                request_id=request_id, token_index=challenge_idx,
                layer_indices=full_layers, tier="full", binary=True)
            audit_full_ms = (time.perf_counter() - t0) * 1000

            # ── F) Verify — routine ──
            t0 = time.perf_counter()
            report_routine = verilm_rs.verify_v4_full_binary(
                bytes(audit_routine), key_binary, artifact_binary)
            verify_routine_ms = (time.perf_counter() - t0) * 1000

            # ── G) Verify — full ──
            t0 = time.perf_counter()
            report_full = verilm_rs.verify_v4_full_binary(
                bytes(audit_full), key_binary, artifact_binary)
            verify_full_ms = (time.perf_counter() - t0) * 1000

            sample = {
                "n_gen": n_gen,
                "n_gen_raw": n_gen_raw,
                "gen_hooks_ms_per_tok": round(gen_hooks_ms_per_tok, 2),
                "chat_ms_per_tok": round(chat_ms_per_tok, 2),
                "commit_ms": round(commit_ms, 1),
                "bytes_per_token": bytes_per_token,
                "total_retained_kib": round(total_retained / 1024, 1),
                "audit_routine_ms": round(audit_routine_ms, 2),
                "audit_full_ms": round(audit_full_ms, 2),
                "payload_routine_kib": round(
                    len(audit_routine) / 1024, 1),
                "payload_full_kib": round(
                    len(audit_full) / 1024, 1),
                "verify_routine_ms": round(verify_routine_ms, 2),
                "verify_full_ms": round(verify_full_ms, 2),
            }
            samples.append(sample)

            print(
                f"  [{i}] n_gen={n_gen} "
                f"gen_hooks={gen_hooks_ms:.0f}ms "
                f"chat={chat_ms:.0f}ms "
                f"commit~{commit_ms:.0f}ms "
                f"retained={total_retained/1024:.0f}KiB")
            print(
                f"       audit: "
                f"rout={audit_routine_ms:.0f}ms/"
                f"{len(audit_routine)/1024:.0f}KiB "
                f"full={audit_full_ms:.0f}ms/"
                f"{len(audit_full)/1024:.0f}KiB")
            print(
                f"       verify: "
                f"rout={verify_routine_ms:.0f}ms "
                f"full={verify_full_ms:.0f}ms")

        if not samples:
            continue

        def avg(vals):
            return round(statistics.mean(vals), 2) if vals else 0

        agg = {
            "max_tokens": max_tokens,
            "n_gen_mean": round(
                statistics.mean([s["n_gen"] for s in samples]), 1),
            "gen_hooks_ms_per_tok": avg(
                [s["gen_hooks_ms_per_tok"] for s in samples]),
            "chat_ms_per_tok": avg(
                [s["chat_ms_per_tok"] for s in samples]),
            "commit_ms": avg(
                [s["commit_ms"] for s in samples]),
            "bytes_per_token": samples[0]["bytes_per_token"],
            "total_retained_kib": avg(
                [s["total_retained_kib"] for s in samples]),
            "audit_routine_ms": avg(
                [s["audit_routine_ms"] for s in samples]),
            "audit_full_ms": avg(
                [s["audit_full_ms"] for s in samples]),
            "payload_routine_kib": avg(
                [s["payload_routine_kib"] for s in samples]),
            "payload_full_kib": avg(
                [s["payload_full_kib"] for s in samples]),
            "verify_routine_ms": avg(
                [s["verify_routine_ms"] for s in samples]),
            "verify_full_ms": avg(
                [s["verify_full_ms"] for s in samples]),
        }
        all_results.append(agg)

        print(f"\n  --- max_tokens={max_tokens} summary ---")
        print(f"  gen+hooks: {agg['gen_hooks_ms_per_tok']} ms/tok")
        print(f"  chat+commit: {agg['chat_ms_per_tok']} ms/tok")
        print(f"  commit: ~{agg['commit_ms']} ms")
        print(f"  retained: {agg['total_retained_kib']} KiB total "
              f"({agg['bytes_per_token']/1024:.0f} KiB/tok)")
        print(f"  audit: rout={agg['audit_routine_ms']}ms "
              f"full={agg['audit_full_ms']}ms")
        print(f"  payload: rout={agg['payload_routine_kib']}KiB "
              f"full={agg['payload_full_kib']}KiB")
        print(f"  verify: rout={agg['verify_routine_ms']}ms "
              f"full={agg['verify_full_ms']}ms")

    return {
        "model": model_name,
        "type": "capture",
        "keygen_s": keygen_s,
        "key_mib": round(len(key_binary) / 1024 / 1024, 1),
        "artifact_mib": round(
            len(artifact_binary) / 1024 / 1024, 1
        ) if artifact_binary else 0,
        "results": all_results,
    }


# ── Modal functions ─────────────────────────────────

@app.function(image=baseline_image, gpu="A100-80GB", timeout=3600)
def baseline_qwen():
    return _run_baseline("qwen", MODELS["qwen"])


@app.function(image=baseline_image, gpu="A100-80GB", timeout=3600)
def baseline_llama():
    return _run_baseline("llama", MODELS["llama"])


@app.function(image=capture_image, gpu="A100-80GB", timeout=3600)
def capture_qwen():
    return _run_capture("qwen", MODELS["qwen"])


@app.function(image=capture_image, gpu="A100-80GB", timeout=3600)
def capture_llama():
    return _run_capture("llama", MODELS["llama"])


# ── Entrypoint ──────────────────────────────────────

@app.local_entrypoint()
def main():
    print("CapturedLogits A/B Capture Overhead Benchmark")
    print("=" * 70)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Gen lengths: {GEN_LENGTHS}")
    print(f"Iterations per length: {N_ITERS}")
    print()

    # Spawn all 4 in parallel.
    bq = baseline_qwen.spawn()
    bl = baseline_llama.spawn()
    cq = capture_qwen.spawn()
    cl = capture_llama.spawn()

    bq_r = bq.get()
    bl_r = bl.get()
    cq_r = cq.get()
    cl_r = cl.get()

    # ── Combined comparison ──
    print("\n" + "=" * 70)
    print("A/B CAPTURE OVERHEAD RESULTS")
    print("=" * 70)

    for model_name in ["qwen", "llama"]:
        base = bq_r if model_name == "qwen" else bl_r
        capt = cq_r if model_name == "qwen" else cl_r

        print(f"\n{'─'*70}")
        print(f"  {model_name.upper()}")
        if capt["type"] == "capture":
            print(f"  Key: {capt['key_mib']} MiB  "
                  f"Artifact: {capt['artifact_mib']} MiB  "
                  f"Keygen: {capt['keygen_s']:.1f}s")
        print(f"{'─'*70}")

        # ── Generation timing comparison ──
        print(f"\n  GENERATION TIMING (ms/token)")
        print(f"  {'tokens':>8} {'Baseline':>10} {'Gen+Hook':>10} "
              f"{'Chat+Cmt':>10} │ {'Hook Δ':>8} {'Hook%':>7} "
              f"{'Commit':>8}")
        print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} "
              f"│ {'─'*8} {'─'*7} {'─'*8}")

        for b in base["results"]:
            mt = b["max_tokens"]
            c = next((r for r in capt["results"]
                      if r["max_tokens"] == mt), None)
            if not c:
                continue

            bms = b["ms_per_tok_mean"]
            hms = c["gen_hooks_ms_per_tok"]
            cms = c["chat_ms_per_tok"]
            delta = hms - bms
            pct = (delta / bms * 100) if bms > 0 else 0
            commit = c["commit_ms"]

            print(f"  {mt:>8} {bms:>10.1f} {hms:>10.1f} "
                  f"{cms:>10.1f} │ {delta:>+8.1f} {pct:>+6.1f}% "
                  f"{commit:>8.0f}")

        # ── Retained state ──
        print(f"\n  RETAINED STATE")
        for c in capt["results"]:
            kib_tok = c["bytes_per_token"] / 1024
            total = c["total_retained_kib"]
            total_mib = total / 1024
            print(f"  {c['max_tokens']:>8} tokens: "
                  f"{kib_tok:.0f} KiB/tok × {c['n_gen_mean']:.0f} = "
                  f"{total_mib:.1f} MiB retained")

        # ── Payload by tier ──
        print(f"\n  OPEN PAYLOAD (per challenged token)")
        print(f"  {'tokens':>8} {'Routine':>12} {'Full':>12}")
        print(f"  {'─'*8} {'─'*12} {'─'*12}")
        for c in capt["results"]:
            rk = c["payload_routine_kib"]
            fk = c["payload_full_kib"]
            print(f"  {c['max_tokens']:>8} "
                  f"{rk:>10.0f}Ki {fk:>10.0f}Ki")

        # ── Audit time by tier ──
        print(f"\n  AUDIT TIME (ms)")
        print(f"  {'tokens':>8} {'Routine':>10} {'Full':>10}")
        print(f"  {'─'*8} {'─'*10} {'─'*10}")
        for c in capt["results"]:
            print(f"  {c['max_tokens']:>8} "
                  f"{c['audit_routine_ms']:>10.0f} "
                  f"{c['audit_full_ms']:>10.0f}")

        # ── Verify time by tier ──
        print(f"\n  VERIFY TIME (ms)")
        print(f"  {'tokens':>8} {'Routine':>10} {'Full':>10}")
        print(f"  {'─'*8} {'─'*10} {'─'*10}")
        for c in capt["results"]:
            print(f"  {c['max_tokens']:>8} "
                  f"{c['verify_routine_ms']:>10.0f} "
                  f"{c['verify_full_ms']:>10.0f}")

    print(f"\n{'='*70}")
