"""
Llama 3.1 8B W8A8 — shell/decode benchmark.

Measures the current E2E verification surface (everything except
arbitrary-position attention):
  - Binary payload size (audit response bytes)
  - Verifier CPU time (µs, from Rust Duration)
  - Audit-open time (prover-side, ms)
  - Per-token metrics (µs/tok, bytes/tok)
  - Greedy vs sampled decode
  - Routine vs full tiers

Runs N iterations per configuration for stable statistics.
Reports mean / p50 / p99.

Usage:
    modal run --detach scripts/modal/tests/llama/bench_e2e.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("commitllm-bench-llama-e2e")

MODEL_ID = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"

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
        ".git", "target", "**/__pycache__", "*.pyc", "*.pdf", "*.md", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

# Benchmark parameters
PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
    "What is the significance of the Turing test?",
    "How does public key cryptography work?",
    "Write a short function in Python that computes Fibonacci numbers.",
    "What are the key principles of operating system design?",
    "Explain how hash tables work and why they are useful.",
]
N_WARMUP = 2
N_ITERS = 10
MAX_TOKENS_SHORT = 32
MAX_TOKENS_LONG = 256


def _run():
    import hashlib
    import json
    import statistics
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"\nLoading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    print(f"  {n_layers} layers\n")

    # ── Keygen (binary) ──
    print("Generating verifier key (binary)...")
    verifier_secret = hashlib.sha256(b"bench-llama-e2e-secret").digest()
    t0 = time.time()
    key_bin, artifact_bin = verilm_rs.generate_key_binary(model_dir, verifier_secret)
    keygen_ms = (time.time() - t0) * 1000
    print(f"  key: {len(key_bin)/1024/1024:.1f} MB, keygen: {keygen_ms:.0f} ms")
    if artifact_bin:
        print(f"  decode artifact: {len(artifact_bin)/1024/1024:.1f} MB")
    else:
        print("  decode artifact: none")

    key_meta = verilm_rs.inspect_key_binary(key_bin)
    profile = key_meta.get("verification_profile", {})
    print(f"  profile: {profile.get('name', 'unknown')}")
    print(f"  attention: {profile.get('attention_mode', 'unknown')}")
    print(f"  decode: {profile.get('decode_acceptance', 'unknown')}")
    print()

    # ── Challenge helper ──
    def derive_challenge_seed(commitment):
        merkle_root = commitment.get("merkle_root", "")
        io_root = commitment.get("io_root", "")
        return hashlib.sha256(
            f"{merkle_root}:{io_root}".encode() + verifier_secret
        ).digest()

    def build_challenge(commitment, n_tokens, tier, *, generated_only=True):
        challenge_seed = derive_challenge_seed(commitment)
        gen_start = max(commitment.get("n_prompt_tokens", 1) - 1, 0)
        for counter in range(64):
            seed = challenge_seed if counter == 0 else hashlib.sha256(
                challenge_seed + counter.to_bytes(4, "little")
            ).digest()
            challenge = verilm_rs.build_audit_challenge(
                list(seed), n_tokens, n_layers, tier
            )
            if not generated_only or challenge["token_index"] >= gen_start:
                return challenge
        raise RuntimeError("could not derive generated-token challenge")

    # ── Single audit cycle ──
    def run_cycle(prompt, max_tokens, temperature, tier, *, top_k=0, top_p=1.0):
        """Run chat → challenge → audit → verify. Returns metrics dict."""
        t_chat = time.time()
        result = server.chat(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        chat_ms = (time.time() - t_chat) * 1000

        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = len(result.get("token_ids", [])) - n_prompt

        challenge = build_challenge(commitment, result["n_tokens"], tier)
        tok_idx = challenge["token_index"]
        layers = challenge["layer_indices"]

        t_audit = time.time()
        audit = server.audit(
            request_id=result["request_id"],
            token_index=tok_idx,
            layer_indices=layers,
            tier=tier,
            binary=True,
            include_kv=False,
        )
        audit_ms = (time.time() - t_audit) * 1000
        payload_bytes = len(audit)

        t_verify = time.time()
        report = verilm_rs.verify_v4_full_binary(bytes(audit), key_bin, artifact_bin)
        verify_wall_ms = (time.time() - t_verify) * 1000
        verify_us = report["duration_us"]

        return {
            "n_gen": n_gen,
            "n_prompt": n_prompt,
            "n_layers_opened": len(layers),
            "chat_ms": round(chat_ms, 2),
            "audit_ms": round(audit_ms, 2),
            "verify_wall_ms": round(verify_wall_ms, 2),
            "verify_us": verify_us,
            "payload_bytes": payload_bytes,
            "payload_per_tok": round(payload_bytes / max(n_gen, 1)),
            "verify_per_tok_us": round(verify_us / max(n_gen, 1)),
            "checks_run": report["checks_run"],
            "checks_passed": report["checks_passed"],
            "passed": report["passed"],
            "skipped": report.get("skipped", []),
            "failures": report.get("failures", []),
        }

    # ── Warmup ──
    print(f"Warmup ({N_WARMUP} iterations)...")
    for i in range(N_WARMUP):
        m = run_cycle(PROMPTS[0], MAX_TOKENS_SHORT, 0.0, "full")
        print(f"  warmup {i}: {m['verify_us']}µs, {m['payload_bytes']}B, "
              f"{'PASS' if m['passed'] else 'FAIL'}")
    print()

    # ── Benchmark matrix ──
    configs = [
        {"name": "greedy-short-full",    "temp": 0.0, "max_tok": MAX_TOKENS_SHORT, "tier": "full"},
        {"name": "greedy-short-routine",  "temp": 0.0, "max_tok": MAX_TOKENS_SHORT, "tier": "routine"},
        {"name": "greedy-long-full",      "temp": 0.0, "max_tok": MAX_TOKENS_LONG,  "tier": "full"},
        {"name": "greedy-long-routine",   "temp": 0.0, "max_tok": MAX_TOKENS_LONG,  "tier": "routine"},
        {"name": "sampled-short-full",    "temp": 0.8, "max_tok": MAX_TOKENS_SHORT, "tier": "full",
         "top_k": 50, "top_p": 0.9},
        {"name": "sampled-short-routine", "temp": 0.8, "max_tok": MAX_TOKENS_SHORT, "tier": "routine",
         "top_k": 50, "top_p": 0.9},
        {"name": "sampled-long-full",     "temp": 0.8, "max_tok": MAX_TOKENS_LONG,  "tier": "full",
         "top_k": 50, "top_p": 0.9},
        {"name": "sampled-long-routine",  "temp": 0.8, "max_tok": MAX_TOKENS_LONG,  "tier": "routine",
         "top_k": 50, "top_p": 0.9},
    ]

    all_results = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"[{name}] ({N_ITERS} iterations)...")
        samples = []
        for i in range(N_ITERS):
            prompt = PROMPTS[i % len(PROMPTS)]
            m = run_cycle(
                prompt, cfg["max_tok"], cfg["temp"], cfg["tier"],
                top_k=cfg.get("top_k", 0), top_p=cfg.get("top_p", 1.0),
            )
            status = "PASS" if m["passed"] else f"FAIL: {m['failures']}"
            print(f"  [{i}] {m['n_gen']}tok verify={m['verify_us']}µs "
                  f"audit={m['audit_ms']:.0f}ms payload={m['payload_bytes']}B "
                  f"{m['checks_passed']}/{m['checks_run']} {status}")
            samples.append(m)
        all_results[name] = samples
        print()

    # ── Aggregate ──
    print("=" * 78)
    print(f"BENCHMARK RESULTS — {MODEL_ID}")
    print(f"  {N_ITERS} iterations per config, {n_layers} layers")
    print("=" * 78)

    def stat(values):
        if not values:
            return {"mean": 0, "p50": 0, "p99": 0, "min": 0, "max": 0}
        s = sorted(values)
        return {
            "mean": round(statistics.mean(s)),
            "p50": round(statistics.median(s)),
            "p99": round(s[min(len(s)-1, int(len(s)*0.99))]),
            "min": round(min(s)),
            "max": round(max(s)),
        }

    summary = {}
    for name, samples in all_results.items():
        all_passed = all(s["passed"] for s in samples)
        checks = f"{samples[0]['checks_passed']}/{samples[0]['checks_run']}" if samples else "?"
        n_layers_opened = samples[0]["n_layers_opened"] if samples else 0

        verify_us = stat([s["verify_us"] for s in samples])
        audit_ms = stat([s["audit_ms"] for s in samples])
        payload = stat([s["payload_bytes"] for s in samples])
        per_tok_us = stat([s["verify_per_tok_us"] for s in samples])
        per_tok_bytes = stat([s["payload_per_tok"] for s in samples])
        n_gen = stat([s["n_gen"] for s in samples])

        entry = {
            "all_passed": all_passed,
            "checks": checks,
            "n_layers_opened": n_layers_opened,
            "n_gen": n_gen,
            "verify_us": verify_us,
            "audit_ms": audit_ms,
            "payload_bytes": payload,
            "verify_per_tok_us": per_tok_us,
            "payload_per_tok_bytes": per_tok_bytes,
        }
        summary[name] = entry

        print(f"\n  [{name}]  layers={n_layers_opened}/{n_layers}  checks={checks}  passed={all_passed}  gen={n_gen['mean']}tok")
        print(f"    verify:      mean={verify_us['mean']}µs  p50={verify_us['p50']}µs  p99={verify_us['p99']}µs  [{verify_us['min']}–{verify_us['max']}]")
        print(f"    verify/tok:  mean={per_tok_us['mean']}µs/tok")
        print(f"    audit-open:  mean={audit_ms['mean']}ms  p50={audit_ms['p50']}ms  p99={audit_ms['p99']}ms  [{audit_ms['min']}–{audit_ms['max']}]")
        print(f"    payload:     mean={payload['mean']}B ({payload['mean']/1024:.1f}KB)  p50={payload['p50']}B  [{payload['min']}–{payload['max']}]")
        print(f"    payload/tok: mean={per_tok_bytes['mean']}B/tok")
        if samples and samples[0].get("skipped"):
            print(f"    skipped: {samples[0]['skipped']}")

    # ── Compact summary table ──
    print(f"\n{'=' * 78}")
    print("COMPACT SUMMARY")
    print(f"{'config':<28} {'pass':>4} {'verify_µs':>10} {'audit_ms':>9} {'payload_KB':>10} {'µs/tok':>7} {'B/tok':>6}")
    print("-" * 78)
    for name, s in summary.items():
        print(f"{name:<28} {'yes' if s['all_passed'] else 'NO':>4} "
              f"{s['verify_us']['mean']:>10} {s['audit_ms']['mean']:>9} "
              f"{s['payload_bytes']['mean']/1024:>10.1f} "
              f"{s['verify_per_tok_us']['mean']:>7} {s['payload_per_tok_bytes']['mean']:>6}")

    return summary


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_bench():
    return _run()


@app.local_entrypoint()
def main():
    print("CommitLLM Llama E2E Benchmark")
    print("=" * 60)
    summary = run_bench.remote()
    print("\nDone.")
