"""
E2E V4 benchmark: captured x_attn + tier-aware verification.

Measures per-tier:
  - Payload size (JSON, binary)
  - Verifier wall time (µs)
  - Checks run / passed / skipped
  - Score anchor gap (if anchoring enabled)

Runs N iterations for stable timing. Reports mean/p50/p99.

Usage:
    modal run --detach scripts/modal/bench_e2e_v4.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-bench-e2e-v4")

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
        "VERILM_CAPTURE_X_ATTN": "1",
    })
    .pip_install(*VERIFICATION)
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
    "What is the significance of the Turing test?",
    "How does public key cryptography work?",
]
N_WARMUP = 2
N_ITERS = 10
MAX_TOKENS = 32


def _run_bench():
    import hashlib
    import json
    import time
    import statistics

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM, SamplingParams
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir

    seed = hashlib.sha256(PROMPTS[0].encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)
    full_layers = list(range(n_layers))

    # ── Warmup ──
    print(f"\nWarmup ({N_WARMUP} iters)...")
    cap._capture_mode = "minimal"
    for i in range(N_WARMUP):
        r = server.chat(prompt=PROMPTS[0], max_tokens=MAX_TOKENS)
        server.audit(request_id=r["request_id"], token_index=0,
                     layer_indices=full_layers, tier="full", binary=True)

    # ── Collect samples ──
    tiers = ["full", "routine"]
    results = {t: [] for t in tiers}

    print(f"\nBenchmark ({N_ITERS} iters per tier)...")
    for i in range(N_ITERS):
        prompt = PROMPTS[i % len(PROMPTS)]

        for tier in tiers:
            # Chat
            t0 = time.perf_counter()
            r = server.chat(prompt=prompt, max_tokens=MAX_TOKENS)
            chat_ms = (time.perf_counter() - t0) * 1000

            rid = r["request_id"]
            n_tokens = r["n_tokens"]
            commitment = r["commitment"]

            # Determine layers for this tier
            if tier == "full":
                layers = full_layers
                token_idx = 0
            else:
                challenge_seed = hashlib.sha256(f"bench-{i}".encode()).digest()
                challenge = verilm_rs.build_audit_challenge(
                    list(challenge_seed), n_tokens, n_layers, "routine"
                )
                layers = challenge["layer_indices"]
                token_idx = challenge["token_index"]

            # Audit (JSON)
            t0 = time.perf_counter()
            audit_json = server.audit(
                request_id=rid, token_index=token_idx,
                layer_indices=layers, tier=tier, binary=False,
            )
            audit_ms = (time.perf_counter() - t0) * 1000
            json_bytes = len(audit_json.encode("utf-8")) if isinstance(audit_json, str) else len(audit_json)

            # Verify JSON
            report_json = verilm_rs.verify_v4(audit_json, key_json)
            verify_json_us = report_json["duration_us"]

            # Audit again for binary
            r2 = server.chat(prompt=prompt, max_tokens=MAX_TOKENS)
            if tier == "routine":
                challenge = verilm_rs.build_audit_challenge(
                    list(challenge_seed), r2["n_tokens"], n_layers, "routine"
                )
                layers = challenge["layer_indices"]
                token_idx = challenge["token_index"]

            audit_bin = server.audit(
                request_id=r2["request_id"], token_index=token_idx,
                layer_indices=layers, tier=tier, binary=True,
            )
            bin_bytes = len(audit_bin)

            # Verify binary
            report_bin = verilm_rs.verify_v4_binary(bytes(audit_bin), key_json)
            verify_bin_us = report_bin["duration_us"]

            sample = {
                "iter": i,
                "tier": tier,
                "n_tokens": n_tokens,
                "n_layers_opened": len(layers),
                "chat_ms": round(chat_ms, 2),
                "audit_ms": round(audit_ms, 2),
                "json_bytes": json_bytes,
                "binary_bytes": bin_bytes,
                "compression_ratio": round(json_bytes / max(bin_bytes, 1), 2),
                "verify_json_us": verify_json_us,
                "verify_bin_us": verify_bin_us,
                "checks_run": report_json["checks_run"],
                "checks_passed": report_json["checks_passed"],
                "passed": report_json["passed"],
                "skipped": report_json.get("skipped", []),
                "failures": report_json.get("failures", []),
            }
            results[tier].append(sample)
            print(f"  [{tier}][{i}] verify={verify_bin_us}µs json={json_bytes}B bin={bin_bytes}B checks={report_json['checks_passed']}/{report_json['checks_run']} {'PASS' if report_json['passed'] else 'FAIL'}")

    # ── Aggregate ──
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS — {MODEL_ID}")
    print(f"  {N_ITERS} iterations, {MAX_TOKENS} max_tokens, {n_layers} layers")
    print(f"{'='*70}")

    summary = {}
    for tier in tiers:
        samples = results[tier]
        if not samples:
            continue

        verify_us = [s["verify_bin_us"] for s in samples]
        json_bytes = [s["json_bytes"] for s in samples]
        bin_bytes = [s["binary_bytes"] for s in samples]
        checks_run = samples[0]["checks_run"]
        checks_passed = samples[0]["checks_passed"]
        n_opened = samples[0]["n_layers_opened"]
        skipped = samples[0].get("skipped", [])
        all_passed = all(s["passed"] for s in samples)

        tier_summary = {
            "tier": tier,
            "n_layers_opened": n_opened,
            "checks_run": checks_run,
            "checks_passed": checks_passed,
            "all_passed": all_passed,
            "skipped": skipped,
            "verify_us_mean": round(statistics.mean(verify_us)),
            "verify_us_p50": round(statistics.median(verify_us)),
            "verify_us_p99": round(sorted(verify_us)[min(len(verify_us)-1, int(len(verify_us)*0.99))]),
            "json_bytes_mean": round(statistics.mean(json_bytes)),
            "binary_bytes_mean": round(statistics.mean(bin_bytes)),
            "compression_ratio": round(statistics.mean(json_bytes) / max(statistics.mean(bin_bytes), 1), 2),
        }
        summary[tier] = tier_summary

        print(f"\n  [{tier.upper()}] layers={n_opened}/{n_layers}, checks={checks_passed}/{checks_run}, all_passed={all_passed}")
        print(f"    verify (binary):  mean={tier_summary['verify_us_mean']}µs  p50={tier_summary['verify_us_p50']}µs  p99={tier_summary['verify_us_p99']}µs")
        print(f"    verify per token: {round(tier_summary['verify_us_mean'] / max(MAX_TOKENS, 1))}µs/tok")
        print(f"    payload JSON:     mean={tier_summary['json_bytes_mean']}B")
        print(f"    payload binary:   mean={tier_summary['binary_bytes_mean']}B  ({tier_summary['compression_ratio']}x)")
        if skipped:
            print(f"    skipped: {skipped}")

    # ── Per-iteration detail ──
    print(f"\n{'='*70}")
    print("PER-ITERATION DETAIL")
    for tier in tiers:
        for s in results[tier]:
            status = "PASS" if s["passed"] else f"FAIL: {s['failures']}"
            print(f"  [{tier}][{s['iter']}] verify={s['verify_bin_us']}µs json={s['json_bytes']}B bin={s['binary_bytes']}B {status}")

    return summary


@app.function(image=image, gpu="A100-80GB", timeout=1200)
def run_bench():
    return _run_bench()


@app.local_entrypoint()
def main():
    print("VeriLM E2E V4 Benchmark")
    print("=" * 60)
    summary = run_bench.remote()
    print("\nSUMMARY:")
    for tier, s in summary.items():
        print(f"  [{tier}] verify_mean={s['verify_us_mean']}µs payload_bin={s['binary_bytes_mean']}B checks={s['checks_passed']}/{s['checks_run']} passed={s['all_passed']}")
