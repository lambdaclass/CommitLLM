"""
Qwen exact-attention benchmark — measures the 4 key numbers for production policy decisions.

Measures:
  1. Per fully attention-audited token payload (total bytes, broken down)
  2. Verifier CPU time (total, attention replay portion)
  3. Prover overhead (KV transcript commit cost, retained state growth)
  4. Pass/fail behavior (honest pass, tamper fail, skip without KV)

Usage:
    modal run --detach scripts/modal/bench_exact_attention.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-bench-exact-attention")

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
        ".git", "target", "scripts/__pycache__", "*.pdf", "*.md", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

# Prompts that produce enough tokens to measure attention cost
BENCH_PROMPTS = [
    ("short", "What is 2+2? Explain step by step.", 32),
    ("medium", "Explain how a CPU pipeline works in detail.", 128),
    ("long", "Write a detailed tutorial on implementing a B-tree in Rust, covering insertion, deletion, and search operations.", 512),
]


def _run_benchmark():
    import hashlib
    import json
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))

    results = {}

    # ── Step 1: Generate verifier key + decode artifact ──
    print("\n" + "=" * 70)
    print("STEP 1: Keygen")
    print("=" * 70)
    seed = hashlib.sha256(b"bench-exact-attention").digest()
    t0 = time.time()
    key_binary, artifact_binary = verilm_rs.generate_key_binary(model_dir, seed)
    keygen_ms = (time.time() - t0) * 1000
    print(f"  keygen: {keygen_ms:.0f} ms")
    print(f"  key size: {len(key_binary)} bytes ({len(key_binary)/1024/1024:.1f} MB)")
    if artifact_binary:
        print(f"  decode artifact: {len(artifact_binary)} bytes ({len(artifact_binary)/1024/1024:.1f} MB)")
    else:
        print("  decode artifact: None")

    results["keygen"] = {
        "time_ms": keygen_ms,
        "key_bytes": len(key_binary),
        "artifact_bytes": len(artifact_binary) if artifact_binary else 0,
    }

    # ── Step 2: Payload size benchmarks ──
    print("\n" + "=" * 70)
    print("STEP 2: Payload sizes (per-token, attention-audited)")
    print("=" * 70)

    payload_results = []
    for label, prompt, max_tokens in BENCH_PROMPTS:
        print(f"\n  --- {label}: max_tokens={max_tokens} ---")
        result = server.chat(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt
        print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
        print(f"  text preview: {result['generated_text'][:60]}...")

        # Audit token 0 with full layers (includes KV transcript for attention)
        t0 = time.time()
        audit_binary = server.audit(
            request_id=request_id,
            token_index=0,
            layer_indices=full_layers,
            tier="full",
            binary=True,
            use_captured_x_attn=True,
        )
        audit_ms = (time.time() - t0) * 1000
        audit_bytes = len(audit_binary)

        # Also audit without KV (shell-only) for comparison
        audit_shell_only = server.audit(
            request_id=request_id,
            token_index=0,
            layer_indices=full_layers,
            tier="routine",
            binary=True,
            use_captured_x_attn=True,
        )
        shell_only_bytes = len(audit_shell_only)

        # Audit a later token too (to see how payload grows with position)
        late_idx = min(n_gen - 1, max(0, n_gen // 2))
        if late_idx > 0:
            audit_late = server.audit(
                request_id=request_id,
                token_index=late_idx,
                layer_indices=full_layers,
                tier="full",
                binary=True,
                use_captured_x_attn=True,
            )
            late_bytes = len(audit_late)
        else:
            late_bytes = audit_bytes

        entry = {
            "label": label,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "token_0_full_bytes": audit_bytes,
            "token_0_shell_only_bytes": shell_only_bytes,
            "token_0_kv_overhead_bytes": audit_bytes - shell_only_bytes,
            f"token_{late_idx}_full_bytes": late_bytes,
            "audit_open_ms": audit_ms,
        }
        payload_results.append(entry)

        print(f"  token 0 full payload: {audit_bytes:,} bytes ({audit_bytes/1024:.1f} KB)")
        print(f"  token 0 shell-only:   {shell_only_bytes:,} bytes ({shell_only_bytes/1024:.1f} KB)")
        print(f"  KV overhead:          {audit_bytes - shell_only_bytes:,} bytes ({(audit_bytes - shell_only_bytes)/1024:.1f} KB)")
        print(f"  token {late_idx} full payload: {late_bytes:,} bytes ({late_bytes/1024:.1f} KB)")
        print(f"  audit open time: {audit_ms:.1f} ms")

    results["payloads"] = payload_results

    # ── Step 3: Verifier CPU time ──
    print("\n" + "=" * 70)
    print("STEP 3: Verifier CPU time")
    print("=" * 70)

    verify_results = []
    # Use the medium prompt for detailed timing
    result = server.chat(
        prompt="Explain how a CPU pipeline works in detail.",
        max_tokens=128,
        temperature=0.0,
    )
    request_id = result["request_id"]
    n_tokens = result["n_tokens"]
    commitment = result["commitment"]
    n_prompt = commitment.get("n_prompt_tokens", 0)
    n_gen = n_tokens - n_prompt

    for token_idx in [0, min(n_gen - 1, n_gen // 4), min(n_gen - 1, n_gen // 2)]:
        print(f"\n  --- token_index={token_idx} ---")

        # Full audit (with KV for attention)
        audit_full = server.audit(
            request_id=request_id,
            token_index=token_idx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
            use_captured_x_attn=True,
        )

        # Shell-only audit
        audit_shell = server.audit(
            request_id=request_id,
            token_index=token_idx,
            layer_indices=full_layers,
            tier="routine",
            binary=True,
            use_captured_x_attn=True,
        )

        # Time full verification (includes attention replay)
        N_REPS = 5
        full_times = []
        for _ in range(N_REPS):
            t0 = time.time()
            report_full = verilm_rs.verify_v4_full_binary(
                bytes(audit_full), key_binary, artifact_binary
            )
            full_times.append((time.time() - t0) * 1000)

        # Time shell-only verification
        shell_times = []
        for _ in range(N_REPS):
            t0 = time.time()
            report_shell = verilm_rs.verify_v4_full_binary(
                bytes(audit_shell), key_binary, artifact_binary
            )
            shell_times.append((time.time() - t0) * 1000)

        full_median = sorted(full_times)[N_REPS // 2]
        shell_median = sorted(shell_times)[N_REPS // 2]
        attn_overhead = full_median - shell_median

        entry = {
            "token_index": token_idx,
            "full_verify_ms": full_median,
            "shell_only_verify_ms": shell_median,
            "attention_overhead_ms": attn_overhead,
            "full_checks": report_full["checks_run"],
            "full_passed": report_full["checks_passed"],
            "shell_checks": report_shell["checks_run"],
            "full_report_passed": report_full["passed"],
            "full_duration_us": report_full.get("duration_us", 0),
        }
        verify_results.append(entry)

        print(f"  full verify (median of {N_REPS}):  {full_median:.1f} ms  ({report_full['checks_passed']}/{report_full['checks_run']} checks)")
        print(f"  shell-only (median of {N_REPS}):   {shell_median:.1f} ms  ({report_shell['checks_passed']}/{report_shell['checks_run']} checks)")
        print(f"  attention overhead:      {attn_overhead:.1f} ms")
        if not report_full["passed"]:
            print(f"  FAILURES: {report_full.get('failures', [])}")
        skipped = report_full.get("skipped", [])
        if skipped:
            print(f"  skipped: {skipped}")

    results["verify_timing"] = verify_results

    # ── Step 4: Prover overhead (KV transcript commit cost) ──
    print("\n" + "=" * 70)
    print("STEP 4: Prover overhead")
    print("=" * 70)

    prover_results = []
    for label, prompt, max_tokens in BENCH_PROMPTS:
        print(f"\n  --- {label}: max_tokens={max_tokens} ---")

        # Time the full chat (includes KV transcript commit)
        t0 = time.time()
        result = server.chat(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        chat_ms = (time.time() - t0) * 1000

        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt

        # Look at commitment size
        commitment_json = json.dumps(commitment)

        entry = {
            "label": label,
            "n_gen": n_gen,
            "chat_total_ms": chat_ms,
            "ms_per_token": chat_ms / max(n_gen, 1),
            "commitment_json_bytes": len(commitment_json),
            "has_kv_roots": "kv_roots" in commitment,
        }
        prover_results.append(entry)

        print(f"  chat total: {chat_ms:.0f} ms ({n_gen} tokens, {chat_ms/max(n_gen,1):.1f} ms/token)")
        print(f"  commitment JSON: {len(commitment_json)} bytes")
        print(f"  has kv_roots: {'kv_roots' in commitment}")

    results["prover_overhead"] = prover_results

    # ── Step 5: Pass/fail behavior ──
    print("\n" + "=" * 70)
    print("STEP 5: Pass/fail behavior")
    print("=" * 70)

    passfail_results = {}

    # 5a: Honest pass
    print("\n  5a. Honest pass...")
    result = server.chat(prompt="What is gravity?", max_tokens=32, temperature=0.0)
    audit = server.audit(
        request_id=result["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )
    report = verilm_rs.verify_v4_full_binary(bytes(audit), key_binary, artifact_binary)
    passfail_results["honest_pass"] = {
        "passed": report["passed"],
        "checks": f"{report['checks_passed']}/{report['checks_run']}",
        "failures": report.get("failures", []),
        "skipped": report.get("skipped", []),
    }
    print(f"  passed={report['passed']}, checks={report['checks_passed']}/{report['checks_run']}")
    if report.get("skipped"):
        print(f"  skipped: {report['skipped']}")

    # 5b: Tampered payload (flip byte in shell region)
    print("\n  5b. Tampered payload...")
    tampered = bytearray(audit)
    if len(tampered) > 100:
        tampered[100] ^= 0xFF
    report_tamper = verilm_rs.verify_v4_full_binary(bytes(tampered), key_binary, artifact_binary)
    passfail_results["tampered_payload"] = {
        "passed": report_tamper["passed"],
        "checks": f"{report_tamper['checks_passed']}/{report_tamper['checks_run']}",
        "failures": report_tamper.get("failures", [])[:5],
    }
    print(f"  passed={report_tamper['passed']} (expected False)")
    if report_tamper.get("failures"):
        for f in report_tamper["failures"][:3]:
            print(f"    {f}")

    # 5c: Shell-only audit (attention should be skipped, not failed)
    print("\n  5c. Shell-only (no KV, attention skipped)...")
    audit_shell = server.audit(
        request_id=result["request_id"],
        token_index=0,
        layer_indices=full_layers,
        tier="routine",
        binary=True,
        use_captured_x_attn=True,
    )
    report_shell = verilm_rs.verify_v4_full_binary(bytes(audit_shell), key_binary, artifact_binary)
    passfail_results["shell_only_skip"] = {
        "passed": report_shell["passed"],
        "checks": f"{report_shell['checks_passed']}/{report_shell['checks_run']}",
        "skipped": report_shell.get("skipped", []),
    }
    print(f"  passed={report_shell['passed']}, checks={report_shell['checks_passed']}/{report_shell['checks_run']}")
    if report_shell.get("skipped"):
        print(f"  skipped: {report_shell['skipped']}")

    results["pass_fail"] = passfail_results

    # ── Summary ──
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n--- Payload sizes (token 0, full attention audit) ---")
    for p in results["payloads"]:
        print(f"  {p['label']:8s}: total={p['token_0_full_bytes']:>8,} B  shell={p['token_0_shell_only_bytes']:>8,} B  kv_overhead={p['token_0_kv_overhead_bytes']:>8,} B  ({p['n_gen']} gen tokens)")

    print("\n--- Verifier CPU time ---")
    for v in results["verify_timing"]:
        print(f"  token {v['token_index']:3d}: full={v['full_verify_ms']:>7.1f} ms  shell={v['shell_only_verify_ms']:>7.1f} ms  attn_overhead={v['attention_overhead_ms']:>7.1f} ms")

    print("\n--- Prover overhead ---")
    for p in results["prover_overhead"]:
        print(f"  {p['label']:8s}: {p['chat_total_ms']:>7.0f} ms total  ({p['ms_per_token']:.1f} ms/tok, {p['n_gen']} tokens)  kv_roots={p['has_kv_roots']}")

    print("\n--- Pass/fail ---")
    for k, v in results["pass_fail"].items():
        print(f"  {k:25s}: passed={v['passed']}  checks={v['checks']}")

    return results


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_benchmark():
    return _run_benchmark()


@app.local_entrypoint()
def main():
    print("CommitLLM Exact Attention Benchmark — Qwen 7B W8A8")
    print("=" * 70)
    results = run_benchmark.remote()

    import json
    print("\n\nFull results JSON:")
    print(json.dumps(results, indent=2, default=str))
