"""
Qwen score-witness strong-tier benchmark.

Measures whether witnessed pre-softmax scores give a tight enough `a` bound
to serve as the production strong-tier verification path.

Key question: does softmax(GPU_scores) @ committed_V → requantize produce
a tight match against committed `a`, even at later token positions where
exact Q·K^T/√d replay diverges by 100-170?

Measures:
  1. Anchor gap (witnessed vs canonical scores)
  2. a L-inf (witnessed-score replay vs committed a) at token 0 / mid / late
  3. Payload sizes (shell + KV + scores)
  4. Verifier CPU time
  5. Pass/fail through canonical verifier

Usage:
    modal run scripts/modal/bench_score_witness.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-bench-score-witness")

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
        "VERILM_SCORE_WITNESS": "1",
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

    # Check score witnessing is enabled (initialized by server constructor)
    buf = cap.get_capture_buffer()
    print(f"Score witness enabled: {buf._sw_enabled}")
    if not buf._sw_enabled:
        raise RuntimeError("Score witnessing not enabled! Check VERILM_SCORE_WITNESS=1")

    # ── Keygen (binary for verifier, JSON for corridor APIs) ──
    print("\n" + "=" * 70)
    print("KEYGEN")
    print("=" * 70)
    seed = hashlib.sha256(b"bench-score-witness").digest()
    t0 = time.time()
    key_binary, artifact_binary = verilm_rs.generate_key_binary(model_dir, seed)
    keygen_ms = (time.time() - t0) * 1000
    print(f"  binary keygen: {keygen_ms:.0f} ms")
    print(f"  key: {len(key_binary)/1024/1024:.1f} MB, artifact: {len(artifact_binary)/1024/1024:.1f} MB" if artifact_binary else f"  key: {len(key_binary)/1024/1024:.1f} MB, no artifact")

    # JSON key for corridor measurement APIs (same seed = same key)
    t0 = time.time()
    key_json = verilm_rs.generate_key(model_dir, seed)
    print(f"  json keygen: {(time.time() - t0)*1000:.0f} ms")

    results["keygen_ms"] = keygen_ms

    # ── Per-prompt benchmark ──
    prompt_results = []
    for label, prompt, max_tokens in BENCH_PROMPTS:
        print(f"\n{'=' * 70}")
        print(f"PROMPT: {label} (max_tokens={max_tokens})")
        print(f"{'=' * 70}")

        result = server.chat(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt
        print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
        print(f"  text: {result['generated_text'][:60]}...")

        # Check KV roots and score witness
        kv_roots = commitment.get("kv_roots", [])
        print(f"  kv_roots: {len(kv_roots)} layers")

        # Test positions: first gen, mid gen, last gen.
        # token_index is 0-based over ALL committed tokens (prompt + gen).
        # Gen tokens start at index n_prompt.
        #
        # NOTE: score witness only captures Q for the LAST decode step.
        # Only the last gen token will have matching witnessed scores seq_len.
        # Earlier tokens will fail the structural check (expected behavior).
        gen_offsets = [0]
        if n_gen > 4:
            gen_offsets.append(n_gen // 2)
        if n_gen > 2:
            gen_offsets.append(n_gen - 1)
        test_positions = [n_prompt + g for g in gen_offsets]

        position_results = []
        for token_idx in test_positions:
            print(f"\n  --- token_index={token_idx} ---")

            # Audit with captured x_attn + score witness
            t0 = time.time()
            audit_binary = server.audit(
                request_id=request_id,
                token_index=token_idx,
                layer_indices=full_layers,
                tier="full",
                binary=True,
                use_captured_x_attn=True,
            )
            audit_ms = (time.time() - t0) * 1000
            audit_bytes = len(audit_binary)

            # Verify through canonical verifier (will use witnessed scores if available)
            N_REPS = 3
            verify_times = []
            for _ in range(N_REPS):
                t0 = time.time()
                report = verilm_rs.verify_v4_full_binary(
                    bytes(audit_binary), key_binary, artifact_binary
                )
                verify_times.append((time.time() - t0) * 1000)

            verify_median = sorted(verify_times)[N_REPS // 2]

            pos_entry = {
                "token_index": token_idx,
                "audit_bytes": audit_bytes,
                "audit_open_ms": audit_ms,
                "verify_ms": verify_median,
                "passed": report["passed"],
                "checks_run": report["checks_run"],
                "checks_passed": report["checks_passed"],
                "failures": report.get("failures", []),
                "skipped": report.get("skipped", []),
            }
            position_results.append(pos_entry)

            print(f"    payload: {audit_bytes:,} bytes ({audit_bytes/1024:.1f} KB)")
            print(f"    verify: {verify_median:.1f} ms ({report['checks_passed']}/{report['checks_run']} checks)")
            print(f"    passed: {report['passed']}")

            if report.get("failures"):
                # Show attention-related failures
                attn_fails = [f for f in report["failures"] if "attention" in f.lower() or "anchor" in f.lower() or "score" in f.lower()]
                other_fails = [f for f in report["failures"] if f not in attn_fails]
                if attn_fails:
                    print(f"    attention failures ({len(attn_fails)}):")
                    for f in attn_fails[:5]:
                        print(f"      {f}")
                    if len(attn_fails) > 5:
                        print(f"      ... and {len(attn_fails) - 5} more")
                if other_fails:
                    print(f"    other failures ({len(other_fails)}):")
                    for f in other_fails[:3]:
                        print(f"      {f}")

            if report.get("skipped"):
                print(f"    skipped: {report['skipped']}")

        # Also run corridor measurement for witnessed scores to get L-inf details
        # (the verifier only reports pass/fail, corridor gives per-layer L-inf)
        print(f"\n  --- corridor measurement (last generated token) ---")
        last_token = n_prompt + n_gen - 1  # absolute token_index
        audit_last = server.audit(
            request_id=request_id,
            token_index=last_token,
            layer_indices=full_layers,
            tier="full",
            binary=True,
            use_captured_x_attn=True,
        )
        try:
            corridor_ws = json.loads(verilm_rs.measure_corridor_witnessed_scores(
                audit_last, key_json, None,
            ))
            corridor_std = json.loads(verilm_rs.measure_corridor_committed_kv(
                audit_last, key_json, None,
            ))
            print(f"    standard replay L-inf:  {corridor_std['global_linf']}")
            print(f"    witnessed score L-inf:  {corridor_ws['global_linf']}")

            # Anchoring stats
            anchor = json.loads(verilm_rs.verify_witnessed_score_anchoring(
                audit_last, key_json, None,
            ))
            print(f"    anchor gap (f64):       {anchor['global_max_gap']:.4f}")

            try:
                anchor_bf16 = json.loads(verilm_rs.verify_witnessed_score_anchoring_bf16(
                    audit_last, key_json, None,
                ))
                print(f"    anchor gap (bf16):      {anchor_bf16['global_max_gap']:.4f}")
            except Exception as e:
                print(f"    anchor gap (bf16):      error: {e}")
                anchor_bf16 = None

        except Exception as e:
            print(f"    corridor measurement error: {e}")
            corridor_ws = corridor_std = anchor = anchor_bf16 = None

        prompt_entry = {
            "label": label,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "positions": position_results,
            "corridor": {
                "standard_linf": corridor_std["global_linf"] if corridor_std else None,
                "witnessed_linf": corridor_ws["global_linf"] if corridor_ws else None,
                "anchor_gap_f64": anchor["global_max_gap"] if anchor else None,
                "anchor_gap_bf16": anchor_bf16["global_max_gap"] if anchor_bf16 else None,
                "standard_per_layer": corridor_std.get("per_layer_max_linf") if corridor_std else None,
                "witnessed_per_layer": corridor_ws.get("per_layer_max_linf") if corridor_ws else None,
            },
        }
        prompt_results.append(prompt_entry)

    results["prompts"] = prompt_results

    # ── Tamper detection ──
    print(f"\n{'=' * 70}")
    print("TAMPER DETECTION")
    print(f"{'=' * 70}")

    result = server.chat(prompt="What is gravity?", max_tokens=16, temperature=0.0)
    tamper_n_prompt = result["commitment"].get("n_prompt_tokens", 0)
    audit = server.audit(
        request_id=result["request_id"],
        token_index=tamper_n_prompt,  # first gen token
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )

    # Honest
    report_honest = verilm_rs.verify_v4_full_binary(bytes(audit), key_binary, artifact_binary)
    print(f"  honest: passed={report_honest['passed']} ({report_honest['checks_passed']}/{report_honest['checks_run']})")

    # Tampered
    tampered = bytearray(audit)
    if len(tampered) > 100:
        tampered[100] ^= 0xFF
    report_tamper = verilm_rs.verify_v4_full_binary(bytes(tampered), key_binary, artifact_binary)
    print(f"  tampered: passed={report_tamper['passed']} (expected False)")

    results["tamper"] = {
        "honest_passed": report_honest["passed"],
        "honest_checks": f"{report_honest['checks_passed']}/{report_honest['checks_run']}",
        "tampered_passed": report_tamper["passed"],
    }

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for p in results["prompts"]:
        c = p["corridor"]
        print(f"\n  {p['label']} ({p['n_gen']} tokens):")
        print(f"    corridor: standard_linf={c['standard_linf']}  witnessed_linf={c['witnessed_linf']}")
        print(f"    anchor:   f64={c['anchor_gap_f64']:.4f}" + (f"  bf16={c['anchor_gap_bf16']:.4f}" if c['anchor_gap_bf16'] else ""))
        for pos in p["positions"]:
            status = "PASS" if pos["passed"] else "FAIL"
            print(f"    token {pos['token_index']:4d}: {status}  {pos['checks_passed']}/{pos['checks_run']} checks  payload={pos['audit_bytes']/1024:.0f}KB  verify={pos['verify_ms']:.0f}ms")
            if not pos["passed"] and pos["failures"]:
                attn = [f for f in pos["failures"] if "attention" in f.lower() or "anchor" in f.lower()]
                if attn:
                    print(f"             attn failures: {len(attn)}")

    return results


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_benchmark():
    return _run_benchmark()


@app.local_entrypoint()
def main():
    print("CommitLLM Score-Witness Strong-Tier Benchmark — Qwen 7B W8A8")
    print("=" * 70)
    results = run_benchmark.remote()

    import json
    print("\n\nFull results JSON:")
    print(json.dumps(results, indent=2, default=str))
