"""
E2E test: sampled decoding through the live HTTP/FastAPI server path.

Unlike test_e2e_v4.py (which calls VerifiedInferenceServer directly),
this test exercises the full HTTP boundary:

  1. POST /chat  — greedy and sampled, through FastAPI request parsing
  2. POST /audit — JSON and binary wire formats, through HTTP response serialization
  3. verilm_rs.verify_v4() / verify_v4_binary() — key-only verification

Also tests HTTP error paths: missing fields, unknown request_id, etc.

Usage:
    modal run --detach scripts/modal/test_e2e_http.py
"""

import modal

app = modal.App("verilm-test-e2e-http")

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
    .pip_install("vllm>=0.8", "torch", "numpy", "fastapi", "httpx", "maturin")
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

PROMPT = "What is the capital of France?"
MAX_TOKENS = 16


def _run_test():
    import hashlib
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from fastapi.testclient import TestClient
    from vllm import LLM

    from verilm import capture as cap
    from verilm.server import create_app

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)

    # Create FastAPI app and test client.
    # create_app() instantiates its own VerifiedInferenceServer internally.
    # TestClient wraps httpx + ASGI transport correctly for sync usage.
    fastapi_app = create_app(llm)
    http = TestClient(fastapi_app)

    n_layers = cap._n_layers
    # Resolve model_dir for keygen (same logic as server.__init__).
    model_id = llm.llm_engine.model_config.model
    if os.path.isdir(model_id):
        model_dir = model_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_id)

    # Generate verifier key.
    seed = hashlib.sha256(PROMPT.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)

    failures = []
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK: {msg}")

    # ── 1. Health endpoint ──
    print("\n1. GET /health")
    resp = http.get("/health")
    ok(resp.status_code == 200, f"health status={resp.status_code}")
    ok(resp.json().get("status") == "ok", "health body is ok")

    # ── 2. Greedy chat via HTTP ──
    print("\n2. POST /chat (greedy, temperature=0)")
    resp = http.post("/chat", json={
        "prompt": PROMPT,
        "n_tokens": MAX_TOKENS,
        "temperature": 0.0,
    })
    ok(resp.status_code == 200, f"greedy chat status={resp.status_code}")
    chat_greedy = resp.json()
    ok(chat_greedy.get("n_tokens", 0) > 0, f"greedy generated {chat_greedy.get('n_tokens')} tokens")
    ok(chat_greedy.get("commitment", {}).get("version") == "V4", "commitment is V4")
    ok("request_id" in chat_greedy, "response has request_id")
    ok("generated_text" in chat_greedy, "response has generated_text")
    print(f"    text: {chat_greedy.get('generated_text', '')[:80]}")

    # ── 3. Audit greedy via HTTP (JSON) ──
    print("\n3. POST /audit (greedy, JSON)")
    request_id = chat_greedy["request_id"]
    resp = http.post("/audit", json={
        "request_id": request_id,
        "token_index": 0,
        "layer_indices": list(range(n_layers)),
        "tier": "full",
        "binary": False,
    })
    ok(resp.status_code == 200, f"audit status={resp.status_code}")
    audit_greedy_json = resp.text
    audit_greedy = json.loads(audit_greedy_json)
    ok("shell_opening" in audit_greedy, "audit has shell_opening")
    ok(audit_greedy.get("shell_opening", {}).get("initial_residual") is not None,
       "audit has initial_residual (full bridge)")

    # ── 4. Verify greedy audit ──
    print("\n4. Verify greedy (JSON)")
    report = verilm_rs.verify_v4(audit_greedy_json, key_json)
    ok(report["passed"], f"greedy verify passed ({report['checks_passed']}/{report['checks_run']})")
    if not report["passed"]:
        print(f"    failures: {report['failures'][:3]}")

    # ── 5. Sampled chat via HTTP ──
    print("\n5. POST /chat (sampled, temperature=0.8)")
    resp = http.post("/chat", json={
        "prompt": PROMPT,
        "n_tokens": MAX_TOKENS,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
    })
    ok(resp.status_code == 200, f"sampled chat status={resp.status_code}")
    chat_sampled = resp.json()
    ok(chat_sampled.get("n_tokens", 0) > 0, f"sampled generated {chat_sampled.get('n_tokens')} tokens")
    print(f"    text: {chat_sampled.get('generated_text', '')[:80]}")

    # ── 6. Audit sampled via HTTP (JSON) + verify ──
    print("\n6. Audit + verify sampled (JSON)")
    resp = http.post("/audit", json={
        "request_id": chat_sampled["request_id"],
        "token_index": 0,
        "layer_indices": list(range(n_layers)),
        "tier": "full",
        "binary": False,
    })
    ok(resp.status_code == 200, f"sampled audit status={resp.status_code}")
    audit_sampled_json = resp.text
    audit_sampled = json.loads(audit_sampled_json)
    report_s = verilm_rs.verify_v4(audit_sampled_json, key_json)
    ok(report_s["passed"], f"sampled verify passed ({report_s['checks_passed']}/{report_s['checks_run']})")
    if not report_s["passed"]:
        print(f"    failures: {report_s['failures'][:3]}")

    # ── 7. Audit sampled via HTTP (binary) + verify ──
    # Audit does not consume the entry, so we can re-audit the same request.
    # Use a fresh chat to test an independent binary path end-to-end.
    print("\n7. Audit + verify (binary wire format)")
    chat_bin = http.post("/chat", json={
        "prompt": PROMPT,
        "n_tokens": MAX_TOKENS,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
    }).json()
    resp_bin = http.post("/audit", json={
        "request_id": chat_bin["request_id"],
        "token_index": 0,
        "layer_indices": list(range(n_layers)),
        "tier": "full",
        "binary": True,
    })
    ok(resp_bin.status_code == 200, f"binary audit status={resp_bin.status_code}")
    ok(resp_bin.headers.get("content-type", "").startswith("application/octet-stream"),
       f"binary content-type is octet-stream (got {resp_bin.headers.get('content-type')})")
    audit_bytes = resp_bin.content
    ok(len(audit_bytes) > 0, f"binary payload is {len(audit_bytes)} bytes")
    report_bin = verilm_rs.verify_v4_binary(audit_bytes, key_json)
    ok(report_bin["passed"], f"binary verify passed ({report_bin['checks_passed']}/{report_bin['checks_run']})")
    if not report_bin["passed"]:
        print(f"    failures: {report_bin['failures'][:3]}")

    # ── 8. Routine tier via HTTP ──
    print("\n8. Routine tier audit via HTTP")
    chat_routine = http.post("/chat", json={
        "prompt": PROMPT,
        "n_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }).json()
    challenge_seed = hashlib.sha256(b"http-test-verifier-secret").digest()
    challenge = verilm_rs.build_audit_challenge(
        list(challenge_seed), chat_routine["n_tokens"], n_layers, "routine"
    )
    resp_routine = http.post("/audit", json={
        "request_id": chat_routine["request_id"],
        "token_index": challenge["token_index"],
        "layer_indices": challenge["layer_indices"],
        "tier": "routine",
        "binary": False,
    })
    ok(resp_routine.status_code == 200, f"routine audit status={resp_routine.status_code}")
    audit_routine_json = resp_routine.text
    report_routine = verilm_rs.verify_v4(audit_routine_json, key_json)
    ok(report_routine["passed"],
       f"routine verify passed ({report_routine['checks_passed']}/{report_routine['checks_run']})")
    if not report_routine["passed"]:
        print(f"    failures: {report_routine['failures'][:3]}")

    # ── 9. Non-zero token index via HTTP (sampled) ──
    print("\n9. Non-zero token index (sampled)")
    chat_multi = http.post("/chat", json={
        "prompt": PROMPT,
        "n_tokens": MAX_TOKENS,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
    }).json()
    token_idx = min(1, chat_multi["n_tokens"] - 1)
    resp_multi = http.post("/audit", json={
        "request_id": chat_multi["request_id"],
        "token_index": token_idx,
        "layer_indices": list(range(n_layers)),
        "tier": "full",
        "binary": False,
    })
    ok(resp_multi.status_code == 200, f"token[{token_idx}] audit status={resp_multi.status_code}")
    audit_multi_json = resp_multi.text
    report_multi = verilm_rs.verify_v4(audit_multi_json, key_json)
    ok(report_multi["passed"],
       f"token[{token_idx}] verify passed ({report_multi['checks_passed']}/{report_multi['checks_run']})")
    if not report_multi["passed"]:
        print(f"    failures: {report_multi['failures'][:3]}")

    # ── 10. HTTP error paths ──
    print("\n10. HTTP error paths")

    # Missing required fields.
    resp_err = http.post("/audit", json={
        "request_id": "fake",
    })
    ok(resp_err.status_code == 400, f"missing fields → 400 (got {resp_err.status_code})")

    # Unknown request_id → 404.
    resp_err2 = http.post("/audit", json={
        "request_id": "nonexistent-id",
        "token_index": 0,
        "layer_indices": [0],
    })
    ok(resp_err2.status_code == 404, f"unknown request_id → 404 (got {resp_err2.status_code})")

    # ── 11. EOS before max_tokens (HTTP path) ──
    print("\n11. EOS before max_tokens")
    resp_eos = http.post("/chat", json={
        "prompt": "What is 2+2? Answer with just the number.",
        "n_tokens": 256,
        "temperature": 0.0,
    })
    ok(resp_eos.status_code == 200, f"EOS chat status={resp_eos.status_code}")
    chat_eos = resp_eos.json()
    ok(chat_eos.get("n_tokens", 256) < 256,
       f"EOS before max_tokens ({chat_eos.get('n_tokens')} < 256)")
    resp_eos_audit = http.post("/audit", json={
        "request_id": chat_eos["request_id"],
        "token_index": 0,
        "layer_indices": list(range(n_layers)),
        "tier": "full",
        "binary": False,
    })
    ok(resp_eos_audit.status_code == 200, "EOS audit status=200")
    audit_eos_json = resp_eos_audit.text
    report_eos = verilm_rs.verify_v4(audit_eos_json, key_json)
    ok(report_eos["passed"],
       f"EOS verify passed ({report_eos['checks_passed']}/{report_eos['checks_run']})")
    if not report_eos["passed"]:
        print(f"    failures: {report_eos['failures'][:3]}")

    # ── 12. Tamper detection (negative case) ──
    print("\n12. Tamper detection")
    tampered = json.loads(audit_greedy_json)
    if tampered.get("shell_opening") and tampered["shell_opening"].get("layers"):
        layer0 = tampered["shell_opening"]["layers"][0]
        if "attn_out" in layer0 and layer0["attn_out"]:
            layer0["attn_out"][0] ^= 0x7F
    tampered_json = json.dumps(tampered)
    report_tampered = verilm_rs.verify_v4(tampered_json, key_json)
    ok(not report_tampered["passed"], "tampered audit correctly rejected")
    if report_tampered["failures"]:
        print(f"    detected: {report_tampered['failures'][0][:80]}...")

    # ── Summary ──
    print(f"\n{'='*60}")
    if failures:
        print(f"HTTP E2E TEST FAILED — {len(failures)}/{checks} checks failed:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"HTTP E2E TEST PASSED — {checks}/{checks} checks")
    print(f"{'='*60}")

    return {
        "passed": len(failures) == 0,
        "checks": checks,
        "n_failures": len(failures),
        "failures": failures,
    }


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_test():
    return _run_test()


@app.local_entrypoint()
def main():
    print("VeriLM HTTP E2E Test: POST /chat → POST /audit → verify")
    print("=" * 60)
    result = run_test.remote()
    if result["passed"]:
        print(f"\nHTTP E2E TEST PASSED — {result['checks']} checks")
    else:
        print(f"\nHTTP E2E TEST FAILED — {result['n_failures']} failure(s):")
        for f in result["failures"]:
            print(f"  - {f}")
        raise SystemExit(1)
