"""
Real over-the-wire HTTP smoke test.

Unlike test_e2e_http.py (which uses FastAPI TestClient / ASGI transport),
this test starts an actual uvicorn server on a real socket and hits it
with httpx over TCP. Proves the real HTTP path works end to end.

Covers:
  - POST /chat (greedy + sampled)
  - POST /audit (JSON + binary)
  - verilm_rs.verify_v4() / verify_v4_binary()
  - Error paths (missing fields → 400, bad request_id → 404)

Usage:
    modal run --detach scripts/modal/test_e2e_http_wire.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-test-e2e-http-wire")

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
    .pip_install(*VERIFICATION, "uvicorn", "httpx")
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf", "CHANGELOG.md",
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
PORT = 18923


def _run_test():
    import hashlib
    import json
    import os
    import socket
    import threading
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import httpx
    import uvicorn
    import verilm_rs
    from vllm import LLM

    from verilm import capture as cap
    from verilm.server import create_app

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)

    n_layers = cap._n_layers
    model_id = llm.llm_engine.model_config.model
    if os.path.isdir(model_id):
        model_dir = model_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_id)

    # Start real HTTP server in a background thread.
    fastapi_app = create_app(llm)
    server_config = uvicorn.Config(
        fastapi_app, host="127.0.0.1", port=PORT, log_level="warning",
    )
    server = uvicorn.Server(server_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready.
    base_url = f"http://127.0.0.1:{PORT}"
    for _ in range(50):
        try:
            sock = socket.create_connection(("127.0.0.1", PORT), timeout=0.5)
            sock.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    else:
        raise RuntimeError("Server did not start within 5 seconds")

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

    with httpx.Client(base_url=base_url, timeout=120) as http:
        # ── 1. Health ──
        print("\n1. GET /health (real TCP)")
        resp = http.get("/health")
        ok(resp.status_code == 200, f"health status={resp.status_code}")

        # ── 2. Greedy chat → audit → verify ──
        print("\n2. Greedy: chat → audit → verify (real TCP)")
        resp = http.post("/chat", json={
            "prompt": PROMPT, "n_tokens": MAX_TOKENS, "temperature": 0.0,
        })
        ok(resp.status_code == 200, f"greedy chat status={resp.status_code}")
        chat = resp.json()
        ok(chat.get("n_tokens", 0) > 0, f"greedy generated {chat.get('n_tokens')} tokens")

        resp = http.post("/audit", json={
            "request_id": chat["request_id"],
            "token_index": 0,
            "layer_indices": list(range(n_layers)),
            "tier": "full", "binary": False,
        })
        ok(resp.status_code == 200, f"greedy audit status={resp.status_code}")
        report = verilm_rs.verify_v4(resp.text, key_json)
        ok(report["passed"], f"greedy verify passed ({report['checks_passed']}/{report['checks_run']})")
        if not report["passed"]:
            print(f"    failures: {report['failures'][:3]}")

        # ── 3. Sampled chat → audit → verify ──
        print("\n3. Sampled: chat → audit → verify (real TCP)")
        resp = http.post("/chat", json={
            "prompt": PROMPT, "n_tokens": MAX_TOKENS,
            "temperature": 0.8, "top_k": 50, "top_p": 0.9,
        })
        ok(resp.status_code == 200, f"sampled chat status={resp.status_code}")
        chat_s = resp.json()
        ok(chat_s.get("n_tokens", 0) > 0, f"sampled generated {chat_s.get('n_tokens')} tokens")
        print(f"    text: {chat_s.get('generated_text', '')[:80]}")

        resp = http.post("/audit", json={
            "request_id": chat_s["request_id"],
            "token_index": 0,
            "layer_indices": list(range(n_layers)),
            "tier": "full", "binary": False,
        })
        ok(resp.status_code == 200, f"sampled audit status={resp.status_code}")
        report_s = verilm_rs.verify_v4(resp.text, key_json)
        ok(report_s["passed"], f"sampled verify passed ({report_s['checks_passed']}/{report_s['checks_run']})")
        if not report_s["passed"]:
            print(f"    failures: {report_s['failures'][:3]}")

        # ── 4. Binary audit over real TCP ──
        print("\n4. Binary audit (real TCP)")
        chat_b = http.post("/chat", json={
            "prompt": PROMPT, "n_tokens": MAX_TOKENS,
            "temperature": 0.8, "top_k": 50, "top_p": 0.9,
        }).json()
        resp_bin = http.post("/audit", json={
            "request_id": chat_b["request_id"],
            "token_index": 0,
            "layer_indices": list(range(n_layers)),
            "tier": "full", "binary": True,
        })
        ok(resp_bin.status_code == 200, f"binary audit status={resp_bin.status_code}")
        ok("octet-stream" in resp_bin.headers.get("content-type", ""),
           "binary content-type is octet-stream")
        report_bin = verilm_rs.verify_v4_binary(resp_bin.content, key_json)
        ok(report_bin["passed"], f"binary verify passed ({report_bin['checks_passed']}/{report_bin['checks_run']})")
        if not report_bin["passed"]:
            print(f"    failures: {report_bin['failures'][:3]}")

        # ── 5. Error paths over real TCP ──
        print("\n5. Error paths (real TCP)")
        resp_err = http.post("/audit", json={"request_id": "fake"})
        ok(resp_err.status_code == 400, f"missing fields → 400 (got {resp_err.status_code})")

        resp_err2 = http.post("/audit", json={
            "request_id": "nonexistent", "token_index": 0, "layer_indices": [0],
        })
        ok(resp_err2.status_code == 404,
           f"bad request_id → 404 (got {resp_err2.status_code})")

    # Shutdown server.
    server.should_exit = True
    thread.join(timeout=5)

    # ── Summary ──
    print(f"\n{'='*60}")
    if failures:
        print(f"HTTP WIRE TEST FAILED — {len(failures)}/{checks} checks failed:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"HTTP WIRE TEST PASSED — {checks}/{checks} checks")
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
    print("VeriLM HTTP Wire Test: real TCP → POST /chat → POST /audit → verify")
    print("=" * 60)
    result = run_test.remote()
    if result["passed"]:
        print(f"\nHTTP WIRE TEST PASSED — {result['checks']} checks")
    else:
        print(f"\nHTTP WIRE TEST FAILED — {result['n_failures']} failure(s):")
        for f in result["failures"]:
            print(f"  - {f}")
        raise SystemExit(1)
