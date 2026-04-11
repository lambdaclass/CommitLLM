"""
Red-team campaign: cross-model substitution.

Serve an honest audit from one model, then verify the same audit against a
verifier key generated from a different model family/checkpoint.

Expected result: rejection. A provider should not be able to serve Qwen while
claiming Llama, or vice versa, and still pass verification.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/modal"))
from _pins import VERIFICATION

import modal

app = modal.App("verilm-redteam-model-substitution")

SERVED_MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
ALT_MODEL_ID = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 24
BUILD_IGNORE = [
    ".git",
    "target",
    "lean",
    "article",
    "docs",
    "paper",
    "research",
    "site",
    "scripts/__pycache__",
    "*.pdf",
    "CHANGELOG.md",
]

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
    .pip_install(*VERIFICATION, "httpx")
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=BUILD_IGNORE)
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)


def _snapshot_dir(model_id: str):
    import os

    if os.path.isdir(model_id):
        return model_id

    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


def _run_test():
    import hashlib
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from fastapi.testclient import TestClient
    from vllm import LLM

    from verilm import capture as cap
    from verilm.server import create_app

    print(f"Loading served model: {SERVED_MODEL_ID}")
    llm = LLM(
        model=SERVED_MODEL_ID,
        dtype="auto",
        max_model_len=2048,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    http = TestClient(create_app(llm))

    n_layers = cap._n_layers
    served_dir = _snapshot_dir(SERVED_MODEL_ID)
    alt_dir = _snapshot_dir(ALT_MODEL_ID)
    seed = hashlib.sha256(PROMPT.encode()).digest()

    print("Generating verifier keys...")
    served_key_json = verilm_rs.generate_key(served_dir, seed)
    alt_key_json = verilm_rs.generate_key(alt_dir, seed)

    resp = http.post("/chat", json={
        "prompt": PROMPT,
        "n_tokens": MAX_TOKENS,
        "temperature": 0.0,
    })
    assert resp.status_code == 200, f"chat failed: {resp.status_code}"
    chat = resp.json()

    resp = http.post("/audit", json={
        "request_id": chat["request_id"],
        "token_index": 0,
        "layer_indices": list(range(n_layers)),
        "tier": "full",
        "binary": True,
    })
    assert resp.status_code == 200, f"audit failed: {resp.status_code}"
    audit_binary = resp.content

    honest = verilm_rs.verify_v4_binary(audit_binary, served_key_json)
    assert honest["passed"], f"honest audit must pass under served-model key: {honest['failures']}"
    print(f"Baseline pass: {honest['checks_passed']}/{honest['checks_run']} checks")

    substituted = verilm_rs.verify_v4_binary(audit_binary, alt_key_json)
    assert not substituted["passed"], "cross-model substitution must be rejected"

    classified = substituted.get("classified_failures", [])
    assert len(classified) > 0, "model substitution rejection should produce classified failures"

    accepted_categories = {
        "SpecMismatch",
        "CryptographicBinding",
        "Structural",
        "spec_mismatch",
        "cryptographic_binding",
        "structural",
    }
    seen_categories = {cf.get("category") for cf in classified}
    assert seen_categories & accepted_categories, (
        f"expected model substitution to fail as spec/binding mismatch, got {classified}"
    )

    print(f"Rejected under alternate model key with categories: {sorted(seen_categories)}")
    print(f"Codes: {[cf.get('code') for cf in classified]}")

    return {
        "passed": True,
        "served_model": SERVED_MODEL_ID,
        "alternate_model": ALT_MODEL_ID,
        "failure_categories": sorted(seen_categories),
        "failure_codes": [cf.get("code") for cf in classified],
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_test():
    return _run_test()


@app.local_entrypoint()
def main():
    print("CommitLLM Red Team: model substitution")
    print("=" * 60)
    result = run_test.remote()
    print(f"\nPASS — alternate-model verification rejected")
    print(f"served: {result['served_model']}")
    print(f"alternate: {result['alternate_model']}")
    print(f"categories: {result['failure_categories']}")
