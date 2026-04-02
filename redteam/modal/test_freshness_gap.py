"""
Red-team probe: receipt replay / freshness gap.

This is intentionally not a "must reject" campaign. It demonstrates the
current accepted limitation that an old honest receipt remains a valid proof
object for the same prompt/output unless a verifier-side freshness mechanism
is added.

Expected result today: the replayed audit still verifies, and the script
reports that as an observed limitation rather than a verifier bug.
"""

import modal

app = modal.App("verilm-redteam-freshness-gap")

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 24

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
        ".git", "target", "scripts/__pycache__", "*.pdf", "CHANGELOG.md",
    ])
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

    print(f"Loading model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        dtype="auto",
        max_model_len=2048,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    http = TestClient(create_app(llm))
    n_layers = cap._n_layers

    model_dir = _snapshot_dir(MODEL_ID)
    seed = hashlib.sha256(PROMPT.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)

    def chat_and_audit():
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
            "binary": False,
        })
        assert resp.status_code == 200, f"audit failed: {resp.status_code}"
        return chat, resp.text

    first_chat, first_audit_json = chat_and_audit()
    first_report = verilm_rs.verify_v4(first_audit_json, key_json)
    assert first_report["passed"], f"baseline audit must pass: {first_report['failures']}"

    second_chat, second_audit_json = chat_and_audit()
    second_report = verilm_rs.verify_v4(second_audit_json, key_json)
    assert second_report["passed"], f"second baseline audit must pass: {second_report['failures']}"

    assert first_chat["request_id"] != second_chat["request_id"], "need distinct requests to probe replay/freshness"

    replay_report = verilm_rs.verify_v4(first_audit_json, key_json)
    replay_accepted = replay_report["passed"]
    assert replay_accepted, "expected replayed honest audit to remain valid without freshness binding"

    print("Observed accepted limitation: old honest audit still verifies on replay.")
    print(f"request A: {first_chat['request_id']}")
    print(f"request B: {second_chat['request_id']}")
    print(f"text A == text B: {first_chat.get('generated_text') == second_chat.get('generated_text')}")

    return {
        "gap_observed": True,
        "model": MODEL_ID,
        "request_a": first_chat["request_id"],
        "request_b": second_chat["request_id"],
        "same_text": first_chat.get("generated_text") == second_chat.get("generated_text"),
        "note": "No verifier-issued nonce or timestamp binding yet; replay of an old honest receipt remains acceptable.",
    }


@app.function(image=image, gpu="A100-80GB", timeout=2400)
def run_test():
    return _run_test()


@app.local_entrypoint()
def main():
    print("CommitLLM Red Team: freshness / replay gap")
    print("=" * 60)
    result = run_test.remote()
    print("\nOBSERVED LIMITATION — replayed honest audit still verifies")
    print(result["note"])
    print(f"same_text={result['same_text']}")
