"""
Diagnostic: verify KV entries are actually present in audit response.

Checks:
  1. kv_roots in commitment (count, not just presence)
  2. kv_entries in deserialized audit
  3. payload size with/without KV
  4. use_captured_x_attn flag at audit time

Usage:
    modal run scripts/modal/diag_kv_opening.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-kv-opening")

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


def _run_diag():
    import hashlib
    import json

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

    # Keygen
    seed = hashlib.sha256(b"diag-kv").digest()
    key_binary, artifact_binary = verilm_rs.generate_key_binary(model_dir, seed)
    print(f"key: {len(key_binary)} bytes, artifact: {len(artifact_binary) if artifact_binary else 0} bytes")

    # Generate
    result = server.chat(prompt="What is 2+2?", max_tokens=16, temperature=0.0)
    request_id = result["request_id"]
    commitment = result["commitment"]
    n_prompt = commitment.get("n_prompt_tokens", 0)
    n_gen = result["n_tokens"] - n_prompt

    # Check commitment
    kv_roots = commitment.get("kv_roots", [])
    print(f"\n=== Commitment ===")
    print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
    print(f"  kv_roots count: {len(kv_roots)}")
    if kv_roots:
        print(f"  kv_roots[0]: {kv_roots[0][:16]}...")
    else:
        print(f"  kv_roots: EMPTY — KV transcript was not committed!")

    # Check audit state
    entry = server._audit_store.get(request_id)
    state = entry["state"]
    print(f"\n=== Audit state ===")
    print(f"  type: {type(state).__name__}")
    print(f"  has_captured_x_attn: {state.has_captured_x_attn()}")
    print(f"  use_captured_x_attn (before set): {state.use_captured_x_attn}")

    # Check if state has kv-related attributes
    for attr in ["kv_tree_count", "kv_entry_count"]:
        if hasattr(state, attr):
            print(f"  {attr}: {getattr(state, attr)()}")

    # Set use_captured_x_attn and audit
    state.use_captured_x_attn = True
    print(f"  use_captured_x_attn (after set): {state.use_captured_x_attn}")

    # Binary audit
    audit_binary = server.audit(
        request_id=request_id,
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=True,
        use_captured_x_attn=True,
    )
    print(f"\n=== Binary audit ===")
    print(f"  size: {len(audit_binary)} bytes ({len(audit_binary)/1024:.1f} KB)")

    # JSON audit for inspection
    audit_json_str = server.audit(
        request_id=request_id,
        token_index=0,
        layer_indices=full_layers,
        tier="full",
        binary=False,
        use_captured_x_attn=True,
    )
    audit_json = json.loads(audit_json_str)
    print(f"\n=== JSON audit ===")
    print(f"  top-level keys: {list(audit_json.keys())}")
    kv_entries = audit_json.get("kv_entries")
    kv_proofs = audit_json.get("kv_proofs")
    print(f"  kv_entries: {type(kv_entries).__name__}, is None: {kv_entries is None}")
    if kv_entries is not None:
        print(f"  kv_entries layers: {len(kv_entries)}")
        if kv_entries:
            print(f"  kv_entries[0] positions: {len(kv_entries[0])}")
            if kv_entries[0]:
                entry0 = kv_entries[0][0]
                print(f"  kv_entries[0][0] keys: {list(entry0.keys()) if isinstance(entry0, dict) else 'not a dict'}")
    else:
        print(f"  kv_entries is NULL — open_v4 did not populate them!")

    if kv_proofs is not None:
        print(f"  kv_proofs layers: {len(kv_proofs)}")
    else:
        print(f"  kv_proofs is NULL")

    # Verify
    report = verilm_rs.verify_v4_full_binary(bytes(audit_binary), key_binary, artifact_binary)
    print(f"\n=== Verify ===")
    print(f"  passed: {report['passed']}")
    print(f"  checks: {report['checks_passed']}/{report['checks_run']}")
    if report.get("failures"):
        for f in report["failures"][:10]:
            print(f"  FAIL: {f}")
    if report.get("skipped"):
        for s in report["skipped"]:
            print(f"  SKIP: {s}")

    return {"done": True}


@app.function(image=image, gpu="A100-80GB", timeout=600)
def run_diag():
    return _run_diag()


@app.local_entrypoint()
def main():
    print("KV Opening Diagnostic")
    print("=" * 60)
    run_diag.remote()
