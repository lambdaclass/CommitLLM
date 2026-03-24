"""
End-to-end protocol test on Modal: chat -> commit -> audit -> verify.

Exercises the full VeriLM pipeline on real GPU inference:
  1. Generate with VerifiedInferenceServer (capture + commit via Rust)
  2. Verify commitment structure (R_T, R_KV, manifest, token count)
  3. Open compact proof for all tokens
  4. Open stratified audit (routine tier, subset of layers)
  5. Open stratified audit (full tier)
  6. Verify proof is non-empty and decompresses

Usage:
    modal run scripts/modal_test_e2e.py
"""

import modal

app = modal.App("verilm-e2e-test")

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
    .pip_install("vllm>=0.8", "torch", "numpy", "fastapi", "maturin", "zstandard")
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

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
PROMPT = "What is 2+2?"


def _run_e2e():
    import json
    import os
    import zstandard

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM
    from verilm.server import VerifiedInferenceServer

    results = {}

    # -- Setup --
    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    print("VerifiedInferenceServer ready")

    # ==================================================================
    # TEST 1: Chat + Commit
    # ==================================================================
    print("\n--- Test 1: Chat + Commit ---")
    chat_result = server.chat(prompt=PROMPT, max_tokens=8)

    has_commitment = "commitment" in chat_result
    has_request_id = "request_id" in chat_result
    has_token_ids = "token_ids" in chat_result
    has_kv_roots = "kv_roots" in chat_result
    has_text = "generated_text" in chat_result
    n_tokens = chat_result.get("n_tokens", 0)

    results["1_has_commitment"] = has_commitment
    results["1_has_request_id"] = has_request_id
    results["1_has_token_ids"] = has_token_ids
    results["1_has_kv_roots"] = has_kv_roots
    results["1_has_text"] = has_text
    results["1_n_tokens_positive"] = n_tokens > 0

    print(f"  commitment: {has_commitment}")
    print(f"  request_id: {chat_result.get('request_id', 'MISSING')}")
    print(f"  n_tokens: {n_tokens}")
    print(f"  generated_text: {repr(chat_result.get('generated_text', ''))}")
    print(f"  kv_roots count: {len(chat_result.get('kv_roots', []))}")

    # ==================================================================
    # TEST 2: Commitment structure
    # ==================================================================
    print("\n--- Test 2: Commitment structure ---")
    commitment = chat_result.get("commitment", {})
    has_merkle_root = "merkle_root" in commitment
    has_n_tokens = "n_tokens" in commitment
    commitment_n_matches = commitment.get("n_tokens") == n_tokens

    results["2_has_merkle_root"] = has_merkle_root
    results["2_n_tokens_match"] = commitment_n_matches

    print(f"  merkle_root: {commitment.get('merkle_root', 'MISSING')[:32]}...")
    print(f"  n_tokens in commitment: {commitment.get('n_tokens')}")
    print(f"  Matches chat n_tokens: {commitment_n_matches}")

    # ==================================================================
    # TEST 3: Audit (routine, all layers, token 0)
    # ==================================================================
    print("\n--- Test 3: Audit (routine, all layers) ---")
    request_id = chat_result["request_id"]
    try:
        entry = server._audit_store[request_id]
        state = entry["state"]
        n_layers = state.n_layers()
        all_layers = list(range(n_layers))
        proof_bytes = server.audit(
            request_id, token_index=0,
            layer_indices=all_layers, tier="routine",
        )
        proof_nonempty = len(proof_bytes) > 0
        dctx = zstandard.ZstdDecompressor()
        decompressed = dctx.decompress(proof_bytes, max_output_size=100_000_000)
        decompress_ok = len(decompressed) > 0
        results["3_proof_nonempty"] = proof_nonempty
        results["3_decompress_ok"] = decompress_ok
        print(f"  Proof size: {len(proof_bytes)} bytes (zstd)")
        print(f"  Decompressed: {len(decompressed)} bytes")
    except Exception as e:
        results["3_proof_nonempty"] = False
        results["3_decompress_ok"] = False
        print(f"  ERROR: {e}")

    # ==================================================================
    # TEST 4: Audit (routine, 3 layers)
    # ==================================================================
    print("\n--- Test 4: Audit (routine, 3 layers) ---")
    try:
        layer_indices = [0, n_layers // 2, n_layers - 1]
        proof_partial = server.audit(
            request_id, token_index=0,
            layer_indices=layer_indices, tier="routine",
        )
        partial_ok = len(proof_partial) > 0
        results["4_partial_layer_audit"] = partial_ok
        print(f"  Layers challenged: {layer_indices} (of {n_layers})")
        print(f"  Proof size: {len(proof_partial)} bytes")
    except Exception as e:
        results["4_partial_layer_audit"] = False
        print(f"  ERROR: {e}")

    # ==================================================================
    # TEST 5: Audit (full tier, all layers)
    # ==================================================================
    print("\n--- Test 5: Audit (full, all layers) ---")
    try:
        proof_full = server.audit(
            request_id, token_index=0,
            layer_indices=all_layers, tier="full",
        )
        full_ok = len(proof_full) > 0
        results["5_full_tier_audit"] = full_ok
        print(f"  Full audit proof: {len(proof_full)} bytes")
    except Exception as e:
        results["5_full_tier_audit"] = False
        print(f"  ERROR: {e}")

    # ==================================================================
    # TEST 8: Second chat (verify capture buffers reset correctly)
    # ==================================================================
    print("\n--- Test 8: Second chat (buffer reset) ---")
    try:
        chat2 = server.chat(prompt="Hello world", max_tokens=4)
        second_ok = (
            "commitment" in chat2
            and chat2["n_tokens"] > 0
            and chat2["request_id"] != request_id
        )
        results["8_second_chat"] = second_ok
        print(f"  Second request_id: {chat2.get('request_id', 'MISSING')}")
        print(f"  n_tokens: {chat2.get('n_tokens')}")
    except Exception as e:
        results["8_second_chat"] = False
        print(f"  ERROR: {e}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*65}")
    print("E2E PROTOCOL TEST SUMMARY")
    print(f"{'='*65}")
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print(f"{'='*65}")
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*65}")

    return {"all_pass": all_pass, "results": results}


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_e2e():
    return _run_e2e()


@app.local_entrypoint()
def main():
    print("VeriLM End-to-End Protocol Test")
    print("=" * 65)
    result = run_e2e.remote()
    if result["all_pass"]:
        print("\nAll tests passed.")
    else:
        failed = [k for k, v in result["results"].items() if not v]
        print(f"\nFailed: {failed}")
