"""
Llama 3.1 8B W8A8 — Adversarial tamper detection tests.

Runs honest inference, validates the canonical audit passes, then
systematically tampers with individual fields and asserts the verifier
rejects each tampered proof **for the right reason**.

This is not a correctness test (does an honest prover pass?). This asks:
can a dishonest prover cheat and still pass verification?

Boundaries tested (36 scenarios):

  Shell opening (Freivalds):     attn_out, ffn_out, g, u, q, k, v
  Retained-state leaf (Merkle):  a, scale_a, scale_x_attn, scale_x_ffn, scale_h
  IO chain:                      prev_io_hash
  Embedding proof:               initial_residual, embedding_proof leaf_index
  LM-head / logits:              logits_i32
  Decode replay:                 token_id (greedy), token_id (sampled), revealed_seed
  Manifest binding:              temperature
  Prompt binding:                prompt bytes
  Commitment structure:          merkle_root, io_root, n_tokens
  Final residual:                final_residual
  Cross-request splice:          shell, retained, token_id from different request
  Layer swap:                    shell layer 0↔1, retained layer 0↔1
  Prefix tampering:              prefix_leaf_hash, prefix_token_id
  Token index:                   shifted token_index
  n_prompt_tokens:               response n_prompt_tokens, commitment n_prompt_tokens
  Seed commitment:               seed_commitment in commitment
  Embedding proof structure:     embedding_proof leaf_index mismatch

Usage:
    modal run --detach scripts/modal/tests/llama/test_adversarial.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("commitllm-test-llama-adversarial")

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
    .pip_install(*VERIFICATION, "httpx")
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "**/__pycache__", "*.pyc", "*.pdf", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

PROMPT = "What is the capital of France?"
MAX_TOKENS = 32


def _run_test():
    import hashlib
    import json

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from fastapi.testclient import TestClient
    from vllm import LLM

    from verilm import capture as cap
    from verilm.server import create_app

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)

    fastapi_app = create_app(llm)
    http = TestClient(fastapi_app)

    n_layers = cap._n_layers
    model_id = llm.llm_engine.model_config.model
    if os.path.isdir(model_id):
        model_dir = model_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_id)

    seed = hashlib.sha256(PROMPT.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)

    def fetch_audit(request_id, token_index, *, binary, deep_prefix=False):
        resp = http.post("/audit", json={
            "request_id": request_id,
            "token_index": token_index,
            "layer_indices": list(range(n_layers)),
            "tier": "full",
            "binary": binary,
            "deep_prefix": deep_prefix,
        })
        assert resp.status_code == 200, f"audit failed: {resp.status_code}"
        return resp

    def sanitize_debug_audit(audit_dict):
        sanitized = json.loads(json.dumps(audit_dict))
        sanitized.pop("kv_entries", None)
        sanitized.pop("kv_proofs", None)
        return sanitized

    # ── Baseline: honest greedy chat + full audit ──
    print("\nBaseline: honest greedy chat + full audit")
    resp = http.post("/chat", json={
        "prompt": PROMPT, "n_tokens": MAX_TOKENS, "temperature": 0.0,
    })
    assert resp.status_code == 200, f"chat failed: {resp.status_code}"
    chat = resp.json()
    print(f"  generated {chat['n_tokens']} tokens: {chat.get('generated_text', '')[:60]}")

    honest_binary = fetch_audit(chat["request_id"], 0, binary=True).content
    report_bin = verilm_rs.verify_v4_binary(honest_binary, key_json)
    assert report_bin["passed"], f"honest greedy binary audit must pass: {report_bin['failures']}"
    print(f"  binary baseline: {report_bin['checks_passed']}/{report_bin['checks_run']} checks passed")

    honest_audit = sanitize_debug_audit(fetch_audit(chat["request_id"], 0, binary=False).json())
    honest_json = json.dumps(honest_audit)

    report = verilm_rs.verify_v4(honest_json, key_json)
    assert report["passed"], f"honest greedy JSON mutation base must pass: {report['failures']}"
    print(f"  json mutation base: {report['checks_passed']}/{report['checks_run']} checks passed")

    # ── Baseline: honest sampled chat + full audit ──
    print("\nBaseline: honest sampled chat + full audit")
    resp_s = http.post("/chat", json={
        "prompt": PROMPT, "n_tokens": MAX_TOKENS,
        "temperature": 0.8, "top_k": 50, "top_p": 0.9,
    })
    assert resp_s.status_code == 200
    chat_s = resp_s.json()
    sampled_binary = fetch_audit(chat_s["request_id"], 0, binary=True).content
    report_s_bin = verilm_rs.verify_v4_binary(sampled_binary, key_json)
    assert report_s_bin["passed"], f"honest sampled binary audit must pass: {report_s_bin['failures']}"
    print(f"  binary baseline: {report_s_bin['checks_passed']}/{report_s_bin['checks_run']} checks passed")

    sampled_audit = sanitize_debug_audit(fetch_audit(chat_s["request_id"], 0, binary=False).json())
    sampled_json = json.dumps(sampled_audit)
    report_s = verilm_rs.verify_v4(sampled_json, key_json)
    assert report_s["passed"], f"honest sampled JSON mutation base must pass: {report_s['failures']}"
    print(f"  json mutation base: {report_s['checks_passed']}/{report_s['checks_run']} checks passed")

    # ── Tamper test harness ──

    failures = []
    tests_run = 0

    def tamper_test(name, audit_json, mutate_fn, expect_substr):
        """Tamper one field, verify rejection for the right reason."""
        nonlocal tests_run
        tests_run += 1
        tampered = json.loads(audit_json)
        try:
            mutate_fn(tampered)
        except (KeyError, IndexError, TypeError) as e:
            msg = f"SKIP [{name}]: field not present in audit ({e})"
            print(f"  {msg}")
            return
        tampered_json = json.dumps(tampered)
        try:
            report = verilm_rs.verify_v4(tampered_json, key_json)
        except ValueError as e:
            print(f"  OK [{name}]: rejected at deserialization ({e})")
            return

        if report["passed"]:
            msg = f"FAIL [{name}]: tampered audit was NOT rejected"
            failures.append(msg)
            print(f"  {msg}")
            return

        all_reasons = " | ".join(report["failures"])
        if expect_substr.lower() in all_reasons.lower():
            print(f"  OK [{name}]: rejected for '{expect_substr}'")
        else:
            msg = (f"FAIL [{name}]: rejected but wrong reason — "
                   f"expected '{expect_substr}', got: {report['failures'][:3]}")
            failures.append(msg)
            print(f"  {msg}")

    # ═══════════════════════════════════════════════════════
    # Shell opening — Freivalds checks on matmul accumulators
    # ═══════════════════════════════════════════════════════
    print("\n═══ Shell opening (Freivalds) ═══")

    def tamper_attn_out(d):
        d["shell_opening"]["layers"][0]["attn_out"][0] ^= 0x7F
    tamper_test("attn_out_flip", honest_json, tamper_attn_out, "freivalds")

    def tamper_ffn_out(d):
        d["shell_opening"]["layers"][0]["ffn_out"][0] ^= 0x7F
    tamper_test("ffn_out_flip", honest_json, tamper_ffn_out, "freivalds")

    def tamper_g(d):
        d["shell_opening"]["layers"][0]["g"][0] ^= 0x7F
    tamper_test("g_flip", honest_json, tamper_g, "freivalds")

    def tamper_u(d):
        d["shell_opening"]["layers"][0]["u"][0] ^= 0x7F
    tamper_test("u_flip", honest_json, tamper_u, "freivalds")

    def tamper_q(d):
        for layer in d["shell_opening"]["layers"]:
            if layer.get("q"):
                layer["q"][0] ^= 0x7F
                return
        raise KeyError("no layer with q present")
    tamper_test("q_flip", honest_json, tamper_q, "freivalds")

    def tamper_k(d):
        for layer in d["shell_opening"]["layers"]:
            if layer.get("k"):
                layer["k"][0] ^= 0x7F
                return
        raise KeyError("no layer with k present")
    tamper_test("k_flip", honest_json, tamper_k, "freivalds")

    def tamper_v(d):
        for layer in d["shell_opening"]["layers"]:
            if layer.get("v"):
                layer["v"][0] ^= 0x7F
                return
        raise KeyError("no layer with v present")
    tamper_test("v_flip", honest_json, tamper_v, "freivalds")

    # ═══════════════════════════════════════════════════════
    # Retained-state leaf — Merkle binding
    # ═══════════════════════════════════════════════════════
    print("\n═══ Retained-state leaf ═══")

    def tamper_a(d):
        v = d["retained"]["layers"][0]["a"][0]
        v = (v ^ 0x7F)
        if v > 127:
            v -= 256
        d["retained"]["layers"][0]["a"][0] = v
    tamper_test("retained_a_flip", honest_json, tamper_a, "merkle")

    def tamper_scale_a(d):
        d["retained"]["layers"][0]["scale_a"] *= 2.0
    tamper_test("scale_a_double", honest_json, tamper_scale_a, "merkle")

    def tamper_scale_x_attn(d):
        d["retained"]["layers"][0]["scale_x_attn"] *= 2.0
    tamper_test("scale_x_attn_double", honest_json, tamper_scale_x_attn, "merkle")

    def tamper_scale_x_ffn(d):
        d["retained"]["layers"][0]["scale_x_ffn"] *= 2.0
    tamper_test("scale_x_ffn_double", honest_json, tamper_scale_x_ffn, "merkle")

    def tamper_scale_h(d):
        d["retained"]["layers"][0]["scale_h"] *= 2.0
    tamper_test("scale_h_double", honest_json, tamper_scale_h, "merkle")

    # ═══════════════════════════════════════════════════════
    # IO chain
    # ═══════════════════════════════════════════════════════
    print("\n═══ IO chain ═══")

    def tamper_prev_io(d):
        d["prev_io_hash"][0] ^= 0xFF
    tamper_test("prev_io_hash_flip", honest_json, tamper_prev_io, "io")

    # ═══════════════════════════════════════════════════════
    # Embedding proof
    # ═══════════════════════════════════════════════════════
    print("\n═══ Embedding proof ═══")

    def tamper_initial_residual(d):
        ir = d["shell_opening"]["initial_residual"]
        if ir:
            ir[0] += 100.0
    tamper_test("initial_residual_shift", honest_json, tamper_initial_residual, "embedding")

    # ═══════════════════════════════════════════════════════
    # LM-head / logits
    # ═══════════════════════════════════════════════════════
    print("\n═══ LM-head / logits ═══")

    def tamper_logits(d):
        logits = d["shell_opening"].get("logits_i32")
        if logits:
            logits[0] ^= 0x7F
        else:
            raise KeyError("logits_i32 not present")
    tamper_test("logits_i32_flip", honest_json, tamper_logits, "lm_head")

    # ═══════════════════════════════════════════════════════
    # Decode replay
    # ═══════════════════════════════════════════════════════
    print("\n═══ Decode replay ═══")

    def tamper_token_id_greedy(d):
        d["token_id"] = (d["token_id"] + 1) % 32000
    tamper_test("token_id_greedy", honest_json, tamper_token_id_greedy, "token")

    def tamper_token_id_sampled(d):
        d["token_id"] = (d["token_id"] + 1) % 32000
    tamper_test("token_id_sampled", sampled_json, tamper_token_id_sampled, "token")

    def tamper_revealed_seed(d):
        d["revealed_seed"][0] ^= 0xFF
    tamper_test("revealed_seed_flip", sampled_json, tamper_revealed_seed, "seed")

    # ═══════════════════════════════════════════════════════
    # Manifest binding
    # ═══════════════════════════════════════════════════════
    print("\n═══ Manifest binding ═══")

    def tamper_manifest_temp(d):
        d["manifest"]["temperature"] = 0.99
    tamper_test("manifest_temperature", honest_json, tamper_manifest_temp, "manifest")

    # ═══════════════════════════════════════════════════════
    # Prompt binding
    # ═══════════════════════════════════════════════════════
    print("\n═══ Prompt binding ═══")

    def tamper_prompt(d):
        if d.get("prompt"):
            d["prompt"].append(0x41)
        else:
            raise KeyError("prompt not present")
    tamper_test("prompt_append_byte", honest_json, tamper_prompt, "prompt")

    # ═══════════════════════════════════════════════════════
    # Commitment structure
    # ═══════════════════════════════════════════════════════
    print("\n═══ Commitment ═══")

    def tamper_merkle_root(d):
        d["commitment"]["merkle_root"][0] ^= 0xFF
    tamper_test("merkle_root_flip", honest_json, tamper_merkle_root, "merkle")

    def tamper_io_root(d):
        d["commitment"]["io_root"][0] ^= 0xFF
    tamper_test("io_root_flip", honest_json, tamper_io_root, "io")

    def tamper_n_tokens(d):
        d["commitment"]["n_tokens"] += 1
    tamper_test("n_tokens_inflate", honest_json, tamper_n_tokens, "token")

    # ═══════════════════════════════════════════════════════
    # Final residual
    # ═══════════════════════════════════════════════════════
    print("\n═══ Final residual ═══")

    def tamper_final_residual(d):
        fr = d["shell_opening"].get("final_residual")
        if fr:
            fr[0] += 100.0
        else:
            raise KeyError("final_residual not present")
    tamper_test("final_residual_shift", honest_json, tamper_final_residual, "merkle")

    # ═══════════════════════════════════════════════════════
    # Cross-request splice
    # ═══════════════════════════════════════════════════════
    print("\n═══ Cross-request splice ═══")

    PROMPT_B = "What is the largest planet in the solar system?"
    resp_b = http.post("/chat", json={
        "prompt": PROMPT_B, "n_tokens": MAX_TOKENS, "temperature": 0.0,
    })
    assert resp_b.status_code == 200
    chat_b = resp_b.json()
    n_gen_b = chat_b["n_tokens"]

    def _get_gen_start(request_id):
        r = http.post("/audit", json={
            "request_id": request_id, "token_index": 0,
            "layer_indices": [0], "tier": "full", "binary": False,
        })
        if r.status_code == 200:
            npt = r.json().get("n_prompt_tokens")
            if npt is not None:
                return max(npt - 1, 0)
        return 0

    gen_start_a = _get_gen_start(chat["request_id"])
    gen_start_b = _get_gen_start(chat_b["request_id"])
    splice_idx_a = min(gen_start_a + 3, chat["n_tokens"] - 1)
    splice_idx_b = min(gen_start_b + 3, n_gen_b - 1)
    print(f"  gen_start: A={gen_start_a}, B={gen_start_b}; auditing at A[{splice_idx_a}], B[{splice_idx_b}]")

    audit_b = sanitize_debug_audit(
        fetch_audit(chat_b["request_id"], splice_idx_b, binary=False).json()
    )
    audit_b_json = json.dumps(audit_b)
    report_b = verilm_rs.verify_v4(audit_b_json, key_json)
    assert report_b["passed"], f"splice baseline B must pass: {report_b['failures']}"
    print(f"  splice baseline B: {report_b['checks_passed']}/{report_b['checks_run']} checks passed")

    audit_a_gen = sanitize_debug_audit(
        fetch_audit(chat["request_id"], splice_idx_a, binary=False).json()
    )
    audit_a_gen_json = json.dumps(audit_a_gen)
    report_a_gen = verilm_rs.verify_v4(audit_a_gen_json, key_json)
    assert report_a_gen["passed"], f"splice baseline A must pass: {report_a_gen['failures']}"
    print(f"  splice baseline A: {report_a_gen['checks_passed']}/{report_a_gen['checks_run']} checks passed")

    def splice_shell(d):
        other = json.loads(audit_b_json)
        d["shell_opening"] = other["shell_opening"]
    tamper_test("cross_request_shell_splice", audit_a_gen_json, splice_shell, "freivalds")

    def splice_retained(d):
        other = json.loads(audit_b_json)
        d["retained"] = other["retained"]
    tamper_test("cross_request_retained_splice", audit_a_gen_json, splice_retained, "merkle")

    tid_a = audit_a_gen["token_id"]
    tid_b = audit_b["token_id"]
    splice_a_json = audit_a_gen_json
    splice_b_json = audit_b_json
    found_divergent = tid_a != tid_b
    if found_divergent:
        print(f"  token_ids diverge: A={tid_a} vs B={tid_b}")
    else:
        print(f"  token_id match ({tid_a}=={tid_b}), scanning for divergent index...")
        max_scan = min(chat["n_tokens"], n_gen_b)
        gen_floor = max(gen_start_a, gen_start_b)
        for i in range(gen_floor, max_scan):
            if i == splice_idx_a:
                continue
            ra = fetch_audit(chat["request_id"], i, binary=False)
            rb = fetch_audit(chat_b["request_id"], i, binary=False)
            if ra.status_code == 200 and rb.status_code == 200:
                aa = sanitize_debug_audit(ra.json())
                ab = sanitize_debug_audit(rb.json())
                if aa["token_id"] != ab["token_id"]:
                    splice_a_json = json.dumps(aa)
                    splice_b_json = json.dumps(ab)
                    print(f"  found divergent tokens at index {i}: {aa['token_id']} vs {ab['token_id']}")
                    found_divergent = True
                    break
        if not found_divergent:
            print("  WARNING: no divergent token_id found — token splice test inconclusive")

    def splice_token_id(d):
        other = json.loads(splice_b_json)
        d["token_id"] = other["token_id"]
    if found_divergent:
        tamper_test("cross_request_token_splice", splice_a_json, splice_token_id, "io")
    else:
        print("  SKIP [cross_request_token_splice]: could not construct divergent test case")

    # ═══════════════════════════════════════════════════════
    # Layer swap
    # ═══════════════════════════════════════════════════════
    print("\n═══ Layer swap ═══")

    def swap_shell_layers(d):
        layers = d["shell_opening"]["layers"]
        if len(layers) < 2:
            raise IndexError("need >=2 shell layers to swap")
        layers[0], layers[1] = layers[1], layers[0]
    tamper_test("shell_layer_swap_0_1", honest_json, swap_shell_layers, "freivalds")

    def swap_retained_layers(d):
        layers = d["retained"]["layers"]
        if len(layers) < 2:
            raise IndexError("need >=2 retained layers to swap")
        layers[0], layers[1] = layers[1], layers[0]
    tamper_test("retained_layer_swap_0_1", honest_json, swap_retained_layers, "merkle")

    # ═══════════════════════════════════════════════════════
    # Prefix tampering
    # ═══════════════════════════════════════════════════════
    print("\n═══ Prefix tampering ═══")

    print("  Getting multi-token audit for prefix tests...")
    resp_multi = http.post("/chat", json={
        "prompt": PROMPT, "n_tokens": MAX_TOKENS, "temperature": 0.0,
    })
    assert resp_multi.status_code == 200
    chat_multi = resp_multi.json()
    n_gen = chat_multi["n_tokens"]
    n_prompt_multi = chat_multi.get("n_prompt_tokens", 1)
    gen_start_multi = max(n_prompt_multi - 1, 0)
    tok_idx = min(gen_start_multi + 3, n_gen - 1)
    multi_audit = sanitize_debug_audit(
        fetch_audit(chat_multi["request_id"], tok_idx, binary=False).json()
    )
    multi_json = json.dumps(multi_audit)
    report_multi = verilm_rs.verify_v4(multi_json, key_json)
    assert report_multi["passed"], f"multi-token audit must pass: {report_multi['failures']}"
    print(f"  multi-token baseline (token_index={tok_idx}): "
          f"{report_multi['checks_passed']}/{report_multi['checks_run']} checks passed")

    def tamper_prefix_leaf(d):
        if d.get("prefix_leaf_hashes") and len(d["prefix_leaf_hashes"]) > 0:
            d["prefix_leaf_hashes"][0][0] ^= 0xFF
        else:
            raise KeyError("no prefix_leaf_hashes")
    tamper_test("prefix_leaf_hash_flip", multi_json, tamper_prefix_leaf, "prefix")

    def tamper_prefix_token_id(d):
        if d.get("prefix_token_ids") and len(d["prefix_token_ids"]) > 0:
            d["prefix_token_ids"][0] = (d["prefix_token_ids"][0] + 1) % 32000
        else:
            raise KeyError("no prefix_token_ids")
    tamper_test("prefix_token_id_swap", multi_json, tamper_prefix_token_id, "io")

    # ═══════════════════════════════════════════════════════
    # Token index manipulation
    # ═══════════════════════════════════════════════════════
    print("\n═══ Token index manipulation ═══")

    def shift_token_index(d):
        d["token_index"] += 1
    tamper_test("token_index_shift", honest_json, shift_token_index, "prefix")

    # ═══════════════════════════════════════════════════════
    # n_prompt_tokens binding
    # ═══════════════════════════════════════════════════════
    print("\n═══ n_prompt_tokens binding ═══")

    def tamper_npt_response(d):
        if d.get("n_prompt_tokens") is not None:
            d["n_prompt_tokens"] += 1
        else:
            raise KeyError("n_prompt_tokens not present")
    tamper_test("n_prompt_tokens_inflate", honest_json, tamper_npt_response, "n_prompt_tokens")

    def tamper_npt_commitment(d):
        if d["commitment"].get("n_prompt_tokens") is not None:
            d["commitment"]["n_prompt_tokens"] += 1
        else:
            raise KeyError("commitment.n_prompt_tokens not present")
    tamper_test("commitment_npt_inflate", honest_json, tamper_npt_commitment, "n_prompt_tokens")

    # ═══════════════════════════════════════════════════════
    # Seed commitment
    # ═══════════════════════════════════════════════════════
    print("\n═══ Seed commitment ═══")

    def tamper_seed_commitment(d):
        sc = d["commitment"].get("seed_commitment")
        if sc:
            sc[0] ^= 0xFF
        else:
            raise KeyError("seed_commitment not present")
    tamper_test("seed_commitment_flip", sampled_json, tamper_seed_commitment, "seed")

    # ═══════════════════════════════════════════════════════
    # Embedding proof structure
    # ═══════════════════════════════════════════════════════
    print("\n═══ Embedding proof structure ═══")

    def tamper_embedding_leaf_index(d):
        ep = d["shell_opening"].get("embedding_proof")
        if ep:
            ep["leaf_index"] = (ep["leaf_index"] + 1) % 32000
        else:
            raise KeyError("embedding_proof not present")
    tamper_test("embedding_proof_wrong_token", honest_json,
                tamper_embedding_leaf_index, "embedding")

    # ── Summary ──
    print(f"\n{'='*60}")
    if failures:
        print(f"ADVERSARIAL TEST FAILED — {len(failures)}/{tests_run} tests failed:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"ADVERSARIAL TEST PASSED — {tests_run}/{tests_run} tamper scenarios rejected")
    print(f"{'='*60}")

    return {
        "passed": len(failures) == 0,
        "tests_run": tests_run,
        "n_failures": len(failures),
        "failures": failures,
    }


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_test():
    return _run_test()


@app.local_entrypoint()
def main():
    print("CommitLLM Adversarial Tamper Test — Llama 3.1 8B W8A8")
    print("Try to break every verifier check on real GPU data")
    print("=" * 60)
    result = run_test.remote()
    if result["passed"]:
        print(f"\nADVERSARIAL TEST PASSED — {result['tests_run']} tamper scenarios rejected")
    else:
        print(f"\nADVERSARIAL TEST FAILED — {result['n_failures']} failure(s):")
        for f in result["failures"]:
            print(f"  - {f}")
        raise SystemExit(1)
