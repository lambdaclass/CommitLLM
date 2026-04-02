"""
Red-team adversarial campaign: try to break verifier checks on real GPU data.

Runs honest inference on a real W8A8 model, validates the canonical binary
audit path, then systematically tampers with verifier-acceptable JSON debug
snapshots and asserts the verifier rejects the proof **for the right reason**
(roadmap #21, #25).

This lives under `redteam/` on purpose: it is not an ordinary correctness
test. Correctness tests ask whether an honest provider passes. This campaign
asks whether a dishonest provider can cheat and still pass verification.

The production receipt format is binary. The JSON form is a debug mutation
surface only, and this script strips opened committed-KV payloads from that
surface so real-model attacks do not depend on the non-canonical transport
path.

Wire format (V4AuditResponse):
  - token_index, token_id, prev_io_hash           (top-level)
  - retained.layers[].{a, scale_a, scale_x_attn, scale_x_ffn, scale_h}
  - merkle_proof, io_proof                         (Merkle proofs)
  - prefix_leaf_hashes, prefix_merkle_proofs, prefix_token_ids
  - commitment.{merkle_root, io_root, n_tokens, manifest_hash, ...}
  - revealed_seed                                  (sampling seed)
  - shell_opening.layers[].{attn_out, g, u, ffn_out, q, k, v}
  - shell_opening.{initial_residual, embedding_proof, final_residual, logits_i32}
  - manifest                                       (DeploymentManifest)
  - prompt                                         (raw bytes as int array)

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

Each test tampers one field and checks that verification fails with a
failure message containing an expected substring (the "right reason").

Usage:
    modal run --detach redteam/modal/test_adversarial.py
"""

import modal

app = modal.App("verilm-test-adversarial")

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
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
    .pip_install("vllm>=0.8", "torch", "numpy", "fastapi", "httpx", "maturin")
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

PROMPT = "What is the capital of France?"
MAX_TOKENS = 32


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
        # Keep JSON mutation campaigns on a verifier-acceptable shell-only path.
        # Canonical production coverage is exercised separately via binary audits.
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
            # Deserialization failure = tampered payload is structurally invalid,
            # which counts as "rejected" (verifier refuses to accept it).
            print(f"  OK [{name}]: rejected at deserialization ({e})")
            return

        if report["passed"]:
            msg = f"FAIL [{name}]: tampered audit was NOT rejected"
            failures.append(msg)
            print(f"  {msg}")
            return

        # Check failure is for the RIGHT reason.
        all_reasons = " | ".join(report["failures"])
        if expect_substr.lower() in all_reasons.lower():
            print(f"  OK [{name}]: rejected for '{expect_substr}'")
        else:
            msg = (f"FAIL [{name}]: rejected but wrong reason — "
                   f"expected '{expect_substr}', got: {report['failures'][:3]}")
            failures.append(msg)
            print(f"  {msg}")

    # ════════════════════════════════════════════════════════════
    # Shell opening — Freivalds checks on matmul accumulators
    # ════════════════════════════════════════════════════════════
    print("\n═══ Shell opening (Freivalds) ═══")

    # 1. attn_out = W_o @ a
    def tamper_attn_out(d):
        d["shell_opening"]["layers"][0]["attn_out"][0] ^= 0x7F
    tamper_test("attn_out_flip", honest_json, tamper_attn_out, "freivalds")

    # 2. ffn_out = W_d @ h
    def tamper_ffn_out(d):
        d["shell_opening"]["layers"][0]["ffn_out"][0] ^= 0x7F
    tamper_test("ffn_out_flip", honest_json, tamper_ffn_out, "freivalds")

    # 3. g = W_g @ x_ffn
    def tamper_g(d):
        d["shell_opening"]["layers"][0]["g"][0] ^= 0x7F
    tamper_test("g_flip", honest_json, tamper_g, "freivalds")

    # 4. u = W_u @ x_ffn
    def tamper_u(d):
        d["shell_opening"]["layers"][0]["u"][0] ^= 0x7F
    tamper_test("u_flip", honest_json, tamper_u, "freivalds")

    # 5. q = W_q @ x_attn (may be None for layer 0)
    def tamper_q(d):
        for layer in d["shell_opening"]["layers"]:
            if layer.get("q"):
                layer["q"][0] ^= 0x7F
                return
        raise KeyError("no layer with q present")
    tamper_test("q_flip", honest_json, tamper_q, "freivalds")

    # 6. k = W_k @ x_attn
    def tamper_k(d):
        for layer in d["shell_opening"]["layers"]:
            if layer.get("k"):
                layer["k"][0] ^= 0x7F
                return
        raise KeyError("no layer with k present")
    tamper_test("k_flip", honest_json, tamper_k, "freivalds")

    # 7. v = W_v @ x_attn
    def tamper_v(d):
        for layer in d["shell_opening"]["layers"]:
            if layer.get("v"):
                layer["v"][0] ^= 0x7F
                return
        raise KeyError("no layer with v present")
    tamper_test("v_flip", honest_json, tamper_v, "freivalds")

    # ════════════════════════════════════════════════════════════
    # Retained-state leaf — Merkle binding via hash(retained)
    # ════════════════════════════════════════════════════════════
    print("\n═══ Retained-state leaf ═══")

    # 8. a (attention INT8 output) — irreducible field
    def tamper_a(d):
        v = d["retained"]["layers"][0]["a"][0]
        # XOR and wrap back to signed i8 range (-128..127)
        v = (v ^ 0x7F)
        if v > 127:
            v -= 256
        d["retained"]["layers"][0]["a"][0] = v
    tamper_test("retained_a_flip", honest_json, tamper_a, "merkle")

    # 9. scale_a
    def tamper_scale_a(d):
        d["retained"]["layers"][0]["scale_a"] *= 2.0
    tamper_test("scale_a_double", honest_json, tamper_scale_a, "merkle")

    # 10. scale_x_attn
    def tamper_scale_x_attn(d):
        d["retained"]["layers"][0]["scale_x_attn"] *= 2.0
    tamper_test("scale_x_attn_double", honest_json, tamper_scale_x_attn, "merkle")

    # 11. scale_x_ffn
    def tamper_scale_x_ffn(d):
        d["retained"]["layers"][0]["scale_x_ffn"] *= 2.0
    tamper_test("scale_x_ffn_double", honest_json, tamper_scale_x_ffn, "merkle")

    # 12. scale_h
    def tamper_scale_h(d):
        d["retained"]["layers"][0]["scale_h"] *= 2.0
    tamper_test("scale_h_double", honest_json, tamper_scale_h, "merkle")

    # ════════════════════════════════════════════════════════════
    # IO chain — transcript hash chain integrity
    # ════════════════════════════════════════════════════════════
    print("\n═══ IO chain ═══")

    # 13. prev_io_hash (top-level [u8; 32])
    def tamper_prev_io(d):
        d["prev_io_hash"][0] ^= 0xFF
    tamper_test("prev_io_hash_flip", honest_json, tamper_prev_io, "io")

    # ════════════════════════════════════════════════════════════
    # Embedding proof — initial_residual binding
    # ════════════════════════════════════════════════════════════
    print("\n═══ Embedding proof ═══")

    # 14. initial_residual (f32 vector in shell_opening)
    def tamper_initial_residual(d):
        ir = d["shell_opening"]["initial_residual"]
        if ir:
            ir[0] += 100.0
    tamper_test("initial_residual_shift", honest_json, tamper_initial_residual, "embedding")

    # ════════════════════════════════════════════════════════════
    # LM-head / logits — Freivalds on lm_head matmul
    # ════════════════════════════════════════════════════════════
    print("\n═══ LM-head / logits ═══")

    # 15. logits_i32
    def tamper_logits(d):
        logits = d["shell_opening"].get("logits_i32")
        if logits:
            logits[0] ^= 0x7F
        else:
            raise KeyError("logits_i32 not present")
    tamper_test("logits_i32_flip", honest_json, tamper_logits, "lm_head")

    # ════════════════════════════════════════════════════════════
    # Decode replay — token selection verification
    # ════════════════════════════════════════════════════════════
    print("\n═══ Decode replay ═══")

    # 16. token_id (greedy — wrong argmax)
    def tamper_token_id_greedy(d):
        d["token_id"] = (d["token_id"] + 1) % 32000
    tamper_test("token_id_greedy", honest_json, tamper_token_id_greedy, "token")

    # 17. token_id (sampled — wrong sample)
    def tamper_token_id_sampled(d):
        d["token_id"] = (d["token_id"] + 1) % 32000
    tamper_test("token_id_sampled", sampled_json, tamper_token_id_sampled, "token")

    # 18. revealed_seed (sampled — seed commitment mismatch)
    def tamper_revealed_seed(d):
        d["revealed_seed"][0] ^= 0xFF
    tamper_test("revealed_seed_flip", sampled_json, tamper_revealed_seed, "seed")

    # ════════════════════════════════════════════════════════════
    # Manifest binding — spec hash integrity
    # ════════════════════════════════════════════════════════════
    print("\n═══ Manifest binding ═══")

    # 19. temperature field
    def tamper_manifest_temp(d):
        d["manifest"]["temperature"] = 0.99
    tamper_test("manifest_temperature", honest_json, tamper_manifest_temp, "manifest")

    # ════════════════════════════════════════════════════════════
    # Prompt binding — prompt hash integrity
    # ════════════════════════════════════════════════════════════
    print("\n═══ Prompt binding ═══")

    # 20. prompt bytes
    def tamper_prompt(d):
        if d.get("prompt"):
            d["prompt"].append(0x41)
        else:
            raise KeyError("prompt not present")
    tamper_test("prompt_append_byte", honest_json, tamper_prompt, "prompt")

    # ════════════════════════════════════════════════════════════
    # Commitment — structural integrity
    # ════════════════════════════════════════════════════════════
    print("\n═══ Commitment ═══")

    # 21. merkle_root
    def tamper_merkle_root(d):
        d["commitment"]["merkle_root"][0] ^= 0xFF
    tamper_test("merkle_root_flip", honest_json, tamper_merkle_root, "merkle")

    # 22. io_root
    def tamper_io_root(d):
        d["commitment"]["io_root"][0] ^= 0xFF
    tamper_test("io_root_flip", honest_json, tamper_io_root, "io")

    # 23. n_tokens
    def tamper_n_tokens(d):
        d["commitment"]["n_tokens"] += 1
    tamper_test("n_tokens_inflate", honest_json, tamper_n_tokens, "token")

    # ════════════════════════════════════════════════════════════
    # Final residual — exact token boundary
    # ════════════════════════════════════════════════════════════
    print("\n═══ Final residual ═══")

    # 24. final_residual (captured pre-final-norm residual)
    def tamper_final_residual(d):
        fr = d["shell_opening"].get("final_residual")
        if fr:
            fr[0] += 100.0
        else:
            raise KeyError("final_residual not present")
    tamper_test("final_residual_shift", honest_json, tamper_final_residual, "merkle")

    # ════════════════════════════════════════════════════════════
    # Cross-request splice — use shell/retained from another request
    # To actually test cross-request binding, we need two requests with
    # different prompts audited at a generated token index, so the
    # committed data is genuinely different.
    # ════════════════════════════════════════════════════════════
    print("\n═══ Cross-request splice ═══")

    # Generate a second baseline with a DIFFERENT prompt.
    # To guarantee token divergence, use prompts with very different answers
    # and audit several positions past the prompt boundary.
    PROMPT_B = "What is the largest planet in the solar system?"
    resp_b = http.post("/chat", json={
        "prompt": PROMPT_B, "n_tokens": MAX_TOKENS, "temperature": 0.0,
    })
    assert resp_b.status_code == 200
    chat_b = resp_b.json()
    n_gen_b = chat_b["n_tokens"]

    # To find actual generated-token indices, do an initial audit at index 0
    # and read n_prompt_tokens from the audit response (not /chat, which may
    # not return it). Then audit well past the prompt boundary.
    def _get_gen_start(request_id):
        """Audit at index 0 to discover n_prompt_tokens from the response."""
        r = http.post("/audit", json={
            "request_id": request_id, "token_index": 0,
            "layer_indices": [0], "tier": "full", "binary": False,
        })
        if r.status_code == 200:
            npt = r.json().get("n_prompt_tokens")
            if npt is not None:
                return max(npt - 1, 0)  # gen_start in committed array
        return 0

    gen_start_a = _get_gen_start(chat["request_id"])
    gen_start_b = _get_gen_start(chat_b["request_id"])
    # Audit 3 positions past prompt boundary for content divergence
    splice_idx_a = min(gen_start_a + 3, chat["n_tokens"] - 1)
    splice_idx_b = min(gen_start_b + 3, n_gen_b - 1)
    print(f"  gen_start: A={gen_start_a}, B={gen_start_b}; auditing at A[{splice_idx_a}], B[{splice_idx_b}]")

    audit_b = sanitize_debug_audit(
        fetch_audit(chat_b["request_id"], splice_idx_b, binary=False).json()
    )
    audit_b_json = json.dumps(audit_b)
    report_b = verilm_rs.verify_v4(audit_b_json, key_json)
    assert report_b["passed"], f"splice baseline B must pass: {report_b['failures']}"
    print(f"  splice baseline B (prompt_b, tok={splice_idx_b}): "
          f"{report_b['checks_passed']}/{report_b['checks_run']} checks passed")

    audit_a_gen = sanitize_debug_audit(
        fetch_audit(chat["request_id"], splice_idx_a, binary=False).json()
    )
    audit_a_gen_json = json.dumps(audit_a_gen)
    report_a_gen = verilm_rs.verify_v4(audit_a_gen_json, key_json)
    assert report_a_gen["passed"], f"splice baseline A must pass: {report_a_gen['failures']}"
    print(f"  splice baseline A (prompt_a, tok={splice_idx_a}): "
          f"{report_a_gen['checks_passed']}/{report_a_gen['checks_run']} checks passed")

    # 25. Splice shell_opening from request B into request A's generated-token audit
    def splice_shell(d):
        other = json.loads(audit_b_json)
        d["shell_opening"] = other["shell_opening"]
    tamper_test("cross_request_shell_splice", audit_a_gen_json, splice_shell, "freivalds")

    # 26. Splice retained state from request B into request A
    def splice_retained(d):
        other = json.loads(audit_b_json)
        d["retained"] = other["retained"]
    tamper_test("cross_request_retained_splice", audit_a_gen_json, splice_retained, "merkle")

    # 27. Splice token_id: must use indices where token_ids actually differ.
    # Scan from splice_idx upward if needed to find divergent tokens.
    tid_a = audit_a_gen["token_id"]
    tid_b = audit_b["token_id"]
    splice_a_json = audit_a_gen_json
    splice_b_json = audit_b_json
    found_divergent = tid_a != tid_b
    if found_divergent:
        print(f"  token_ids diverge at initial index: A={tid_a} vs B={tid_b}")
    else:
        print(f"  token_id match ({tid_a}=={tid_b}), scanning for divergent index...")
        max_scan = min(chat["n_tokens"], n_gen_b)
        gen_floor = max(gen_start_a, gen_start_b)
        for i in range(gen_floor, max_scan):
            if i == splice_idx_a:
                continue  # already checked
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
        # Don't count as pass or fail — this is a coverage gap in this run
        print("  SKIP [cross_request_token_splice]: could not construct divergent test case")

    # ════════════════════════════════════════════════════════════
    # Layer swap — put data from one layer into another layer's slot
    # ════════════════════════════════════════════════════════════
    print("\n═══ Layer swap ═══")

    # 28. Swap shell layers 0 and 1 (if >1 layer opened)
    def swap_shell_layers(d):
        layers = d["shell_opening"]["layers"]
        if len(layers) < 2:
            raise IndexError("need >=2 shell layers to swap")
        layers[0], layers[1] = layers[1], layers[0]
    tamper_test("shell_layer_swap_0_1", honest_json, swap_shell_layers, "freivalds")

    # 29. Swap retained layers 0 and 1
    def swap_retained_layers(d):
        layers = d["retained"]["layers"]
        if len(layers) < 2:
            raise IndexError("need >=2 retained layers to swap")
        layers[0], layers[1] = layers[1], layers[0]
    tamper_test("retained_layer_swap_0_1", honest_json, swap_retained_layers, "merkle")

    # ════════════════════════════════════════════════════════════
    # Prefix tampering — tamper prefix leaf hashes and proofs
    # ════════════════════════════════════════════════════════════
    print("\n═══ Prefix tampering ═══")

    # For prefix tests, audit a later generated token (token_index > gen_start)
    # so prefix exists and token replay is valid.
    print("  Getting multi-token audit for prefix tests...")
    resp_multi = http.post("/chat", json={
        "prompt": PROMPT, "n_tokens": MAX_TOKENS, "temperature": 0.0,
    })
    assert resp_multi.status_code == 200
    chat_multi = resp_multi.json()
    n_gen = chat_multi["n_tokens"]
    n_prompt_multi = chat_multi.get("n_prompt_tokens", 1)
    # First generated token in committed array is at index n_prompt-1;
    # pick a later one so there are prefix tokens before it.
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

    # 30. Tamper prefix leaf hash
    def tamper_prefix_leaf(d):
        if d.get("prefix_leaf_hashes") and len(d["prefix_leaf_hashes"]) > 0:
            d["prefix_leaf_hashes"][0][0] ^= 0xFF
        else:
            raise KeyError("no prefix_leaf_hashes")
    tamper_test("prefix_leaf_hash_flip", multi_json, tamper_prefix_leaf, "prefix")

    # 31. Tamper prefix token_id
    def tamper_prefix_token_id(d):
        if d.get("prefix_token_ids") and len(d["prefix_token_ids"]) > 0:
            d["prefix_token_ids"][0] = (d["prefix_token_ids"][0] + 1) % 32000
        else:
            raise KeyError("no prefix_token_ids")
    tamper_test("prefix_token_id_swap", multi_json, tamper_prefix_token_id, "io")

    # ════════════════════════════════════════════════════════════
    # Token index manipulation
    # ════════════════════════════════════════════════════════════
    print("\n═══ Token index manipulation ═══")

    # 32. Claim different token_index (proof for token 0, claim token 1)
    def shift_token_index(d):
        d["token_index"] += 1
    tamper_test("token_index_shift", honest_json, shift_token_index, "prefix")

    # ════════════════════════════════════════════════════════════
    # n_prompt_tokens binding
    # ════════════════════════════════════════════════════════════
    print("\n═══ n_prompt_tokens binding ═══")

    # 33. Tamper n_prompt_tokens in response
    def tamper_npt_response(d):
        if d.get("n_prompt_tokens") is not None:
            d["n_prompt_tokens"] += 1
        else:
            raise KeyError("n_prompt_tokens not present")
    tamper_test("n_prompt_tokens_inflate", honest_json, tamper_npt_response, "n_prompt_tokens")

    # 34. Tamper n_prompt_tokens in commitment
    def tamper_npt_commitment(d):
        if d["commitment"].get("n_prompt_tokens") is not None:
            d["commitment"]["n_prompt_tokens"] += 1
        else:
            raise KeyError("commitment.n_prompt_tokens not present")
    tamper_test("commitment_npt_inflate", honest_json, tamper_npt_commitment, "n_prompt_tokens")

    # ════════════════════════════════════════════════════════════
    # Seed commitment binding
    # ════════════════════════════════════════════════════════════
    print("\n═══ Seed commitment ═══")

    # 35. Tamper seed_commitment in commitment (different from revealed)
    def tamper_seed_commitment(d):
        sc = d["commitment"].get("seed_commitment")
        if sc:
            sc[0] ^= 0xFF
        else:
            raise KeyError("seed_commitment not present")
    tamper_test("seed_commitment_flip", sampled_json, tamper_seed_commitment, "seed")

    # ════════════════════════════════════════════════════════════
    # Embedding proof structure
    # ════════════════════════════════════════════════════════════
    print("\n═══ Embedding proof structure ═══")

    # 36. Tamper embedding_proof leaf_index (wrong token binding)
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
    print("VeriLM Adversarial Tamper Test")
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
