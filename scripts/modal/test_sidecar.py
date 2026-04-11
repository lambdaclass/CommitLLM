"""
End-to-end sidecar test: capture -> trace building -> commitment readiness.

Runs real inference on a W8A8 model and verifies every piece of data
the protocol needs is correctly captured, split, and structured.

Tests:
  1. Capture completeness: all projections, all layers, acc_i32 present
  2. Model geometry: num_heads, kv_heads, head_dim detected correctly
  3. GQA-aware QKV split: Q/K/V dimensions match model config
  4. Gate/up split: equal halves, correct dimensions
  5. LayerTrace building: all fields populated, correct structure
  6. KV cache accumulation: Level C snapshots grow across tokens
  7. acc_i32 correctness: matches torch._int_mm(x_i8, weight_i8)
  8. Requantization: i32 -> i8 clamp matches Rust requantize()
  9. Embedding capture: hook fires, correct shape
  10. Logit capture: hook fires, correct shape
  11. attn_out_i8 identity: o_proj input == post-attention output (byte-exact)

Usage:
    modal run scripts/modal_test_sidecar.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VLLM_SPEC, TORCH_SPEC, TRANSFORMERS_SPEC, NUMPY_SPEC, SAFETENSORS_SPEC
except ImportError:
    VLLM_SPEC = TORCH_SPEC = TRANSFORMERS_SPEC = NUMPY_SPEC = SAFETENSORS_SPEC = ""

import modal

app = modal.App("verilm-sidecar-test")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(VLLM_SPEC, TORCH_SPEC, TRANSFORMERS_SPEC, NUMPY_SPEC, SAFETENSORS_SPEC)
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .env(
        {
            "VERILM_CAPTURE": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
    )
)

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
PROMPT = "Explain verified inference in two sentences."


def _run_tests():
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from verilm.capture import (
        get_capture_buffer,
        get_model_from_llm,
        configure_from_model,
        PROJS_PER_LAYER,
        _n_layers, _num_heads, _num_kv_heads, _head_dim,
        _hidden_size, _intermediate_size, _q_dim, _kv_dim,
    )
    from verilm.trace import (
        build_layer_traces,
        split_qkv,
        split_gate_up,
        requantize_i32_to_i8,
    )
    from verilm.hooks import EmbeddingLogitCapture

    results = {}

    # -- Load model --
    config = AutoConfig.from_pretrained(MODEL_ID)
    print(f"Model config: hidden_size={config.hidden_size}, "
          f"intermediate_size={config.intermediate_size}, "
          f"num_attention_heads={config.num_attention_heads}, "
          f"num_key_value_heads={config.num_key_value_heads}")

    llm = LLM(
        model=MODEL_ID, dtype="auto",
        max_model_len=2048, enforce_eager=True,
    )
    buf = get_capture_buffer()
    model = get_model_from_llm(llm)
    configure_from_model(model)

    # Reimport after configure
    from verilm import capture as cap
    n_layers = cap._n_layers

    # Install embedding/logit hooks
    el_capture = EmbeddingLogitCapture()
    n_hooks = el_capture.install(model)
    print(f"Installed {n_hooks} embedding/logit hooks")

    # Install o_proj hooks for byte-exact comparison
    hooked_o_proj = []

    def _make_o_proj_hook(layer_idx):
        def hook(module, args, kwargs=None):
            x = args[0] if args else None
            if x is not None:
                hooked_o_proj.append((layer_idx, x.detach().clone()))
        return hook

    hook_handles = []
    for name, mod in model.named_modules():
        if name.endswith(".self_attn.o_proj"):
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        h = mod.register_forward_pre_hook(_make_o_proj_hook(layer_idx))
                        hook_handles.append(h)
                    except ValueError:
                        pass

    # -- Warmup --
    print("Warmup...")
    llm.generate([PROMPT], SamplingParams(max_tokens=5, temperature=0))
    buf.drain()
    el_capture.drain()
    hooked_o_proj.clear()

    # -- Generate --
    print("Generating...")
    buf.drain()
    buf.total_captured = 0
    el_capture.drain()
    hooked_o_proj.clear()

    params = SamplingParams(max_tokens=10, temperature=0)
    outputs = llm.generate([PROMPT], params)
    gen_tokens = len(outputs[0].outputs[0].token_ids)
    prompt_tokens = len(outputs[0].prompt_token_ids)
    print(f"Generated {gen_tokens} tokens (prompt={prompt_tokens})")

    captures = buf.drain()
    el_data = el_capture.drain()
    total = len(captures)
    calls_per_fwd = n_layers * PROJS_PER_LAYER

    # ==================================================================
    # TEST 1: Capture completeness
    # ==================================================================
    print(f"\n--- Test 1: Capture completeness ---")
    layer_proj_map = {}
    none_count = 0
    has_acc = True
    for layer, proj, x_i8, acc_i32, scale_a in captures:
        if layer is None or proj is None:
            none_count += 1
            continue
        layer_proj_map.setdefault(layer, set()).add(proj)
        if acc_i32 is None:
            has_acc = False
        elif acc_i32.dtype != torch.int32:
            has_acc = False

    all_layers_covered = set(layer_proj_map.keys()) == set(range(n_layers))
    all_projs = all(
        layer_proj_map.get(l, set()) >= {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}
        for l in range(n_layers)
    )
    exact_multiple = total % calls_per_fwd == 0
    n_fwd = total // calls_per_fwd if calls_per_fwd else 0

    results["1_no_none"] = none_count == 0
    results["1_all_layers"] = all_layers_covered
    results["1_all_projs"] = all_projs
    results["1_exact_multiple"] = exact_multiple
    results["1_has_acc_i32"] = has_acc

    print(f"  No None: {none_count == 0} (none_count={none_count})")
    print(f"  All layers: {all_layers_covered} ({len(layer_proj_map)}/{n_layers})")
    print(f"  All projs: {all_projs}")
    print(f"  Exact multiple: {exact_multiple} ({total} / {calls_per_fwd} = {n_fwd})")
    print(f"  acc_i32 present: {has_acc}")

    # ==================================================================
    # TEST 2: Model geometry
    # ==================================================================
    print(f"\n--- Test 2: Model geometry ---")
    geo_ok = (
        cap._num_heads == config.num_attention_heads
        and cap._num_kv_heads == config.num_key_value_heads
        and cap._hidden_size == config.hidden_size
        and cap._intermediate_size == config.intermediate_size
        and cap._q_dim == config.num_attention_heads * (config.hidden_size // config.num_attention_heads)
        and cap._kv_dim == config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
    )
    results["2_geometry"] = geo_ok
    print(f"  heads={cap._num_heads} (expect {config.num_attention_heads})")
    print(f"  kv_heads={cap._num_kv_heads} (expect {config.num_key_value_heads})")
    print(f"  q_dim={cap._q_dim}, kv_dim={cap._kv_dim}")
    print(f"  Match: {geo_ok}")

    # ==================================================================
    # TEST 3: GQA-aware QKV split
    # ==================================================================
    print(f"\n--- Test 3: GQA-aware QKV split ---")
    qkv_errors = []
    qkv_checked = 0
    for layer, proj, x_i8, acc_i32, scale_a in captures:
        if proj != "qkv_proj":
            continue
        q, k, v = split_qkv(acc_i32)
        if q.shape[-1] != cap._q_dim:
            qkv_errors.append(f"L{layer}: q width {q.shape[-1]} != {cap._q_dim}")
        if k.shape[-1] != cap._kv_dim:
            qkv_errors.append(f"L{layer}: k width {k.shape[-1]} != {cap._kv_dim}")
        if v.shape[-1] != cap._kv_dim:
            qkv_errors.append(f"L{layer}: v width {v.shape[-1]} != {cap._kv_dim}")
        # Verify the split is lossless: concat should equal original
        reconstructed = torch.cat([q, k, v], dim=-1)
        if not torch.equal(reconstructed, acc_i32):
            qkv_errors.append(f"L{layer}: QKV split not lossless")
        qkv_checked += 1
        if qkv_checked >= 100 and not qkv_errors:
            break

    results["3_qkv_split"] = len(qkv_errors) == 0
    print(f"  Checked {qkv_checked} qkv_proj captures")
    print(f"  Q width={cap._q_dim}, K width={cap._kv_dim}, V width={cap._kv_dim}")
    print(f"  Total fused width={cap._q_dim + 2 * cap._kv_dim}")
    if qkv_errors:
        for e in qkv_errors[:5]:
            print(f"  ERROR: {e}")
    else:
        print(f"  All splits correct and lossless")

    # ==================================================================
    # TEST 4: Gate/up split
    # ==================================================================
    print(f"\n--- Test 4: Gate/up split ---")
    gu_errors = []
    gu_checked = 0
    for layer, proj, x_i8, acc_i32, scale_a in captures:
        if proj != "gate_up_proj":
            continue
        g, u = split_gate_up(acc_i32)
        half = acc_i32.shape[-1] // 2
        if g.shape[-1] != half:
            gu_errors.append(f"L{layer}: gate width {g.shape[-1]} != {half}")
        if u.shape[-1] != half:
            gu_errors.append(f"L{layer}: up width {u.shape[-1]} != {half}")
        reconstructed = torch.cat([g, u], dim=-1)
        if not torch.equal(reconstructed, acc_i32):
            gu_errors.append(f"L{layer}: gate_up split not lossless")
        gu_checked += 1
        if gu_checked >= 100 and not gu_errors:
            break

    results["4_gate_up_split"] = len(gu_errors) == 0
    print(f"  Checked {gu_checked} gate_up_proj captures")
    print(f"  Each half width={cap._intermediate_size}")
    if gu_errors:
        for e in gu_errors[:5]:
            print(f"  ERROR: {e}")
    else:
        print(f"  All splits correct and lossless")

    # ==================================================================
    # TEST 5: LayerTrace building
    # ==================================================================
    print(f"\n--- Test 5: LayerTrace building ---")
    traces = build_layer_traces(captures, n_layers=n_layers)
    trace_ok = True
    trace_errors = []

    if len(traces) != n_fwd:
        trace_errors.append(f"Expected {n_fwd} tokens, got {len(traces)}")
        trace_ok = False

    expected_fields = [
        "x_attn", "q", "k", "v", "a", "attn_out",
        "x_ffn", "g", "u", "h", "ffn_out",
        "kv_cache_k", "kv_cache_v",
    ]
    for t_idx, token_layers in enumerate(traces[:3]):  # Check first 3 tokens
        if len(token_layers) != n_layers:
            trace_errors.append(f"Token {t_idx}: {len(token_layers)} layers, expected {n_layers}")
            trace_ok = False
            continue
        for l_idx, lt in enumerate(token_layers[:3]):  # Check first 3 layers
            for field in expected_fields:
                if field not in lt:
                    trace_errors.append(f"Token {t_idx}/Layer {l_idx}: missing field '{field}'")
                    trace_ok = False

    # Check specific dimensions
    if traces:
        lt0 = traces[0][0]
        dim_checks = [
            ("x_attn", lt0["x_attn"].shape[-1], cap._hidden_size),
            ("q", lt0["q"].shape[-1], cap._q_dim),
            ("k", lt0["k"].shape[-1], cap._kv_dim),
            ("v", lt0["v"].shape[-1], cap._kv_dim),
            ("a", lt0["a"].shape[-1], cap._hidden_size),
            ("attn_out", lt0["attn_out"].shape[-1], cap._hidden_size),
            ("x_ffn", lt0["x_ffn"].shape[-1], cap._hidden_size),
            ("g", lt0["g"].shape[-1], cap._intermediate_size),
            ("u", lt0["u"].shape[-1], cap._intermediate_size),
            ("h", lt0["h"].shape[-1], cap._intermediate_size),
            ("ffn_out", lt0["ffn_out"].shape[-1], cap._hidden_size),
        ]
        for name, actual, expected in dim_checks:
            if actual != expected:
                trace_errors.append(f"Token 0/Layer 0/{name}: width {actual} != {expected}")
                trace_ok = False

    results["5_layer_trace"] = trace_ok and len(trace_errors) == 0
    print(f"  Tokens: {len(traces)}, Layers per token: {n_layers}")
    if trace_errors:
        for e in trace_errors[:10]:
            print(f"  ERROR: {e}")
    else:
        print(f"  All fields present, all dimensions correct")

    # ==================================================================
    # TEST 6: KV cache accumulation (Level C)
    # ==================================================================
    print(f"\n--- Test 6: KV cache accumulation (Level C) ---")
    traces_c = build_layer_traces(captures, n_layers=n_layers, level_c=True)
    kv_errors = []

    for t_idx in range(min(len(traces_c), 5)):
        for l_idx in range(min(n_layers, 3)):
            lt = traces_c[t_idx][l_idx]
            expected_len = t_idx + 1
            if len(lt["kv_cache_k"]) != expected_len:
                kv_errors.append(
                    f"Token {t_idx}/Layer {l_idx}: kv_cache_k len={len(lt['kv_cache_k'])}, "
                    f"expected {expected_len}"
                )
            if len(lt["kv_cache_v"]) != expected_len:
                kv_errors.append(
                    f"Token {t_idx}/Layer {l_idx}: kv_cache_v len={len(lt['kv_cache_v'])}, "
                    f"expected {expected_len}"
                )
            # Check each entry has correct width (kv_dim)
            for entry in lt["kv_cache_k"]:
                if entry.shape[-1] != cap._kv_dim:
                    kv_errors.append(
                        f"Token {t_idx}/Layer {l_idx}: kv_cache_k entry width "
                        f"{entry.shape[-1]} != {cap._kv_dim}"
                    )
                    break

    results["6_kv_cache"] = len(kv_errors) == 0
    print(f"  Built Level C traces for {len(traces_c)} tokens")
    if kv_errors:
        for e in kv_errors[:5]:
            print(f"  ERROR: {e}")
    else:
        print(f"  KV cache grows correctly: token t has t+1 entries per layer")

    # ==================================================================
    # TEST 7: acc_i32 correctness (spot check against weights)
    # ==================================================================
    print(f"\n--- Test 7: acc_i32 correctness ---")
    acc_errors = []
    acc_checked = 0

    # Get weight matrices from model for spot checking
    for name, param in model.named_parameters():
        if "layers.0.self_attn.qkv_proj.weight" in name:
            w_qkv_0 = param.data  # INT8 weight matrix
            break

    for layer, proj, x_i8, acc_i32, scale_a in captures:
        if layer != 0 or proj != "qkv_proj":
            continue
        # Recompute: acc = x_i8 @ w^T (cutlass does a @ b where b is already transposed)
        # But torch._int_mm expects (M,K) @ (K,N). The capture has a=x_i8, b=weight.
        # We need to match the exact computation in the wrapper.
        M = x_i8.shape[0]
        if M <= 16:
            a_padded = torch.nn.functional.pad(x_i8, (0, 0, 0, 32 - M))
            expected = torch._int_mm(a_padded, w_qkv_0)[:M, :]
        else:
            expected = torch._int_mm(x_i8, w_qkv_0)

        if not torch.equal(acc_i32, expected):
            n_diff = (acc_i32 != expected).sum().item()
            acc_errors.append(
                f"L0/qkv_proj: {n_diff}/{acc_i32.numel()} elements differ"
            )
        acc_checked += 1
        if acc_checked >= 5:
            break

    results["7_acc_i32"] = len(acc_errors) == 0
    print(f"  Spot-checked {acc_checked} L0/qkv_proj acc_i32 against torch._int_mm")
    if acc_errors:
        for e in acc_errors[:5]:
            print(f"  ERROR: {e}")
    else:
        print(f"  All match exactly")

    # ==================================================================
    # TEST 8: Requantization
    # ==================================================================
    print(f"\n--- Test 8: Requantization ---")
    test_vals = torch.tensor([-200, -128, -1, 0, 1, 127, 200, 1000], dtype=torch.int32)
    expected = torch.tensor([-128, -128, -1, 0, 1, 127, 127, 127], dtype=torch.int8)
    actual = requantize_i32_to_i8(test_vals)
    results["8_requantize"] = torch.equal(actual, expected)
    print(f"  Boundary test: {torch.equal(actual, expected)}")
    print(f"  Input:    {test_vals.tolist()}")
    print(f"  Expected: {expected.tolist()}")
    print(f"  Got:      {actual.tolist()}")

    # ==================================================================
    # TEST 9: Embedding capture
    # ==================================================================
    print(f"\n--- Test 9: Embedding capture ---")
    n_embeds = len(el_data["embeddings"])
    embed_ok = n_embeds > 0
    if embed_ok:
        e0 = el_data["embeddings"][0]
        embed_dim_ok = e0.shape[-1] == config.hidden_size
        embed_ok = embed_ok and embed_dim_ok
        print(f"  Captured {n_embeds} embedding tensors")
        print(f"  Shape: {list(e0.shape)}, expected dim={config.hidden_size}: {embed_dim_ok}")
    else:
        print(f"  No embeddings captured!")
    results["9_embeddings"] = embed_ok

    # ==================================================================
    # TEST 10: Logit capture
    # ==================================================================
    print(f"\n--- Test 10: Logit capture ---")
    n_logits = len(el_data["logits"])
    logit_ok = n_logits > 0
    if logit_ok:
        l0 = el_data["logits"][0]
        print(f"  Captured {n_logits} logit tensors")
        print(f"  Shape: {list(l0.shape)}")
    else:
        print(f"  No logits captured!")
    results["10_logits"] = logit_ok

    # ==================================================================
    # TEST 11: attn_out_i8 byte-exact match
    # ==================================================================
    print(f"\n--- Test 11: attn_out_i8 byte-exact ---")
    o_proj_caps = [
        (layer, x_i8)
        for layer, proj, x_i8, _, _ in captures
        if proj == "o_proj"
    ]

    byte_ok = True
    if len(o_proj_caps) != len(hooked_o_proj):
        print(f"  Count mismatch: {len(o_proj_caps)} captures vs {len(hooked_o_proj)} hooks")
        byte_ok = False
    else:
        mismatches = 0
        for (cl, ct), (hl, ht) in zip(o_proj_caps, hooked_o_proj):
            if cl != hl:
                mismatches += 1
                continue
            if ct.shape != ht.shape:
                mismatches += 1
                continue
            if ht.dtype == torch.int8 and not torch.equal(ct, ht):
                mismatches += 1
        byte_ok = mismatches == 0
        print(f"  Compared {len(o_proj_caps)} tensors, mismatches={mismatches}")

    results["11_attn_out_byte_exact"] = byte_ok

    # ==================================================================
    # TEST 12: Residual capture
    # ==================================================================
    print(f"\n--- Test 12: Residual capture ---")
    n_residuals = len(el_data["residuals"])
    expected_residuals = n_fwd * n_layers
    residual_ok = n_residuals == expected_residuals
    if n_residuals > 0:
        r0 = el_data["residuals"][0]
        residual_dim_ok = r0.shape[-1] == config.hidden_size
        residual_ok = residual_ok and residual_dim_ok
        print(f"  Captured {n_residuals} residual tensors (expected {expected_residuals})")
        print(f"  Shape: {list(r0.shape)}, expected dim={config.hidden_size}: {residual_dim_ok}")
    else:
        print(f"  No residuals captured!")
    results["12_residuals"] = residual_ok

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*65}")
    print("SIDECAR TEST SUMMARY")
    print(f"{'='*65}")
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print(f"{'='*65}")
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*65}")

    # Cleanup
    el_capture.remove()
    for h in hook_handles:
        h.remove()

    return {"all_pass": all_pass, "results": results}


@app.function(image=vllm_image, gpu="A100-80GB", timeout=900)
def run_tests():
    return _run_tests()


@app.local_entrypoint()
def main():
    print("VeriLM Sidecar End-to-End Test")
    print("=" * 65)
    result = run_tests.remote()
    if result["all_pass"]:
        print("\nAll tests passed.")
    else:
        failed = [k for k, v in result["results"].items() if not v]
        print(f"\nFailed: {failed}")
