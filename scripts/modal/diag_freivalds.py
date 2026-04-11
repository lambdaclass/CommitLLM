"""
Freivalds failure classifier: determine whether failures are bridge or weight-path.

One Modal run that:
  1. Runs honest full audit after rmsnorm_eps fix
  2. Classifies failing Freivalds checks by matrix family
  3. If Wo fails → compares runtime vLLM params vs safetensors on disk
  4. Lightweight evidence: shape, dtype, SHA-256, first 8 values, max abs diff

Usage:
    modal run scripts/modal/diag_freivalds.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-freivalds")

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
    .pip_install(*VERIFICATION, "httpx")
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
MAX_TOKENS = 64


def _run_diag():
    import hashlib
    import json
    import os
    import re

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import numpy as np
    import torch
    import verilm_rs
    from fastapi.testclient import TestClient
    from safetensors import safe_open
    from vllm import LLM

    from verilm import capture as cap
    from verilm.server import create_app

    results = {}

    # ── Load model ──
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

    results["model_dir"] = model_dir
    results["n_layers"] = n_layers

    # ── Generate key ──
    seed = hashlib.sha256(PROMPT.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)
    key_data = json.loads(key_json)
    results["key_rmsnorm_eps"] = key_data.get("rmsnorm_eps")
    print(f"Key rmsnorm_eps: {key_data.get('rmsnorm_eps')}")

    # Check provider rmsnorm_eps
    provider = verilm_rs.WeightProvider(model_dir)
    # Provider eps is used at audit time — not directly queryable from Python,
    # but we can check config.json
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        import json as json_mod
        with open(cfg_path) as f:
            model_cfg = json_mod.load(f)
        results["config_json_rms_norm_eps"] = model_cfg.get("rms_norm_eps")
        results["config_json_rope_theta"] = model_cfg.get("rope_theta")
        print(f"config.json rms_norm_eps: {model_cfg.get('rms_norm_eps')}")
        print(f"config.json rope_theta: {model_cfg.get('rope_theta')}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Run honest audit, classify failures
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Honest audit + Freivalds classification")
    print("=" * 60)

    resp = http.post("/chat", json={
        "prompt": PROMPT, "n_tokens": MAX_TOKENS, "temperature": 0.0,
    })
    assert resp.status_code == 200, f"chat failed: {resp.status_code}"
    chat = resp.json()
    print(f"Generated {chat['n_tokens']} tokens")

    resp = http.post("/audit", json={
        "request_id": chat["request_id"],
        "token_index": 0,
        "layer_indices": list(range(n_layers)),
        "tier": "full",
        "binary": False,
    })
    assert resp.status_code == 200, f"audit failed: {resp.status_code}"
    audit_json = resp.text
    audit = json.loads(audit_json)

    report = verilm_rs.verify_v4(audit_json, key_json)
    results["checks_run"] = report["checks_run"]
    results["checks_passed"] = report["checks_passed"]
    results["passed"] = report["passed"]
    results["total_failures"] = len(report["failures"])

    print(f"Verdict: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Checks: {report['checks_passed']}/{report['checks_run']}")

    # Classify failures by matrix family
    freivalds_pattern = re.compile(r"layer (\d+) (W[a-z]+): Freivalds")
    lmhead_pattern = re.compile(r"lm_head.*[Ff]reivalds|[Ff]reivalds.*lm_head")

    families = {"Wo": [], "Wq": [], "Wk": [], "Wv": [], "Wg": [], "Wu": [], "Wd": [], "lm_head": []}
    non_freivalds = []

    for f in report["failures"]:
        m = freivalds_pattern.search(f)
        if m:
            layer, matrix = int(m.group(1)), m.group(2)
            if matrix in families:
                families[matrix].append(layer)
            continue
        if lmhead_pattern.search(f):
            families["lm_head"].append(-1)
            continue
        non_freivalds.append(f)

    print("\n── Freivalds failure classification ──")
    for matrix, layers in families.items():
        if layers:
            if matrix == "lm_head":
                print(f"  {matrix}: FAIL")
            else:
                print(f"  {matrix}: FAIL on {len(layers)} layers (first: {layers[0]})")
        else:
            print(f"  {matrix}: PASS")

    if non_freivalds:
        print(f"\n── Non-Freivalds failures ({len(non_freivalds)}) ──")
        for f in non_freivalds[:10]:
            print(f"  - {f}")

    results["freivalds_families"] = {k: len(v) for k, v in families.items()}
    results["non_freivalds_failures"] = non_freivalds

    # ── Classification verdict ──
    wo_fails = len(families["Wo"]) > 0
    qkv_fails = any(len(families[m]) > 0 for m in ["Wq", "Wk", "Wv"])
    ffn_fails = any(len(families[m]) > 0 for m in ["Wg", "Wu", "Wd"])
    lmhead_fails = len(families["lm_head"]) > 0

    print("\n── Diagnosis ──")
    if wo_fails:
        print("  Wo FAILS → weight-path mismatch (not just bridge)")
    elif qkv_fails or ffn_fails:
        print("  Wo PASSES, QKV/FFN fail → bridge/replay issue (rmsnorm_eps fix may help)")
    elif lmhead_fails:
        print("  Only lm_head fails → tail path problem")
    elif not report["passed"]:
        print("  No Freivalds failures, but other checks fail")
    else:
        print("  ALL PASS — rmsnorm_eps fix resolved the issue")

    results["diagnosis"] = (
        "weight_path_mismatch" if wo_fails else
        "bridge_replay_issue" if (qkv_fails or ffn_fails) else
        "tail_path_issue" if lmhead_fails else
        "non_freivalds_issue" if not report["passed"] else
        "all_pass"
    )

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Runtime vs safetensors weight comparison
    # (only if Wo fails, suggesting weight-path mismatch)
    # ══════════════════════════════════════════════════════════════
    if wo_fails or qkv_fails or ffn_fails:
        print("\n" + "=" * 60)
        print("PHASE 2: Runtime vLLM params vs safetensors weight comparison")
        print("=" * 60)

        # Get the actual vLLM model
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model

        # Find safetensors files
        st_files = sorted([
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir) if f.endswith(".safetensors")
        ])

        def load_safetensors_tensor(name):
            """Load a tensor from safetensors by name."""
            for f in st_files:
                with safe_open(f, framework="pt") as st:
                    if name in st.keys():
                        return st.get_tensor(name)
            return None

        def tensor_summary(name, t):
            """Lightweight summary: shape, dtype, SHA-256 prefix, first 8 values."""
            flat = t.contiguous().view(-1)
            raw = flat.numpy().tobytes() if t.device.type == "cpu" else flat.cpu().numpy().tobytes()
            sha = hashlib.sha256(raw).hexdigest()[:16]
            first8 = flat[:8].tolist()
            return {
                "name": name,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "sha256_prefix": sha,
                "first_8": first8,
            }

        def compare_tensors(name, runtime_t, safetensors_t):
            """Compare two tensors: max abs diff, shape match, byte identity."""
            rt = runtime_t.cpu() if runtime_t.device.type != "cpu" else runtime_t
            st = safetensors_t.cpu() if safetensors_t.device.type != "cpu" else safetensors_t

            result = {
                "name": name,
                "runtime": tensor_summary("runtime", rt),
                "safetensors": tensor_summary("safetensors", st),
                "shape_match": list(rt.shape) == list(st.shape),
            }

            if list(rt.shape) == list(st.shape) and rt.dtype == st.dtype:
                if rt.dtype in (torch.int8, torch.uint8):
                    diff = (rt.to(torch.int16) - st.to(torch.int16)).abs()
                else:
                    diff = (rt.float() - st.float()).abs()
                result["max_abs_diff"] = diff.max().item()
                result["byte_identical"] = torch.equal(rt, st)
                result["n_different"] = int((diff > 0).sum().item())
            else:
                result["max_abs_diff"] = None
                result["byte_identical"] = False

            return result

        comparisons = []

        # Layer 0 comparisons
        print("\n── Layer 0 weight comparisons ──")

        # 1. o_proj: runtime vs safetensors (Wo — the critical discriminator)
        for layer_idx in [0]:
            prefix = f"model.layers.{layer_idx}"

            # o_proj — unfused, direct comparison
            rt_o = None
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}.self_attn.o_proj.weight" in name:
                    rt_o = param.data
                    break
            st_o = load_safetensors_tensor(f"{prefix}.self_attn.o_proj.weight")
            if rt_o is not None and st_o is not None:
                cmp = compare_tensors(f"layer{layer_idx}.o_proj", rt_o, st_o)
                comparisons.append(cmp)
                print(f"\n  o_proj (Wo):")
                print(f"    runtime:     shape={cmp['runtime']['shape']} dtype={cmp['runtime']['dtype']} sha={cmp['runtime']['sha256_prefix']}")
                print(f"    safetensors: shape={cmp['safetensors']['shape']} dtype={cmp['safetensors']['dtype']} sha={cmp['safetensors']['sha256_prefix']}")
                print(f"    byte_identical={cmp['byte_identical']}  max_abs_diff={cmp['max_abs_diff']}  n_different={cmp.get('n_different', '?')}")

            # down_proj — unfused, direct comparison
            rt_d = None
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}.mlp.down_proj.weight" in name:
                    rt_d = param.data
                    break
            st_d = load_safetensors_tensor(f"{prefix}.mlp.down_proj.weight")
            if rt_d is not None and st_d is not None:
                cmp = compare_tensors(f"layer{layer_idx}.down_proj", rt_d, st_d)
                comparisons.append(cmp)
                print(f"\n  down_proj (Wd):")
                print(f"    runtime:     shape={cmp['runtime']['shape']} dtype={cmp['runtime']['dtype']} sha={cmp['runtime']['sha256_prefix']}")
                print(f"    safetensors: shape={cmp['safetensors']['shape']} dtype={cmp['safetensors']['dtype']} sha={cmp['safetensors']['sha256_prefix']}")
                print(f"    byte_identical={cmp['byte_identical']}  max_abs_diff={cmp['max_abs_diff']}  n_different={cmp.get('n_different', '?')}")

            # qkv_proj: runtime fused vs safetensors separate (concatenated)
            rt_qkv = None
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}.self_attn.qkv_proj.weight" in name:
                    rt_qkv = param.data
                    break
            st_q = load_safetensors_tensor(f"{prefix}.self_attn.q_proj.weight")
            st_k = load_safetensors_tensor(f"{prefix}.self_attn.k_proj.weight")
            st_v = load_safetensors_tensor(f"{prefix}.self_attn.v_proj.weight")
            if rt_qkv is not None and st_q is not None:
                st_qkv = torch.cat([st_q, st_k, st_v], dim=0)
                cmp = compare_tensors(f"layer{layer_idx}.qkv_proj", rt_qkv, st_qkv)
                comparisons.append(cmp)
                print(f"\n  qkv_proj (runtime fused) vs cat(q,k,v) (safetensors):")
                print(f"    runtime:     shape={cmp['runtime']['shape']} dtype={cmp['runtime']['dtype']} sha={cmp['runtime']['sha256_prefix']}")
                print(f"    safetensors: shape={cmp['safetensors']['shape']} dtype={cmp['safetensors']['dtype']} sha={cmp['safetensors']['sha256_prefix']}")
                print(f"    byte_identical={cmp['byte_identical']}  max_abs_diff={cmp['max_abs_diff']}  n_different={cmp.get('n_different', '?')}")
                print(f"    runtime first 8:     {cmp['runtime']['first_8']}")
                print(f"    safetensors first 8: {cmp['safetensors']['first_8']}")

            # gate_up_proj: runtime fused vs safetensors separate
            rt_gu = None
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}.mlp.gate_up_proj.weight" in name:
                    rt_gu = param.data
                    break
            st_g = load_safetensors_tensor(f"{prefix}.mlp.gate_proj.weight")
            st_u = load_safetensors_tensor(f"{prefix}.mlp.up_proj.weight")
            if rt_gu is not None and st_g is not None:
                st_gu = torch.cat([st_g, st_u], dim=0)
                cmp = compare_tensors(f"layer{layer_idx}.gate_up_proj", rt_gu, st_gu)
                comparisons.append(cmp)
                print(f"\n  gate_up_proj (runtime fused) vs cat(gate,up) (safetensors):")
                print(f"    runtime:     shape={cmp['runtime']['shape']} dtype={cmp['runtime']['dtype']} sha={cmp['runtime']['sha256_prefix']}")
                print(f"    safetensors: shape={cmp['safetensors']['shape']} dtype={cmp['safetensors']['dtype']} sha={cmp['safetensors']['sha256_prefix']}")
                print(f"    byte_identical={cmp['byte_identical']}  max_abs_diff={cmp['max_abs_diff']}  n_different={cmp.get('n_different', '?')}")

            # Also check if safetensors has separate q/k/v or already fused
            print(f"\n── Safetensors tensor inventory (layer 0 attn) ──")
            for f in st_files:
                with safe_open(f, framework="pt") as st:
                    for k in sorted(st.keys()):
                        if "layers.0.self_attn" in k:
                            t = st.get_tensor(k)
                            print(f"    {k}: shape={list(t.shape)} dtype={t.dtype}")

            print(f"\n── Runtime parameter inventory (layer 0 attn) ──")
            for name, param in model.named_parameters():
                if "layers.0.self_attn" in name:
                    print(f"    {name}: shape={list(param.shape)} dtype={param.dtype} device={param.device}")

        results["weight_comparisons"] = comparisons

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Diagnosis: {results['diagnosis']}")
    print(f"  Key rmsnorm_eps: {results.get('key_rmsnorm_eps')}")
    print(f"  Config rms_norm_eps: {results.get('config_json_rms_norm_eps')}")
    print(f"  Freivalds families: {results.get('freivalds_families')}")
    if "weight_comparisons" in results:
        for c in results["weight_comparisons"]:
            ident = "IDENTICAL" if c.get("byte_identical") else f"DIFFER (max_diff={c.get('max_abs_diff')}, n={c.get('n_different')})"
            print(f"  {c['name']}: {ident}")

    return results


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_diag():
    return _run_diag()


@app.local_entrypoint()
def main():
    print("VeriLM Freivalds Diagnostic")
    print("=" * 60)
    result = run_diag.remote()
    print(f"\nDiagnosis: {result['diagnosis']}")
