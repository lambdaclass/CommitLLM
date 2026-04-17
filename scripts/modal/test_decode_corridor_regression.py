"""
Fail-fast smoke for the prefill -> decode corridor regression.

Checks one short greedy generation on W8A8 models and asserts that:
  - a prefill-side committed token stays in a very tight corridor
  - the first generated token stays in the historical single-digit corridor
  - the first decode token stays in the historical single-digit corridor

This is specifically meant to catch V1-engine / capture-path regressions
where prefill remains correct but decode-phase capture drifts catastrophically.

Usage:
    modal run scripts/modal/test_decode_corridor_regression.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION, VLLM_SPEC, TORCH_SPEC, TRANSFORMERS_SPEC, COMPRESSED_TENSORS_SPEC
except ImportError:
    VERIFICATION = []
    VLLM_SPEC = TORCH_SPEC = TRANSFORMERS_SPEC = COMPRESSED_TENSORS_SPEC = ""

import modal

app = modal.App("verilm-decode-corridor-regression")

PREFILL_MAX_LINF = int(os.environ.get("VERILM_PREFILL_MAX_LINF", "2"))
DECODE_MAX_LINF = int(os.environ.get("VERILM_DECODE_MAX_LINF", "12"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "ca-certificates")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
        "VERILM_CAPTURE": "1",
    })
    .pip_install(*VERIFICATION)
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf", "*.md",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)


def _extract_weight_scales(llm, n_layers, cfg):
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    hidden_dim = cfg["hidden_dim"]
    kv_dim = cfg["kv_dim"]
    scales = {"wq": [], "wk": [], "wv": []}

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        ws = layer.self_attn.qkv_proj.weight_scale
        ws_np = ws.detach().cpu().float().numpy().flatten()
        scales["wq"].append(ws_np[:hidden_dim].tolist())
        scales["wk"].append(ws_np[hidden_dim:hidden_dim + kv_dim].tolist())
        scales["wv"].append(ws_np[hidden_dim + kv_dim:].tolist())

    return scales


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def check_model(model_id: str):
    import hashlib
    import json

    import torch
    import verilm_rs
    import vllm
    from vllm import LLM

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    buf = get_capture_buffer()
    print(f"\n{'=' * 70}")
    print(f"Decode corridor regression smoke: {model_id}")
    print(f"{'=' * 70}")
    print(
        "Pinned runtime specs: "
        f"vllm={VLLM_SPEC}, transformers={TRANSFORMERS_SPEC}, "
        f"compressed_tensors={COMPRESSED_TENSORS_SPEC}, torch={TORCH_SPEC}"
    )

    llm = LLM(
        model=model_id,
        dtype="auto",
        max_model_len=4096,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    server = VerifiedInferenceServer(llm)
    print(
        f"Runtime: vllm={vllm.__version__} torch={torch.__version__} "
        f"backend={server._attn_backend} dtype={server._attn_dtype}"
    )

    n_layers = cap._n_layers
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    for _ in range(3):
        server.chat(prompt="Hello", max_tokens=8)

    key_seed = hashlib.sha256(model_id.encode()).digest()
    key_json = verilm_rs.generate_key(server._model_dir, key_seed)
    cfg = json.loads(key_json)["config"]
    cfg_dict = {
        "hidden_dim": cfg["hidden_dim"],
        "kv_dim": cfg["n_kv_heads"] * cfg["d_head"],
    }
    scale_overrides_json = json.dumps(_extract_weight_scales(llm, n_layers, cfg_dict))
    full_layers = list(range(n_layers))

    chat_r = server.chat(prompt="What is 2+2?", max_tokens=32, temperature=0.0)
    request_id = chat_r["request_id"]
    n_tokens = chat_r["n_tokens"]
    n_prompt = chat_r["commitment"]["n_prompt_tokens"]

    positions = [
        ("prefill", 0, PREFILL_MAX_LINF),
        ("first_gen", n_prompt - 1, DECODE_MAX_LINF),
        ("first_decode", min(n_prompt, n_tokens - 1), DECODE_MAX_LINF),
    ]

    results = []
    for label, pos, threshold in positions:
        if pos >= n_tokens:
            continue
        audit_binary = server.audit(
            request_id=request_id,
            token_index=pos,
            layer_indices=full_layers,
            tier="full",
            binary=True,
            use_captured_x_attn=True,
        )
        report = json.loads(
            verilm_rs.measure_corridor_committed_kv(
                audit_binary, key_json, scale_overrides_json
            )
        )
        frac_eq = sum(m["frac_eq"] for m in report["measurements"]) / max(
            len(report["measurements"]), 1
        )
        entry = {
            "label": label,
            "pos": pos,
            "global_linf": report["global_linf"],
            "threshold": threshold,
            "avg_frac_eq": frac_eq,
        }
        results.append(entry)
        print(
            f"  {label:>11s} pos={pos:3d} "
            f"L-inf={report['global_linf']:3d} "
            f"frac_eq={frac_eq:.4f} "
            f"threshold={threshold}"
        )
        if report["global_linf"] > threshold:
            raise AssertionError(
                f"{model_id} {label} corridor regression: "
                f"L-inf={report['global_linf']} > threshold={threshold}"
            )

    return {
        "model": model_id,
        "results": results,
        "thresholds": {
            "prefill": PREFILL_MAX_LINF,
            "decode": DECODE_MAX_LINF,
        },
    }


@app.local_entrypoint()
def main():
    models = os.environ.get(
        "VERILM_CORRIDOR_MODELS",
        "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8,"
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
    )
    for model_id in [m.strip() for m in models.split(",") if m.strip()]:
        check_model.remote(model_id)
