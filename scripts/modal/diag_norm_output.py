"""
Quick diagnostic: what does model.norm actually return?

Check the type, shape, and tuple structure of model.norm's forward output
to understand why the captured post-norm hidden doesn't produce correct tokens.

Also check: does the LogitsProcessor see the same hidden state?

Usage:
    modal run --detach scripts/modal/diag_norm_output.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-norm-output")

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
    import torch
    import numpy as np
    import inspect

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from verilm import capture as cap

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # 1. Inspect model.norm module
    norm_mod = None
    for name, mod in model.named_modules():
        if name == "model.norm":
            norm_mod = mod
            print(f"\nmodel.norm: {type(mod).__name__}")
            print(f"  module: {mod}")
            # Print the forward method source
            try:
                src = inspect.getsource(type(mod).forward)
                print(f"  forward source:\n{src[:1500]}")
            except Exception as e:
                print(f"  forward source error: {e}")
            break

    # 2. Inspect LogitsProcessor
    logits_proc = None
    for name, mod in model.named_modules():
        if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
            logits_proc = mod
            print(f"\nLogitsProcessor: {type(mod).__name__} at '{name}'")
            try:
                src = inspect.getsource(type(mod).forward)
                print(f"  forward source:\n{src[:2000]}")
            except Exception as e:
                print(f"  forward source error: {e}")
            break

    # 3. Hook model.norm to inspect args and output
    norm_data = []

    def norm_hook(module, args, output):
        info = {
            "n_args": len(args),
            "args_types": [type(a).__name__ for a in args],
            "output_type": type(output).__name__,
        }
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                info[f"arg{i}_shape"] = tuple(a.shape)
                info[f"arg{i}_dtype"] = str(a.dtype)
                info[f"arg{i}_abs_mean"] = float(a.float().abs().mean())
            elif isinstance(a, (tuple, list)):
                info[f"arg{i}_len"] = len(a)
                for j, sub in enumerate(a[:3]):
                    if isinstance(sub, torch.Tensor):
                        info[f"arg{i}[{j}]_shape"] = tuple(sub.shape)

        if isinstance(output, tuple):
            info["output_len"] = len(output)
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    info[f"out{i}_shape"] = tuple(o.shape)
                    info[f"out{i}_dtype"] = str(o.dtype)
                    info[f"out{i}_abs_mean"] = float(o.float().abs().mean())
                elif o is None:
                    info[f"out{i}"] = "None"
        elif isinstance(output, torch.Tensor):
            info["out_shape"] = tuple(output.shape)
            info["out_dtype"] = str(output.dtype)
            info["out_abs_mean"] = float(output.float().abs().mean())

        norm_data.append(info)

    h1 = norm_mod.register_forward_hook(norm_hook)

    # 4. Hook LogitsProcessor to see what it receives
    lp_data = []

    def lp_hook(module, args, output):
        info = {
            "n_args": len(args),
            "args_types": [type(a).__name__ for a in args],
            "output_type": type(output).__name__,
        }
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                info[f"arg{i}_shape"] = tuple(a.shape)
                info[f"arg{i}_dtype"] = str(a.dtype)
                info[f"arg{i}_abs_mean"] = float(a.float().abs().mean())
        if isinstance(output, torch.Tensor):
            info["out_shape"] = tuple(output.shape)
            info["out_dtype"] = str(output.dtype)
        lp_data.append(info)

    h2 = logits_proc.register_forward_hook(lp_hook) if logits_proc else None

    # 5. Also hook to capture norm input vs output and compare with lm_head
    norm_io = []

    def norm_io_hook(module, args, output):
        inp = args[0] if len(args) > 0 and isinstance(args[0], torch.Tensor) else None
        out = output[0] if isinstance(output, tuple) else output
        if inp is not None and isinstance(out, torch.Tensor):
            norm_io.append({
                "input": inp.detach().cpu().float(),
                "output": out.detach().cpu().float(),
            })

    h3 = norm_mod.register_forward_hook(norm_io_hook)

    # 6. Run one generation
    print("\nRunning generation...")
    params = SamplingParams(max_tokens=4, temperature=0.0)
    output = llm.generate(["What is 2+2?"], params)[0]
    gen_tokens = list(output.outputs[0].token_ids)
    print(f"  Generated tokens: {gen_tokens}")
    print(f"  Generated text: {output.outputs[0].text}")

    # 7. Print captured data
    print(f"\n{'='*60}")
    print(f"model.norm hook data ({len(norm_data)} calls):")
    for i, d in enumerate(norm_data):
        print(f"  call {i}: {d}")

    print(f"\nLogitsProcessor hook data ({len(lp_data)} calls):")
    for i, d in enumerate(lp_data):
        print(f"  call {i}: {d}")

    # 8. Compare norm input vs output
    print(f"\n{'='*60}")
    print(f"Norm input vs output comparison ({len(norm_io)} calls):")
    for i, d in enumerate(norm_io):
        inp = d["input"]
        out = d["output"]
        print(f"  call {i}:")
        print(f"    input:  shape={inp.shape}, dtype=float32, abs_mean={inp.abs().mean():.4f}, max={inp.abs().max():.4f}")
        print(f"    output: shape={out.shape}, dtype=float32, abs_mean={out.abs().mean():.4f}, max={out.abs().max():.4f}")

        # Check if input == output (fused norm might return residual as output[0])
        if inp.shape == out.shape:
            diff = (inp - out).abs().max()
            print(f"    input==output? max_diff={diff:.6f}")

    # 9. Try to find what LogitsProcessor actually receives as hidden_states
    # by comparing norm output with LP input
    if norm_io and lp_data:
        print(f"\n  Norm output shape: {norm_io[0]['output'].shape}")
        if 'arg0_shape' in lp_data[0]:
            print(f"  LP arg0 shape: {lp_data[0]['arg0_shape']}")

    h1.remove()
    if h2:
        h2.remove()
    h3.remove()

    return {"norm_calls": len(norm_data), "lp_calls": len(lp_data)}


@app.function(image=image, gpu="A100-80GB", timeout=600)
def run_diag():
    return _run_diag()


@app.local_entrypoint()
def main():
    print("Diagnostic: model.norm output inspection")
    print("=" * 60)
    result = run_diag.remote()
    print(f"\nResult: {result}")
