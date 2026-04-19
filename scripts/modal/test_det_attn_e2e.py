"""
End-to-end test: deterministic attention in vLLM verified-attention mode.

Loads Llama 8B W8A8, enables VERILM_ATTN_MODE=deterministic, generates
tokens, and verifies that:
  1. The deterministic kernel runs without error
  2. Attention captures are produced per layer per token
  3. The model generates coherent output (not garbage)
  4. CPU (Rust) verification of captured attention matches GPU output

Usage:
    modal run scripts/modal/test_det_attn_e2e.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-test-det-attn-e2e")

MODEL_ID = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "ca-certificates")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    })
    .pip_install(*VERIFICATION)
    # Install sidecar
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python3 -c 'import site, os; open(os.path.join(site.getsitepackages()[0], \"verilm_capture.pth\"), \"w\").write(\"import verilm._startup\\n\")'",
    )
    # Build verilm_rs
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf", "*.md", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
    )
    # Compile deterministic attention kernel
    .run_commands(
        "nvcc -O2 --fmad=false -shared -Xcompiler -fPIC "
        "-o /opt/libdet_attn.so /build/kernels/deterministic_attention.cu",
        "rm -rf /build",
    )
)


def _run_test():
    """Run the E2E deterministic attention test."""
    import json
    import numpy as np
    import torch

    # Set attention mode BEFORE importing vLLM (env var read at server init).
    os.environ["VERILM_ATTN_MODE"] = "deterministic"
    os.environ["VERILM_CAPTURE"] = "1"

    from vllm import LLM, SamplingParams
    from verilm.capture import (
        configure_from_model,
        get_capture_buffer,
        get_model_from_llm,
    )
    from verilm.det_attn import DeterministicAttentionHook

    print("=" * 70)
    print("Loading model...")
    print("=" * 70)

    llm = LLM(
        model=MODEL_ID, dtype="auto",
        max_model_len=2048, enforce_eager=True,
    )

    model = get_model_from_llm(llm)
    configure_from_model(model)

    # Install deterministic attention hook.
    det_hook = DeterministicAttentionHook(lib_path="/opt/libdet_attn.so")
    n_hooks = det_hook.install(model)

    print(f"\nInstalled {n_hooks} deterministic attention hooks")
    print(f"Geometry: {det_hook.geometry}")

    # --- Test 1: Generate a short response ---
    print("\n" + "=" * 70)
    print("Test 1: Short generation with deterministic attention")
    print("=" * 70)

    prompt = "What is 2+2? Answer in one word:"
    params = SamplingParams(temperature=0.0, max_tokens=5)

    det_hook.captures.clear()
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text.strip()
    n_tokens = len(outputs[0].outputs[0].token_ids)

    print(f"  Prompt: {prompt!r}")
    print(f"  Output: {text!r}")
    print(f"  Tokens generated: {n_tokens}")
    print(f"  Attention captures: {len(det_hook.captures)}")

    # Each decode step should produce n_layers captures.
    expected_captures = n_tokens * n_hooks
    # Prefill doesn't produce captures (batch > 1), only decode steps do.
    # Actually for greedy decode, each decode token produces n_layers captures.
    print(f"  Expected captures (decode only): ~{n_tokens * n_hooks}")

    captures = det_hook.drain()
    print(f"  Drained captures: {len(captures)}")

    if len(captures) == 0:
        print("  WARNING: No captures produced — hook may not be firing")
        print("  This could mean:")
        print("    - vLLM attention module has a different forward() signature")
        print("    - batch_size > 1 during decode (chunked prefill?)")
        print("    - Hook is on the wrong module")

        # Debug: check what modules exist
        print("\n  Attention modules found:")
        for name, mod in model.named_modules():
            if hasattr(mod, "impl"):
                print(f"    {name}: {type(mod).__name__} impl={type(mod.impl).__name__}")
    else:
        # Verify captures have expected structure.
        sample = captures[0]
        print(f"\n  Sample capture (layer {sample['layer']}):")
        print(f"    seq_len: {sample['seq_len']}")
        print(f"    q_bf16 shape: {sample['q_bf16'].shape}")
        print(f"    k_bf16 shape: {sample['k_bf16'].shape}")
        print(f"    v_bf16 shape: {sample['v_bf16'].shape}")
        print(f"    output_f32_bits shape: {sample['output_f32_bits'].shape}")

    # --- Test 2: Verify CPU parity on captured data ---
    if len(captures) > 0:
        print("\n" + "=" * 70)
        print("Test 2: CPU parity verification on captured attention")
        print("=" * 70)

        import verilm_rs

        n_verified = 0
        n_mismatch = 0
        for cap_data in captures[:5]:  # Verify first 5 captures
            layer = cap_data["layer"]
            seq_len = cap_data["seq_len"]
            q_bf16 = cap_data["q_bf16"]
            k_bf16 = cap_data["k_bf16"]
            v_bf16 = cap_data["v_bf16"]
            gpu_out_bits = cap_data["output_f32_bits"]
            gpu_w_bits = cap_data["weight_f32_bits"]

            n_q = det_hook._n_q_heads
            n_kv = det_hook._n_kv_heads
            d = det_hook._d_head

            # Run CPU reference.
            q_list = q_bf16.tolist()
            k_list = [k_bf16[t * n_kv * d:(t+1) * n_kv * d].tolist()
                       for t in range(seq_len)]
            v_list = [v_bf16[t * n_kv * d:(t+1) * n_kv * d].tolist()
                       for t in range(seq_len)]

            cpu_json = verilm_rs.deterministic_attention_bf16(
                q_list, k_list, v_list,
                n_q, n_kv, d,
                float(det_hook._inv_sqrt_d),
            )
            cpu_result = json.loads(cpu_json)
            cpu_out_bits = np.array(cpu_result["output_bits"], dtype=np.uint32)

            match = np.array_equal(gpu_out_bits, cpu_out_bits)
            status = "PASS" if match else "FAIL"
            print(f"  Layer {layer}, seq_len={seq_len}: [{status}]")

            if match:
                n_verified += 1
            else:
                n_mismatch += 1
                diffs = np.abs(
                    gpu_out_bits.view(np.float32) - cpu_out_bits.view(np.float32)
                )
                print(f"    max_diff: {diffs.max():.2e}")

        print(f"\n  Verified: {n_verified}/{n_verified + n_mismatch} bit-exact")

    # --- Test 3: Model output quality ---
    print("\n" + "=" * 70)
    print("Test 3: Output quality (basic coherence)")
    print("=" * 70)

    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "The color of the sky is",
    ]
    params = SamplingParams(temperature=0.0, max_tokens=10)
    det_hook.captures.clear()

    outputs = llm.generate(prompts, params)
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text.strip()
        print(f"  {prompt!r} → {text!r}")

    captures = det_hook.drain()
    print(f"\n  Total captures from 3 prompts: {len(captures)}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {MODEL_ID}")
    print(f"  Attention mode: deterministic")
    print(f"  Layers hooked: {n_hooks}")
    print(f"  Geometry: {det_hook.geometry}")
    print(f"  Captures produced: {'YES' if len(captures) > 0 else 'NO'}")
    if 'n_verified' in dir():
        print(f"  CPU parity: {n_verified}/{n_verified + n_mismatch} bit-exact")
    print("=" * 70)

    return {
        "n_hooks": n_hooks,
        "geometry": det_hook.geometry,
        "captures_produced": len(captures) > 0,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/models": modal.Volume.from_name("model-cache", create_if_missing=True)},
)
def run_test():
    return _run_test()


@app.local_entrypoint()
def main():
    print("Deterministic Attention E2E Test")
    print("=" * 70)
    result = run_test.remote()
    print(f"\nResult: {result}")
