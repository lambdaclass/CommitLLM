"""
Diagnostic test for capture-count mismatch on specific prompts.

Runs the two previously-crashing prompts with detailed logging to
identify exactly why cutlass_scaled_mm call counts don't match
the expected 4*n_layers pattern.

Usage:
    modal run --detach scripts/modal_debug_capture_mismatch.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-debug-capture")

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
    .pip_install(*VERIFICATION)
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

FAILING_CASES = [
    {"prompt": "Write a haiku about CUDA synchronization.", "max_tokens": 16, "label": "haiku"},
    {"prompt": "A" * 200, "max_tokens": 4, "label": "AAA"},
]

PASSING_CASES = [
    {"prompt": "What is 2+2?", "max_tokens": 8, "label": "2+2"},
    {"prompt": "Hi", "max_tokens": 128, "label": "Hi"},
]


@app.function(image=image, gpu="A100-80GB", timeout=600)
def run_debug():
    import logging
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Enable detailed logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("verilm")
    logger.setLevel(logging.DEBUG)

    from vllm import LLM, SamplingParams

    from verilm import capture as cap
    from verilm.capture import get_capture_buffer
    from verilm.server import VerifiedInferenceServer

    buf = get_capture_buffer()

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048, enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    calls_per_fwd = n_layers * cap.PROJS_PER_LAYER

    print(f"\nn_layers={n_layers}, calls_per_fwd={calls_per_fwd}")

    # Warmup
    cap._capture_mode = "minimal"
    buf.set_sync_mode("global")
    buf.enabled = True
    for _ in range(3):
        buf.drain()
        buf.reset_counter()
        params = SamplingParams(max_tokens=4, temperature=0.0)
        llm.generate(["Hello"], params)
        buf.wait_for_transfers()
        warmup_caps = buf.drain()
        print(f"Warmup: captures={len(warmup_caps)}, counter={cap._call_counter}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Raw capture counts per request")
    print("=" * 60)

    all_cases = PASSING_CASES + FAILING_CASES

    for case in all_cases:
        prompt = case["prompt"]
        max_tokens = case["max_tokens"]
        label = case["label"]

        print(f"\n--- {label}: prompt={prompt[:40]!r}{'...' if len(prompt)>40 else ''}, max_tokens={max_tokens} ---")

        # Reset and run raw inference (no server.chat, to isolate capture counting)
        buf.drain()
        buf.reset_counter()
        cap._capture_mode = "minimal"
        buf.enabled = True

        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        outputs = llm.generate([prompt], params)
        output = outputs[0]

        buf.wait_for_transfers()
        captures = buf.drain()

        prompt_token_ids = list(output.prompt_token_ids)
        gen_token_ids = list(output.outputs[0].token_ids)
        all_tokens = len(prompt_token_ids) + len(gen_token_ids)
        expected_traces = all_tokens - 1

        print(f"  prompt_tokens={len(prompt_token_ids)}, gen_tokens={len(gen_token_ids)}, "
              f"all_tokens={all_tokens}")
        print(f"  captures={len(captures)}, calls_per_fwd={calls_per_fwd}")
        print(f"  n_fwd={len(captures) / calls_per_fwd if calls_per_fwd > 0 else '?'}")
        print(f"  call_counter={cap._call_counter}")
        print(f"  captures % calls_per_fwd = {len(captures) % calls_per_fwd}")

        if len(captures) % calls_per_fwd == 0:
            n_fwd = len(captures) // calls_per_fwd
            # In minimal mode, o_proj is at index 1 in each layer's 4-call group
            batch_sizes = []
            for i in range(n_fwd):
                entry = captures[i * calls_per_fwd + 1]  # o_proj
                if entry[2] is not None:
                    batch_sizes.append(entry[2].shape[0])
                else:
                    batch_sizes.append("None")
            n_tokens = sum(b for b in batch_sizes if isinstance(b, int))
            print(f"  fwd_batch_sizes={batch_sizes}")
            print(f"  n_tokens={n_tokens}, expected_traces={expected_traces}")
            if n_tokens == expected_traces:
                print(f"  RESULT: OK")
            else:
                print(f"  RESULT: TOKEN MISMATCH ({n_tokens} != {expected_traces})")
        else:
            print(f"  RESULT: CAPTURE COUNT MISMATCH")

            # Dump layer/proj of every capture for debugging
            print(f"  First 10 captures (layer, proj):")
            for i, c in enumerate(captures[:10]):
                print(f"    [{i}] layer={c[0]}, proj={c[1]}, "
                      f"tensor={'shape='+str(c[2].shape) if c[2] is not None else 'None'}")
            if len(captures) > 10:
                print(f"  Last 5 captures:")
                for i, c in enumerate(captures[-5:]):
                    print(f"    [{len(captures)-5+i}] layer={c[0]}, proj={c[1]}, "
                          f"tensor={'shape='+str(c[2].shape) if c[2] is not None else 'None'}")

        # Also try through server.chat to see if it matches
        print(f"\n  Trying through server.chat()...")
        try:
            result = server.chat(prompt=prompt, max_tokens=max_tokens)
            print(f"  server.chat(): OK — n_tokens={result['n_tokens']}, "
                  f"root={result['commitment']['merkle_root'][:16]}...")
        except RuntimeError as e:
            print(f"  server.chat(): FAILED — {e}")

    return "done"


@app.local_entrypoint()
def main():
    print("VeriLM Capture Mismatch Diagnostic")
    print("=" * 60)
    run_debug.remote()
    print("Done.")
