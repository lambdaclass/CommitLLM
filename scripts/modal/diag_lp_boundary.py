"""
Decode boundary diagnostic: LogitsProcessor input as the real capture point.

Previous diagnostics showed model.norm output[1] != LogitsProcessor input
(diffs 38-318). This script hooks LogitsProcessor directly to:

1. Inspect its argument structure (don't assume args[1])
2. Capture the exact hidden_states it receives
3. Capture the exact logits it produces
4. Verify: hidden_at_lp @ lm_head == LP output logits
5. Verify: argmax(LP output logits) == actual generated token

If step 4+5 pass, LP input is the real decode boundary.
If step 4 fails, there's a transform inside LP we can't replicate.
If step 4 passes but step 5 fails, the sampler is doing something unexpected.

Usage:
    modal run --detach scripts/modal/diag_lp_boundary.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-lp-boundary")

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
        ".git", "target", "scripts/__pycache__", "*.pdf", "site",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)

PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
]
MAX_TOKENS = 64


def _run_diag():
    import inspect
    import numpy as np
    import torch

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # ── Step 1: Find LogitsProcessor and inspect its signature ──
    print(f"\n{'='*60}")
    print("STEP 1: Find LogitsProcessor and inspect arguments")
    print(f"{'='*60}")

    logits_proc = None
    for name, mod in model.named_modules():
        if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
            logits_proc = mod
            print(f"  Found: {type(mod).__name__} at '{name}'")
            try:
                sig = inspect.signature(type(mod).forward)
                print(f"  forward signature: {sig}")
            except Exception as e:
                print(f"  signature error: {e}")
            try:
                src = inspect.getsource(type(mod).forward)
                print(f"  forward source:\n{src[:3000]}")
            except Exception as e:
                print(f"  source error: {e}")
            break

    if logits_proc is None:
        print("ERROR: LogitsProcessor not found")
        return {"error": "no LogitsProcessor"}

    # Also find lm_head weight for local matmul verification
    lm_head_weight = None
    for name, param in model.named_parameters():
        if "lm_head" in name and "weight" in name:
            lm_head_weight = param
            print(f"\n  lm_head: {name}, shape={param.shape}, dtype={param.dtype}")
            break

    # ── Step 2: Probe LP arguments with a short generation ──
    print(f"\n{'='*60}")
    print("STEP 2: Probe LogitsProcessor argument structure")
    print(f"{'='*60}")

    probe_data = []

    def probe_hook(module, args, kwargs, output):
        info = {
            "n_args": len(args),
            "n_kwargs": len(kwargs) if kwargs else 0,
            "kwargs_keys": list(kwargs.keys()) if kwargs else [],
            "output_type": type(output).__name__,
        }
        for i, a in enumerate(args):
            info[f"arg{i}_type"] = type(a).__name__
            if isinstance(a, torch.Tensor):
                info[f"arg{i}_shape"] = tuple(a.shape)
                info[f"arg{i}_dtype"] = str(a.dtype)
            elif hasattr(a, '__len__'):
                info[f"arg{i}_len"] = len(a)

        if isinstance(output, torch.Tensor):
            info["out_shape"] = tuple(output.shape)
            info["out_dtype"] = str(output.dtype)
        elif isinstance(output, tuple):
            info["out_len"] = len(output)
            for i, o in enumerate(output[:3]):
                if isinstance(o, torch.Tensor):
                    info[f"out{i}_shape"] = tuple(o.shape)
                    info[f"out{i}_dtype"] = str(o.dtype)

        probe_data.append(info)

    h_probe = logits_proc.register_forward_hook(probe_hook, with_kwargs=True)

    probe_out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=4, temperature=0.0))[0]
    print(f"  Probe gen: {list(probe_out.outputs[0].token_ids)}")

    h_probe.remove()

    print(f"\n  LP called {len(probe_data)} times:")
    for i, d in enumerate(probe_data):
        print(f"    call {i}: {d}")

    # Identify the hidden-state argument by shape
    # It should be (N, hidden_dim) where hidden_dim matches lm_head shape[1]
    hidden_dim = lm_head_weight.shape[1] if lm_head_weight is not None else 3584
    vocab_size = lm_head_weight.shape[0] if lm_head_weight is not None else 152064

    hidden_arg_idx = None
    for i in range(probe_data[0]["n_args"]):
        key = f"arg{i}_shape"
        if key in probe_data[0]:
            shape = probe_data[0][key]
            if len(shape) == 2 and shape[1] == hidden_dim:
                hidden_arg_idx = i
                print(f"\n  Hidden-state argument: args[{i}] (shape={shape}, hidden_dim={hidden_dim})")
                break
            elif len(shape) == 1 and shape[0] == hidden_dim:
                hidden_arg_idx = i
                print(f"\n  Hidden-state argument: args[{i}] (shape={shape}, 1D hidden_dim={hidden_dim})")
                break

    if hidden_arg_idx is None:
        # Check if hidden_states is passed as kwarg
        print("\n  WARNING: Could not identify hidden-state arg by shape. Dumping all args:")
        for i in range(probe_data[0]["n_args"]):
            print(f"    arg{i}: type={probe_data[0].get(f'arg{i}_type')}, "
                  f"shape={probe_data[0].get(f'arg{i}_shape')}, "
                  f"dtype={probe_data[0].get(f'arg{i}_dtype')}")

    # ── Step 3: Capture LP hidden input + logit output for real generation ──
    print(f"\n{'='*60}")
    print("STEP 3: Capture LP hidden input + logit output")
    print(f"{'='*60}")

    lp_captures = []

    def capture_hook(module, args, kwargs, output):
        entry = {"fwd_idx": len(lp_captures)}

        # Capture all tensor args for safety
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                entry[f"arg{i}"] = a.detach().cpu().float()
                entry[f"arg{i}_shape"] = tuple(a.shape)
                entry[f"arg{i}_dtype"] = str(a.dtype)

        # Capture output logits
        if isinstance(output, torch.Tensor):
            entry["logits"] = output.detach().cpu().float()
            entry["logits_shape"] = tuple(output.shape)
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            entry["logits"] = output[0].detach().cpu().float()
            entry["logits_shape"] = tuple(output[0].shape)

        lp_captures.append(entry)

    h_cap = logits_proc.register_forward_hook(capture_hook, with_kwargs=True)

    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {pi}: {prompt[:50]}... ---")
        lp_captures.clear()

        result = llm.generate([prompt], SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0))[0]
        gen_tokens = list(result.outputs[0].token_ids)
        n_gen = len(gen_tokens)

        print(f"  Generated {n_gen} tokens, LP called {len(lp_captures)} times")
        print(f"  First 10 tokens: {gen_tokens[:10]}")

        if not lp_captures:
            print("  ERROR: no LP captures")
            continue

        # ── Step 4: Verify local identity: hidden @ lm_head.T == LP logits ──
        # For prefill (call 0): LP gets pruned hidden (last row only for generation)
        # For decode (calls 1+): LP gets 1 row

        lm_head_cpu = lm_head_weight.detach().cpu()

        for ci, cap_entry in enumerate(lp_captures):
            if "logits" not in cap_entry:
                print(f"  LP call {ci}: no logits captured")
                continue

            logits_actual = cap_entry["logits"]  # (N, vocab_size) or (1, vocab_size)

            # Find hidden state tensor
            hidden = None
            hidden_key = None
            if hidden_arg_idx is not None and f"arg{hidden_arg_idx}" in cap_entry:
                hidden = cap_entry[f"arg{hidden_arg_idx}"]
                hidden_key = f"arg{hidden_arg_idx}"
            else:
                # Fallback: find any arg with hidden_dim in last dimension
                for k, v in cap_entry.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[1] == hidden_dim:
                        hidden = v
                        hidden_key = k
                        break

            if hidden is None:
                print(f"  LP call {ci}: could not find hidden-state tensor")
                continue

            # Number of rows in logits tells us how many tokens LP is producing
            n_rows = logits_actual.shape[0]

            # Recompute logits: hidden @ lm_head.T
            # Use same dtype path as GPU: bf16
            hidden_bf16 = hidden.bfloat16()
            lm_head_bf16 = lm_head_cpu.bfloat16()
            logits_recomputed_bf16 = (hidden_bf16 @ lm_head_bf16.T).float()

            # Also fp32 path
            logits_recomputed_fp32 = (hidden.float() @ lm_head_cpu.float().T)

            # Compare recomputed vs actual
            diff_bf16 = (logits_recomputed_bf16 - logits_actual).abs()
            diff_fp32 = (logits_recomputed_fp32 - logits_actual).abs()

            max_diff_bf16 = diff_bf16.max().item()
            mean_diff_bf16 = diff_bf16.mean().item()
            max_diff_fp32 = diff_fp32.max().item()
            mean_diff_fp32 = diff_fp32.mean().item()

            # For first few calls, print detailed info
            if ci < 3 or ci == len(lp_captures) - 1:
                print(f"\n  LP call {ci} ({hidden_key}):")
                print(f"    hidden: {tuple(hidden.shape)}, logits: {tuple(logits_actual.shape)}")
                print(f"    recomputed vs actual (bf16 matmul): max_diff={max_diff_bf16:.4f}, mean={mean_diff_bf16:.6f}")
                print(f"    recomputed vs actual (fp32 matmul): max_diff={max_diff_fp32:.4f}, mean={mean_diff_fp32:.6f}")

            # ── Step 5: Check token identity ──
            # For prefill: only last row matters (predicts first gen token)
            # For decode: single row predicts next token
            if ci == 0:
                # Prefill: last row predicts gen_tokens[0]
                token_idx = 0
                logit_row = logits_actual[-1]
            else:
                # Decode call ci predicts gen_tokens[ci] (call 1 -> gen[1], etc.)
                token_idx = ci
                logit_row = logits_actual[-1]  # Should be (1, vocab) -> take last row

            if token_idx < n_gen:
                actual_token = gen_tokens[token_idx]
                argmax_actual = int(logit_row.argmax())
                is_exact = (argmax_actual == actual_token)

                # Also check recomputed
                recomp_row_bf16 = logits_recomputed_bf16[-1] if ci == 0 else logits_recomputed_bf16[-1]
                recomp_row_fp32 = logits_recomputed_fp32[-1] if ci == 0 else logits_recomputed_fp32[-1]
                argmax_recomp_bf16 = int(recomp_row_bf16.argmax())
                argmax_recomp_fp32 = int(recomp_row_fp32.argmax())

                result_entry = {
                    "prompt": pi, "lp_call": ci, "token_idx": token_idx,
                    "actual_token": actual_token,
                    "argmax_lp_logits": argmax_actual,
                    "argmax_recomp_bf16": argmax_recomp_bf16,
                    "argmax_recomp_fp32": argmax_recomp_fp32,
                    "exact_lp": is_exact,
                    "exact_recomp_bf16": argmax_recomp_bf16 == actual_token,
                    "exact_recomp_fp32": argmax_recomp_fp32 == actual_token,
                    "matmul_diff_bf16": max_diff_bf16,
                    "matmul_diff_fp32": max_diff_fp32,
                }
                all_results.append(result_entry)

                if not is_exact or argmax_recomp_bf16 != actual_token:
                    gap_lp = float(logit_row[argmax_actual] - logit_row[actual_token])
                    rank_lp = int((logit_row > logit_row[actual_token]).sum())
                    print(f"    gen[{token_idx}] tid={actual_token}: "
                          f"LP_logits={'OK' if is_exact else f'MISS(argmax={argmax_actual},rank={rank_lp},gap={gap_lp:.2f})'} "
                          f"recomp_bf16={'OK' if argmax_recomp_bf16 == actual_token else f'MISS(argmax={argmax_recomp_bf16})'} "
                          f"recomp_fp32={'OK' if argmax_recomp_fp32 == actual_token else f'MISS(argmax={argmax_recomp_fp32})'}")

    h_cap.remove()

    # ── Summary ──
    total = len(all_results)
    exact_lp = sum(1 for r in all_results if r["exact_lp"])
    exact_bf16 = sum(1 for r in all_results if r["exact_recomp_bf16"])
    exact_fp32 = sum(1 for r in all_results if r["exact_recomp_fp32"])

    avg_diff_bf16 = np.mean([r["matmul_diff_bf16"] for r in all_results]) if all_results else 0
    avg_diff_fp32 = np.mean([r["matmul_diff_fp32"] for r in all_results]) if all_results else 0
    max_diff_bf16_all = max(r["matmul_diff_bf16"] for r in all_results) if all_results else 0
    max_diff_fp32_all = max(r["matmul_diff_fp32"] for r in all_results) if all_results else 0

    print(f"\n{'='*60}")
    print(f"SUMMARY ({total} generated tokens across {len(PROMPTS)} prompts)")
    print(f"{'='*60}")
    print(f"\n  Token identity:")
    print(f"    LP output logits argmax == token:     {exact_lp}/{total} ({100*exact_lp/max(total,1):.1f}%)")
    print(f"    Recomputed bf16 matmul argmax == token: {exact_bf16}/{total} ({100*exact_bf16/max(total,1):.1f}%)")
    print(f"    Recomputed fp32 matmul argmax == token: {exact_fp32}/{total} ({100*exact_fp32/max(total,1):.1f}%)")

    print(f"\n  Matmul fidelity (recomputed vs LP output):")
    print(f"    bf16: avg_max_diff={avg_diff_bf16:.4f}, overall_max_diff={max_diff_bf16_all:.4f}")
    print(f"    fp32: avg_max_diff={avg_diff_fp32:.4f}, overall_max_diff={max_diff_fp32_all:.4f}")

    print(f"\n  DECISION:")
    if exact_lp == total:
        print(f"    LP output logits give EXACT token identity for all {total} tokens.")
        if exact_bf16 == total:
            print(f"    Recomputed bf16 matmul ALSO exact — LP input hidden is the real decode boundary.")
            print(f"    → Option C viable at LP input (not model.norm).")
        elif exact_fp32 == total:
            print(f"    Recomputed fp32 exact but bf16 not — bf16 matmul loses precision.")
            print(f"    → Need fp32 logits or deterministic lm_head.")
        else:
            print(f"    Recomputed matmul does NOT match — LP has an internal transform.")
            print(f"    → Must capture LP OUTPUT logits (Option A), not LP input hidden.")
    else:
        n_miss = total - exact_lp
        print(f"    LP output logits miss {n_miss}/{total} tokens.")
        print(f"    → Something between LP output and sampler changes token selection.")
        print(f"    → Investigate sampler/logit-processing pipeline.")

    return {
        "total": total,
        "exact_lp": exact_lp,
        "exact_recomp_bf16": exact_bf16,
        "exact_recomp_fp32": exact_fp32,
        "max_diff_bf16": float(max_diff_bf16_all),
        "max_diff_fp32": float(max_diff_fp32_all),
    }


@app.function(image=image, gpu="A100-80GB", timeout=1200)
def run_diag():
    return _run_diag()


@app.local_entrypoint()
def main():
    print("Decode Boundary Diagnostic: LogitsProcessor input")
    print("=" * 60)
    result = run_diag.remote()
    print(f"\nResults: {result}")
