"""
Validate LP-hidden capture hook correctness.

Checks:
  1. count == generated tokens (after EOS trim)
  2. each capture is shape (1, hidden_dim), dtype bf16
  3. captured row matches LogitsProcessor input exactly (independent hook)
  4. bf16 lm_head from captured hidden reproduces greedy token identity
  5. sampled mode (temperature>0) has correct per-token alignment
  6. prefill rows are NOT accidentally retained

Usage:
    modal run --detach scripts/modal/diag_lp_hook.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-lp-hook")

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

PROMPTS = [
    "Explain the theory of relativity in one paragraph.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
]


def _run_diag():
    import torch
    import numpy as np

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)

    # Get lm_head for token identity checks
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    lm_head_weight = None
    for name, param in model.named_parameters():
        if "lm_head" in name and "weight" in name:
            lm_head_weight = param.detach().cpu().bfloat16()
            print(f"  lm_head: {name}, shape={param.shape}")
            break

    hidden_dim = lm_head_weight.shape[1]
    print(f"  hidden_dim={hidden_dim}")

    # Also install an independent reference hook on LogitsProcessor
    # to compare against what LPHiddenCapture sees.
    ref_captures = []
    logits_proc = None
    for name, mod in model.named_modules():
        if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
            logits_proc = mod
            break

    def ref_hook(module, args, output):
        if len(args) >= 2 and isinstance(args[1], torch.Tensor):
            # Immutable snapshot: clone on GPU, then sync D2H. Preserve native dtype (bf16).
            ref_captures.append(args[1].detach().contiguous().clone().cpu())

    h_ref = logits_proc.register_forward_hook(ref_hook)

    results = {"tests": []}

    def test(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}{': ' + detail if detail else ''}")
        results["tests"].append({"name": name, "passed": passed, "detail": detail})

    # ── Test 1-4: Greedy mode ──
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"GREEDY — Prompt {pi}: {prompt[:50]}...")
        print(f"{'='*60}")

        ref_captures.clear()

        result = server.chat(prompt=prompt, max_tokens=64, temperature=0.0)
        gen_tokens = result["token_ids"]
        n_prompt = result["commitment"].get("n_prompt_tokens", 0)
        n_gen = len(gen_tokens) - n_prompt
        gen_token_ids = gen_tokens[n_prompt:]

        # Read LP hidden stored by server.chat() after drain+trim.
        lp_raw = getattr(server, '_last_lp_hidden', None) or []

        print(f"  n_prompt={n_prompt}, n_gen={n_gen}, lp_captures={len(lp_raw)}, ref_captures={len(ref_captures)}")

        # Test 1: count == generated tokens
        # LP hook fires once per LogitsProcessor call.
        # For greedy with server.chat: prefill=1 LP call (pruned to last row),
        # then 1 per decode step = n_gen total LP calls.
        # But LP hook captures ALL calls including prefill.
        # The server trims trailing EOS, so lp_raw may have been trimmed too.
        # ref_captures is our independent count.
        test(f"p{pi}_greedy_ref_count",
             len(ref_captures) == n_gen,
             f"ref={len(ref_captures)}, n_gen={n_gen}")

        test(f"p{pi}_greedy_lp_count",
             len(lp_raw) == n_gen or len(lp_raw) == len(ref_captures),
             f"lp={len(lp_raw)}, n_gen={n_gen}, ref={len(ref_captures)}")

        # Test 2: shape and dtype
        if lp_raw:
            shapes_ok = all(t.shape[-1] == hidden_dim for t in lp_raw)
            dtypes_ok = all(t.dtype == torch.bfloat16 for t in lp_raw)
            test(f"p{pi}_greedy_shape",
                 shapes_ok,
                 f"shapes={[tuple(t.shape) for t in lp_raw[:3]]}...")
            test(f"p{pi}_greedy_dtype",
                 dtypes_ok,
                 f"dtypes={[str(t.dtype) for t in lp_raw[:3]]}...")

        # Test 3: captured row matches ref hook exactly
        if lp_raw and ref_captures:
            n_compare = min(len(lp_raw), len(ref_captures))
            max_diff = 0.0
            worst_i = -1
            for i in range(n_compare):
                lp_t = lp_raw[i]
                ref_t = ref_captures[i]
                lp_flat = lp_t.float().view(-1)[:hidden_dim]
                ref_flat = ref_t.float().view(-1)[:hidden_dim]
                diff = (lp_flat - ref_flat).abs().max().item()
                if diff > max_diff:
                    max_diff = diff
                    worst_i = i
            # Print detailed debug for first capture and worst capture
            for idx, label in [(0, "first"), (worst_i, "worst")]:
                if idx < 0 or idx >= n_compare:
                    continue
                lp_t = lp_raw[idx]
                ref_t = ref_captures[idx]
                print(f"  [{label} i={idx}] lp: shape={tuple(lp_t.shape)}, dtype={lp_t.dtype}, "
                      f"vals[:5]={lp_t.float().view(-1)[:5].tolist()}")
                print(f"  [{label} i={idx}] ref: shape={tuple(ref_t.shape)}, dtype={ref_t.dtype}, "
                      f"vals[:5]={ref_t.float().view(-1)[:5].tolist()}")
                d = (lp_t.float().view(-1)[:hidden_dim] - ref_t.float().view(-1)[:hidden_dim]).abs()
                print(f"  [{label} i={idx}] max_diff={d.max().item()}, "
                      f"nonzero={int((d > 0).sum())}/{hidden_dim}, "
                      f"argmax_diff={int(d.argmax())}")
            test(f"p{pi}_greedy_ref_match",
                 max_diff == 0.0,
                 f"max_diff={max_diff} at i={worst_i} across {n_compare} captures")

        # Test 4: bf16 lm_head from captured hidden reproduces token identity
        if lp_raw and gen_token_ids:
            exact = 0
            total = 0
            misses = []
            for i in range(min(len(lp_raw), len(gen_token_ids))):
                lp_t = lp_raw[i]
                hidden_bf16 = lp_t.bfloat16().view(-1)[:hidden_dim]
                logits = (lm_head_weight @ hidden_bf16).float()
                argmax = int(logits.argmax())
                actual = gen_token_ids[i]
                if argmax == actual:
                    exact += 1
                else:
                    gap = float(logits[argmax] - logits[actual])
                    misses.append(f"gen[{i}] tid={actual} argmax={argmax} gap={gap:.2f}")
                total += 1
            test(f"p{pi}_greedy_token_identity",
                 exact == total,
                 f"{exact}/{total} exact" + (f", misses: {misses[:3]}" if misses else ""))

        # Test 6: prefill rows NOT retained
        # LP hook should fire for prefill too, but server.chat should
        # have drained/trimmed correctly. Check that lp_raw count
        # matches gen tokens, not gen+prompt tokens.
        test(f"p{pi}_greedy_no_prefill_leak",
             len(lp_raw) <= n_gen + 1,  # +1 tolerance for EOS
             f"lp={len(lp_raw)}, n_gen={n_gen}, n_prompt={n_prompt}")

    # ── Test 5: Sampled mode — exact canonical replay ──
    for pi, prompt in enumerate(PROMPTS[:1]):  # Just one prompt for sampled
        print(f"\n{'='*60}")
        print(f"SAMPLED — Prompt {pi}: {prompt[:50]}...")
        print(f"{'='*60}")

        ref_captures.clear()

        temperature = 0.8
        top_k = 50
        top_p = 0.9
        result = server.chat(prompt=prompt, max_tokens=32,
                             temperature=temperature, top_k=top_k, top_p=top_p)
        gen_tokens = result["token_ids"]
        n_prompt = result["commitment"].get("n_prompt_tokens", 0)
        n_gen = len(gen_tokens) - n_prompt
        gen_token_ids = gen_tokens[n_prompt:]

        lp_raw = getattr(server, '_last_lp_hidden', None) or []
        batch_seed = getattr(server, '_last_seed', None)

        print(f"  n_prompt={n_prompt}, n_gen={n_gen}, lp_captures={len(lp_raw)}, ref_captures={len(ref_captures)}")
        print(f"  First 10 tokens: {gen_token_ids[:10]}")
        print(f"  batch_seed present: {batch_seed is not None}")

        test(f"sampled_count",
             len(lp_raw) == n_gen or len(lp_raw) == len(ref_captures),
             f"lp={len(lp_raw)}, n_gen={n_gen}, ref={len(ref_captures)}")

        # Exact canonical replay: recompute logits from LP hidden via bf16
        # lm_head, then run the same Rust canonical sampler with the committed
        # seed to verify the token choice is deterministic.
        if lp_raw and gen_token_ids and batch_seed is not None:
            import verilm_rs

            exact = 0
            total = 0
            misses = []
            for i in range(min(len(lp_raw), len(gen_token_ids))):
                lp_t = lp_raw[i]
                hidden_bf16 = lp_t.bfloat16().view(-1)[:hidden_dim]
                logits = (lm_head_weight @ hidden_bf16).float()
                logits_np = logits.numpy()

                # Derive per-token seed and replay canonical sampler.
                token_seed = verilm_rs.derive_token_seed(batch_seed, i)
                replayed = verilm_rs.canonical_sample(
                    logits_np, temperature, top_k, top_p, token_seed,
                )

                actual = gen_token_ids[i]
                if replayed == actual:
                    exact += 1
                else:
                    misses.append(f"gen[{i}] tid={actual} replayed={replayed}")
                total += 1
            test(f"sampled_exact_replay",
                 exact == total,
                 f"{exact}/{total} exact" + (f", misses: {misses[:5]}" if misses else ""))

    h_ref.remove()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    n_pass = sum(1 for t in results["tests"] if t["passed"])
    n_total = len(results["tests"])
    for t in results["tests"]:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  [{status}] {t['name']}: {t['detail']}")
    print(f"\n  {n_pass}/{n_total} passed")

    results["n_pass"] = n_pass
    results["n_total"] = n_total
    return results


@app.function(image=image, gpu="A100-80GB", timeout=1200)
def run_diag():
    return _run_diag()


@app.local_entrypoint()
def main():
    print("LP Hidden Hook Validation")
    print("=" * 60)
    result = run_diag.remote()
    print(f"\nResults: {result['n_pass']}/{result['n_total']} passed")
