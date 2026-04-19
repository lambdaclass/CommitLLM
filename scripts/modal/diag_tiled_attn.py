"""
Diagnostic: tiled online-softmax attention replay vs global softmax.

Measures L-inf gap for three replay strategies on real GPU data:
  1. Global f64 softmax (current baseline)
  2. Tiled online-softmax, block_n=64 (FA2 standard decode)
  3. Tiled online-softmax, block_n=128 (FA2 split-KV decode on A100)

Both Llama and Qwen, multiple token positions, sampled decode.
This is a time-boxed measurement spike — if tiled does not close the gap
(or makes it worse), we stop and escalate.

Usage:
    modal run --detach scripts/modal/diag_tiled_attn.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-tiled-attn")

MODELS = {
    "qwen": "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",
    "llama": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
}

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
        "VERILM_SCORE_WITNESS": "1",
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
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Rust?",
    "Describe how a compiler works in simple terms.",
    "Tell me a fun fact about space.",
]

N_ITERS = 5
MAX_TOKENS = 32


def _run_model(model_name, model_id):
    import hashlib
    import json
    import random

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"\nLoading {model_id}...")
    llm = LLM(model=model_id, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))

    # Generate verifier key (JSON for corridor measurement).
    print(f"Generating verifier key...")
    seed = hashlib.sha256(f"diag-tiled-attn-{model_name}".encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, list(seed))

    # Warmup.
    print("Warmup...")
    for _ in range(2):
        server.chat(prompt="Hello", max_tokens=8, temperature=0.8, top_k=50, top_p=0.9)

    rng = random.Random(42)
    all_results = []

    for i in range(N_ITERS):
        prompt = PROMPTS[i % len(PROMPTS)]
        print(f"\n{'='*60}")
        print(f"[{model_name}] iter {i}: {prompt[:50]}...")

        result = server.chat(
            prompt=prompt, max_tokens=MAX_TOKENS,
            temperature=0.8, top_k=50, top_p=0.9,
        )
        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        n_prompt = result["commitment"].get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt
        print(f"  n_prompt={n_prompt}, n_gen={n_gen}")

        if n_gen <= 0:
            print("  skip: no generated tokens")
            continue

        # Challenge a random generated token.
        gen_start = n_prompt - 1
        gen_offset = rng.randint(0, n_gen - 1)
        challenge_idx = gen_start + gen_offset

        # Get audit binary with full opening.
        audit_binary = server.audit(
            request_id=request_id,
            token_index=challenge_idx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        print(f"  challenge: gen_offset={gen_offset}, token_index={challenge_idx}, "
              f"payload={len(audit_binary)} bytes")

        # --- Measure all three replay strategies ---

        # 1. Global softmax (current baseline).
        report_global = json.loads(verilm_rs.measure_corridor_witnessed_scores(
            bytes(audit_binary), key_json,
        ))

        # 2. Tiled block_n=64 (FA2 standard decode).
        report_tiled_64 = json.loads(
            verilm_rs.measure_corridor_witnessed_scores_tiled(
                bytes(audit_binary), key_json, 64,
            )
        )

        # 3. Tiled block_n=128 (FA2 split-KV decode on A100).
        report_tiled_128 = json.loads(
            verilm_rs.measure_corridor_witnessed_scores_tiled(
                bytes(audit_binary), key_json, 128,
            )
        )

        # Extract L-inf stats.
        def extract(report):
            per_layer = report.get("per_layer_max_linf", [])
            measurements = report.get("measurements", [])
            linf_vals = [m["linf"] for m in measurements] if measurements else []
            return {
                "linf_max": report.get("global_linf", -1),
                "linf_mean": sum(linf_vals) / len(linf_vals) if linf_vals else -1,
                "per_layer_max": per_layer,
                "n_layers": len(per_layer),
            }

        g = extract(report_global)
        t64 = extract(report_tiled_64)
        t128 = extract(report_tiled_128)

        row = {
            "iter": i,
            "model": model_name,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "challenge_idx": challenge_idx,
            "global": g,
            "tiled_64": t64,
            "tiled_128": t128,
        }
        all_results.append(row)

        print(f"  global:     linf_max={g['linf_max']:>3}  mean={g['linf_mean']:.1f}  layers={g['n_layers']}")
        print(f"  tiled_64:   linf_max={t64['linf_max']:>3}  mean={t64['linf_mean']:.1f}  layers={t64['n_layers']}")
        print(f"  tiled_128:  linf_max={t128['linf_max']:>3}  mean={t128['linf_mean']:.1f}  layers={t128['n_layers']}")

        delta_64 = t64["linf_max"] - g["linf_max"]
        delta_128 = t128["linf_max"] - g["linf_max"]
        print(f"  delta vs global: tiled_64={delta_64:+d}, tiled_128={delta_128:+d}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"[{model_name}] Tiled Attention Diagnostic Summary")
    print(f"{'='*60}")
    print(f"  {'iter':>4} {'n_gen':>5} {'global':>8} {'tiled64':>8} {'tiled128':>9} {'d64':>5} {'d128':>5}")
    print(f"  {'-'*4} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*5} {'-'*5}")
    for r in all_results:
        g = r["global"]["linf_max"]
        t64 = r["tiled_64"]["linf_max"]
        t128 = r["tiled_128"]["linf_max"]
        print(f"  {r['iter']:>4} {r['n_gen']:>5} {g:>8} {t64:>8} {t128:>9} "
              f"{t64-g:>+5} {t128-g:>+5}")

    if all_results:
        g_vals = [r["global"]["linf_max"] for r in all_results]
        t64_vals = [r["tiled_64"]["linf_max"] for r in all_results]
        t128_vals = [r["tiled_128"]["linf_max"] for r in all_results]
        print(f"\n  Global  linf_max: min={min(g_vals)}, max={max(g_vals)}, "
              f"mean={sum(g_vals)/len(g_vals):.1f}")
        print(f"  Tiled64  linf_max: min={min(t64_vals)}, max={max(t64_vals)}, "
              f"mean={sum(t64_vals)/len(t64_vals):.1f}")
        print(f"  Tiled128 linf_max: min={min(t128_vals)}, max={max(t128_vals)}, "
              f"mean={sum(t128_vals)/len(t128_vals):.1f}")

    print(f"{'='*60}")

    return {
        "model": model_name,
        "results": all_results,
    }


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_qwen():
    return _run_model("qwen", MODELS["qwen"])


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_llama():
    return _run_model("llama", MODELS["llama"])


@app.local_entrypoint()
def main():
    print("Tiled Attention Diagnostic: global vs tiled online-softmax")
    print("=" * 60)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Iterations: {N_ITERS}")
    print(f"Block sizes: 64 (standard FA2), 128 (split-KV FA2)")
    print()

    qwen_future = run_qwen.spawn()
    llama_future = run_llama.spawn()

    qwen_result = qwen_future.get()
    llama_result = llama_future.get()

    print("\n" + "=" * 60)
    print("COMBINED RESULTS")
    print("=" * 60)

    for result in [qwen_result, llama_result]:
        model = result["model"]
        rs = result["results"]
        if not rs:
            print(f"\n{model}: no results")
            continue
        g_vals = [r["global"]["linf_max"] for r in rs]
        t64_vals = [r["tiled_64"]["linf_max"] for r in rs]
        t128_vals = [r["tiled_128"]["linf_max"] for r in rs]
        print(f"\n{model}:")
        print(f"  Global   linf_max: {min(g_vals)}-{max(g_vals)} (mean {sum(g_vals)/len(g_vals):.1f})")
        print(f"  Tiled64  linf_max: {min(t64_vals)}-{max(t64_vals)} (mean {sum(t64_vals)/len(t64_vals):.1f})")
        print(f"  Tiled128 linf_max: {min(t128_vals)}-{max(t128_vals)} (mean {sum(t128_vals)/len(t128_vals):.1f})")

        deltas_64 = [t - g for t, g in zip(t64_vals, g_vals)]
        deltas_128 = [t - g for t, g in zip(t128_vals, g_vals)]
        print(f"  Delta64  range: {min(deltas_64):+d} to {max(deltas_64):+d}")
        print(f"  Delta128 range: {min(deltas_128):+d} to {max(deltas_128):+d}")

    print()
    print("If tiled L-inf < global L-inf: tiling matches kernel better → proceed")
    print("If tiled L-inf ≈ global L-inf: tiling is not the gap source → stop, escalate")
    print("If tiled L-inf > global L-inf: tiling hurts → stop immediately")
