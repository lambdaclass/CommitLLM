"""
Broad witnessed-score tolerance sweep for Qwen 7B W8A8.

Runs 40 diverse prompts across multiple families and context-length buckets
to validate whether the ±3 LSB tolerance holds across the tail distribution.

Families: prose, code, math, retrieval/factual, adversarial/repetitive
Buckets:  very-short (~16 tok), short (~64), medium (~256), long (~512)
Decode:   greedy (temperature=0) + sampled (temperature=0.8)

For each prompt, audits the LAST generated token (the only one with valid
witnessed scores) and measures per-layer max_diff.

Output: per-prompt max_diff, per-layer histogram, global max_diff, fraction
of layers at each diff bucket (0, 1, 2, 3, >3).

Usage:
    modal run --detach scripts/modal/bench_tolerance_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-tolerance-sweep")

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

# ── Prompt corpus: 40 prompts, 5 families × 4 context buckets × mixed decode ──
# (label, family, prompt, max_tokens, temperature)
SWEEP_PROMPTS = [
    # ── PROSE ──
    ("prose-vshort-g", "prose", "What is water?", 16, 0.0),
    ("prose-short-g", "prose", "Describe the process of photosynthesis in simple terms.", 64, 0.0),
    ("prose-med-g", "prose", "Write a short essay about the impact of the printing press on European society.", 256, 0.0),
    ("prose-long-g", "prose", "Write a comprehensive analysis of how climate change affects ocean ecosystems, covering coral bleaching, ocean acidification, sea level rise, and impacts on marine biodiversity.", 512, 0.0),
    ("prose-short-s", "prose", "Tell me a creative story about a robot who learns to paint.", 64, 0.8),
    ("prose-med-s", "prose", "Write a dialogue between a philosopher and a scientist debating free will.", 256, 0.8),
    ("prose-long-s", "prose", "Write a detailed travel guide for someone visiting Tokyo for the first time, covering neighborhoods, food, transport, etiquette, and hidden gems.", 512, 0.8),

    # ── CODE ──
    ("code-vshort-g", "code", "Write a Python one-liner to reverse a string.", 16, 0.0),
    ("code-short-g", "code", "Write a Python function that checks if a number is prime.", 64, 0.0),
    ("code-med-g", "code", "Implement a binary search tree in Python with insert, search, and delete methods. Include docstrings.", 256, 0.0),
    ("code-long-g", "code", "Write a complete implementation of a Redis-like in-memory key-value store in Python, supporting GET, SET, DEL, EXPIRE, KEYS, and TTL commands. Include a simple command parser.", 512, 0.0),
    ("code-short-s", "code", "Write a creative fizzbuzz implementation in an unusual programming style.", 64, 0.8),
    ("code-med-s", "code", "Write a Rust function that parses a simplified JSON format (strings, numbers, arrays, objects) without external crates.", 256, 0.8),
    ("code-long-s", "code", "Design and implement a simple HTTP router in Go that supports path parameters, middleware, and method-based routing. Include examples.", 512, 0.8),

    # ── MATH ──
    ("math-vshort-g", "math", "What is 17 × 23?", 16, 0.0),
    ("math-short-g", "math", "Prove that the square root of 2 is irrational.", 64, 0.0),
    ("math-med-g", "math", "Explain the fundamental theorem of calculus and provide three worked examples of increasing difficulty.", 256, 0.0),
    ("math-long-g", "math", "Derive the formula for the volume of a sphere using triple integrals in spherical coordinates. Then extend the derivation to n-dimensional hyperspheres, showing the recurrence relation for the volume formula.", 512, 0.0),
    ("math-short-s", "math", "Create a fun word problem involving probability and pirates.", 64, 0.8),
    ("math-med-s", "math", "Explain eigenvalues and eigenvectors with an intuitive geometric interpretation and three examples.", 256, 0.8),

    # ── RETRIEVAL / FACTUAL ──
    ("fact-vshort-g", "factual", "What year did the Berlin Wall fall?", 16, 0.0),
    ("fact-short-g", "factual", "List the planets of the solar system in order from the Sun, with one fact about each.", 64, 0.0),
    ("fact-med-g", "factual", "Explain the major differences between TCP and UDP, including use cases, header format, and reliability guarantees.", 256, 0.0),
    ("fact-long-g", "factual", "Write a detailed comparison of the x86-64 and ARM64 instruction set architectures, covering register files, memory models, SIMD extensions, virtualization support, and power efficiency trade-offs.", 512, 0.0),
    ("fact-short-s", "factual", "Describe three surprising facts about octopuses.", 64, 0.8),
    ("fact-med-s", "factual", "Explain the history of cryptography from Caesar ciphers to modern public-key systems.", 256, 0.8),

    # ── ADVERSARIAL / REPETITIVE ──
    ("rep-vshort-g", "adversarial", "Repeat the word 'hello' ten times.", 16, 0.0),
    ("rep-short-g", "adversarial", "Count from 1 to 50, one number per line.", 64, 0.0),
    ("rep-med-g", "adversarial", "Generate a multiplication table from 1×1 to 12×12, formatted as a grid.", 256, 0.0),
    ("rep-long-g", "adversarial", "List all prime numbers between 1 and 500, showing your work for each number you test.", 512, 0.0),
    ("rep-short-s", "adversarial", "Write a tongue twister about a purple porcupine.", 64, 0.8),
    ("rep-med-s", "adversarial", "Write the lyrics to an original song where every line starts with a successive letter of the alphabet.", 256, 0.8),

    # ── MIXED / EDGE CASES ──
    ("mix-vshort-g", "mixed", "Hi", 16, 0.0),
    ("mix-short-g", "mixed", "Translate 'The quick brown fox jumps over the lazy dog' into French, German, and Japanese.", 64, 0.0),
    ("mix-med-g", "mixed", "You are a medieval knight. Describe your typical day, including training, meals, and duties, staying in character throughout.", 256, 0.0),
    ("mix-long-g", "mixed", "Write a technical blog post explaining how transformers work, starting from self-attention, building up to multi-head attention, then covering positional encoding, feed-forward layers, and the full encoder-decoder architecture. Include mathematical notation.", 512, 0.0),
    ("mix-short-s", "mixed", "Invent a new word and write its dictionary entry.", 64, 0.8),
    ("mix-med-s", "mixed", "Write a recipe for a dish that doesn't exist yet, using unexpected ingredient combinations.", 256, 0.8),
    ("mix-long-s", "mixed", "Write a short science fiction story set in a world where gravity works in reverse above a certain altitude.", 512, 0.8),
]


def _run_sweep():
    import hashlib
    import json
    import time
    from collections import defaultdict

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=4096,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir
    full_layers = list(range(n_layers))

    buf = cap.get_capture_buffer()
    print(f"Score witness enabled: {buf._sw_enabled}")
    if not buf._sw_enabled:
        raise RuntimeError("Score witnessing not enabled!")

    seed = hashlib.sha256(b"tolerance-sweep-v1").digest()
    key_binary, artifact_binary = verilm_rs.generate_key_binary(model_dir, seed)
    key_json = verilm_rs.generate_key(model_dir, seed)
    print(f"Key ready: {len(key_binary)/1024/1024:.1f} MB")

    # ── Aggregation state ──
    global_max_diff = 0
    per_layer_max = defaultdict(int)       # layer_idx → max_diff seen
    diff_histogram = defaultdict(int)       # diff_value → count of (prompt, layer) pairs
    layer_fail_count = defaultdict(int)     # layer_idx → count of prompts where diff > 0
    prompt_results = []

    n_prompts = len(SWEEP_PROMPTS)
    for i, (label, family, prompt, max_tokens, temperature) in enumerate(SWEEP_PROMPTS):
        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{n_prompts}] {label} (family={family}, max_tok={max_tokens}, temp={temperature})")
        print(f"{'=' * 70}")

        try:
            result = server.chat(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            print(f"  CHAT ERROR: {e}")
            prompt_results.append({
                "label": label, "family": family, "error": str(e),
            })
            continue

        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt
        print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
        print(f"  text[:80]: {result['generated_text'][:80]}...")

        if n_gen < 1:
            print("  SKIP: no generated tokens")
            prompt_results.append({
                "label": label, "family": family, "n_prompt": n_prompt,
                "n_gen": 0, "error": "no gen tokens",
            })
            continue

        # Audit LAST generated token (only one with valid witnessed scores)
        last_token_idx = n_prompt + n_gen - 1
        print(f"  auditing token_index={last_token_idx} (last gen)")

        try:
            t0 = time.time()
            audit_binary = server.audit(
                request_id=request_id,
                token_index=last_token_idx,
                layer_indices=full_layers,
                tier="full",
                binary=True,
                use_captured_x_attn=True,
            )
            audit_ms = (time.time() - t0) * 1000
        except Exception as e:
            print(f"  AUDIT ERROR: {e}")
            prompt_results.append({
                "label": label, "family": family, "n_prompt": n_prompt,
                "n_gen": n_gen, "error": f"audit: {e}",
            })
            continue

        # Verify
        t0 = time.time()
        report = verilm_rs.verify_v4_full_binary(
            bytes(audit_binary), key_binary, artifact_binary
        )
        verify_ms = (time.time() - t0) * 1000

        print(f"  payload: {len(audit_binary)/1024:.0f} KB, verify: {verify_ms:.0f} ms")
        print(f"  passed: {report['passed']} ({report['checks_passed']}/{report['checks_run']})")

        # Corridor measurement for per-layer L-inf
        per_layer_linf = None
        global_linf = None
        try:
            corridor = json.loads(verilm_rs.measure_corridor_witnessed_scores(
                audit_binary, key_json, None,
            ))
            global_linf = corridor["global_linf"]
            per_layer_linf = corridor.get("per_layer_max_linf", {})
            print(f"  witnessed L-inf: {global_linf}")

            # Update aggregations
            if global_linf > global_max_diff:
                global_max_diff = global_linf

            for layer_str, linf in per_layer_linf.items():
                layer_int = int(layer_str)
                if linf > per_layer_max[layer_int]:
                    per_layer_max[layer_int] = linf
                diff_histogram[linf] += 1
                if linf > 0:
                    layer_fail_count[layer_int] += 1

            # Print layers with diff > 1
            hot_layers = {k: v for k, v in per_layer_linf.items() if v > 1}
            if hot_layers:
                print(f"  layers with diff>1: {hot_layers}")

        except Exception as e:
            print(f"  corridor error: {e}")

        # Failures detail
        if report.get("failures"):
            attn_fails = [f for f in report["failures"]
                         if "attention" in f.lower() or "anchor" in f.lower() or "score" in f.lower()]
            if attn_fails:
                print(f"  attn failures ({len(attn_fails)}):")
                for f in attn_fails[:3]:
                    print(f"    {f}")

        prompt_results.append({
            "label": label,
            "family": family,
            "temperature": temperature,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "token_index": last_token_idx,
            "payload_kb": len(audit_binary) / 1024,
            "verify_ms": verify_ms,
            "passed": report["passed"],
            "checks": f"{report['checks_passed']}/{report['checks_run']}",
            "failures": report.get("failures", []),
            "global_linf": global_linf,
            "per_layer_linf": per_layer_linf,
        })

    # ── Summary ──
    print(f"\n\n{'=' * 70}")
    print("TOLERANCE SWEEP SUMMARY")
    print(f"{'=' * 70}")

    n_run = sum(1 for p in prompt_results if "error" not in p)
    n_pass = sum(1 for p in prompt_results if p.get("passed"))
    n_fail = sum(1 for p in prompt_results if "passed" in p and not p["passed"])
    n_err = sum(1 for p in prompt_results if "error" in p)

    print(f"\nPrompts: {n_run} run, {n_pass} pass, {n_fail} fail, {n_err} errors")
    print(f"Global max_diff (witnessed L-inf): {global_max_diff}")

    # Diff histogram
    print(f"\nDiff histogram (across all prompt×layer pairs):")
    for d in sorted(diff_histogram.keys()):
        count = diff_histogram[d]
        pct = 100.0 * count / sum(diff_histogram.values()) if diff_histogram else 0
        bar = "#" * max(1, int(pct))
        print(f"  diff={d}: {count:5d} ({pct:5.1f}%) {bar}")

    # Fraction at each bucket
    total_pairs = sum(diff_histogram.values())
    if total_pairs > 0:
        exact = diff_histogram.get(0, 0)
        within_1 = exact + diff_histogram.get(1, 0)
        within_2 = within_1 + diff_histogram.get(2, 0)
        within_3 = within_2 + diff_histogram.get(3, 0)
        beyond_3 = total_pairs - within_3
        print(f"\n  Fraction exact (diff=0):  {exact}/{total_pairs} ({100*exact/total_pairs:.1f}%)")
        print(f"  Fraction ≤1:             {within_1}/{total_pairs} ({100*within_1/total_pairs:.1f}%)")
        print(f"  Fraction ≤2:             {within_2}/{total_pairs} ({100*within_2/total_pairs:.1f}%)")
        print(f"  Fraction ≤3:             {within_3}/{total_pairs} ({100*within_3/total_pairs:.1f}%)")
        print(f"  Fraction >3 (BREACHES):  {beyond_3}/{total_pairs} ({100*beyond_3/total_pairs:.1f}%)")

    # Per-layer breakdown
    print(f"\nPer-layer max_diff across all prompts:")
    for layer in sorted(per_layer_max.keys()):
        mx = per_layer_max[layer]
        fail_ct = layer_fail_count.get(layer, 0)
        marker = " <<<" if mx > 3 else ""
        print(f"  layer {layer:2d}: max_diff={mx}  non-exact in {fail_ct}/{n_run} prompts{marker}")

    # Hottest layers
    hot = [(l, m) for l, m in per_layer_max.items() if m >= 2]
    hot.sort(key=lambda x: -x[1])
    if hot:
        print(f"\nHottest layers (diff≥2):")
        for l, m in hot:
            print(f"  layer {l}: max_diff={m}, non-exact in {layer_fail_count[l]}/{n_run}")

    # Per-family summary
    print(f"\nPer-family results:")
    families = sorted(set(p.get("family", "?") for p in prompt_results))
    for fam in families:
        fam_results = [p for p in prompt_results if p.get("family") == fam and "error" not in p]
        if not fam_results:
            continue
        fam_pass = sum(1 for p in fam_results if p.get("passed"))
        fam_max = max((p.get("global_linf", 0) or 0) for p in fam_results)
        print(f"  {fam:12s}: {fam_pass}/{len(fam_results)} pass, max_linf={fam_max}")

    # Greedy vs sampled
    print(f"\nGreedy vs sampled:")
    for temp_label, temp_val in [("greedy (t=0)", 0.0), ("sampled (t=0.8)", 0.8)]:
        t_results = [p for p in prompt_results
                     if p.get("temperature") == temp_val and "error" not in p]
        if not t_results:
            continue
        t_pass = sum(1 for p in t_results if p.get("passed"))
        t_max = max((p.get("global_linf", 0) or 0) for p in t_results)
        print(f"  {temp_label:20s}: {t_pass}/{len(t_results)} pass, max_linf={t_max}")

    # Verdict
    print(f"\n{'=' * 70}")
    if global_max_diff <= 3:
        print(f"VERDICT: ±3 tolerance HOLDS (global max_diff={global_max_diff})")
    else:
        print(f"VERDICT: ±3 tolerance BREACHED (global max_diff={global_max_diff})")
        print(f"  Breaching layers: {[l for l, m in per_layer_max.items() if m > 3]}")
    print(f"{'=' * 70}")

    return {
        "global_max_diff": global_max_diff,
        "n_prompts": n_run,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "diff_histogram": dict(diff_histogram),
        "per_layer_max": dict(per_layer_max),
        "layer_fail_count": dict(layer_fail_count),
        "prompts": prompt_results,
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_sweep():
    return _run_sweep()


@app.local_entrypoint()
def main():
    print("CommitLLM Tolerance Sweep — Qwen 7B W8A8")
    print("40 prompts × 5 families × 4 context buckets × greedy+sampled")
    print("=" * 70)
    results = run_sweep.remote()

    import json
    print("\n\nFull results JSON:")
    print(json.dumps(results, indent=2, default=str))
