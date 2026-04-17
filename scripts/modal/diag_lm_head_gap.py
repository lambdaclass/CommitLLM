"""
Diagnostic: measure lm_head logit gap distribution across all generated tokens.

For each generated token, computes:
  gap = max(logits_i32) - logits_i32[actual_token_id]

This gives the empirical distribution needed to calibrate lm_head_logit_tolerance.

Usage:
    modal run --detach scripts/modal/diag_lm_head_gap.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-lm-head-gap")

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
    "What is the significance of the Turing test?",
    "How does public key cryptography work?",
]
MAX_TOKENS = 64


def _run_diag():
    import hashlib
    import json

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import verilm_rs
    from vllm import LLM
    from verilm import capture as cap
    from verilm.server import VerifiedInferenceServer

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="auto", max_model_len=2048,
              enforce_eager=True, enable_prefix_caching=False)
    server = VerifiedInferenceServer(llm)
    n_layers = cap._n_layers
    model_dir = server._model_dir

    seed = hashlib.sha256(PROMPTS[0].encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)
    full_layers = list(range(n_layers))

    all_gaps = []

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"Prompt {pi}: {prompt[:50]}...")

        cap._capture_mode = "minimal"
        result = server.chat(prompt=prompt, max_tokens=MAX_TOKENS)
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        gen_start = n_prompt - 1 if n_prompt > 0 else 0

        print(f"  n_tokens={n_tokens}, n_prompt={n_prompt}, gen_start={gen_start}")

        # Audit every generated token position
        for token_idx in range(gen_start, n_tokens):
            # Need fresh chat for each audit after the first
            if token_idx > gen_start:
                result = server.chat(prompt=prompt, max_tokens=MAX_TOKENS)
                rid = result["request_id"]
            else:
                rid = result["request_id"]

            audit_json = server.audit(
                request_id=rid,
                token_index=token_idx,
                layer_indices=full_layers,
                tier="full",
                binary=False,
            )
            audit = json.loads(audit_json)
            shell = audit.get("shell_opening", {})
            logits_i32 = shell.get("logits_i32")
            token_id = audit.get("token_id")

            if logits_i32 and token_id is not None and token_id < len(logits_i32):
                max_logit = max(logits_i32)
                token_logit = logits_i32[token_id]
                gap = max_logit - token_logit
                argmax = logits_i32.index(max_logit)
                all_gaps.append(gap)

                status = "EXACT" if gap == 0 else f"gap={gap}"
                if gap > 100:
                    print(f"  token_idx={token_idx} tid={token_id} argmax={argmax} {status} max_logit={max_logit}")
            else:
                print(f"  token_idx={token_idx} NO LOGITS")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"GAP DISTRIBUTION ({len(all_gaps)} generated tokens)")
    print(f"{'='*60}")
    if all_gaps:
        all_gaps.sort()
        exact = sum(1 for g in all_gaps if g == 0)
        print(f"  exact (gap=0): {exact}/{len(all_gaps)} ({100*exact/len(all_gaps):.1f}%)")
        for threshold in [50, 100, 200, 500, 750, 1000, 1500, 2000]:
            within = sum(1 for g in all_gaps if g <= threshold)
            print(f"  gap<={threshold}: {within}/{len(all_gaps)} ({100*within/len(all_gaps):.1f}%)")
        print(f"  max gap: {all_gaps[-1]}")
        print(f"  p50: {all_gaps[len(all_gaps)//2]}")
        print(f"  p90: {all_gaps[int(len(all_gaps)*0.9)]}")
        print(f"  p95: {all_gaps[int(len(all_gaps)*0.95)]}")
        print(f"  p99: {all_gaps[int(len(all_gaps)*0.99)]}")

        # Top 10 worst gaps
        print(f"\n  Top 10 worst gaps: {all_gaps[-10:]}")

    return {"gaps": all_gaps}


@app.function(image=image, gpu="A100-80GB", timeout=1800)
def run_diag():
    return _run_diag()


@app.local_entrypoint()
def main():
    print("VeriLM Diagnostic: lm_head logit gap distribution")
    print("=" * 60)
    result = run_diag.remote()
    gaps = result["gaps"]
    if gaps:
        print(f"\nMax gap: {max(gaps)}, tokens measured: {len(gaps)}")
