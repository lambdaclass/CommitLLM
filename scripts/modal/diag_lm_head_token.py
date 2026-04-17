"""
Diagnostic: lm_head token replay failure at non-zero decode positions.

Hypothesis: the verifier's quantized logits (i32 from i8 matmul) diverge
from the GPU's float logits enough to change the sampled/argmax token,
especially with temperature=1.0.

Tests:
  1. Greedy (temperature=0) vs sampled (temperature=1.0)
  2. Challenge token 0, middle token, last token
  3. Compare the prover's logits_i32 argmax vs the actual generated token

Usage:
    modal run --detach scripts/modal/diag_lm_head_token.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-diag-lm-head-token")

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

PROMPT = "Explain the theory of relativity in one paragraph."
MAX_TOKENS = 32


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

    seed = hashlib.sha256(PROMPT.encode()).digest()
    key_json = verilm_rs.generate_key(model_dir, seed)
    full_layers = list(range(n_layers))

    # Test both greedy and sampled
    for temperature in [0.0, 1.0]:
        mode = "greedy" if temperature == 0.0 else "sampled"
        print(f"\n{'='*70}")
        print(f"MODE: {mode} (temperature={temperature})")
        print(f"{'='*70}")

        cap._capture_mode = "minimal"
        result = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS,
                             temperature=temperature)
        rid = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)

        print(f"  n_tokens={n_tokens}, n_prompt={n_prompt}, generated={result['generated_text'][:60]}...")

        # Challenge each token position
        test_indices = [0]
        if n_tokens > 2:
            test_indices.append(n_tokens // 2)
        if n_tokens > 1:
            test_indices.append(n_tokens - 1)

        for token_idx in test_indices:
            # Need fresh chat for each audit (audit consumes state)
            if token_idx > 0:
                result = server.chat(prompt=PROMPT, max_tokens=MAX_TOKENS,
                                     temperature=temperature)
                rid = result["request_id"]

            print(f"\n  --- token_index={token_idx} (gen_start={n_prompt - 1 if n_prompt > 0 else 0}) ---")
            is_generated = token_idx >= (n_prompt - 1 if n_prompt > 0 else 0)
            print(f"  is_generated={is_generated}")

            # JSON audit
            audit_json = server.audit(
                request_id=rid,
                token_index=token_idx,
                layer_indices=full_layers,
                tier="full",
                binary=False,
            )
            report = verilm_rs.verify_v4(audit_json, key_json)

            # Parse audit to inspect logits
            audit = json.loads(audit_json)
            shell = audit.get("shell_opening", {})
            logits_i32 = shell.get("logits_i32")
            final_residual = shell.get("final_residual")

            token_id = audit.get("token_id")

            if logits_i32:
                # Find argmax of logits_i32
                max_val = max(logits_i32)
                argmax_token = logits_i32.index(max_val)

                # Find top-5
                indexed = sorted(enumerate(logits_i32), key=lambda x: -x[1])[:5]
                top5 = [(idx, val) for idx, val in indexed]

                # Find rank of actual token
                rank = next((i for i, (idx, _) in enumerate(indexed) if idx == token_id), -1)

                print(f"  token_id={token_id}, argmax={argmax_token}, argmax_val={max_val}")
                print(f"  actual token rank in logits_i32: {rank}")
                print(f"  top5: {top5}")

                if token_id != argmax_token:
                    actual_val = logits_i32[token_id] if token_id < len(logits_i32) else None
                    print(f"  MISMATCH: argmax={argmax_token}(val={max_val}) vs actual={token_id}(val={actual_val})")
                    if actual_val is not None:
                        print(f"  logit gap: {max_val - actual_val}")
            else:
                print(f"  no logits_i32 in shell opening")

            if final_residual:
                fr_abs = [abs(v) for v in final_residual]
                print(f"  final_residual: len={len(final_residual)}, max_abs={max(fr_abs):.4f}, mean_abs={sum(fr_abs)/len(fr_abs):.4f}")

            print(f"  verify: {report['checks_passed']}/{report['checks_run']} {'PASS' if report['passed'] else 'FAIL'}")
            if not report["passed"]:
                for f in report["failures"]:
                    print(f"    FAIL: {f}")
            if report.get("skipped"):
                for s in report["skipped"]:
                    print(f"    SKIP: {s}")


@app.function(image=image, gpu="A100-80GB", timeout=900)
def run_diag():
    return _run_diag()


@app.local_entrypoint()
def main():
    print("VeriLM Diagnostic: lm_head token replay")
    print("=" * 60)
    run_diag.remote()
