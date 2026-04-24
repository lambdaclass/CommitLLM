"""
E2E validation: audit-only attention path on llama-w8a8-audited / qwen-w8a8-audited.

Proves the post-B3 attention audit status populates on real GPU:

  1. VerifiedInferenceServer.chat() — capture + commit (witnessed scores
     auto-install for W8A8 llama/qwen).
  2. verilm_rs.generate_key_binary_with_profile("{model}-w8a8-audited", ...)
  3. server.audit() on a random generated-token index.
  4. verilm_rs.verify_v4_full_binary() returns report["attention_status_json"].
  5. Assertions on the decoded status:
       - mode == "audited_inputs_not_verified"
       - score_anchor.checked == True
       - kv_provenance.checked == True
       - wiring is present (any sub-field populated)
       - local_replay may be None (B4 not wired yet)

Both llama and qwen run in parallel.

Usage:
    modal run --detach scripts/modal/test_audited_inputs_e2e.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import VERIFICATION
except ImportError:
    VERIFICATION = []

import modal

app = modal.App("verilm-test-audited-inputs-e2e")

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
    "Write one sentence about the ocean.",
    "Tell me a fun fact about space.",
]


def _run_model(model_name, model_id):
    import hashlib
    import json
    import random
    import time

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

    profile_name = f"{model_name}-w8a8-audited"
    print(f"\n1. Generate verifier key (profile={profile_name})...")
    seed = hashlib.sha256(f"audited-e2e-{model_name}".encode()).digest()
    t0 = time.time()
    key_binary, artifact_binary = verilm_rs.generate_key_binary_with_profile(
        model_dir, list(seed), profile_name,
    )
    keygen_ms = (time.time() - t0) * 1000
    print(f"  keygen: {keygen_ms:.0f}ms, key={len(key_binary)/1024/1024:.1f} MB")

    full_layers = list(range(n_layers))
    rng = random.Random(42)

    results = []
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(f"[{model_name}] {msg}")
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK: {msg}")

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"[{model_name}] prompt {pi}: {prompt}")
        print(f"{'='*60}")

        result = server.chat(
            prompt=prompt, max_tokens=24,
            temperature=0.8, top_k=50, top_p=0.9,
        )
        request_id = result["request_id"]
        n_tokens = result["n_tokens"]
        commitment = result["commitment"]
        n_prompt = commitment.get("n_prompt_tokens", 0)
        n_gen = n_tokens - n_prompt

        print(f"  n_prompt={n_prompt}, n_gen={n_gen}")
        print(f"  text: {result['generated_text'][:80]}...")
        check(n_gen > 0, f"generated {n_gen} tokens")

        gen_start = n_prompt - 1
        gen_offset = rng.randint(0, max(0, n_gen - 1))
        challenge_idx = gen_start + gen_offset

        audit_binary = server.audit(
            request_id=request_id,
            token_index=challenge_idx,
            layer_indices=full_layers,
            tier="full",
            binary=True,
        )
        print(f"  audit token_index={challenge_idx}, payload={len(audit_binary)/1024:.1f} KiB")

        report = verilm_rs.verify_v4_full_binary(
            bytes(audit_binary), key_binary, artifact_binary,
        )

        status_json = report.get("attention_status_json")
        check(status_json is not None, "report carries attention_status_json")
        if not status_json:
            continue
        status = json.loads(status_json)
        print(f"  attention_status: {json.dumps(status, indent=2)}")

        check(
            status.get("mode") == "audited_inputs_not_verified",
            f"mode == audited_inputs_not_verified (got {status.get('mode')!r})",
        )

        sa = status.get("score_anchor")
        check(sa is not None and sa.get("checked") is True,
              f"score_anchor.checked == True (got {sa!r})")

        kv = status.get("kv_provenance")
        check(kv is not None and kv.get("checked") is True,
              f"kv_provenance.checked == True (got {kv!r})")

        wiring = status.get("wiring")
        check(wiring is not None, f"wiring populated (got {wiring!r})")
        if wiring is not None:
            any_set = any(wiring.get(k) is not None
                          for k in ("mask_ok", "rope_ok", "gqa_ok"))
            check(any_set, f"wiring has at least one sub-audit (got {wiring!r})")

        # Decode path must still pass on audited profile.
        decode_fs = [f for f in report.get("failures", [])
                     if "attention" not in f.lower()]
        check(len(decode_fs) == 0,
              f"no decode failures (got {len(decode_fs)}: {decode_fs[:3]})")

        results.append({
            "prompt_idx": pi,
            "token_index": challenge_idx,
            "status": status,
            "passed": report["passed"],
            "checks_passed": report["checks_passed"],
            "checks_run": report["checks_run"],
        })

    print(f"\n{'='*60}")
    print(f"[{model_name}] Summary")
    print(f"  Runs: {len(results)}")
    print(f"  Failures: {len(failures)}")
    for f in failures:
        print(f"    - {f}")
    print(f"{'='*60}")

    return {
        "model": model_name,
        "results": results,
        "failures": failures,
        "passed": len(failures) == 0,
    }


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_qwen():
    return _run_model("qwen", MODELS["qwen"])


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_llama():
    return _run_model("llama", MODELS["llama"])


@app.local_entrypoint()
def main():
    print("VeriLM E2E: audit-only attention (score_anchor + kv_provenance + wiring)")
    print("=" * 60)

    qwen_future = run_qwen.spawn()
    llama_future = run_llama.spawn()

    qwen_result = qwen_future.get()
    llama_result = llama_future.get()

    all_passed = True
    for result in [qwen_result, llama_result]:
        model = result["model"]
        n = len(result["results"])
        nf = len(result["failures"])
        print(f"\n{model}: {n} runs, {nf} failure(s)")
        if nf:
            all_passed = False
            for f in result["failures"]:
                print(f"  - {f}")

    print()
    if all_passed:
        print("AUDIT-ONLY E2E: PASSED (both models)")
    else:
        print("AUDIT-ONLY E2E: FAILED")
        raise SystemExit(1)
