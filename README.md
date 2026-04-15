# CommitLLM

CommitLLM is a cryptographic commit-and-audit protocol for open-weight LLM inference.

When you call an LLM provider, you have no proof they ran the model they claim, with the configuration they advertise, or that the output wasn't tampered with. CommitLLM closes that gap: the provider serves responses normally on GPU and returns a compact receipt. Trace data is opened only when challenged. A verifier checks the opening on CPU against committed model weights, deployment configuration, transcript state, and decode policy.

> Status: active development. Lean 4 formalization is in progress. Internal crate names still use the `verilm` prefix and are pending rename to `commitllm`.

## What It Gives You

- Exact checks for the large linear shell, quantization and dequantization boundaries, and supported canonical replay steps
- Exact audit detection for wrong weights, wrong deployment configuration, and wrong decode policy on opened traces
- Approximate replay only at the attention interior, where native GPU FP16 and BF16 execution is not bit reproducible
- Statistical routine-audit provenance for unopened prefix KV state
- CPU-only verification on the client side
- A normal GPU serving path for the provider, without a proving circuit or per-response proof generation

## Current Prototype

- Measured online tracing overhead is about 12-14% during generation
- Measured verifier cost for Llama 70B is about 1.3 ms per challenged token under the 10-layer routine audit, and about 10 ms for a 1-token full audit
- The corrected replay path is currently measured on Qwen2.5-7B-W8A8 and Llama-3.1-8B-W8A8
- On those measured paths, worst-case INT8 attention mismatch beyond 1k tokens is `L_inf = 8` for Qwen and `L_inf = 9` for Llama, with more than 99.8% of elements staying within one quantization bucket

## Protocol

1. **Setup**. The verifier builds a key from the public checkpoint. It contains a Merkle root over weights, secret Freivalds vectors for each matrix family (`Wq`, `Wk`, `Wv`, `Wo`, `Wgate`, `Wup`, `Wdown`, `LM_head`), and the model configuration needed for replay.

2. **Commit**. During inference the provider runs the model normally, captures retained state, and returns a compact receipt binding the execution trace, KV state, deployment manifest, prompt, sampling randomness, and token count.

3. **Audit**. When challenged, the provider opens the requested token positions and layer range. Routine audit samples prefix state. Deep audit opens all requested layers and prefix state.

4. **Verify**. The verifier checks:
   - Embedding Merkle proof
   - Shell matmuls via Freivalds
   - Exact INT8 bridge tensors by canonical recomputation
   - KV provenance
   - Attention replay against the committed post-attention output
   - Final-token tail from the captured residual
   - LM-head binding
   - Decode and output policy replay

The protocol is commitment-bound end-to-end. Within that binding, large linear components are verified by verifier-secret, information-theoretically sound algebraic checks, exact bridge tensors and supported nonlinear subcomputations are checked by canonical re-execution, attention is checked by bounded approximate replay, and routine KV provenance is statistical unless deep audit is used. Unsupported semantics fail closed.

## Try It

You need a [Modal](https://modal.com) account. All tests run on an A100 GPU.

### Prerequisites

```bash
pip install modal
modal setup   # one-time auth
```

### Per-Model E2E Tests

Each test loads a real model, generates text with cryptographic capture, commits, generates a verifier key, audits at multiple tiers, and verifies. Takes ~5-10 minutes per model (most of that is image build on first run).

```bash
# Llama 3.1 8B W8A8 — strongest tier: exact attention replay, exact token identity
modal run --detach scripts/modal/tests/llama/test_e2e.py

# Qwen 2.5 7B W8A8 — decode path regression test (attention verification disabled, known open problem)
modal run --detach scripts/modal/tests/qwen/test_e2e.py
```

What `test_e2e.py` checks:
- Full-tier verify (all layers, binary key path)
- Routine-tier verify (subset of layers via random challenge)
- Tamper detection (bit-flip → verifier rejects)
- EOS early-stop handling
- Multi-position verification (Llama) / last-token verification (Qwen)
- Profile assertion (ExactReplay vs WitnessedScores, ExactTokenIdentity vs LpHiddenBf16)

### Adversarial Tamper Tests

36 scenarios that systematically tamper with individual fields in an honest audit and assert the verifier rejects each one **for the right reason**. This is not "does an honest prover pass?" — this is "can a dishonest prover cheat?"

```bash
modal run --detach scripts/modal/tests/llama/test_adversarial.py
```

Boundaries tested: Freivalds (all 7 matmul families), Merkle binding (retained state, final residual), IO chain, embedding proof, LM-head logits, decode replay (greedy + sampled), manifest/prompt binding, cross-request splice, layer swap, prefix tampering, token index shift, seed commitment.

### Demo Script

Polished output with timings, payload sizes, and per-prompt breakdown. Suitable for walkthroughs.

```bash
modal run --detach scripts/modal/demo_llama_e2e.py
```

### Checking Results

`--detach` runs the job server-side so you can close your terminal. To check results:

```bash
# List recent runs
modal app list

# Stream logs from a running/completed app
modal app logs <app-id>
```

Or check the Modal dashboard at `modal.com/apps`.

## Repository Layout

| Component | Path |
|---|---|
| Core types and traits | [`crates/verilm-core/`](crates/verilm-core/) |
| Key generation | [`crates/verilm-keygen/`](crates/verilm-keygen/) |
| Verifier | [`crates/verilm-verify/`](crates/verilm-verify/) |
| Prover (Rust) | [`crates/verilm-prover/`](crates/verilm-prover/) |
| Prover (Python sidecar) | [`sidecar/`](sidecar/) |
| Python bindings | [`crates/verilm-py/`](crates/verilm-py/) |
| Test vectors | [`crates/verilm-test-vectors/`](crates/verilm-test-vectors/) |
| Per-model E2E tests | [`scripts/modal/tests/`](scripts/modal/tests/) |
| Adversarial tamper tests | [`scripts/modal/tests/llama/test_adversarial.py`](scripts/modal/tests/llama/test_adversarial.py) |
| Demo script | [`scripts/modal/demo_llama_e2e.py`](scripts/modal/demo_llama_e2e.py) |
| Benchmarks and diagnostics | [`scripts/modal/`](scripts/modal/) |
| Lean formalization | [`lean/`](lean/) |
| Paper | [`paper/`](paper/) |

## Abstract

Large language models are increasingly used in settings where integrity matters, but users still lack technical assurance that a provider actually ran the claimed model, decode policy, and output behavior. Fingerprinting and statistical heuristics can provide signals, but not exact per-response verification. Zero-knowledge proof systems provide stronger guarantees, but at prover costs that remain impractical for production LLM serving.

We present CommitLLM, a cryptographic commit-and-audit protocol for open-weight LLM inference. CommitLLM keeps the provider on the normal serving path and keeps verifier work fast and CPU-only. It combines commitment binding, direct audit, and randomized algebraic fingerprints, including Freivalds-style checks for large matrix products, rather than per-response proof generation or full re-execution. Its main costs are retained-state memory over the audit window and audit bandwidth, not per-response proving.

The protocol is commitment-bound end-to-end. Within that binding, large linear layers are verified by verifier-secret, information-theoretically sound algebraic checks, quantization/dequantization boundaries and supported nonlinear subcomputations are checked by canonical re-execution, attention is verified by bounded approximate replay, and routine prefix-state provenance is statistical unless deep audit is used. Unsupported semantics fail closed.

In the current prototype, online tracing adds roughly 12-14% during generation. The larger commitment and finalization cost is measured separately, currently runs synchronously, and is a candidate for asynchronous deferral in production. For Llama 70B, the measured verifier cost is about 1.3 ms per challenged token under the 10-layer routine audit, and about 10 ms for a 1-token full audit. On the corrected replay path, the measured attention corridor is small on Qwen2.5-7B-W8A8 and Llama-3.1-8B-W8A8: at context lengths beyond 1k tokens, worst-case INT8 mismatch is single-digit (`L_inf = 8` and `9`) and more than 99.8% of elements remain within one quantization bucket.
