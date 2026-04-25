# CommitLLM

CommitLLM is a cryptographic commit-and-audit protocol for open-weight LLM inference.

When you call an LLM provider, you have no proof they ran the model they claim, with the configuration they advertise, or that the output wasn't tampered with. CommitLLM closes that gap: the provider serves responses normally on GPU and returns a compact receipt. Trace data is opened only when challenged. A verifier checks the opening on CPU against committed model weights, deployment configuration, transcript state, and decode policy.

> Status: active development. Lean 4 formalization is in progress. Internal crate names still use the `verilm` prefix and are pending rename to `commitllm`.

## What It Gives You

- Exact checks for the large linear shell (all weight matrix families via Freivalds), quantization/dequantization boundaries, and supported decode policies
- Exact audit detection for wrong weights, wrong deployment configuration, wrong sampling, and wrong decode policy on opened traces
- Commitment-bound execution trace: the provider cannot change its story after committing
- CPU-only verification on the client side
- A normal GPU serving path for the provider, without a proving circuit or per-response proof generation

## Verification Coverage

The table below shows what the protocol verifies today and what remains open. "Verified" means the verifier independently checks the claim from committed data and verifier-secret randomness. "Open" means the protocol commits the data but does not yet independently verify the computation.

```
 COMPONENT                CLAIM                           STATUS
 ─────────────────────────────────────────────────────────────────────
 Embedding                token → embedding row            ✅ Verified
                          Merkle proof against weights     ✅ Verified

 Linear shell (per layer)
   QKV projection         Wq·x, Wk·x, Wv·x Freivalds    ✅ Llama
                                                          ⚠️  Qwen (bridge gap)
   FFN projections        Wg·x, Wu·x, Wd·x Freivalds    ✅ Verified
   Output projection      Wo·a Freivalds                  ✅ Verified
   LM-head                lm_head binding                 ✅ Verified

 Bridge (per layer)
   Residual chain         x_attn → QKV → attn → FFN →    ✅ Verified
                          residual → next layer
   Nonlinear gates        SiLU, RMSNorm canonical replay  ✅ Verified

 Attention (audit-only)                                   ⚠️  NOT VERIFIED
   Arbitrary-position     softmax(QKᵀ/√d)·V output        ❌ Out of scope
   Score anchoring        recompute QKᵀ/√d for the last   ✅ Audited
                          generated token, compare to
                          witnessed pre-softmax scores
   KV provenance          opened K/V rows match committed ✅ Audited
                          token positions / cache rows
   Wiring (GQA, RoPE,     causal mask + RoPE config hash  ✅ Audited
   causal mask)           + GQA head mapping; mask check
                          is last-generated-token only
   Local replay smoke     token-0 exact attention replay   ✅ Token 0 only

 Decode
   Greedy                 argmax(logits) must match        ✅ Validated
   Sampled                exact GPU-logit sampling         ✅ CapturedLogits
   LP-hidden bf16         bf16 lm_head replay              ⚙️  Intermediate

 Binding
   Prompt                 hash binding                     ✅ Verified
   Sampling manifest      temperature, top_k, top_p, seed ✅ Verified
   Token count            n_tokens commitment              ✅ Verified
   KV transcript          Merkle proofs vs kv_roots        ✅ Token 0 only
   IO chain               per-token hash chain             ✅ Verified
   Tamper detection        36 adversarial scenarios         ✅ Llama
 ─────────────────────────────────────────────────────────────────────
```

**Arbitrary-position attention outputs are not verified in the shipped product.** GPU FlashAttention uses bf16 tiled accumulation that does not match any CPU f64/f32 replay the verifier can do within a meaningful tolerance. Every attempted production verification path — exact stock-kernel replay, tiled/LSE replay, stock-bounded certification, and deterministic kernels — is closed. The kept claim is narrower and honest: exact decode plus audited attention inputs and wiring. Stock-mode attention audits cover score anchoring (last generated token), KV provenance, GQA / RoPE-config / causal-mask (last generated token) wiring, and token-0 local replay smoke. The score-anchor and causal-mask audits are scoped to the last generated token because the prover-side score witness retains Q only for the final decode step; per-step Q retention is a deferred extension. See the [roadmap](roadmap.md) for the audit implementation plan.

Everything else — embedding, all linear projections, bridge chain, decode, and all bindings — is independently verified with exact or information-theoretically sound checks on the supported path. For sampled decode, the kept exact path is `CapturedLogits`: capture the exact GPU logits, verify the exact sampled token, and Freivalds-bind those logits back to `lm_head`.

## Current Prototype

- Measured online tracing overhead is about 12–14% during generation
- Measured verifier cost for Llama 70B is about 1.3 ms per challenged token under the 10-layer routine audit, and about 10 ms for a 1-token full audit
- End-to-end tested on Qwen2.5-7B-W8A8 and Llama-3.1-8B-W8A8 with binary key path, commitment-derived challenges, greedy and sampled decode, mixed audit tiers, tamper detection, and EOS handling
- Exact sampled decode is GPU-validated on both models via `CapturedLogits`:
  - Qwen: `17,336/17,336` decode checks passed
  - Llama: `23,335/23,335` decode checks passed
- Captured-logits economics are now the decode-side tradeoff:
  - retained state: ~`501 KiB/token` (Llama), ~`594 KiB/token` (Qwen)
  - opened payload: ~`0.5–0.6 MiB` per challenged token
  - verifier CPU: two dot products per challenged token
- Arbitrary-position attention outputs are not verified in the shipped product; the kept stock-mode attention claim is audit-only (score anchoring on the last generated token, KV provenance, GQA / RoPE-config / causal-mask wiring with the mask audit scoped to the last generated token, token-0 local replay smoke). Audit-only E2E is green on `llama-w8a8-audited` and `qwen-w8a8-audited`.

## Protocol

1. **Setup**. The verifier builds a key from the public checkpoint. It contains a Merkle root over weights, secret Freivalds vectors for each matrix family (`Wq`, `Wk`, `Wv`, `Wo`, `Wgate`, `Wup`, `Wdown`, `LM_head`), and the model configuration needed for replay.

2. **Commit**. During inference the provider runs the model normally, captures retained state, and returns a compact receipt binding the execution trace, KV state, deployment manifest, prompt, sampling randomness, and token count.

3. **Audit**. When challenged, the provider opens the requested token positions and layer range. Routine audit samples prefix state. Deep audit opens all requested layers and prefix state.

4. **Verify**. The verifier checks:
   - Embedding Merkle proof
   - Shell matmuls via Freivalds (all weight matrix families)
   - Exact INT8 bridge tensors by canonical recomputation
   - KV transcript Merkle proofs (when KV data is opened)
   - Attention: token-0 exact replay as regression smoke, plus audited attention inputs/wiring (score anchoring and causal-mask check scoped to the last generated token; KV provenance, GQA, and RoPE-config hash apply to the full opened range); arbitrary-position attention outputs are not verified
   - Final-token tail from the captured residual
   - Exact sampled decode from captured GPU logits plus LM-head algebraic binding
   - Prompt, sampling manifest, and IO chain binding

The protocol is commitment-bound end-to-end. Within that binding, large linear components are verified by verifier-secret, information-theoretically sound algebraic checks. Exact bridge tensors and supported nonlinear subcomputations are checked by canonical re-execution. Sampled decode correctness is checked by exact GPU-logit capture plus algebraic LM-head binding; cheaper replay-based paths remain useful for greedy decode where validated. Arbitrary-position attention outputs are not verified on stock GPU kernels — the kept stock-mode attention claim is audit-only (score anchoring on the last generated token, KV provenance, GQA / RoPE-config / causal-mask wiring with the mask audit scoped to the last generated token, token-0 local replay smoke). Unsupported semantics fail closed.

## Try It

You need a [Modal](https://modal.com) account. All tests run on an A100 GPU.

### Prerequisites

```bash
pip install modal
modal setup   # one-time auth
```

### Maintained E2E Surface

These tests exercise the maintained production wiring: shell/decode/binding coverage with everything except arbitrary-position attention. Takes ~5-10 minutes per model on first run (mostly image build).

```bash
# Llama 3.1 8B W8A8 — shell/decode surface, token-0 attention smoke, QKV Freivalds
modal run --detach scripts/modal/tests/llama/test_e2e.py

# Qwen 2.5 7B W8A8 — shell/decode surface, token-0 attention smoke, FFN Freivalds
modal run --detach scripts/modal/tests/qwen/test_e2e.py
```

What `test_e2e.py` checks:
- Binary key path with commitment-derived random challenges
- Full-tier (all layers) and routine-tier (random subset) shell/decode audits
- Greedy and sampled decoding (temperature, top_k, top_p, seed commitment)
- Long prompts (~500 tokens) and long generations (512 tokens)
- Multi-position escalation (same request, multiple challenges)
- Tamper detection (bit-flip → verifier rejects)
- EOS early-stop handling
- Token-0 exact attention smoke check (Llama)
- Profile assertion (ExactReplay vs WitnessedScores, ExactTokenIdentity vs LpHiddenBf16)
- Payload size and verification timing reporting

### Exact Sampled-Decode Validation

This test exercises the kept sampled-decode path directly: `CapturedLogits`.

```bash
# Llama + Qwen captured-logits profiles
modal run --detach scripts/modal/test_e2e_captured_logits.py
```

What `test_e2e_captured_logits.py` checks:
- `llama-w8a8-captured-logits` and `qwen-w8a8-captured-logits`
- Exact sampled token verification from captured GPU logits
- Freivalds binding from captured logits back to `lp_hidden × lm_head_bf16`
- Greedy regression checks on the same profiles
- Repeated sampled runs on the same prompts
- Tamper rejection

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

The protocol is commitment-bound end-to-end. Within that binding, large linear layers are verified by verifier-secret, information-theoretically sound algebraic checks, quantization/dequantization boundaries and supported nonlinear subcomputations are checked by canonical re-execution, and sampled decode correctness is checked by exact GPU-logit capture plus algebraic LM-head binding. Attention at arbitrary token positions is an open problem: stock GPU FlashAttention kernels use bf16 tiled accumulation that does not match verifier-side f64 replay beyond the first decode step. Routine prefix-state provenance is statistical unless deep audit is used. Unsupported semantics fail closed.

In the current prototype, online tracing adds roughly 12-14% during generation. The larger commitment and finalization cost is measured separately, currently runs synchronously, and is a candidate for asynchronous deferral in production. For Llama 70B, the measured verifier cost is about 1.3 ms per challenged token under the 10-layer routine audit, and about 10 ms for a 1-token full audit.
