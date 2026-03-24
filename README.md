# VeriLM

VeriLM is a commit-and-audit protocol for open-weight LLM inference.

A provider serves responses normally and returns a compact receipt. Later, a verifier can challenge specific tokens and layers, request an opening, and check the opened computation against:

- the committed model identity
- the committed decode policy
- the committed transcript state
- the public weights and verifier key

The design goal is:

> Everything except the attention interior is exact, replayable, and cryptographically bound.

That goal is not fully complete in the current repo yet. This README describes the current state of the codebase and the intended protocol boundary.

## Current Status

VeriLM is currently between the `V5` and `V6` milestones.

### Implemented today

- Retained-state commit-and-audit path
- Binary audit / receipt path
- Verifier-secret Freivalds checks for the linear shell
- Exact or canonical bridge checks around the shell:
  - requantization
  - residual add
  - RMSNorm
  - RoPE
  - SiLU
- Embedding / layer-0 binding via committed embedding data
- Weight identity binding via committed `R_W`
- Manifest binding for the currently wired deployment fields
- Verifier-side canonical sampler in Rust
- Live canonical sampled-serving hook through a vLLM `logits_processor`
- Capture reliability fixes for EOS trimming, prefix-caching disablement, and mixed-shape stability

### Still approximate or statistical

- The attention interior remains approximate
- Prefix/KV anchoring is still statistical unless a stronger deep-audit mode is used

### Still missing before the strongest claim

- Final-token verification must start from a captured post-attention boundary state, not from a shell-replayed hidden state after many approximate attention layers
- Full decode/output policy completeness:
  - penalties
  - logit bias
  - grammar / guided decoding
  - stop/output policy
  - detokenization replay
- Exact full-prefix deep-audit mode
- Adversarial hardening
- Conformance vectors, challenge-spec documentation, and interoperability tests

For the concrete execution plan, see [roadmap.md](./roadmap.md).

## Verification Classes

VeriLM has three different verification classes:

| Area | Status | Meaning |
|---|---|---|
| Linear shell | Exact / cryptographically checked | Weight matmuls are checked with verifier-secret Freivalds; bridge operations are recomputed exactly or canonically |
| Prefix/KV provenance | Statistical | Merkle binding is exact, but correctness of unsampled prefix positions depends on challenge sampling unless deep audit is used |
| Attention interior | Approximate | Attention replay constrains behavior, but native FP16/BF16 attention is not bit-reproducible |

## What "Linear Shell" Means

The linear shell is everything around the attention interior:

- Q / K / V / O projections
- FFN projections
- residual / RMSNorm / requant bridge steps
- LM head / final token tail
- decode-policy logic after logits

In other words:

- **inside the approximate boundary:** attention scores, softmax, weighted aggregation
- **outside the approximate boundary:** the linear shell and the final token path

This distinction matters. The verifier should not derive the final token from a hidden state reconstructed through approximate attention replay and then call that exact. The exact tail must start from a captured live boundary state after the approximate attention interior.

## Protocol Sketch

### 1. Setup

For a given model deployment:

- compute or publish a committed weight identity `R_W`
- derive a verifier key with verifier-secret Freivalds vectors
- bind deployment fields through a manifest

The verifier key lets the verifier check linear shell matmuls without keeping the full weights in memory.

### 2. Commit

During inference, the provider:

- runs the model normally
- captures the retained state needed for later audit
- commits to the response transcript
- commits to deployment metadata
- commits to sampling randomness

The receipt binds at least:

- trace / retained-state commitment
- KV or prefix commitment
- manifest binding
- seed commitment for sampled decoding

### 3. Audit

When challenged, the provider opens:

- the challenged token position
- the requested layer subset or prefix
- the retained state / shell openings
- the randomness opening
- any policy fields needed for replay

### 4. Verify

The verifier checks:

- linear shell matmuls with Freivalds
- bridge operations exactly or canonically
- embedding / layer-0 binding
- manifest binding
- decode-policy replay
- sampled token replay
- attention replay and prefix anchoring according to the current audit mode

## What Is Exact Today

These are the parts of the protocol that are already intended to be exact in the current design:

- model identity binding via `R_W`
- linear shell weight checks via verifier-secret Freivalds
- quantized bridge semantics around the shell
- deployment-manifest binding for wired fields
- receipt / audit binary format
- sampled token randomness derivation and canonical sampling logic, where the live path and verifier both use the same Rust sampler

## What Is Not Exact Today

### Attention

The core remaining protocol gap is still the attention interior:

- score computation
- masking
- softmax
- weighted aggregation over `V`

VeriLM can constrain attention and replay it, but cannot make native GPU FP16/BF16 attention exactly reproducible in the current sidecar design.

### Prefix correctness

Without exact full-prefix deep audit, unsampled prefix positions remain statistically anchored rather than exactly proven.

### Final-token boundary

The final-token exactness story is still being cleaned up. The verifier should use a captured live post-attention boundary state for the exact LM-head / logits / sampler tail, rather than a shell-replayed hidden state that has accumulated approximation from attention replay.

## Sampled Serving

Sampled serving is the intended default production mode.

The intended architecture is:

- vLLM owns:
  - forward passes
  - batching
  - KV/cache management
  - logits production
- VeriLM owns:
  - per-request randomness commitment
  - per-token seed derivation
  - canonical decode policy
  - final sampled token choice

The live sampled path uses a custom `logits_processor` so that:

1. vLLM computes logits normally
2. VeriLM runs the canonical Rust sampler
3. the chosen token is forced by masking all other logits to `-inf`
4. vLLM's internal sampler becomes irrelevant

This avoids depending on opaque or version-fragile sampler behavior inside vLLM.

## Manifest Direction

The repo is moving toward a four-spec manifest structure:

- `input_spec_hash`
- `model_spec_hash`
- `decode_spec_hash`
- `output_spec_hash`

That split is not fully complete yet. The remaining work includes full binding of:

- tokenizer / normalization / template policy
- model/runtime knobs that affect outputs
- penalties / grammar / logit bias / sampler version
- stop/output / detokenization policy

## Audit Modes

There are two protocol styles in the repo:

- **routine audit**: cheaper, narrower openings, still useful for probabilistic checking
- **deep audit**: stronger checking, intended to support exact full-prefix mode

The final exact/statistical boundary depends on which mode is used. Until exact full-prefix deep audit is finished, prefix anchoring remains statistical.

## Operational Notes

Verified serving requires fail-closed operational settings.

The live verified server should enforce at startup:

- model identity matches committed `R_W`
- prefix caching is disabled in verified mode
- sync/capture mode matches the intended verified configuration

Silent drift here is an operational correctness bug, not just a deployment issue.

## Development Workflow

The project currently uses two GPU workflows:

- **persistent SSH GPU box** for inner-loop development
- **Modal** for outer-loop validation and deployment-style checks

This is intentional: Modal cold starts and image rebuilds are too expensive for every protocol iteration.

## What Happens Before Publication

Before making the strongest protocol claims, the repo still needs:

- exact final-token boundary cleanup
- full decode/output policy completeness
- exact full-prefix deep-audit mode
- adversarial verifier-hardening
- golden/conformance vectors
- challenge protocol spec
- binary interoperability/versioning tests
- final V5/V6 benchmarks
- full protocol documentation updates

## Roadmap

The authoritative execution order is in [roadmap.md](./roadmap.md).

At a high level:

1. finish live sampled serving and decode/output binding
2. canonicalize the V5 retained-state path
3. add exact full-prefix mode
4. harden the verifier adversarially
5. add conformance/interoperability work
6. benchmark the final protocol
7. update the paper and docs

## End State

The target end state for `V6` is:

- exact shell / bridge / final-token path
- exact preprocessing, model, decode, and output policy binding
- exact sampled replay as the default production mode
- greedy mode as the `temperature=0` special case
- exact prefix mode available through deep audit
- only the attention interior remains approximate

After `V6`, the main remaining protocol gap is attention exactness. A `V7` only makes sense if that story improves materially.
