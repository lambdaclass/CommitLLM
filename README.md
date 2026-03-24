# VeriLM

VeriLM is a commit-and-audit protocol for open-weight LLM inference.

A provider serves responses normally and returns a compact receipt. Later, a verifier can challenge specific tokens and layers, request an opening, and check the opened computation against:

- the committed model identity
- the committed preprocessing policy
- the committed decode and output policy
- the committed transcript state
- the public weights and verifier key

The design goal is:

> Everything except the attention interior is exact, replayable, and cryptographically bound.

This README describes the final protocol boundary. The implementation roadmap for reaching that boundary is in [roadmap.md](./roadmap.md).

## Guarantee Boundary

VeriLM uses four guarantee classes:

| Class | Meaning |
|---|---|
| Exact | Cryptographically checked or canonically recomputed with no ambiguity in the protocol semantics |
| Approximate | Replayed and constrained, but not bit-reproducible because native FP16/BF16 attention is hardware-sensitive |
| Statistical | Commitment binding is exact, but correctness of unopened positions depends on challenge sampling unless deep audit is used |
| Fail-closed | A feature is either replayed exactly or rejected explicitly; the verifier never silently accepts unsupported semantics |

## Verification Coverage

| Area | Verification method | Guarantee |
|---|---|---|
| Input preprocessing | Bound by `input_spec_hash` | Exact |
| Embedding lookup | Merkle proof against committed embedding root | Exact |
| Shell matmuls (`W_q`, `W_k`, `W_v`, `W_o`, `W_gate`, `W_up`, `W_down`) | Verifier-secret Freivalds | Exact |
| Bridge operations (requantization, residual, RMSNorm, RoPE, SiLU) | Canonical recomputation | Exact |
| Prefix / KV provenance | Merkle commitment plus sampled shell verification, or exact deep audit | Statistical by default; Exact in deep audit |
| Attention interior | Independent replay against committed prefix and committed post-attention output | Approximate |
| Final-token boundary | Start from captured pre-final-norm residual | Exact |
| LM head | Freivalds binding plus exact logits replay | Exact |
| Decode policy | Canonical sampler replay or explicit fail-closed rejection | Exact / Fail-closed |
| Output policy | Exact stopping / output replay or explicit fail-closed rejection | Exact / Fail-closed |

## Protocol Overview

### 1. Setup

For a given deployment, VeriLM binds a public model identity `R_W` and a verifier key.

- `R_W` is a Merkle root over the published checkpoint in canonical order.
- The verifier samples secret Freivalds vectors and precomputes verifier-side checks for eight matrix families:
  - `W_q`
  - `W_k`
  - `W_v`
  - `W_o`
  - `W_gate`
  - `W_up`
  - `W_down`
  - `LM_head`
- The verifier key also binds the model configuration needed for canonical replay:
  - hidden sizes and head counts
  - RoPE/scaling configuration
  - RMSNorm epsilon
  - weight/quantization metadata
  - final RMSNorm weights
  - embedding commitment data

### 2. Commit

During inference, the provider:

- runs the model normally
- captures the retained state needed for later audit
- commits to the response transcript
- commits to deployment metadata
- commits to sampling randomness

The receipt binds:

- a trace / retained-state commitment
- a prefix / KV commitment
- a manifest commitment
- a prompt / transcript binding
- a seed commitment for sampled decoding
- the committed token count

### 3. Audit

When challenged, the provider opens:

- the challenged token position
- the requested layer subset or prefix
- the retained-state / shell openings
- the randomness opening
- the committed decode/output policy needed for replay

### 4. Verify

The verifier checks:

- embedding binding
- shell matmuls with Freivalds
- bridge operations exactly or canonically
- prefix/KV provenance according to the audit mode
- attention replay
- the exact final-token tail starting from the captured pre-final-norm residual
- LM-head binding and exact logits replay
- canonical decode policy replay
- output-policy replay

## Final Exact Boundary

The final exact tail starts from a captured live boundary state:

- captured pre-final-norm residual
- exact final RMSNorm
- exact quantization into the LM-head input space
- Freivalds binding of `LM_head`
- exact logits computation
- exact decode-policy replay
- exact token selection replay
- exact output-policy replay when final text is claimed

The verifier must not derive the final token from a hidden state reconstructed through many layers of approximate attention replay and then call that exact.

## Manifest Surface

The final protocol binds deployment semantics through four committed specs:

- `input_spec_hash`
- `model_spec_hash`
- `decode_spec_hash`
- `output_spec_hash`

### Input Spec

- tokenizer / normalization semantics
- chat template
- BOS / EOS preprocessing policy
- truncation / padding policy
- special-token handling
- system prompt semantics

### Model Spec

- `R_W`
- quantization scheme and configuration
- adapter / LoRA / merged-checkpoint identity
- RoPE / scaling configuration
- RMSNorm epsilon
- any other architecture-affecting knob that changes outputs

### Decode Spec

- sampler ID / version
- mode choice
- temperature
- top-k
- top-p
- repetition / frequency / presence penalties
- logit bias
- bad-word masks
- guided decoding / grammar constraints
- tie-breaking rules
- transcript randomness derivation

### Output Spec

- EOS policy
- ignore-EOS behavior
- stop strings
- min/max stopping rules
- special-token stripping
- detokenization / cleanup / whitespace normalization

Every decode/output feature is either replayed exactly or rejected explicitly. VeriLM does not permit partially wired semantics that are bound in the manifest but silently ignored by the verifier.

## Audit Modes

VeriLM has two protocol styles:

- **routine audit**: cheaper, narrower openings, still useful for probabilistic checking
- **deep audit**: stronger checking, intended to support exact full-prefix mode

The final guarantee boundary is:

- exact shell / bridge / final-token tail
- exact prefix anchoring in deep audit
- statistical prefix anchoring in routine audit
- approximate attention interior only

## Verified-Mode Requirements

Verified mode is not “trust”; it is enforced configuration.

The verified server must fail closed unless:

- model identity matches committed `R_W`
- the required capture hooks are active
- verified-mode sync/capture settings match the committed configuration
- unsupported serving optimizations are disabled or explicitly accounted for
- audit-state retention and audit-window policy are configured correctly

Unsupported configurations must be rejected, not silently tolerated.

## Supported Architectures

The final protocol targets autoregressive decoder-only transformers with the committed capture layout and replay semantics.

At minimum, the implementation must:

- detect the intended model family and layout assumptions
- reject unsupported architectures explicitly
- fail closed rather than silently assuming Llama-style geometry everywhere

Generalization beyond constant-width decoder models is a separate engineering task, not an implicit protocol guarantee.

## Assumptions Outside Protocol Scope

VeriLM does not assume honest prover hardware or honest provider runtime behavior. Those are what the protocol is designed to remove.

The remaining explicit assumptions outside protocol scope are:

- standard cryptographic assumptions:
  - collision resistance of the hash functions used for Merkle commitments and manifest binding
  - the usual finite-field soundness assumptions behind Freivalds checks
- verifier-secret secrecy:
  - the prover does not learn the verifier’s secret Freivalds vectors or equivalent verifier-only randomness
- no side-channel leakage of verifier-secret material
- correct execution of the verifier itself

## Why the Attention Interior Is Still Special

The core remaining protocol gap is the attention interior:

- score computation
- masking over the committed prefix
- softmax
- weighted aggregation over `V`

VeriLM constrains attention from both sides, but cannot make native GPU FP16/BF16 attention exactly reproducible in the sidecar design. That is the only intended approximate region of the final protocol.

## Roadmap

This README is the final protocol definition. The implementation sequence for reaching it is in [roadmap.md](./roadmap.md).
