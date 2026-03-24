# Roadmap

This roadmap reflects the current execution order for VeriLM:

1. Finish live sampled serving so production behavior is exactly replayable.
2. Canonicalize the V5 retained-state path.
3. Add exact-strengthening features needed for the strongest claim.
4. Adversarially harden the verifier before trusting final benchmarks.
5. Add conformance, interoperability, and operational fail-closed checks.
6. Only then optimize further, benchmark the final protocol, and publish.

## Claim-Critical Checklist

Do not claim "everything except attention" until all of these are complete:

- [ ] live sampled serving is replayable end to end
- [ ] decode/output policy completeness is finished
- [ ] the exact final-token boundary starts at the captured pre-final-norm residual
- [ ] exact full-prefix deep-audit mode exists
- [ ] the V5 retained-state path is canonical
- [ ] adversarial hardening is complete
- [ ] conformance vectors and challenge-spec documentation exist

## Publication-Credibility Checklist

Do not treat the protocol as publication-ready until all of these are complete:

- [ ] final V5/V6 benchmarks are run
- [ ] exact/statistical/approximate boundaries are documented clearly
- [ ] canonical semantics vs trust assumptions are documented clearly
- [ ] binary interoperability/versioning behavior is tested
- [ ] operational fail-closed startup checks exist
- [ ] at least one independent non-Rust verifier consumer can consume the golden vectors

## 0. Closed Blockers

These are no longer on the critical path:

- [x] Sync-equivalence: `global` and `event` modes were checked and matched.
- [x] Capture reliability: EOS trailing-forward trim, prefix-caching disablement, counter reset, and mixed-shape stability are in place.
- [x] Deterministic capture mismatch investigation: no remaining evidence that modulo-based projection identification must be replaced.

## 1. V6 First: Live Canonical Sampled Serving and Decode/Output Binding

Sampled serving is required for V6. Greedy remains the `temperature=0` special case.

- [ ] **Implement live canonical sampled decoding through a vLLM `logits_processor`**
  - keep vLLM responsible for forward passes, batching, KV/cache, and logits production
  - keep VeriLM responsible for the final token-selection semantics
  - do not fork vLLM if the `logits_processor` path is sufficient

- [ ] **Finish the decode/output parts of the manifest before declaring sampled serving done**
  - decode spec: sampler ID / version, temperature, top-k, top-p, repetition / frequency / presence penalties, logit bias, bad-word masks, grammar / guided-decoding constraints, mode choice, tie-breaking rules
  - output spec: EOS policy, stop strings, min/max stopping rules, ignore-EOS behavior, special-token stripping, detokenization / cleanup / whitespace normalization

- [ ] **Bind transcript randomness end to end**
  - generate a fresh random per-request `batch_seed`
  - commit `seed_commitment = H(batch_seed)` in the receipt
  - reveal `batch_seed` on audit
  - derive per-token seeds deterministically from `batch_seed` and token index

- [ ] **Apply the canonical decode policy in the live path**
  - temperature
  - top-k
  - top-p
  - repetition / frequency / presence penalties
  - grammar / guided decoding
  - logit bias
  - tie-breaking rules

- [ ] **Force the chosen token**
  - mask all other logits so vLLM's internal sampler becomes irrelevant
  - neutralize or bypass vLLM's built-in sampling knobs so decode policy is not applied twice

- [ ] **Keep greedy mode as a supported special case**
  - `temperature=0` triggers argmax behavior
  - same receipt structure
  - useful for deterministic benchmarks and debugging

- [ ] **Add sampled end-to-end tests through the live HTTP/server path**
  - honest sampled pass
  - wrong seed fails
  - wrong manifest fails
  - wrong sampled token fails
  - cross-request splice fails

- [ ] **Add verifier logic for decode/output policy completeness needed by sampled serving**
  - recompute penalties
  - verify grammar / constraint behavior
  - verify stop-string and stop-policy behavior
  - replay detokenization / cleanup when final text is claimed

- [ ] **Add sampler drift protection**
  - version-lock / conformance tests so sampler behavior cannot drift silently across vLLM upgrades

- [ ] **Make sampled decoding the default production mode**
  - only after the live canonical sampler is stable, replayable, and fully tested

## 2. Canonical V5 Path

- [ ] **Make the key-only retained-state path the canonical protocol path**
- [ ] **Keep weight-backed replay debug-only / oracle-only**
- [ ] **Remove or demote transitional V4 framing**
- [ ] **Keep only irreducible retained state long-term**
- [ ] **Keep the binary audit / receipt format as the canonical protocol format**
- [ ] **Ensure audit-time weight loading stays bound to the committed `R_W`**

## 3. Remaining Exactness and Strengthening

- [ ] **Only claim "everything except attention" after all of the following are done**
  - sampled serving is live and replayable end to end
  - decode/output policy completeness is finished
  - the exact final-token boundary starts at the captured pre-final-norm residual
  - exact full-prefix deep-audit mode exists

- [ ] **Move the exact final-token boundary to the strongest post-attention state**
  - capture the **pre-final-norm residual** in the live path
  - do not derive the final token from a shell-replayed hidden state after approximate attention replay
  - make the verifier recompute the exact tail from that captured boundary:
    - final RMSNorm
    - LM head
    - logits
    - decode policy
    - sampled token

- [ ] **Complete the manifest in its final protocol form**
  - input spec: tokenizer / normalization, chat template, BOS / EOS policy, truncation / padding, special-token handling, system prompt
  - model spec: `R_W`, quantization config, adapter / LoRA / merged-checkpoint identity, RoPE / scaling config, RMSNorm epsilon, other architecture-affecting knobs
  - decode spec and output spec are already required in Section 1 for sampled-serving completion

- [ ] **Add verifier logic for policy completeness**
  - any remaining input/model-spec verification beyond the decode/output checks already required in Section 1

- [ ] **Keep the final token tail exact end to end in the published protocol**
  - final hidden
  - LM head / logits
  - logit modifiers
  - token selection
  - stopping behavior
  - output-text claim boundaries

- [ ] **Add exact full-prefix deep-audit mode**
  - without this, prefix anchoring remains statistical

- [ ] **Ensure deep-audit batching uses verifier-secret randomness**

- [ ] **Run a boundary-fix benchmark checkpoint**
  - after the final-token boundary moves to the captured pre-final-norm residual
  - measure any change in audit open cost, verifier cost, and payload size

## 4. Adversarial Hardening

This comes before trusting final benchmarks or making strong publication claims.

- [ ] **Add an explicit adversarial verifier-hardening phase**

- [ ] **Build a tamper corpus for every verifier boundary**
  - shell openings
  - retained-state leaves
  - IO chain
  - embedding proofs
  - manifest fields
  - prompt hashes
  - seed commitments
  - LM-head / final-token step
  - decode replay
  - stop policy
  - prefix openings
  - routine / full audit semantics
  - malformed binary payloads

- [ ] **Add cross-proof and splice attacks**
  - mix request A receipts with request B audits
  - mix prefixes across requests
  - mix manifests / weights / prompts / seeds across requests

- [ ] **Add boundary-condition fuzzing**
  - EOS-shortened requests
  - long prompt / short output
  - short prompt / long output
  - weird tier / layer requests
  - malformed serialization
  - unknown versions

- [ ] **Assert failures for the right reason**
  - not just "some error"
  - where possible, specific verifier failure messages

## 5. Conformance, Interoperability, and Challenge Specification

- [ ] **Add golden / conformance vectors**
  - fixed receipts
  - fixed audit responses
  - fixed verifier outputs
  - usable by future Rust / Python / TypeScript verifiers

- [ ] **Write the challenge protocol specification**
  - when `challenge_seed` is sampled
  - how token challenges are derived
  - how layer challenges are derived
  - how prefix challenges are derived
  - routine vs deep-audit parameters
  - exact interactive verifier / prover flow

- [ ] **Add cross-version / interoperability tests for the binary format**
  - golden binary payloads
  - backward rejection behavior
  - forward rejection behavior
  - independent decoder compatibility

## 6. Live Server Operational Checks

- [ ] **Add startup / self-check tasks for the live verified server**
  - assert model identity matches committed `R_W`
  - assert prefix caching is disabled in verified mode
  - assert sync mode / capture mode match intended verified settings
  - fail closed on operational mismatch instead of silently drifting

## 7. Performance and Audit Cost

Performance is no longer ahead of protocol completion, but still matters before release.

### 7.1 Inference Cost

- [ ] **Lower online inference overhead further if it remains worthwhile**
  - pinned host buffers for `a_i8`
  - flatter minimal-capture layout
  - less Python postprocessing
  - less PyO3 / Rust ingress copying
  - possibly a more native capture implementation if needed

- [ ] **Rebenchmark after each meaningful online-path change**

### 7.2 Audit Payload and Audit Open Cost

- [ ] **Inspect the binary payload by field**
- [ ] **Reduce audit payload structurally where possible**
  - remove unnecessary full vectors
  - preserve strongest full/deep-audit semantics
  - use sampling only for routine / probabilistic tiers

- [ ] **Add streaming / incremental audit open if it materially reduces peak memory or open time**

## 8. Benchmarks and Research Validation

- [ ] **Benchmark the final V5 path**
  - baseline
  - online overhead
  - commit time
  - audit open time
  - verify time
  - retained state per token
  - binary payload size

- [ ] **Benchmark the protocol-complete V6 path**
  - greedy path cost
  - sampled path cost
  - routine vs deep audit
  - exact-prefix premium
  - retained state / payload / verifier time

- [ ] **Measure retained-state memory and audit bandwidth**
- [ ] **Measure the attention corridor empirically**
- [ ] **Measure real audit-window storage costs**
- [ ] **Calibrate detection probabilities**
- [ ] **Benchmark batch verification**

## 9. Documentation and Release

These tasks must explicitly update the full protocol, not just the README narrative.

- [ ] **Run an explicit hidden-trust-assumptions review**
  - enumerate what is exact
  - enumerate what is approximate
  - enumerate what is statistical
  - enumerate what is still operationally trusted
  - ensure docs and claims match that list

- [ ] **Update the full protocol specification in the paper**
  - exact / statistical / approximate boundaries
  - routine vs deep audit
  - greedy vs sampled mode status
  - retained-state canonical path
  - verifier randomness model

- [ ] **Update the full protocol documentation in the README**
  - architecture
  - verifier model
  - exact / statistical / approximate guarantees
  - binary protocol path
  - routine vs deep audit

- [ ] **Update the article / writeup to match the full protocol**

- [ ] **Add a pipeline figure**
  - exact
  - statistical
  - approximate
  - trust-assumption boundaries

- [ ] **Document canonical semantics vs trust assumptions clearly**

- [ ] **Add fail-on-unknown versioning rules to the full protocol docs and code-facing docs**

- [ ] **Run the final kept-path benchmark suite**

- [ ] **Publish around V6 when the story is coherent**

## 10. Longer-Term Engineering

- [ ] **Client SDKs** — Python and TypeScript wrappers around the verifier
- [ ] **llama.cpp tracing plugin**
- [ ] **Fine-tuned models / LoRA support**
- [ ] **Formalization in Lean**
- [ ] **Receipt format specification**
- [ ] **API extensions** — OpenAI-compatible receipt field in response metadata

## 11. First Product

- [ ] **OpenAI-compatible proxy with receipts**
  - standard API
  - receipts in response headers / metadata
  - default audit window
  - challenge endpoint
  - simple verifier CLI

## 12. Ecosystem

- [ ] **Ritual plugin**
- [ ] **Bittensor plugin**
- [ ] **Gensyn plugin**
- [ ] **Inference marketplace**

---

## Critical Path

```text
Live canonical sampled serving
    + decode/output manifest binding + verifier replay
    ↓
Canonical V5 retained-state path
    ↓
Exact full-prefix mode
    ↓
Adversarial verifier hardening
    ↓
Conformance / challenge specification / interoperability
    ↓
Performance and audit-cost cleanup
    ↓
Final V5/V6 benchmarks
    ↓
Update the full protocol docs
    ↓
Publish
```

## End State

The target for V6 is:

- exact shell / bridge / final-token path
- exact preprocessing, model, decode, and output policy binding
- exact sampled replay as the default production mode
- greedy mode available as the `temperature=0` special case
- exact prefix mode available via deep audit
- only the attention interior remains approximate

After V6, the main remaining protocol gap is attention exactness. A V7 only makes sense if the attention story improves materially.
