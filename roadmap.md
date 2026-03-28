# Roadmap

This roadmap is a single ordered checklist for getting the codebase to match the final protocol described in the README, article, and paper.

Mark a task done only when the sentence is true in code, tests, docs, and operational behavior.

This file now tracks the remaining and partial work only. Completed protocol milestones live in [CHANGELOG.md](./CHANGELOG.md).

The narrow core canonical sampled path already exists in the live server. At this point, the kept canonical protocol is structurally complete; most remaining tasks are semantic/spec edge-case closure, optimization, hardening, conformance, benchmarking, and documentation work.

The following blockers are already closed and are therefore omitted from the remaining checklist: sync-equivalence between `global` and `event` modes, capture reliability for EOS/counter/prefix-caching edge cases, the deterministic capture-mismatch investigation, prompt-hash binding in the canonical verifier path, and prompt/generation-boundary count binding in the commitment/opening path.

1. [ ] Use verifier-secret randomness in deep-audit batching, done when any deep-audit batching or compression uses verifier-only randomness that the prover cannot predict before commitment.
2. [ ] Benchmark exact deep-audit open cost, verifier cost, and payload cost, done when the chosen exact-prefix procedure has real benchmark numbers attached to it.
3. [ ] Keep only irreducible retained state long term, done when the retained-state schema excludes derivable intermediates and preserves only what the verifier cannot reconstruct. Partial: RetainedLayerState now explicitly documents which fields are irreducible (a, scale_a) vs derivable with weight access (scale_x_attn, scale_x_ffn, scale_h). Actual removal of derivable scales blocked on #14 (verifier rewrite with weight access).
4. [ ] Add a dedicated adversarial verifier-hardening gate to the execution plan, done when no strong claims or final benchmarks can land without passing a specific adversarial-hardening stage.
5. [x] Build a tamper corpus for every verifier boundary, done when shell openings, retained-state leaves, IO chain, embedding proofs, manifest/spec fields, prompt hashes, seed commitments, LM-head/final-token checks, decode replay, stop policy, prefix openings, and malformed binary payloads all have adversarial cases.
6. [x] Add cross-proof attacks to the tamper corpus, done when receipt/proof components from different responses can be mixed in tests and are rejected.
7. [x] Add splice attacks to the tamper corpus, done when prefix, transcript, or request-chain splices are tested and rejected.
8. [ ] Add boundary-condition fuzzing, done when EOS-shortened requests, long/short prompt-output combinations, weird challenge tiers, malformed serialization, and unknown versions are fuzzed and fail correctly.
9. [x] Require adversarial tests to fail for the right reason, done when hardening tests assert stable, specific verifier failure reasons instead of generic failure.
10. [x] Define a verification failure taxonomy, done when cryptographic failures, semantic failures, operational/configuration failures, and approximate/statistical outcomes are explicitly classified and reported differently.
11. [ ] Add structured audit-failure reporting, done when verifier output identifies the failed check and the relevant token/layer/proof component in a stable machine-consumable form.
12. [ ] Define partial-audit semantics, done when routine-audit and deep-audit outcomes are reported distinctly and partial statistical coverage cannot be mistaken for full exact success.
13. [ ] Add cross-version tests for the binary format, done when backward rejection, forward rejection, and decoder compatibility are all tested against golden payloads. Partial: golden_conformance.rs covers V4 roundtrip, unknown magic rejection, truncated payload rejection, cross-format rejection, and version field preservation. Still missing: checked-in frozen binary fixtures and a multi-version compatibility matrix.
14. [ ] Rewrite the verifier from scratch against the frozen final spec, done when a new minimal linear verifier consumes the canonical receipt/audit format, passes the golden vectors, and replaces transitional branches in the trusted verification path.
15. [ ] Add at least one independent non-Rust verifier consumer, done when some non-Rust implementation can consume the golden vectors and reproduce expected verification results.
16. [ ] Add startup self-checks for the live verified server, done when startup fails closed on model-identity mismatch, bad capture settings, or unsupported verified-mode configuration.
17. [ ] Add supported-architecture detection and fail-closed behavior, done when unsupported model families and layouts are detected explicitly instead of being silently treated as decoder-only/Llama-style.
18. [ ] Define and enforce buffer-pressure behavior, done when overflow behavior for capture and audit-state buffers is explicit and tested.
19. [ ] Define and enforce audit-state retention behavior, done when retention TTL, cleanup, and purge behavior are explicit and tested.
20. [ ] Handle verifier-key rotation explicitly, done when key versions are tracked and old receipts remain auditable with the correct historical key.
21. [ ] Add health and readiness checks, done when the live server can report whether capture hooks, audit buffers, model identity, and verified-mode settings are all healthy and aligned.
22. [ ] Add audit-endpoint abuse protection, done when audit requests are rate-limited, bounds-checked, and protected against amplification via unreasonable token or layer requests.
23. [ ] Define a stable benchmark protocol, done when workload corpus, model/settings, warmup policy, hardware class, remote environment, and reporting format are fixed for milestone comparisons.
24. [ ] Record benchmark baselines before and after each runtime-affecting milestone, done when every change to the live sampler, final-token boundary, deep-audit path, sync/capture behavior, or retained-state layout has a before/after benchmark entry.
25. [ ] Run periodic remote-GPU benchmark checkpoints, done when milestone comparisons are repeated on the same representative remote GPU class rather than relying only on local numbers.
26. [ ] Treat unexplained regressions as blockers, done when each regression is either explained or explicitly recorded as an accepted protocol-strengthening tradeoff before further work proceeds.
27. [ ] Lower online inference overhead further if it remains worthwhile, done when the remaining online-cost work has either been completed and benchmarked or consciously stopped because the returns are too small.
28. [ ] Rebenchmark after each meaningful online-path change, done when every material serving-path change has fresh benchmark data attached to it.
29. [ ] Inspect the binary payload by field, done when the receipt and audit format is measured field-by-field rather than only as a total size.
30. [ ] Reduce audit payload structurally where possible, done when unnecessary full vectors are removed without weakening the strongest routine/deep-audit semantics.
31. [ ] Add streaming or incremental audit open if it materially helps, done when it measurably reduces peak memory or audit-open time and is benchmarked against the baseline.
32. [ ] Benchmark the canonical routine-audit path, done when baseline, online overhead, commit time, audit-open time, verifier time, retained-state size, and binary payload size are all measured on the reference environment.
33. [ ] Benchmark the canonical exact-prefix path, done when greedy cost, sampled cost, routine-versus-exact-prefix audit cost, exact-prefix premium, retained-state size, payload size, and verifier time are all measured on the reference environment.
34. [ ] Measure retained-state memory and audit bandwidth, done when per-token storage cost and audit-transfer cost are quantified for the target deployments.
35. [ ] Measure the attention corridor empirically, done when the FP16/BF16-versus-replay disagreement envelope is measured on real workloads and turned into an evidence-based acceptance threshold.
36. [ ] Measure real audit-window storage costs, done when RAM, NVMe, and network storage costs for realistic retention windows are quantified.
37. [ ] Calibrate routine-audit detection probabilities, done when routine-audit sampling rates are tied to measured or modeled detection probabilities rather than informal intuition.
38. [ ] Benchmark batch verification, done when verifier throughput for batched audits is measured rather than assumed.
39. [ ] Run an explicit hidden-trust-assumptions review, done when exact, approximate, statistical, fail-closed, and out-of-scope trust assumptions are enumerated and checked against the code and claims.
40. [ ] Keep the paper, README, and article normative to the final protocol, done when they describe the destination protocol while benchmark and result sections remain factual about current implementation and measurements.
41. [ ] Update the full protocol specification in the paper, done when the paper contains the exact/statistical/approximate/fail-closed boundary, four-spec structure, routine-versus-deep audit, retained-state path, and verifier randomness model.
42. [ ] Add an explicit input-verification procedure to the paper, README, and article, done when all three docs explain raw request reconstruction, tokenization under the committed input spec, and binding to the embedding/input path.
43. [ ] Add an explicit transcript-chain and anti-splice procedure to the paper, README, and article, done when all three docs describe prompt genesis, token/transcript chaining, and rejection of reorder/deletion/splice/cross-request mixups.
44. [ ] Make one canonical deep-audit procedure explicit in the paper, README, and article, done when the same exact full-prefix algorithm is described as the normative deep-audit mode everywhere.
45. [ ] Add a decode/output support matrix to the paper, done when the paper has one place that says which features are bound, replayed exactly, or fail-closed.
46. [ ] Use one canonical randomness story everywhere, done when the docs use the same wording and procedure for seed commitment, seed reveal, and per-token randomness derivation.
47. [ ] Make the implementation-status split explicit in the docs, done when the docs say they describe the target final protocol and the roadmap is clearly the implementation tracker.
48. [ ] Document verified-mode deviations from stock vLLM, done when the paper and docs explain what verified mode changes operationally without overstating serving-path differences.
49. [ ] Update the full protocol documentation in the README, done when the README covers the final protocol framing, verification coverage, manifest surface, architecture, verifier model, and routine-versus-deep audit semantics.
50. [ ] Document privacy implications of receipts and audits, done when the docs explain what receipts reveal, what audit openings reveal, and what transport/deployment practices are recommended.
51. [ ] Document supported and unsupported architectures explicitly, done when the docs state the supported model/layout families and the implementation’s fail-closed behavior matches that support matrix.
52. [ ] Update the article or writeup to match the full protocol, done when the article has the final trust-boundary figure, four-spec explanation, guarantee boundary, and explicit unsupported-feature fail-closed rule.
53. [ ] Add a pipeline or boundary figure, done when at least one maintained doc has a figure that clearly marks exact, statistical, approximate, and trust-assumption boundaries.
54. [ ] Document canonical semantics versus trust assumptions clearly, done when the docs separate protocol semantics from what is simply trusted outside scope.
55. [ ] Add an explicit trusted-assumptions section to the paper, README, and article, done when standard cryptographic assumptions, verifier-key secrecy, side-channel assumptions, and verifier-correctness assumptions are all stated explicitly.
56. [ ] Add fail-on-unknown versioning rules to the docs and code-facing docs, done when unknown receipt or protocol versions are specified to fail closed rather than being interpreted leniently.
57. [ ] Run the final kept-path benchmark suite, done when the final benchmark set is rerun after the implementation converges on the kept protocol path.
58. [ ] Explain benchmark methodology and regression gates in the paper and docs, done when the published docs say what hardware and workloads were used and how protocol-strengthening tradeoffs were benchmark-gated.
59. [ ] Publish only when the story is coherent, done when the code, benchmarks, docs, and claims all line up on the same kept path and unfinished claim-critical items are closed.
60. [ ] Build client SDKs, done when at least Python and TypeScript wrappers exist for receipt handling and verification.
61. [ ] Build a `llama.cpp` tracing plugin, done when the protocol can capture the required retained state from a `llama.cpp` serving path.
62. [ ] Add fine-tuned-model and LoRA support, done when adapters or merged checkpoints can be committed, identified, and verified under the model spec.
63. [ ] Add optional receipt or audit encryption for verifier-targeted privacy, done when the product can encrypt receipts or audit payloads to the verifier’s public key without changing the core claim-critical path.
64. [ ] Generalize the packed retained-state path beyond constant-width decoder models, done when the retained-state format is schema-driven and can carry per-layer shapes and layout metadata safely.
65. [ ] Formalize the protocol in Lean, done when the core verification claims have a maintained Lean formalization.
66. [ ] Write a receipt-format specification, done when the receipt and audit binary format is documented independently of the implementation code.
67. [ ] Add API extensions for receipt fields, done when the serving API can carry receipt metadata in a stable, documented way.
68. [ ] Build an OpenAI-compatible proxy with receipts, done when the proxy serves a standard API, returns receipt metadata, exposes a challenge endpoint, and ships with a simple verifier CLI.
69. [ ] Build a Ritual plugin, done when Ritual can consume or expose VeriLM receipts and audits.
70. [ ] Build a Bittensor plugin, done when Bittensor-side inference can expose or consume VeriLM receipts and audits.
71. [ ] Build a Gensyn plugin, done when Gensyn-side inference can expose or consume VeriLM receipts and audits.
72. [ ] Build an inference-marketplace integration, done when marketplace providers can attach receipts and clients can challenge or verify them through the market surface.
