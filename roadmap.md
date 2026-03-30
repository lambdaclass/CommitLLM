# Roadmap

This roadmap is a single ordered checklist for getting the codebase to match the final protocol described in the README, article, and paper.

Mark a task done only when the sentence is true in code, tests, docs, and operational behavior.

This file tracks the remaining and partial work only. Completed milestones live in [CHANGELOG.md](./CHANGELOG.md).

The kept canonical sampled path already exists in the live server. At this point the protocol is structurally complete; the remaining work is mostly verifier finalization, prover/server hardening, benchmarking, documentation, and ecosystem work.

Current release critical path:
1. settle the attention replay inputs in protocol form: quantization semantics (#1), bridge trust boundary (#2), committed full-causal KV transcript (#3)
2. measure the attention corridor with structurally complete replay (#4), then set evidence-based tau
3. bind the committed KV transcript back to real computation with provenance checks (#5)
4. freeze the canonical verifier and key-provenance story against that settled attention design
5. finish fail-closed prover/live-server hardening
6. clean the public product/docs surface for the first release — including audit tier separation (#12) and guarantee language (#13)
7. only then expand ecosystem and integrations

## Verifier Finalization

1. [ ] Make quantization semantics first-class protocol data, done when the protocol commits quantization metadata (quant family, scale derivation, block/group size, exact scale layout for K/V and attention output) and the replay path uses it faithfully. This is the first attention blocker: until the replay knows the real scale layout, corridor numbers and attention claims are not trustworthy. Current replay treats scale_a as scalar-per-tensor; if the GPU uses per-token or grouped scales, the protocol must stop pretending otherwise.
2. [ ] Fix the bridge trust boundary for attention, done when audited tokens/layers either commit and open actual bridge outputs (x_attn_i8, scale_x_attn) so the verifier can check them, or the protocol defines a canonical bridge so verifier and prover derive the same x_attn_i8. This must be decided before finalizing attention replay because otherwise QKV can diverge before attention starts.
3. [ ] Add committed KV transcript with kv_root covering full causal prefix, done when the protocol commits post-RoPE K/V for every (layer, position) including prompt and generated tokens under a separate kv_root, and audit openings can open the challenged token plus its causal K/V prefix for exact-attention replay. Current deep-prefix replay is structurally incomplete: `server.py` commits generated tokens only, `corridor.rs` and `canonical.rs` build kv_k/kv_v only from generated-prefix shell openings, so prompt/prefill KV is absent from replay. Partial short-term: add prompt KV capture to the measurement-only path without changing the production commit format.
4. [ ] Measure the attention corridor empirically, done when the FP16/BF16-versus-replay disagreement envelope is measured on real workloads and turned into an evidence-based acceptance threshold. This is now the main paper-critical measurement blocker. Blocked until the replay is structurally faithful: current corridor numbers are invalid because the replay cache is missing prompt/prefill KV entries (only generated-token shells are committed) and the quantization/bridge boundary is not yet a settled protocol object. Short-term unblock: add prompt KV capture to the measurement-only path. Long-term: depends on #1, #2, and #3.
5. [ ] Bind committed KV to real computation via provenance checks, done when the prover cannot commit arbitrary K/V — sampled or batched Freivalds checks verify that committed K/V are consistent with the committed Wk/Wv weights and the bridge inputs that feed them. Important for the final exact-attention protocol, but not a prerequisite for measuring the corridor once the replay inputs are structurally faithful.
6. [ ] Rewrite the verifier from scratch against the frozen final spec, done when the canonical verifier is the frozen trusted path, the legacy verifier is no longer needed for rollback, and the remaining verifier contract is explicit in tests and GPU evidence. Partial: the canonical verifier is now the trusted public path; legacy-vs-canonical parity tests, canonical frozen pass/reject fixtures, and full verifier-suite coverage (`canonical`, `v4_e2e`, `boundary_fuzz`, `cross_version`, `golden_conformance`, `hardening_gate`) exist; external challenge-response matching now lives in the client-side wrapper rather than inside the canonical trust boundary; phase-level tests cover all 9 phases (structural, embedding, specs, output policy, bridge, LM-head, deep prefix, tokenization, detokenization) plus Ctx::new() unit tests; token-0 attention replay closes the single-token computation gap; deep-prefix attention replay verifies multi-token attention for both prefix tokens and the opened token in the toy/reference path; routine-audit known-gap test documents that self-consistent fake `a` passes for token > 0 without deep-prefix. Remaining: fresh real-GPU confirmation on the current canonical code path through `scripts/modal/test_e2e_v4.py` and `scripts/modal/test_adversarial.py`. Production RoPE-aware deep-prefix replay already landed (see CHANGELOG); not yet confirmed on real GPU — tracked in #7.
7. [ ] Add explicit GPU smoke tests for the current attention-replay coverage, done when maintained GPU tests exercise token-0 attention replay and the kept deep-prefix attention path on the current canonical verifier. Partial boundary: this confirms the currently implemented coverage, but does not by itself close the broader attention protocol work in #1-#5.
8. [ ] Define verifier-key distribution and pinning, done when clients have one canonical procedure for trusted key provenance, key hash/version pinning, historical key lookup, and fail-closed handling of unknown verifier keys.
9. [ ] Handle verifier-key rotation explicitly, done when key versions are tracked and old receipts remain auditable with the correct historical key.
10. [ ] Use verifier-secret randomness in deep-audit batching, done when any deep-audit batching or compression uses verifier-only randomness that the prover cannot predict before commitment.
11. [ ] Run an explicit hidden-trust-assumptions review, done when exact, approximate, statistical, fail-closed, and out-of-scope trust assumptions are enumerated and checked against the code and claims.
12. [ ] Explicitly separate audit tiers in the protocol, done when the protocol defines distinct tiers — receipt-only, routine approximate, routine exact-attention, deep/full — with explicit coverage and cost for each, rather than blurring them.
13. [ ] Update guarantee language to match actual verification boundaries, done when the protocol docs clearly separate exact (shell/tail/bindings), statistical (sampled provenance checks), approximate (FP attention replay), and fail-closed (versioning/unsupported paths) guarantees.
14. [ ] Optional: canonical deterministic attention arithmetic, done when a canonical attention kernel or arithmetic specification removes the FP16-vs-replay corridor entirely. Without this, even perfect kv_root only fixes provenance, not FP precision mismatch.
15. [ ] Remove the legacy verifier after a soak period, done when `verify_v4_legacy` and any rollback-only verifier scaffolding are deleted and the canonical verifier is the only maintained trusted verification path.
16. [ ] Add at least one independent non-Rust verifier consumer, done when some non-Rust implementation can consume the golden vectors and reproduce expected verification results.

## Prover / Live Server / Ops

17. [ ] Add startup self-checks for the live verified server, done when startup fails closed on model-identity mismatch, bad capture settings, or unsupported verified-mode configuration.
18. [ ] Add supported-architecture detection and fail-closed behavior, done when unsupported model families and layouts are detected explicitly instead of being silently treated as decoder-only/Llama-style.
19. [ ] Define and enforce buffer-pressure behavior, done when overflow behavior for capture and audit-state buffers is explicit and tested.
20. [ ] Define and enforce audit-state retention behavior, done when retention TTL, cleanup, and purge behavior are explicit and tested.
21. [ ] Add health and readiness checks, done when the live server can report whether capture hooks, audit buffers, model identity, and verified-mode settings are all healthy and aligned.
22. [ ] Add audit-endpoint abuse protection, done when audit requests are rate-limited, bounds-checked, and protected against amplification via unreasonable token or layer requests.
23. [ ] Add explicit remote-GPU evidence for the EOS-trim regression path, done when a targeted Modal run triggers EOS-before-max_tokens on real GPU and verifies commit + audit + verify on the trimmed trailing-forward-pass path.

## Benchmarks / Performance

24. [ ] Define a stable benchmark protocol, done when workload corpus, model/settings, warmup policy, hardware class, remote environment, and reporting format are fixed for milestone comparisons.
25. [ ] Record benchmark baselines before and after each runtime-affecting milestone, done when every change to the live sampler, final-token boundary, deep-audit path, sync/capture behavior, or retained-state layout has a before/after benchmark entry.
26. [ ] Run periodic remote-GPU benchmark checkpoints, done when milestone comparisons are repeated on the same representative remote GPU class rather than relying only on local numbers.
27. [ ] Treat unexplained regressions as blockers, done when each regression is either explained or explicitly recorded as an accepted protocol-strengthening tradeoff before further work proceeds.
28. [ ] Benchmark exact deep-audit open cost, verifier cost, and payload cost, done when the chosen exact-prefix procedure has real benchmark numbers attached to it.
29. [ ] Benchmark the canonical routine-audit path, done when baseline, online overhead, commit time, audit-open time, verifier time, retained-state size, and binary payload size are all measured on the reference environment.
30. [ ] Benchmark the canonical exact-prefix path, done when greedy cost, sampled cost, routine-versus-exact-prefix audit cost, exact-prefix premium, retained-state size, payload size, and verifier time are all measured on the reference environment.
31. [ ] Inspect the binary payload by field, done when the receipt and audit format is measured field-by-field rather than only as a total size.
32. [ ] Measure retained-state memory and audit bandwidth, done when per-token storage cost and audit-transfer cost are quantified for the target deployments.
33. [ ] Measure real audit-window storage costs, done when RAM, NVMe, and network storage costs for realistic retention windows are quantified.
34. [ ] Calibrate routine-audit detection probabilities, done when routine-audit sampling rates are tied to measured or modeled detection probabilities rather than informal intuition.
35. [ ] Benchmark batch verification, done when verifier throughput for batched audits is measured rather than assumed.
36. [ ] Reduce audit payload structurally where possible, done when unnecessary full vectors are removed without weakening the strongest routine/deep-audit semantics.
37. [ ] Add streaming or incremental audit open if it materially helps, done when it measurably reduces peak memory or audit-open time and is benchmarked against the baseline.
38. [ ] Lower online inference overhead further if it remains worthwhile, done when the remaining online-cost work has either been completed and benchmarked or consciously stopped because the returns are too small.
39. [ ] Rebenchmark after each meaningful online-path change, done when every material serving-path change has fresh benchmark data attached to it.

## Docs / Publication

40. [ ] Remove protocol-generation naming from the product surface, done when maintained docs, APIs, CLI/help text, and user-facing flows stop framing the first release as “V4/V5” and instead use one canonical first-release product naming story. Internal binary/version fields may stay versioned, but public naming should not imply a prior shipped generation.
41. [ ] Make the implementation-status split explicit in the docs, done when the docs say they describe the target final protocol and the roadmap is clearly the implementation tracker.
42. [ ] Make the production verification surface explicitly binary-only, done when maintained docs and public verification APIs clearly mark the canonical binary format as the normative production path and any JSON path as debug/development-only compatibility tooling.
43. [ ] Freeze the verifier-facing report contract, done when `V4VerifyReport`-equivalent report fields, failure codes, coverage fields, and the client-wrapper contract are pinned by maintained golden tests and documented as a stable public verification surface across Rust and Python.
44. [ ] Ship a reference client verification flow, done when one canonical verifier-facing CLI or reference flow binds trusted verifier key, exact receipt/commitment, exact served output, verifier-issued challenge, audit bytes, and coverage interpretation in one maintained path.
45. [ ] Keep the paper, README, and article normative to the final protocol, done when they describe the destination protocol while benchmark and result sections remain factual about current implementation and measurements.
46. [ ] Update the full protocol documentation in the README, done when the README covers the final protocol framing, verification coverage, manifest surface, architecture, verifier model, and routine-versus-deep audit semantics.
47. [ ] Update the full protocol specification in the paper, done when the paper contains the exact/statistical/approximate/fail-closed boundary, four-spec structure, routine-versus-deep audit, retained-state path, and verifier randomness model.
48. [ ] Add an explicit input-verification procedure to the paper, README, and article, done when all three docs explain raw request reconstruction, tokenization under the committed input spec, and binding to the embedding/input path.
49. [ ] Add an explicit transcript-chain and anti-splice procedure to the paper, README, and article, done when all three docs describe prompt genesis, token/transcript chaining, and rejection of reorder/deletion/splice/cross-request mixups.
50. [ ] Make one canonical deep-audit procedure explicit in the paper, README, and article, done when the same exact full-prefix algorithm is described as the normative deep-audit mode everywhere.
51. [ ] Add a decode/output support matrix to the paper, done when the paper has one place that says which features are bound, replayed exactly, or fail-closed.
52. [ ] Use one canonical randomness story everywhere, done when the docs use the same wording and procedure for seed commitment, seed reveal, and per-token randomness derivation.
53. [ ] Document verified-mode deviations from stock vLLM, done when the paper and docs explain what verified mode changes operationally without overstating serving-path differences.
54. [ ] Document supported and unsupported architectures explicitly, done when the docs state the supported model/layout families and the implementation’s fail-closed behavior matches that support matrix.
55. [ ] Document canonical semantics versus trust assumptions clearly, done when the docs separate protocol semantics from what is simply trusted outside scope.
56. [ ] Add an explicit trusted-assumptions section to the paper, README, and article, done when standard cryptographic assumptions, verifier-key secrecy, side-channel assumptions, and verifier-correctness assumptions are all stated explicitly.
57. [ ] Add fail-on-unknown versioning rules to the docs and code-facing docs, done when unknown receipt or protocol versions are specified to fail closed rather than being interpreted leniently.
58. [ ] Document privacy implications of receipts and audits, done when the docs explain what receipts reveal, what audit openings reveal, and what transport/deployment practices are recommended.
59. [ ] Add a pipeline or boundary figure, done when at least one maintained doc has a figure that clearly marks exact, statistical, approximate, and trust-assumption boundaries.
60. [ ] Update the article or writeup to match the full protocol, done when the article has the final trust-boundary figure, four-spec explanation, guarantee boundary, and explicit unsupported-feature fail-closed rule.
61. [ ] Run the final kept-path benchmark suite, done when the final benchmark set is rerun after the implementation converges on the kept protocol path.
62. [ ] Explain benchmark methodology and regression gates in the paper and docs, done when the published docs say what hardware and workloads were used and how protocol-strengthening tradeoffs were benchmark-gated.
63. [ ] Publish only when the story is coherent, done when the code, benchmarks, docs, and claims all line up on the same kept path and unfinished claim-critical items are closed.

## Ecosystem / Future

64. [ ] Write a receipt-format specification, done when the receipt and audit binary format is documented independently of the implementation code.
65. [ ] Add API extensions for receipt fields, done when the serving API can carry receipt metadata in a stable, documented way.
66. [ ] Build client SDKs, done when at least Python and TypeScript wrappers exist for receipt handling and verification.
67. [ ] Build an OpenAI-compatible proxy with receipts, done when the proxy serves a standard API, returns receipt metadata, exposes a challenge endpoint, and ships with a simple verifier CLI.
68. [ ] Build a `llama.cpp` tracing plugin, done when the protocol can capture the required retained state from a `llama.cpp` serving path.
69. [ ] Add fine-tuned-model and LoRA support, done when adapters or merged checkpoints can be committed, identified, and verified under the model spec.
70. [ ] Add optional receipt or audit encryption for verifier-targeted privacy, done when the product can encrypt receipts or audit payloads to the verifier’s public key without changing the core claim-critical path.
71. [ ] Generalize the packed retained-state path beyond constant-width decoder models, done when the retained-state format is schema-driven and can carry per-layer shapes and layout metadata safely.
72. [ ] Build a Ritual plugin, done when Ritual can consume or expose VeriLM receipts and audits.
73. [ ] Build a Bittensor plugin, done when Bittensor-side inference can expose or consume VeriLM receipts and audits.
74. [ ] Build a Gensyn plugin, done when Gensyn-side inference can expose or consume VeriLM receipts and audits.
75. [ ] Build an inference-marketplace integration, done when marketplace providers can attach receipts and clients can challenge or verify them through the market surface.
76. [ ] Formalize the protocol in Lean, done when the core verification claims have a maintained Lean formalization.
