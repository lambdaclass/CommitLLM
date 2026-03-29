# Roadmap

This roadmap is a single ordered checklist for getting the codebase to match the final protocol described in the README, article, and paper.

Mark a task done only when the sentence is true in code, tests, docs, and operational behavior.

This file tracks the remaining and partial work only. Completed milestones live in [CHANGELOG.md](./CHANGELOG.md).

The kept canonical sampled path already exists in the live server. At this point the protocol is structurally complete; the remaining work is mostly verifier finalization, prover/server hardening, benchmarking, documentation, and ecosystem work.

Current release critical path:
1. freeze the canonical verifier and key-provenance story
2. close the remaining attention story and measure the attention corridor
3. finish fail-closed prover/live-server hardening
4. clean the public product/docs surface for the first release
5. only then expand ecosystem and integrations

## Verifier Finalization

1. [ ] Rewrite the verifier from scratch against the frozen final spec, done when the canonical verifier is the frozen trusted path, the legacy verifier is no longer needed for rollback, and the remaining verifier contract is explicit in tests and GPU evidence. Partial: the canonical verifier is now the trusted public path; legacy-vs-canonical parity tests, canonical frozen pass/reject fixtures, and full verifier-suite coverage (`canonical`, `v4_e2e`, `boundary_fuzz`, `cross_version`, `golden_conformance`, `hardening_gate`) exist; external challenge-response matching now lives in the client-side wrapper rather than inside the canonical trust boundary; phase-level tests cover all 9 phases (structural, embedding, specs, output policy, bridge, LM-head, deep prefix, tokenization, detokenization) plus Ctx::new() unit tests; token-0 attention replay closes the single-token computation gap; deep-prefix attention replay verifies multi-token attention for both prefix tokens and the opened token in the toy/reference path; routine-audit known-gap test documents that self-consistent fake `a` passes for token > 0 without deep-prefix. Remaining: fresh real-GPU confirmation on the current canonical code path through `scripts/modal/test_e2e_v4.py` and `scripts/modal/test_adversarial.py`. Not fully done for production attention replay: RoPE-aware deep-prefix replay is still tracked separately in `#71`.
71. [ ] Make deep-prefix attention replay RoPE-aware for production models, done when the deep-prefix attention replay path dequantizes Q/K accumulators using weight and activation scales, applies RoPE at each position via `verilm_core::rope::apply_rope_{q,k}`, requantizes, and replays attention in the correct post-RoPE space. Current state: toy-model replay (raw requantize, no RoPE) is implemented and tested; production models with RoPE still use the approximate path.
75. [ ] Add explicit GPU smoke tests for the current attention-replay coverage, done when maintained GPU tests exercise token-0 attention replay and the kept deep-prefix attention path on the current canonical verifier. Partial boundary: this confirms the currently implemented coverage, but does not by itself close the production RoPE-aware path tracked in `#71`.
66. [ ] Define verifier-key distribution and pinning, done when clients have one canonical procedure for trusted key provenance, key hash/version pinning, historical key lookup, and fail-closed handling of unknown verifier keys.
6. [ ] Handle verifier-key rotation explicitly, done when key versions are tracked and old receipts remain auditable with the correct historical key.
5. [ ] Use verifier-secret randomness in deep-audit batching, done when any deep-audit batching or compression uses verifier-only randomness that the prover cannot predict before commitment.
7. [ ] Run an explicit hidden-trust-assumptions review, done when exact, approximate, statistical, fail-closed, and out-of-scope trust assumptions are enumerated and checked against the code and claims.
72. [ ] Add bounded KV-cache commitment for arbitrary-token attention replay, done when the protocol supports committed/openable prior-token KV data so that any single token's attention can be replayed in routine audits, not just token 0 or deep-prefix co-opened tokens. This is a protocol extension requiring new wire format fields and prover-side KV provenance.
65. [ ] Remove the legacy verifier after a soak period, done when `verify_v4_legacy` and any rollback-only verifier scaffolding are deleted and the canonical verifier is the only maintained trusted verification path.
3. [ ] Add at least one independent non-Rust verifier consumer, done when some non-Rust implementation can consume the golden vectors and reproduce expected verification results.

## Prover / Live Server / Ops

8. [ ] Add startup self-checks for the live verified server, done when startup fails closed on model-identity mismatch, bad capture settings, or unsupported verified-mode configuration.
9. [ ] Add supported-architecture detection and fail-closed behavior, done when unsupported model families and layouts are detected explicitly instead of being silently treated as decoder-only/Llama-style.
10. [ ] Define and enforce buffer-pressure behavior, done when overflow behavior for capture and audit-state buffers is explicit and tested.
11. [ ] Define and enforce audit-state retention behavior, done when retention TTL, cleanup, and purge behavior are explicit and tested.
12. [ ] Add health and readiness checks, done when the live server can report whether capture hooks, audit buffers, model identity, and verified-mode settings are all healthy and aligned.
13. [ ] Add audit-endpoint abuse protection, done when audit requests are rate-limited, bounds-checked, and protected against amplification via unreasonable token or layer requests.
14. [ ] Add explicit remote-GPU evidence for the EOS-trim regression path, done when a targeted Modal run triggers EOS-before-max_tokens on real GPU and verifies commit + audit + verify on the trimmed trailing-forward-pass path.

## Benchmarks / Performance

16. [ ] Define a stable benchmark protocol, done when workload corpus, model/settings, warmup policy, hardware class, remote environment, and reporting format are fixed for milestone comparisons.
28. [ ] Measure the attention corridor empirically, done when the FP16/BF16-versus-replay disagreement envelope is measured on real workloads and turned into an evidence-based acceptance threshold.
17. [ ] Record benchmark baselines before and after each runtime-affecting milestone, done when every change to the live sampler, final-token boundary, deep-audit path, sync/capture behavior, or retained-state layout has a before/after benchmark entry.
18. [ ] Run periodic remote-GPU benchmark checkpoints, done when milestone comparisons are repeated on the same representative remote GPU class rather than relying only on local numbers.
19. [ ] Treat unexplained regressions as blockers, done when each regression is either explained or explicitly recorded as an accepted protocol-strengthening tradeoff before further work proceeds.
15. [ ] Benchmark exact deep-audit open cost, verifier cost, and payload cost, done when the chosen exact-prefix procedure has real benchmark numbers attached to it.
25. [ ] Benchmark the canonical routine-audit path, done when baseline, online overhead, commit time, audit-open time, verifier time, retained-state size, and binary payload size are all measured on the reference environment.
26. [ ] Benchmark the canonical exact-prefix path, done when greedy cost, sampled cost, routine-versus-exact-prefix audit cost, exact-prefix premium, retained-state size, payload size, and verifier time are all measured on the reference environment.
22. [ ] Inspect the binary payload by field, done when the receipt and audit format is measured field-by-field rather than only as a total size.
27. [ ] Measure retained-state memory and audit bandwidth, done when per-token storage cost and audit-transfer cost are quantified for the target deployments.
29. [ ] Measure real audit-window storage costs, done when RAM, NVMe, and network storage costs for realistic retention windows are quantified.
30. [ ] Calibrate routine-audit detection probabilities, done when routine-audit sampling rates are tied to measured or modeled detection probabilities rather than informal intuition.
31. [ ] Benchmark batch verification, done when verifier throughput for batched audits is measured rather than assumed.
23. [ ] Reduce audit payload structurally where possible, done when unnecessary full vectors are removed without weakening the strongest routine/deep-audit semantics.
24. [ ] Add streaming or incremental audit open if it materially helps, done when it measurably reduces peak memory or audit-open time and is benchmarked against the baseline.
20. [ ] Lower online inference overhead further if it remains worthwhile, done when the remaining online-cost work has either been completed and benchmarked or consciously stopped because the returns are too small.
21. [ ] Rebenchmark after each meaningful online-path change, done when every material serving-path change has fresh benchmark data attached to it.

## Docs / Publication

67. [ ] Remove protocol-generation naming from the product surface, done when maintained docs, APIs, CLI/help text, and user-facing flows stop framing the first release as “V4/V5” and instead use one canonical first-release product naming story. Internal binary/version fields may stay versioned, but public naming should not imply a prior shipped generation.
39. [ ] Make the implementation-status split explicit in the docs, done when the docs say they describe the target final protocol and the roadmap is clearly the implementation tracker.
70. [ ] Make the production verification surface explicitly binary-only, done when maintained docs and public verification APIs clearly mark the canonical binary format as the normative production path and any JSON path as debug/development-only compatibility tooling.
68. [ ] Freeze the verifier-facing report contract, done when `V4VerifyReport`-equivalent report fields, failure codes, coverage fields, and the client-wrapper contract are pinned by maintained golden tests and documented as a stable public verification surface across Rust and Python.
69. [ ] Ship a reference client verification flow, done when one canonical verifier-facing CLI or reference flow binds trusted verifier key, exact receipt/commitment, exact served output, verifier-issued challenge, audit bytes, and coverage interpretation in one maintained path.
32. [ ] Keep the paper, README, and article normative to the final protocol, done when they describe the destination protocol while benchmark and result sections remain factual about current implementation and measurements.
41. [ ] Update the full protocol documentation in the README, done when the README covers the final protocol framing, verification coverage, manifest surface, architecture, verifier model, and routine-versus-deep audit semantics.
33. [ ] Update the full protocol specification in the paper, done when the paper contains the exact/statistical/approximate/fail-closed boundary, four-spec structure, routine-versus-deep audit, retained-state path, and verifier randomness model.
34. [ ] Add an explicit input-verification procedure to the paper, README, and article, done when all three docs explain raw request reconstruction, tokenization under the committed input spec, and binding to the embedding/input path.
35. [ ] Add an explicit transcript-chain and anti-splice procedure to the paper, README, and article, done when all three docs describe prompt genesis, token/transcript chaining, and rejection of reorder/deletion/splice/cross-request mixups.
36. [ ] Make one canonical deep-audit procedure explicit in the paper, README, and article, done when the same exact full-prefix algorithm is described as the normative deep-audit mode everywhere.
37. [ ] Add a decode/output support matrix to the paper, done when the paper has one place that says which features are bound, replayed exactly, or fail-closed.
38. [ ] Use one canonical randomness story everywhere, done when the docs use the same wording and procedure for seed commitment, seed reveal, and per-token randomness derivation.
40. [ ] Document verified-mode deviations from stock vLLM, done when the paper and docs explain what verified mode changes operationally without overstating serving-path differences.
43. [ ] Document supported and unsupported architectures explicitly, done when the docs state the supported model/layout families and the implementation’s fail-closed behavior matches that support matrix.
46. [ ] Document canonical semantics versus trust assumptions clearly, done when the docs separate protocol semantics from what is simply trusted outside scope.
47. [ ] Add an explicit trusted-assumptions section to the paper, README, and article, done when standard cryptographic assumptions, verifier-key secrecy, side-channel assumptions, and verifier-correctness assumptions are all stated explicitly.
48. [ ] Add fail-on-unknown versioning rules to the docs and code-facing docs, done when unknown receipt or protocol versions are specified to fail closed rather than being interpreted leniently.
42. [ ] Document privacy implications of receipts and audits, done when the docs explain what receipts reveal, what audit openings reveal, and what transport/deployment practices are recommended.
45. [ ] Add a pipeline or boundary figure, done when at least one maintained doc has a figure that clearly marks exact, statistical, approximate, and trust-assumption boundaries.
44. [ ] Update the article or writeup to match the full protocol, done when the article has the final trust-boundary figure, four-spec explanation, guarantee boundary, and explicit unsupported-feature fail-closed rule.
49. [ ] Run the final kept-path benchmark suite, done when the final benchmark set is rerun after the implementation converges on the kept protocol path.
50. [ ] Explain benchmark methodology and regression gates in the paper and docs, done when the published docs say what hardware and workloads were used and how protocol-strengthening tradeoffs were benchmark-gated.
51. [ ] Publish only when the story is coherent, done when the code, benchmarks, docs, and claims all line up on the same kept path and unfinished claim-critical items are closed.

## Ecosystem / Future

58. [ ] Write a receipt-format specification, done when the receipt and audit binary format is documented independently of the implementation code.
59. [ ] Add API extensions for receipt fields, done when the serving API can carry receipt metadata in a stable, documented way.
52. [ ] Build client SDKs, done when at least Python and TypeScript wrappers exist for receipt handling and verification.
60. [ ] Build an OpenAI-compatible proxy with receipts, done when the proxy serves a standard API, returns receipt metadata, exposes a challenge endpoint, and ships with a simple verifier CLI.
53. [ ] Build a `llama.cpp` tracing plugin, done when the protocol can capture the required retained state from a `llama.cpp` serving path.
54. [ ] Add fine-tuned-model and LoRA support, done when adapters or merged checkpoints can be committed, identified, and verified under the model spec.
55. [ ] Add optional receipt or audit encryption for verifier-targeted privacy, done when the product can encrypt receipts or audit payloads to the verifier’s public key without changing the core claim-critical path.
56. [ ] Generalize the packed retained-state path beyond constant-width decoder models, done when the retained-state format is schema-driven and can carry per-layer shapes and layout metadata safely.
61. [ ] Build a Ritual plugin, done when Ritual can consume or expose VeriLM receipts and audits.
62. [ ] Build a Bittensor plugin, done when Bittensor-side inference can expose or consume VeriLM receipts and audits.
63. [ ] Build a Gensyn plugin, done when Gensyn-side inference can expose or consume VeriLM receipts and audits.
64. [ ] Build an inference-marketplace integration, done when marketplace providers can attach receipts and clients can challenge or verify them through the market surface.
57. [ ] Formalize the protocol in Lean, done when the core verification claims have a maintained Lean formalization.
