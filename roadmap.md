# Roadmap

This roadmap is a single ordered checklist for getting the codebase to match the final protocol described in the README, article, and paper.

Mark a task done only when the sentence is true in code, tests, docs, and operational behavior.

The narrow core canonical sampled path already exists in the live server. Completed dependencies remain checked below for context, and unchecked items are the remaining unfinished work.

The following blockers are already closed and are therefore omitted from the remaining checklist: sync-equivalence between `global` and `event` modes, capture reliability for EOS/counter/prefix-caching edge cases, and the deterministic capture-mismatch investigation.

1. [x] Add `MatrixType::LM_HEAD`, done when the codebase treats the LM head as an explicit Freivalds-checked matrix family instead of an implicit special case.
2. [x] Extend key generation with verifier-secret LM-head Freivalds material, done when keygen samples `r_lm_head`, precomputes `v_lm_head = r^T W_lm_head mod p`, and stores the resulting verifier-side LM-head check material in the verifier key.
3. [x] Add verifier-side LM-head Freivalds verification, done when the verifier checks the LM head with Freivalds instead of relying only on exact recomputation.
4. [x] Keep exact logits replay after LM-head Freivalds lands, done when LM-head Freivalds binds the linear map and exact logits replay still determines the verified token selection path.
5. [x] Capture the pre-final-norm residual in every verified live path, done when all verified-serving modes that claim the exact final-token tail actually commit the captured pre-final-norm residual.
6. [x] Start exact final-token replay from the captured pre-final-norm residual, done when the verifier no longer derives the final token from shell-replayed hidden state after approximate attention replay.
7. [x] Fail closed when the exact final-token boundary state is missing or inconsistent, done when any response that lacks the required captured final boundary is rejected rather than partially verified.
8. [x] Add literal `input_spec_hash`, `model_spec_hash`, `decode_spec_hash`, and `output_spec_hash` fields to the receipt format, done when the receipt schema and binary format carry those four spec hashes explicitly.
9. [x] Commit the four literal spec hashes in the prover path, done when the prover computes and emits all four spec hashes for every committed response.
10. [x] Verify the four literal spec hashes in the verifier path, done when the verifier recomputes and checks all four spec hashes instead of only checking a monolithic manifest hash.
11. [ ] Bind tokenizer identity and normalization semantics in the input spec, done when tokenizer files and normalization behavior are committed as part of the input spec rather than described informally.
12. [x] Bind chat-template semantics in the input spec, done when the chat template or its canonical hash/policy is committed and checked as part of input semantics.
13. [x] Bind BOS/EOS preprocessing policy in the input spec, done when add-BOS/add-EOS behavior is committed and verified rather than assumed.
14. [ ] Bind truncation and padding policy in the input spec, done when truncation side/method and padding semantics are committed and verified.
15. [x] Bind special-token handling in the input spec, done when special-token insertion, stripping, or preservation semantics are committed and verified.
16. [x] Bind system-prompt semantics in the input spec, done when the committed input spec fully captures the system-prompt behavior that affects tokenization and model input.
17. [x] Bind public model identity `R_W` in the model spec, done when the model spec explicitly commits the intended published checkpoint identity.
18. [ ] Bind quantization identity and configuration in the model spec, done when quantization family, quantization config, and quantization hash or equivalent committed identity are all part of the model spec.
19. [ ] Bind quantization scale-derivation semantics in the model spec, done when the verifier can tell which scale derivation rules and layout rules are intended for the committed quantization scheme.
20. [ ] Bind block size and layout parameters for blockwise quantization schemes in the model spec, done when blockwise quantization cannot drift silently under the same high-level quantization label.
21. [x] Bind adapter, LoRA, or merged-checkpoint identity in the model spec, done when uncommitted adapters cannot change outputs without changing the committed model spec.
22. [x] Bind RoPE base and scaling configuration in the model spec, done when `rope_theta`, scaling family, and any rope-scaling knobs that affect outputs are committed and verified.
23. [x] Bind RMSNorm epsilon in the model spec, done when `rmsnorm_eps` is explicitly committed and verified instead of living only verifier-side.
24. [ ] Bind any other architecture-affecting model knobs in the model spec, done when no output-affecting model parameter remains verifier-side-only or implicit.
25. [x] Bind sampler identity and version in the decode spec, done when the decode spec identifies the exact sampler semantics that the verifier replays.
26. [ ] Bind decode mode choice in the decode spec, done when greedy versus sampled mode and any equivalent mode distinctions are committed explicitly.
27. [x] Bind temperature, top-k, and top-p in the decode spec, done when those core decode knobs are committed explicitly under the decode spec.
28. [x] Bind repetition, frequency, and presence penalties in the decode spec, done when those penalties are committed explicitly rather than only carried opportunistically in the manifest.
29. [ ] Bind logit bias and bad-word masking in the decode spec, done when those token-level logit modifications are committed explicitly.
30. [x] Bind guided decoding or grammar constraints in the decode spec, done when grammar/schema constraints are committed explicitly.
31. [ ] Bind tie-breaking rules in the decode spec, done when any ambiguity in equal-logit or equal-probability selection is committed explicitly.
32. [ ] Bind transcript-randomness derivation in the decode spec, done when the decode spec commits exactly how per-request and per-token randomness are derived.
33. [x] Bind EOS policy and ignore-EOS behavior in the output spec, done when end-of-sequence semantics are committed explicitly in the output spec.
34. [x] Bind stop strings and min/max stopping rules in the output spec, done when all stopping conditions that can alter final output are committed explicitly.
35. [ ] Bind special-token stripping in the output spec, done when output-token cleanup behavior is committed explicitly.
36. [ ] Bind detokenization, cleanup, and whitespace-normalization semantics in the output spec, done when output text formatting cannot drift without changing the committed output spec.
37. [ ] Reconstruct token IDs from the raw request under the committed input spec, done when the verifier can deterministically reproduce the exact token sequence from the raw request and committed input spec.
38. [ ] Check reconstructed token IDs against the committed input path, done when the verifier explicitly checks that reconstructed token IDs match the committed request/input semantics rather than trusting a claimed token list.
39. [ ] Check reconstructed token IDs against the embedding path, done when the verifier explicitly checks that token reconstruction and embedding lookup are consistent with the committed token IDs and embedding proof path.
40. [ ] Bind prompt genesis in the transcript chain, done when the transcript has one explicit request genesis value that all later transcript chaining derives from.
41. [ ] Bind token IDs into the transcript chain, done when every emitted token is committed in a way that later audit openings cannot reorder or substitute.
42. [ ] Bind prior transcript state into the transcript chain, done when every transcript step commits to the prior transcript state so deletion and splice attacks fail.
43. [ ] Bind cross-request separation into the transcript chain, done when receipts and openings from different requests cannot be mixed without detection.
44. [ ] Add verifier-side checks for the full committed input/model surface, done when the verifier explicitly checks the committed input/model semantics instead of only checking a hash digest.
45. [ ] Choose one canonical exact-prefix deep-audit algorithm, done when there is a single normative exact full-prefix procedure rather than multiple competing “deep audit” descriptions.
46. [ ] Implement the canonical exact-prefix deep-audit algorithm in code, done when the verifier can actually run the chosen full-prefix algorithm and upgrade prefix anchoring from statistical to exact.
47. [ ] Use verifier-secret randomness in deep-audit batching, done when any deep-audit batching or compression uses verifier-only randomness that the prover cannot predict before commitment.
48. [ ] Add tests for the canonical deep-audit algorithm, done when exact full-prefix behavior is covered by tests instead of only described in docs.
49. [ ] Benchmark exact deep-audit open cost, verifier cost, and payload cost, done when the chosen exact-prefix procedure has real benchmark numbers attached to it.
50. [x] Keep the live canonical sampled path replayable end to end, done when live serving, receipt contents, audit openings, and verifier replay all agree on sampled token selection under the committed decode spec.
51. [x] Generate a fresh `batch_seed` per request in the live path, done when every sampled request uses a new per-request seed generated at response time.
52. [x] Commit `seed_commitment = H(batch_seed)` in the receipt, done when every sampled receipt carries the committed per-request seed commitment.
53. [x] Reveal the committed seed on audit, done when sampled audits always include the exact seed needed for verifier replay and seed-commitment checking.
54. [x] Derive per-token seeds deterministically from the committed per-request seed, done when per-token randomness derivation is fixed, tested, and verifier-replayable.
55. [x] Apply all supported live decode semantics under VeriLM control, done when no supported decode feature depends on incidental stock-vLLM behavior outside the committed canonical path.
56. [ ] Add sampled end-to-end tests through the live HTTP/server path, done when honest sampled pass, wrong seed, wrong manifest, wrong sampled token, and cross-request splice failures are all covered by tests.
57. [ ] Implement verifier-side replay for every supported decode feature, done when every decode feature that the live server supports in verified mode is replayed by the verifier rather than merely tolerated.
58. [ ] Implement verifier-side replay for every supported output feature, done when every output-policy feature that the live server supports in verified mode is replayed by the verifier rather than merely tolerated.
59. [ ] Fail closed on every unsupported decode feature, done when unsupported non-default decode features are rejected explicitly with stable failure reasons.
60. [ ] Fail closed on every unsupported output feature, done when unsupported non-default output features are rejected explicitly with stable failure reasons.
61. [ ] Add sampler drift protection, done when version-locked conformance tests catch silent changes in sampler behavior across dependency or implementation updates.
62. [ ] Make sampled decoding the default verified-serving mode, done when verified production mode defaults to the canonical sampled path and greedy remains the explicit `temperature=0` special case.
63. [x] Make the key-only retained-state path the canonical verifier path, done when the protocol and implementation both treat the retained-state path as the normal verifier path.
64. [x] Restrict weight-backed replay to debug or oracle use, done when weight-backed replay is no longer presented as part of the canonical protocol.
65. [ ] Remove or clearly demote transitional V4 framing, done when docs, code comments, and tests no longer present transitional V4 semantics as the long-term target.
66. [ ] Keep only irreducible retained state long term, done when the retained-state schema excludes derivable intermediates and preserves only what the verifier cannot reconstruct.
67. [ ] Keep the binary receipt and audit format canonical, done when there is one normative receipt/audit representation rather than multiple equally “official” paths.
68. [ ] Bind any audit-time weight loading to `R_W`, done when audit-time weight access is explicitly checked against the committed published checkpoint identity.
69. [ ] Add a dedicated adversarial verifier-hardening gate to the execution plan, done when no strong claims or final benchmarks can land without passing a specific adversarial-hardening stage.
70. [ ] Build a tamper corpus for every verifier boundary, done when shell openings, retained-state leaves, IO chain, embedding proofs, manifest/spec fields, prompt hashes, seed commitments, LM-head/final-token checks, decode replay, stop policy, prefix openings, and malformed binary payloads all have adversarial cases.
71. [ ] Add cross-proof attacks to the tamper corpus, done when receipt/proof components from different responses can be mixed in tests and are rejected.
72. [ ] Add splice attacks to the tamper corpus, done when prefix, transcript, or request-chain splices are tested and rejected.
73. [ ] Add boundary-condition fuzzing, done when EOS-shortened requests, long/short prompt-output combinations, weird challenge tiers, malformed serialization, and unknown versions are fuzzed and fail correctly.
74. [ ] Require adversarial tests to fail for the right reason, done when hardening tests assert stable, specific verifier failure reasons instead of generic failure.
75. [ ] Define a verification failure taxonomy, done when cryptographic failures, semantic failures, operational/configuration failures, and approximate/statistical outcomes are explicitly classified and reported differently.
76. [ ] Add structured audit-failure reporting, done when verifier output identifies the failed check and the relevant token/layer/proof component in a stable machine-consumable form.
77. [ ] Define partial-audit semantics, done when routine-audit and deep-audit outcomes are reported distinctly and partial statistical coverage cannot be mistaken for full exact success.
78. [ ] Add golden/conformance vectors, done when fixed receipts, fixed audit responses, and fixed verifier outputs exist for future consumers.
79. [ ] Write the challenge protocol specification, done when the timing and derivation of token, layer, and prefix challenges and the exact prover/verifier flow are specified precisely.
80. [ ] Add cross-version tests for the binary format, done when backward rejection, forward rejection, and decoder compatibility are all tested against golden payloads.
81. [ ] Rewrite the verifier from scratch against the frozen final spec, done when a new minimal linear verifier consumes the canonical receipt/audit format, passes the golden vectors, and replaces transitional branches in the trusted verification path.
82. [ ] Add at least one independent non-Rust verifier consumer, done when some non-Rust implementation can consume the golden vectors and reproduce expected verification results.
83. [ ] Add startup self-checks for the live verified server, done when startup fails closed on model-identity mismatch, bad capture settings, or unsupported verified-mode configuration.
84. [ ] Add supported-architecture detection and fail-closed behavior, done when unsupported model families and layouts are detected explicitly instead of being silently treated as decoder-only/Llama-style.
85. [ ] Define and enforce buffer-pressure behavior, done when overflow behavior for capture and audit-state buffers is explicit and tested.
86. [ ] Define and enforce audit-state retention behavior, done when retention TTL, cleanup, and purge behavior are explicit and tested.
87. [ ] Handle verifier-key rotation explicitly, done when key versions are tracked and old receipts remain auditable with the correct historical key.
88. [ ] Add health and readiness checks, done when the live server can report whether capture hooks, audit buffers, model identity, and verified-mode settings are all healthy and aligned.
89. [ ] Add audit-endpoint abuse protection, done when audit requests are rate-limited, bounds-checked, and protected against amplification via unreasonable token or layer requests.
90. [ ] Define a stable benchmark protocol, done when workload corpus, model/settings, warmup policy, hardware class, remote environment, and reporting format are fixed for milestone comparisons.
91. [ ] Record benchmark baselines before and after each runtime-affecting milestone, done when every change to the live sampler, final-token boundary, deep-audit path, sync/capture behavior, or retained-state layout has a before/after benchmark entry.
92. [ ] Run periodic remote-GPU benchmark checkpoints, done when milestone comparisons are repeated on the same representative remote GPU class rather than relying only on local numbers.
93. [ ] Treat unexplained regressions as blockers, done when each regression is either explained or explicitly recorded as an accepted protocol-strengthening tradeoff before further work proceeds.
94. [ ] Lower online inference overhead further if it remains worthwhile, done when the remaining online-cost work has either been completed and benchmarked or consciously stopped because the returns are too small.
95. [ ] Rebenchmark after each meaningful online-path change, done when every material serving-path change has fresh benchmark data attached to it.
96. [ ] Inspect the binary payload by field, done when the receipt and audit format is measured field-by-field rather than only as a total size.
97. [ ] Reduce audit payload structurally where possible, done when unnecessary full vectors are removed without weakening the strongest routine/deep-audit semantics.
98. [ ] Add streaming or incremental audit open if it materially helps, done when it measurably reduces peak memory or audit-open time and is benchmarked against the baseline.
99. [ ] Benchmark the final V5 path, done when baseline, online overhead, commit time, audit-open time, verifier time, retained-state size, and binary payload size are all measured on the reference environment.
100. [ ] Benchmark the protocol-complete V6 path, done when greedy cost, sampled cost, routine-versus-deep audit cost, exact-prefix premium, retained-state size, payload size, and verifier time are all measured on the reference environment.
101. [ ] Measure retained-state memory and audit bandwidth, done when per-token storage cost and audit-transfer cost are quantified for the target deployments.
102. [ ] Measure the attention corridor empirically, done when the FP16/BF16-versus-replay disagreement envelope is measured on real workloads and turned into an evidence-based acceptance threshold.
103. [ ] Measure real audit-window storage costs, done when RAM, NVMe, and network storage costs for realistic retention windows are quantified.
104. [ ] Calibrate routine-audit detection probabilities, done when routine-audit sampling rates are tied to measured or modeled detection probabilities rather than informal intuition.
105. [ ] Benchmark batch verification, done when verifier throughput for batched audits is measured rather than assumed.
106. [ ] Run an explicit hidden-trust-assumptions review, done when exact, approximate, statistical, fail-closed, and out-of-scope trust assumptions are enumerated and checked against the code and claims.
107. [ ] Keep the paper, README, and article normative to the final protocol, done when they describe the destination protocol while benchmark and result sections remain factual about current implementation and measurements.
108. [ ] Update the full protocol specification in the paper, done when the paper contains the exact/statistical/approximate/fail-closed boundary, four-spec structure, routine-versus-deep audit, retained-state path, and verifier randomness model.
109. [ ] Add an explicit input-verification procedure to the paper, README, and article, done when all three docs explain raw request reconstruction, tokenization under the committed input spec, and binding to the embedding/input path.
110. [ ] Add an explicit transcript-chain and anti-splice procedure to the paper, README, and article, done when all three docs describe prompt genesis, token/transcript chaining, and rejection of reorder/deletion/splice/cross-request mixups.
111. [ ] Make one canonical deep-audit procedure explicit in the paper, README, and article, done when the same exact full-prefix algorithm is described as the normative deep-audit mode everywhere.
112. [ ] Add a decode/output support matrix to the paper, done when the paper has one place that says which features are bound, replayed exactly, or fail-closed.
113. [ ] Use one canonical randomness story everywhere, done when the docs use the same wording and procedure for seed commitment, seed reveal, and per-token randomness derivation.
114. [ ] Make the implementation-status split explicit in the docs, done when the docs say they describe the target final protocol and the roadmap is clearly the implementation tracker.
115. [ ] Document verified-mode deviations from stock vLLM, done when the paper and docs explain what verified mode changes operationally without overstating serving-path differences.
116. [ ] Update the full protocol documentation in the README, done when the README covers the final protocol framing, verification coverage, manifest surface, architecture, verifier model, and routine-versus-deep audit semantics.
117. [ ] Document privacy implications of receipts and audits, done when the docs explain what receipts reveal, what audit openings reveal, and what transport/deployment practices are recommended.
118. [ ] Document supported and unsupported architectures explicitly, done when the docs state the supported model/layout families and the implementation’s fail-closed behavior matches that support matrix.
119. [ ] Update the article or writeup to match the full protocol, done when the article has the final trust-boundary figure, four-spec explanation, guarantee boundary, and explicit unsupported-feature fail-closed rule.
120. [ ] Add a pipeline or boundary figure, done when at least one maintained doc has a figure that clearly marks exact, statistical, approximate, and trust-assumption boundaries.
121. [ ] Document canonical semantics versus trust assumptions clearly, done when the docs separate protocol semantics from what is simply trusted outside scope.
122. [ ] Add an explicit trusted-assumptions section to the paper, README, and article, done when standard cryptographic assumptions, verifier-key secrecy, side-channel assumptions, and verifier-correctness assumptions are all stated explicitly.
123. [ ] Add fail-on-unknown versioning rules to the docs and code-facing docs, done when unknown receipt or protocol versions are specified to fail closed rather than being interpreted leniently.
124. [ ] Run the final kept-path benchmark suite, done when the final benchmark set is rerun after the implementation converges on the kept protocol path.
125. [ ] Explain benchmark methodology and regression gates in the paper and docs, done when the published docs say what hardware and workloads were used and how protocol-strengthening tradeoffs were benchmark-gated.
126. [ ] Publish around V6 only when the story is coherent, done when the code, benchmarks, docs, and claims all line up on the same kept path and unfinished claim-critical items are closed.
127. [ ] Build client SDKs, done when at least Python and TypeScript wrappers exist for receipt handling and verification.
128. [ ] Build a `llama.cpp` tracing plugin, done when the protocol can capture the required retained state from a `llama.cpp` serving path.
129. [ ] Add fine-tuned-model and LoRA support, done when adapters or merged checkpoints can be committed, identified, and verified under the model spec.
130. [ ] Add optional receipt or audit encryption for verifier-targeted privacy, done when the product can encrypt receipts or audit payloads to the verifier’s public key without changing the core claim-critical path.
131. [ ] Generalize the packed retained-state path beyond constant-width decoder models, done when the retained-state format is schema-driven and can carry per-layer shapes and layout metadata safely.
132. [ ] Formalize the protocol in Lean, done when the core verification claims have a maintained Lean formalization.
133. [ ] Write a receipt-format specification, done when the receipt and audit binary format is documented independently of the implementation code.
134. [ ] Add API extensions for receipt fields, done when the serving API can carry receipt metadata in a stable, documented way.
135. [ ] Build an OpenAI-compatible proxy with receipts, done when the proxy serves a standard API, returns receipt metadata, exposes a challenge endpoint, and ships with a simple verifier CLI.
136. [ ] Build a Ritual plugin, done when Ritual can consume or expose VeriLM receipts and audits.
137. [ ] Build a Bittensor plugin, done when Bittensor-side inference can expose or consume VeriLM receipts and audits.
138. [ ] Build a Gensyn plugin, done when Gensyn-side inference can expose or consume VeriLM receipts and audits.
139. [ ] Build an inference-marketplace integration, done when marketplace providers can attach receipts and clients can challenge or verify them through the market surface.
