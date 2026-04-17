# Roadmap

All remaining work for CommitLLM, organized by priority. The execution-order sections use plain linear numbering. The detailed tables below retain older stable tracking IDs, including suffix letters, so existing references do not break.

Mark a task done only when it is true in code, tests, docs, and operational behavior. Completed milestones live in [CHANGELOG.md](./CHANGELOG.md).

The shell/tail/binding protocol is structurally complete. The maintained E2E target is now explicit: **exercise everything except arbitrary-position attention replay**. That means binary key/decode artifact, commitment-derived challenges, greedy and sampled decode, full shell/decode openings, routine/full coverage reporting, long prompts/generations, multi-position escalation, tamper rejection, and EOS behavior. Attention is excluded from random-position E2E audits by omitting KV openings; the only kept attention check in E2E is a token-0 smoke test. The Freivalds shell checks are gated by verification profile: Llama checks all 7 matrices (`Wq/Wk/Wv/Wo/Wg/Wu/Wd`); Qwen skips `Wq/Wk/Wv` because the bridge replay cannot match the GPU's quantized QKV GEMM. Token-identity acceptance is now cleanly split by decode mode: **sampled decode is shipped on both Llama and Qwen via `DecodeAcceptanceMode::CapturedLogits`**, which captures the exact GPU logits used by the sampler, verifies exact token sampling, and Freivalds-binds those logits back to `lp_hidden × lm_head_bf16`. GPU E2E is green on both models: Llama `23,335/23,335` decode checks passed, Qwen `17,336/17,336`. Greedy decode can still use cheaper validated paths (`ExactTokenIdentity` on Llama, `LpHiddenBf16` where useful) because argmax is robust to small replay differences. The economics question is now separate from correctness: CapturedLogits eliminates the ~1.87s/token CPU matmul replay cost but requires prover-side retained logits (~501 KiB/token on Llama, ~594 KiB/token on Qwen; ~125-149 MiB retained for a 256-token answer). Audit bandwidth is per *opened* token only (~0.5-0.6 MiB), not per whole answer. On the attention side, **neither Llama nor Qwen should claim arbitrary-position stock-kernel attention replay as solved**: Llama f64 exact replay passes token-0 smoke but diverges at non-zero positions under FlashAttention/bf16; Qwen brute-force replay and fixed-tolerance witnessed-score replay are both rejected as production strong-tier paths. Capturing or committing the attention output `a` is useful plumbing and narrows the prover's degrees of freedom, but it is not an attention-correctness proof unless the verifier also checks that `a = softmax(QK^T / sqrt(d)) @ V`. The `lm_head` Freivalds matmul check remains exact for all profiles. The remaining open problems are: (1) **Correctness blocker — attention**: run one strict FlashAttention/kernel-aligned witness attempt, then move to deterministic attention kernels if it does not pass broadly and cleanly, (2) **Cost blocker — captured-logits economics**: measure capture overhead, retained footprint, open payload delta, and steady-state verifier time with cached keys/artifacts, (3) large decode-artifact distribution/storage (Qwen 7B `lm_head_bf16` is still ~1.57 GB), and (4) a production audit policy that clearly separates shell/decode coverage from still-open attention evidence. **Decode policy**: sampled decode → CapturedLogits path (exact). Greedy decode → cheaper validated paths remain acceptable.

## Linear Execution Order

Use this sequence as the single top-level plan. The tiers below provide detail;
this list defines the actual order to execute.

1. ~~**Lock the E2E test surface: everything except arbitrary-position attention**~~ — **DONE (2026-04-16).** Both Llama and Qwen E2E exercise real verifier wiring, binary keys, sampled/greedy decode, random generated-token challenges, full shell/decode audits, tamper rejection, EOS, and payload/timing reporting. Adversarial suite 36/36 green. KV attention openings intentionally omitted except for token-0 smoke. Two verifier bugs fixed to reach this state: f32 accumulator for LP-hidden bf16 lm_head replay, and generation-local seed indexing in `derive_token_seed`.
2. ~~**Benchmark the current shell/decode surface**~~ — **MEASURED (2026-04-17).** Llama shell/decode is cheap (~15ms verify for full tier, 32 tokens). Qwen LP-hidden bf16 decode verification exposed the replay-cost problem (~1.87s per audited token on CPU), which is now superseded for sampled decode by CapturedLogits.
3. **Benchmark captured-logits economics in steady state** — with sampled decode correctness closed, the next gating question is operational: prover capture overhead, retained bytes per answer, open payload bytes per challenged token, verifier time, and key/artifact caching behavior. Treat keygen as an offline artifact build, not a steady-state cost.
4. **Fix arbitrary-position dense attention ASAP** — do not widen tolerances to make tests pass. Run one strict FlashAttention/kernel-aligned witness attempt that proves `a = softmax(QK^T / sqrt(d)) @ V`. If that cannot produce honest low-bandwidth verification, promote deterministic attention kernels. **(Main correctness blocker.)**
5. ~~**Stabilize routine-tier decode semantics for approximate profiles**~~ — **DONE.** Bounded `lm_head` token acceptance was empirically tested on Qwen W8A8 (A100) and rejected: honest gaps range 0–6556, making any threshold meaningless. The kept answer is not tolerance tuning; it is a captured runtime boundary.
6. ~~**Prototype the decode-side captured boundary**~~ — **SHIPPED.** LP hidden → bf16 lm_head → canonical sampler passed the intermediate Qwen promotion gate and identified the correct runtime decode boundary. This path remains useful for greedy / intermediate capture work, but it is not the kept sampled-decode exactness answer.
7. ~~**Promote the LP-hidden decode path end-to-end**~~ — **DONE as an intermediate milestone.** Real-weight Qwen E2E passed on Modal A100-80GB and proved the decode boundary was correctly identified. Later stress testing showed replay arithmetic still misses the GPU sampling tail. The shipped sampled-decode path is now CapturedLogits, not LP-hidden replay.
8. **Use committed attention-output boundaries only as plumbing** — capture/bind `a` or nearby attention outputs only to stabilize trace plumbing, payload measurement, and later witness checks. Do not present this as attention verification unless the witness also proves the softmax/V aggregation that produced `a`.
9. **Close prefix trust and guarantee language** — KV provenance, audit tiers, guarantee wording, and the formal cheating-game story. Include security comparison of bridge vs local-boundary verification.
10. **Establish one exact dense-attention reference tier** — keep the brute-force exact attention audit as the diagnostic upper bound for payload/CPU and token-0 smoke, not as arbitrary-position production acceptance.
11. **Prototype the dense attention witness or escalate to deterministic attention kernels** — fixed-tolerance f64 replay is rejected for Qwen and arbitrary-position exact replay is not solved for Llama. Candidate witnesses include FlashAttention block-state, online-softmax running max/sum/accumulator state, post-softmax weights plus proof they came from scores, or a later boundary that proves the actual GPU aggregation. If that is not viable, deterministic attention kernels become the mainline fix before DeepSeek-V3.
12. **Take the next production-relevant frontier jump with DeepSeek-V3** — once dense attention is closed well enough to state the Llama/Qwen story honestly, start the DeepSeek-V3 track: MoE, MLA, and MTP.
13. **Broaden dense coverage and portability** — third dense family, larger dense model, longer context, and one non-W8A8 dense quant path.
14. **Take the broader frontier jump** — FP8 boundary policy first, then FP8 KV and wider architecture variants.
15. **Improve serving/runtime compatibility** — native capture, batching, paging, tensor parallelism, CUDA graphs, and payload reduction.
16. **Pursue verified-mode research** — deterministic kernels / deterministic runtime only after the current stock-kernel path is shipped and benchmarked.

### Architecture fork

Two verification paths, ordered by time-to-value:

1. **Stock-kernel path with captured local boundaries** (near-term) — keep vLLM's existing kernels and move strong-tier verification toward captured local boundaries instead of long-chain bridge replay. On the decode side, the kept path is now **CapturedLogits** for sampled decode: capture the actual GPU f32 logits, verify exact sampling, and Freivalds-bind back to `lp_hidden × lm_head_bf16`. For greedy decode, cheaper paths (i32 ExactTokenIdentity or LP-hidden bf16) remain acceptable. On the attention side, captured `a` / attention-output boundaries are useful trace plumbing and reduce degrees of freedom, but they do not prove attention correctness by themselves. The real stock-kernel attention target is a kernel-aligned witness that proves the captured/committed `a` was produced by the actual FlashAttention computation.
2. **Verified-mode deterministic kernels** (backstop / clean path) — custom CUDA kernels with exact arithmetic. Eliminates the boundary-approximation class entirely but requires kernel engineering per op and a deeper runtime fork. Promote this from backstop to mainline if the FlashAttention-aligned witness cannot produce honest low-bandwidth arbitrary-position attention verification. Deterministic kernels mainly solve dense-model arithmetic: they make newer dense families and FP8-style dense serving much easier to support by restoring exact or near-exact GEMM semantics and removing most linear-layer tolerance tuning. They do **not** by themselves solve MoE routing, MLA / nonstandard KV state, speculative decoding / MTP semantics, sparse attention semantics, or cross-request cache semantics; those remain protocol/transcript work even if the arithmetic layer becomes exact.

The long-chain bridge is downgraded from the main verification path to a routine-tier fallback / diagnostic tool. It remains useful where captured boundaries are unavailable, but it should no longer define strong-tier semantics for dense models once a measured local-boundary policy exists.

### Near-term decision gates

Use these as explicit promotion criteria between phases, not just as vague "next work":

1. **E2E must be honest before it is broad** — Llama/Qwen E2E may cover all protocol wiring, shell checks, decode, sampling, tamper, EOS, and random generated-token challenges, but it must not imply arbitrary-position attention is verified while KV attention openings are omitted.
2. **Attention is the ASAP blocker before DeepSeek** — promote neither Llama nor Qwen to arbitrary-position stock-kernel attention verification by widening tolerance. The next accepted paths are a FlashAttention/kernel-aligned witness or deterministic attention kernels.
3. ~~**Decode boundary promotion depends on real-weight E2E, not just diagnostics**~~ — **GATE PASSED.** Real-weight GPU E2E passed 261/261 verifier checks on all runs (greedy and sampled) on Modal A100-80GB. Remaining follow-up: capture overhead benchmark. New concern: large decode-artifact packaging/distribution (Qwen 7B `lm_head_bf16` is still ~1.57 GB).
4. **Local-boundary path starts with `x_attn` on attention and `LogitsProcessor` input on decode** — decode no longer needs float-logit capture or tolerance tuning. Do not pre-commit to `x_ffn`, `h`, residual capture, or full-logit capture until the cheap boundary candidates are measured first.
5. **Committed attention-output boundaries are not enough** — binding `a` or an attention output proves downstream consistency, not that the provider computed `softmax(QK^T / sqrt(d)) @ V`. It is allowed as infrastructure and measurement plumbing, but not as an attention-correctness claim.
6. **Exact dense-attention audit stays a reference tier before optimization** — use brute-force exact attention as the diagnostic upper bound for payload/CPU and token-0 smoke. Do not treat it as arbitrary-position production acceptance under stock FlashAttention.
7. **Fixed-tolerance replay is rejected as the attention answer** — the Qwen 39-prompt sweep breached `±3` (global max diff `9`, only `23/39` prompts passed, layer 11 dominant), and Llama arbitrary-position exact replay shows much larger FlashAttention-vs-f64 gaps at non-zero positions. Do not promote attention by widening tolerances. The next stock-kernel attempt must move the witness boundary later / closer to kernel state; otherwise deterministic attention kernels move forward as the mainline fix.
8. **Native captured-boundary backend is conditional, not automatic** — only invest in the native path after the protocol explicitly chooses local captured boundaries as the kept dense-model direction and after the boundary cost benchmark shows Python copy/materialization is the limiting factor. The LP-hidden decode path may justify native capture sooner if synchronous D2H proves too expensive in production.

---

## Tier 0 — Critical path

Highest-priority work. Do in this order.

Strict near-term execution order inside this tier:
1. ~~Keep Llama/Qwen E2E green for everything except arbitrary-position attention.~~ **DONE (2026-04-16).**
2. ~~Benchmark the current shell/decode bandwidth, verifier CPU time, and audit-open time.~~ **MEASURED (2026-04-17).** Llama greedy: ~15ms verify (full, 32tok), clean. Qwen LP-hidden bf16 replay exposed the old cost problem (~1.87s/token CPU) and Llama A/B diagnostics confirmed replay was the wrong sampled-decode architecture (`ExactTokenIdentity` rejected, LP-hidden only ~98.7%). This benchmark is now historical context for why CapturedLogits replaced replay on sampled decode.
2b. **Solve sampled decode exactness — GPU E2E VALIDATED (2026-04-17).** `DecodeAcceptanceMode::CapturedLogits` captures the actual GPU f32 logits from `CanonicalSamplerHook`, commits them in the Merkle leaf, verifies exact sampling, and Freivalds-binds them back to `lp_hidden × lm_head_bf16` via a secret ±1 random projection (two dot products, not a full matmul). **Decode proof structure: exact token sampling from captured GPU logits + probabilistic algebraic binding back to lm_head.** Profiles: `llama-w8a8-captured-logits`, `qwen-w8a8-captured-logits`. Cost structure: (a) prover retains ~501 KiB/token (Llama) or ~594 KiB/token (Qwen) — for 256 tokens: ~125-149 MiB retained until audit, (b) audit bandwidth is per *opened* token only (~0.5-0.6 MiB/opened token), not per whole answer, (c) verifier CPU: two dot products (negligible). The prover must retain/commit all tokens' logits since the challenge is chosen after generation.
2b-i. ~~**GPU E2E validation for captured-logits**~~ **DONE (2026-04-17).** Both models pass all decode checks: Llama 23,335/23,335, Qwen 17,336/17,336 (3 greedy + 20 sampled each). Tamper rejection works. Remaining 736 Llama attention failures are the known pre-existing gap (not CapturedLogits).
2b-ii. **Benchmark captured-logits economics** — four separate numbers: (1) capture overhead ms/token + GPU→CPU transfer cost, (2) prover retained footprint bytes/token and bytes/answer + peak resident memory, (3) open payload delta per challenged token, (4) verifier CPU ms/token for Freivalds check.
2c. ~~**Run Qwen A/B sampled stress test**~~ — **Superseded by CapturedLogits.** The LP-hidden stress-test question became irrelevant once sampled decode moved to exact captured GPU logits on both families. Keep LP-hidden only as historical evidence / fallback for cheaper greedy paths, not as the shipped sampled-decode rule.
2d. ~~Reduce LP-hidden bf16 decode verification cost~~ — **Superseded by CapturedLogits.** The captured-logits path eliminates the ~1.87s/token CPU matmul entirely (replaced by two dot products). The new cost question is receipt bandwidth, not CPU verify time.
3. Treat committed attention-output boundaries as plumbing only; do not claim they verify attention.
4. Prototype the FlashAttention/kernel-aligned attention witness.
5. If the witness path fails broad randomized sweeps, promote deterministic attention kernels as the mainline fix.
6. Keep exact attention replay as a reference/token-0 smoke tier, not production arbitrary-position acceptance.
7. Define production audit policy from measured shell/decode cost plus the chosen attention path.
8. Define strong-tier boundary policy per family from measured results, not assumptions.
9. Freeze the canonical verifier and update docs/claims.
10. Cut the release.

The detailed items below remain useful tracking references, but they should not
override the linear near-term execution order above.

| # | Item | Status |
|---|------|--------|
| 1 | **RMSNorm contraction / local operator — MEASURED** — crude contraction ratio `ρ_j = γ·‖W_o‖₂/RMS(x^(j))` and tighter local sensitivity/operator tests have now been run on real checkpoints. **Qwen-7B W8A8**: crude contraction mixed; tighter local operator puts 21/28 layers below 1, but 3 layers (0, 1, 27) can still individually flip the output and many middle layers remain near the margin. **Llama-3.1-8B**: crude contraction fails everywhere and tighter local operator still leaves 0/32 layers below 1; 6 layers can individually flip the output. Conclusion: contraction and local linearization are useful diagnostics, but they do not rescue the corridor by themselves. Next step: refine the local map further with prompt diversity and the full within-layer propagation channel. Subtests: projected norm `‖J_RMSNorm(x_j) W_o^(j)‖`, head-aware/block-aware operators, mixed `∞→∞` and `∞→2` bounds consistent with the protocol corridor, finite-difference local sensitivity on real traces, and inclusion of the MLP propagation channel after the perturbed residual enters the rest of the layer. | **measured** |
| 2 | **Shrink the honest corridor / land arbitrary-position attention verification** — prover-side determinism confirmed (§1h), sidecar-to-verifier gap measured at L∞=8–9 (§1i), but later experiments show that stock-kernel f64 replay is not enough for production arbitrary-position attention claims. **Llama:** token-0 exact attention can pass, but non-zero positions can diverge heavily from FlashAttention/bf16; keep arbitrary-position E2E KV attention disabled until a better witness/kernel path exists. **Qwen:** brute-force replay fails at later positions, and witnessed-score f64 softmax·V replay breached the proposed `±3` bound in the 39-prompt sweep. Backend binding is done (`attn_backend`/`attn_dtype` in manifest/key/verifier, SDPA mandated for W8A8, fail closed on eager/unknown). The next accepted paths are kernel-aligned attention witness or deterministic attention kernels. | **blocked on attention witness/kernel** |
| 2a | ~~**Bind attention backend in manifest**~~ — `attn_backend`, `attn_dtype` added to manifest/key/verifier with cross-checks. W8A8 mandates SDPA, fail closed on eager and unknown. Keygen populates from model config. | **done** |
| 2b | ~~**Measure real protocol gap (sidecar → verifier)**~~ — L∞=8 (Qwen 7B), L∞=9 (Llama 8B, Llama 70B) on A100-80GB with SDPA. ~93–97% exact, >99.9% within ±1. Gap grows weakly with sequence length. See §1i. | **done** |
| 2c | **Cross-hardware stability** — same backend on A100 vs H100. If hardware changes the gap, τ must accommodate hardware diversity. | open |
| 2d | ~~**Better `a` quantization**~~ — measured and ruled out as the main τ-reduction path. Per-head `scale_a` does not help (one head still dominates; INT8 corridor gets worse), and INT16 retained `a` does not materially shrink the float-space honest gap. No longer on the critical path. | **done** |
| 2e | **Layer/head-specific tolerances** — keep only as a routine-tier fallback. Strong-tier direction is family-specific score anchoring, not ever-finer τ tuning. Corridor worst layers remain useful for diagnostics: Qwen 1/27, Llama-8B 25, Llama-70B 7/13/70. | open |
| 2f | **Qwen `x_attn` boundary: captured path exact, bridge approximate** — 10-run Modal diagnostic (2026-04-08) proved: (1) shell K from captured `x_attn` + bf16 CUTLASS epilogue matches GPU QKV output K-slice with L-inf=0 for ALL prefix positions; (2) bridge-derived `x_attn` (f64 RMSNorm + quantize) diverges from vLLM's fused `norm_quant` kernel output for all positions; (3) the apparent pos 1+ divergence in earlier diagnostics was a measurement bug (RoPE applied in-place, hook captured post-RoPE K). This remains useful for shell/KV provenance, but it does not by itself solve attention routing because softmax·V replay still diverges under stock kernels. Regression test: `test_qwen_shell_exact.py`. | **partial** |
| 2g | **Local boundary prototype** — prototype local captured-boundary verification on one family (Qwen or Llama). Replace long-chain bridge replay with per-layer local checks using captured boundaries where available. Start with `x_attn` and, if useful, the committed attention-output boundary `a` as plumbing. Explicitly distinguish the claim: committed `a` proves downstream consistency, not attention correctness, unless paired with a witness proving `a = softmax(QK^T / sqrt(d)) @ V`. | open |
| 2h | **Strong-tier boundary policy** — for each supported family, specify which boundaries are: (a) captured exactly, (b) locally checked, (c) replayed approximately, (d) unsupported. Current policy until the attention witness/kernel work lands: E2E random generated-token audits verify shell/decode and omit KV attention openings; Llama keeps token-0 exact-attention smoke only; Qwen attention is unsupported beyond diagnostics; sampled decode is exact on both families via `CapturedLogits`; greedy decode may use cheaper validated replay paths where explicitly allowed. | partial |
| 2i | **Boundary cost benchmark** — quantify retained-state growth and online overhead for: (a) captured `x_attn` only (current), (b) captured `x_attn` + `x_ffn` + `h`, (c) optional residual capture. Report per-token/per-layer bytes, total prefill volume at 4K/32K, end-to-end latency impact, and keep prover-internal capture/hash volume separate from client-visible audit payload. | open |
| 3 | **Real-model corridor attacks — PARTIALLY MEASURED** — multi-prompt finite-difference sensitivity measured: Qwen 28/28 layers flip on every prompt, Llama 0–6 flips depending on margin. All-layers-simultaneously always flips. Remaining: constrained adversarial optimization (true worst-case under L∞), multi-token accumulation across autoregressive decode. See `research/attention-gap.md` §1e, §3l–3m. | **partial** |
| 4 | **Unify attention path** — one `x_attn` boundary, one QKV replay story across prover/verifier/corridor, and one correct historical-prefix binding story. No silent mixing of bridge-derived Q with committed K/V. The captured `x_attn` is the canonical QKV boundary for strong-tier verification. Remaining: make `server.audit()` default to captured `x_attn` for strong/full/score-witness audit opens whenever captured data is available (currently defaulted for `deep_prefix`), and document the bridge path explicitly as a routine-tier approximation. | open |
| 4a | **Routine-tier `lm_head` acceptance calibration — DONE / REJECTED** — for approximate profiles like Qwen, the `lm_head` matmul/Freivalds check remains exact, but final token identity can drift because quantized verifier logits reorder top tokens relative to GPU bf16 logits. Honest-gap measurement across real runs showed a heavy tail (up to ~6556 on a ~7000 range), so bounded token acceptance was rejected as a meaningful rule. Result: no threshold tuning. Future exact paths are captured `LogitsProcessor`-input hidden with bf16 `lm_head` replay or deterministic `lm_head` kernels. | **done** |
| 4b | ~~**Decode boundary at `LogitsProcessor` input — SHIPPED**~~ — LP-hidden capture passed exact hook-level validation (23/23 checks) and real-weight GPU E2E on Qwen, proving the correct runtime decode boundary and shipping the first exact decode-boundary path. Later stress testing showed replay arithmetic still misses the GPU sampling tail, so LP-hidden is now an intermediate milestone rather than the kept sampled-decode path. | **done / intermediate** |
| 4c | ~~**Qwen real-weight LP-hidden E2E — PASSED**~~ — real prover/verifier path on Modal A100-80GB with keygen-populated `lm_head_bf16`: 261/261 checks on all runs (greedy × 3, sampled × 2, EOS trim). The large `lm_head_bf16` payload is now split out of `VerifierKey` into a separate decode artifact, but the artifact is still ~1.57 GB for Qwen 7B. Remaining follow-up: capture overhead benchmark (sync D2H), artifact caching/distribution, and size reduction where it preserves exact replay. | **done** |
| 5 | **Adversarial testing** — cheating-provider test suite: receipt forgery, transcript splicing, intermediate tampering, model substitution, selective layer cheating, KV injection, and selective-abort / audit-policy gaming. Every attack must fail verification or be documented as accepted gap. Includes real-model constrained corridor attacks (see #3). | **in progress** |
| 5a | **Middleware/router adversarial tests** — test suite for man-in-the-middle attacks by an untrusted router or API gateway sitting between an honest inference provider and the end user. Attacks to implement and verify detection: (1) single-token substitution (change one token_id in displayed output), (2) token insertion/deletion (add or remove tokens), (3) prompt rewriting (modify prompt before forwarding to provider), (4) full response replacement (substitute entire response from a different model/prompt), (5) selective field tampering in audit payload (modify shell opening, retained state, or Merkle proofs), (6) replay attack (serve a cached valid audit from a different request), (7) audit stripping (remove audit data entirely). All attacks except #7 must produce a verifier FAIL with 100% detection via IO chain / Merkle binding / Freivalds. #7 is a policy decision (no proof = no trust). Each test should confirm the specific `FailureCode` raised. | open |
| 6 | **Formal security argument** — define the cheating game, state Freivalds soundness bound (1/p), state detection probability as f(sampling rate, cheating fraction), state what commitment binding prevents. Include random audit conditioning analysis: detection probability vs cheating fraction vs audit rate. | open |
| 7 | **KV provenance** — sampled/batched Freivalds checks verify committed K/V are consistent with committed Wk/Wv weights and the bound `x_attn` inputs. Tightening current-token attention alone is insufficient if the adversary can manipulate prefix state; KV provenance closes the upstream path. With #2f resolved (captured `x_attn` gives exact K), the provenance check can now use exact equality for models where captured `x_attn` is available, falling back to corridor bounds only for the bridge path. | open |
| 7a | **Exact attention reference phase (Qwen first)** — the canonical verifier phase now exists: reconstruct Q for the challenged token from shell `q`, validate the committed K/V transcript, replay canonical attention (`QK^T / sqrt(d)` → softmax → `weights @ V`), requantize to committed `a`, and compare. Measured result on Qwen W8A8: token 0 can pass with a local `±1` LSB tolerance at the final requantization boundary, but later tokens still diverge heavily (max diff on the order of 100–170), so this is a **reference / diagnostic tier only**, not the kept Qwen strong-tier path. It remains valuable as the real upper-bound payload/CPU benchmark (~243 MB at 1K context with the current f64 K/V transcript and proofs) and as the baseline for later witness/sketch optimizations. | **partial** |
| 8 | **Score witnessing** — generated-token score witnessing remains useful for anchoring and diagnostics, but **fixed-tolerance f64 softmax·V replay is rejected as the kept strong-tier path**. Qwen breached `±3` in the 39-prompt sweep (global max diff `9`, only `23/39` prompts passed; layer 11 dominant). Llama arbitrary-position exact replay also shows large FlashAttention-vs-f64 gaps at non-zero positions, so exact replay cannot be treated as a solved production attention claim. Do not promote either family by widening tolerances. | **partial / fixed-tolerance rejected** |
| 8a | **Dense FlashAttention-aligned attention witness — ASAP** — replace rejected f64 softmax·V replay with a later-boundary / kernel-state witness experiment for Llama and Qwen. Candidate witnesses, in priority order: (1) online-softmax running max/sum/accumulator state from the FlashAttention path, (2) post-softmax weights plus a proof they came from anchored scores, (3) post-aggregation `a` plus enough kernel state to prove the aggregation, (4) another object that avoids replaying attention with CPU f64 semantics. Captured `a` alone is not sufficient; it is only the committed downstream boundary. Success criteria: honest pass on broad randomized prompt sweeps, bounded tail with clear source, payload in the single-digit-to-low-double-digit MB range per fully attention-audited token, and verifier attention time in the ms / low-tens-of-ms range. If this cannot produce an honest low-bandwidth path, promote deterministic attention kernels as the mainline fix before DeepSeek-V3. | **ASAP / open** |
| 9 | **W_o conditioning** — compute σ_min(W_o), ‖W_o‖₂, and more faithful projected/local operator bounds per layer on real models. **Confirmed critical by #1**: Llama ρ>1 everywhere means the crude spectral bound is loose — need tighter per-layer operator norms to identify which layers genuinely amplify vs where the bound is just conservative. Quantifies adversary freedom when attn_out_i8 is commitment-bound but not independently replayed. Also translate hidden-state drift into per-layer logit-margin bounds using the real LM head, and measure margin-plus-frequency on real prompts to see when dangerous layers actually coincide with small output margins. | open |
| 10 | **Deep-audit payload sizing / policy** — budget the per-token cost of the current shell/decode E2E surface separately from future attention evidence. Current practical policy: random generated-token audits open full shell/decode with `include_kv=false`; token-0 attention smoke is allowed; arbitrary-position attention remains unsupported until the attention witness/kernel work lands. Future sizing should evaluate kernel-aligned witnesses or deterministic kernels, not wider softmax·V replay tolerances. **Must benchmark CPU verification time in the Rust verifier**, not estimate from FLOPs. | open |
| 11 | **Rename to CommitLLM** — all docs, APIs, CLI, scripts, packages use `CommitLLM`. | open |
| 12 | **Remove V4/V5 naming** — public surface uses one canonical release name, not internal version numbers | open |
| 13 | **Clean repo** — delete stale corridor/debug paths, one-off hooks, duplicate helpers, obsolete protocol-era code. Repo should reflect the kept path, not an archaeology of experiments. | open |
| 14 | **E2E everything-except-attention policy** — Llama and Qwen E2E tests should cover binary key/decode-artifact paths, commitment-derived random generated-token challenges, greedy and sampled decode, routine/full shell coverage, long prompt/generation cases, multi-position escalation, tamper rejection, EOS, payload size, and verify timing. They must omit KV attention openings for random non-zero token positions (`include_kv=false`) and state clearly that arbitrary-position attention is not being claimed. Token-0 attention smoke remains as the maintained GPU attention check. | **in progress** |
| 15 | **Freeze canonical verifier** — canonical verifier is the frozen trusted path; legacy verifier no longer needed. Fresh real-GPU confirmation via `test_e2e_v4.py` and `test_adversarial.py`. | partial |
| 16 | **Adversarial methodology research** — move from defender testing (flip bits, check rejection) to attacker simulation (white-box forgery, adaptive adversary, composition attacks, probabilistic security curves). See [`research/adversarial-methodology.md`](./research/adversarial-methodology.md). | open |
| 17 | **Fuzz binary parsers** — coverage-guided fuzzing (cargo-fuzz) on verifier's binary deserialization. Malformed receipts fail closed, never panic. | open |
| 18 | **Stable benchmark protocol** — fix workload corpus, model/settings, warmup, hardware class, reporting format | open |
| 19 | **Benchmark routine-audit path** — baseline, online overhead, commit time, audit-open time, verifier time, retained-state size, payload size | open |
| 20 | **Paper/README/article consistency** — claims match code match measurements. See [Docs detail](#docs--publication-detail) below. | open |
| 21 | **v1.0 release** — cut clean tagged release only when code, benchmarks, docs, and claims all line up | open |

---

## Tier 1 — Security hardening

Blocks credibility with serious reviewers.

| # | Item |
|---|------|
| 21a | **Trust assumptions review** — enumerate exact, approximate, statistical, fail-closed, and out-of-scope assumptions; check against code and claims |
| 22 | **Freshness / temporal binding** — verifier-issued nonce or timestamp in receipts; prevent replay of cached honest responses. Also define selective-abort / denial-of-audit policy: response deadlines, retention horizon, and what evidence is required when a provider refuses or misses an audit. |
| 23 | **Tolerance bounds** — analytical attention-gap story for the kept path. Empirical data exists (Qwen L∞=8, Llama L∞=9); a full composed theorem does not. Do not assume plain per-layer `L∞` alone gives a strong semantic guarantee. Use norm-consistent bounds: if the corridor is stated in `L∞`, the downstream analysis must use compatible induced norms or explicitly justify the `L2`/spectral relaxation. Combine per-step corridor bounds with the local operator measurements from #1, real-model attack evidence from #2, `W_o` conditioning (#8), and score witnessing (#7) where needed. Add margin-based safety as an operational corollary where full worst-case theorems are too loose. RMSNorm contraction is supporting evidence, not the primary fix. |
| 24 | **Audit tiers** — formalize receipt-only / routine / deep / full with explicit coverage and cost per tier. The default production shape should be: (a) decode on all generated tokens, (b) sampled attention-row checks on all generated tokens, (c) a small number of full-shell and optionally exact-attention audits on randomly chosen tokens, (d) escalated full audits for high-stakes answers. Separate clearly what is exact, statistical, and still unsupported. |
| 25 | **Guarantee language** — docs clearly separate exact (Freivalds/INT8), bounded approximate (attention), statistical (sampled provenance), and fail-closed |
| 25a | **Bridge vs local-boundary security comparison** — explicitly compare: (a) long-chain bridge replay, (b) local captured-boundary checks, (c) what each proves and does not prove about nonlinear transitions. The guarantee language changes: local-boundary checks verify each nonlinear op independently (no accumulated drift) but still require per-op tolerance or exactness claims. Document which nonlinear ops (RMSNorm, SiLU, quantization, residual add) are exact vs approximate under each verification mode, and state clearly that the bridge remains a routine-tier fallback once local boundaries are adopted for strong-tier dense models. |
| 25b | **Local nonlinear transition checks** — formalize RMSNorm, quantization, SiLU/gating, and residual update checks as local constraints with exact vs approximate status clearly stated per op. Quantization should be exact; RMSNorm and SiLU may have small fused-kernel gaps that need measurement. |
| 25c | **Decode acceptance semantics** — specify per profile whether final-token acceptance is exact token identity, captured-logits exact, LP-hidden boundary replay, bounded token acceptance, or unsupported. This is separate from `lm_head` Freivalds: the linear algebra check can remain exact while the decode-acceptance rule changes by profile. Current state: sampled decode on both Llama and Qwen is exact via `CapturedLogits`; greedy decode may use `ExactTokenIdentity` (Llama) or other cheaper validated paths where explicitly allowed; LP-hidden remains an intermediate boundary and fallback, not the shipped sampled-decode rule. Document how greedy vs sampled decode is handled, what randomness is or is not replayed exactly, and how each decode mode is reported to users. |
| 26 | **Verifier-key distribution** — canonical procedure for trusted key provenance, hash pinning, historical lookup, fail-closed on unknown keys |
| 27 | **Key rotation** — key versions tracked; old receipts remain auditable with historical key |
| 28 | **Verifier-secret randomness** — one canonical verifier-secret randomness story. Make explicit which claims rely only on secrecy (single Freivalds check soundness `≈ 1/p`) and which rely on independence (cross-layer amplification, if claimed). If the protocol keeps shared `r_j` per matrix family, document that this is sufficient for base per-check soundness but does not provide multiplicative amplification across layers. If stronger composed claims are desired, derive per-layer Freivalds randomness from a verifier-secret seed. Deep-audit batching randomness and challenge selection must remain unpredictable until after commitment. |

---

## Tier 2 — Future Coverage After Core Attention Fix

These are important scope-expansion tracks, but they are **not** on the
critical path before the current attention / score-witness story is solid.
First make one strong protocol work well for the currently supported dense
decoder slice; then generalize.

Near-term order within this tier is:
1. DeepSeek-V3 frontier feasibility / compact-policy track.
2. Third dense family: `Mistral` / `Ministral` / `Nemo` or `Gemma`.
3. FP8 support.

Reason: once the dense Qwen attention story is honest, the next production-relevant
architecture is DeepSeek-V3 (MoE + MLA + MTP). A third dense family remains
important to prove the path is not overfit to Llama/Qwen, but it no longer
needs to block the first frontier architecture track.

| # | Item |
|---|------|
| 29a | **DeepSeek-V3 frontier feasibility / compact-policy track** — after Qwen attention is closed enough to state the dense story cleanly, start the first DeepSeek-V3 audit design pass explicitly. Goal: quantify what a production-usable policy looks like for DeepSeek-V3 under MoE + MLA + MTP, including decode-all cost, partial-attention cost, and a small number of full-shell / full-attention audits. This is the next production-relevant architecture track, not a side note under generic MoE. |
| 29 | **MoE support** — verify expert routing decisions and per-expert shell matmuls for Mixtral, DeepSeek-V2/V3, Qwen-MoE. Commit router logits/top-k, Freivalds on selected expert weights, fail-closed on unsupported routing. This is a serious protocol extension and should follow the first DeepSeek-V3 feasibility/design pass; broad MoE rollout still should not outrun the dense attention story or the first third-family dense validation. |
| 30 | **FP8 quantization** — split into two explicit tracks: (a) compat mode with bounded replay against vendor FP8 kernels, and (b) verified mode with a canonical exact/fixed-point lowering or custom deterministic kernel. Do not start before the current dense slice and third-family support are stable. If the deterministic-kernel path is pursued, FP8 should move closer to that work because owning the arithmetic is the main reason to do FP8 in a verifier-friendly way. |
| 30a | **FP8 boundary policy** — decide whether FP8 uses captured boundaries, compat replay, or verified-mode kernels. This should be explicit before FP8 implementation starts, not implied from the W8A8 path. FP8 decode semantics differ from INT8 and may require different boundary capture or tolerance strategies. |
| 30b | **Kernel payoff boundary** — document explicitly what deterministic kernels buy: dense-model GEMM exactness, cleaner Freivalds semantics, much less tolerance tuning, and a much stronger path to FP8 dense serving. Also document what they do **not** buy automatically: routing semantics, nonstandard cache objects, speculative accept/reject logic, sparse attention rules, or shared-cache provenance. |
| 31 | **Third family** — prioritize `Mistral` / `Ministral` / `Nemo` or `Gemma` after the first DeepSeek-V3 feasibility pass. Goal: prove the kept dense path is not overfit to Llama/Qwen while staying in dense-decoder territory. |
| 31a | **Third-family local-boundary validation** — validate the local captured-boundary path on the third dense family (Mistral or Gemma) with the same architecture, after the first DeepSeek-V3 feasibility pass and before broad FP8 rollout. Confirms the local-boundary design generalizes beyond Llama/Qwen. |
| 32 | **Larger model** — at least one 30B/70B-class datapoint on the corrected path |
| 33 | **Long-context 128K+** — validate corridor at production context lengths where attention numerics are most stressed |
| 34 | **GPTQ/AWQ/grouped quant** — at least one non-W8A8 family with a fully validated verifier replay path |
| 35 | **Verification profiles** — per-family configuration (tolerance table, context lengths, audit policy) under the same core protocol |
| 35a | **FP8 KV cache** — quantify how FP8 KV storage changes the committed-KV / attention corridor and what additional tolerance or deep-audit support is required. Separate from FP8 shell matmuls. |
| 35b | **MLA support** — support multi-latent attention / non-standard KV cache layouts (older DeepSeek-style MLA). This is the hardest frontier-architecture extension because it changes the committed state object itself, not just the arithmetic. Requires a redesigned committed-cache transcript and a new verifier replay object for latent / nonstandard cache semantics. |
| 35c | **Native sparse attention / NSA** — commit and verify the sparsity pattern or active blocks plus the sparse attention transcript for models like GLM-5 / DeepSeek-style sparse attention. The verifier must know which positions/blocks were active and verify attention only over that committed sparse structure. |
| 35d | **MTP / speculative multi-token decoding** — define retained-state and audit semantics for draft/accept transcripts. Commit the draft model outputs, acceptance/rejection mask, and final accepted token path; verify both the draft model behavior and the accept/reject logic. |
| 35e | **Probabilistic / compressed KV schemes** — explicitly classify which KV compression methods are supported, unsupported, or require a weaker proof tier (e.g. randomized projections / QJL-style residuals). |

---

## Tier 3 — Performance & serving compatibility

Determines whether CommitLLM is deployable at production throughput.

| # | Item |
|---|------|
| 36 | **Native capture backend** — C++/CUDA/Triton hot path replacing Python interception. Milestones: (a) profile dominant cost, (b) prototype native `cutlass_scaled_mm` hook with ring buffer, (c) benchmark vs Python path, (d) integrate. Also enables CUDA graph compat (#44). |
| 36a | **Native captured-boundary backend** — if local-boundary verification becomes the main dense-model path, the capture path needs a native implementation sooner because bandwidth/copy becomes the dominant bottleneck. The additional `x_ffn` + `h` captures (3-4× current volume) through Python→Rust is the likely scaling wall. Do not start this before the `x_attn`-only prototype and cost benchmark establish that local boundaries are the kept path. |
| 37 | **Cross-request prefix caching** — committed prefix cache becomes a first-class committed object. New requests must reference a committed cache root explicitly, and the verifier must check that the continuation was computed from that exact shared cache state. Highest-impact unsupported optimization (~2-5x TTFT). Important sooner than many frontier features because deployment economics depend on it. |
| 38 | **Test continuous batching** — tracer splits batched prefill into per-token traces; commit+audit+verify on real GPU with concurrent requests |
| 39 | **Test paged attention** — paged KV doesn't interfere with committed KV transcript integrity |
| 40 | **Test tensor parallelism** — TP=2 and TP=4 produce valid receipts. Startup hook patches TP workers but needs real multi-GPU confirmation. |
| 41 | **Test fused kernels** — fused QKV and gate-up projections produce correct shell accumulators, pass Freivalds |
| 42 | **Test FlashAttention** — attention outputs within established corridor bounds with FA enabled |
| 43 | **Test more W8A8 families** — overlaps with #31 and #32 |
| 44 | **CUDA graph compatibility** — capture hooks survive graph replay. Largely solved by #36. Impact: ~10-30% decode latency. |
| 45 | **Speculative decoding** — commit draft model identity, draft outputs/tokens, acceptance masks, and the final accepted token path. New sub-protocol for propose → verify → accept/reject transcript semantics. High serving relevance because many real deployments depend on speculative decode for throughput. |
| 46 | **LoRA/adapter verification** — verifier confirms served weights match `base + adapter`. Model spec already hashes adapter identity. |
| 47 | **Pipeline parallelism** — PP configs produce valid receipts. Capture layer handles cross-GPU activation transfers. |
| 48 | **Lower online overhead** — complete remaining cost work or consciously stop if returns are too small |
| 49 | **Reduce audit payload** — lossless shell compression: width-packed accumulators, tighter binary layouts. Don't drop shell matrices. |

---

## Tier 4 — Production infrastructure

Needed for real multi-tenant deployment.

| # | Item |
|---|------|
| 50 | **Streaming / SSE** — incremental commitments during token-by-token streaming; partial receipts before generation completes |
| 51 | **Multi-turn binding** — chain receipts across conversation turns (hash of prior receipt in next commitment) |
| 52 | **Monitoring / observability** — export metrics: failure rate, corridor histograms, audit latency, buffer utilization, receipt throughput (Prometheus/OTel) |
| 53 | **Receipt storage** — indexing by request ID, time, model, verification status. Append-only log, object store, or blockchain anchoring. |
| 54 | **Verifier deployment** — hosted service vs client-side library vs hybrid. Address key distribution, receipt transport, offline vs real-time. |
| 55 | **Receipt compression** — compact aggregate receipts or compressed streams at thousands of req/sec |
| 56 | **Audit delegation** — verifier delegates authority to trusted third-party auditor. Define trust model, revocation. |
| 57 | **OpenAI-compatible proxy** — standard API with receipt metadata, challenge endpoint, verifier CLI |
| 58 | **Client SDKs** — Python and TypeScript wrappers for receipt handling and verification |
| 59 | **Reference client flow** — one canonical CLI: trusted key → receipt → served output → challenge → audit → coverage interpretation |
| 60 | **Startup self-checks** — fail closed on model-identity mismatch, bad capture settings, unsupported config |
| 61 | **Architecture detection** — unsupported families detected explicitly, not silently treated as Llama-style |
| 62 | **Buffer-pressure behavior** — capture and audit-state buffer overflow is explicit and tested |
| 63 | **Retention behavior** — audit-state TTL, cleanup, and purge are explicit and tested |
| 64 | **Health checks** — server reports capture hook, audit buffer, model identity, verified-mode health |
| 65 | **Abuse protection** — audit requests rate-limited, bounds-checked, no amplification |
| 66 | **EOS-trim evidence** — real-GPU test: EOS-before-max_tokens triggers trim, commit+audit+verify passes |

---

## Tier 5 — Strengthening & research

Important for the long-term story but not blocking anything above.

| # | Item |
|---|------|
| 67 | **Deterministic inference mode** — eager attention (`attn_implementation="eager"`), `torch.use_deterministic_algorithms(True)`, fixed CUDA seeds. If the forward pass is bit-exact and the verifier replays with the same arithmetic spec, τ=0 closes the attention gap for opened tokens without score witnessing. Requires a defined arithmetic/kernel spec for cross-hardware determinism — "deterministic on one GPU" is not enough. Note: τ=0 alone does not solve prefix anchoring; replay only proves consistency with committed prefix, not true earlier execution. KV provenance (#5) or deep audit still needed for upstream trust. Keep as a research track, not a blocker before shipping the current dense strong tier. |
| 68 | **Deterministic attention / linear kernels** — custom Triton/CUDA kernels with fixed accumulation order and explicit cast/round semantics. Strong long-term direction for eliminating the bridge-boundary approximation class entirely (the Qwen `norm_quant` divergence is one instance of this class). No longer blocking Qwen in the short term (captured `x_attn` gives exact shell K), but still the clean fix for making the protocol independent of vLLM's fused kernels. Measure corridor=0 and throughput cost versus FlashAttention / stock CUTLASS. |
| 68a | **Kernel payoff / non-goals note** — make explicit in docs and roadmap that deterministic kernels mainly simplify dense-model arithmetic: newer dense Qwen/Llama/Mistral/Gemma-style families, tighter Freivalds semantics, and a much stronger FP8 story. They do not automatically solve MoE routing, MLA / nonstandard KV objects, speculative decode / MTP semantics, sparse attention rules, or shared-cache provenance. |
| 68b | **Post-kernel compatibility backlog** — even if deterministic kernels land, keep a separate protocol backlog ordered by practical importance: (1) cross-request cache semantics, (2) speculative / MTP semantics, (3) MoE routing, (4) sparse attention semantics, (5) MLA / unusual KV-cache architectures. Kernel work reduces arithmetic pain, but these remain transcript/control-flow extensions. |
| 69 | **Cheating-incentive analysis** — quantify cost/benefit of model substitution, detection probability vs audit sampling rate, equilibrium conditions |
| 70 | **Lean formalization** — machine-checked core verification claims |
| 71 | **Non-Rust verifier** — independent implementation consuming golden vectors |
| 72 | **llama.cpp plugin** — capture retained state from llama.cpp serving path |
| 73 | **Marketplace integration** — providers attach receipts, clients challenge through market surface |
| 74 | **Receipt encryption** — encrypt receipts/audit payloads to verifier's public key |
| 75 | **Generalized retained state** — schema-driven format beyond constant-width decoder models |
| 76 | **Full shell recomputation** — bandwidth-light, verifier-heavy mode. Future experiment requiring local weights. |
| 77 | **Receipt-format spec** — binary format documented independently of implementation |
| 78 | **API extensions** — serving API carries receipt metadata in stable, documented way |
| 79 | **Remove legacy verifier** — delete `verify_v4_legacy` after soak period |

---

## Benchmarks checklist

All benchmark items in one place for tracking.

| # | Item |
|---|------|
| 80 | Stable benchmark protocol |
| 81 | Record baselines before/after each runtime-affecting milestone |
| 82 | Periodic remote-GPU benchmark checkpoints |
| 83 | Treat unexplained regressions as blockers |
| 84 | Benchmark deep-audit open cost |
| 85 | Benchmark routine-audit path |
| 86 | Benchmark exact-prefix path |
| 87 | Inspect binary payload by field |
| 88 | Measure retained-state memory and audit bandwidth |
| 89 | Measure audit-window storage costs |
| 90 | Calibrate routine-audit detection probabilities |
| 91 | Benchmark batch verification |
| 92 | Rebenchmark after each online-path change |
| 93 | Run final kept-path benchmark suite |
| 94 | Explain benchmark methodology in paper and docs |

---

## Docs / publication detail

| # | Item |
|---|------|
| 95 | Make implementation-status split explicit |
| 96 | Production verification surface is binary-only |
| 97 | Freeze verifier-facing report contract |
| 98 | Paper/README/article normative to final protocol |
| 99 | Update with landed attention evidence: restored single-digit corridor, Llama bf16 strong-tier anchoring (~0.06), and Qwen diagnosis resolved — shell K is exact with captured `x_attn`; the L-inf 6–10 gap was bridge `x_attn` divergence under fused `norm_quant`, not a capture-binding bug |
| 100 | Full protocol documentation in README |
| 101 | Full protocol specification in paper |
| 102 | Explicit input-verification procedure |
| 103 | Transcript-chain and anti-splice procedure |
| 104 | Canonical deep-audit procedure |
| 105 | Decode/output support matrix in paper |
| 106 | One canonical randomness story: secrecy vs independence made explicit |
| 107 | Document verified-mode deviations from stock vLLM |
| 108 | Document supported/unsupported architectures |
| 109 | Document canonical semantics vs trust assumptions |
| 110 | Explicit trusted-assumptions section |
| 111 | Fail-on-unknown versioning rules |
| 112 | Document privacy implications of receipts and audits |
| 113 | Pipeline / boundary figure |
| 114 | Update article to match full protocol |

---

## Completed

| # | Item |
|---|------|
| — | ✓ Make quantization semantics first-class protocol data |
| — | ✓ Fix bridge trust boundary for attention (`x_attn_i8`, `scale_x_attn`, check-and-gate, `vi-retained-v3`) |
| — | ✓ Add committed KV transcript with `kv_root` covering full causal prefix |
| — | ✓ Clean old attention-gap measurement strategy (no confounded L∞=127/255, fail-closed on missing data, clean naming) |
