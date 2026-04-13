# Roadmap

All remaining work for CommitLLM, organized by priority. Items are numbered globally, with suffix letters for follow-on work inside an existing track.

Mark a task done only when it is true in code, tests, docs, and operational behavior. Completed milestones live in [CHANGELOG.md](./CHANGELOG.md).

The shell/tail/binding protocol is structurally complete. The Freivalds shell checks are gated by verification profile: Llama checks all 7 matrices (`Wq/Wk/Wv/Wo/Wg/Wu/Wd`); Qwen skips `Wq/Wk/Wv` because the bridge replay cannot match the GPU's quantized QKV GEMM (exact QKV Freivalds remains Llama-only until attention-side local verification or deterministic kernels close the gap). Token-identity acceptance is also profile-gated: Llama uses `ExactTokenIdentity`; Qwen now has a kept decode-side path in code via `DecodeAcceptanceMode::LpHiddenBf16`. The quantized `i8→i32` `lm_head` surrogate was rejected (honest gaps up to ~6500 on a ~7000 range), and the measured exact decode boundary is instead the hidden state at `LogitsProcessor args[1]`: captured LP hidden + verifier-side `bf16 lm_head` replay matched both greedy and sampled decode exactly on the measured Qwen runs at ~7 KB/token, far cheaper than captured logits. The remaining open problem is no longer decode; it is attention-side / prefix-side local verification and the real-weight end-to-end promotion gate for the new decode path. The `lm_head` Freivalds matmul check remains exact for all profiles. Tiers differ in attention evidence and payload cost.

## Linear Execution Order

Use this sequence as the single top-level plan. The tiers below provide detail;
this list defines the actual order to execute.

1. **Ship the first strong tier** — land the current Llama strong-tier path (`#2`, `#8`, `#15`, `#20`).
2. **Land Qwen strong tier via captured x_attn** — bridge x_attn diverges under fused norm+quant; captured x_attn gives exact shell K. Next proof point is an end-to-end Qwen strong-tier benchmark with captured x_attn: anchor gap, witnessed-replay L∞, payload, and verifier time (`#2f`, `#4`, `#8`).
3. ~~**Stabilize routine-tier decode semantics for approximate profiles**~~ — **DONE.** Bounded `lm_head` token acceptance was empirically tested on Qwen W8A8 (A100) and rejected: honest gaps range 0–6556, making any threshold meaningless. The kept answer is not tolerance tuning; it is a captured runtime boundary (`#4a`, `#4b`, `#25c`).
4. ~~**Prototype the decode-side captured boundary**~~ — **IMPLEMENTED / VALIDATED AT THE HOOK LEVEL.** `LogitsProcessor`-input hidden is now wired through the protocol surface with verifier-side `bf16 lm_head` replay. Remaining gate: real-weight GPU E2E and overhead measurement before this is described as fully shipped behavior (`#4b`, `#4c`, `#25c`).
5. **Promote the LP-hidden decode path end-to-end** — run real-weight Qwen greedy+sampled E2E on the `LpHiddenBf16` path, measure payload and capture overhead, and update docs/claims so Qwen decode is no longer described as unsupported (`#4c`, `#20`, `#25c`).
6. **Prototype local captured-boundary verification on the attention side** — replace long-chain bridge replay with per-layer local checks using captured boundaries. Measure exactness, payload growth, and verifier time. Define boundary policy per family (`#2g`, `#2h`, `#2i`).
7. **Close prefix trust and guarantee language** — KV provenance, audit tiers, guarantee wording, and the formal cheating-game story (`#7`, `#24`, `#25`, `#6`). Include security comparison of bridge vs local-boundary verification (`#25a`).
8. **Prove the protocol generalizes across dense families** — add one third dense family using the local-boundary path before any frontier architecture jump (`#31`, `#31a`).
9. **Broaden dense coverage** — larger dense model, longer context, and one non-W8A8 dense quant path (`#32`, `#33`, `#34`, `#35`).
10. **Take the frontier jump** — FP8 boundary policy first (`#30a`), then FP8 KV, then MoE and architecture variants (`#30`, `#35a`, `#29`, `#35b`, `#35c`, `#35d`, `#35e`).
11. **Improve serving/runtime compatibility** — native capture (including native captured-boundary backend `#36a`), batching, paging, TP, CUDA graphs, payload reduction (`#36` onward).
12. **Pursue verified-mode research** — deterministic kernels / deterministic runtime only after the current stock-kernel path is shipped and benchmarked (`#67`, `#68`).

### Architecture fork

Two verification paths, ordered by time-to-value:

1. **Stock-kernel path with captured local boundaries** (near-term) — keep vLLM's existing kernels and move strong-tier verification toward captured local boundaries instead of long-chain bridge replay. On the decode side, this is now the kept path: `LogitsProcessor`-input hidden with verifier-side `bf16 lm_head` replay. On the attention side, start with the already-proven `x_attn` boundary; extend to `x_ffn`, `h`, or residual capture only after `#2g/#2h/#2i` show that the local checks are accurate and the capture cost is acceptable. This is the intended dense-model path if the prototype succeeds.
2. **Verified-mode deterministic kernels** (research / backstop) — custom CUDA kernels with exact arithmetic. Eliminates the boundary-approximation class entirely but requires kernel engineering per op and a deeper runtime fork. Keep this as the long-term clean path and as the fallback if the stock-kernel path stalls on unsupported kernels, FP8 semantics, or frontier architectures. Deterministic kernels mainly solve dense-model arithmetic: they make newer dense families and FP8-style dense serving much easier to support by restoring exact or near-exact GEMM semantics and removing most linear-layer tolerance tuning. They do **not** by themselves solve MoE routing, MLA / nonstandard KV state, speculative decoding / MTP semantics, sparse attention semantics, or cross-request cache semantics; those remain protocol/transcript work even if the arithmetic layer becomes exact.

The long-chain bridge is downgraded from the main verification path to a routine-tier fallback / diagnostic tool. It remains useful where captured boundaries are unavailable, but it should no longer define strong-tier semantics for dense models once a measured local-boundary policy exists.

### Near-term decision gates

Use these as explicit promotion criteria between phases, not just as vague "next work":

1. **Llama strong tier ships first** — keep the current Llama path as the reference strong dense family only if GPU smoke tests, canonical verifier behavior, score witnessing, and docs/claims all stay aligned.
2. **Qwen only graduates after the captured-`x_attn` benchmark** — promote Qwen to a supported strong family only if the end-to-end benchmark with captured `x_attn` shows an acceptably tight anchor gap / witnessed-replay gap at reasonable payload and verifier cost. If not, keep Qwen routine-tier only rather than silently widening semantics.
3. **Decode boundary promotion depends on real-weight E2E, not just diagnostics** — `LogitsProcessor`-input hidden has now passed exact greedy + sampled hook-level validation. The remaining promotion gate is a real-weight GPU E2E on the end-to-end prover/verifier path plus a measured capture-overhead number.
4. **Local-boundary path starts with `x_attn` on attention and `LogitsProcessor` input on decode** — decode no longer needs float-logit capture or tolerance tuning. Do not pre-commit to `x_ffn`, `h`, residual capture, or full-logit capture until the cheap boundary candidates are measured first.
5. **Native captured-boundary backend is conditional, not automatic** — only invest in the native path after the protocol explicitly chooses local captured boundaries as the kept dense-model direction and after the boundary cost benchmark shows Python copy/materialization is the limiting factor. The LP-hidden decode path may justify native capture sooner if synchronous D2H proves too expensive in production.

---

## Tier 0 — Critical path

Highest-priority work. Do in this order.

Strict near-term execution order inside this tier:
1. `#2` land the Llama strong attention tier
2. `#2f` run Qwen captured-`x_attn` strong-tier benchmark
3. `#4` default captured `x_attn` for strong/full/score-witness audit opens, not only for `deep_prefix`
4. ~~`#4a` calibrate bounded `lm_head` token acceptance~~ — **DONE/REJECTED.** Bounded acceptance tested, gap tail too heavy (up to 6556). `DecodeAcceptanceMode::Unsupported` for Qwen; exact for Llama
5. `#4b` land `LogitsProcessor`-input hidden as the exact decode boundary (`bf16 lm_head` replay + canonical sampler)
6. `#4c` run real-weight Qwen E2E and capture-overhead benchmark on the LP-hidden path
7. `#2g` prototype local captured-boundary verification for `x_attn` only (per-layer local checks vs long-chain bridge)
8. `#2i` benchmark boundary capture cost and separate prover-internal capture volume from client-visible audit payload
9. `#2h` define strong-tier boundary policy per family from measured results, not assumptions
10. `#8` freeze generated-token score witnessing as the strong tier
11. `#15` freeze the canonical verifier
12. `#14` keep GPU smoke tests green
13. `#20` update docs/claims
14. `#21` cut the release

Items `#1`, `#3`, `#5`, `#6`, `#7`, `#9`, and `#10` remain important, but
they are supporting work around the main sequence above rather than reasons to
delay the first strong-tier ship decision indefinitely.

| # | Item | Status |
|---|------|--------|
| 1 | **RMSNorm contraction / local operator — MEASURED** — crude contraction ratio `ρ_j = γ·‖W_o‖₂/RMS(x^(j))` and tighter local sensitivity/operator tests have now been run on real checkpoints. **Qwen-7B W8A8**: crude contraction mixed; tighter local operator puts 21/28 layers below 1, but 3 layers (0, 1, 27) can still individually flip the output and many middle layers remain near the margin. **Llama-3.1-8B**: crude contraction fails everywhere and tighter local operator still leaves 0/32 layers below 1; 6 layers can individually flip the output. Conclusion: contraction and local linearization are useful diagnostics, but they do not rescue the corridor by themselves. Next step: refine the local map further with prompt diversity and the full within-layer propagation channel. Subtests: projected norm `‖J_RMSNorm(x_j) W_o^(j)‖`, head-aware/block-aware operators, mixed `∞→∞` and `∞→2` bounds consistent with the protocol corridor, finite-difference local sensitivity on real traces, and inclusion of the MLP propagation channel after the perturbed residual enters the rest of the layer. | **measured** |
| 2 | **Shrink the honest corridor / land strong attention tier** — prover-side determinism confirmed (§1h), sidecar-to-verifier gap measured at L∞=8–9 (§1i). That story is now split by family: **Llama-3.1-8B strong tier is real** with bf16 score anchoring gap ~0.06 and threshold tightened to 0.25; the remaining replay gap is downstream of score anchoring. **Qwen-7B strong tier is now unblocked but not yet end-to-end benchmarked**: the L∞ 6–10 anchor gap was caused by bridge-derived `x_attn` diverging from vLLM's fused `norm_quant` kernel — NOT a capture-binding bug. With GPU-captured `x_attn` as the QKV boundary, shell K matches the GPU QKV output with L-inf=0 for all positions (confirmed by 10-run Modal diagnostic). Next required measurement: Qwen captured-`x_attn` strong-tier end-to-end benchmark: anchor gap, witnessed replay L∞, payload MB, verifier ms. Backend binding is done (`attn_backend`/`attn_dtype` in manifest/key/verifier, SDPA mandated for W8A8, fail closed on eager/unknown). | **partial** |
| 2a | ~~**Bind attention backend in manifest**~~ — `attn_backend`, `attn_dtype` added to manifest/key/verifier with cross-checks. W8A8 mandates SDPA, fail closed on eager and unknown. Keygen populates from model config. | **done** |
| 2b | ~~**Measure real protocol gap (sidecar → verifier)**~~ — L∞=8 (Qwen 7B), L∞=9 (Llama 8B, Llama 70B) on A100-80GB with SDPA. ~93–97% exact, >99.9% within ±1. Gap grows weakly with sequence length. See §1i. | **done** |
| 2c | **Cross-hardware stability** — same backend on A100 vs H100. If hardware changes the gap, τ must accommodate hardware diversity. | open |
| 2d | ~~**Better `a` quantization**~~ — measured and ruled out as the main τ-reduction path. Per-head `scale_a` does not help (one head still dominates; INT8 corridor gets worse), and INT16 retained `a` does not materially shrink the float-space honest gap. No longer on the critical path. | **done** |
| 2e | **Layer/head-specific tolerances** — keep only as a routine-tier fallback. Strong-tier direction is family-specific score anchoring, not ever-finer τ tuning. Corridor worst layers remain useful for diagnostics: Qwen 1/27, Llama-8B 25, Llama-70B 7/13/70. | open |
| 2f | **Qwen `x_attn` boundary: captured path exact, bridge approximate** — 10-run Modal diagnostic (2026-04-08) proved: (1) shell K from captured `x_attn` + bf16 CUTLASS epilogue matches GPU QKV output K-slice with L-inf=0 for ALL prefix positions; (2) bridge-derived `x_attn` (f64 RMSNorm + quantize) diverges from vLLM's fused `norm_quant` kernel output for all positions; (3) the apparent pos 1+ divergence in earlier diagnostics was a measurement bug (RoPE applied in-place, hook captured post-RoPE K). **Next**: run Qwen score-witness strong-tier benchmark with captured `x_attn`, then enable the Qwen strong profile only if the score anchor gap is tight. Regression test: `test_qwen_shell_exact.py`. Long-term: canonical deterministic norm+quant kernel eliminates the bridge approximation class entirely. | **in progress** |
| 2g | **Local boundary prototype** — prototype local captured-boundary verification on one family (Qwen or Llama). Replace long-chain bridge replay with per-layer local checks using captured boundaries where available. Start with `x_attn` only for Qwen/Llama strong tier and measure exactness, payload, and verifier time. Only if the single-op RMSNorm/quant gap is tight enough should the protocol extend to `x_ffn` and `h`. | open |
| 2h | **Strong-tier boundary policy** — for each supported family, specify which boundaries are: (a) captured exactly, (b) locally checked, (c) replayed approximately, (d) unsupported. This prevents the protocol from silently mixing bridge-derived and captured semantics across audit tiers. | open |
| 2i | **Boundary cost benchmark** — quantify retained-state growth and online overhead for: (a) captured `x_attn` only (current), (b) captured `x_attn` + `x_ffn` + `h`, (c) optional residual capture. Report per-token/per-layer bytes, total prefill volume at 4K/32K, end-to-end latency impact, and keep prover-internal capture/hash volume separate from client-visible audit payload. | open |
| 3 | **Real-model corridor attacks — PARTIALLY MEASURED** — multi-prompt finite-difference sensitivity measured: Qwen 28/28 layers flip on every prompt, Llama 0–6 flips depending on margin. All-layers-simultaneously always flips. Remaining: constrained adversarial optimization (true worst-case under L∞), multi-token accumulation across autoregressive decode. See `research/attention-gap.md` §1e, §3l–3m. | **partial** |
| 4 | **Unify attention path** — one `x_attn` boundary, one QKV replay story across prover/verifier/corridor, and one correct historical-prefix binding story. No silent mixing of bridge-derived Q with committed K/V. The captured `x_attn` is the canonical QKV boundary for strong-tier verification. Remaining: make `server.audit()` default to captured `x_attn` for strong/full/score-witness audit opens whenever captured data is available (currently defaulted for `deep_prefix`), and document the bridge path explicitly as a routine-tier approximation. | open |
| 4a | **Routine-tier `lm_head` acceptance calibration — DONE / REJECTED** — for approximate profiles like Qwen, the `lm_head` matmul/Freivalds check remains exact, but final token identity can drift because quantized verifier logits reorder top tokens relative to GPU bf16 logits. Honest-gap measurement across real runs showed a heavy tail (up to ~6556 on a ~7000 range), so bounded token acceptance was rejected as a meaningful rule. Result: no threshold tuning. Future exact paths are captured `LogitsProcessor`-input hidden with bf16 `lm_head` replay or deterministic `lm_head` kernels. | **done** |
| 4b | **Decode boundary at `LogitsProcessor` input — IMPLEMENTED / DIAGNOSTICALLY VALIDATED** — measured on Qwen W8A8: `model.norm` is NOT the decode boundary (1/192 exact; diff 38–318 vs LP input), but the hidden state at `LogitsProcessor args[1]` is. After fixing the CUDA snapshot race, LP-hidden capture now passes exact hook-level validation: count alignment exact, ref match `max_diff=0`, greedy token replay `192/192`, sampled replay `32/32`, no prefill leak. The protocol surface now carries committed `lp_hidden_bf16`, `DecodeAcceptanceMode::LpHiddenBf16`, verifier-side `bf16 lm_head` replay, and keygen support for `lm_head_bf16`. Cost target remains ~7 KB/token, ~42× cheaper than captured bf16 logits. Remaining gate: real-weight GPU E2E and overhead measurement. | **partial** |
| 4c | **Qwen real-weight LP-hidden E2E + overhead** — run the real prover/verifier path on GPU with keygen-populated `lm_head_bf16`, prove greedy and sampled decode on the full protocol path, and report capture overhead for the synchronous D2H snapshot. Only after this should docs/claims stop describing Qwen decode identity as unsupported. Follow-up: if sync D2H overhead is too high, design a safe async snapshot scheme without allocator-reuse races. | open |
| 5 | **Adversarial testing** — cheating-provider test suite: receipt forgery, transcript splicing, intermediate tampering, model substitution, selective layer cheating, KV injection, and selective-abort / audit-policy gaming. Every attack must fail verification or be documented as accepted gap. Includes real-model constrained corridor attacks (see #3). | **in progress** |
| 6 | **Formal security argument** — define the cheating game, state Freivalds soundness bound (1/p), state detection probability as f(sampling rate, cheating fraction), state what commitment binding prevents. Include random audit conditioning analysis: detection probability vs cheating fraction vs audit rate. | open |
| 7 | **KV provenance** — sampled/batched Freivalds checks verify committed K/V are consistent with committed Wk/Wv weights and the bound `x_attn` inputs. Tightening current-token attention alone is insufficient if the adversary can manipulate prefix state; KV provenance closes the upstream path. With #2f resolved (captured `x_attn` gives exact K), the provenance check can now use exact equality for models where captured `x_attn` is available, falling back to corridor bounds only for the bridge path. | open |
| 8 | **Score witnessing** — generated-token score witnessing is now the live strong-tier path. **Llama:** bf16 score anchoring is validated and should be treated as the first supported strong family. **Qwen:** no longer blocked — with captured `x_attn` as the QKV boundary (#2f), shell K is exact and score witnessing can proceed on the same footing as Llama. Keep the generated token exact by default; sample prefix tokens only as a later probabilistic extension. Compression ideas (top-k + tail bounds) are optimization work after the plain exact tier is benchmarked. | **partial** |
| 9 | **W_o conditioning** — compute σ_min(W_o), ‖W_o‖₂, and more faithful projected/local operator bounds per layer on real models. **Confirmed critical by #1**: Llama ρ>1 everywhere means the crude spectral bound is loose — need tighter per-layer operator norms to identify which layers genuinely amplify vs where the bound is just conservative. Quantifies adversary freedom when attn_out_i8 is commitment-bound but not independently replayed. Also translate hidden-state drift into per-layer logit-margin bounds using the real LM head, and measure margin-plus-frequency on real prompts to see when dangerous layers actually coincide with small output margins. | open |
| 10 | **Deep-audit payload sizing** — budget the per-token cost of score witnessing and KV provenance at production context lengths (4K, 32K, 128K). Determine storage, bandwidth, and verification-time scaling. **Must benchmark CPU verification time in the Rust verifier**, not estimate from FLOPs (likely memory-bound). Note: for Fiat-Shamir-sampled decode tokens, average prefix length applies, not max context. Informs audit tier design and whether long-context deployments need different policies. | open |
| 11 | **Rename to CommitLLM** — all docs, APIs, CLI, scripts, packages use `CommitLLM`. | open |
| 12 | **Remove V4/V5 naming** — public surface uses one canonical release name, not internal version numbers | open |
| 13 | **Clean repo** — delete stale corridor/debug paths, one-off hooks, duplicate helpers, obsolete protocol-era code. Repo should reflect the kept path, not an archaeology of experiments. | open |
| 14 | **GPU smoke tests** — maintained GPU tests exercise token-0 attention replay and deep-prefix on the canonical verifier | open |
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
| 24 | **Audit tiers** — formalize receipt-only / routine / deep / full with explicit coverage and cost per tier |
| 25 | **Guarantee language** — docs clearly separate exact (Freivalds/INT8), bounded approximate (attention), statistical (sampled provenance), and fail-closed |
| 25a | **Bridge vs local-boundary security comparison** — explicitly compare: (a) long-chain bridge replay, (b) local captured-boundary checks, (c) what each proves and does not prove about nonlinear transitions. The guarantee language changes: local-boundary checks verify each nonlinear op independently (no accumulated drift) but still require per-op tolerance or exactness claims. Document which nonlinear ops (RMSNorm, SiLU, quantization, residual add) are exact vs approximate under each verification mode, and state clearly that the bridge remains a routine-tier fallback once local boundaries are adopted for strong-tier dense models. |
| 25b | **Local nonlinear transition checks** — formalize RMSNorm, quantization, SiLU/gating, and residual update checks as local constraints with exact vs approximate status clearly stated per op. Quantization should be exact; RMSNorm and SiLU may have small fused-kernel gaps that need measurement. |
| 25c | **Decode acceptance semantics** — specify per profile whether final-token acceptance is exact token identity, bounded token acceptance under quantized replay, unsupported, or captured-boundary exact (`LpHiddenBf16`). This is separate from `lm_head` Freivalds: the linear algebra check can remain exact while the decode-acceptance rule changes by profile. Current state: Llama uses `ExactTokenIdentity`; Qwen has an implemented `LpHiddenBf16` path in code, pending real-weight E2E before docs/claims treat it as fully shipped. Document how greedy vs sampled decode is handled, what randomness is or is not replayed exactly, and how each decode mode is reported to users. |
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
1. `#31` Third dense family (`Mistral` / `Ministral` / `Nemo` or `Gemma`)
2. `#30` FP8 support
3. `#29` MoE support

Reason: prove the protocol is not overfit to Llama/Qwen before taking on the
frontier FP8+MoE architecture jump.

Without broader model support, CommitLLM remains a Qwen+Llama-focused protocol
rather than a general verifier for the full frontier serving stack.

| # | Item |
|---|------|
| 29 | **MoE support** — verify expert routing decisions and per-expert shell matmuls for Mixtral, DeepSeek-V2/V3, Qwen-MoE. Commit router logits/top-k, Freivalds on selected expert weights, fail-closed on unsupported routing. This is a serious protocol extension and should come after FP8 and a third dense family. |
| 30 | **FP8 quantization** — split into two explicit tracks: (a) compat mode with bounded replay against vendor FP8 kernels, and (b) verified mode with a canonical exact/fixed-point lowering or custom deterministic kernel. Do not start before the current dense slice and third-family support are stable. If the deterministic-kernel path is pursued, FP8 should move closer to that work because owning the arithmetic is the main reason to do FP8 in a verifier-friendly way. |
| 30a | **FP8 boundary policy** — decide whether FP8 uses captured boundaries, compat replay, or verified-mode kernels. This should be explicit before FP8 implementation starts, not implied from the W8A8 path. FP8 decode semantics differ from INT8 and may require different boundary capture or tolerance strategies. |
| 30b | **Kernel payoff boundary** — document explicitly what deterministic kernels buy: dense-model GEMM exactness, cleaner Freivalds semantics, much less tolerance tuning, and a much stronger path to FP8 dense serving. Also document what they do **not** buy automatically: routing semantics, nonstandard cache objects, speculative accept/reject logic, sparse attention rules, or shared-cache provenance. |
| 31 | **Third family** — prioritize `Mistral` / `Ministral` / `Nemo` or `Gemma` before FP8/MoE. Goal: prove the kept path is not overfit to Llama/Qwen while staying in dense-decoder territory. |
| 31a | **Third-family local-boundary validation** — validate the local captured-boundary path on the third dense family (Mistral or Gemma) with the same architecture, before FP8/MoE. Confirms the local-boundary design generalizes beyond Llama/Qwen. |
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
