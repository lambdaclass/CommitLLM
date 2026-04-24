# Changelog

This changelog tracks the kept canonical VeriLM protocol and its major implementation milestones.

Historical references below to “roadmap #N” refer to the pre-2026-03-30 roadmap numbering. On 2026-03-30 the roadmap was renumbered into a single linear open-items-only sequence.

## 2026-04-21

### Added

- **`verilm_rs.deserialize_v4_audit(audit_binary) -> dict`** in `verilm-py`. Exposes the publicly-committed fields of a V4 audit binary (`output_text`, `prompt`, `prompt_hash`, `input/model/decode/output_spec_hash`, `manifest_hash`, `n_tokens`, etc.) as a Python dict. Internal verification state (Merkle proofs, retained state, shell openings, KV entries/roots/proofs, prefix embeddings, witnessed scores) is intentionally not surfaced. Enables relay-layer callers that already trust the binary (having passed `verify_v4_binary`) to extract the committed output text directly rather than accepting a parallel "claimed output" value from the client.

## 2026-04-20

### Measured

- **`StockBounded` attention is now conclusively dead as a shipped guarantee.** The first `k=16` sweep was already weak (`23%` Qwen, `28%` Llama). Follow-up work made that verdict decisive rather than provisional:
  - adaptive-`k` + tail-bound + margin gating
  - explicit budgeted greedy allocation
  - exact `o_proj` sensitivity (`alpha[l,h]`)
  - scalar downstream gain calibration (`beta[l]`)
  - directional margin sensitivity (`gamma[l,h]`)
  All still failed by orders of magnitude. The budgeted certifier reached **0% certification** on the broad rerun, almost entirely as `budget_exhausted`, and the directional-gamma diagnostic still left the bound roughly `10^3–10^4` above the `margin / 2` target. This is not a threshold-tuning problem; the entire stock-bounded certification line is rejected.
- **Deterministic exact attention works technically but is not the product path.** The deterministic kernel, CPU replay, and commit/open/verify plumbing are all real and exact on their supported slice, but the operational benchmark makes the product tradeoff clear:
  - stock decode median: `21.58 ms/token`
  - deterministic decode median: `71.16 ms/token`
  - overhead: `+49.58 ms/token` (`+230%`)
  - greedy agreement vs stock: `1/30`
  This is valuable as a research/reference path, not as the kept shipped runtime.

### Decided

- **The kept product claim is now: exact decode, audited attention inputs/wiring, no arbitrary-position attention verification.** The project no longer treats either `StockBounded` certification or deterministic attention mode as part of the mainline product guarantee.
- **The attention audit surface is now explicitly narrower and honest.** The kept stock-mode attention work is:
  - exact score anchoring (`QK^T / sqrt(d)`)
  - KV provenance / cache-row correctness
  - mask / RoPE / GQA wiring checks
  - token-0 / local replay smoke checks and offline differential testing as regression tools
  These are useful audits, but they are not full attention verification.
- **The failed attention branches are archived, not promoted.** No more stock-bounded certification work, no more replay-shape or witness tuning, and no more deterministic-kernel optimization on the main product path.

### Known issues

- **Arbitrary-position attention outputs are not verified in stock mode.** The verifier can still audit scores, KV provenance, and attention wiring, but it cannot honestly claim full `softmax(scores) @ V` verification on arbitrary positions.
- **Paper cleanup still pending.** The verifier, README, and roadmap now carry the audit-only claim: the `StockBounded` variants on `AttentionVerificationMode` and `AttentionStatus` are renamed to `AuditedInputsOnly` / `AuditedInputsNotVerified`, the stock-bounded certification hot path is removed, profile builders/names are `*_audited`, and the README verification-coverage table states the audit-only claim explicitly. `paper/main.typ` still specifies attention replay (§Attention Replay, Proposition 2, Table 1 CommitLLM row) and is the remaining cleanup; it is scheduled as roadmap item 11a (deferred).

## 2026-04-19

### Measured

- **CapturedLogits economics are now measured enough to isolate the dominant decode-side cost.** Across both dense families and 128/512/1024-token generations, retained sampled-decode state stays in the same `~500–600 KiB/token` range already established. The new A/B split shows:
  - **hook/copy overhead**: about `~3–4 ms/token` (`~20–25%` over plain generation)
  - **commit/finalize cost**: dominant, roughly `~55–100 ms/token`
  The decode-side operational bottleneck is commit/materialization, not the sampler hook itself.
- **`StockBounded` attention certification was measured and rejected as the mainline attention answer.** On the first broad certification sweep (`k=16`, mass threshold `0.9`, temperature `0.8`):
  - **Qwen 7B W8A8**: `23/100` certified
  - **Llama 8B W8A8**: `28/100` certified
  - **Low logit margin was not the bottleneck**: `0/100` failures on margin for both families.
  - **Attention concentration was the bottleneck**: roughly `72–77%` of challenged tokens failed because top-k mass was too diffuse. This is too weak for the main shipped attention story.

### Added

- **`StockBounded` verifier plumbing and attention evidence prototype.** The verifier surface now includes a stock-bounded attention mode, explicit attention status reporting, attention-evidence computation, and token-certification plumbing. This keeps the practical stock-compatible tier as an explicit best-effort path rather than an implicit replay/tolerance story.
- **Deterministic attention arithmetic spec, CPU reference, and CUDA parity harness.** The exact verified-attention path now has:
  - a frozen arithmetic spec in [docs/design/deterministic-attention-spec.md](./docs/design/deterministic-attention-spec.md)
  - CPU reference replay in Rust
  - standalone CUDA kernel and parity harness
  - explicit round-to-nearest-even handling (`rintf` / `round_ties_even`) after fixing the only observed parity bug

### Validated

- **Deterministic attention arithmetic is now frozen for the first verified-attention slice.** After the rounding fix, the standalone CPU and CUDA implementations are bit-exact on:
  - **10,000/10,000 randomized parity tests**
  - **45/45 targeted edge cases**
  Scope: decode-time attention, `seqlen_q = 1`, dense GQA, `head_dim = 128`, A100/SM80, post-RoPE `Q/K`, bf16 inputs, f32 arithmetic. This validates the arithmetic contract strongly enough to freeze it and treat future arithmetic changes as protocol changes.

### Decided

- **`StockBounded` remains a narrow stock-compatible tier, not the mainline attention guarantee.** It is useful as an opportunistic certification layer when the bound passes, but the measured `23–28%` certification rate is too low for the main production attention story.
- **Verified-attention mode is now the mainline exact-attention path.** The next engineering step is no longer arithmetic research; it is integration of the frozen deterministic attention contract into vLLM as an explicit verified-attention mode, followed by drift and throughput measurement against stock mode.

### Known issues

- **Verified-attention mode still needs runtime integration.** The arithmetic contract is frozen and the standalone parity harness is green, but vLLM integration, profile/runtime wiring, fail-closed serving behavior, and drift/throughput benchmarks versus stock mode are still open.
- **Stock-compatible arbitrary-position attention remains best-effort only.** Until the low-bandwidth bound can certify more than a minority of tokens, stock-compatible mode should treat arbitrary-position attention as opportunistic/experimental certification or unsupported beyond token-0 smoke/reference.

## 2026-04-18

### Measured

- **Tiled online-softmax replay was rejected as the stock-kernel attention fix.** On both Qwen and Llama, global replay and tiled replay (`block_n=64` / `128`) produced identical `L∞` gaps on every measured run (`delta = 0`). Matching FlashAttention tile order does not reduce the arbitrary-position attention gap.
- **`LSE` feasibility was also rejected.** Capturing FlashAttention decode-time `LSE` does not let the verifier recover stock-kernel attention closely enough to matter:
  - **Qwen 7B W8A8**: max `|CPU_LSE - GPU_LSE| = 6160`
  - **Llama 8B W8A8**: max `|CPU_LSE - GPU_LSE| = 15.6`
  - Even when replay uses the exact GPU `LSE`, the attention-output gap stays in the same `L∞ ≈ 182–254` range. This isolates the remaining mismatch to the stock kernel's `P @ V` aggregation / cast path, not softmax normalization.
- **CapturedLogits A/B economics are partly isolated.** On the completed Qwen run, capture hooks added about `~3.6 ms/token` over baseline generation, while commit/finalize added about `~37 ms/token`. This is enough to say capture itself is not the dominant decode-side cost; commit/materialization is. The Llama capture A/B still needs a clean rerun because one Modal capture leg failed to receive a GPU.

### Decided

- **Exact arbitrary-position stock-kernel attention replay is closed.** The project no longer treats kernel-aligned replay as the mainline exact-attention path. Tiled replay and `LSE` replay both failed; no more replay-shape or softmax-normalization tuning belongs on the critical path.
- **The dense-model story now has two explicit modes.**
  - **Stock-compatible mode** keeps the current vLLM / FlashAttention runtime and therefore preserves current answer semantics. Shell and decode remain exact on the kept paths (`CapturedLogits` for sampled decode). Arbitrary-position attention is either certified by a practical low-bandwidth non-hackability tier (`top-k` / tail bound / final-logit margin gating) or reported as unsupported.
  - **Verified-attention mode** replaces stock attention on the verified path with deterministic attention kernels. This is the exact-attention mode. It may change answers relative to stock mode and must be documented as a distinct runtime/profile.
- **Deterministic attention kernels are now a separate verified mode, not the default runtime.** The default plan is to keep stock semantics where possible and only use verified-attention mode when the stock-compatible practical tier cannot certify the answer or when stronger guarantees are required.

### Known issues

- **Stock-compatible arbitrary-position attention is still open.** `LSE` and tiled replay are both rejected. The remaining stock-kernel option is a practical low-bandwidth tier based on tiny attention evidence, tail bounds against committed `V`, and exact final-logit margin gating. Until that lands, stock-compatible mode should treat arbitrary-position attention as unsupported beyond token-0 smoke/reference.
- **Verified-attention mode is not implemented yet.** Deterministic kernels are the clean exact path, but they imply a distinct runtime mode and may drift from stock answers. CPU/GPU arithmetic spec, harness, integration, and drift benchmarks are all still open.

## 2026-04-17

### Measured

- **Shell/decode benchmark on A100-80GB** — first systematic measurement of the current E2E verification surface (everything except arbitrary-position attention), across greedy/sampled × short/long generation × routine/full tiers.
  - **Llama 8B W8A8**: ~15ms verify (full tier, 32 tokens), ~4.1MB payload per audit, ~1.7s audit-open time. Cheap enough for routine use.
  - **Qwen 7B W8A8**: **~1.87s verify** per audited token (full tier, 32 tokens), ~4.2MB payload, ~1.75s audit-open time. The CPU cost is dominated by the full 152,064×3,584 bf16 lm_head matmul in `LpHiddenBf16` decode verification. Correct but too expensive for routine audits.
  - Qwen: all configurations pass (greedy + sampled, short + long, routine + full).
  - Llama greedy: all configurations pass.
  - **Llama sampled: intermittent failures on ExactTokenIdentity (i32) path** — ~20% of sampled decode audits fail `lm_head: expected token X but got Y`. Failures are stochastic (same prompt sometimes passes, sometimes fails), not clustered at gen_index==0 (no off-by-one), and hit diverse token positions. Root cause: verifier samples from i32 quantized logits which diverge from the GPU's real bf16 logits; with top_k/top_p this difference is enough to flip the sampled token. Llama greedy on i32 is unaffected because the argmax is stable.
- **Llama A/B diagnostic (30 runs × 10 prompts = 300 comparisons, T=0.8 top_k=50 top_p=0.9)**:
  - Path A (i32 ExactTokenIdentity): **239/300 (79.7%)** — fail rate varies 7-33% per prompt.
  - Path B (LP-hidden bf16): **296/300 (98.7%)** — dramatically better, but not 100%. All 4 Path B failures also failed Path A (hardest cases where GPU output diverges from both CPU replay methods).
  - LP-hidden bf16 verify cost on Llama is comparable to Qwen (~1.87s/token CPU) — the same cost blocker applies.
  - Run timed out at 3600s after 10/12 prompts; remaining 2 prompts would not change the conclusion.

### Added

- **Profile override for keygen**: `generate_key_binary_with_profile()` API allows generating verifier keys with non-default verification profiles (e.g., `llama-w8a8-lp-hidden` for A/B testing Llama with LP-hidden bf16 decode instead of its default ExactTokenIdentity).
- **`DecodeAcceptanceMode::CapturedLogits` — exact sampled decode via captured GPU logits.** Full end-to-end implementation across all crates:
  - **Core types**: `CapturedLogits` variant, `captured_logits_f32` on `ShellTokenOpening`, `v_lm_head_f64` and `captured_logits_freivalds_seed` on `VerifierKey`, profiles `llama-w8a8-captured-logits` and `qwen-w8a8-captured-logits`.
  - **Merkle**: `hash_retained_with_captured_logits()` with domain separator `"vi-retained-logits-v1"`.
  - **Freivalds**: `derive_pm1_vector()` (CSPRNG ±1 from seed), `precompute_v_lm_head_f64()` (v = r^T @ lm_head_bf16 in f64), `check_captured_logits()` (r · logits ≈ v · hidden).
  - **Keygen**: CapturedLogits profiles generate Freivalds seed + precomputed v_lm_head_f64 from bf16 weights.
  - **Prover**: `MinimalBatchState.captured_logits_f32`, bound into Merkle leaf at commit, included in shell opening.
  - **Verifier**: `phase_lm_head_captured_logits()` — exact sampling check + Freivalds binding.
  - **Python bindings**: `commit_minimal_from_captures()` accepts `captured_logits_f32`; profile names registered.
  - **Sidecar**: `CanonicalSamplerHook` captures `logits_np` per-token; server drains and passes to Rust commit.
  - **Decode proof structure**: exact token sampling from captured GPU logits + probabilistic algebraic binding (Freivalds) back to lm_head. No CPU matmul replay needed.
  - **Cost structure**: (a) Prover-side retained state: ~501 KiB/token (Llama) or ~594 KiB/token (Qwen) held in memory until audit. For 256 tokens: ~125 MiB (Llama) or ~149 MiB (Qwen) retained. (b) Audit bandwidth: +~0.5-0.6 MiB per *opened* (challenged) token — not per whole answer. The Merkle commitment binds the logits; only opened tokens ship them. (c) Verifier CPU: two dot products per challenged token (negligible vs the eliminated ~1.87s/token matmul replay).
  - **Decode policy**: sampled decode → CapturedLogits (exact). Greedy decode → cheaper paths (i32 or LP-hidden) remain acceptable.

### Validated

- **CapturedLogits decode is GPU-validated end-to-end on both dense families.** Modal A100-80GB E2E runs with the `llama-w8a8-captured-logits` and `qwen-w8a8-captured-logits` profiles now pass decode verification cleanly:
  - **Qwen 7B W8A8**: `3/3` greedy, `20/20` sampled, `17,336/17,336` decode checks passed.
  - **Llama 8B W8A8**: `3/3` greedy, `20/20` sampled, `23,335/23,335` decode checks passed.
  - Tamper rejection passes on both models.
  - The remaining Llama failures in those runs are the known arbitrary-position attention gap, not decode.
- **Sampled decode exactness is now closed by boundary capture, not replay.** The replay-based paths remain historically important diagnostics:
  - `ExactTokenIdentity` on sampled Llama is rejected (`239/300 = 79.7%`).
  - `LpHiddenBf16` improves sampled replay (`296/300 = 98.7%`) but is still not exact.
  - `CapturedLogits` is the kept path because it samples from the exact GPU logits and only uses algebraic binding to tie them back to `lm_head`.
- **CapturedLogits keygen is expensive but offline.** First-run keygen for captured-logits profiles now includes the extra `v = r^T @ lm_head_bf16` precompute:
  - **Qwen 7B**: ~55.4s
  - **Llama 8B**: ~152.0s
  This is an artifact-build cost, not a steady-state verifier cost. Keys and decode artifacts should be cached.

### Known issues

- **Llama sampled decode on i32 path: rejected.** ExactTokenIdentity fails ~20% of sampled decode audits. The i32 quantized logits diverge enough from GPU bf16 logits that top_k/top_p sampling flips tokens. Greedy is unaffected (argmax is stable). This path is dead for sampled decode.
- **Llama sampled decode on LP-hidden bf16 replay: promising but still not exact.** LP-hidden bf16 passes 98.7% (296/300) vs i32's 79.7% (239/300). The improvement confirms the main problem was the quantized logit path. But 98.7% is not "exact sampled decode." The 4 residual failures prove CPU replay arithmetic ≠ GPU tensor-core arithmetic for sampled-token probabilities — this is not about quantization anymore, it is about accumulation-order divergence in the lm_head matmul. Neither current replay path achieves exact sampled decode.
- **CapturedLogits economics are now the decode-side question.** Correctness is closed; the remaining work is operational: measure capture overhead, retained memory, open payload, and steady-state verifier cost with cached keys/artifacts. The prover must retain/commit all sampled-token logits because the challenge is chosen after generation.
- **Arbitrary-position attention is the main remaining dense-model correctness blocker.** Decode and shell audits are now green on the kept path. Attention still needs either a FlashAttention/kernel-aligned witness with hard success criteria or deterministic attention kernels. Captured attention outputs are useful plumbing, not proof.

## 2026-04-16

### Fixed

- **f32 accumulator for LP-hidden bf16 lm_head replay**: the canonical verifier was accumulating the bf16 lm_head matmul in bf16, but GPU tensor cores accumulate in f32 within tiles. Changed the accumulator from `half::bf16::ZERO` to `0.0_f32` in `canonical.rs`. This fixed 3/~20 argmax divergences on Qwen sampled decode.
- **Generation-local seed indexing in all verifier sampling sites**: the prover's sampler uses a generation-local `_call_count` (0-based from the first generated token), but the verifier was passing the absolute `token_index` to `derive_token_seed`. Fixed all three call sites (`canonical.rs` ExactTokenIdentity path, `canonical.rs` LpHiddenBf16 path, `lib.rs` adversarial path) to use `token_index - gen_start` where `gen_start = n_prompt - 1`. This fixed the remaining 2/~20 sampled decode divergences.
- **Tier assertion in E2E tests**: routine-tier audits on 28-layer Qwen can randomly select all layers, causing the verifier to report "full" coverage. Tests now accept "full" when "routine" was requested.

### Confirmed

- **Qwen sampled decode passes E2E on LP-hidden bf16 path (261/261)**: after the f32 accumulator and seed indexing fixes, Qwen E2E passes all checks (greedy + sampled decode, routine + full tiers, tamper rejection). Adversarial 36/36 passed. **Caveat (added 2026-04-17)**: the Llama A/B diagnostic at 300 runs exposed a ~1.3% residual failure rate on LP-hidden bf16 sampled decode due to CPU-vs-GPU accumulation-order divergence. Qwen's 261/261 was a smaller sample; the same failure mode may exist. A Qwen-specific A/B stress test at comparable scale is needed before claiming Qwen sampled is exact.
- **Llama greedy decode is fully green on ExactTokenIdentity (i32) path**: Llama E2E greedy checks all pass. Llama sampled appeared green in the narrow E2E test but the broader benchmark (2026-04-17) exposed ~20% failure rate on the i32 path and ~1.3% on LP-hidden bf16.
- **The LP-hidden bf16 decode path failures were verifier bugs, not fundamental** (partially revised): the Qwen sampled-decode divergences at the time traced to accumulator and indexing bugs. Fixing those bugs made the path much better. However, the Llama A/B diagnostic (2026-04-17) proved that even with correct accumulator and indexing, CPU bf16 replay does not perfectly match GPU tensor-core arithmetic for sampled decode — a residual ~1.3% gap remains.

### Known issues

- **Arbitrary-position attention remains the only major dense-model gap**: decode and shell audits are shipped and green. Attention at arbitrary positions is still not verified on stock kernels. The next accepted paths are a FlashAttention/kernel-aligned witness (Path B) or deterministic attention kernels (Path C). Committed attention-output boundaries (Path A) are useful plumbing but do not prove attention correctness.

## 2026-04-14

### Added

- **Explicit attention-verification routing in the canonical verifier**: `VerificationProfile` now carries `AttentionVerificationMode` with `ExactReplay` and `WitnessedScores`. The verifier routes attention checking through an explicit mode instead of overloading `score_anchor_threshold` as an implicit selector. Current production test policy is stricter than the available phases: Llama keeps exact attention replay only as a token-0 smoke/reference check, and Qwen witnessed-score replay is diagnostic only until a kernel-aligned witness or deterministic attention kernel lands.
- **Standalone witnessed-score attention phase**: `phase_witnessed_score_attention` is now a first-class verifier phase. It performs (1) witnessed-score structural checks, (2) score anchoring against canonical `QK^T / sqrt(d)`, (3) softmax replay from the witnessed score rows, and (4) comparison against committed `a` with the local requantization tolerance. This separates the “witnessed score” claim cleanly from raw `Q/K/V` replay, but it is not promoted as a production strong-tier rule after the Qwen sweep breached the fixed tolerance.
- **Exact attention reference phase**: `phase_exact_attention` is now kept as a pure reference-tier phase that replays canonical attention directly from reconstructed `Q` plus committed `K/V`. It no longer doubles as the stock-kernel Qwen strong-tier path.
- **E2E everything-except-attention policy**: the maintained Modal E2E tests now target the full production wiring except arbitrary-position attention. They cover binary key/decode artifacts, commitment-derived random generated-token challenges, greedy and sampled decode, routine/full shell openings, long prompts/generations, multi-position escalation, tamper rejection, EOS, payload size, and verifier timing. Random-position audits omit KV attention openings (`include_kv=false`); token-0 attention remains a smoke/reference check only.
- **Binary-key inspection API**: `inspect_key_binary()` exposes small verifier-key metadata to Python tests without JSON-serializing production-scale keys.

### Measured

- **Qwen brute-force exact attention replay is a reference tier, not the kept path**: with committed KV transcript and canonical f64 replay, token 0 passes under a `±1` LSB tolerance at the final `a` requantization boundary, but later tokens still diverge heavily (`max_diff` on the order of `100–170` in i8 space). This is not a small rounding issue; it is the stock-kernel attention arithmetic gap showing up across longer prefixes. Conclusion: brute-force exact replay remains useful as an upper-bound payload/CPU benchmark and diagnostic tier, but it is not the honest Qwen production attention path.
- **Witnessed-score and exact replay are now treated as different claims, not two checks on the same claim**: the earlier benchmark confusion came from routing witnessed-score logic through a separate corridor/anchoring path while `phase_exact_attention` still enforced raw f64 replay first. The verifier now has an explicit phase split so Qwen is no longer blocked by the wrong attention claim.
- **Initial Qwen witnessed-score attention looked close on narrow late-token audits, but this was superseded by the broader sweep**: reruns on short / medium / long prompts with the **last generated token** reduced the old `100–170` scale gap down to `max_diff = 2–3` on only a few layers. That result was useful diagnostically, but it was not a valid production bound.
- **Qwen witnessed-score f64 softmax·V replay was stress-tested and rejected as a production strong-tier path**: the broader tolerance sweep over 39 prompts breached the proposed `±3` bound. Global max diff reached `9`; only `23/39` prompts passed at `±3`; failures were deterministic and dominated by layer 11. Breakdown: factual prompts were worst (`1/6`, max `9`), adversarial `3/6` max `4`, code `4/7` max `4`, prose `5/7` max `4`, math `5/6` max `6`, mixed `5/7` max `6`; greedy was not safer than sampled (`15/24`, max `9` vs sampled `8/15`, max `6`). Conclusion: fixed-tolerance replay of softmax·V from witnessed scores is not an honest Qwen strong-tier rule. The remaining serious stock-kernel branch is a later/kernel-aligned witness (for example FlashAttention block-state or post-aggregation boundary witness); otherwise the clean fix is a deterministic attention kernel path.
- **Llama arbitrary-position exact attention is not solved by stock f64 replay either**: token-0 exact attention can remain as a smoke/reference check, but non-zero token positions can diverge substantially from GPU FlashAttention/bf16. The E2E tests therefore omit KV attention openings for random generated-token audits rather than widening attention tolerance.

### Known issues

- **Last-token LP-hidden opening bug remains separate from the attention result**: in the same benchmark runs, the final generated token can still miss `lp_hidden_bf16` in the shell opening even when the attention path is otherwise behaving as expected. This is an independent commit/open bug and should not be conflated with the witnessed-score attention tail.
- **Arbitrary-position attention remains unresolved on stock kernels**: decode and shell audits are shipped/covered, but Llama and Qwen should not claim arbitrary-position attention verification via brute-force f64 replay or fixed-tolerance witnessed-score replay. The next accepted paths are either a later/kernel-aligned witness or deterministic attention kernels.

## 2026-04-13

### Shipped

- **Qwen W8A8 decode verification is now fully shipped on the binary-key path.** The complete LP hidden → bf16 lm_head matmul → canonical sampler path passes real-weight E2E on Modal A100-80GB: **261/261 verifier checks on every run**, across 3 greedy prompts, 2 sampled prompts (temp=0.8, top_k=50, top_p=0.9), and an EOS-trim regression test. Tamper detection confirmed. This closes the decode-side promotion gate — Qwen token identity is no longer “unsupported in principle.”

### Measured

- **LP-hidden decode capture is exact on the real sampler boundary**: the `LogitsProcessor` capture hook passes `23/23` validation checks on Qwen W8A8. Count alignment is exact on both greedy and sampled runs, dtype is preserved as `bf16`, reference-hook match is exact (`max_diff = 0.0`), greedy token replay is `192/192`, sampled replay is `32/32` exact under the canonical sampler with the committed seed, and no prefill rows leak into the decode trace. Root cause of the earlier mismatch was CUDA allocator reuse before an async D2H copy completed; the kept path now snapshots on the producer stream before copying.

### Added

- **Binary verifier-key API**: `generate_key_binary()` and `verify_v4_full_binary()` added to the Python API. The JSON key path is impractical for models with `lm_head_bf16` (Qwen 7B key is 1.57 GB binary / ~2 GB JSON). The binary path (VKEY magic + bincode) is now mandatory for `LpHiddenBf16` profiles.
- **Committed LP-hidden decode path in the protocol surface**: `ShellTokenOpening` now carries `lp_hidden_bf16`, the prover commit/open path stores per-token LP hidden, and the retained-state hash binds LP hidden with a dedicated domain separator. The canonical verifier gained `DecodeAcceptanceMode::LpHiddenBf16`, a `bf16` `lm_head` replay path for greedy and sampled decode, and `VerifierKey.lm_head_bf16` so token identity can be checked from the committed runtime boundary instead of the broken `i8→i32` surrogate.
- **Keygen support for decode-boundary replay**: key generation now loads raw `bf16` bit patterns from `lm_head.weight` and stores them in the verifier key when the selected verification profile requires `LpHiddenBf16`. The verifier therefore replays the exact `bf16` `lm_head` surface the decode path expects, rather than silently re-deriving a different dtype path.

### Known issues

- **Verifier key size**: `lm_head_bf16` for Qwen 7B is 544M u16 elements (152064 × 3584) = 1.57 GB binary. This makes verifier-key distribution a first-class product concern. Options: keep in key, split to separate artifact, or derive differently. JSON key path is dead for these models.

## 2026-04-12

### Measured

- **Qwen W8A8 tier-aware benchmark passes cleanly on the honest supported profile**: Modal A100 benchmark `scripts/modal/bench_e2e_v4.py` passes **20/20** across both audit tiers. **Full tier**: 10/10 pass, 261/261 checks, mean verify time ~16.6 ms, mean binary payload ~4.1 MB. **Routine tier**: 10/10 pass, 397/397 checks, mean verify time ~14.7 ms, mean binary payload ~5.4 MB. Skipped checks are explicit rather than silent: Qwen skips `Wq/Wk/Wv` Freivalds and `lm_head` token identity; all supported checks pass.

- **Bounded `lm_head` token acceptance was tested and rejected on Qwen W8A8**: honest-gap measurement on the quantized replay path showed a heavy tail from `0` to `6556` i32 units on a ~7000 range, making any threshold meaningless. Result: Qwen uses `DecodeAcceptanceMode::Unsupported`; Llama keeps exact token identity. The `lm_head` Freivalds matmul check remains exact for all profiles.

- **The decode boundary is `LogitsProcessor` input, not `model.norm` output**: diagnostic runs on Qwen W8A8 showed that `model.norm` does not feed the sampler boundary directly. `model.norm output[1]` differs from `LogitsProcessor args[1]` by 38–318 and gives only 1/192 exact token matches. In contrast, the pruned hidden at `LogitsProcessor args[1]` is the real decode boundary: `bf16 lm_head` replay from that captured hidden reproduced token identity **192/192** on greedy decode, while `fp32` replay gave 189/192 and the `i8→i32` path stayed dead. This makes captured LP-input hidden the leading decode-boundary candidate.

### Added

- **Profile-gated decode acceptance in the canonical verifier**: `VerificationProfile.decode_acceptance` / `DecodeAcceptanceMode` now control whether final-token identity is checked exactly or reported as unsupported. Qwen is explicitly unsupported for token identity on the stock quantized replay path; Llama remains exact. The verifier still runs exact `lm_head` Freivalds regardless of profile.

## 2026-04-04

### Measured

- **Sidecar-to-verifier gap — the real protocol metric (roadmap #2b)**: measured the actual L∞ between GPU-committed attention outputs (vLLM sidecar, SDPA, A100-80GB) and Rust verifier CPU f64 replay. **Qwen 7B W8A8**: L∞=8, 92.8% exact. **Llama 8B W8A8**: L∞=9, 95.6% exact. **Llama 70B W8A8**: L∞=9, 96.8% exact. >99.9% of elements within ±1 across all models. Gap grows weakly with sequence length. Worst-case layers: Qwen 1/27, Llama-8B 25, Llama-70B 7/13/70. **The honest corridor is NOT zero — τ≥9 required for honest providers.** Gap is from GPU fp16 vs CPU f64 arithmetic, not backend non-determinism (which is zero). Score witnessing remains load-bearing until τ < adversarial sensitivity threshold (~4–5). See [`research/attention-gap.md`](./research/attention-gap.md) §1i. Script: `scripts/modal/measure_corridor.py`. Results: Modal volume `corridor-results/`.

### Added

- **Attention backend binding in protocol (roadmap #2a)**: `attn_backend` and `attn_dtype` fields added to `DeploymentManifest`, `ModelSpec`, and `VerifierKey`. Canonical verifier cross-checks these between manifest and key. W8A8 mandates SDPA — fail closed on `eager`, unknown, and missing `attn_backend`. Keygen auto-populates `attn_backend=sdpa` for W8A8 models and `attn_dtype` from `config.json:torch_dtype`. Tests: `canonical_w8a8_eager_rejected`, `canonical_w8a8_sdpa_accepted`, `canonical_w8a8_missing_backend_rejected`, `canonical_w8a8_unknown_backend_rejected`, `canonical_attn_backend_mismatch_rejected`, `canonical_attn_backend_match_accepted`.

## 2026-04-03

### Measured

- **RMSNorm contraction ratio on real checkpoints (roadmap #1, partial)**: ρ_j = γ·‖W_o‖₂/RMS(x^(j)) measured via Modal A100-40GB on two families. **Qwen-7B W8A8**: 24/28 layers contract (86%), 4 early layers expand (ρ up to 63.8), accumulated drift/residual=0.63. **Llama-3.1-8B fp16**: 0/32 layers contract, every layer expands (ρ 1.7–691), accumulated drift bound diverges (10^13). RMSNorm contraction alone does NOT close the corridor for Llama — the small residual norms (~1.5–2.0) and large γ weights (~1.0) prevent dampening. Score witnessing (#7) and W_o conditioning (#8) confirmed critical, not optional. Measurement script: `redteam/modal/measure_contraction.py`.

- **Advanced corridor sensitivity (roadmap #1, #2, #8)**: local operator norm, finite-difference sensitivity, and bad-layer localization measured via Modal A100-40GB. See [`research/attention-gap.md`](./research/attention-gap.md) for full analysis. Key results:
  - **Local operator norm** ‖J_RMSNorm(x)·W_o‖ is 2–4× tighter than crude ρ. Qwen: 21/28 layers < 1.0. Llama: 0/32 < 1.0 (still all expanding).
  - **Finite-difference token flips**: injecting worst-case ±τ=8 perturbations through W_o at a single layer flips the output token on **3/28 layers** (Qwen) and **6/32 layers** (Llama). Layers 0, 1, 27 (Qwen) and 3, 7, 8, 16, 17, 30, 31 (Llama) are the most dangerous.
  - **Logit margin**: Qwen margin=1.69, Llama margin=0.11. Even middle layers consume 70–115% of Qwen's margin and 500–1700% of Llama's.
  - **Per-head W_o norms**: max per-head spectral norm ~2–4.5 (both models), confirming head substructure matters.
  - **L∞→L∞ induced norm**: 87–331 (Qwen), 98–190 (Llama) — much larger than spectral norms, confirming L∞ corridor metric needs L∞-compatible analysis.
  - **Conclusion**: corridor tolerance is not safe without score witnessing on either model. Perturbations are not merely theoretical — they flip real tokens on real weights.
  - Measurement script: `redteam/modal/measure_sensitivity.py`.

- **Multi-prompt sensitivity + MLP propagation (roadmap #1, #2)**: extended sensitivity measurement across 6 diverse prompts (factual, code, math, uncertain, long-context, true/false) with MLP propagation and all-layers-simultaneously experiments. See [`research/attention-gap.md`](./research/attention-gap.md) §1e–1g. Key results:
  - **Qwen**: 28/28 layers flip the token on every prompt tested. The corridor is structurally exploitable, not edge-case. Layers 24–25 consistently most dangerous.
  - **Llama**: 0–6 single-layer flips depending on prompt. High-margin prompts (fibonacci, margin=9.81) survive all single-layer attacks. Layer 31 is top-3 dangerous on all 6 prompts.
  - **MLP dampens perturbations on Llama** (ratio 0.67), neutral on Qwen. MLP is not a threat channel.
  - **All-layers-simultaneously flips every prompt on both models.** Compounding drift is real — the verifier follows committed state forward (`canonical.rs:1223, 1904`), so ±τ errors accumulate across layers.
  - Measurement script: `redteam/modal/measure_sensitivity_v2.py`.

- **Backend determinism comparison (roadmap #2)**: measured attention output mismatch and run-to-run determinism across eager, SDPA, and eager+deterministic backends on both Qwen W8A8 and Llama fp16. **All backends are bit-exact run-to-run on the prover side** (L∞ = 0.0). Backend mismatch appears to dominate the previously measured honest corridor — not stochastic non-determinism. This makes deterministic, backend-aligned replay a much more promising path. **Caveat**: prover-side determinism does not yet prove protocol-level τ=0 — the verifier does CPU FP64 replay, not the same GPU backend; vLLM sidecar path and cross-hardware (A100 vs H100) are untested. Eager attention is broken with W8A8 compressed_tensors (wrong tokens); SDPA is the only correct backend for quantized models. For fp16 (Llama), eager vs SDPA differs by only L∞=0.004 with identical tokens. See [`research/attention-gap.md`](./research/attention-gap.md) §1h. Measurement script: `redteam/modal/measure_corridor_backends.py`.

### Added

- **Attention gap research document** (`research/attention-gap.md`): comprehensive analysis of corridor sensitivity with all measurement results, interpretation, protocol implications, and remaining open experiments.

## 2026-03-31

### Added

- **Score-witness deep-audit scaffolding (roadmap #9, partial)**: added `ScoreWitness` to the audit wire/types surface and `V4AuditResponse.score_witnesses` as an optional per-layer field for deep-audit openings. Binary fixtures / cross-version fixtures were regenerated so the wire format remains pinned while the score-witness path is prototyped. This is only the first step of roadmap `#9`: the measurement path, canonical verifier phase, and prover deep-audit opening logic are still pending.

### Fixed

- **RoPE convention**: `apply_rope_head` used interleaved pairing `[2i, 2i+1]` but vLLM/Qwen/LLaMA use half-rotary `[i, i+half]`. K corridor dropped from L-inf=255 to <0.5 after fix.
- **Q/KV input boundary asymmetry in corridor measurement**: shell opening Q was computed from bridge-derived x_attn while committed K/V used GPU-captured x_attn. Added `use_captured_x_attn` flag to `open_v4` so corridor measurement uses the same authoritative boundary for Q and K/V. Corridor dropped from L-inf=67–117 to L-inf=8.

### Measured

- **Attention corridor on Qwen2.5-7B-W8A8 (A100-80GB)**: 672 measurements across 6 workloads (short through 1164-token long_context), all 28 layers, all decode positions. Global max L-inf = 8. First generated token max L-inf = 5. >92% of elements are exact matches; >99.8% within ±1. No growth with sequence length. Worst layers spread across the stack (not concentrated).

- **Attention corridor on Llama-3.1-8B-W8A8 (A100-80GB) — cross-family control (corrected)**: Rerun after Llama3 `rope_scaling` (factor=8, type="llama3") landed in the verifier. K/Q diagnostic confirms fix: K L-inf dropped from 49 to 0.049 (1000× improvement), Q L-inf = 0.033, max score diff = 0.020 — all in Qwen's range. Corridor: global max L-inf = 9, first-generated-token max = 5, decode max = 9, `frac_eq ≈ 94–96%`, `frac≤1 > 99.9%`. Mild growth with context (3 at short, 9 at 1165 tokens). Layer 25 consistently worst. The prior L-inf=49 result was caused by missing `rope_scaling` in the verifier, not a family limitation — it is invalid historical data. Both families now support a shared `attention_tolerance = 10`. Profiles remain for architecture metadata (RoPE semantics, support scope), not different tolerance values.

### Changed

- **Attention-tightening priority**: with corrected Llama now in the same single-digit corridor regime as Qwen, the main next protocol-tightening step is score witnessing / score anchoring rather than further family-specific tolerance splits. Third-family and larger-model controls remain required, but they should preferentially target the tightened post-score-witness path once the measurement and verifier phases land.

## 2026-03-30

### Added

- **Roadmap #3 closed: committed KV transcript with kv_root covering full causal prefix.** `KvEntry { k_roped, v_deq }` committed per (layer, position) under per-layer Merkle `kv_roots` with domain separator `vi-kv-v1`. Prover derives post-RoPE K/V at commit time via deterministic INT8 matmul on captured `x_attn_i8` + weights (no GPU-side K/V hooks needed). Canonical verifier `phase_kv_transcript` verifies Merkle proofs against `kv_roots`. `replay_deep_prefix_toy/roped` consumes committed KV when available, falling back to shell accumulator reconstruction for legacy. GPU evidence on `neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8` (A100-80GB): kv_roots present on default path (28/28 layers); binary deep-prefix KV Merkle proofs pass (0 failures); binary canonical path has 0 KV failures (only attention replay mismatch remains — tracked in #4). Three non-attention confounders fixed during this milestone: per-channel dequant in prover bridge (Freivalds 0/168 failures, was 168/168), shell QKV using bridge-derived input instead of GPU-captured x_attn, and quant_family/scale_derivation vocabulary alignment between weight provider and manifest. JSON debug path has a known f64 round-trip precision issue on KV entries (not a protocol blocker — binary is the canonical production format).

- x_attn capture enabled by default when `VERILM_CAPTURE=1`. The non-packed commit path (which computes KV transcripts) is now the default for verified inference. The packed path remains available via `VERILM_CAPTURE_X_ATTN=0` but does not produce kv_roots. This ensures a supported verified run always includes committed KV.

### Fixed

- Per-channel dequantization in prover `compute_shell_opening` — previously only the verifier dispatched per-channel, causing W8A8 Freivalds failures on all layers/matrices (per-tensor scale was 0.0 for W8A8).
- Shell QKV now uses bridge-derived x_attn (not GPU-captured) so that QKV accumulators match the verifier's own derivation for Freivalds checks. Captured x_attn is only for KV transcript computation at commit time.
- quant_family/scale_derivation metadata now derived from `WeightProvider` using the same logic as keygen, instead of extracting HuggingFace `quant_method` strings that didn't match key vocabulary.
- `_weight_provider` initialization order in `server.py` — quant fields deferred until after weight provider is created.

### Previously added

- **Roadmap #1 closed: quantization semantics are now first-class protocol data.** Per-channel weight scales flow end-to-end from keygen through verification. Keygen discovers `weight_scale` tensors in W8A8 safetensors (handling both `[N]` and `[N,1]` shapes, BF16/FP16/F32 dtypes), populates `VerifierKey.per_channel_weight_scales`, and sets `quant_family` / `scale_derivation` metadata. The canonical verifier dispatches to per-channel dequant for QKV attention, bridge residual, and SiLU gate paths. Corridor tooling uses key-embedded scales instead of external overrides. Confirmed on real `neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8` via Modal smoke test: 28 layers × 7 matrices of per-channel scales loaded with correct shapes and realistic nonzero ranges. Formal worst-case attention corridor bounds derived in `bounds.rs`. Per-token prefill `scale_a` and grouped-quant `quant_block_size` are real issues tracked separately, not blocking #2.

- **Roadmap #2 closed: bridge trust boundary is now committed and checked with check-and-gate semantics.** `RetainedLayerState` now commits the actual attention-input bridge object (`x_attn_i8`, `scale_x_attn`) alongside `a`, and the retained Merkle hash domain is bumped from `vi-retained-v2` to `vi-retained-v3` with presence markers so missing-versus-present bridge data is hash-distinct. The canonical verifier now gates on the opened bridge boundary before downstream replay: it canonically recomputes `x_attn_i8`, compares against the committed bridge value, and if the check passes continues from the committed value rather than the verifier replay. This makes the bridge the authoritative QKV input boundary and prevents approximation error from propagating across layers. Binary fixtures and golden pins were regenerated; dedicated tests cover honest pass, tampered `x_attn_i8`, tampered `scale_x_attn`, presence-marker hashing, and fail-closed handling when bridge data is missing.

- Production RoPE-aware deep-prefix attention replay landed in the canonical verifier and corridor tooling. `VerifierKey.rope_aware_replay` now dispatches between toy/reference replay and a production path that dequantizes Q/K/V accumulators with activation and weight scales, applies RoPE at each position, and replays attention in f64 post-RoPE space. The deep-prefix path covers both prefix tokens and the opened token, and tests cover honest pass plus fake-`a` rejection on the roped path.

### Changed

- The roadmap was renumbered into a single linear sequence of remaining tasks, and completed items were removed from `roadmap.md` so the roadmap is now strictly an open-items tracker while this changelog carries completed milestones.
- With roadmap `#1` and `#2` closed, the active attention-critical path now starts at roadmap `#3`: full-causal `kv_root`, cleanup of the old confounded gap path, and only then analytical/empirical attention-gap work.

## 2026-03-29

### Added

- Client-side verification wrapper (`verilm_verify::client`) for protocol-level checks that sit outside the canonical verifier's trust boundary. `verify_challenged_binary()` and `verify_challenged_response()` now bind an audit to the verifier-issued `AuditChallenge` by checking the challenged `token_index` and exact opened `layer_indices`, using stable `FailureCode` values (`ChallengeTokenMismatch`, `ChallengeLayerMismatch`) rather than message parsing. Tests cover pass, wrong token, wrong layers, canonical-failure propagation, combined failures, and the baseline no-wrapper path.

### Changed

- The canonical verifier is now the trusted public verification path. Public `verify_v4()` / `verify_v4_full()` entrypoints delegate to the canonical verifier; the old monolithic verifier remains only for differential testing and rollback during the freeze-out period.
- The canonical verifier was cleaned up into a small explicit pipeline: `Ctx::new()` owns precomputed facts, `run()` is the orchestrator, typed phase state (`StructuralState`, `SpecState`, `BridgeState`) wires the phases together, and the bridge path is split into named subchecks instead of transitional branch soup. The trust boundary is now explicit: `canonical` verifies proof correctness, while `client` handles external challenge/receipt protocol checks.
- Retained-state schema trimmed to irreducible fields only. `RetainedLayerState` now commits only `a` and `scale_a`; derivable replay scales (`scale_x_attn`, `scale_x_ffn`, `scale_h`) moved to `ShellLayerOpening`, the retained hash domain bumped to `vi-retained-v2`, prover-side `CapturedLayerScales` carries those values between commit and open, and frozen fixtures / SHA-256 pins were regenerated. Tampering any revealed replay scale is caught by the dependent bridge/Freivalds equations. This closes roadmap `#2`.

## 2026-03-28

### Fixed

- Keygen `rmsnorm_eps` split-brain: key generation and `SafetensorsWeightProvider` both now read `rms_norm_eps` from the model's `config.json` instead of hardcoding `1e-5`. Fixes bridge replay divergence on models like Qwen that use `1e-6`.
- Keygen `rope_theta` now sourced from `config.json` instead of a heuristic based on `hidden_dim`. Fixes mismatch on models with non-standard `rope_theta` values.
- Keygen `vocab_size` early detection from `lm_head.weight` tensor shape in `detect_config()`, before `r` vector generation. Fixes panic when `vocab_size` was 0 at LmHead Freivalds setup.
- Verifier `n_tokens` generation-length semantics: `n_generated = n_tokens - (n_prompt - 1)`, accounting for the committed token_ids array omitting the first embedding token. Fixes off-by-one that allowed `n_tokens_inflate` to pass.
- Verifier LM-head token replay now skipped for prompt-side tokens (where `token_index < gen_start`). Prompt tokens are chosen by the tokenizer, not by argmax/sampling over logits. Fixes false rejection on multi-token audits where the challenged position is inside the prompt.

### Added

- Verification failure taxonomy: every verifier failure is classified into one of six categories (`structural`, `cryptographic_binding`, `spec_mismatch`, `unsupported`, `semantic_violation`, `operational`) via a stable `FailureCode` enum (~48 variants). Each `VerificationFailure` carries `code`, `category`, `message`, and optional `FailureContext` (token index, layer, matrix, field, spec, expected/actual values). Consumers match on codes — no message-text parsing needed. The substring-based `classify_failure()` classifier is removed.
- Structured audit-failure reporting: `V4VerifyReport.failures` is now `Vec<VerificationFailure>` with stable codes and context. `failure_messages()` accessor provides backward-compatible `Vec<&str>`. Python bridge exposes `classified_failures` as `list[dict]` with `code`, `category`, `message`, and optional `context` fields.
- Partial-audit semantics: `V4VerifyReport` now carries `coverage: AuditCoverage` distinguishing `Full` (all layers checked), `Routine` (contiguous prefix), and `Unknown` (no shell opening). Display and JSON serialization include coverage level. Python bridge exposes `coverage` as a dict with `level`, `layers_checked`, and optionally `layers_total`. Consumers can programmatically distinguish routine-audit passes from full-audit passes — partial statistical coverage cannot be mistaken for full exact success.
- Boundary-condition fuzzing suite (`boundary_fuzz.rs`, 55 tests): malformed/truncated binary payloads (empty, magic-only, garbage zstd, wrong bincode, single-bit flips, every-offset truncation), structural field absence/mismatch (seed, prompt hash, n_prompt_tokens bounds), prefix count boundaries, non-contiguous/reversed/empty layer_indices, EOS policy edge cases (stop-not-at-end, unknown policy, ignore_eos, min_tokens without eos_token_id), decode-mode/temperature inconsistency, unsupported decode features, manifest hash binding, unknown version/magic rejection, long/short prompt-output combinations, sampler version rejection, coverage semantics, and failure context metadata. All assert specific `FailureCode` and `FailureCategory` values. This closes roadmap `#8`.
- Cross-version binary format tests (`cross_version.rs`, 19 tests) with frozen on-disk fixtures (`tests/fixtures/`). Compatibility matrix: canonical V4 audit and key fixtures deserialize and verify, byte-stability checks detect format drift, SHA-256 checksums pinned, rejection fixtures for unknown magic, truncated payloads, cross-format misuse, and corrupted bincode. Forward-compatibility: VV5A/VV9Z/VKE2 future magics fail-closed. Cross-format: key-as-audit and audit-as-key rejected. This closes roadmap `#13`.
- Adversarial hardening gate (`hardening_gate.rs` + `make hardening-gate`): explicit named gate that must pass before strong claims or final benchmarks land. Aggregates 8 suites (265+ tests): boundary_fuzz, cross_version, v4_e2e, golden_conformance, weight_chain_adversarial, fiat_shamir_soundness, quantization_parity, plus GPU adversarial existence check. Gate meta-tests verify suite existence, minimum test counts, fixture presence, taxonomy completeness, and coverage semantics. `make gpu-test-adversarial` runs the remote GPU adversarial suite. This closes roadmap `#4`.

### Changed

- Adversarial tamper test (`test_adversarial.py`) hardened:
  - Baselines restored to strict assertions (224/224 checks pass on real W8A8 GPU).
  - Splice tests now use different prompts and audit at generated-token indices past the prompt boundary. Token splice requires verified token divergence before asserting rejection.
  - `final_residual_shift` accepts `merkle` as a valid rejection reason (stronger than a custom "final" reason since it proves retained-leaf hash binding).
  - Multi-token prefix baseline audits at a generated token index, avoiding prompt-side token replay.
  - Freivalds diagnostic script (`diag_freivalds.py`) added for targeted GPU-side failure classification.
- Adversarial suite now achieves 36/36 on the real W8A8 Modal run. Token-splice construction fixed by reading `n_prompt_tokens` from the audit response and auditing past the prompt/template boundary where content tokens actually diverge.

## 2026-03-27

### Added

- Four-spec commitment wiring for `input_spec_hash`, `model_spec_hash`, `decode_spec_hash`, and `output_spec_hash`, with verifier-side recomputation and composed manifest checking.
- Canonical commitment of the full four-spec surface, including chat template, BOS/EOS preprocessing, special-token handling, system-prompt semantics, public model identity `R_W`, adapter identity, RoPE configuration, RMSNorm epsilon, sampler identity, decode knobs, and output stopping rules.
- Explicit model-surface commitments for `n_layers`, `hidden_dim`, `vocab_size`, and `embedding_merkle_root`, with verifier-side cross-checks against the verifier key/config.
- `padding_policy` field in InputSpec and DeploymentManifest, hashed in `hash_input_spec()`, populated by the live server manifest path, and enforced fail-closed in the Python verifier path. Completes truncation/padding binding (roadmap #4).
- `decode_mode` field in DecodeSpec and DeploymentManifest, hashed in `hash_decode_spec()`, with verifier cross-check against temperature: greedy requires temp=0, sampled requires temp>0, unknown modes fail closed. Tests cover all mismatch and pass cases (roadmap #9).
- Quantization identity fields `quant_family`, `scale_derivation`, `quant_block_size` in ModelSpec, DeploymentManifest, and VerifierKey, hashed in `hash_model_spec()`, cross-checked against verifier key, extracted from model quantization config in the live server, and parsed in the Python bridge. Tests cover mismatch rejection and consistent pass (roadmap #5-7).
- Remaining architecture fields in ModelSpec: `kv_dim`, `ffn_dim`, `d_head`, `n_q_heads`, `n_kv_heads`, `rope_theta`, hashed in `hash_model_spec()`, cross-checked against verifier key config, extracted from model config in the live server, parsed in the Python bridge. Tests cover all 6 mismatch rejections plus full-architecture pass (roadmap #8).
- LM-head Freivalds as an explicit eighth matrix family, including verifier-secret key material and canonical verifier checks.
- Exact logits replay after LM-head binding so verified token selection still depends on the committed logits path.
- Exact final-token verification boundary from the captured pre-final-norm residual, with fail-closed behavior when the boundary state is missing.
- Canonical sampled-path replay with fresh per-request `batch_seed`, `seed_commitment`, audit-time seed reveal, and deterministic per-token randomness derivation.
- Request-bound transcript genesis `H("vi-io-genesis-v4" || prompt_hash)` and canonical V4 IO chaining over `leaf_hash`, `token_id`, and prior transcript state.
- Fail-closed prompt binding and prompt/generation-boundary count binding, including `n_prompt_tokens`.
- End-to-end canonical input reconstruction via `PromptTokenizer`, exposed through the Python verifier bridge.
- Full `InputSpec` replay in the Python verifier path, covering tokenizer identity, system prompt, chat template, BOS/EOS preprocessing, special-token policy, and truncation policy.
- Prefix embedding binding for prefix tokens, including per-prefix embedding rows and Merkle proofs in rich-prefix mode.
- Exact deep-prefix audit support, including `prefix_retained`, `prefix_shell_openings`, retained-hash consistency checks, and Freivalds/bridge checks on prefix tokens.
- End-to-end detokenization verification via `Detokenizer`, `output_text` in audit responses, and Python verifier plumbing for detokenizer callbacks.
- End-to-end HTTP coverage for `/chat` and `/audit`, including greedy and sampled JSON audit verification, binary audit verification, routine-tier coverage, and HTTP error-path coverage.
- Tokenizer identity commitment based on canonicalized `tokenizer.json` content via `backend_tokenizer.to_str()`, with deterministic JSON normalization and legacy vocab-only fallback.
- Explicit `weight_hash` / `R_W` cross-checking in the canonical verifier path, in addition to model-spec hash binding.
- Version-locked sampler conformance vectors: golden `derive_token_seed` SHA-256 vectors and golden `sample()` token outputs for temperature, top-k, top-p, combined filtering, and greedy modes. Any silent drift in the `chacha20-vi-sample-v1` pipeline breaks these tests.
- Greedy tie-breaking is fixed to lowest-index selection and pinned as part of the version-locked `chacha20-vi-sample-v1` sampler semantics.
- Golden conformance vectors embedded in the test suite for challenge derivation (`build_audit_challenge` token index and layer indices for 4 seed/token/layer configs), manifest hashing (pinned `hash_manifest` and `hash_model_spec` digests), and end-to-end verification (pinned merkle root, IO root, verdict, and checks_run for deterministic commit→open→verify).
- Challenge protocol specification (now folded into the maintained paper/README/article surface): exact derivation of token index, layer depth (routine/full), sampling seed, Freivalds block coefficients, all domain separators, and binary wire format.
- Binary format robustness tests: V4 audit response roundtrip through serialize/deserialize, unknown magic rejection, truncated payload rejection, cross-format rejection (key bytes to audit deserializer), version field preservation, and verifier key roundtrip. (Not yet a full cross-version compatibility matrix with frozen binary fixtures.)

### Changed

- Verified serving mode now defaults to sampled decoding (temperature=1.0). Greedy decoding remains available as the explicit `temperature=0` special case. Server API, HTTP endpoint, sampler hook, and Modal endpoint all updated.
- Terminology and documentation framing now consistently describe Freivalds checks as information-theoretically sound / statistically sound rather than cryptographic.
- Runtime-populated input, decode, and output policy fields now flow into the committed four-spec surface instead of relying on static defaults.
- The live canonical sampled path is replayable end to end under the committed decode spec instead of depending on incidental stock serving behavior.
- The key-only retained-state path is now the canonical verifier path for the kept protocol.
- The canonical verifier path now checks reconstructed prompt tokens against the committed input path and the embedding path.
- Output-policy enforcement now covers `eos_policy`, `min_tokens`, `ignore_eos`, stop conditions, special-token stripping, detokenization policy, cleanup behavior, and fail-closed handling for unsupported or unknown policy values.
- Input-policy replay is fail-closed for unknown `bos_eos_policy`, `special_token_policy`, `truncation_policy`, and `detokenization_policy`.
- The verifier now cross-checks committed model-spec values against verifier-key values for `rmsnorm_eps`, `rope_config_hash`, and `weight_hash`.
- The verifier now also cross-checks committed model geometry and embedding commitment values against the verifier key/config: `n_layers`, `hidden_dim`, `vocab_size`, and `embedding_merkle_root`.
- Decode-spec handling is fail-closed for unsupported sampler versions and unsupported non-default decode features; only the canonical sampler version is accepted and supported decode features are replayed.
- Audit retrieval now preserves claimed output text end to end so detokenization checks can run in the canonical verifier flow.
- Bridge documentation and call sites now explicitly treat `bridge_residual_rmsnorm()` as the canonical production W8A8 bridge, while `bridge_requantize()` is demoted to toy-model and last-layer fallback use only.
- The roadmap and protocol framing now treat the kept canonical protocol as structurally complete, with remaining work focused on hardening, conformance, benchmarks, and documentation.
- Removed transitional V4 framing: "dropped in V5" comments and "transitional replay scales" qualifiers replaced with canonical bridge replay scale framing. V4 is the kept protocol, not a stepping stone.
- `RetainedLayerState` documentation now explicitly classifies fields as irreducible (`a`, `scale_a`) vs derivable-with-weight-access (`scale_x_attn`, `scale_x_ffn`, `scale_h`), with derivation paths documented.
- Binary (bincode+zstd with VV4A magic) is now explicitly framed as the sole normative wire format for receipts and audits. JSON verification path demoted to debug/development convenience.
- Audit-time weight loading confirmed bound to `R_W`: server validates WeightProvider hash against manifest at startup, verifier cross-checks manifest weight_hash against key weight_hash.
- `bad_word_ids` field added to DecodeSpec and DeploymentManifest, hashed in `hash_decode_spec()`, fail-closed in verifier (non-empty rejected), parsed in Python bridge, set to empty in live server. Completes decode-surface logit modification binding (roadmap #10).

### Removed

- Legacy V1/V2/V3 protocol framing and code paths from the kept protocol story.
- Dead full-trace artifacts and stale legacy comments that only served the removed pre-V4 protocol paths.
- Dead `requantize_bridge()` and `verify_bridge()` bridge helpers superseded by `bridge_residual_rmsnorm()`.
- Unused `kv_chain_root` commitment field and related dead code.
- Transitional reliance on weight-backed replay as part of the canonical verifier path; weight-backed replay remains debug/oracle-only.
