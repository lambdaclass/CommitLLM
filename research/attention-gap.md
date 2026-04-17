# Attention Gap: Corridor Sensitivity Analysis

This document tracks measurements, open questions, and the evolving understanding of the attention corridor's security properties. The corridor allows ±τ tolerance on INT8 attention outputs per layer. The central question: **can an adversary exploit this tolerance to flip output tokens?**

Current protocol interpretation as of 2026-04-17:

- sampled decode exactness is now closed separately via `CapturedLogits`
- greedy shell/decode is in good shape on the supported paths
- arbitrary-position attention is the main remaining dense-model correctness blocker
- the next accepted stock-kernel attempt is a FlashAttention/kernel-aligned witness
- if that fails broad randomized tests cleanly, the fallback is deterministic attention kernels

---

## 1. Measurement results

### 1a. RMSNorm contraction ratio (2026-04-03)

The crude scalar bound ρ_j = max(γ_j) · ‖W_o^(j)‖₂ / RMS(x^(j)) measures whether the RMSNorm + W_o path dampens or amplifies perturbations.

**Qwen-7B W8A8** (`neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8`):

| Metric | Value |
|--------|-------|
| Contracting layers (ρ < 1) | 24/28 (86%) |
| Expanding layers | 4 (layers 0–3, ρ up to 63.8) |
| ρ range | [0.020, 63.8] |
| Accumulated drift bound (τ=10) | 11.6 |
| Drift / final RMS(residual) | 0.63 |

**Llama-3.1-8B** (`NousResearch/Meta-Llama-3.1-8B-Instruct`):

| Metric | Value |
|--------|-------|
| Contracting layers (ρ < 1) | 0/32 (0%) |
| ρ range | [1.68, 691.4] |
| Accumulated drift bound (τ=10) | 2.37 × 10¹³ (divergent) |
| Drift / final RMS(residual) | 1.19 × 10¹³ |

**Why Llama is worse**: small residual norms (~1.5–2.0 vs Qwen's ~45–50) and large RMSNorm γ weights (~1.0 vs Qwen's ~0.25–0.35). The normalization doesn't dampen perturbations.

Script: `redteam/modal/measure_contraction.py`

### 1b. Local operator norm (2026-04-03)

The tighter bound ‖J_RMSNorm(x_j) · W_o^(j)‖ accounts for the actual Jacobian of RMSNorm at the observed residual, rather than taking worst-case products of individual norms.

**Qwen-7B W8A8**:

| Metric | Value |
|--------|-------|
| Layers with local norm < 1 | 21/28 |
| Local norm range | [0.10, 44.0] |
| Average tightening vs crude ρ | 2.3× |

**Llama-3.1-8B**:

| Metric | Value |
|--------|-------|
| Layers with local norm < 1 | 0/32 |
| Local norm range | [2.16, 352.8] |
| Average tightening vs crude ρ | 2.8× |

The local operator norm is meaningfully tighter (2–4×) but does not change the qualitative picture: Llama has no contracting layers even under the tighter bound.

### 1c. Finite-difference token flips (2026-04-03)

Injected worst-case ±τ=8 perturbations (SVD-aligned, L∞-bounded, scaled by measured per-layer scale_a) through W_o at each layer individually. Measured actual logit change and whether the output token flips.

**Qwen-7B W8A8** (honest token: "Paris", margin: 1.69):

| Layer | Δlogit_max | Margin consumed | Flips? |
|-------|-----------|----------------|--------|
| 0 | 15.0 | 890% | YES → "," |
| 1 | 14.4 | 855% | YES → "1" |
| 27 | 2.8 | 167% | YES → "__" |
| 6 | 1.9 | 116% | no |
| 17 | 1.9 | 112% | no |

3/28 layers flip the token. Most middle layers consume 70–115% of the margin — borderline.

**Llama-3.1-8B** (honest token: "a", margin: 0.11):

| Layer | Δlogit_max | Margin consumed | Flips? |
|-------|-----------|----------------|--------|
| 31 | 6.9 | 6266% | YES → "Paris" |
| 3 | 3.5 | 3204% | no |
| 29 | 2.1 | 1896% | no |
| 7 | 0.9 | 850% | YES → "Paris" |
| 8 | 0.7 | 639% | YES → "Paris" |

6/32 layers flip the token. The model was already uncertain (margin 0.11), so every layer's perturbation vastly exceeds the margin.

### 1d. Additional measurements (2026-04-03)

**Per-head W_o spectral norms**: max per-head norm ~2–4.5 for both models. The head substructure means not all directions in the attention output space are equally dangerous.

**L∞→L∞ induced norms** (max absolute row sum of W_o): 87–331 (Qwen), 98–190 (Llama). These are much larger than spectral (2→2) norms and are the correct metric when the corridor is stated in L∞. This confirms that spectral-norm analysis alone underestimates the worst case for L∞-bounded adversaries.

Script: `redteam/modal/measure_sensitivity.py`

### 1e. Multi-prompt sensitivity (2026-04-03)

Ran finite-difference token flips on 6 diverse prompts per model: factual, code, math, uncertain, long-context, true/false.

**Qwen-7B W8A8**:

| Prompt | Margin | Attn-only flips | All-layers Δlogit |
|--------|--------|-----------------|-------------------|
| Capital of France | 1.69 | 28/28 | 23.1 |
| Best language | 0.53 | 28/28 | 24.6 |
| Fibonacci code | 6.61 | 28/28 | 26.1 |
| Math (7*8) | 7.53 | 28/28 | 27.1 |
| Long context | 0.55 | 28/28 | 28.6 |
| True/false | 0.00 | 27/28 | 30.6 |

**Every layer flips the token on every prompt.** This is worse than v1 because v2 correctly scales perturbations by measured scale_a and uses SVD-aligned worst-case direction. Consistently dangerous layers: **25, 24** (top-3 on all 6 prompts).

**Llama-3.1-8B**:

| Prompt | Margin | Attn-only flips | Post-layer flips | All-layers Δlogit |
|--------|--------|-----------------|------------------|-------------------|
| Capital of France | 0.11 | 6/32 | 11/32 | 14.7 |
| Best language | 0.56 | 0/32 | 1/32 | 16.9 |
| Fibonacci code | 9.81 | 0/32 | 0/32 | 26.1 |
| Math (7*8) | 2.66 | 0/32 | 1/32 | 16.5 |
| Long context | 0.41 | 2/32 | 4/32 | 16.1 |
| True/false | 1.64 | 2/32 | 2/32 | 14.0 |

Llama is more resilient per-layer than Qwen (fewer single-layer flips), but high-margin prompts (fibonacci, margin=9.81) survive all single-layer attacks. Low-margin prompts are vulnerable. Consistently dangerous layer: **31** (top-3 on all 6 prompts), followed by **0** and **28**.

**All-layers-simultaneously flips every prompt on both models.**

### 1f. MLP propagation (2026-04-03)

Compared attn-only perturbation (injected at self_attn output, flows through same-layer MLP) vs post-layer perturbation (injected after full layer, bypasses same-layer MLP).

**Qwen**: MLP is roughly neutral. Attn-path / post-layer ratio: mean=1.00, range [0.91, 1.22].

**Llama**: **MLP dampens perturbations**. Ratio: mean=0.67, range [0.13, 1.28]. This is the opposite of the feared MLP amplification — the MLP actually reduces the perturbation impact. However, in some cases post-layer injection causes MORE flips than attn-only (11 vs 6 on prompt 0), suggesting the MLP reshapes the perturbation direction rather than simply scaling it.

### 1g. Compounding drift (architectural note)

The current verifier is **non-state-replacing**: it follows the provider's committed state forward. At each layer, the verifier checks that the committed attention output is within ±τ of replay, but then continues with the committed value, not the replayed value (`canonical.rs:1223`, `canonical.rs:1904`). This means:

- Layer j accepts error e_j with ‖e_j‖∞ ≤ τ
- Layer j+1 operates on the already-drifted committed state
- Layer j+1 introduces another accepted error e_{j+1}
- Errors compound across L layers

The all-layers-simultaneously experiment directly measures this compounding: Δlogit from all layers is 14–31 (much larger than any single layer's contribution), confirming accumulation.

A hypothetical **state-replacing** verifier (replace committed state with canonical replay at each check) would not have this issue, but would require the verifier to maintain full inference state — incompatible with the current commitment-following design.

Script: `redteam/modal/measure_sensitivity_v2.py`

### 1h. Backend determinism comparison (2026-04-03)

Measured attention output mismatch between backends (eager, SDPA, eager+deterministic) and run-to-run determinism for each backend on both models.

**Llama-3.1-8B** (fp16):

| Comparison | Max L∞ | Mean L∞ | Tokens |
|------------|--------|---------|--------|
| eager vs eager_det | **0.000000** | 0.000000 | SAME |
| eager vs sdpa | **0.003906** | 0.001011 | SAME |
| eager_det vs sdpa | **0.003906** | 0.001011 | SAME |

Run-to-run determinism:
- eager: **BIT-EXACT** (L∞ = 0.0, 32/32 layers)
- eager_det: **BIT-EXACT** (L∞ = 0.0, 32/32 layers)
- sdpa: **BIT-EXACT** (L∞ = 0.0, 32/32 layers)

All backends produce correct, identical tokens (`a`, `fibonacci`, `the`).

**Qwen-7B W8A8**:

| Comparison | Max L∞ | Mean L∞ | Tokens |
|------------|--------|---------|--------|
| eager vs eager_det | **0.000000** | 0.000000 | SAME |
| eager vs sdpa | **1.468750** | 0.256953 | DIFFER |
| eager_det vs sdpa | **1.468750** | 0.256953 | DIFFER |

Run-to-run determinism:
- eager: **BIT-EXACT** (L∞ = 0.0, 27/28 layers)
- eager_det: **BIT-EXACT** (L∞ = 0.0, 27/28 layers)
- sdpa: **BIT-EXACT** (L∞ = 0.0, 28/28 layers)

**Critical**: Eager produces wrong tokens (`!` for all prompts) with W8A8 compressed_tensors. SDPA produces correct tokens (`Paris`, `fibonacci`, `the`). Eager attention is **incompatible** with compressed_tensors W8A8.

**Key finding**: Every backend is **bit-exact run-to-run on the prover side**. Backend mismatch (e.g., FlashAttention vs eager) appears to dominate the previously measured honest corridor, not stochastic non-determinism. This makes deterministic, backend-aligned replay a much more promising path than previously thought.

**Caveats — this does not yet prove protocol-level τ=0**:
1. The current verifier does CPU-side FP64 replay, not the same GPU backend as the provider. Prover-side determinism alone does not close the gap unless the verifier matches the same arithmetic/kernel semantics.
2. This measures HuggingFace backend behavior (`AutoModelForCausalLM`), not the full vLLM sidecar path used in production.
3. Same-backend requires same arithmetic spec, same hardware class (or a proved cross-hardware equivalence), and same implementation on both prover and verifier.
4. Cross-hardware stability (A100 vs H100) is untested.

**What must be tested before changing the protocol story**:
- Reproduce in the actual vLLM sidecar path
- Compare provider output to verifier CPU replay (not just provider vs provider)
- Check cross-hardware stability (at least A100 vs H100)

Script: `redteam/modal/measure_corridor_backends.py`

### 1i. Sidecar-to-verifier gap (GPU fp16 SDPA → Rust CPU f64 replay)

**Measured the actual protocol gap**: vLLM sidecar with SDPA attention on A100-80GB, committed INT8 attention outputs, compared against the Rust verifier's CPU f64 replay using `measure_corridor_committed_kv()`. This is the real decision metric for τ.

**Hardware**: A100-80GB (Modal), SDPA attention backend, W8A8 compressed_tensors quantization.

**Results (L∞ = max |GPU_committed_a_i8 − verifier_replayed_a_i8|)**:

| Model | Global max L∞ | Typical L∞ | frac_eq | frac≤1 | frac≤2 |
|-------|--------------|------------|---------|--------|--------|
| Qwen 7B W8A8 | **8** | 3–5 | 92.8% | 99.93% | 99.99% |
| Llama 8B W8A8 | **9** | 3–5 | 95.6% | 99.97% | 99.99% |
| Llama 70B W8A8 | **9** | 3–5 | 96.8% | 99.99% | 99.99% |

**Histogram (aggregate across all workloads/positions)**:

| diff | Qwen 7B | Llama 8B | Llama 70B |
|------|---------|----------|-----------|
| =0 | 92.79% | 95.61% | 96.76% |
| =1 | 7.13% | 4.37% | 3.23% |
| =2 | 0.063% | 0.017% | 0.010% |
| =3 | 0.009% | 0.002% | 0.001% |
| =4 | 0.002% | 0.0005% | 0.0001% |
| ≥5 | 0.0008% | 0.0004% | 0.0001% |

**Per-model worst-case layers**:
- **Qwen 7B**: layer 1 (L∞=8), layer 27 (L∞=7). Early and final layers worst.
- **Llama 8B**: layer 25 (L∞=9). Concentrated in a single layer; grows with sequence length (pos=1165 has worst case).
- **Llama 70B**: layer 70 (L∞=9), layer 7 (L∞=6), layer 13 (L∞=6). Scattered across depth.

**Sequence length trend**: L∞ weakly increases with sequence length. Worst cases consistently appear at the longest positions tested (pos=1165 for 1K-token generations). This suggests softmax precision diverges more with longer KV caches.

**Key interpretation**:
1. **The honest corridor is NOT zero.** τ ≥ 9 is required for honest W8A8 providers on A100 with SDPA. This is the same magnitude as the previously measured adversarial sensitivity threshold.
2. **~93–97% of elements are exactly equal** between GPU and verifier. The gap is concentrated in rare outliers, not systemic drift.
3. **>99.9% of elements are within ±1.** The bulk of the distribution is tight; only a few elements per layer per token reach diff≥5.
4. **The gap is NOT from backend non-determinism** (which was shown to be zero in §1h). It's from GPU fp16 + SDPA vs CPU f64 arithmetic — different rounding, different softmax intermediate precision, different accumulation order.
5. **Score witnessing remains load-bearing.** With τ=9, the sensitivity analysis from §1c–1g shows single-layer perturbations can flip tokens. The corridor cannot be closed by backend alignment alone.

**What this means for the protocol**:
- The current honest corridor (τ≈8–9) is dominated by the fp16-to-f64 arithmetic mismatch, not by stochastic variation or backend differences.
- Reducing τ requires either: (a) matching the verifier's arithmetic to the GPU's (fp16 replay), (b) using higher-precision retained state (INT16/FP16), or (c) per-head quantization scales.
- Until τ can be reduced below the adversarial sensitivity threshold (~4–5 for single-layer attacks), score witnessing or a state-replacing verifier remains necessary.

Script: `scripts/modal/measure_corridor.py`
Results: Modal volume `corridor-results/` (`qwen_corridor.json`, `llama_corridor.json`, `llama70b_corridor.json`)

### 1j. Replay precision ladder (2026-04-04)

**Question**: Can the sidecar-to-verifier gap be reduced by matching the verifier's replay precision to the GPU's?

**Method**: Parameterized replay with 4 precision levels:
- `F64` — current verifier default (over-precise vs GPU)
- `F32` — tests whether f32 accumulation alone closes the gap
- `Fp16InputsF32Accum` — fp16 input truncation + f32 accumulation (closest to GPU SDPA with fp16 tensors)
- `Bf16InputsF32Accum` — bf16 input truncation + f32 accumulation (for bf16 models)

Implementation: `ReplayPrecision` enum in `verilm_core::attention`, `measure_corridor_committed_kv_precision()` in `verilm_verify::corridor`, `measure_corridor_precision()` PyO3 binding. Uses `half` crate for real IEEE fp16/bf16 round-trips.

**Invariant check**: On real GPU audit payloads, `measure_corridor_precision(..., "f64")` is **exactly equal** to the legacy `measure_corridor_committed_kv(...)`. The new dispatch path is verified correct.

**Results** (vLLM 0.18.0 V1 engine, A100-80GB, Qwen 7B + Llama 8B W8A8):

| Precision | Max delta vs f64 (L∞) |
|-----------|----------------------|
| f64 → f32 | **0** (identical everywhere) |
| f64 → fp16_f32 | **±1** |
| f64 → bf16_f32 | **±2** |

All four precision levels produce L∞ within ±2 of each other across all workloads, positions, layers, and both models.

**Interpretation**:
1. **Replay precision is definitively NOT the gap source.** Changing from f64 to f32 makes zero difference. Adding fp16/bf16 input truncation changes L∞ by at most ±2.
2. **The gap is entirely upstream of the attention inner loop** — in the capture/dequant/requant pipeline, not in how QK^T/softmax/V-sum is computed.
3. This rules out the "over-precise verifier" hypothesis and the "fp16 input truncation" hypothesis simultaneously.

**Note on absolute L∞**: These runs were taken before the RoPE off-by-one fix (§1k) and showed L∞=100-186 due to that bug. With the fix applied, absolute L∞ returns to 8-9 (§1i). The precision comparison remains valid regardless — it depends on relative differences, not absolute magnitudes.

Script: `redteam/modal/measure_corridor_precision.py`
Results: Modal volume `corridor-results/qwen_precision_ab.json`

### 1k. L-inf regression root cause (2026-04-04)

**Problem**: §1i showed L∞=8-9 on the original code. Subsequent runs showed L∞=100-186 for decode tokens while prefill (pos=0) remained L∞=1.

**Root cause**: An off-by-one error in `compute_kv_transcript()` (`crates/verilm-prover/src/lib.rs`). The line `let absolute_pos = token_pos + 1` was introduced after the initial snapshot, shifting all KV RoPE positions by +1 relative to the GPU's actual positions.

**Why pos=0 was unaffected**: With a single KV entry, softmax is trivially [1.0] regardless of the score. The RoPE offset on K doesn't affect the attention output because there's only one entry to attend to — the output is just V. With multiple entries (decode tokens), the wrong RoPE positions produce wrong QK^T inner products, wrong attention distributions, and L∞=100+ differences.

**Fix**: Reverted `absolute_pos = token_pos + 1` back to `token_pos`, matching the original code. All 480+ workspace tests pass.

**Eliminated during investigation**:
- vLLM V1 vs V0 engine (both showed same L∞)
- vLLM version bisection 0.8.3–0.18.0 (all pinned same torch/compressed-tensors)
- Replay precision (f64/f32/fp16/bf16 — all within ±2)
- Sidecar capture code (unchanged since initial snapshot, Rust hook path correct)
- C++ native capture .max() reduction (not in critical path)

**GPU confirmation (2026-04-04)**: Re-ran `test_decode_corridor_regression.py` on A100-80GB with the fix:

| Model | Position | L∞ | Threshold |
|-------|----------|-----|-----------|
| Qwen 7B W8A8 | prefill (pos=0) | 1 | ≤ 2 |
| Qwen 7B W8A8 | first_gen (pos=6) | 4 | ≤ 12 |
| Qwen 7B W8A8 | first_decode (pos=7) | 8 | ≤ 12 |
| Llama 8B W8A8 | prefill (pos=0) | 1 | ≤ 2 |
| Llama 8B W8A8 | first_gen (pos=7) | 4 | ≤ 12 |
| Llama 8B W8A8 | first_decode (pos=8) | 2 | ≤ 12 |

**Status**: Fixed and confirmed. L∞ is back to single digits, consistent with §1i.

### 1l. Per-head `scale_a` boundary simulation (2026-04-04)

**Question**: Would replacing the current tensor-wide `scale_a` with one scale per Q-head materially shrink the honest corridor or the adversarial float-space room?

**Method**: Keep the same audit payloads and replay path, but simulate an alternative retained boundary:
- current protocol: one tensor-wide `scale_a`
- alternative: one `scale_a` per Q-head

Measure:
- L∞ under the simulated INT8 boundary
- head-scale distribution relative to the tensor scale
- worst-case float-space room ratio

**Results**:

| Metric | Qwen 7B | Llama 8B |
|--------|---------|----------|
| Tensor L∞ | 11 | 6 |
| Per-head L∞ | 44 | 47 |
| Worst float room ratio | 1.06 | 1.05 |
| Head scale max ratio | 1.00 | 1.05 |
| Head scale median ratio | 0.39 | 0.30 |

**Interpretation**:
1. **Per-head `scale_a` does not reduce the worst-case adversarial room.** One head always saturates the tensor-wide scale, so the worst head's room stays essentially unchanged (`ratio ≈ 1.0`).
2. **The honest INT8 corridor gets worse.** Smaller heads get smaller scales, which magnifies the same float mismatch when re-expressed in INT8 units. The simulated per-head L∞ is 4–8× larger.
3. **The scale distribution is genuinely skewed.** Most heads are only 30–40% of the tensor scale, but that does not help because the dominant head still sets the attack surface.

**Conclusion**: Per-head `scale_a` is a dead end as a τ-reduction path for the current protocol. The remaining realistic retained-boundary candidate is **INT16 / FP16 retained `a`**.

Script: `redteam/modal/simulate_perhead_boundary.py`

---

## 2. Interpretation

### What is established

1. **RMSNorm contraction does not hold for Llama**. The corridor tolerance is not self-dampening on this architecture.
2. **Single-layer perturbations can flip tokens** on both Qwen and Llama with real production weights. This is not a toy-model artifact.
3. **On Qwen, nearly every layer can flip the token on every prompt tested.** The corridor is structurally exploitable, not just edge-case vulnerable.
4. **Early and final layers dominate Llama's risk**. Layer 31 is consistently the most dangerous (top-3 on all 6 prompts). Layers 0 and 28 are secondary.
5. **The crude ρ bound is loose by 2–4×** but the local operator norm does not rescue Llama — it remains far above 1.0 at every layer.
6. **Multi-prompt sensitivity varies significantly**: Llama with high-margin prompts (fibonacci, margin=9.81) survives all single-layer attacks. Low-margin prompts are vulnerable. Qwen is vulnerable regardless of margin.
7. **MLP dampens perturbations on Llama** (ratio 0.67) and is neutral on Qwen (ratio 1.0). MLP amplification is NOT a concern.
8. **All-layers-simultaneously always flips tokens** on both models across all prompts. Compounding drift is real.
9. **Drift compounds because the verifier is non-state-replacing**: it follows the committed state forward, accepting ±τ per layer.
10. **All attention backends (eager, SDPA, eager+det) are bit-exact run-to-run on the prover side.** Backend mismatch appears to dominate the previously measured honest corridor. This makes deterministic, backend-aligned replay a much more promising path. But this does NOT yet prove protocol-level τ=0 — the verifier does CPU FP64 replay, not the same GPU backend.
11. **Eager attention is broken with W8A8 compressed_tensors** — produces wrong tokens. SDPA is the only correct backend for W8A8.
12. **For fp16 models (Llama), eager vs SDPA differs by only L∞=0.004** — and produces identical tokens. Backend choice barely matters for fp16.
13. **The sidecar-to-verifier gap is L∞=8–9 for W8A8 models on A100.** Measured on the actual vLLM sidecar path with SDPA attention vs Rust verifier CPU f64 replay. The gap is NOT zero — it matches the previously measured adversarial sensitivity threshold. ~93–97% of elements are exact, but outliers reach diff=8–9 in specific layers.
14. **The honest corridor is dominated by fp16-to-f64 arithmetic mismatch**, not by backend non-determinism (which is zero) or stochastic variation. Reducing τ requires matching arithmetic precision, not just fixing backends.
15. **Worst-case layers are model-specific**: Qwen layers 1, 27; Llama-8B layer 25; Llama-70B layers 7, 13, 70. Sequence length weakly increases the gap.
16. **Replay precision is NOT the gap source (§1j).** All four replay precisions (f64, f32, fp16+f32, bf16+f32) produce L∞ within ±2 of each other. The gap is entirely upstream of the attention inner loop — in the capture/dequant/requant pipeline, not in how QK^T/softmax/V-sum is computed.
17. **The L∞ regression was a RoPE position off-by-one (§1k).** The `absolute_pos = token_pos + 1` bug in `compute_kv_transcript` shifted all committed KV RoPE positions by +1, producing L∞=100-186 for decode tokens. Prefill (pos=0) was unaffected because single-entry softmax is trivially [1.0]. Fix: revert to `token_pos`. **GPU-confirmed**: L∞ returns to 1-8 (Qwen) and 1-4 (Llama).
18. **Per-head `scale_a` is ruled out (§1l).** It does not shrink the worst-case float room because one head always matches the tensor-wide maximum, and it makes the INT8 corridor substantially worse on the quieter heads.

### What is NOT yet established

1. **Multi-token accumulation**: autoregressive decode means small per-token drifts may accumulate across the generated sequence. Single-token sensitivity may underestimate the real risk.
2. **Constrained adversarial optimization**: the SVD-aligned perturbation is a heuristic worst case under L∞ constraint. True worst-case requires solving a constrained optimization problem on real layers.
3. **Fiat-Shamir / random sampling conditioning**: the adversary doesn't know which layers will be audited until after commitment. If deep audit samples layers randomly, the adversary must hedge across all layers rather than concentrating on the most dangerous ones. This probabilistic conditioning may substantially reduce effective risk compared to worst-case single-layer analysis.
4. **Whether a state-replacing verifier is feasible**: if the verifier canonicalized after each check, drift wouldn't compound. Cost and design implications are unexplored.
5. **Cross-hardware stability.** A100 vs H100 (or other GPU generations) may differ in floating-point behavior even with the same backend. Untested.
6. **Whether INT16 / FP16 retained `a` materially works.** fp16 replay is **ruled out** (§1j — it doesn't help) and per-head scales are **ruled out** (§1l — they do not shrink worst-case room). The remaining path's implementation cost and τ improvement are unknown.
7. ~~**Root cause of the L∞ regression — FIXED and GPU-CONFIRMED (§1k).**~~ Off-by-one in `compute_kv_transcript`: `absolute_pos = token_pos + 1` shifted all KV RoPE positions by +1. Fixed by reverting to `token_pos`. GPU-confirmed on A100: L∞ returns to single digits. See §1k.

### Protocol implications

- **The honest corridor is τ≈8–9, confirmed end-to-end (§1i).** The sidecar-to-verifier measurement shows L∞=8 (Qwen) and L∞=9 (Llama) on A100 with SDPA. This is the actual protocol gap, not a proxy.
- **The gap is dominated by fp16-to-f64 arithmetic mismatch.** Backend non-determinism is zero (§1h); the remaining gap is from GPU fp16 softmax/accumulation vs CPU f64 replay.
- **Score witnessing remains load-bearing.** With τ=9, the sensitivity analysis (§1c–1g) shows single-layer perturbations can flip tokens. The corridor cannot be closed by backend alignment alone.
- **For W8A8 models, mandate SDPA and fail closed on eager.** Eager produces wrong outputs with compressed_tensors.
- **For fp16 models, backend choice barely matters** on the prover side — eager vs SDPA differs by only L∞=0.004 with identical tokens.
- **Reducing τ is now the priority.** The remaining retained-boundary path is **INT16/FP16 retained state**. fp16 verifier replay is ruled out (§1j — precision matching doesn't help) and per-head scales are ruled out (§1l — one head always dominates).
- **KV provenance remains critical regardless.** Even perfect current-token attention only proves consistency with the committed prefix, not true earlier execution.
- **Bad-layer targeting** could make score witnessing cheaper: Llama layer 31 alone accounts for the largest adversarial risk; Qwen layers 24-25 are consistently worst. Corridor worst layers (Qwen 1/27, Llama 25) are partially different from adversarial worst layers.
- **Random audit sampling provides probabilistic conditioning**: an adversary who doesn't know which layers will be opened must spread risk, reducing the expected payoff from corridor exploitation.
- **Compounding drift is the core issue under τ>0**: even if single-layer perturbations were small, they accumulate because the verifier follows committed state.

### Practical next steps (ordered)

1. ~~**Done**: Bind attention backend in the manifest/protocol.~~ `attn_backend`, `attn_dtype` in manifest/key/verifier. SDPA mandated for W8A8, fail closed on eager and unknown.
2. ~~**Done**: Measure the real protocol gap — sidecar-to-verifier.~~ L∞=8–9 on A100. See §1i.
3. ~~**Done**: Replay precision ladder.~~ fp16/f32/bf16 replay does not reduce the gap (§1j). Precision matching is ruled out as a τ-reduction path.
4. ~~**Done**: Fix the L∞ regression.~~ Root cause: off-by-one RoPE position in `compute_kv_transcript` (`absolute_pos = token_pos + 1`). Reverted to `token_pos`. See §1k. GPU-confirmed: L∞ returns to single digits on both Qwen and Llama.
5. **Next**: Cross-hardware checks — same backend on A100 vs H100. If hardware changes the gap, τ must accommodate hardware diversity.
6. **Next**: Investigate τ reduction paths (priority order):
   a. INT16 retained state (256× finer granularity, moderate protocol change)
   b. FP16 retained state (same storage as INT16, different rounding / compatibility tradeoffs)
7. **Keep regardless**: Score witnessing until τ < adversarial sensitivity threshold (~4–5). KV provenance — independent of attention gap resolution.

---

## 3. Open experiments

Two separate goals:
- **Shrink the honest corridor**: make τ smaller (reduce the gap at the source)
- **Shrink adversarial freedom**: make passing within τ less useful (reduce what the adversary can do with the gap)

### ~~3a. Multi-prompt sensitivity~~ DONE (1e)

### ~~3b. MLP propagation~~ DONE (1f)

### ~~3c. All-layers-simultaneously~~ DONE (1e)

---

### Shrink the honest corridor (make τ smaller)

### 3d. Better `a` boundary quantization

Keep the same protocol shape, but store attention output with a tighter boundary:
- **INT16 or FP16 retained `a`** instead of INT8

This is probably the cleanest remaining way to reduce the honest corridor without changing the whole protocol. INT16 gives 256× finer granularity. Per-head scales looked attractive in theory but were ruled out empirically in §1l.

**Why**: the current ±τ in i8 units maps to a large float perturbation because the retained INT8 boundary is coarse. Higher-precision storage directly shrinks the float-space perturbation.

### ~~3e. Alternative attention backends~~ PARTIALLY DONE (1h)

**Result**: All backends (eager, SDPA, eager+det) are **bit-exact run-to-run on the prover side** (HF backends, A100). Backend mismatch dominates the previously measured corridor. See §1h for full results.

**Still open**: (a) reproduce in vLLM sidecar path, (b) measure provider GPU output vs verifier CPU FP64 replay, (c) cross-hardware stability (A100 vs H100).

### 3f. Layer-specific and head-specific tolerances

Don't use one global τ. Calibrate by:
- Model family
- Layer (tight on dangerous layers 24-25/31, loose on safe ones)
- Head group (if some heads are consistently noisier)

**Why**: if only 3–6 layers are dangerous, tighten those to τ=1 and leave the rest at τ=10. The dangerous layers drive the exploitability.

---

### Shrink adversarial freedom (make ±τ less useful)

### 3g. Generated-token exactness

Always strengthen the current token being returned. The cheapest strong version:
- Exact or stronger evidence on the generated token (always witness, don't sample)
- Weaker sampled checks on earlier prefix tokens

**Why**: much more value than sampling random tokens uniformly. The generated token is the one the adversary wants to flip.

### 3h. Partial score witnessing

Before full score witnessing, test:
- Worst layers only (3 bad layers per model)
- Worst heads only (if head-subspace analysis shows concentration)
- Top-k scores + certified tail bound (see 3k)

**Why**: may capture most of the security gain at much lower payload cost.

### 3i. Head-subspace analysis

Measure whether the dangerous perturbation directions (those that cause flips) lie in specific attention head subspaces. If yes, per-head score witnessing may suffice.

**Why**: could reduce score witnessing cost from O(all heads) to O(dangerous heads).

### 3j. State-replacing on bad layers only

Canonicalize the hidden state after the 3 most dangerous layers (replace committed state with replay result). Cheaper than full state-replacing, breaks the compounding chain where it matters most.

**Why**: eliminates compounding drift at the critical points without requiring full inference state.

### 3k. Post-W_o verification

Check the residual delta (after W_o projection) instead of pre-W_o attention output. The W_o amplification is the core problem; checking after it catches the amplified value directly.

**Why**: moves the verification boundary to after the amplification step.

---

### Remaining analysis experiments

### 3l. Multi-token accumulation

Generate a 50-token sequence with and without per-token ±τ perturbations. Measure whether small per-token drifts accumulate into semantic divergence across autoregressive decode.

**Why**: single-token analysis may underestimate the risk. The verifier follows committed state forward, so even small per-token drift may compound into semantic divergence.

### 3m. Constrained adversarial optimization

For the top-5 dangerous layers, solve: maximize logit drift subject to ‖Δa‖∞ ≤ τ using projected gradient descent on real traces. Compare to the SVD-aligned heuristic.

**Why**: the SVD-aligned perturbation may not be the true worst case under L∞ constraint.

### 3n. Logit-margin translation

For each layer, compute the maximum logit change per unit hidden-state drift using the actual LM head matrix. Map the corridor tolerance directly to a logit-space bound.

**Why**: provides an operational safety criterion without needing the full composed theorem.

### 3o. Fiat-Shamir / random audit conditioning

Model the adversary's optimization problem under random layer sampling. Compute expected detection probability as f(perturbation strategy, sampling rate).

**Why**: quantifies the probabilistic security that sampling provides on top of score witnessing.

### 3p. KV tightening in parallel

Even perfect current-token attention only proves correctness relative to the committed prefix. Unless KV provenance (#6) is strong, the adversary can manipulate the prefix state that feeds into the verified attention computation.

**Why**: attention gap and KV provenance are complementary — closing one without the other leaves a path open.

### 3q. State-replacing verifier feasibility (full)

Analyze whether the verifier could canonicalize the hidden state after every layer check. Would eliminate compounding drift entirely but requires the verifier to maintain full inference state.

**Why**: if feasible, closes the attention gap without score witnessing. Big architectural change.

---

### Payload and cost experiments

### 3r. Score witnessing payload and verification cost

Measure actual CPU verification time for score witness replay in the Rust verifier path. Back-of-envelope FLOP estimates are unreliable because verification is likely memory-bound (softmax, serialization, audit plumbing), not FLOP-bound. Must benchmark, not estimate.

**Payload formula**: `bytes = sampled_tokens × witnessed_layers × avg_prefix_len × n_q_heads × bytes_per_score`. Note: for Fiat-Shamir-sampled decode tokens, the relevant prefix length is the **average prefix** (tokens span the decode window), not always the max context. This can halve worst-case estimates.

Approximate payload at different scales (fp16, 28 heads):

| Strategy | seq=4K | seq=32K | seq=128K |
|----------|--------|---------|----------|
| 64 tok × 3 bad layers | ~41 MB | ~330 MB | ~1.3 GB |
| 8 tok × 3 bad layers | ~5 MB | ~41 MB | ~164 MB |

Long-context deployments need payload compression.

### 3k. Top-k score sparsification with tail certification

Commit only top-k pre-softmax scores per head instead of full seq_len. This makes payload constant in sequence length. However, **top-k alone is not sound**: the tail of the softmax can still carry manipulated probability mass. A sound variant requires:

- Top-k scores per head
- **Plus** a certified tail bound or tail mass summary (e.g., log-sum-exp of tail scores)

This ensures the verifier can confirm that the committed top-k scores, combined with the tail bound, fully determine the softmax output to within a provable tolerance.

**Status**: promising compression idea, not yet a drop-in protocol replacement for full score witnessing. Add to research agenda, not mainline roadmap.

---

## 4. Relationship to protocol design

| Finding | Protocol response |
|---------|------------------|
| Token flips from single-layer ±τ | Score witnessing (#7) required for safety |
| 28/28 layers flip Qwen on every prompt | Corridor is structurally exploitable, not edge-case |
| Layer 31 dominates Llama risk | Bad-layer-targeted deep audit for Llama |
| Layers 24-25 dominate Qwen risk | Bad-layer-targeted deep audit for Qwen |
| L∞→L∞ norms ≫ spectral norms | Tolerance bounds (#23) must use L∞-compatible analysis |
| High-margin prompts survive on Llama | Margin-based safety is model and prompt dependent |
| MLP dampens (not amplifies) | MLP is not a threat channel |
| All-layers flips on every prompt | Compounding drift is real and dangerous |
| Verifier follows committed state | State-replacing design would eliminate compounding |
| Random audit conditioning (unmeasured) | Probabilistic security argument (#5) should quantify |
| Fiat-Shamir alone doesn't fix corridor | ±τ perturbations pass all checks regardless of sampling |
| Fiat-Shamir IS useful for | Cost reduction, shell/KV statistical coverage, making score witnessing affordable |
| Score witnessing on generated token | Closes attention gap for that token (τ=0); always witness, don't sample |
| Fiat-Shamir on prefix tokens | Probabilistic coverage for prefix integrity |
| CPU verification time unknown | Back-of-envelope FLOP estimates unreliable; must benchmark in verifier |
| Top-k score sparsification | Promising but requires certified tail bound; research, not mainline yet |
| Payload scales with context | Long-context (32K+) needs bad-layer targeting or compression |

---

## 5. Scripts

| Script | What it measures |
|--------|-----------------|
| `redteam/modal/measure_contraction.py` | Crude ρ per layer, accumulated drift bound |
| `redteam/modal/measure_sensitivity.py` | Local operator norm, finite-difference flips, per-head norms, L∞ norms, bad-layer ranking (single prompt) |
| `redteam/modal/measure_sensitivity_v2.py` | Multi-prompt sensitivity, MLP propagation, all-layers-simultaneously, bad-layer consistency |
