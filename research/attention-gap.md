# Attention Gap: Corridor Sensitivity Analysis

This document tracks measurements, open questions, and the evolving understanding of the attention corridor's security properties. The corridor allows ±τ tolerance on INT8 attention outputs per layer. The central question: **can an adversary exploit this tolerance to flip output tokens?**

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

### What is NOT yet established

1. **Multi-token accumulation**: autoregressive decode means small per-token drifts may accumulate across the generated sequence. Single-token sensitivity may underestimate the real risk.
2. **Constrained adversarial optimization**: the SVD-aligned perturbation is a heuristic worst case under L∞ constraint. True worst-case requires solving a constrained optimization problem on real layers.
3. **Fiat-Shamir / random sampling conditioning**: the adversary doesn't know which layers will be audited until after commitment. If deep audit samples layers randomly, the adversary must hedge across all layers rather than concentrating on the most dangerous ones. This probabilistic conditioning may substantially reduce effective risk compared to worst-case single-layer analysis.
4. **Whether a state-replacing verifier is feasible**: if the verifier canonicalized after each check, drift wouldn't compound. Cost and design implications are unexplored.

### Protocol implications

- **Score witnessing is critical**, not optional. Without it, the corridor tolerance is exploitable on both families.
- **Bad-layer targeting** could make score witnessing cheaper: Llama layer 31 alone accounts for the largest risk; Qwen layers 24-25 are consistently worst.
- **The L∞ metric mismatch matters**: corridor is stated in L∞, but most analysis uses L2/spectral norms. Future bounds should use L∞-compatible analysis or explicitly justify the relaxation.
- **Random audit sampling provides probabilistic conditioning**: an adversary who doesn't know which layers will be opened must spread risk, reducing the expected payoff from corridor exploitation.
- **MLP is not a threat channel**: it dampens or is neutral, not amplifying.
- **Compounding drift is the core issue**: even if single-layer perturbations were small, they accumulate because the verifier follows committed state.

---

## 3. Open experiments

Ordered by expected decision value.

### ~~3a. Multi-prompt sensitivity~~ DONE (1e)

### ~~3b. MLP propagation~~ DONE (1f)

### ~~3c. All-layers-simultaneously~~ DONE (1e)

### 3d. Constrained adversarial optimization

For the top-5 dangerous layers, solve: maximize logit drift subject to ‖Δa‖∞ ≤ τ using projected gradient descent on real traces. Compare to the SVD-aligned heuristic.

**Why**: the SVD-aligned perturbation may not be the true worst case under L∞ constraint.

### 3e. Multi-token accumulation

Generate a 50-token sequence with and without per-token ±τ perturbations. Measure whether small per-token drifts accumulate into semantic divergence.

**Why**: single-token analysis may underestimate the risk for longer generations.

### 3f. Head-subspace analysis

Measure whether the dangerous perturbation directions (those that cause flips) lie in specific attention head subspaces. If yes, per-head score witnessing may suffice.

**Why**: could reduce score witnessing cost from O(all heads) to O(dangerous heads).

### 3g. Logit-margin translation

For each layer, compute the maximum logit change per unit hidden-state drift using the actual LM head matrix. Map the corridor tolerance directly to a logit-space bound.

**Why**: provides an operational safety criterion without needing the full composed theorem.

### 3h. Fiat-Shamir / random audit conditioning

Model the adversary's optimization problem under random layer sampling. If the auditor opens k of L layers uniformly, the adversary must choose which layers to perturb without knowing k. Compute the expected detection probability as a function of (perturbation strategy, sampling rate).

**Why**: worst-case single-layer analysis assumes the adversary can target the most dangerous layer. Random sampling forces them to hedge, reducing effective risk.

### 3i. State-replacing verifier feasibility

Analyze whether the verifier could canonicalize the hidden state after each layer check (replace committed state with replay result). This would eliminate compounding drift but requires the verifier to maintain full inference state during verification.

**Why**: if feasible, this architectural change could close the attention gap without score witnessing.

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

---

## 5. Scripts

| Script | What it measures |
|--------|-----------------|
| `redteam/modal/measure_contraction.py` | Crude ρ per layer, accumulated drift bound |
| `redteam/modal/measure_sensitivity.py` | Local operator norm, finite-difference flips, per-head norms, L∞ norms, bad-layer ranking (single prompt) |
| `redteam/modal/measure_sensitivity_v2.py` | Multi-prompt sensitivity, MLP propagation, all-layers-simultaneously, bad-layer consistency |
