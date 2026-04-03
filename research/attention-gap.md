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

---

## 2. Interpretation

### What is established

1. **RMSNorm contraction does not hold for Llama**. The corridor tolerance is not self-dampening on this architecture.
2. **Single-layer perturbations can flip tokens** on both Qwen and Llama with real production weights. This is not a toy-model artifact.
3. **Early and final layers dominate the risk**. Layers 0–1 and the last layer are consistently the most dangerous on both models. Middle layers are less sensitive but still consume significant margin.
4. **The crude ρ bound is loose by 2–4×** but the local operator norm does not rescue Llama — it remains far above 1.0 at every layer.

### What is NOT yet established

1. **Multi-prompt generality**: all measurements use a single short prompt ("The capital of France is"). Residual magnitudes, logit margins, and scale_a values are prompt-dependent. Results may be better or worse on other prompts.
2. **MLP amplification**: current finite-difference injects at the self_attn output (post-W_o). The perturbation also flows through the MLP in the same layer, which may amplify further. This channel is not yet measured.
3. **All-layers-simultaneously**: current tests perturb one layer at a time. An adversary perturbing all layers within ±τ could cause larger accumulated drift. But perturbations may partially cancel.
4. **Multi-token accumulation**: autoregressive decode means small per-token drifts may accumulate across the generated sequence. Single-token sensitivity may underestimate the real risk.
5. **Constrained adversarial optimization**: the SVD-aligned perturbation is a heuristic worst case under L∞ constraint. True worst-case requires solving a constrained optimization problem on real layers.
6. **Fiat-Shamir / random sampling conditioning**: the adversary doesn't know which layers will be audited until after commitment. If deep audit samples layers randomly, the adversary must hedge across all layers rather than concentrating on the most dangerous ones. This probabilistic conditioning may substantially reduce effective risk compared to worst-case single-layer analysis.

### Protocol implications

- **Score witnessing is critical**, not optional. Without it, the corridor tolerance is exploitable on both families.
- **Bad-layer targeting** could make score witnessing cheaper: if only 3–6 layers are dangerous, deep audit can focus on those rather than all layers.
- **The L∞ metric mismatch matters**: corridor is stated in L∞, but most analysis uses L2/spectral norms. Future bounds should use L∞-compatible analysis or explicitly justify the relaxation.
- **Random audit sampling provides probabilistic conditioning**: an adversary who doesn't know which layers will be opened must spread risk, reducing the expected payoff from corridor exploitation.

---

## 3. Open experiments

Ordered by expected decision value.

### 3a. Multi-prompt sensitivity

Run the finite-difference measurement on diverse prompts: short factual, long context, code, math, uncertain outputs, high-confidence outputs. Determine whether flips are prompt-dependent or structural.

**Why**: logit margin varies enormously by prompt. "The capital of France" may be unusually easy/hard.

### 3b. MLP propagation

Inject the perturbation before the attention residual addition (not after the full layer) and let it flow through the post-attention LayerNorm and MLP. Measure whether the MLP amplifies, dampens, or is neutral.

**Why**: current measurement only captures the W_o → residual path. The MLP is a second amplification channel that may dominate.

### 3c. All-layers-simultaneously

Inject independent ±τ perturbations at every layer and measure cumulative logit change. Compare worst-case-aligned vs random-sign perturbations.

**Why**: single-layer analysis doesn't capture accumulation. But this should come after 3a and 3b to ensure correct interpretation.

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

---

## 4. Relationship to protocol design

| Finding | Protocol response |
|---------|------------------|
| Token flips from single-layer ±τ | Score witnessing (#7) required for safety |
| 3–6 layers dominate risk | Bad-layer-targeted deep audit reduces cost |
| L∞→L∞ norms ≫ spectral norms | Tolerance bounds (#23) must use L∞-compatible analysis |
| Prompt-dependent margins (untested) | Multi-prompt calibration needed before setting policy |
| MLP amplification (unmeasured) | Must measure before any safety claims |
| Random audit conditioning | Probabilistic security argument (#5) should quantify |

---

## 5. Scripts

| Script | What it measures |
|--------|-----------------|
| `redteam/modal/measure_contraction.py` | Crude ρ per layer, accumulated drift bound |
| `redteam/modal/measure_sensitivity.py` | Local operator norm, finite-difference flips, per-head norms, L∞ norms, bad-layer ranking |
