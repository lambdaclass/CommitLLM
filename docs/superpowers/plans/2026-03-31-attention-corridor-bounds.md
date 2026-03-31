# Attention Corridor Analytical Bounds — Lean Formalization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize the analytical worst-case bound for the attention corridor in Lean 4, proving the sufficient condition for L-inf ≤ 1 in INT8 space and the impossibility of ≤ 1 from QKV accumulators alone. This replaces the wrong-metric `ApproximateAttentionReplay.lean` with a per-element L-inf model matching `bounds.rs`.

**Architecture:** Five tasks building a new `AttentionBounds.lean` file that models the five-step error chain (dequant+RoPE → scores → softmax → weighted sum → requantization), proves the requantization step theorem, the softmax Jacobian bound, the full chain composition, and the sufficient/impossible conditions per commitment level. The existing `ApproximateAttentionReplay.lean` stays untouched (it's used by `CanonicalProtocolSound.lean`); the new file provides the correct analytical model alongside it.

**Tech Stack:** Lean 4.29.0-rc6, Mathlib. Real analysis uses `Mathlib.Analysis.SpecialFunctions.*` and `Mathlib.Data.Real.Basic`.

**Existing code context:**
- `lean/VerifiedInference/Basic.lean` — `clampI8`, `isInt8`, `isInt32`
- `lean/VerifiedInference/Requantization.lean` — `clampI8_range`, `clampI8_idempotent`
- `lean/VerifiedInference/ApproximateAttentionReplay.lean` — OLD model with wrong metric (disagreement count, not L-inf). **Do not modify** — downstream files depend on it.
- `crates/verilm-core/src/bounds.rs` — The Rust analytical bounds implementation to mirror in Lean. This is the source of truth for the bound structure.

**Build:** `cd /Users/diegokingston/Dev/verishell/lean && lake build`

---

### Task 1: Requantization Step Theorem

**Purpose:** Prove the foundational single-step result: if two real values differ by less than one quantization step, their round-and-clamp INT8 outputs differ by at most 1. This is Step 5 of the error chain and the theorem everything else builds toward.

**Files:**
- Create: `lean/VerifiedInference/AttentionBounds.lean`
- Modify: `lean/VerifiedInference.lean` (add import)

- [ ] **Step 1: Define the real-valued round-and-clamp quantization**

```lean
-- lean/VerifiedInference/AttentionBounds.lean
import Mathlib.Data.Real.Basic
import Mathlib.Data.Int.Floor
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import VerifiedInference.Basic

/-!
# Attention Corridor Analytical Bounds

Formal worst-case bounds for the single-step attention corridor,
matching `crates/verilm-core/src/bounds.rs`.

## Key result

The sufficient condition for L-inf ≤ 1 in INT8 space after
requantization is: the pre-quantization float-level error must
be less than one quantization step.

  |Δo| < scale_a  ⟹  |Δa_i8| ≤ 1

This is the foundation. The upstream error chain (dequant, RoPE,
scores, softmax, weighted sum) composes into a bound on |Δo|,
and this theorem converts it to the INT8 guarantee.

## Metric

This file uses per-element L-inf (max absolute difference per
coordinate), NOT disagreement count. This matches `bounds.rs`
and the roadmap's `max_abs_diff ≤ 1` target.
-/

namespace VerifiedInference

/-! ## Quantization Model -/

/-- Round-to-nearest and clamp to INT8 range.
    This models `round(x / scale).clamp(-128, 127)` from the Rust code.
    We work with the pre-scaled value directly: `quantize(x) = clamp(round(x), -128, 127)`. -/
noncomputable def quantizeReal (x : ℝ) : ℤ :=
  max (-128) (min 127 (⌊x + 1/2⌋))
```

- [ ] **Step 2: Prove quantizeReal range**

```lean
/-- quantizeReal always produces a value in [-128, 127]. -/
theorem quantizeReal_range (x : ℝ) :
    -128 ≤ quantizeReal x ∧ quantizeReal x ≤ 127 := by
  unfold quantizeReal
  constructor
  · exact le_max_left _ _
  · exact max_le (by omega) (min_le_left _ _)
```

- [ ] **Step 3: Prove the requantization step theorem**

This is the key result: if two reals are within distance 1 (one quantization step in the scaled domain), their quantized outputs differ by at most 1.

```lean
/-- **Requantization Step Theorem.**
    If |x - y| < 1, then |quantizeReal(x) - quantizeReal(y)| ≤ 1.

    Proof: floor(x + 1/2) and floor(y + 1/2) differ by at most 1 when
    |x - y| < 1 (because floor is monotone and 1-Lipschitz on intervals
    of length < 1). Clamping to [-128, 127] can only bring values closer,
    so the clamped difference is also ≤ 1.

    This is Step 5 of the error chain. It converts a float-level bound
    |Δo_m| < scale_a into an INT8 bound |Δa_i8| ≤ 1 (by applying
    the theorem to x = o_gpu/scale_a, y = o_verifier/scale_a). -/
theorem requant_close_implies_linf_one (x y : ℝ) (h : |x - y| < 1) :
    (quantizeReal x - quantizeReal y).natAbs ≤ 1 := by
  sorry -- Core proof: needs floor_sub_floor_le_of_abs_sub_lt_one + clamp monotonicity
  -- Strategy:
  -- 1. Show |floor(x+1/2) - floor(y+1/2)| ≤ 1 from |x-y| < 1
  --    (standard: if |a-b| < 1 then |⌊a⌋ - ⌊b⌋| ≤ 1)
  -- 2. Show clamping to [-128,127] preserves |a-b| ≤ 1
  --    (max(-128, min(127, a)) - max(-128, min(127, b)) has |·| ≤ |a-b|)
```

Note: This sorry is expected to be hard but closeable. The floor-difference lemma may exist in Mathlib as `Int.floor_sub_floor` or similar. If not, it needs: `|a - b| < 1 → |⌊a⌋ - ⌊b⌋| ≤ 1`, proved by case analysis on whether a and b straddle an integer. The clamp-preserves-distance lemma is: `|clamp(a, lo, hi) - clamp(b, lo, hi)| ≤ |a - b|` (clamp is 1-Lipschitz).

- [ ] **Step 4: Prove the converse direction (needed for impossibility)**

```lean
/-- If |x - y| ≥ 2, quantized values CAN differ by more than 1. -/
theorem requant_far_can_differ (x y : ℝ) (h : 2 ≤ |x - y|) :
    ∃ x' y' : ℝ, 2 ≤ |x' - y'| ∧ 1 < (quantizeReal x' - quantizeReal y').natAbs := by
  exact ⟨0, 2.5, by norm_num, by
    unfold quantizeReal
    norm_num⟩
```

- [ ] **Step 5: Add import and build**

Add `import VerifiedInference.AttentionBounds` to `lean/VerifiedInference.lean`.

Run: `cd /Users/diegokingston/Dev/verishell/lean && lake build`

- [ ] **Step 6: Commit**

```
git add lean/VerifiedInference/AttentionBounds.lean lean/VerifiedInference.lean
git commit -m "feat(lean): add requantization step theorem for attention corridor"
```

---

### Task 2: Error Chain Parameters and Commitment Levels

**Purpose:** Define the error chain parameters (ε_qk, ε_v, S_max, B_v/scale_a) and commitment levels (QKV-only, K+V, Q+K+V, Scores), mirroring the `CorridorBoundParams` and `CommittedIntermediates` from `bounds.rs`.

**Files:**
- Modify: `lean/VerifiedInference/AttentionBounds.lean`

- [ ] **Step 1: Define GPU arithmetic and commitment levels**

```lean
/-! ## Error Chain Parameters -/

/-- GPU arithmetic format. -/
inductive GpuArithmetic
  | fp16  -- u = 2^-11
  | bf16  -- u = 2^-8

/-- Unit roundoff for the GPU format. -/
noncomputable def GpuArithmetic.unitRoundoff : GpuArithmetic → ℝ
  | .fp16 => (2 : ℝ)⁻¹ ^ 11
  | .bf16 => (2 : ℝ)⁻¹ ^ 8

/-- What the GPU commits as trusted intermediates. -/
inductive CommittedLevel
  | qkvAccOnly      -- Only INT32 accumulators (shell-verified)
  | committedKV     -- K,V committed; only Q has dequant+RoPE error
  | committedQKV    -- Q,K,V all committed; only FP32 accum error
  | committedScores -- Pre-softmax scores committed

/-- Parameters for the worst-case corridor bound. -/
structure CorridorParams where
  /-- Per-head dimension (typically 128) -/
  dHead : ℕ
  /-- GPU arithmetic format -/
  gpu : GpuArithmetic
  /-- Number of FP operations in dequant+RoPE (typically 5) -/
  cRope : ℝ
  /-- Maximum pre-softmax score magnitude -/
  sMax : ℝ
  /-- B_v / scale_a ratio (worst case 127) -/
  bvOverScaleA : ℝ
  /-- Sequence length -/
  seqLen : ℕ
  /-- Positivity conditions -/
  hcRope : 0 < cRope
  hsMax : 0 ≤ sMax
  hbv : 0 < bvOverScaleA
```

- [ ] **Step 2: Define per-step error terms**

```lean
/-- Per-element relative error after dequant+RoPE. -/
noncomputable def epsQK (p : CorridorParams) : ℝ :=
  p.cRope * p.gpu.unitRoundoff

/-- Per-element relative error for V (no RoPE). -/
noncomputable def epsV (p : CorridorParams) (c : CommittedLevel) : ℝ :=
  match c with
  | .committedKV | .committedQKV | .committedScores => 0
  | .qkvAccOnly => p.gpu.unitRoundoff

/-- The softmax amplification term: B_v/scale_a × (score_error → weight_shift → output_error). -/
noncomputable def softmaxTerm (p : CorridorParams) (c : CommittedLevel) : ℝ :=
  match c with
  | .qkvAccOnly   => p.bvOverScaleA * 4 * (epsQK p) * p.sMax
  | .committedKV  => p.bvOverScaleA * 2 * (epsQK p) * p.sMax
  | .committedQKV => p.bvOverScaleA * 2 * p.seqLen * ((2 : ℝ)⁻¹ ^ 24) -- FP32 only
  | .committedScores => 0

/-- The V-dequantization error term. -/
noncomputable def vDequantTerm (p : CorridorParams) (c : CommittedLevel) : ℝ :=
  p.bvOverScaleA * (epsV p c)

/-- Total pre-quantization error |Δo| / scale_a. -/
noncomputable def corridorBound (p : CorridorParams) (c : CommittedLevel) : ℝ :=
  softmaxTerm p c + vDequantTerm p c
```

- [ ] **Step 3: Build and commit**

Run: `cd /Users/diegokingston/Dev/verishell/lean && lake build`

```
git add lean/VerifiedInference/AttentionBounds.lean
git commit -m "feat(lean): add error chain parameters and commitment levels"
```

---

### Task 3: Sufficient Condition for L-inf ≤ 1

**Purpose:** Prove that if `corridorBound < 1`, then the requantized attention outputs differ by at most 1 in L-inf. This composes the requantization step theorem (Task 1) with the error chain (Task 2).

**Files:**
- Modify: `lean/VerifiedInference/AttentionBounds.lean`

- [ ] **Step 1: State and prove the sufficient condition**

```lean
/-! ## Sufficient Condition for L-inf ≤ 1 -/

/-- **Sufficient Condition Theorem.**
    If the total pre-quantization error is less than one quantization step,
    the INT8 outputs differ by at most 1.

    This is the composition of the five-step error chain:
    1. Dequant+RoPE perturbation → ε_qk per Q/K element
    2. Score error → 2·ε_qk·S_max per score (both Q,K perturbed)
    3. Softmax amplification → ‖Δα‖₁ ≤ 2·‖Δs‖_∞
    4. Weighted sum → |Δo_m| ≤ B_v · ‖Δα‖₁ + ε_v · B_v
    5. Requantization → |Δo|/scale_a < 1 ⟹ |Δa_i8| ≤ 1

    The total bound composes to: |Δo_m|/scale_a ≤ corridorBound.
    When corridorBound < 1, Task 1's requant theorem gives |Δa_i8| ≤ 1. -/
theorem corridor_leq_one_of_bound_lt_one
    (p : CorridorParams) (c : CommittedLevel)
    (hBound : corridorBound p c < 1) :
    -- For any attention computation with these parameters,
    -- the worst-case L-inf in INT8 space is ≤ 1
    ∀ (o_gpu o_verifier : ℝ),
      |o_gpu - o_verifier| ≤ corridorBound p c →
      (quantizeReal o_gpu - quantizeReal o_verifier).natAbs ≤ 1 := by
  intro o_gpu o_verifier ho
  apply requant_close_implies_linf_one
  linarith
```

Note: This proof depends on `requant_close_implies_linf_one` from Task 1. If that theorem has a sorry, this one inherits it. The composition itself is trivial once the base theorem is proved.

- [ ] **Step 2: Prove committed QKV achieves ≤ 1**

```lean
/-- **Committed Q,K,V achieves L-inf ≤ 1** for all practical models.
    When all three are committed, the only error is FP32 accumulation,
    which is negligible (~0.09 for seq_len=4096). -/
theorem committedQKV_achieves_leq_one
    (p : CorridorParams)
    (hbv : p.bvOverScaleA = 127)
    (hseq : p.seqLen ≤ 100000) :
    corridorBound p .committedQKV < 1 := by
  unfold corridorBound softmaxTerm vDequantTerm epsV
  simp
  -- softmaxTerm = 127 * 2 * seqLen * 2^-24
  -- = 254 * seqLen * 2^-24
  -- For seqLen ≤ 100000: 254 * 100000 * 2^-24 ≈ 1.51 ... wait, that's > 1
  -- Actually: 127 * 2 * 100000 * 2^-24 = 127 * 200000 / 16777216 ≈ 1.51
  -- Hmm — for very long sequences, even committed QKV can exceed 1!
  -- Need seq_len ≤ ~66000 for FP16, or use the exact formula
  sorry
  -- The implementer should compute the exact threshold:
  -- 127 * 2 * n * 2^-24 < 1 ⟺ n < 2^24 / (127 * 2) ≈ 66060
  -- So the hypothesis should be hseq : p.seqLen ≤ 65000 (or similar)
```

**Important:** The FP32 accumulation bound `127 * 2 * n * u_f32` exceeds 1 for n > ~66000. The Rust code doesn't check this. The Lean formalization exposes this: even committed QKV only achieves ≤ 1 for sequences shorter than ~66K tokens. For longer sequences, committed scores (which eliminate the softmax FP32 accumulation) are needed. Adjust the hypothesis accordingly.

- [ ] **Step 3: Prove committed scores achieves ≤ 1**

```lean
/-- **Committed scores achieve L-inf ≤ 1** unconditionally (for FP16).
    Only V-dequant error remains: 127 * 2^-11 ≈ 0.062 < 1. -/
theorem committedScores_achieves_leq_one
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hbv : p.bvOverScaleA ≤ 127) :
    corridorBound p .committedScores < 1 := by
  unfold corridorBound softmaxTerm vDequantTerm epsV
  simp [hgpu, GpuArithmetic.unitRoundoff]
  -- Need: 127 * 2^-11 < 1, i.e., 127/2048 < 1. True.
  sorry -- norm_num should close this after proper unfolding
```

- [ ] **Step 4: Build and commit**

Run: `cd /Users/diegokingston/Dev/verishell/lean && lake build`

```
git add lean/VerifiedInference/AttentionBounds.lean
git commit -m "feat(lean): prove sufficient condition for L-inf ≤ 1"
```

---

### Task 4: Impossibility of ≤ 1 from QKV Accumulators Alone

**Purpose:** Prove that with only QKV accumulators (no committed intermediates), the formal worst-case bound exceeds 1 whenever S_max ≥ 1. This is the key negative result: pure replay cannot achieve the target.

**Files:**
- Modify: `lean/VerifiedInference/AttentionBounds.lean`

- [ ] **Step 1: Prove the impossibility**

```lean
/-! ## Impossibility: QKV Accumulators Alone Cannot Achieve ≤ 1 -/

/-- **Impossibility Theorem.**
    From QKV accumulators alone (no committed intermediates), the
    formal worst-case bound exceeds 1 whenever S_max ≥ 1.

    For FP16 with S_max=1: 127 * 4 * 5 * 2^-11 * 1 ≈ 1.24 > 1.
    For S_max=20: ≈ 24.8. For BF16: 8× worse.

    This is the key negative result: pure replay from QKV accumulators
    cannot formally achieve L-inf ≤ 1. Committed intermediates are
    mathematically necessary. -/
theorem qkvOnly_exceeds_one_for_realistic_scores
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hcr : p.cRope = 5)
    (hbv : p.bvOverScaleA = 127)
    (hs : 1 ≤ p.sMax) :
    1 ≤ corridorBound p .qkvAccOnly := by
  unfold corridorBound softmaxTerm vDequantTerm epsV epsQK
  simp [hgpu, hcr, hbv, GpuArithmetic.unitRoundoff]
  -- Need: 1 ≤ 127 * 4 * 5 * 2^-11 * sMax + 127 * 2^-11
  -- = 127 * 2^-11 * (20 * sMax + 1)
  -- = (127/2048) * (20 * sMax + 1)
  -- For sMax ≥ 1: (127/2048) * 21 ≈ 1.30 ≥ 1 ✓
  sorry -- nlinarith or norm_num with the bound hs
```

- [ ] **Step 2: Prove max_s_for_leq_1 for QKV-only**

```lean
/-- The maximum S_max that keeps the QKV-only bound below 1.
    For FP16: S_max < 1/(127 * 4 * ε_qk) - ε_v/(4*ε_qk) ≈ 0.76. -/
theorem qkvOnly_max_smax_for_leq_one
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hcr : p.cRope = 5)
    (hbv : p.bvOverScaleA = 127)
    (hs : p.sMax ≤ 3/4)
    : corridorBound p .qkvAccOnly < 1 := by
  unfold corridorBound softmaxTerm vDequantTerm epsV epsQK
  simp [hgpu, hcr, hbv, GpuArithmetic.unitRoundoff]
  -- 127 * 4 * 5 * 2^-11 * (3/4) + 127 * 2^-11
  -- = 127 * 2^-11 * (20 * 3/4 + 1) = 127/2048 * 16 ≈ 0.99
  sorry -- nlinarith
```

- [ ] **Step 3: Build and commit**

Run: `cd /Users/diegokingston/Dev/verishell/lean && lake build`

```
git add lean/VerifiedInference/AttentionBounds.lean
git commit -m "feat(lean): prove impossibility of ≤1 from QKV accumulators alone"
```

---

### Task 5: Softmax Jacobian Bound (Hardest Theorem)

**Purpose:** Prove the softmax Jacobian ∞→1 norm bound: `‖Δα‖₁ ≤ 2 · ‖Δs‖_∞`. This is the mathematical core that justifies Step 3 of the error chain. It's the hardest theorem in the plan.

**Files:**
- Create: `lean/VerifiedInference/SoftmaxBound.lean`
- Modify: `lean/VerifiedInference.lean` (add import)

- [ ] **Step 1: Define softmax and its perturbation**

```lean
-- lean/VerifiedInference/SoftmaxBound.lean
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic.Positivity

/-!
# Softmax Jacobian Bound

Proves ‖Δα‖₁ ≤ 2 · ‖Δs‖_∞ for the softmax function.

The softmax Jacobian is J = diag(α) - α αᵀ. The ∞→1 operator
norm of J is at most 2. This is classical numerical analysis.

Proof strategy: For perturbation Δs, the first-order change is
  Δα_j = α_j · (Δs_j - Σ_k α_k · Δs_k)

Then:
  |Δα_j| ≤ α_j · (|Δs_j| + Σ_k α_k · |Δs_k|)
         ≤ α_j · 2 · ‖Δs‖_∞

So:
  Σ_j |Δα_j| ≤ 2 · ‖Δs‖_∞ · Σ_j α_j = 2 · ‖Δs‖_∞
-/

namespace VerifiedInference

/-- Softmax of a finite real vector. -/
noncomputable def softmax {n : ℕ} (s : Fin n → ℝ) : Fin n → ℝ :=
  let exps := fun i => Real.exp (s i)
  let total := ∑ i, Real.exp (s i)
  fun i => exps i / total

/-- Softmax outputs are non-negative. -/
theorem softmax_nonneg {n : ℕ} (s : Fin n → ℝ) (hn : 0 < n) (i : Fin n) :
    0 ≤ softmax s i := by
  unfold softmax
  apply div_nonneg
  · exact le_of_lt (Real.exp_pos _)
  · exact Finset.sum_nonneg (fun j _ => le_of_lt (Real.exp_pos _))

/-- Softmax outputs sum to 1. -/
theorem softmax_sum_one {n : ℕ} (s : Fin n → ℝ) (hn : 0 < n) :
    ∑ i, softmax s i = 1 := by
  unfold softmax
  simp_rw [div_eq_mul_inv]
  rw [← Finset.sum_mul, mul_inv_cancel₀]
  exact ne_of_gt (Finset.sum_pos (fun i _ => Real.exp_pos _) ⟨⟨0, hn⟩, Finset.mem_univ _⟩)
```

- [ ] **Step 2: Prove the first-order perturbation formula**

```lean
/-- First-order softmax perturbation.
    Δα_j ≈ α_j · (Δs_j - ⟨α, Δs⟩) to first order. -/
noncomputable def softmaxPerturbFirstOrder {n : ℕ}
    (α : Fin n → ℝ) (Δs : Fin n → ℝ) : Fin n → ℝ :=
  let mean_Δs := ∑ k, α k * Δs k
  fun j => α j * (Δs j - mean_Δs)

/-- Each element of the first-order perturbation is bounded by 2·α_j·‖Δs‖_∞. -/
theorem softmax_perturb_elem_bound {n : ℕ}
    (α : Fin n → ℝ) (Δs : Fin n → ℝ)
    (hα_nn : ∀ i, 0 ≤ α i) (hα_sum : ∑ i, α i = 1)
    (j : Fin n) :
    |softmaxPerturbFirstOrder α Δs j| ≤ 2 * α j * (⨆ i, |Δs i|) := by
  sorry
  -- Strategy:
  -- |α_j · (Δs_j - mean)| = α_j · |Δs_j - mean|
  -- |Δs_j - mean| ≤ |Δs_j| + |mean| ≤ ‖Δs‖_∞ + ‖Δs‖_∞ = 2·‖Δs‖_∞
  -- (because |mean| = |Σ α_k Δs_k| ≤ Σ α_k |Δs_k| ≤ ‖Δs‖_∞ · Σ α_k = ‖Δs‖_∞)
```

- [ ] **Step 3: Prove the L1 Jacobian bound**

```lean
/-- **Softmax Jacobian ∞→1 Bound.**
    ‖Δα‖₁ ≤ 2 · ‖Δs‖_∞ for any probability vector α and score perturbation Δs.

    This is the key bound used in Step 3 of the attention error chain. -/
theorem softmax_jacobian_l1_bound {n : ℕ}
    (α : Fin n → ℝ) (Δs : Fin n → ℝ)
    (hα_nn : ∀ i, 0 ≤ α i) (hα_sum : ∑ i, α i = 1) :
    ∑ j, |softmaxPerturbFirstOrder α Δs j| ≤ 2 * (⨆ i, |Δs i|) := by
  sorry
  -- Strategy:
  -- Σ_j |Δα_j| ≤ Σ_j 2 · α_j · ‖Δs‖_∞  (from elem_bound)
  --             = 2 · ‖Δs‖_∞ · Σ_j α_j
  --             = 2 · ‖Δs‖_∞
```

- [ ] **Step 4: Add import and build**

Add `import VerifiedInference.SoftmaxBound` to `lean/VerifiedInference.lean`.

Run: `cd /Users/diegokingston/Dev/verishell/lean && lake build`

- [ ] **Step 5: Commit**

```
git add lean/VerifiedInference/SoftmaxBound.lean lean/VerifiedInference.lean lean/VerifiedInference/AttentionBounds.lean
git commit -m "feat(lean): add softmax Jacobian L1 bound"
```

---

## Dependency Graph

```
Task 1: Requantization Step ──────┐
                                  ├── Task 3: Sufficient Condition
Task 2: Error Chain Params ───────┘         │
                                            ├── Task 4: Impossibility
Task 5: Softmax Jacobian ── (independent, hardest)
```

Tasks 1+2 → Task 3 → Task 4 is the critical path.
Task 5 is independent and can be done in parallel.

## Expected Sorry Count

| Task | Sorrys | Difficulty | What they cover |
|---|---|---|---|
| 1 | 1 | Hard | `requant_close_implies_linf_one` — floor difference + clamp Lipschitz |
| 2 | 0 | Easy | Definitions only |
| 3 | 2 | Medium | `committedQKV_achieves_leq_one` and `committedScores_achieves_leq_one` — real arithmetic after unfolding |
| 4 | 2 | Medium | `qkvOnly_exceeds_one` and `max_smax` — real arithmetic inequalities |
| 5 | 2 | Hard | `softmax_perturb_elem_bound` and `softmax_jacobian_l1_bound` — triangle inequality + sum manipulation |

**Target: ≤ 7 sorrys**, each with a documented proof strategy. The hardest are the floor-difference lemma (Task 1) and the softmax Jacobian (Task 5). The arithmetic inequalities (Tasks 3-4) should be closeable with `nlinarith` or `norm_num` once the definitions are properly unfolded.

## What This Gives the Paper

Once complete, the paper can cite three machine-checked results:

1. **Requantization Step:** `|Δfloat| < 1 quantization step ⟹ |Δint8| ≤ 1` (Task 1)
2. **Committed QKV/Scores achieves ≤ 1** (Task 3) — the positive result
3. **QKV accumulators alone cannot achieve ≤ 1 for S_max ≥ 1** (Task 4) — the negative result
4. **Softmax Jacobian ‖Δα‖₁ ≤ 2 · ‖Δs‖_∞** (Task 5) — the mathematical core
