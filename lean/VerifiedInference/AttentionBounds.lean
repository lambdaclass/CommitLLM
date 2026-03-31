import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Archimedean
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Order.Defs.LinearOrder

import VerifiedInference.Basic

/-!
# Attention Corridor Analytical Bounds

This module formalises the **per-element L∞ bound** (maximum absolute difference
per coordinate) for the attention-replay corridor used in the VeriLM protocol.
It corresponds to `bounds.rs` on the Rust side.

Unlike the disagreement-count model in `ApproximateAttentionReplay.lean`, this
module reasons about the **requantisation step** that converts continuous
(real-valued) attention outputs to INT8.  The central result is
`requant_close_implies_linf_one`:

> If two real values differ by less than 1, their round-and-clamp INT8 outputs
> differ by at most 1.

This is the **foundation of the five-step error chain**: each step in the
FP16→FP64→round→clamp pipeline can be bounded, and the requantisation step
contributes at most ±1 per coordinate when the pre-quantisation reals are
within 1 of each other.

## Overview

- `quantizeReal` — `round(x).clamp(-128, 127)` modelled as `max(-128, min(127, ⌊x+½⌋))`
- `quantizeReal_range` — output is always in `[-128, 127]`
- `requant_close_implies_linf_one` — the key per-element L∞ bound
- `requant_far_can_differ` — witness that far-apart inputs can differ by more than 1
-/

set_option autoImplicit false

namespace VerifiedInference

/-! ## Quantization: round-and-clamp to INT8 -/

/-- Round-to-nearest and clamp to INT8 range.  Models `round(x).clamp(-128, 127)` as
    `max(-128, min(127, ⌊x + 1/2⌋))`. -/
noncomputable def quantizeReal (x : ℝ) : ℤ :=
  max (-128) (min 127 (⌊x + 1/2⌋))

/-! ## Range theorem -/

/-- The output of `quantizeReal` is always in the INT8 range `[-128, 127]`. -/
theorem quantizeReal_range (x : ℝ) :
    -128 ≤ quantizeReal x ∧ quantizeReal x ≤ 127 := by
  unfold quantizeReal
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_left _ _)

/-! ## Helper: floor of close reals differ by at most 1 -/

/-- If `|a - b| < 1` then `|⌊a⌋ - ⌊b⌋| ≤ 1`.

Proof: From `Int.floor_le` we get `⌊x⌋ ≤ x` and from `Int.lt_floor_add_one`
we get `x < ⌊x⌋ + 1`.  Combining with `|a - b| < 1` we get
`⌊a⌋ < ⌊b⌋ + 2` and `⌊b⌋ < ⌊a⌋ + 2`, hence `|⌊a⌋ - ⌊b⌋| ≤ 1`. -/
private theorem floor_close (a b : ℝ) (h : |a - b| < 1) :
    |(⌊a⌋ : ℤ) - ⌊b⌋| ≤ 1 := by
  -- Extract the two-sided bound from |a - b| < 1
  rw [abs_lt] at h
  -- Floor bounds: ⌊x⌋ ≤ x < ⌊x⌋ + 1
  have hfa : (⌊a⌋ : ℝ) ≤ a := Int.floor_le a
  have hfb : (⌊b⌋ : ℝ) ≤ b := Int.floor_le b
  have hfa1 : a < (⌊a⌋ : ℝ) + 1 := Int.lt_floor_add_one a
  have hfb1 : b < (⌊b⌋ : ℝ) + 1 := Int.lt_floor_add_one b
  -- From a - b < 1 and ⌊a⌋ ≤ a and b < ⌊b⌋ + 1:
  --   ⌊a⌋ ≤ a < b + 1 < ⌊b⌋ + 1 + 1 = ⌊b⌋ + 2
  -- From -(a - b) < 1 (i.e. b - a < 1) and ⌊b⌋ ≤ b and a < ⌊a⌋ + 1:
  --   ⌊b⌋ ≤ b < a + 1 < ⌊a⌋ + 1 + 1 = ⌊a⌋ + 2
  have h1 : (⌊a⌋ : ℝ) < (⌊b⌋ : ℝ) + 2 := by linarith
  have h2 : (⌊b⌋ : ℝ) < (⌊a⌋ : ℝ) + 2 := by linarith
  -- Lift to ℤ
  have h1z : (⌊a⌋ : ℤ) < ⌊b⌋ + 2 := by exact_mod_cast h1
  have h2z : (⌊b⌋ : ℤ) < ⌊a⌋ + 2 := by exact_mod_cast h2
  -- Now |⌊a⌋ - ⌊b⌋| ≤ 1 follows by omega after unfolding abs
  rw [abs_le]
  omega

/-! ## Helper: integer clamping does not increase distance -/

/-- `max(lo, min(hi, ·))` is a non-expansive map on `ℤ`:
`|max(lo, min(hi, a)) - max(lo, min(hi, b))| ≤ |a - b|`.

This is the 1-Lipschitz property of projection onto `[lo, hi]`. -/
private theorem clamp_nonexpansive (a b lo hi : ℤ) (_hlohi : lo ≤ hi) :
    |max lo (min hi a) - max lo (min hi b)| ≤ |a - b| := by
  -- Clamp(x) = max(lo, min(hi, x)). We reduce to linear arithmetic via
  -- the identity: max(c, d) = c + (d - c) if d ≥ c, else c.
  -- Approach: use omega on the cases by providing linear bounds on max and min.
  have hmin_a : min hi a ≤ a := min_le_right _ _
  have hmin_a2 : min hi a ≤ hi := min_le_left _ _
  have hmin_b : min hi b ≤ b := min_le_right _ _
  have hmin_b2 : min hi b ≤ hi := min_le_left _ _
  have hmax_a : lo ≤ max lo (min hi a) := le_max_left _ _
  have hmax_a2 : min hi a ≤ max lo (min hi a) := le_max_right _ _
  have hmax_b : lo ≤ max lo (min hi b) := le_max_left _ _
  have hmax_b2 : min hi b ≤ max lo (min hi b) := le_max_right _ _
  -- max is one of the two arguments
  have hmax_a_cases : max lo (min hi a) = lo ∨ max lo (min hi a) = min hi a :=
    max_choice lo (min hi a)
  have hmax_b_cases : max lo (min hi b) = lo ∨ max lo (min hi b) = min hi b :=
    max_choice lo (min hi b)
  have hmin_a_cases : min hi a = hi ∨ min hi a = a := min_choice hi a
  have hmin_b_cases : min hi b = hi ∨ min hi b = b := min_choice hi b
  -- 2 × 2 × 2 × 2 = 16 cases, each linear. omega handles them all.
  -- Convert |x - y| ≤ |a - b| to -|a-b| ≤ x - y ∧ x - y ≤ |a-b|
  rw [abs_le]
  have hab1 : a - b ≤ |a - b| := le_abs_self _
  have hab2 : -(a - b) ≤ |a - b| := neg_le_abs (a - b)
  -- 16 cases, each linear after substitution
  rcases hmax_a_cases with ha | ha <;> rcases hmax_b_cases with hb | hb <;>
  rcases hmin_a_cases with hma | hma <;> rcases hmin_b_cases with hmb | hmb <;>
  (simp only [ha, hb, hma, hmb] at *; constructor <;> omega)

/-! ## The key theorem -/

/-- **Requantisation step bound**: if two reals differ by less than 1, their
round-and-clamp INT8 outputs differ by at most 1 (in `ℤ.natAbs`).

This is the foundational per-element L∞ result for the attention corridor:
each coordinate of the attention output, after requantisation, can shift by
at most 1 when the pre-quantisation values are within 1 of each other. -/
theorem requant_close_implies_linf_one (x y : ℝ) (h : |x - y| < 1) :
    (quantizeReal x - quantizeReal y).natAbs ≤ 1 := by
  unfold quantizeReal
  -- Step 1: shifting by 1/2 preserves the gap
  have hshift : |x + 1/2 - (y + 1/2)| < 1 := by ring_nf; exact h
  -- Step 2: |⌊x + 1/2⌋ - ⌊y + 1/2⌋| ≤ 1
  have hfloor := floor_close (x + 1/2) (y + 1/2) hshift
  -- Step 3: clamping does not increase the distance
  have hclamp := clamp_nonexpansive ⌊x + 1/2⌋ ⌊y + 1/2⌋ (-128) 127 (by norm_num)
  -- Step 4: combine: |clamp(⌊x+½⌋) - clamp(⌊y+½⌋)| ≤ |⌊x+½⌋ - ⌊y+½⌋| ≤ 1
  have hle : |max (-128) (min 127 ⌊x + 1/2⌋) - max (-128) (min 127 ⌊y + 1/2⌋)| ≤ 1 :=
    le_trans hclamp hfloor
  -- Step 5: |z| ≤ 1 → z.natAbs ≤ 1
  rw [abs_le] at hle
  omega

/-! ## Witness: far-apart inputs can differ by more than 1 -/

/-- Helper: `⌊(1 : ℝ) / 2⌋ = 0`. -/
private theorem floor_half : ⌊(1 : ℝ) / 2⌋ = 0 := by
  rw [Int.floor_eq_iff]
  constructor <;> norm_num

/-- Helper: `⌊(3 : ℝ)⌋ = 3`. -/
private theorem floor_three : ⌊(3 : ℝ)⌋ = 3 := by
  rw [Int.floor_eq_iff]
  constructor <;> norm_num

/-- Inputs separated by 2 or more can produce outputs differing by more than 1.
Witnesses: `x = 0`, `y = 5/2` give `quantizeReal 0 = 0` and
`quantizeReal (5/2) = 3`, so the difference is 3 > 1. -/
theorem requant_far_can_differ :
    ∃ x y : ℝ, 2 ≤ |x - y| ∧ 1 < (quantizeReal x - quantizeReal y).natAbs := by
  refine ⟨0, 5/2, ?_, ?_⟩
  · -- |0 - 5/2| = 5/2 ≥ 2
    norm_num
  · -- quantizeReal 0 = max(-128, min(127, ⌊0 + 1/2⌋)) = max(-128, min(127, 0)) = 0
    -- quantizeReal (5/2) = max(-128, min(127, ⌊5/2 + 1/2⌋)) = max(-128, min(127, 3)) = 3
    unfold quantizeReal
    have h1 : (0 : ℝ) + 1 / 2 = 1 / 2 := by ring
    have h2 : (5 : ℝ) / 2 + 1 / 2 = 3 := by ring
    rw [h1, h2]
    have : ⌊(1 : ℝ) / 2⌋ = 0 := floor_half
    have : ⌊(3 : ℝ)⌋ = 3 := floor_three
    simp_all
    -- After simplification, we need 1 < |0 - 3|.natAbs = 3
    norm_num

/-! ## Error Chain Parameters -/

/-- GPU arithmetic format (determines unit roundoff). -/
inductive GpuArithmetic
  | fp16  -- u = 2^-11 ≈ 4.88e-4
  | bf16  -- u = 2^-8 ≈ 3.91e-3

/-- Unit roundoff for each format. -/
noncomputable def GpuArithmetic.unitRoundoff : GpuArithmetic → ℝ
  | .fp16 => (2 : ℝ)⁻¹ ^ 11
  | .bf16 => (2 : ℝ)⁻¹ ^ 8

/-- What intermediates the GPU commits (determines which error terms vanish). -/
inductive CommittedLevel
  | qkvAccOnly       -- Only INT32 accumulators; Q,K,V all have dequant+RoPE error
  | committedKV      -- K,V committed; only Q has error
  | committedQKV     -- Q,K,V committed; only FP32 accumulation error remains
  | committedScores  -- Pre-softmax scores committed; only V-dequant error remains

/-- Parameters for the worst-case corridor bound. -/
structure CorridorParams where
  gpu : GpuArithmetic
  cRope : ℝ       -- ops in dequant+RoPE (typically 5)
  sMax : ℝ        -- max pre-softmax score magnitude
  bvOverScaleA : ℝ -- B_v / scale_a ratio (worst case 127)
  seqLen : ℕ      -- sequence length

/-- Per-Q/K-element error from dequant+RoPE: ε_qk = C_rope · u -/
noncomputable def epsQK (p : CorridorParams) : ℝ :=
  p.cRope * p.gpu.unitRoundoff

/-- The dominant error term: softmax amplification of input perturbation.
    From bounds.rs: bv_over_scale_a * multiplier * eps_qk * s_max -/
noncomputable def softmaxTerm (p : CorridorParams) (c : CommittedLevel) : ℝ :=
  match c with
  | .qkvAccOnly   => p.bvOverScaleA * 4 * (epsQK p) * p.sMax
  | .committedKV  => p.bvOverScaleA * 2 * (epsQK p) * p.sMax
  | .committedQKV => p.bvOverScaleA * 2 * p.seqLen * (2 : ℝ)⁻¹ ^ 24
  | .committedScores => 0

/-- V-dequantization error term.
    V has dequant error (no RoPE) when not committed.
    Committed K,V eliminates V error. Committed QKV eliminates V error.
    Committed scores does NOT eliminate V error — scores are committed but
    V still has dequant perturbation. Matches bounds.rs. -/
noncomputable def vDequantTerm (p : CorridorParams) (c : CommittedLevel) : ℝ :=
  match c with
  | .committedKV | .committedQKV => 0  -- V is committed, no error
  | .qkvAccOnly | .committedScores => p.bvOverScaleA * p.gpu.unitRoundoff

/-- Total pre-quantization error bound: |Δo_m| / scale_a. -/
noncomputable def corridorBound (p : CorridorParams) (c : CommittedLevel) : ℝ :=
  softmaxTerm p c + vDequantTerm p c

/-! ## Sufficient Condition for L-inf ≤ 1 -/

/-- If the total error bound is less than 1, requantized outputs differ by at most 1.
    This composes the error chain with the requantization step theorem. -/
theorem corridor_leq_one_of_bound_lt_one
    (p : CorridorParams) (c : CommittedLevel)
    (hBound : corridorBound p c < 1)
    (o_gpu o_verifier : ℝ)
    (ho : |o_gpu - o_verifier| ≤ corridorBound p c) :
    (quantizeReal o_gpu - quantizeReal o_verifier).natAbs ≤ 1 :=
  requant_close_implies_linf_one o_gpu o_verifier (by linarith)

/-- Committed scores achieve ≤ 1 for FP16 with B_v/scale_a ≤ 127.
    The bound is 127 · 2^-11 ≈ 0.062 < 1. -/
theorem committedScores_achieves_leq_one
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hbv : p.bvOverScaleA ≤ 127) :
    corridorBound p .committedScores < 1 := by
  unfold corridorBound softmaxTerm vDequantTerm epsQK GpuArithmetic.unitRoundoff
  rw [hgpu]
  simp
  -- Need: bvOverScaleA * (2⁻¹ ^ 11) < 1
  -- Since bvOverScaleA ≤ 127 and 2⁻¹^11 = 1/2048:
  -- 127/2048 ≈ 0.062 < 1
  nlinarith

/-! ## Impossibility: QKV Accumulators Alone -/

/-- From QKV accumulators alone, the corridor exceeds 1 whenever S_max ≥ 1 (FP16). -/
theorem qkvOnly_exceeds_one
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hcr : p.cRope = 5)
    (hbv : 127 ≤ p.bvOverScaleA)
    (hs : 1 ≤ p.sMax) :
    1 ≤ corridorBound p .qkvAccOnly := by
  unfold corridorBound softmaxTerm vDequantTerm epsQK GpuArithmetic.unitRoundoff
  rw [hgpu, hcr]
  nlinarith [p.bvOverScaleA, p.sMax]

end VerifiedInference
