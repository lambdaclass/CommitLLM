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
private theorem clamp_nonexpansive (a b lo hi : ℤ) (hlohi : lo ≤ hi) :
    |max lo (min hi a) - max lo (min hi b)| ≤ |a - b| := by
  -- Clamp is 1-Lipschitz. We prove it by showing the clamped difference
  -- is between -(|a-b|) and |a-b|, i.e., |clamp(a) - clamp(b)| ≤ |a-b|.
  -- Direct omega on integers after converting abs to ± form.
  simp only [max_def, min_def]
  -- Clamp is 1-Lipschitz: |clamp(a) - clamp(b)| ≤ |a - b|.
  -- Proof: 9-way case split on regions of a and b relative to [lo, hi].
  sorry
  -- TODO: Close this sorry. The proof strategy is correct (split_ifs on
  -- max_def/min_def, then linarith with abs_nonneg/le_abs_self).
  -- One branch needs nlinarith or manual manipulation to close.
  -- The main theorem (requant_close_implies_linf_one) depends on this.

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

end VerifiedInference
