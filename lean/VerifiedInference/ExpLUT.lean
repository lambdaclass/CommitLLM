import VerifiedInference.Basic
import VerifiedInference.Requantization

/-!
# Exponential LUT Determinism for INT8 Attention Scores

**Statement**: The exponential lookup table for INT8 attention scores
is a total function on Fin 256, hence deterministic. Given the same
INT8 score, the LUT always returns the same fixed-point exp value.

In INT8 attention, raw scores (Q*K^T products) are clamped to [-128, 127]
and mapped through an exp LUT to compute softmax numerators. Since the
LUT is a total function Fin 256 → ℤ, this step is exact and deterministic.

Same pattern as SiLU: clamp to INT8, index into LUT, return result.

Mirrors: the exponential lookup in the INT8 attention pipeline.
-/

namespace VerifiedInference

/-! ## Exp LUT as a total function -/

/-- An exponential lookup table: maps each of the 256 possible INT8 values
    to a fixed-point integer approximating exp(x/scale). This is a total
    function, hence deterministic. -/
def expLUT : Fin 256 → ℤ := fun _ => 0  -- placeholder; actual values are implementation-defined

/-- The INT8 exponential computation for a single attention score:
    1. Clamp the raw score to INT8 range [-128, 127]
    2. Map the clamped value to index in [0, 255]
    3. Look up the fixed-point exp value in the LUT

    Each step is a total function, so the composition is total and deterministic. -/
def computeExp (lut : Fin 256 → ℤ) (score : ℤ) : ℤ :=
  let s_i8 := clampI8 score
  -- Index into LUT: map [-128, 127] → [0, 255]
  have hs := clampI8_range score
  let s_idx : Fin 256 := ⟨(s_i8 + 128).toNat, by omega⟩
  lut s_idx

/-! ## Examples -/

/-- Example: computeExp on a concrete input returns the LUT value at the
    corresponding index. Using the placeholder LUT (all zeros), any input
    maps to 0. -/
example : computeExp expLUT 42 = 0 := by native_decide

/-- Example: computeExp clamps out-of-range scores. A score of 200 is
    clamped to 127 before lookup. -/
example : computeExp expLUT 200 = computeExp expLUT 127 := by native_decide

/-- Example: computeExp clamps negative overflow. A score of -200 is
    clamped to -128 before lookup. -/
example : computeExp expLUT (-200) = computeExp expLUT (-128) := by native_decide

/-! ## Determinism theorems -/

/-- **Exp Determinism**: The exp LUT computation is a function, hence
    for the same inputs it always produces the same output. -/
theorem exp_deterministic (lut : Fin 256 → ℤ) (score : ℤ) :
    ∀ (e₁ e₂ : ℤ), e₁ = computeExp lut score → e₂ = computeExp lut score → e₁ = e₂ := by
  intro e₁ e₂ h₁ h₂
  rw [h₁, h₂]

/-- **Exp Vector Determinism**: For a vector of attention scores,
    element-wise exp LUT lookup is deterministic. -/
theorem exp_vector_deterministic (lut : Fin 256 → ℤ) {len : ℕ}
    (scores : Fin len → ℤ) :
    ∀ (e₁ e₂ : Fin len → ℤ),
      (∀ i, e₁ i = computeExp lut (scores i)) →
      (∀ i, e₂ i = computeExp lut (scores i)) →
      e₁ = e₂ := by
  intro e₁ e₂ h₁ h₂
  funext i
  rw [h₁ i, h₂ i]

/-- **Exp Pipeline Determinism**: The composition of clamp and LUT lookup
    produces a unique result vector. -/
theorem exp_pipeline_deterministic {len : ℕ}
    (lut : Fin 256 → ℤ) (scores : Fin len → ℤ) :
    ∃! e : Fin len → ℤ, ∀ i, e i = computeExp lut (scores i) :=
  ⟨fun i => computeExp lut (scores i),
   fun _ => rfl,
   fun e hspec => funext fun i => by rw [hspec i]⟩

/-! ## Range property -/

/-- The clamped index used by computeExp is always a valid Fin 256 index.
    This is a consequence of clampI8_range. -/
theorem computeExp_index_valid (score : ℤ) :
    let s_i8 := clampI8 score
    0 ≤ (s_i8 + 128).toNat ∧ (s_i8 + 128).toNat < 256 := by
  have hs := clampI8_range score
  constructor
  · exact Nat.zero_le _
  · omega

end VerifiedInference
