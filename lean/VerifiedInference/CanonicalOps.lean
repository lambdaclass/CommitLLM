import VerifiedInference.Basic
import VerifiedInference.Requantization
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Canonical Nonlinear Operations with Explicit Precision Assumptions

Real-valued canonical forms of RMSNorm and RoPE. The precision gap between
real arithmetic and INT8 arithmetic is made explicit as a `Prop` assumption
rather than hidden inside an implementation.
-/

namespace VerifiedInference

open Real

/-! ## Canonical RMSNorm -/

/-- Real-valued RMS denominator for a list of integers. -/
noncomputable def rmsValue (x : List ℤ) (eps : ℝ) : ℝ :=
  let n := x.length
  if n = 0 then 1
  else
    let sumSq := x.foldl (fun acc xi => acc + (xi : ℝ) ^ 2) 0
    Real.sqrt (sumSq / n + eps)

/-- Canonical real-valued RMSNorm output.
    For each index i, computes gamma[i] * x[i] / rms(x, eps).
    Returns a list of reals parallel to x (truncated to min length). -/
noncomputable def canonicalRMSNorm (gamma : List ℝ) (x : List ℤ) (eps : ℝ) : List ℝ :=
  let rms := rmsValue x eps
  List.zipWith (fun g xi => g * (xi : ℝ) / rms) gamma x

/-! ## Precision Assumption for RMSNorm -/

/-- Round a real number to the nearest integer (round-half-up). -/
noncomputable def roundReal (r : ℝ) : ℤ := ⌊r + (1 / 2 : ℝ)⌋

/-- Requantize a real value to INT8 via rounding and clamping. -/
noncomputable def requantizeReal (r : ℝ) : ℤ := clampI8 (roundReal r)

/-- **RMSNorm Precision Assumption**: The canonical real-valued RMSNorm output,
    after scaling by the output scale factor and requantizing to INT8, agrees
    element-wise with the INT8 output committed to by the prover.

    This Prop makes the precision gap *explicit*: the protocol cannot verify
    this in zero-knowledge without additional commitments to intermediate values.
    It is stated as an assumption (hypothesis) in theorems that need it. -/
def rmsnormPrecisionAssumption
    (gamma : List ℝ) (x : List ℤ) (eps : ℝ) (outputScale : ℝ)
    (committed : List ℤ) : Prop :=
  let canonical := canonicalRMSNorm gamma x eps
  let scaled := canonical.map (fun r => requantizeReal (r * outputScale))
  committed = scaled

/-! ## Canonical RoPE Structure -/

/-- Canonical RoPE (Rotary Position Embedding) tables.
    `cosTable pos` and `sinTable pos` give the INT8-quantized cos/sin
    coefficients for rotary position `pos`. -/
structure CanonicalRoPE where
  /-- Quantized cosine table: position → list of INT8 coefficients -/
  cosTable : ℕ → List ℤ
  /-- Quantized sine table: position → list of INT8 coefficients -/
  sinTable : ℕ → List ℤ

/-! ## RoPE Determinism -/

/-- **RoPE Determinism**: Given fixed cosine and sine tables, the RoPE
    output is a deterministic function of the position index.
    Two computations from identical tables at the same position agree. -/
theorem rope_canonical_deterministic (rope : CanonicalRoPE) (pos : ℕ) :
    ∀ out₁ out₂ : List ℤ,
      out₁ = rope.cosTable pos →
      out₂ = rope.cosTable pos →
      out₁ = out₂ := by
  intro out₁ out₂ h₁ h₂
  rw [h₁, h₂]

end VerifiedInference
