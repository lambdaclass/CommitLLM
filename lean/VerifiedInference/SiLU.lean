import VerifiedInference.Basic
import VerifiedInference.Requantization

/-!
# SiLU Determinism

**Statement**: The INT8 SiLU lookup table is a total function on Fin 256,
hence deterministic. Given the same INT8 input, the LUT always returns
the same output.

The paper Table 1 says SiLU verification is "Exact". The code's
`compute_h_unit_scale` (silu.rs:35) at unit scale is indeed exact:
it's a composition of pure functions (clamp, LUT lookup, multiply, clamp).

The scaled f32 path (check_silu, silu.rs:57) allows +-1 tolerance,
but we formalize only the exact INT8 path.

Mirrors: `crates/vi-core/src/silu.rs` — `compute_h_unit_scale()`
-/

namespace VerifiedInference

/-! ## SiLU LUT as a total function -/

/-- A SiLU lookup table: maps each of the 256 possible INT8 values
    to a fixed integer result. This is a total function, hence deterministic. -/
def siluLUT : Fin 256 → ℤ := fun _ => 0  -- placeholder; actual values are implementation-defined

/-- The INT8 SiLU+multiply computation at unit scale:
    h[i] = clamp(round(SiLU(clamp(g[i])) * clamp(u[i])))

    Each step is a total function, so the composition is total and deterministic. -/
def computeH (lut : Fin 256 → ℤ) (g u : ℤ) : ℤ :=
  let g_i8 := clampI8 g
  let u_i8 := clampI8 u
  -- Index into LUT: map [-128, 127] → [0, 255]
  have hg := clampI8_range g
  let g_idx : Fin 256 := ⟨(g_i8 + 128).toNat, by omega⟩
  let silu_g := lut g_idx
  clampI8 (silu_g * u_i8)

/-- **SiLU Determinism (trivial)**: If two values both equal computeH(lut,g,u),
    they equal each other. This is transitivity of equality, NOT a security
    statement. The actual security value comes from siluLayerValid (Protocol.lean)
    which CHECKS that the prover's claimed h matches computeH. -/
theorem silu_deterministic (lut : Fin 256 → ℤ) (g u : ℤ) :
    ∀ h₁ h₂ : ℤ, h₁ = computeH lut g u → h₂ = computeH lut g u → h₁ = h₂ := by
  intro h₁ h₂ e₁ e₂
  rw [e₁, e₂]

/-- **SiLU with ±1 tolerance**: The Rust verifier allows rounding slack.
    This matches `check_silu` in silu.rs which checks |h[i] - expected| ≤ 1.
    The tolerance accounts for f32 rounding in the scaled path. -/
def computeH_tolerant (lut : Fin 256 → ℤ) (g u : ℤ) (claimed : ℤ) : Prop :=
  let expected := computeH lut g u
  (claimed - expected).natAbs ≤ 1

/-- **Tolerant SiLU Determinism**: With ±1 tolerance, at most 3 values
    are accepted for each (g, u) pair: expected-1, expected, expected+1. -/
theorem silu_tolerant_bounded (lut : Fin 256 → ℤ) (g u : ℤ) (h₁ h₂ : ℤ)
    (ht₁ : computeH_tolerant lut g u h₁)
    (ht₂ : computeH_tolerant lut g u h₂) :
    (h₁ - h₂).natAbs ≤ 2 := by
  unfold computeH_tolerant at ht₁ ht₂
  omega

/-- **SiLU Vector Determinism**: For vectors, element-wise SiLU is deterministic. -/
theorem silu_vector_deterministic (lut : Fin 256 → ℤ) {dim : ℕ}
    (g u : Fin dim → ℤ) :
    ∀ h₁ h₂ : Fin dim → ℤ,
      (∀ i, h₁ i = computeH lut (g i) (u i)) →
      (∀ i, h₂ i = computeH lut (g i) (u i)) →
      h₁ = h₂ := by
  intro h₁ h₂ e₁ e₂
  funext i
  rw [e₁ i, e₂ i]

/-- The composition of deterministic steps is deterministic. This captures
    the pipeline: requantize → LUT lookup → multiply → requantize. -/
theorem silu_pipeline_deterministic {dim : ℕ}
    (lut : Fin 256 → ℤ) (g_acc u_acc : Fin dim → ℤ) :
    ∃! h : Fin dim → ℤ, ∀ i, h i = computeH lut (g_acc i) (u_acc i) :=
  ⟨fun i => computeH lut (g_acc i) (u_acc i),
   fun _ => rfl,
   fun h hspec => funext fun i => by rw [hspec i]⟩

end VerifiedInference
