import VerifiedInference.Basic

/-!
# Requantization Determinism

**Statement**: The clamp function clamp(z, -128, 127) is a total function
on integers, hence deterministic.

This captures the INT8 requantization step that converts i32 accumulators
back to INT8 between layers.

Mirrors: `crates/vi-core/src/types.rs` — `CompactLayerTrace::to_full()`
         where `v.clamp(-128, 127) as i8`
-/

namespace VerifiedInference

/-! ## Clamp is a total function -/

/-- **Requantization Determinism (trivial)**: If two values both equal
    clampI8(z), they equal each other. This is transitivity of equality.
    The actual security value comes from intraLayerChainValid (Protocol.lean)
    which CHECKS that the prover's xFfn matches clamp(attnOut). -/
theorem requantization_deterministic (z : ℤ) :
    ∀ y₁ y₂ : ℤ, y₁ = clampI8 z → y₂ = clampI8 z → y₁ = y₂ := by
  intro y₁ y₂ h₁ h₂
  rw [h₁, h₂]

/-- **Clamp range**: the output of clampI8 is always in [-128, 127]. -/
theorem clampI8_range (z : ℤ) : -128 ≤ clampI8 z ∧ clampI8 z ≤ 127 := by
  unfold clampI8
  constructor
  · exact le_max_left _ _
  · exact max_le (by omega) (min_le_left _ _)

/-- **Clamp idempotence**: clamping an already-clamped value is a no-op. -/
theorem clampI8_idempotent (z : ℤ) : clampI8 (clampI8 z) = clampI8 z := by
  have ⟨hlo, hhi⟩ := clampI8_range z
  unfold clampI8
  omega

/-- Vector requantization is deterministic. -/
theorem requantization_vector_deterministic {dim : ℕ} (z : Fin dim → ℤ) :
    ∃! y : Fin dim → ℤ, ∀ i, y i = clampI8 (z i) :=
  ⟨fun i => clampI8 (z i),
   fun _ => rfl,
   fun y hy => funext fun i => by rw [hy i]⟩

/-- **Chain check determinism**: if output of layer i is determined,
    then input to layer i+1 (= clamp(output)) is determined. -/
theorem chain_deterministic {dim : ℕ}
    (output₁ output₂ : Fin dim → ℤ)
    (hEq : output₁ = output₂) :
    (fun i => clampI8 (output₁ i)) = (fun i => clampI8 (output₂ i)) := by
  rw [hEq]

end VerifiedInference
