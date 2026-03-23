import VerifiedInference.Basic

/-!
# Theorem 3: Layer Determinism

**Statement**: A deterministic function has a unique output for each input.

This is mathematically trivial (functions have unique outputs by definition),
but the formalization serves to make explicit the assumption that INT8 matmul,
SiLU lookup, and requantization are all deterministic operations.

Mirrors: The concept that `check()` in `freivalds.rs` and `compute_h_unit_scale()`
in `silu.rs` are pure functions.
-/

namespace VerifiedInference

/-- **Theorem 3 (Layer Determinism)**:

    Any function f : α → β produces a unique output for each input.
    If f(x) = y₁ and f(x) = y₂, then y₁ = y₂.

    This captures the requirement that all verification-relevant operations
    (INT8 matmul, SiLU LUT, clamp requantization) are deterministic.
-/
theorem layer_determinism {α β : Type*} (f : α → β) (x : α) (y₁ y₂ : β)
    (h₁ : f x = y₁) (h₂ : f x = y₂) : y₁ = y₂ := by
  rw [← h₁, ← h₂]

/-- Corollary for matrix-vector multiplication: W x is unique. -/
theorem matmul_deterministic {p : ℕ} [Fact (Nat.Prime p)] {m n : ℕ}
    (W : Matrix (Fin m) (Fin n) (ZMod p))
    (x : Fin n → ZMod p) :
    ∀ z₁ z₂ : Fin m → ZMod p,
      z₁ = W.mulVec x → z₂ = W.mulVec x → z₁ = z₂ :=
  fun _ _ h₁ h₂ => by rw [h₁, h₂]

end VerifiedInference
