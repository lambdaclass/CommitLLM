import VerifiedInference.Basic

/-!
# Cross-Layer Algebraic Constraints on Fake Attention

This module formalizes the algebraic constraint that links a layer's
`Wo * a` accumulator, the residual stream, and the next layer's input
via INT8 requantization (`clampI8`).
-/

namespace VerifiedInference

/-! ## Single-Layer Constraint -/

/-- The cross-layer constraint: the next layer's input at each position equals
    `clampI8 (wo_times_a i + residual i)`. -/
def crossLayerConstraint {hiddenDim : ℕ}
    (wo_times_a residual nextInput : Fin hiddenDim → ℤ) : Prop :=
  ∀ i, nextInput i = clampI8 (wo_times_a i + residual i)

/-! ## Multi-Layer Constraint -/

/-- For `nOpened` consecutive layers each layer's output must satisfy
    the cross-layer constraint with the next layer's input.

    `woa layers.(i)`, `residual.(i)`, and `inputs.(i+1)` are linked by
    `crossLayerConstraint` for every `i < nOpened`. -/
def multiLayerConstraint (nOpened hiddenDim : ℕ)
    (woa : Fin nOpened → Fin hiddenDim → ℤ)
    (residual : Fin nOpened → Fin hiddenDim → ℤ)
    (inputs : Fin (nOpened + 1) → Fin hiddenDim → ℤ) : Prop :=
  ∀ (i : Fin nOpened),
    crossLayerConstraint (woa i) (residual i)
      (inputs ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩)

/-! ## Uniqueness Theorem -/

/-- If two different `wo_times_a` values both satisfy the same cross-layer
    constraint (same `residual`, same `nextInput`), then their clamped sums
    with the residual are equal at every position. -/
theorem cross_layer_pins_requantized_output {hiddenDim : ℕ}
    (wo_a₁ wo_a₂ residual nextInput : Fin hiddenDim → ℤ)
    (h₁ : crossLayerConstraint wo_a₁ residual nextInput)
    (h₂ : crossLayerConstraint wo_a₂ residual nextInput) :
    ∀ i, clampI8 (wo_a₁ i + residual i) = clampI8 (wo_a₂ i + residual i) := by
  intro i
  -- Both equal nextInput i, so they equal each other.
  exact (h₁ i).symm.trans (h₂ i)

end VerifiedInference
