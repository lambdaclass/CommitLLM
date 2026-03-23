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

/-! ## Adversarial Counting -/

/-- The cross-layer constraint uniquely determines `nextInput` from `wo_times_a` and `residual`.
    Security implication: an adversary who observes the residual stream and the next-layer input
    has no freedom — the requantized value is fully pinned and cannot be manipulated by choosing
    a different `nextInput` while keeping the same `wo_times_a` and `residual`. -/
theorem cross_layer_determines_next_input {hiddenDim : ℕ}
    (wo_times_a residual : Fin hiddenDim → ℤ) :
    ∃! nextInput, crossLayerConstraint wo_times_a residual nextInput :=
  ⟨fun i => clampI8 (wo_times_a i + residual i),
   fun _ => rfl,
   fun next hNext => funext fun i => hNext i⟩

/-- In the interior of the INT8 range (strictly between -128 and 127), two pre-activations that
    produce the same clamped output must have identical sums with the residual.
    Security implication: when the output is not at a saturation boundary, the adversary cannot
    find two distinct `wo_times_a` values that collide after INT8 clamping — the preimage
    width is exactly one, giving no adversarial slack to forge activations. -/
theorem cross_layer_preimage_width {hiddenDim : ℕ}
    (residual nextInput : Fin hiddenDim → ℤ)
    (i : Fin hiddenDim)
    (hInterior : -128 < nextInput i ∧ nextInput i < 127) :
    ∀ a₁ a₂ : ℤ,
      clampI8 (a₁ + residual i) = nextInput i →
      clampI8 (a₂ + residual i) = nextInput i →
      a₁ + residual i = a₂ + residual i := by
  intro a₁ a₂ h₁ h₂
  unfold clampI8 at h₁ h₂
  omega

end VerifiedInference
