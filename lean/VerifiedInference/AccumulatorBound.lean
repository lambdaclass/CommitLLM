import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import VerifiedInference.Basic
import VerifiedInference.Requantization

/-!
# Accumulator Overflow Bounds

Production uses i32 accumulators for INT8×INT8 matmuls. This module proves
that accumulator values are bounded, ensuring they fit in i32 and the
formalization's unbounded ℤ faithfully represents the Rust implementation.

For input dimension n with INT8 inputs (range [-128, 127]) and INT8 weights
(range [-128, 127]):
  |acc[j]| = |Σᵢ W[j,i] · x[i]| ≤ n · 128² = n · 16384

For Llama-8B (n=14336, the largest dim): max = 234,881,024 < 2^31 - 1.

Mirrors: Rust `i32` accumulator type in `types.rs`
-/

namespace VerifiedInference

/-- A single INT8×INT8 product is bounded by 128² = 16384. -/
theorem int8_mul_bound (a b : ℤ) (ha : isInt8 a) (hb : isInt8 b) :
    -16384 ≤ a * b ∧ a * b ≤ 16384 := by
  unfold isInt8 at ha hb
  constructor <;> nlinarith [ha.1, ha.2, hb.1, hb.2]

/-- The accumulator for an INT8 dot product of dimension n is bounded
    by n × 16384. If n × 16384 < 2³¹, the accumulator fits in i32. -/
theorem accumulator_bound_sufficient (n : ℕ) (hn : n * 16384 < 2147483648) :
    ∀ (w x : Fin n → ℤ),
      (∀ i, isInt8 (w i)) →
      (∀ i, isInt8 (x i)) →
      isInt32 (∑ i : Fin n, w i * x i) := by
  intro w x hw hx
  unfold isInt32
  have hbound : ∀ i : Fin n, -16384 ≤ w i * x i ∧ w i * x i ≤ 16384 :=
    fun i => int8_mul_bound (w i) (x i) (hw i) (hx i)
  constructor
  · calc -(2147483648 : ℤ) ≤ -(n * 16384 : ℤ) := by omega
      _ = ∑ _i : Fin n, (-16384 : ℤ) := by simp [Finset.sum_const]
      _ ≤ ∑ i : Fin n, w i * x i := Finset.sum_le_sum fun i _ => (hbound i).1
  · calc ∑ i : Fin n, w i * x i
        ≤ ∑ _i : Fin n, (16384 : ℤ) := Finset.sum_le_sum fun i _ => (hbound i).2
      _ = (n * 16384 : ℤ) := by simp [Finset.sum_const]
      _ ≤ 2147483647 := by omega

/-- For Llama-8B, the largest input dimension is ffn_dim = 14336. -/
example : 14336 * 16384 < 2147483648 := by native_decide

/-- For Llama-405B, the largest input dimension is ffn_dim = 53248. -/
example : 53248 * 16384 < 2147483648 := by native_decide

/-- clampI8 produces INT8 values. -/
theorem clampI8_isInt8 (z : ℤ) : isInt8 (clampI8 z) := by
  unfold isInt8 clampI8
  constructor <;> omega

end VerifiedInference
