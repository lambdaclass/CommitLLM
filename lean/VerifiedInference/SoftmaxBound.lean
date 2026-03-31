import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
# Softmax Jacobian L1 Bound

This module formalizes the softmax Jacobian ∞→1 norm bound used in Step 3 of
the attention error chain in the VeriLM protocol.

The softmax Jacobian `J` satisfies `J_{jk} = α_j (δ_{jk} - α_k)` for a
probability vector `α`. When a score vector is perturbed by `Δs`, the
first-order change in the attention weights is `Δα = J · Δs`, whose
components are `Δα_j = α_j · (Δs_j - Σ_k α_k · Δs_k)`.

The main result is:

  ‖Δα‖₁ ≤ 2 · ‖Δs‖_∞

This bound is classical numerical analysis (see, e.g., the softmax Lipschitz
analysis in Gao & Pavel 2017) and is the bottleneck in the worst-case
corridor derivation: it converts an L∞ score perturbation (from quantization
or Freivalds residuals) into an L1 attention-weight perturbation that
propagates through the value projection.
-/

namespace VerifiedInference

/-- First-order softmax perturbation model.

Given a probability vector `α` (non-negative, sums to 1) and a score
perturbation `Δs`, the softmax Jacobian gives the attention-weight
perturbation as `Δα_j = α_j · (Δs_j - Σ_k α_k · Δs_k)`. -/
noncomputable def softmaxPerturbFirstOrder {n : ℕ}
    (α : Fin n → ℝ) (Δs : Fin n → ℝ) : Fin n → ℝ :=
  let meanΔs := ∑ k : Fin n, α k * Δs k
  fun j => α j * (Δs j - meanΔs)

/-- The weighted mean of a bounded function under a probability distribution
is bounded by the sup norm of the function. -/
private theorem abs_weighted_mean_le {n : ℕ}
    (α : Fin n → ℝ) (Δs : Fin n → ℝ)
    (hα_nn : ∀ i, 0 ≤ α i) (hα_sum : ∑ i, α i = 1)
    (maxΔs : ℝ) (hmaxΔs : ∀ i, |Δs i| ≤ maxΔs) :
    |∑ k : Fin n, α k * Δs k| ≤ maxΔs := by
  calc |∑ k : Fin n, α k * Δs k|
      ≤ ∑ k : Fin n, |α k * Δs k| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ k : Fin n, α k * |Δs k| := by
        congr 1; ext k
        rw [abs_mul, abs_of_nonneg (hα_nn k)]
    _ ≤ ∑ k : Fin n, α k * maxΔs := by
        apply Finset.sum_le_sum
        intro k _
        exact mul_le_mul_of_nonneg_left (hmaxΔs k) (hα_nn k)
    _ = maxΔs * ∑ k : Fin n, α k := by
        trans ((∑ k : Fin n, α k) * maxΔs)
        · exact Finset.sum_mul Finset.univ (fun k => α k) maxΔs |>.symm ▸
            Finset.sum_congr rfl (fun k _ => by ring)
        · ring
    _ = maxΔs := by rw [hα_sum, mul_one]

/-- Per-element bound on the softmax perturbation: each component satisfies
`|Δα_j| ≤ 2 · α_j · maxΔs`. -/
theorem softmax_perturb_elem_bound {n : ℕ}
    (α : Fin n → ℝ) (Δs : Fin n → ℝ)
    (hα_nn : ∀ i, 0 ≤ α i) (hα_sum : ∑ i, α i = 1)
    (maxΔs : ℝ) (hmaxΔs : ∀ i, |Δs i| ≤ maxΔs)
    (j : Fin n) :
    |softmaxPerturbFirstOrder α Δs j| ≤ 2 * α j * maxΔs := by
  unfold softmaxPerturbFirstOrder
  show |α j * (Δs j - ∑ k : Fin n, α k * Δs k)| ≤ 2 * α j * maxΔs
  -- |α_j · (Δs_j - mean)| = α_j · |Δs_j - mean| since α_j ≥ 0
  rw [abs_mul, abs_of_nonneg (hα_nn j)]
  -- Suffices to show |Δs_j - mean| ≤ 2 * maxΔs
  have hmean := abs_weighted_mean_le α Δs hα_nn hα_sum maxΔs hmaxΔs
  have hΔs_j := hmaxΔs j
  have hdev : |Δs j - ∑ k : Fin n, α k * Δs k| ≤ 2 * maxΔs := by
    have htri : |Δs j - ∑ k : Fin n, α k * Δs k|
        ≤ |Δs j| + |∑ k : Fin n, α k * Δs k| := by
      calc |Δs j - ∑ k : Fin n, α k * Δs k|
          = |Δs j + (-(∑ k : Fin n, α k * Δs k))| := by ring_nf
        _ ≤ |Δs j| + |-(∑ k : Fin n, α k * Δs k)| := abs_add_le _ _
        _ = |Δs j| + |∑ k : Fin n, α k * Δs k| := by rw [abs_neg]
    linarith
  calc α j * |Δs j - ∑ k : Fin n, α k * Δs k|
      ≤ α j * (2 * maxΔs) := by
        exact mul_le_mul_of_nonneg_left hdev (hα_nn j)
    _ = 2 * α j * maxΔs := by ring

/-- **Softmax Jacobian L1 bound.** The L1 norm of the softmax perturbation
is at most 2 times the L∞ norm of the score perturbation:

  `Σ_j |Δα_j| ≤ 2 · maxΔs`

where `maxΔs ≥ |Δs_j|` for all `j`. This is the ∞→1 operator norm of
the softmax Jacobian, and is tight when all probability mass is on a
single coordinate. -/
theorem softmax_jacobian_l1_bound {n : ℕ}
    (α : Fin n → ℝ) (Δs : Fin n → ℝ)
    (hα_nn : ∀ i, 0 ≤ α i) (hα_sum : ∑ i, α i = 1)
    (maxΔs : ℝ) (hmaxΔs : ∀ i, |Δs i| ≤ maxΔs) :
    ∑ j, |softmaxPerturbFirstOrder α Δs j| ≤ 2 * maxΔs := by
  -- Lift per-element bound
  have helem : ∀ j ∈ Finset.univ, |softmaxPerturbFirstOrder α Δs j| ≤ 2 * α j * maxΔs :=
    fun j _ => softmax_perturb_elem_bound α Δs hα_nn hα_sum maxΔs hmaxΔs j
  calc ∑ j, |softmaxPerturbFirstOrder α Δs j|
      ≤ ∑ j, 2 * α j * maxΔs := Finset.sum_le_sum helem
    _ = 2 * maxΔs * ∑ j, α j := by
        trans (∑ j : Fin n, (2 * maxΔs) * α j)
        · exact Finset.sum_congr rfl (fun j _ => by ring)
        · exact (Finset.mul_sum Finset.univ (fun j => α j) (2 * maxΔs)).symm
    _ = 2 * maxΔs := by rw [hα_sum, mul_one]

end VerifiedInference
