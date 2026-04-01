import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.BigOperators

import VerifiedInference.Basic
import VerifiedInference.PrecomputedFreivalds

/-!
# Theorem 1: Freivalds Correctness (Schwartz-Zippel)

**Statement**: If AB ≠ C, then Pr_r[r^T(AB - C) = 0] ≤ 1/p.

We use the direct counting approach:
1. Let d = Wx - z ≠ 0 (since z ≠ Wx).
2. r · d is a nonzero linear form in r.
3. Fix all r_i for i ≠ j where d_j ≠ 0. Then r_j is uniquely determined.
4. So |kernel| = p^{m-1}, ratio = 1/p.

Stated combinatorially: |{r ∈ (ZMod p)^m : r · d = 0}| * p ≤ p^m.

Mirrors: `crates/vi-core/src/freivalds.rs` — `check()` soundness
-/

namespace VerifiedInference

variable {p : ℕ} [hp : Fact (Nat.Prime p)]
variable {m n : ℕ}

/-! ## The kernel of a nonzero linear form over ZMod p -/

/-- A linear form over (ZMod p)^m defined by a coefficient vector d. -/
def linearForm (d : Fin m → ZMod p) (r : Fin m → ZMod p) : ZMod p :=
  dotProduct r d

/-! ### Helper lemmas for the kernel bound proof -/

/-- For two vectors in the kernel of a linear form with d_j ≠ 0,
    if they agree on all coordinates except j, then they also agree on j.

    This is the key algebraic fact: in the field ZMod p, if d_j ≠ 0 and
    r . d = r' . d = 0, then (r_j - r'_j) * d_j = 0 forces r_j = r'_j. -/
private lemma kernel_agree_at_j
    (d : Fin m → ZMod p) (j : Fin m) (hdj : d j ≠ 0)
    (r r' : Fin m → ZMod p)
    (hr : dotProduct r d = 0) (hr' : dotProduct r' d = 0)
    (heq : ∀ i, i ≠ j → r i = r' i) : r j = r' j := by
  -- Unfold dotProduct to sums
  unfold dotProduct at hr hr'
  -- For i ≠ j, (r i - r' i) * d i = 0 since r i = r' i by hypothesis
  have hdiff : ∀ i : Fin m, i ≠ j → (r i - r' i) * d i = 0 := by
    intro i hne; rw [heq i hne, sub_self, zero_mul]
  -- Sum of differences is zero: both sums individually equal zero
  have hsum_diff : ∑ i : Fin m, (r i - r' i) * d i = 0 := by
    have : ∑ i, (r i - r' i) * d i = ∑ i, r i * d i - ∑ i, r' i * d i := by
      simp_rw [sub_mul]; exact Finset.sum_sub_distrib ..
    rw [this, hr, hr', sub_self]
  -- Extract single surviving term: only j-th term can be nonzero
  have h_single := Finset.sum_eq_single j
    (fun i _ hne => hdiff i hne) (fun h => absurd (Finset.mem_univ j) h)
  rw [h_single] at hsum_diff
  -- In the field ZMod p: (r j - r' j) * d j = 0 with d j ≠ 0 gives r j = r' j
  cases mul_eq_zero.mp hsum_diff with
  | inl h => exact sub_eq_zero.mp h
  | inr h => exact absurd h hdj

/-- Dot product distributes over vector subtraction. -/
private lemma dotProduct_sub (r a b : Fin m → ZMod p) :
    dotProduct r (a - b) = dotProduct r a - dotProduct r b := by
  simp [dotProduct, Finset.sum_sub_distrib, mul_sub]

/-- The Freivalds accepting set equals the kernel of the error linear form.

    The condition (W^T r) . x = r . z is equivalent to r . (Wx - z) = 0,
    using the precomputed Freivalds equivalence r . (Wx) = (W^T r) . x. -/
private lemma freivalds_filter_eq_kernel (W : Matrix (Fin m) (Fin n) (ZMod p))
    (x : Fin n → ZMod p) (z : Fin m → ZMod p) :
    (Finset.univ.filter fun r : Fin m → ZMod p =>
      dotProduct (W.transpose.mulVec r) x = dotProduct r z) =
    (Finset.univ.filter fun r : Fin m → ZMod p =>
      dotProduct r (W.mulVec x - z) = 0) := by
  ext r
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  rw [dotProduct_sub, ← precomputed_equiv]
  exact ⟨fun h => sub_eq_zero.mpr h, fun h => sub_eq_zero.mp h⟩

/-- **Key Lemma**: The kernel of a nonzero linear form over (ZMod p)^m
    has cardinality at most p^{m-1}.

    Stated as: |ker| * p ≤ |total|.

    **Proof**: We construct an injection from ker x (ZMod p) into (Fin m -> ZMod p)
    via (r, c) |-> Function.update r j c, where j is a coordinate with d_j /= 0.

    Injectivity: if Function.update r j c = Function.update r' j c', then
    - c = c' (from the j-th coordinate)
    - r i = r' i for i /= j (from coordinates i /= j)
    - r j = r' j (from `kernel_agree_at_j`: the kernel constraint forces agreement at j)

    Since |ker x (ZMod p)| = |ker| * p and the target has |total| elements,
    we get |ker| * p <= |total|. -/
theorem kernel_bound_of_nonzero
    (d : Fin m → ZMod p) (hd : d ≠ 0) (_hm : 0 < m) :
    (Finset.univ.filter fun r : Fin m → ZMod p => linearForm d r = 0).card * p
      ≤ Fintype.card (Fin m → ZMod p) := by
  -- Step 1: Since d ≠ 0, pick j with d_j ≠ 0
  obtain ⟨j, hdj⟩ := Function.ne_iff.mp hd
  simp only [Pi.zero_apply] at hdj
  -- Name the kernel set
  set K := Finset.univ.filter (fun r : Fin m → ZMod p => linearForm d r = 0) with hK_def
  -- Step 2: Reduce to |K ×ˢ univ| ≤ |univ| using Finset cardinality arithmetic
  -- K.card * p = K.card * (Finset.univ : Finset (ZMod p)).card = (K ×ˢ univ).card
  suffices h : (K ×ˢ (Finset.univ : Finset (ZMod p))).card
      ≤ (Finset.univ : Finset (Fin m → ZMod p)).card by
    rwa [Finset.card_product, Finset.card_univ, ZMod.card, Finset.card_univ] at h
  -- Step 3: Construct the injection (r, c) ↦ Function.update r j c
  apply Finset.card_le_card_of_injOn
    (fun rc => Function.update rc.1 j rc.2)
  -- The image always lands in Finset.univ (trivially true)
  · intro _ _; exact Finset.mem_univ _
  -- Injectivity on K ×ˢ univ
  · intro ⟨r, c⟩ hrc ⟨r', c'⟩ hrc' heq_update
    -- From the j-th coordinate of the update: c = c'
    have hc : c = c' := by
      have := congr_fun heq_update j
      simp only [Function.update_apply] at this
      exact this
    -- From coordinates i ≠ j of the update: r i = r' i
    have hi : ∀ i, i ≠ j → r i = r' i := by
      intro i hne
      have := congr_fun heq_update i
      simp only [Function.update_apply, hne, ite_false] at this
      exact this
    -- Get kernel membership: linearForm d r = 0 means dotProduct r d = 0
    have hr : dotProduct r d = 0 := by
      have := (Finset.mem_product.mp hrc).1
      rw [Finset.mem_filter] at this; exact this.2
    have hr' : dotProduct r' d = 0 := by
      have := (Finset.mem_product.mp hrc').1
      rw [Finset.mem_filter] at this; exact this.2
    -- From kernel constraint + agreement off j: r j = r' j
    have hj : r j = r' j := kernel_agree_at_j d j hdj r r' hr hr' hi
    -- Combine: (r, c) = (r', c')
    exact Prod.ext (funext fun i => by
      by_cases h : i = j
      · subst h; exact hj
      · exact hi i h) hc

/-! ## Freivalds Soundness -/

/-- The "error vector": d = W x - z. -/
def errorVec (W : Matrix (Fin m) (Fin n) (ZMod p))
    (x : Fin n → ZMod p) (z : Fin m → ZMod p) : Fin m → ZMod p :=
  W.mulVec x - z

section Completeness

variable {p : ℕ}
variable {m n : ℕ}

/-- **Theorem 1 (Freivalds Completeness)**:
    If z = W x, then the Freivalds check passes for every r. -/
theorem freivalds_complete
    (W : Matrix (Fin m) (Fin n) (ZMod p))
    (x : Fin n → ZMod p) (z : Fin m → ZMod p)
    (hz : z = W.mulVec x) (r : Fin m → ZMod p)
    (v : Fin n → ZMod p) (hv : v = W.transpose.mulVec r) :
    freivaldsAccepts v x r z := by
  unfold freivaldsAccepts
  rw [hv, ← precomputed_equiv, hz]

end Completeness

/-- **Theorem 1 (Freivalds Soundness)**:
    If z ≠ W x, then |{r : check passes}| * p ≤ p^m (error ≤ 1/p).

    The accepting set {r | (W^T r) . x = r . z} equals the kernel of the
    linear form r ↦ r . (Wx - z). Since z ≠ Wx, this linear form is nonzero,
    so the kernel bound applies. -/
theorem freivalds_sound
    (W : Matrix (Fin m) (Fin n) (ZMod p))
    (x : Fin n → ZMod p) (z : Fin m → ZMod p)
    (hz : z ≠ W.mulVec x) (hm : 0 < m) :
    (Finset.univ.filter fun r : Fin m → ZMod p =>
      dotProduct (W.transpose.mulVec r) x = dotProduct r z).card * p
      ≤ Fintype.card (Fin m → ZMod p) := by
  -- Step 1: Rewrite the accepting set as the kernel of r ↦ r . (Wx - z)
  rw [freivalds_filter_eq_kernel]
  -- Step 2: The error vector Wx - z is nonzero since z ≠ Wx
  have hd : W.mulVec x - z ≠ 0 := by
    intro h; apply hz; exact eq_of_sub_eq_zero h |>.symm
  -- Step 3: Apply the kernel bound
  exact kernel_bound_of_nonzero (W.mulVec x - z) hd hm

end VerifiedInference
