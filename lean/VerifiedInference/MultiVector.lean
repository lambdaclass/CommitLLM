import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Ring

import VerifiedInference.Basic

/-!
# Theorem 7: Multi-Vector Amplification

**Statement**: If k independent Freivalds checks each pass with
probability ≤ 1/p, then all k checks pass with probability ≤ (1/p)^k.

Stated combinatorially: if each check's "bad set" has at most
|S|/p elements, then the joint bad set in the product space has
at most |S|^k / p^k elements.

This avoids measure theory entirely by working with Fintype cardinalities.

Mirrors: The protocol runs 7 Freivalds checks per layer (one per matrix type).
-/

namespace VerifiedInference

/-! ## Single-event bound -/

/-- A "bad set" bound: the number of elements in a subset of a finite type
    is at most 1/p of the total. Expressed as: |bad| * p ≤ |total|. -/
structure BadSetBound (α : Type*) [Fintype α] (bad : Finset α) (p : ℕ) : Prop where
  bound : bad.card * p ≤ Fintype.card α

/-! ## Product space bound -/

/-- Base case (Theorem 7): for a single event, the bound holds by assumption. -/
theorem amplification_base {α : Type*} [Fintype α]
    {bad : Finset α} {p : ℕ}
    (h : bad.card * p ≤ Fintype.card α) :
    bad.card * p ^ 1 ≤ Fintype.card α := by
  simpa [pow_one] using h

/-- Key lemma: if |A| * p ≤ |S| and |B| * p ≤ |T|,
    then |A × B| * p² ≤ |S × T|. -/
theorem product_bound {α β : Type*} [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    {badA : Finset α} {badB : Finset β} {p : ℕ}
    (hA : badA.card * p ≤ Fintype.card α)
    (hB : badB.card * p ≤ Fintype.card β) :
    (badA ×ˢ badB).card * p ^ 2 ≤ Fintype.card (α × β) := by
  rw [Finset.card_product, Fintype.card_prod, pow_succ, pow_one]
  -- Goal: badA.card * badB.card * (p * p) ≤ card α * card β
  -- From hA: badA.card * p ≤ card α
  -- From hB: badB.card * p ≤ card β
  -- Multiply: badA.card * p * (badB.card * p) ≤ card α * card β
  calc badA.card * badB.card * (p * p)
      = (badA.card * p) * (badB.card * p) := by ring
    _ ≤ Fintype.card α * Fintype.card β := Nat.mul_le_mul hA hB

/-- **Inductive amplification for k independent checks over the same space.**

    If we draw k independent random vectors from a space of size q,
    and each check's bad set has at most q/p elements (i.e., badCard * p ≤ q),
    then the joint bad set in q^k has at most q^k / p^k elements.

    Formally: badCard^k * p^k ≤ q^k. -/
theorem amplification_inductive
    (q p badCard : ℕ)
    (h : badCard * p ≤ q)
    (k : ℕ) :
    badCard ^ k * p ^ k ≤ q ^ k := by
  rw [← Nat.mul_pow]
  exact Nat.pow_le_pow_left h k

end VerifiedInference
