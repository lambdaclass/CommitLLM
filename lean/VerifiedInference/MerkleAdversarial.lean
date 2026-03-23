import VerifiedInference.Basic
import VerifiedInference.HashCommitment
import VerifiedInference.MerkleTree

/-!
# Theorem: Adversarial Merkle Binding

**Statement**: Under collision resistance, if two leaves verify to the same root
at the same index but possibly via *different* sibling lists (of equal length),
then the leaves are equal AND the sibling lists are equal.

This strengthens `merkle_binding_rec` which requires the same siblings.
An adversary who finds two distinct proofs (different siblings) for different
leaves at the same position would imply a hash collision.

Mirrors: `crates/vi-core/src/merkle.rs` — `verify()`, `build_tree()`
-/

namespace VerifiedInference

variable {α : Type*}

/-- **Adversarial Merkle Binding**:

    If H is collision-resistant, and two leaves verify to the same root
    at the same index via sibling lists of the same length, then
    the leaves are equal and the sibling lists are equal.

    This is strictly stronger than `merkle_binding_rec`, which requires
    identical sibling lists. Here the adversary may choose different siblings.

    Proof by induction on `siblings₁`:
    - Base: both lists nil. `verifyMerkleRec` returns the leaf directly,
      so `leaf₁ = leaf₂` from `hRoot`.
    - Step: both lists are cons. Unfold `verifyMerkleRec`. Both sides take the
      same branch (same `idx`). The recursive call gives parents that verify to
      the same root via shorter lists. By IH the parents are equal and the
      rest-siblings are equal. Then `hashPair_injective` gives `leaf₁ = leaf₂`
      and `s₁ = s₂`. -/
theorem merkle_binding_adversarial
    (H : HashFunction α) [hcr : CollisionResistant H]
    (leaf₁ leaf₂ : α)
    (siblings₁ siblings₂ : List α) (idx : ℕ)
    (hLen : siblings₁.length = siblings₂.length)
    (hRoot : verifyMerkleRec H leaf₁ siblings₁ idx =
             verifyMerkleRec H leaf₂ siblings₂ idx) :
    leaf₁ = leaf₂ ∧ siblings₁ = siblings₂ := by
  induction siblings₁ generalizing leaf₁ leaf₂ siblings₂ idx with
  | nil =>
    cases siblings₂ with
    | nil =>
      simp only [verifyMerkleRec] at hRoot
      exact ⟨hRoot, rfl⟩
    | cons _ _ => simp [List.length] at hLen
  | cons s₁ rest₁ ih =>
    cases siblings₂ with
    | nil => simp [List.length] at hLen
    | cons s₂ rest₂ =>
      simp only [List.length_cons] at hLen
      have hLenRest : rest₁.length = rest₂.length := Nat.succ_injective hLen
      simp only [verifyMerkleRec] at hRoot
      split at hRoot
      · -- idx % 2 == 0: parent = hashPair current sibling
        have ihResult := ih _ _ _ (idx / 2) hLenRest hRoot
        obtain ⟨hParentEq, hRestEq⟩ := ihResult
        have hPair := hcr.hashPair_injective _ _ _ _ hParentEq
        exact ⟨hPair.1, by rw [hPair.2, hRestEq]⟩
      · -- idx % 2 != 0: parent = hashPair sibling current
        have ihResult := ih _ _ _ (idx / 2) hLenRest hRoot
        obtain ⟨hParentEq, hRestEq⟩ := ihResult
        have hPair := hcr.hashPair_injective _ _ _ _ hParentEq
        exact ⟨hPair.2, by rw [hPair.1, hRestEq]⟩

end VerifiedInference
