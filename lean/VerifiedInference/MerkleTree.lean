import VerifiedInference.Basic
import VerifiedInference.HashCommitment

/-!
# Theorem 5: Merkle Tree Binding

**Statement**: Under collision resistance, if two different leaves verify
against the same root at the same position, we can extract a hash collision.

By contrapositive under CollisionResistant: identical root + identical
position + valid proofs implies identical leaves.

Mirrors: `crates/vi-core/src/merkle.rs` — `verify()`, `build_tree()`
-/

namespace VerifiedInference

variable {α : Type*}

/-! ## Merkle Tree Definitions -/

/-- A Merkle proof: a list of sibling hashes along the path from leaf to root,
    together with the direction bits (leaf_index). -/
structure MerkleProof (α : Type*) where
  /-- Sibling hashes from leaf to root -/
  siblings : List α
  /-- Leaf index (encodes left/right choices as bits) -/
  leafIndex : ℕ

/-- Recursive Merkle verification, indexed by depth for clean induction. -/
def verifyMerkleRec (H : HashFunction α) (current : α) :
    List α → ℕ → α
  | [], _ => current
  | sibling :: rest, idx =>
    let parent := if idx % 2 == 0
      then H.hashPair current sibling
      else H.hashPair sibling current
    verifyMerkleRec H parent rest (idx / 2)

/-- Verify a Merkle proof by recomputing the root from leaf to top. -/
def verifyMerkleProof (H : HashFunction α) (leaf : α) (proof : MerkleProof α) : α :=
  verifyMerkleRec H leaf proof.siblings proof.leafIndex

/-- **Theorem 5 (Merkle Binding via recursive verification)**:

    If H is collision-resistant, and two leaves verify to the same root
    at the same index via the same siblings, then the leaves are equal.

    Proof by induction on the sibling list:
    - Base: no siblings, leaf IS the root. Same root → same leaf.
    - Step: Collision resistance of hashPair forces children equal.
      Apply IH to deduce leaves equal.
-/
theorem merkle_binding_rec
    (H : HashFunction α) [hcr : CollisionResistant H]
    (leaf₁ leaf₂ : α) (siblings : List α) (idx : ℕ)
    (hRoot : verifyMerkleRec H leaf₁ siblings idx =
             verifyMerkleRec H leaf₂ siblings idx) :
    leaf₁ = leaf₂ := by
  induction siblings generalizing leaf₁ leaf₂ idx with
  | nil =>
    simpa [verifyMerkleRec] using hRoot
  | cons sibling rest ih =>
    simp only [verifyMerkleRec] at hRoot
    split at hRoot
    · exact (hcr.hashPair_injective _ _ _ _ (ih _ _ _ hRoot)).1
    · exact (hcr.hashPair_injective _ _ _ _ (ih _ _ _ hRoot)).2

/-- **Corollary**: Merkle binding for the common case where proofs share
    structure (same index, same siblings). A committed root uniquely determines
    each leaf at each position. -/
theorem merkle_leaf_unique
    (H : HashFunction α) [hcr : CollisionResistant H]
    (root : α) (idx : ℕ)
    (leaf₁ leaf₂ : α)
    (siblings : List α)
    (hVerify₁ : verifyMerkleProof H leaf₁ ⟨siblings, idx⟩ = root)
    (hVerify₂ : verifyMerkleProof H leaf₂ ⟨siblings, idx⟩ = root) :
    leaf₁ = leaf₂ := by
  unfold verifyMerkleProof at hVerify₁ hVerify₂
  simp at hVerify₁ hVerify₂
  exact merkle_binding_rec H leaf₁ leaf₂ siblings idx (by rw [hVerify₁, hVerify₂])

end VerifiedInference
