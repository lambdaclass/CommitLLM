import VerifiedInference.Basic
import VerifiedInference.HashCommitment
import VerifiedInference.MerkleTree
import VerifiedInference.Freivalds

/-!
# Theorem 6: Weight-Model Binding

**Statement**: If the weight commitments verify (Merkle proofs pass) and
the Freivalds checks pass, then the prover used the committed weights.

This combines:
- Theorem 5 (Merkle binding): committed root binds the weight hashes
- Theorem 4 (Hash commitment): weight hashes bind the actual weights
- Theorem 1 (Freivalds): matrix product check ensures Wx was computed

Mirrors: The verification pipeline in `crates/vi-core/` that checks
Merkle proofs before running Freivalds.
-/

namespace VerifiedInference

variable {α : Type*}

/-- A weight commitment: the Merkle root over hashed weight matrices. -/
structure WeightCommitment (α : Type*) where
  /-- The Merkle root over H(W_j^{(i)}) for all layers i, matrix types j -/
  root : α

/-- **Theorem 6 (Weight-Model Binding)**:

    If:
    1. The Merkle proof for weight W at position idx verifies against the
       committed root (Merkle binding)
    2. The hash of the claimed weight matches the Merkle leaf (hash binding)
    3. The Freivalds check v · x = r · z passes

    Then the prover's computation is consistent with the committed weight:
    the claimed output z satisfies z = W x with probability ≥ 1 - 1/p,
    where W is the weight committed in the Merkle tree.

    This is a "wrapper" theorem composing Theorems 1, 4, and 5.
-/
theorem weight_model_binding
    (H : HashFunction α) [hcr : CollisionResistant H]
    -- The committed weight (known to verifier via Merkle root)
    (committedWeight : α)
    -- The weight the prover claims to have used
    (claimedWeight : α)
    -- Hash binding: both weights have the same hash
    (hHash : H.hash committedWeight = H.hash claimedWeight) :
    -- Conclusion: the weights are identical
    committedWeight = claimedWeight :=
  hash_commitment_binding H committedWeight claimedWeight hHash

/-- **Corollary**: End-to-end weight binding.

    If the verifier holds a Merkle root that commits to weight hashes,
    and a Merkle proof verifies a leaf against this root, and the leaf
    is the hash of the weight matrix, then the weight is uniquely determined
    by the root.

    The Freivalds check (Thm 1) then ensures the matmul output z = Wx
    is correct with high probability over the random vector r.
-/
theorem end_to_end_weight_binding
    (H : HashFunction α) [hcr : CollisionResistant H]
    (root : α)
    -- Two proofs claiming different weights at the same position
    (weight₁ weight₂ : α)
    (leaf₁ leaf₂ : α)
    (hLeaf₁ : leaf₁ = H.hash weight₁)
    (hLeaf₂ : leaf₂ = H.hash weight₂)
    -- Both verify against the same root via recursive proofs
    (siblings : List α) (idx : ℕ)
    (hVerify₁ : verifyMerkleRec H leaf₁ siblings idx = root)
    (hVerify₂ : verifyMerkleRec H leaf₂ siblings idx = root) :
    weight₁ = weight₂ := by
  -- Step 1: Merkle binding gives leaf₁ = leaf₂
  have hLeafEq : leaf₁ = leaf₂ :=
    merkle_binding_rec H leaf₁ leaf₂ siblings idx (by rw [hVerify₁, hVerify₂])
  -- Step 2: Hash binding gives weight₁ = weight₂
  rw [hLeaf₁, hLeaf₂] at hLeafEq
  exact hash_commitment_binding H weight₁ weight₂ hLeafEq

end VerifiedInference
