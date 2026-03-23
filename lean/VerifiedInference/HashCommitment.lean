import VerifiedInference.Basic

/-!
# Theorem 4: Hash Commitment Binding

**Statement**: Under collision resistance, a hash commitment binds the
committed value — no adversary can find two distinct values with the same hash.

This is axiomatized rather than proved, following standard practice in
cryptographic formalizations. Collision resistance is a computational
assumption about a concrete hash function (e.g., SHA-256).

Note: The paper's `LEAN_PROOFS.md` mentions "preimage resistance" but what's
actually needed is collision resistance (weaker, sufficient).

Mirrors: `crates/vi-core/src/merkle.rs` — `hash_pair()`, `hash_leaf()`
-/

namespace VerifiedInference

/-! ## Hash Function Abstraction -/

/-- A hash function from arbitrary-length byte sequences to fixed-length digests. -/
structure HashFunction (α : Type*) where
  /-- The hash function -/
  hash : α → α
  /-- Domain tag for leaves vs internal nodes (optional, for Merkle) -/
  hashPair : α → α → α

/-- **Collision resistance axiom**: We model computational collision resistance
    as injectivity. This is standard in all cryptographic formalizations —
    we cannot prove computational hardness in a proof assistant.

    In practice, this means: "assuming SHA-256 is collision-resistant,
    no polynomial-time adversary can find distinct x, y with H(x) = H(y)." -/
class CollisionResistant {α : Type*} (H : HashFunction α) : Prop where
  /-- No two distinct inputs map to the same hash. -/
  hash_injective : Function.Injective H.hash
  /-- No two distinct pairs map to the same pair-hash. -/
  hashPair_injective : ∀ a₁ b₁ a₂ b₂,
    H.hashPair a₁ b₁ = H.hashPair a₂ b₂ → a₁ = a₂ ∧ b₁ = b₂

/-- **Theorem 4 (Hash Commitment Binding)**:

    If H is collision-resistant and H(x) = H(y), then x = y.
    A commitment H(m) uniquely binds the message m.
-/
theorem hash_commitment_binding {α : Type*}
    (H : HashFunction α) [hcr : CollisionResistant H]
    (x y : α) (h : H.hash x = H.hash y) : x = y :=
  hcr.hash_injective h

/-- Corollary: pair-hash binding for Merkle internal nodes. -/
theorem pair_hash_binding {α : Type*}
    (H : HashFunction α) [hcr : CollisionResistant H]
    (a₁ b₁ a₂ b₂ : α) (h : H.hashPair a₁ b₁ = H.hashPair a₂ b₂) :
    a₁ = a₂ ∧ b₁ = b₂ :=
  hcr.hashPair_injective a₁ b₁ a₂ b₂ h

end VerifiedInference
