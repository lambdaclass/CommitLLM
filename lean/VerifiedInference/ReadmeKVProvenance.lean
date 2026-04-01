import Mathlib.Data.Real.Basic

import VerifiedInference.Protocol
import VerifiedInference.MerkleTree
import VerifiedInference.HashCommitment

/-!
# README KV Provenance

This module models Step 3 of the canonical README protocol:

- the provider opens prefix `K,V` values against `R_KV`
- the verifier samples earlier token positions
- sampled earlier positions are anchored by stronger shell checks
- the unsampled portion remains statistical

The current Lean results here are about receipt-aligned data structures,
commitment binding, and the README sampling formula. Full semantic anchoring of
`K,V` to upstream hidden states remains future work.
-/

namespace VerifiedInference

variable {α β : Type*}

/-- One `R_KV` leaf payload for a prefix position. -/
structure KVLeafData (β : Type*) where
  tokenIndex : Nat
  payload : KVPayload β

/-- Hash of one `R_KV` leaf. -/
def kvReceiptLeafHash (H : HashFunction α)
    (encodeKV : KVLeafData β → α) (leaf : KVLeafData β) : α :=
  H.hash (encodeKV leaf)

/-- `R_KV` commitment binding for a single prefix opening. -/
theorem kv_receipt_leaf_binding
    (H : HashFunction α) [CollisionResistant H]
    (encodeKV : KVLeafData β → α) (henc : Function.Injective encodeKV)
    (root : α) (idx : Nat) (siblings : List α)
    (leaf₁ leaf₂ : KVLeafData β)
    (hVerify₁ : verifyMerkleRec H (kvReceiptLeafHash H encodeKV leaf₁) siblings idx = root)
    (hVerify₂ : verifyMerkleRec H (kvReceiptLeafHash H encodeKV leaf₂) siblings idx = root) :
    leaf₁ = leaf₂ := by
  have hLeaf :
      kvReceiptLeafHash H encodeKV leaf₁ = kvReceiptLeafHash H encodeKV leaf₂ :=
    merkle_binding_rec H _ _ siblings idx (by rw [hVerify₁, hVerify₂])
  have hEnc : encodeKV leaf₁ = encodeKV leaf₂ :=
    hash_commitment_binding H _ _ hLeaf
  exact henc hEnc

/-- The sampled prefix positions are anchored when the shell-verified values
agree with the receipt-opened values on every sampled index. -/
def sampledPrefixAnchored
    (shellVerified opened : Nat → KVPayload β)
    (sampled : List Nat) : Prop :=
  ∀ idx : Nat, idx ∈ sampled → shellVerified idx = opened idx

/-- The sampled prefix positions are semantically anchored when the opened
values match the truthful prefix computation on every sampled index. -/
def sampledPrefixSemanticallyAnchored
    (truthful opened : Nat → KVPayload β)
    (sampled : List Nat) : Prop :=
  ∀ idx : Nat, idx ∈ sampled → truthful idx = opened idx

/-- Witness packaging the README's sampled-prefix argument:
shell verification establishes correctness at sampled positions, and the
opened `R_KV` values are compared against those shell-verified positions. -/
structure SampledPrefixWitness (β : Type*) where
  truthful : Nat → KVPayload β
  shellVerified : Nat → KVPayload β
  opened : Nat → KVPayload β
  sampled : List Nat
  shellMatchesTruth : ∀ idx : Nat, idx ∈ sampled → shellVerified idx = truthful idx
  openedMatchesShell : sampledPrefixAnchored shellVerified opened sampled

/-- Pointwise equality on the sampled indices establishes sampled anchoring. -/
theorem sampled_prefix_anchored_of_eq
    (shellVerified opened : Nat → KVPayload β)
    (sampled : List Nat)
    (hEq : ∀ idx : Nat, idx ∈ sampled → shellVerified idx = opened idx) :
    sampledPrefixAnchored shellVerified opened sampled :=
  hEq

/-- Sampled semantic anchoring follows from shell correctness at the sampled
positions together with matching receipt-opened values at those positions. -/
theorem sampled_prefix_semantic_anchoring
    (truthful shellVerified opened : Nat → KVPayload β)
    (sampled : List Nat)
    (hShellTruth : ∀ idx : Nat, idx ∈ sampled → shellVerified idx = truthful idx)
    (hAnchored : sampledPrefixAnchored shellVerified opened sampled) :
    sampledPrefixSemanticallyAnchored truthful opened sampled := by
  intro idx hIdx
  calc
    truthful idx = shellVerified idx := (hShellTruth idx hIdx).symm
    _ = opened idx := hAnchored idx hIdx

/-- The packaged witness yields semantic correctness of the opened prefix on
every sampled index. -/
theorem SampledPrefixWitness.semanticAnchoring
    (w : SampledPrefixWitness β) :
    sampledPrefixSemanticallyAnchored w.truthful w.opened w.sampled := by
  exact sampled_prefix_semantic_anchoring
    w.truthful w.shellVerified w.opened w.sampled
    w.shellMatchesTruth w.openedMatchesShell

/-- If a tampered prefix position is sampled, the sampled-prefix argument
rejects it. -/
theorem sampled_prefix_tampering_caught
    (truthful shellVerified opened : Nat → KVPayload β)
    (sampled : List Nat) {idx : Nat}
    (hShellTruth : ∀ j : Nat, j ∈ sampled → shellVerified j = truthful j)
    (hAnchored : sampledPrefixAnchored shellVerified opened sampled)
    (hIdx : idx ∈ sampled)
    (hTampered : truthful idx ≠ opened idx) :
    False := by
  have hSemantic :=
    sampled_prefix_semantic_anchoring truthful shellVerified opened sampled
      hShellTruth hAnchored
  exact hTampered (hSemantic idx hIdx)

/-- The verifier detects prefix tampering whenever its sampled set intersects a
tampered position. -/
def sampledHitsTampered (tampered : Finset Nat) (sampled : List Nat) : Prop :=
  ∃ idx : Nat, idx ∈ sampled ∧ idx ∈ tampered

/-- A packaged sampled-prefix witness rejects any tampering scenario that hits
one of the sampled prefix positions. -/
theorem SampledPrefixWitness.rejects_if_sampled_tampered
    (w : SampledPrefixWitness β)
    (tampered : Finset Nat)
    (hTampered : ∀ idx : Nat, idx ∈ tampered → w.truthful idx ≠ w.opened idx)
    (hHit : sampledHitsTampered tampered w.sampled) :
    False := by
  rcases hHit with ⟨idx, hSampled, hTamperedIdx⟩
  exact sampled_prefix_tampering_caught
    w.truthful w.shellVerified w.opened w.sampled
    w.shellMatchesTruth w.openedMatchesShell hSampled
    (hTampered idx hTamperedIdx)

/-- README detection probability for `m` tampered positions out of `n`
prefix positions when the verifier samples `k` positions. -/
noncomputable def catchProbability (m n k : Nat) : ℝ :=
  1 - (1 - (m : ℝ) / (n : ℝ)) ^ k

/-- The Lean definition matches the README formula
`P(catch) = 1 - (1 - m / n)^k`. -/
theorem catchProbability_readme_formula (m n k : Nat) :
    catchProbability m n k = 1 - (1 - (m : ℝ) / (n : ℝ)) ^ k := rfl

/-- README-native statistical statement for sampled KV provenance:
if the sample hits a tampered prefix position, rejection is exact, and the
probability of such a hit is the README formula instantiated with the number
of tampered positions and the sample size. -/
theorem SampledPrefixWitness.statistical_detection_theorem
    (w : SampledPrefixWitness β)
    (tampered : Finset Nat)
    (prefixLength : Nat)
    (hTampered : ∀ idx : Nat, idx ∈ tampered → w.truthful idx ≠ w.opened idx) :
    (sampledHitsTampered tampered w.sampled → False) ∧
    catchProbability tampered.card prefixLength w.sampled.length =
      1 - (1 - (tampered.card : ℝ) / (prefixLength : ℝ)) ^ w.sampled.length := by
  refine ⟨?_, ?_⟩
  · intro hHit
    exact w.rejects_if_sampled_tampered tampered hTampered hHit
  · exact catchProbability_readme_formula
      tampered.card prefixLength w.sampled.length

end VerifiedInference
