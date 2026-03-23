import VerifiedInference.Protocol
import VerifiedInference.ReadmeShell
import VerifiedInference.ReadmeKVProvenance
import VerifiedInference.ApproximateAttentionReplay
import VerifiedInference.MerkleTree
import VerifiedInference.HashCommitment
import VerifiedInference.WeightBinding

/-!
# Canonical README Protocol

This module packages the canonical README protocol as the default Lean-facing
surface. It does not claim a full end-to-end formal proof of the README yet;
instead, it:

- exposes the exact / statistical / structural / approximate split directly
- defines the canonical audit-acceptance predicate over that split
- proves the structural binding results that already follow from the exact
  hash / Merkle assumptions
- keeps the weaker attention and sampled-KV layers explicit as weaker
  predicates, matching the README's stated boundary
-/

namespace VerifiedInference

variable {α β : Type*}

/-- Guarantee categories used in the README composition table. -/
inductive GuaranteeKind
  | exact
  | statistical
  | structural
  | approximate
  | none
  deriving DecidableEq, Repr

/-- README guarantee summary. -/
structure GuaranteeSummary where
  shellVerification : GuaranteeKind
  kvProvenance : GuaranteeKind
  crossLayerConsistency : GuaranteeKind
  attentionReplay : GuaranteeKind
  unopenedCoverage : GuaranteeKind

/-- The canonical guarantee summary from the README. -/
def readmeGuaranteeSummary : GuaranteeSummary :=
  { shellVerification := .exact
    kvProvenance := .statistical
    crossLayerConsistency := .structural
    attentionReplay := .approximate
    unopenedCoverage := .none }

/-- Canonical audit checks for one README-style audited response. -/
structure CanonicalAuditChecks (α β : Type*) where
  receipt : Receipt α
  manifest : DeploymentManifest α
  exactShell : ExactShellChecks
  kvCommitmentBinding : Prop
  sampledAnchoring : Prop
  crossLayerConsistency : Prop
  attentionReplay : Prop

/-- README-style audit acceptance predicate.

This keeps the exact shell and the weaker KV / attention layers separate
instead of collapsing them into one undifferentiated theorem. -/
def canonicalAuditAccepts (checks : CanonicalAuditChecks α β) : Prop :=
  checks.exactShell.holds ∧
  checks.kvCommitmentBinding ∧
  checks.sampledAnchoring ∧
  checks.crossLayerConsistency ∧
  checks.attentionReplay

/-- Trace openings are bound by the receipt's `R_T`. -/
theorem receipt_trace_binding
    (H : HashFunction α) [CollisionResistant H]
    (receipt : Receipt α)
    (leaf₁ leaf₂ : α)
    (siblings : List α) (idx : Nat)
    (hVerify₁ : verifyMerkleRec H leaf₁ siblings idx = receipt.traceRoot)
    (hVerify₂ : verifyMerkleRec H leaf₂ siblings idx = receipt.traceRoot) :
    leaf₁ = leaf₂ := by
  exact merkle_binding_rec H leaf₁ leaf₂ siblings idx (by rw [hVerify₁, hVerify₂])

/-- KV openings are bound by the receipt's `R_KV`. -/
theorem receipt_kv_binding
    (H : HashFunction α) [CollisionResistant H]
    (receipt : Receipt α)
    (leaf₁ leaf₂ : α)
    (siblings : List α) (idx : Nat)
    (hVerify₁ : verifyMerkleRec H leaf₁ siblings idx = receipt.kvRoot)
    (hVerify₂ : verifyMerkleRec H leaf₂ siblings idx = receipt.kvRoot) :
    leaf₁ = leaf₂ := by
  exact merkle_binding_rec H leaf₁ leaf₂ siblings idx (by rw [hVerify₁, hVerify₂])

/-- Deployment-manifest binding for the receipt hash `M`. -/
theorem deployment_manifest_binding
    (H : HashFunction α) [CollisionResistant H]
    (encodeManifest : DeploymentManifest α → α)
    (henc : Function.Injective encodeManifest)
    (manifest₁ manifest₂ : DeploymentManifest α)
    (hHash :
      H.hash (encodeManifest manifest₁) =
      H.hash (encodeManifest manifest₂)) :
    manifest₁ = manifest₂ := by
  have hEnc : encodeManifest manifest₁ = encodeManifest manifest₂ :=
    hash_commitment_binding H _ _ hHash
  exact henc hEnc

/-- Weight openings are bound exactly by the weight root committed in the
deployment manifest. -/
theorem manifest_weight_binding
    (H : HashFunction α) [CollisionResistant H]
    (manifest : DeploymentManifest α)
    (weight₁ weight₂ : α)
    (leaf₁ leaf₂ : α)
    (hLeaf₁ : leaf₁ = H.hash weight₁)
    (hLeaf₂ : leaf₂ = H.hash weight₂)
    (siblings : List α) (idx : Nat)
    (hVerify₁ : verifyMerkleRec H leaf₁ siblings idx = manifest.weightRoot)
    (hVerify₂ : verifyMerkleRec H leaf₂ siblings idx = manifest.weightRoot) :
    weight₁ = weight₂ := by
  exact end_to_end_weight_binding
    H manifest.weightRoot weight₁ weight₂ leaf₁ leaf₂
    hLeaf₁ hLeaf₂ siblings idx hVerify₁ hVerify₂

/-- A canonical accepted audit exposes the exact shell as a separately exact
guarantee, matching the README composition table. -/
theorem canonical_audit_exact_shell
    (checks : CanonicalAuditChecks α β)
    (hAccept : canonicalAuditAccepts checks) :
    checks.exactShell.holds := by
  exact hAccept.1

/-- A canonical accepted audit exposes the weaker sampled-KV guarantee
separately from the exact shell. -/
theorem canonical_audit_sampled_kv
    (checks : CanonicalAuditChecks α β)
    (hAccept : canonicalAuditAccepts checks) :
    checks.kvCommitmentBinding ∧ checks.sampledAnchoring := by
  exact ⟨hAccept.2.1, hAccept.2.2.1⟩

/-- A canonical accepted audit exposes the approximate replay guarantee
separately from the exact shell. -/
theorem canonical_audit_attention_replay
    (checks : CanonicalAuditChecks α β)
    (hAccept : canonicalAuditAccepts checks) :
    checks.crossLayerConsistency ∧ checks.attentionReplay := by
  exact ⟨hAccept.2.2.2.1, hAccept.2.2.2.2⟩

/-- Exact opened-layer model identity in the canonical README protocol:
trace openings are bound by `R_T`, the deployment manifest is bound by the
receipt hash, and the opened shell exposes the exact Freivalds-checked weight
matmuls for that slice. -/
theorem canonical_opened_layer_model_identity_exact
    (H : HashFunction α) [CollisionResistant H]
    (checks : CanonicalAuditChecks α β)
    (hAccept : canonicalAuditAccepts checks)
    (traceLeaf₁ traceLeaf₂ : α)
    (traceSiblings : List α) (traceIdx : Nat)
    (hTrace₁ : verifyMerkleRec H traceLeaf₁ traceSiblings traceIdx = checks.receipt.traceRoot)
    (hTrace₂ : verifyMerkleRec H traceLeaf₂ traceSiblings traceIdx = checks.receipt.traceRoot)
    (encodeManifest : DeploymentManifest α → α)
    (henc : Function.Injective encodeManifest)
    (manifestCandidate : DeploymentManifest α)
    (hReceiptManifest :
      checks.receipt.manifestHash = H.hash (encodeManifest checks.manifest))
    (hCandidateManifest :
      checks.receipt.manifestHash = H.hash (encodeManifest manifestCandidate)) :
    traceLeaf₁ = traceLeaf₂ ∧
    checks.manifest = manifestCandidate ∧
    checks.exactShell.weightFreivaldsHold := by
  refine ⟨?_, ?_, ?_⟩
  · exact receipt_trace_binding
      H checks.receipt traceLeaf₁ traceLeaf₂ traceSiblings traceIdx hTrace₁ hTrace₂
  · exact deployment_manifest_binding
      H encodeManifest henc checks.manifest manifestCandidate
      (by rw [← hReceiptManifest, hCandidateManifest])
  · exact exact_shell_implies_model_identity_on_opened_slice checks.exactShell hAccept.1

end VerifiedInference
