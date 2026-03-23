import VerifiedInference.CanonicalProtocol
import VerifiedInference.StorageReconstruction
import VerifiedInference.ReadmeShell
import VerifiedInference.ReadmeKVProvenance
import VerifiedInference.ApproximateAttentionReplay

/-!
# Canonical README Soundness

This module gives the repository's current top-level Lean statement for the
canonical README protocol.

It is intentionally layered rather than overclaiming:

- the exact shell is exact
- prefix KV provenance is statistical
- cross-layer consistency is structural
- attention replay is approximate
- storage reconstruction is deterministic once the retained artifacts and
  transcript are fixed

This matches the README's own statement of protocol boundaries.
-/

namespace VerifiedInference

/-- Top-level README-style guarantees after an accepted audit. -/
structure CanonicalSoundnessResult where
  modelIdentityExact : Prop
  kvProvenanceStatistical : Prop
  crossLayerStructural : Prop
  attentionReplayApproximate : Prop
  storageReconstructionDeterministic : Prop

/-- A README-native package of the current formal boundary. -/
def canonical_protocol_sound
    {α β σ τ : Type} {nLayers : Nat}
    (checks : CanonicalAuditChecks α β)
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers) :
    CanonicalSoundnessResult :=
  { modelIdentityExact := checks.exactShell.holds
    kvProvenanceStatistical := checks.kvCommitmentBinding ∧ checks.sampledAnchoring
    crossLayerStructural := checks.crossLayerConsistency
    attentionReplayApproximate := checks.attentionReplay
    storageReconstructionDeterministic := ∀ candidate₁ candidate₂ : ReconstructedState τ,
      storageReconstructionClaim oracle tokens artifacts candidate₁ →
      storageReconstructionClaim oracle tokens artifacts candidate₂ →
      candidate₁ = candidate₂ }

/-- The model-identity component of the top-level result is exactly the
accepted exact shell. -/
theorem canonical_protocol_model_identity
    {α β σ τ : Type} {nLayers : Nat}
    (checks : CanonicalAuditChecks α β)
    (hAccept : canonicalAuditAccepts checks)
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers) :
    (canonical_protocol_sound checks oracle tokens artifacts).modelIdentityExact := by
  change checks.exactShell.holds
  exact hAccept.1

/-- The KV provenance component remains statistical, matching the README. -/
theorem canonical_protocol_kv_statistical
    {α β σ τ : Type} {nLayers : Nat}
    (checks : CanonicalAuditChecks α β)
    (hAccept : canonicalAuditAccepts checks)
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers) :
    (canonical_protocol_sound checks oracle tokens artifacts).kvProvenanceStatistical := by
  change checks.kvCommitmentBinding ∧ checks.sampledAnchoring
  exact ⟨hAccept.2.1, hAccept.2.2.1⟩

/-- The attention component remains approximate, matching the README. -/
theorem canonical_protocol_attention_approximate
    {α β σ τ : Type} {nLayers : Nat}
    (checks : CanonicalAuditChecks α β)
    (hAccept : canonicalAuditAccepts checks)
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers) :
    (canonical_protocol_sound checks oracle tokens artifacts).attentionReplayApproximate := by
  change checks.attentionReplay
  exact hAccept.2.2.2.2

/-- The storage component is deterministic once transcript and retained
artifacts are fixed. -/
theorem canonical_protocol_storage_reconstruction
    {α β σ τ : Type} {nLayers : Nat}
    (checks : CanonicalAuditChecks α β)
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers) :
    (canonical_protocol_sound checks oracle tokens artifacts).storageReconstructionDeterministic := by
  change ∀ candidate₁ candidate₂ : ReconstructedState τ,
    storageReconstructionClaim oracle tokens artifacts candidate₁ →
    storageReconstructionClaim oracle tokens artifacts candidate₂ →
    candidate₁ = candidate₂
  intro candidate₁ candidate₂ h₁ h₂
  exact storage_reconstruction_unique oracle tokens artifacts candidate₁ candidate₂ h₁ h₂

/-- A single README-style end-to-end theorem package.

This remains honest to the README's guarantee split:

- model identity on the opened slice is exact
- sampled prefix correctness is semantic only on sampled positions
- attention replay acceptance is approximate and depends on an explicit
  unstable-coordinate set
- storage reconstruction is deterministic once transcript and retained
  artifacts are fixed
-/
structure ReadmeEndToEndGuarantees
    {α β σ τ : Type} {nLayers replayWidth : Nat}
    (checks : CanonicalAuditChecks α β)
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers)
    (prefixWitness : SampledPrefixWitness β)
    (corridor : ReplayCorridor)
    (replayWitness : ApproximateReplayWitness replayWidth) : Prop where
  exactModelIdentity : checks.exactShell.holds
  exactOpenedWeightChecks : checks.exactShell.weightFreivaldsHold
  kvCommitmentBindingExact : checks.kvCommitmentBinding
  sampledPrefixSemanticCorrectness :
    sampledPrefixSemanticallyAnchored
      prefixWitness.truthful prefixWitness.opened prefixWitness.sampled
  crossLayerStructural : checks.crossLayerConsistency
  attentionReplayAccepted :
    approximateReplayAccepts corridor replayWitness.replayed replayWitness.committed
  storageReconstructionDeterministic :
    ∀ candidate₁ candidate₂ : ReconstructedState τ,
      storageReconstructionClaim oracle tokens artifacts candidate₁ →
      storageReconstructionClaim oracle tokens artifacts candidate₂ →
      candidate₁ = candidate₂

/-- A single compiled theorem composing the README-native exact shell,
sampled-prefix semantic anchoring, approximate replay acceptance, and storage
reconstruction determinism. -/
theorem canonical_protocol_end_to_end
    {α β σ τ : Type} {nLayers replayWidth : Nat}
    (checks : CanonicalAuditChecks α β)
    (hAccept : canonicalAuditAccepts checks)
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers)
    (prefixWitness : SampledPrefixWitness β)
    (corridor : ReplayCorridor)
    (replayWitness : ApproximateReplayWitness replayWidth)
    (hReplayBound : replayWitness.unstable.card ≤ corridor.maxDisagreements) :
    ReadmeEndToEndGuarantees
      checks oracle tokens artifacts prefixWitness corridor replayWitness := by
  refine
    { exactModelIdentity := hAccept.1
      exactOpenedWeightChecks :=
        exact_shell_implies_model_identity_on_opened_slice checks.exactShell hAccept.1
      kvCommitmentBindingExact := hAccept.2.1
      sampledPrefixSemanticCorrectness := prefixWitness.semanticAnchoring
      crossLayerStructural := hAccept.2.2.2.1
      attentionReplayAccepted := replayWitness.accepts corridor hReplayBound
      storageReconstructionDeterministic := ?_ }
  intro candidate₁ candidate₂ h₁ h₂
  exact storage_reconstruction_unique oracle tokens artifacts candidate₁ candidate₂ h₁ h₂

/-- The README catch-probability formula instantiated with the sampled prefix
size used in the composed end-to-end theorem. -/
theorem canonical_protocol_end_to_end_catch_probability
    {β : Type}
    (prefixWitness : SampledPrefixWitness β)
    (tamperedPositions prefixLength : Nat) :
    catchProbability tamperedPositions prefixLength prefixWitness.sampled.length =
      1 - (1 - (tamperedPositions : ℝ) / (prefixLength : ℝ)) ^ prefixWitness.sampled.length := by
  exact catchProbability_readme_formula
    tamperedPositions prefixLength prefixWitness.sampled.length

end VerifiedInference
