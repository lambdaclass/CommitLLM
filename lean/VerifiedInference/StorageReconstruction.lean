import VerifiedInference.Storage

/-!
# README Storage Reconstruction

The README states that, for a fixed response, the only non-derivable provider
state is the per-layer `attn_out_i8` plus its scale. Everything else is
re-derived from:

- input tokens
- output tokens
- retained attention artifacts

This module captures that claim at the level currently practical for the Lean
tree: reconstruction is modeled abstractly as a pure function of those three
inputs, so uniqueness / determinism follows immediately.
-/

namespace VerifiedInference

/-- Input/output token transcript for one audited response. -/
structure TranscriptTokens where
  inputTokens : List Nat
  outputTokens : List Nat

/-- Abstract reconstruction target for the README storage theorem. -/
structure ReconstructedState (τ : Type) where
  state : τ

/-- Pure reconstruction oracle for README-retained artifacts. -/
structure ReconstructionOracle (σ τ : Type) (nLayers : Nat) where
  reconstruct :
    TranscriptTokens →
    RetainedTokenArtifacts σ nLayers →
    ReconstructedState τ

/-- Reconstruction from transcript + retained artifacts is deterministic. -/
theorem reconstruction_deterministic
    {σ τ : Type} {nLayers : Nat}
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers) :
    ∀ s₁ s₂ : ReconstructedState τ,
      s₁ = oracle.reconstruct tokens artifacts →
      s₂ = oracle.reconstruct tokens artifacts →
      s₁ = s₂ := by
  intro s₁ s₂ h₁ h₂
  rw [h₁, h₂]

/-- A README-style storage reconstruction claim is just the statement that the
provider-side state is the output of the canonical reconstruction oracle. -/
def storageReconstructionClaim
    {σ τ : Type} {nLayers : Nat}
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers)
    (candidate : ReconstructedState τ) : Prop :=
  candidate = oracle.reconstruct tokens artifacts

/-- Any two candidates satisfying the canonical reconstruction claim are equal. -/
theorem storage_reconstruction_unique
    {σ τ : Type} {nLayers : Nat}
    (oracle : ReconstructionOracle σ τ nLayers)
    (tokens : TranscriptTokens)
    (artifacts : RetainedTokenArtifacts σ nLayers)
    (candidate₁ candidate₂ : ReconstructedState τ)
    (h₁ : storageReconstructionClaim oracle tokens artifacts candidate₁)
    (h₂ : storageReconstructionClaim oracle tokens artifacts candidate₂) :
    candidate₁ = candidate₂ := by
  unfold storageReconstructionClaim at h₁ h₂
  rw [h₁, h₂]

end VerifiedInference
