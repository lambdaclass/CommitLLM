/-!
# README Storage Objects

The canonical README storage claim is intentionally narrow: the provider need
only retain the post-attention `attn_out_i8` vector plus its quantization
scale for each layer of each output token. The rest of the exact-shell trace
is re-derived on demand from the input/output tokens and the retained
attention artifacts.

This module introduces only the current retained-artifact shape. It does not
yet prove the full reconstruction theorem from those artifacts back to the
wider README protocol state.
-/

namespace VerifiedInference

/-- The non-derivable per-layer artifact retained for one output token.

`σ` abstracts over the concrete scale representation used by an
implementation. The README treats this as "one float per layer"; the Lean
model keeps the type abstract because the current formalization does not rely
on floating-point details here. -/
structure RetainedAttentionArtifact (σ : Type) where
  attnOutI8 : List Int
  scale : σ

/-- Retained per-layer artifacts for one output token. -/
structure RetainedTokenArtifacts (σ : Type) (nLayers : Nat) where
  perLayer : Fin nLayers → RetainedAttentionArtifact σ

end VerifiedInference
