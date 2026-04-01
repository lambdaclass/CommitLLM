import VerifiedInference.Basic
import VerifiedInference.MerkleTree
import VerifiedInference.SiLU
import VerifiedInference.Requantization

/-!
# Protocol Objects

`README.md` is the canonical protocol description for this repository.
This module contains only README-native protocol objects plus a still-useful
exact-shell slice (`LayerState`) used by the current Lean formalization.
-/

namespace VerifiedInference

variable {α : Type*} [DecidableEq α]

/-! ## Canonical README Receipt Objects -/

/-- Deployment-manifest fields bound into the README receipt hash `M`. -/
structure DeploymentManifest (α : Type*) where
  tokenizerHash : α
  weightRoot : α
  quantHash : α
  samplingParamsHash : α
  eosPolicyHash : α
  systemPromptHash : α

/-- The canonical receipt described in `README.md`.

It binds:
- `traceRoot = R_T`
- `kvRoot = R_KV`
- `manifestHash = M`
- `nTokens = N` -/
structure Receipt (α : Type*) where
  traceRoot : α
  kvRoot : α
  manifestHash : α
  nTokens : ℕ

/-- One opened `K,V` payload against the receipt's `R_KV`.

The payload values are kept abstract here because the current Lean tree does
not yet formalize the sampled semantic anchoring step from the README; it only
needs a typed object for README-aligned audit openings. -/
structure KVPayload (β : Type*) where
  k : β
  v : β

/-- One prefix position opened from the canonical `R_KV` commitment. -/
structure PrefixKVOpening (α : Type*) (β : Type*) where
  tokenIndex : ℕ
  payload : KVPayload β
  merkleProof : MerkleProof α

/-- Audit-time opening object for the README's sampled prefix-provenance step.

This captures the shape of Step 3 in the README: the prover opens prefix
`K,V`, the verifier samples earlier token positions, and those sampled
positions are anchored by stronger shell checks. The current Lean tree does
not yet prove the semantic theorem connecting those pieces end to end. -/
structure SampledPrefixAudit (α : Type*) (β : Type*) where
  challengedToken : ℕ
  prefixOpenings : List (PrefixKVOpening α β)
  sampledAnchorIndices : List ℕ

/-! ## Layer State (formalized exact-shell slice) -/

/-- Full state of one transformer layer, matching Rust's `LayerTrace`.
    INT8 values and i32 accumulators are uniformly represented as integers.
    Inputs (xAttn, a, xFfn, h) are post-requantization INT8.
    Outputs (q, k, v, attnOut, g, u, ffnOut) are i32 accumulators.

    This is only the currently formalized exact-shell slice. It does NOT yet
    model the wider README shell items such as embeddings, RoPE, RMSNorm,
    final RMSNorm, or LM-head / logits. -/
structure LayerState where
  /-- Input to attention block (INT8) -/
  xAttn : List ℤ
  /-- Wq * xAttn (i32 accumulator) -/
  q : List ℤ
  /-- Wk * xAttn (i32 accumulator) -/
  k : List ℤ
  /-- Wv * xAttn (i32 accumulator) -/
  v : List ℤ
  /-- Attention output vector (INT8) -/
  a : List ℤ
  /-- Wo * a (i32 accumulator) -/
  attnOut : List ℤ
  /-- Input to FFN block = clamp(attnOut) (INT8) -/
  xFfn : List ℤ
  /-- Wg * xFfn (i32 accumulator) -/
  g : List ℤ
  /-- Wu * xFfn (i32 accumulator) -/
  u : List ℤ
  /-- SiLU(g) * u, requantized (INT8) -/
  h : List ℤ
  /-- Wd * h (i32 accumulator) -/
  ffnOut : List ℤ

/-- A LayerState is well-formed when all vector lengths are consistent.
    This matches the Rust invariant that all vectors within a layer
    have dimensions determined by the model config. -/
def LayerState.wellFormed (ls : LayerState) (hiddenDim ffnDim : ℕ)
    (_hH : 0 < hiddenDim) (_hF : 0 < ffnDim) : Prop :=
  ls.xAttn.length = hiddenDim ∧
  ls.q.length = hiddenDim ∧
  ls.attnOut.length = hiddenDim ∧
  ls.xFfn.length = hiddenDim ∧
  ls.g.length = ffnDim ∧
  ls.u.length = ffnDim ∧
  ls.h.length = ffnDim ∧
  ls.ffnOut.length = hiddenDim

/-- Elementwise INT8 requantization of a list of accumulator values. -/
def requantizeList (xs : List ℤ) : List ℤ :=
  xs.map clampI8

@[simp] theorem requantizeList_length (xs : List ℤ) :
    (requantizeList xs).length = xs.length := by
  simp [requantizeList]

/-! ## Cross-Layer Chain Predicate -/

/-- Intra-layer chain: xFfn = clamp(attnOut) element-wise. -/
def intraLayerChainValid (ls : LayerState) : Prop :=
  ∀ (i : Fin ls.attnOut.length) (h : i.val < ls.xFfn.length),
    ls.xFfn.get ⟨i.val, h⟩ = clampI8 (ls.attnOut.get i)

/-- Inter-layer chain: next layer's xAttn = clamp(prev layer's ffnOut). -/
def interLayerChainValid (prev next : LayerState) : Prop :=
  ∀ (i : Fin prev.ffnOut.length) (h : i.val < next.xAttn.length),
    next.xAttn.get ⟨i.val, h⟩ = clampI8 (prev.ffnOut.get i)

/-- If the opened FFN input is literally the requantized attention output,
then the intra-layer chain check holds. -/
theorem intraLayerChainValid_of_eq_requantizeList
    (ls : LayerState)
    (hEq : ls.xFfn = requantizeList ls.attnOut) :
    intraLayerChainValid ls := by
  intro i h
  have hReq : i.val < (requantizeList ls.attnOut).length := by
    simpa [hEq, requantizeList] using h
  calc
    ls.xFfn.get ⟨i.val, h⟩ = (requantizeList ls.attnOut).get ⟨i.val, hReq⟩ := by
      simpa [hEq]
    _ = clampI8 (ls.attnOut.get i) := by
      simpa [requantizeList] using
        (List.getElem_map (f := clampI8) (l := ls.attnOut) (n := i.val) (h := i.isLt))

/-- With matching lengths, the intra-layer chain check is equivalent to the
opened FFN input being the elementwise requantization of the attention output. -/
theorem intraLayerChainValid_eq_requantizeList
    (ls : LayerState)
    (hLen : ls.xFfn.length = ls.attnOut.length)
    (hValid : intraLayerChainValid ls) :
    ls.xFfn = requantizeList ls.attnOut := by
  apply List.ext_getElem
  · simpa [requantizeList] using hLen
  · intro n hnX hnReq
    let i : Fin ls.attnOut.length := ⟨n, by simpa [requantizeList] using hnReq⟩
    have hChain : ls.xFfn[n]'hnX = clampI8 (ls.attnOut.get i) :=
      hValid i hnX
    calc
      ls.xFfn[n]'hnX = clampI8 (ls.attnOut.get i) := hChain
      _ = (requantizeList ls.attnOut)[n]'hnReq := by
        simpa [requantizeList, i] using
          (List.getElem_map_rev (f := clampI8) (l := ls.attnOut) (n := n) (h := i.isLt))

/-- If the next layer's attention input is literally the requantized previous
FFN output, then the inter-layer chain check holds. -/
theorem interLayerChainValid_of_eq_requantizeList
    (prev next : LayerState)
    (hEq : next.xAttn = requantizeList prev.ffnOut) :
    interLayerChainValid prev next := by
  intro i h
  have hReq : i.val < (requantizeList prev.ffnOut).length := by
    simpa [hEq, requantizeList] using h
  calc
    next.xAttn.get ⟨i.val, h⟩ = (requantizeList prev.ffnOut).get ⟨i.val, hReq⟩ := by
      simpa [hEq]
    _ = clampI8 (prev.ffnOut.get i) := by
      simpa [requantizeList] using
        (List.getElem_map (f := clampI8) (l := prev.ffnOut) (n := i.val) (h := i.isLt))

/-- With matching lengths, the inter-layer chain check is equivalent to the
next layer consuming the elementwise requantized previous FFN output. -/
theorem interLayerChainValid_eq_requantizeList
    (prev next : LayerState)
    (hLen : next.xAttn.length = prev.ffnOut.length)
    (hValid : interLayerChainValid prev next) :
    next.xAttn = requantizeList prev.ffnOut := by
  apply List.ext_getElem
  · simpa [requantizeList] using hLen
  · intro n hnX hnReq
    let i : Fin prev.ffnOut.length := ⟨n, by simpa [requantizeList] using hnReq⟩
    have hChain : next.xAttn[n]'hnX = clampI8 (prev.ffnOut.get i) :=
      hValid i hnX
    calc
      next.xAttn[n]'hnX = clampI8 (prev.ffnOut.get i) := hChain
      _ = (requantizeList prev.ffnOut)[n]'hnReq := by
        simpa [requantizeList, i] using
          (List.getElem_map_rev (f := clampI8) (l := prev.ffnOut) (n := n) (h := i.isLt))

/-- SiLU check for one layer: h[i] = computeH(lut, g[i], u[i]). -/
def siluLayerValid (lut : Fin 256 -> ℤ) (ls : LayerState) : Prop :=
  ∀ (i : Fin ls.g.length) (hU : i.val < ls.u.length) (hH : i.val < ls.h.length),
    ls.h.get ⟨i.val, hH⟩ = computeH lut (ls.g.get i) (ls.u.get ⟨i.val, hU⟩)

/-- SiLU check with ±1 tolerance, matching Rust's check_silu.
    This is strictly weaker than siluLayerValid (exact). -/
def siluLayerTolerant (lut : Fin 256 → ℤ) (ls : LayerState) : Prop :=
  ∀ (i : Fin ls.g.length) (hU : i.val < ls.u.length) (hH : i.val < ls.h.length),
    computeH_tolerant lut (ls.g.get i) (ls.u.get ⟨i.val, hU⟩) (ls.h.get ⟨i.val, hH⟩)

/-- Exact SiLU implies tolerant SiLU (the tolerance is strictly weaker). -/
theorem siluLayerValid_implies_tolerant (lut : Fin 256 → ℤ) (ls : LayerState)
    (h : siluLayerValid lut ls) : siluLayerTolerant lut ls := by
  intro i hU hH
  unfold computeH_tolerant
  rw [h i hU hH]
  simp

/-- Full cross-layer chain validity for all layers in a token trace.
    Checks both intra-layer (xFfn = clamp(attnOut)) and inter-layer
    (layer[i+1].xAttn = clamp(layer[i].ffnOut)) chains. -/
def crossLayerChainValid {nLayers : ℕ} (layers : Fin nLayers -> LayerState) : Prop :=
  -- Intra-layer chain: every layer's xFfn = clamp(attnOut)
  (∀ (i : Fin nLayers), intraLayerChainValid (layers i)) ∧
  -- Inter-layer chain: consecutive layers linked by clamp(ffnOut)
  (∀ (i : Fin nLayers) (hi : i.val + 1 < nLayers),
    interLayerChainValid (layers i) (layers ⟨i.val + 1, hi⟩))

/-- A well-formed cross-layer trace determines the requantized bridge values
between the opened attention and FFN subpaths. -/
theorem crossLayerChainValid_requantized_links
    {nLayers hiddenDim ffnDim : ℕ}
    {hH : 0 < hiddenDim} {hF : 0 < ffnDim}
    (layers : Fin nLayers → LayerState)
    (hWell : ∀ i : Fin nLayers, (layers i).wellFormed hiddenDim ffnDim hH hF)
    (hValid : crossLayerChainValid layers) :
    (∀ i : Fin nLayers, (layers i).xFfn = requantizeList (layers i).attnOut) ∧
    (∀ (i : Fin nLayers) (hi : i.val + 1 < nLayers),
      (layers ⟨i.val + 1, hi⟩).xAttn = requantizeList (layers i).ffnOut) := by
  rcases hValid with ⟨hIntra, hInter⟩
  refine ⟨?_, ?_⟩
  · intro i
    rcases hWell i with ⟨_, _, hAttnOut, hXFfn, _, _, _, _⟩
    exact intraLayerChainValid_eq_requantizeList
      (layers i) (hXFfn.trans hAttnOut.symm) (hIntra i)
  · intro i hi
    rcases hWell i with ⟨_, _, _, _, _, _, _, hFfnOut⟩
    rcases hWell ⟨i.val + 1, hi⟩ with ⟨hXAttn, _, _, _, _, _, _, _⟩
    exact interLayerChainValid_eq_requantizeList
      (layers i) (layers ⟨i.val + 1, hi⟩)
      (hXAttn.trans hFfnOut.symm) (hInter i hi)

/-- Two intra-layer traces with the same opened attention output must have the
same requantized FFN input, provided both satisfy the chain check. -/
theorem xFfn_eq_of_attnOut_eq
    {hiddenDim ffnDim : ℕ}
    {hH : 0 < hiddenDim} {hF : 0 < ffnDim}
    {ls₁ ls₂ : LayerState}
    (hWell₁ : ls₁.wellFormed hiddenDim ffnDim hH hF)
    (hWell₂ : ls₂.wellFormed hiddenDim ffnDim hH hF)
    (hValid₁ : intraLayerChainValid ls₁)
    (hValid₂ : intraLayerChainValid ls₂)
    (hAttn : ls₁.attnOut = ls₂.attnOut) :
    ls₁.xFfn = ls₂.xFfn := by
  rcases hWell₁ with ⟨_, _, hAttnOut₁, hXFfn₁, _, _, _, _⟩
  rcases hWell₂ with ⟨_, _, hAttnOut₂, hXFfn₂, _, _, _, _⟩
  rw [intraLayerChainValid_eq_requantizeList
        ls₁ (hXFfn₁.trans hAttnOut₁.symm) hValid₁,
      intraLayerChainValid_eq_requantizeList
        ls₂ (hXFfn₂.trans hAttnOut₂.symm) hValid₂,
      hAttn]

/-- Two consecutive-layer openings with the same previous FFN output must have
the same next attention input, provided both satisfy the inter-layer check. -/
theorem xAttn_eq_of_prevFfnOut_eq
    {hiddenDim ffnDim : ℕ}
    {hH : 0 < hiddenDim} {hF : 0 < ffnDim}
    {prev₁ prev₂ next₁ next₂ : LayerState}
    (hPrev₁ : prev₁.wellFormed hiddenDim ffnDim hH hF)
    (hPrev₂ : prev₂.wellFormed hiddenDim ffnDim hH hF)
    (hNext₁ : next₁.wellFormed hiddenDim ffnDim hH hF)
    (hNext₂ : next₂.wellFormed hiddenDim ffnDim hH hF)
    (hValid₁ : interLayerChainValid prev₁ next₁)
    (hValid₂ : interLayerChainValid prev₂ next₂)
    (hFfnOut : prev₁.ffnOut = prev₂.ffnOut) :
    next₁.xAttn = next₂.xAttn := by
  rcases hPrev₁ with ⟨_, _, _, _, _, _, _, hFfnOut₁⟩
  rcases hPrev₂ with ⟨_, _, _, _, _, _, _, hFfnOut₂⟩
  rcases hNext₁ with ⟨hXAttn₁, _, _, _, _, _, _, _⟩
  rcases hNext₂ with ⟨hXAttn₂, _, _, _, _, _, _, _⟩
  rw [interLayerChainValid_eq_requantizeList
        prev₁ next₁ (hXAttn₁.trans hFfnOut₁.symm) hValid₁,
      interLayerChainValid_eq_requantizeList
        prev₂ next₂ (hXAttn₂.trans hFfnOut₂.symm) hValid₂,
      hFfnOut]

/-- Intra-layer chain validity WITH explicit length requirement.
    When lengths match, the predicate is non-vacuous. -/
theorem intraLayerChainValid_nonvacuous (ls : LayerState)
    (hLen : ls.xFfn.length = ls.attnOut.length)
    (hValid : intraLayerChainValid ls) :
    ∀ (i : Fin ls.attnOut.length),
      ls.xFfn.get ⟨i.val, by omega⟩ = clampI8 (ls.attnOut.get i) := by
  intro i
  exact hValid i (by omega)

/-! ## Cross-Token Chain Predicate -/

/-! ## SiLU Validity Across All Layers -/

/-- SiLU check passes for every layer in a token trace. -/
def siluAllLayersValid {nLayers : ℕ}
    (lut : Fin 256 -> ℤ) (layers : Fin nLayers -> LayerState) : Prop :=
  ∀ (i : Fin nLayers), siluLayerValid lut (layers i)

end VerifiedInference
