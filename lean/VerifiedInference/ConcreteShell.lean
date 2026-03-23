import VerifiedInference.Basic
import VerifiedInference.ReadmeShell

/-!
# Concrete Shell Checks

This module connects concrete Freivalds checks (with actual `freivaldsAccepts`
predicates) to the abstract `ExactShellChecks` from `ReadmeShell.lean`.

It provides:
- `ConcreteFreivaldsCheck`: bundles the four vectors needed for one Freivalds check
- `ConcreteLayerShell`: groups the 7 per-layer concrete checks
- `buildAbstractShellChecks`: constructs an `ExactShellChecks` from concrete data
- Theorems linking concrete checks to `weightFreivaldsHold` and `holds`
-/

namespace VerifiedInference

/-- Bundle the four vectors needed for a single Freivalds check:
    `v · x = r · z`. -/
structure ConcreteFreivaldsCheck (p : ℕ) (m n : ℕ) where
  v : FpVec p n
  x : FpVec p n
  r : FpVec p m
  z : FpVec p m

/-- The Freivalds check passes when `freivaldsAccepts` holds on the bundled vectors. -/
def ConcreteFreivaldsCheck.passes {p m n : ℕ} (c : ConcreteFreivaldsCheck p m n) : Prop :=
  freivaldsAccepts c.v c.x c.r c.z

/-- Groups the 7 concrete Freivalds checks for one transformer layer. -/
structure ConcreteLayerShell (p : ℕ) (hiddenDim ffnDim : ℕ) where
  wq    : ConcreteFreivaldsCheck p hiddenDim hiddenDim
  wk    : ConcreteFreivaldsCheck p hiddenDim hiddenDim
  wv    : ConcreteFreivaldsCheck p hiddenDim hiddenDim
  wo    : ConcreteFreivaldsCheck p hiddenDim hiddenDim
  wgate : ConcreteFreivaldsCheck p ffnDim hiddenDim
  wup   : ConcreteFreivaldsCheck p ffnDim hiddenDim
  wdown : ConcreteFreivaldsCheck p hiddenDim ffnDim

/-- Conjunction of all 7 per-layer Freivalds checks. -/
def ConcreteLayerShell.allFreivaldsPass {p hiddenDim ffnDim : ℕ}
    (layer : ConcreteLayerShell p hiddenDim ffnDim) : Prop :=
  layer.wq.passes ∧
  layer.wk.passes ∧
  layer.wv.passes ∧
  layer.wo.passes ∧
  layer.wgate.passes ∧
  layer.wup.passes ∧
  layer.wdown.passes

/-- Build an `ExactShellChecks` from a concrete layer shell and LM-head check,
    plus `Prop` values for every deterministic check. The Freivalds fields are
    filled directly from `.passes` on the concrete checks. -/
def buildAbstractShellChecks
    {p hiddenDim ffnDim : ℕ}
    (layer    : ConcreteLayerShell p hiddenDim ffnDim)
    (lmHead   : ConcreteFreivaldsCheck p hiddenDim hiddenDim)
    -- deterministic checks
    (embeddingExact            : Prop)
    (attnRmsNormExact          : Prop)
    (qRequantExact             : Prop)
    (kRequantExact             : Prop)
    (vRequantExact             : Prop)
    (ropeQExact                : Prop)
    (ropeKExact                : Prop)
    (postAttentionBound        : Prop)
    (woRequantExact            : Prop)
    (residualAfterAttentionExact : Prop)
    (ffnRmsNormExact           : Prop)
    (gateRequantExact          : Prop)
    (upRequantExact            : Prop)
    (siluExact                 : Prop)
    (downRequantExact          : Prop)
    (residualAfterFfnExact     : Prop)
    (finalRmsNormExact         : Prop)
    : ExactShellChecks :=
  { embeddingExact              := embeddingExact
    attnRmsNormExact            := attnRmsNormExact
    qFreivalds                  := layer.wq.passes
    kFreivalds                  := layer.wk.passes
    vFreivalds                  := layer.wv.passes
    qRequantExact               := qRequantExact
    kRequantExact               := kRequantExact
    vRequantExact               := vRequantExact
    ropeQExact                  := ropeQExact
    ropeKExact                  := ropeKExact
    postAttentionBound          := postAttentionBound
    woFreivalds                 := layer.wo.passes
    woRequantExact              := woRequantExact
    residualAfterAttentionExact := residualAfterAttentionExact
    ffnRmsNormExact             := ffnRmsNormExact
    gateFreivalds               := layer.wgate.passes
    upFreivalds                 := layer.wup.passes
    gateRequantExact            := gateRequantExact
    upRequantExact              := upRequantExact
    siluExact                   := siluExact
    downFreivalds               := layer.wdown.passes
    downRequantExact            := downRequantExact
    residualAfterFfnExact       := residualAfterFfnExact
    finalRmsNormExact           := finalRmsNormExact
    lmHeadFreivalds             := lmHead.passes }

/-- When all 7 layer Freivalds checks pass and the LM-head check passes,
    `weightFreivaldsHold` holds on the assembled `ExactShellChecks`. -/
theorem concrete_implies_weight_freivalds
    {p hiddenDim ffnDim : ℕ}
    (layer  : ConcreteLayerShell p hiddenDim ffnDim)
    (lmHead : ConcreteFreivaldsCheck p hiddenDim hiddenDim)
    (hLayer : layer.allFreivaldsPass)
    (hLm    : lmHead.passes)
    -- deterministic check props (existentially supplied)
    (embeddingExact            : Prop)
    (attnRmsNormExact          : Prop)
    (qRequantExact             : Prop)
    (kRequantExact             : Prop)
    (vRequantExact             : Prop)
    (ropeQExact                : Prop)
    (ropeKExact                : Prop)
    (postAttentionBound        : Prop)
    (woRequantExact            : Prop)
    (residualAfterAttentionExact : Prop)
    (ffnRmsNormExact           : Prop)
    (gateRequantExact          : Prop)
    (upRequantExact            : Prop)
    (siluExact                 : Prop)
    (downRequantExact          : Prop)
    (residualAfterFfnExact     : Prop)
    (finalRmsNormExact         : Prop)
    : (buildAbstractShellChecks layer lmHead
        embeddingExact attnRmsNormExact
        qRequantExact kRequantExact vRequantExact
        ropeQExact ropeKExact postAttentionBound
        woRequantExact residualAfterAttentionExact
        ffnRmsNormExact gateRequantExact upRequantExact
        siluExact downRequantExact residualAfterFfnExact
        finalRmsNormExact).weightFreivaldsHold := by
  unfold ExactShellChecks.weightFreivaldsHold buildAbstractShellChecks
  unfold ConcreteLayerShell.allFreivaldsPass at hLayer
  obtain ⟨hQ, hK, hV, hWo, hGate, hUp, hDown⟩ := hLayer
  exact ⟨hQ, hK, hV, hWo, hGate, hUp, hDown, hLm⟩

/-- When ALL concrete checks pass (both Freivalds and deterministic),
    `ExactShellChecks.holds` is true for the assembled shell. -/
theorem concrete_implies_abstract_shell
    {p hiddenDim ffnDim : ℕ}
    (layer  : ConcreteLayerShell p hiddenDim ffnDim)
    (lmHead : ConcreteFreivaldsCheck p hiddenDim hiddenDim)
    (hLayer : layer.allFreivaldsPass)
    (hLm    : lmHead.passes)
    -- deterministic checks
    (embeddingExact            : Prop)
    (attnRmsNormExact          : Prop)
    (qRequantExact             : Prop)
    (kRequantExact             : Prop)
    (vRequantExact             : Prop)
    (ropeQExact                : Prop)
    (ropeKExact                : Prop)
    (postAttentionBound        : Prop)
    (woRequantExact            : Prop)
    (residualAfterAttentionExact : Prop)
    (ffnRmsNormExact           : Prop)
    (gateRequantExact          : Prop)
    (upRequantExact            : Prop)
    (siluExact                 : Prop)
    (downRequantExact          : Prop)
    (residualAfterFfnExact     : Prop)
    (finalRmsNormExact         : Prop)
    (hEmbedding  : embeddingExact)
    (hAttnNorm   : attnRmsNormExact)
    (hQReq       : qRequantExact)
    (hKReq       : kRequantExact)
    (hVReq       : vRequantExact)
    (hRopeQ      : ropeQExact)
    (hRopeK      : ropeKExact)
    (hPostAttn   : postAttentionBound)
    (hWoReq      : woRequantExact)
    (hResAttn    : residualAfterAttentionExact)
    (hFfnNorm    : ffnRmsNormExact)
    (hGateReq    : gateRequantExact)
    (hUpReq      : upRequantExact)
    (hSiLU       : siluExact)
    (hDownReq    : downRequantExact)
    (hResFfn     : residualAfterFfnExact)
    (hFinalNorm  : finalRmsNormExact)
    : (buildAbstractShellChecks layer lmHead
        embeddingExact attnRmsNormExact
        qRequantExact kRequantExact vRequantExact
        ropeQExact ropeKExact postAttentionBound
        woRequantExact residualAfterAttentionExact
        ffnRmsNormExact gateRequantExact upRequantExact
        siluExact downRequantExact residualAfterFfnExact
        finalRmsNormExact).holds := by
  unfold ExactShellChecks.holds buildAbstractShellChecks
  unfold ConcreteLayerShell.allFreivaldsPass at hLayer
  obtain ⟨hQ, hK, hV, hWo, hGate, hUp, hDown⟩ := hLayer
  exact ⟨hEmbedding, hAttnNorm, hQ, hK, hV, hQReq, hKReq, hVReq, hRopeQ, hRopeK,
    hPostAttn, hWo, hWoReq, hResAttn, hFfnNorm, hGate, hUp, hGateReq, hUpReq,
    hSiLU, hDown, hDownReq, hResFfn, hFinalNorm, hLm⟩

end VerifiedInference
