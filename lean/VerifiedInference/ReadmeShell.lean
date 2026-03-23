import VerifiedInference.Protocol
import VerifiedInference.Freivalds
import VerifiedInference.WeightBinding
import VerifiedInference.Requantization
import VerifiedInference.SiLU
import VerifiedInference.MultiVector

/-!
# README Exact Shell

This module models the exact / deterministic portion of the canonical
`README.md` protocol.

The README's exact shell consists of:

- input embeddings
- RMSNorm / final RMSNorm
- INT8 weight matmuls checked by Freivalds
- exact requantization bridges
- exact RoPE recomputation
- exact residual adds
- exact SiLU(gate) ⊙ up
- LM-head matmul

The current Lean code proves the algebraic Freivalds and integer-side
determinism facts that underpin this shell. It does not yet formalize the
full tensor semantics of every shell stage end to end.
-/

namespace VerifiedInference

/-- Weight-bearing exact-shell operations in the canonical README path. -/
inductive ReadmeMatmulType
  | Wq
  | Wk
  | Wv
  | Wo
  | Wgate
  | Wup
  | Wdown
  | LmHead
  deriving DecidableEq, Repr

/-- Number of Freivalds-checked matrix multiplications in the README exact
shell for a model with `nLayers` transformer layers. -/
def exactShellMatmulCount (nLayers : Nat) : Nat :=
  7 * nLayers + 1

/-- Abstract embedding table used by the README's exact input-embedding step. -/
structure EmbeddingTable where
  lookup : Nat → List Int

/-- Canonical recomputation functions treated as exact in the README shell. -/
abbrev RMSNormFn := List Int → List Int
abbrev RopeFn := Nat → List Int → List Int

/-- Exact residual add in the quantized shell. -/
def residualAdd (lhs rhs : List Int) : List Int :=
  List.zipWith (fun x y => x + y) lhs rhs

/-- README-style exact shell checks for one opened layer / token slice. -/
structure ExactShellChecks where
  embeddingExact : Prop
  attnRmsNormExact : Prop
  qFreivalds : Prop
  kFreivalds : Prop
  vFreivalds : Prop
  qRequantExact : Prop
  kRequantExact : Prop
  vRequantExact : Prop
  ropeQExact : Prop
  ropeKExact : Prop
  postAttentionBound : Prop
  woFreivalds : Prop
  woRequantExact : Prop
  residualAfterAttentionExact : Prop
  ffnRmsNormExact : Prop
  gateFreivalds : Prop
  upFreivalds : Prop
  gateRequantExact : Prop
  upRequantExact : Prop
  siluExact : Prop
  downFreivalds : Prop
  downRequantExact : Prop
  residualAfterFfnExact : Prop
  finalRmsNormExact : Prop
  lmHeadFreivalds : Prop

/-- Conjunction of all README exact-shell checks. -/
def ExactShellChecks.holds (checks : ExactShellChecks) : Prop :=
  checks.embeddingExact ∧
  checks.attnRmsNormExact ∧
  checks.qFreivalds ∧
  checks.kFreivalds ∧
  checks.vFreivalds ∧
  checks.qRequantExact ∧
  checks.kRequantExact ∧
  checks.vRequantExact ∧
  checks.ropeQExact ∧
  checks.ropeKExact ∧
  checks.postAttentionBound ∧
  checks.woFreivalds ∧
  checks.woRequantExact ∧
  checks.residualAfterAttentionExact ∧
  checks.ffnRmsNormExact ∧
  checks.gateFreivalds ∧
  checks.upFreivalds ∧
  checks.gateRequantExact ∧
  checks.upRequantExact ∧
  checks.siluExact ∧
  checks.downFreivalds ∧
  checks.downRequantExact ∧
  checks.residualAfterFfnExact ∧
  checks.finalRmsNormExact ∧
  checks.lmHeadFreivalds

/-- The exact shell's weight-bearing Freivalds checks, collected separately
for the README's model-identity claim on an opened slice. -/
def ExactShellChecks.weightFreivaldsHold (checks : ExactShellChecks) : Prop :=
  checks.qFreivalds ∧
  checks.kFreivalds ∧
  checks.vFreivalds ∧
  checks.woFreivalds ∧
  checks.gateFreivalds ∧
  checks.upFreivalds ∧
  checks.downFreivalds ∧
  checks.lmHeadFreivalds

/-- Embedding lookup is deterministic because the table is a pure function. -/
theorem embedding_lookup_deterministic (table : EmbeddingTable) (token : Nat) :
    ∀ out₁ out₂ : List Int,
      out₁ = table.lookup token →
      out₂ = table.lookup token →
      out₁ = out₂ := by
  intro out₁ out₂ h₁ h₂
  rw [h₁, h₂]

/-- Canonical RMSNorm recomputation is deterministic when modeled as a pure function. -/
theorem rmsNorm_deterministic (norm : RMSNormFn) (x : List Int) :
    ∀ y₁ y₂ : List Int, y₁ = norm x → y₂ = norm x → y₁ = y₂ := by
  intro y₁ y₂ h₁ h₂
  rw [h₁, h₂]

/-- Exact RoPE recomputation is deterministic for a fixed position and input. -/
theorem rope_deterministic (rope : RopeFn) (pos : Nat) (x : List Int) :
    ∀ y₁ y₂ : List Int, y₁ = rope pos x → y₂ = rope pos x → y₁ = y₂ := by
  intro y₁ y₂ h₁ h₂
  rw [h₁, h₂]

/-- Residual addition is deterministic. -/
theorem residual_add_deterministic (lhs rhs : List Int) :
    ∀ out₁ out₂ : List Int,
      out₁ = residualAdd lhs rhs →
      out₂ = residualAdd lhs rhs →
      out₁ = out₂ := by
  intro out₁ out₂ h₁ h₂
  rw [h₁, h₂]

/-- Any opened matrix multiplication in the README shell inherits the standard
Freivalds soundness bound. This covers the seven per-layer matrices and the
LM head alike. -/
theorem readme_matmul_freivalds_sound
    {p : Nat} [Fact (Nat.Prime p)]
    {m n : Nat}
    (W : Matrix (Fin m) (Fin n) (ZMod p))
    (x : Fin n → ZMod p)
    (z : Fin m → ZMod p)
    (hCheat : z ≠ W.mulVec x)
    (hm : 0 < m) :
    (Finset.univ.filter fun r : Fin m → ZMod p =>
      dotProduct (W.transpose.mulVec r) x = dotProduct r z).card * p
      ≤ Fintype.card (Fin m → ZMod p) :=
  freivalds_sound W x z hCheat hm

/-- If each exact-shell matrix check has bad-set bound `badCard * p ≤ total`,
then the whole README exact shell has amplified bound
`badCard^(7*nLayers+1) * p^(7*nLayers+1) ≤ total^(7*nLayers+1)`. -/
theorem exact_shell_freivalds_amplification
    (totalVectors p badCard nLayers : Nat)
    (hSingle : badCard * p ≤ totalVectors) :
    badCard ^ exactShellMatmulCount nLayers * p ^ exactShellMatmulCount nLayers
      ≤ totalVectors ^ exactShellMatmulCount nLayers := by
  unfold exactShellMatmulCount
  exact amplification_inductive totalVectors p badCard hSingle (7 * nLayers + 1)

/-- If every exact-shell check proposition is available, the README exact shell
holds for the opened slice. -/
theorem exact_shell_checks_hold (checks : ExactShellChecks)
    (hEmbedding : checks.embeddingExact)
    (hAttnNorm : checks.attnRmsNormExact)
    (hQ : checks.qFreivalds)
    (hK : checks.kFreivalds)
    (hV : checks.vFreivalds)
    (hQReq : checks.qRequantExact)
    (hKReq : checks.kRequantExact)
    (hVReq : checks.vRequantExact)
    (hRopeQ : checks.ropeQExact)
    (hRopeK : checks.ropeKExact)
    (hPostAttn : checks.postAttentionBound)
    (hWo : checks.woFreivalds)
    (hWoReq : checks.woRequantExact)
    (hResAttn : checks.residualAfterAttentionExact)
    (hFfnNorm : checks.ffnRmsNormExact)
    (hGate : checks.gateFreivalds)
    (hUp : checks.upFreivalds)
    (hGateReq : checks.gateRequantExact)
    (hUpReq : checks.upRequantExact)
    (hSiLU : checks.siluExact)
    (hDown : checks.downFreivalds)
    (hDownReq : checks.downRequantExact)
    (hResFfn : checks.residualAfterFfnExact)
    (hFinalNorm : checks.finalRmsNormExact)
    (hLmHead : checks.lmHeadFreivalds) :
    checks.holds := by
  unfold ExactShellChecks.holds
  exact ⟨hEmbedding, hAttnNorm, hQ, hK, hV, hQReq, hKReq, hVReq, hRopeQ, hRopeK,
    hPostAttn, hWo, hWoReq, hResAttn, hFfnNorm, hGate, hUp, hGateReq, hUpReq,
    hSiLU, hDown, hDownReq, hResFfn, hFinalNorm, hLmHead⟩

/-- The README's exact shell implies exact model-identity auditing for the
opened slice once all of its checks hold. This theorem intentionally keeps the
claim propositional: the exact shell is the part of the protocol with the hard
exact guarantee, while weaker attention/KV layers are handled elsewhere. -/
theorem exact_shell_implies_model_identity_on_opened_slice
    (checks : ExactShellChecks)
    (hShell : checks.holds) :
    checks.weightFreivaldsHold := by
  unfold ExactShellChecks.weightFreivaldsHold
  unfold ExactShellChecks.holds at hShell
  rcases hShell with ⟨_, _, hQ, hK, hV, _, _, _, _, _, _, hWo, _, _, _, hGate, hUp,
    _, _, _, hDown, _, _, _, hLm⟩
  exact ⟨hQ, hK, hV, hWo, hGate, hUp, hDown, hLm⟩

end VerifiedInference
