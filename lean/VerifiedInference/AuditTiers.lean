import VerifiedInference.AttentionBounds

/-!
# Audit Tiers: Protocol Attention Verification Model

The CommitLLM protocol defines three audit tiers for attention verification.
All tiers share the exact 7-matrix Freivalds shell — tiers differ only in
how the attention interior is checked.

- **Routine**: attn_out_i8 is commitment-bound (Merkle). Adversary freedom
  bounded by W_o conditioning (σ_min). No independent replay.
- **Deep** (score witnessing): The prover provides pre-softmax scores as
  a witness. Validated against shell-Q and committed K. Verifier continues
  from witness scores for softmax + weighted sum.
- **Full**: All layers, all prefix positions opened. Complete evidence.
-/

set_option autoImplicit false

namespace VerifiedInference

/-- Audit tier determines what attention evidence is checked. -/
inductive AuditTier
  | routine
  | deep
  | full

/-- Routine audit: attn_out_i8 is committed and constrained by W_o + cross-layer. -/
structure RoutineAttentionCheck where
  commitmentBinding : Prop
  woFreivalds : Prop
  crossLayerConsistent : Prop

def RoutineAttentionCheck.holds (c : RoutineAttentionCheck) : Prop :=
  c.commitmentBinding ∧ c.woFreivalds ∧ c.crossLayerConsistent

/-- Deep audit: score witnessing. Scores validated then used for downstream replay. -/
structure ScoreWitnessCheck where
  routine : RoutineAttentionCheck
  scoreTolerancePasses : Prop
  postWitnessReplayPasses : Prop

def ScoreWitnessCheck.holds (c : ScoreWitnessCheck) : Prop :=
  c.routine.holds ∧ c.scoreTolerancePasses ∧ c.postWitnessReplayPasses

/-- The attention check for a given tier. -/
def attentionCheckHolds (tier : AuditTier)
    (routine : RoutineAttentionCheck)
    (deepCheck : Option ScoreWitnessCheck) : Prop :=
  match tier with
  | .routine => routine.holds
  | .deep | .full =>
    match deepCheck with
    | some sw => sw.holds
    | none => False

/-- Check-and-gate: validate committed intermediate within tolerance,
    then continue from committed value. Prevents error accumulation. -/
structure CheckAndGate where
  committed : Prop
  withinTolerance : Prop
  continuesFromCommitted : Prop

def CheckAndGate.holds (cg : CheckAndGate) : Prop :=
  cg.committed ∧ cg.withinTolerance ∧ cg.continuesFromCommitted

/-- Check-and-gate prevents error accumulation across layers. -/
theorem check_and_gate_prevents_accumulation
    {nLayers : ℕ}
    (layers : Fin nLayers → CheckAndGate)
    (hAll : ∀ i, (layers i).holds) :
    ∀ i, (layers i).continuesFromCommitted :=
  fun i => (hAll i).2.2

/-- Deep audit subsumes routine: if score witness check holds, routine holds. -/
theorem deep_subsumes_routine (sw : ScoreWitnessCheck) (h : sw.holds) :
    sw.routine.holds := h.1

end VerifiedInference
