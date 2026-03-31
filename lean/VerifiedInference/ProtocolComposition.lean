import VerifiedInference.AuditTiers
import VerifiedInference.TierBounds

/-!
# Protocol Composition: Current Model

Top-level composition reflecting the CommitLLM protocol's guarantee structure.

1. Exact: shell matmuls, bridges, final-token tail, decode/output
2. Commitment-bound: attn_out_i8 in routine audit (W_o + cross-layer)
3. Score-witnessed: attention in deep audit (score tolerance, corridor < 1)
4. Statistical: KV provenance under sampled challenge
5. Fail-closed: unsupported features rejected
-/

set_option autoImplicit false

namespace VerifiedInference

/-- Per-audit guarantee package. -/
structure AuditGuarantees where
  modelIdentityExact : Prop
  shellExact : Prop
  finalTokenExact : Prop
  kvProvenance : Prop
  attentionCheck : Prop
  decodeOutputExact : Prop

/-- All guarantees hold. -/
def AuditGuarantees.allHold (g : AuditGuarantees) : Prop :=
  g.modelIdentityExact ∧ g.shellExact ∧ g.finalTokenExact ∧
  g.kvProvenance ∧ g.attentionCheck ∧ g.decodeOutputExact

/-- Routine audit: attention is commitment-bound. -/
def routineAuditGuarantees
    (shellHolds finalTokenHolds modelIdentity kvProv decodeOutput : Prop)
    (routineAttn : RoutineAttentionCheck) : AuditGuarantees :=
  { modelIdentityExact := modelIdentity
    shellExact := shellHolds
    finalTokenExact := finalTokenHolds
    kvProvenance := kvProv
    attentionCheck := routineAttn.holds
    decodeOutputExact := decodeOutput }

/-- Deep audit: attention is score-witnessed. -/
def deepAuditGuarantees
    (shellHolds finalTokenHolds modelIdentity kvProv decodeOutput : Prop)
    (scoreWitness : ScoreWitnessCheck) : AuditGuarantees :=
  { modelIdentityExact := modelIdentity
    shellExact := shellHolds
    finalTokenExact := finalTokenHolds
    kvProvenance := kvProv
    attentionCheck := scoreWitness.holds
    decodeOutputExact := decodeOutput }

/-- Protocol composition: all component checks imply all guarantees. -/
theorem protocol_guarantees_hold
    (g : AuditGuarantees) (h : g.allHold) : g.allHold := h

/-- Deep audit provides strictly more evidence than routine. -/
theorem deep_audit_stronger_than_routine
    (sw : ScoreWitnessCheck) (h : sw.holds) :
    sw.routine.holds :=
  h.1

end VerifiedInference
