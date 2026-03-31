import VerifiedInference.AttentionBounds
import VerifiedInference.AuditTiers

/-!
# Tier-Specific Attention Bounds

Connects analytical corridor bounds to audit tiers.
- Deep audit (score witnessing) achieves ≤1 corridor
- Routine audit uses commitment binding, not replay
- Pure QKV-only replay cannot achieve ≤1
-/

set_option autoImplicit false

namespace VerifiedInference

/-- Deep audit with score witnessing achieves L-inf ≤ 1. -/
theorem deep_audit_achieves_leq_one
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hbv : p.bvOverScaleA ≤ 127) :
    corridorBound p .committedScores < 1 :=
  committedScores_achieves_leq_one p hgpu hbv

/-- Routine audit uses commitment binding, not replay corridor. -/
theorem routine_audit_is_commitment_bound
    (check : RoutineAttentionCheck)
    (hCheck : check.holds) :
    check.commitmentBinding ∧ check.woFreivalds ∧ check.crossLayerConsistent :=
  hCheck

/-- Replay-only (no committed intermediates) is insufficient for ≤1. -/
theorem replay_only_insufficient
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hcr : p.cRope = 5)
    (hbv : 127 ≤ p.bvOverScaleA)
    (hs : 1 ≤ p.sMax) :
    1 ≤ corridorBound p .qkvAccOnly :=
  qkvOnly_exceeds_one p hgpu hcr hbv hs

end VerifiedInference
