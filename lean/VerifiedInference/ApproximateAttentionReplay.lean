import Mathlib.Data.Finset.Card

import VerifiedInference.Protocol

/-!
# Approximate Attention Replay

This module models the README's attention-replay layer.

Unlike the exact shell, attention replay is not exact: the verifier
recomputes attention in FP64 from shell-verified `Q` and committed prefix
`K,V`, quantizes the replayed output, and accepts within an empirically
calibrated corridor. The theorems here therefore formalize only
consistency-within-a-corridor statements, not exact equality.
-/

namespace VerifiedInference

/-- Acceptance corridor for approximate replay, measured as a maximum number of
INT8 positions allowed to disagree with the committed post-attention output. -/
structure ReplayCorridor where
  maxDisagreements : Nat

/-- The set of coordinates where replayed and committed outputs differ. -/
def disagreementSet {n : Nat}
    (reference committed : Fin n → Int) : Finset (Fin n) :=
  Finset.univ.filter (fun i => reference i ≠ committed i)

/-- Number of differing INT8 coordinates. -/
def disagreementCount {n : Nat}
    (reference committed : Fin n → Int) : Nat :=
  (disagreementSet reference committed).card

/-- Approximate replay acceptance: the committed output falls within the
configured disagreement corridor relative to the replayed output. -/
def approximateReplayAccepts {n : Nat}
    (corridor : ReplayCorridor)
    (reference committed : Fin n → Int) : Prop :=
  disagreementCount reference committed ≤ corridor.maxDisagreements

/-- Quantized agreement outside an explicit set of unstable coordinates. -/
def quantizedAgreementOutside {n : Nat}
    (unstable : Finset (Fin n))
    (reference committed : Fin n → Int) : Prop :=
  ∀ i : Fin n, i ∉ unstable → reference i = committed i

/-- Witness for the README's FP16→FP64 replay story:
the committed output is the honest quantized serving output, and the verifier's
replay can differ only on an explicit set of unstable coordinates. -/
structure ApproximateReplayWitness (n : Nat) where
  replayed : Fin n → Int
  honestServed : Fin n → Int
  committed : Fin n → Int
  unstable : Finset (Fin n)
  committedMatchesServed : committed = honestServed
  replayMatchesServedOutside : quantizedAgreementOutside unstable replayed honestServed

/-- Explicit assumption boundary for the README replay layer.

This is the piece the protocol does not try to prove from first principles:
measurements or calibration establish that an honest serving implementation
differs from the verifier's FP64 replay only on a bounded set of unstable
coordinates. -/
structure ReplayCalibrationAssumption (n : Nat) where
  corridor : ReplayCorridor
  replayed : Fin n → Int
  honestServed : Fin n → Int
  committed : Fin n → Int
  unstable : Finset (Fin n)
  committedMatchesServed : committed = honestServed
  replayMatchesServedOutside : quantizedAgreementOutside unstable replayed honestServed
  unstableWithinCorridor : unstable.card ≤ corridor.maxDisagreements

/-- A replay calibration assumption packages directly into the README replay
witness used by the top-level protocol theorem. -/
def ReplayCalibrationAssumption.toWitness
    {n : Nat}
    (a : ReplayCalibrationAssumption n) :
    ApproximateReplayWitness n :=
  { replayed := a.replayed
    honestServed := a.honestServed
    committed := a.committed
    unstable := a.unstable
    committedMatchesServed := a.committedMatchesServed
    replayMatchesServedOutside := a.replayMatchesServedOutside }

/-- Exact agreement is always accepted, regardless of the corridor. -/
theorem approximate_replay_accepts_of_equal {n : Nat}
    (corridor : ReplayCorridor)
    (reference committed : Fin n → Int)
    (hEq : reference = committed) :
    approximateReplayAccepts corridor reference committed := by
  unfold approximateReplayAccepts disagreementCount disagreementSet
  subst hEq
  simp

/-- Enlarging the corridor preserves acceptance. -/
theorem approximate_replay_monotone {n : Nat}
    (small large : ReplayCorridor)
    (reference committed : Fin n → Int)
    (hLe : small.maxDisagreements ≤ large.maxDisagreements)
    (hAccept : approximateReplayAccepts small reference committed) :
    approximateReplayAccepts large reference committed := by
  unfold approximateReplayAccepts at *
  exact Nat.le_trans hAccept hLe

/-- Any disagreement must lie in the explicit unstable set if reference and
committed outputs agree everywhere else. -/
theorem disagreementSet_subset_of_agreementOutside {n : Nat}
    (unstable : Finset (Fin n))
    (reference committed : Fin n → Int)
    (hAgree : quantizedAgreementOutside unstable reference committed) :
    disagreementSet reference committed ⊆ unstable := by
  intro i hi
  by_contra hNotIn
  have hNe : reference i ≠ committed i := by
    unfold disagreementSet at hi
    simp at hi
    exact hi
  exact hNe (hAgree i hNotIn)

/-- If quantized FP64 replay differs from the honest served output only on an
explicit set of unstable coordinates whose cardinality fits inside the
acceptance corridor, the README replay check accepts. -/
theorem approximate_replay_accepts_of_agreementOutside {n : Nat}
    (corridor : ReplayCorridor)
    (reference honestServed committed : Fin n → Int)
    (unstable : Finset (Fin n))
    (hCommitted : committed = honestServed)
    (hAgree : quantizedAgreementOutside unstable reference honestServed)
    (hBound : unstable.card ≤ corridor.maxDisagreements) :
    approximateReplayAccepts corridor reference committed := by
  unfold approximateReplayAccepts disagreementCount
  have hSubset :
      disagreementSet reference honestServed ⊆ unstable :=
    disagreementSet_subset_of_agreementOutside unstable reference honestServed hAgree
  rw [hCommitted]
  exact Nat.le_trans (Finset.card_le_card hSubset) hBound

/-- The packaged replay witness suffices to establish approximate replay
acceptance once the unstable set fits inside the corridor. -/
theorem ApproximateReplayWitness.accepts
    {n : Nat}
    (w : ApproximateReplayWitness n)
    (corridor : ReplayCorridor)
    (hBound : w.unstable.card ≤ corridor.maxDisagreements) :
    approximateReplayAccepts corridor w.replayed w.committed := by
  exact approximate_replay_accepts_of_agreementOutside
    corridor w.replayed w.honestServed w.committed w.unstable
    w.committedMatchesServed w.replayMatchesServedOutside hBound

/-- An explicit replay calibration assumption suffices to justify replay
acceptance. This keeps the floating-point boundary assumption named and local. -/
theorem ReplayCalibrationAssumption.accepts
    {n : Nat}
    (a : ReplayCalibrationAssumption n) :
    approximateReplayAccepts a.corridor a.replayed a.committed := by
  exact (a.toWitness).accepts a.corridor a.unstableWithinCorridor

/-- Acceptance proves only bounded disagreement with the replayed output on the
committed prefix, not semantic correctness of unsampled earlier-token state. -/
theorem approximate_replay_is_committed_prefix_consistency {n : Nat}
    (corridor : ReplayCorridor)
    (reference committed : Fin n → Int)
    (hAccept : approximateReplayAccepts corridor reference committed) :
    disagreementCount reference committed ≤ corridor.maxDisagreements :=
  hAccept

end VerifiedInference
