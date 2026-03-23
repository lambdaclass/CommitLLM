import VerifiedInference.SecurityGame
import VerifiedInference.MultiVector
import VerifiedInference.GameToFreivalds

/-!
# Independent Checks: Independence Model for Multi-Check Amplification

This module formalizes the independence model underlying VeriLM's k-round
soundness guarantee.  The central idea is simple: to amplify the Freivalds
1/p per-check error bound to (1/p)^k across k rounds, the k random vectors
must be drawn independently.

## Independence model

We represent independence structurally: each of the k checks owns its own
`MatmulOpening` (containing a distinct claimed input/output pair) and its own
weight matrix.  The random vector for check i is drawn uniformly from
`FpVec p m` independently of all other rounds — this is encoded implicitly by
the fact that `escapingSet (checks.weights i) (checks.openings i)` is a subset
of the *per-round* randomness space rather than a joint space.

Because each escaping set lives in an independent copy of the randomness space,
the product-space amplification argument (`amplification_inductive`) applies
directly: if each per-round bad set has at most 1/p of the vectors, then the
joint probability of evading all k checks is at most (1/p)^k.

## Relationship to `GameToFreivalds`

`GameToFreivalds` establishes the per-matrix 1/p bound for a *single* cheated
matrix.  This file lifts that bound to a bundle of k matrices/openings,
making the amplification step fully explicit at the level of the
`IndependentChecks` structure.
-/

namespace VerifiedInference

variable {p : ℕ} [hp : Fact (Nat.Prime p)]

/-- k independent Freivalds checks with k weight matrices and k openings. -/
structure IndependentChecks (p : ℕ) (m n k : ℕ) where
  weights : Fin k → FpMatrix p m n
  openings : Fin k → MatmulOpening p m n

def IndependentChecks.cheatedAt {m n k : ℕ}
    (checks : IndependentChecks p m n k) (i : Fin k) : Prop :=
  cheatedAtMatrix (checks.weights i) (checks.openings i)

def IndependentChecks.escapingAt {m n k : ℕ}
    (checks : IndependentChecks p m n k) (i : Fin k) : Finset (FpVec p m) :=
  escapingSet (checks.weights i) (checks.openings i)

/-- If the adversary cheated at check i, the escaping set for that check
    gives the per-matrix bound, which amplifies to (1/p)^k over k checks. -/
theorem independent_checks_amplified
    {m n k : ℕ} (hm : 0 < m)
    (checks : IndependentChecks p m n k)
    (i : Fin k) (hCheat : checks.cheatedAt i) :
    (checks.escapingAt i).card ^ k * p ^ k ≤
      (Fintype.card (FpVec p m)) ^ k :=
  amplification_inductive _ p _
    (escaping_set_bound (checks.weights i) (checks.openings i) hCheat hm) k

end VerifiedInference
