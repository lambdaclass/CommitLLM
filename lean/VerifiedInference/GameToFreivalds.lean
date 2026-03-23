import VerifiedInference.Basic
import VerifiedInference.SecurityGame
import VerifiedInference.MultiVector
import VerifiedInference.ReadmeShell

/-!
# Game-to-Freivalds Bridge

This module bridges the security game (per-matrix escaping set cardinality
bounds) to the multi-vector amplification theorem, instantiating concrete
cardinalities.

## Why this file is critical

Without this bridge, the final composition theorem would merely take
`badCard * p ≤ total` as a bare assumption — the bound would be non-vacuous
only by fiat.  Here we derive that assumption directly from the game-theoretic
hypotheses: if the adversary cheated at a matrix, the escaping set is a
*concrete* set with a *concrete* cardinality, and `escaping_set_bound` provides
the inequality.  `amplification_inductive` then lifts this single-round bound
to the k-round product bound.

The chain is:

  `cheatedAtMatrix`
      → `escaping_set_bound`     (single-round, concrete card)
      → `amplification_inductive` (k rounds, product space)

This makes `cheating_amplified_bound` fully self-contained: every quantity in
the final inequality is a concrete cardinality, not an opaque hypothesis.
-/

namespace VerifiedInference

variable {p : ℕ} [hp : Fact (Nat.Prime p)]

/-! ## Single-matrix bound -/

/-- **Single-matrix game bound**: for a single cheated matrix, the escaping set
    satisfies the Freivalds 1/p bound with concrete cardinalities.

    This is a thin, legibility-oriented wrapper around `escaping_set_bound`.
    Its purpose is to give the composition theorems below a named lemma whose
    statement makes the role of each term transparent. -/
theorem single_matrix_game_bound {m n : ℕ} (hm : 0 < m)
    (W : FpMatrix p m n)
    (opening : MatmulOpening p m n)
    (hCheat : cheatedAtMatrix W opening) :
    (escapingSet W opening).card * p ≤ Fintype.card (FpVec p m) :=
  escaping_set_bound W opening hCheat hm

/-! ## Multi-matrix bound -/

/-- **Multi-matrix game bound**: given a uniform per-matrix escaping-set bound
    `badCard * p ≤ totalVectors`, the amplified bound holds for k checks.

    This is the composition step: a single-matrix 1/p bound lifts to (1/p)^k
    across k independent checks. The caller obtains `badCard` from
    `single_matrix_game_bound` applied to the worst-case cheated matrix. -/
theorem multi_matrix_game_bound
    (totalVectors badCard : ℕ) (k : ℕ)
    (hBound : badCard * p ≤ totalVectors) :
    badCard ^ k * p ^ k ≤ totalVectors ^ k :=
  amplification_inductive totalVectors p badCard hBound k

/-! ## Layer audit bound -/

/-- **Layer audit game bound**: if the adversary cheated at any of the matrices
    in the opened layers, each individually cheated matrix contributes its own
    Freivalds escaping-set bound.

    This collects the per-matrix bound uniformly across all
    `exactShellMatmulCount nOpenedLayers` matrices opened during the audit.
    The existential `hSomeCheat` is not used for this direction — the bound
    holds for *every* cheated matrix, regardless of whether others were honest.
    The caller supplies `hSomeCheat` to make the non-vacuousness of the
    protocol audit hypothesis explicit. -/
theorem layer_audit_game_bound
    {m n : ℕ} (hm : 0 < m)
    (nOpenedLayers : ℕ)
    (weights : Fin (exactShellMatmulCount nOpenedLayers) → FpMatrix p m n)
    (openings : Fin (exactShellMatmulCount nOpenedLayers) → MatmulOpening p m n)
    (_hSomeCheat : ∃ _i, cheatedAtMatrix (weights _i) (openings _i)) :
    ∀ i, cheatedAtMatrix (weights i) (openings i) →
      (escapingSet (weights i) (openings i)).card * p ≤ Fintype.card (FpVec p m) :=
  fun i hc => single_matrix_game_bound hm (weights i) (openings i) hc

/-! ## Amplified cheating bound -/

/-- **Cheating amplified bound**: the key theorem connecting the security game
    to the multi-round amplification.

    If the adversary cheated at a single matrix, the escaping-set cardinality
    satisfies the 1/p bound (`single_matrix_game_bound`), and
    `amplification_inductive` immediately lifts this to the k-th power:

        `(escapingSet W opening).card ^ k * p ^ k ≤ (Fintype.card (FpVec p m)) ^ k`

    This is the composition theorem that makes the VeriLM protocol's soundness
    claim non-vacuous: the probability that an adversary fools all k independent
    Freivalds checks is at most (1/p)^k, expressed combinatorially as a ratio of
    concrete Fintype cardinalities. -/
theorem cheating_amplified_bound
    {m n : ℕ} (hm : 0 < m)
    (W : FpMatrix p m n)
    (opening : MatmulOpening p m n)
    (hCheat : cheatedAtMatrix W opening)
    (k : ℕ) :
    (escapingSet W opening).card ^ k * p ^ k ≤ (Fintype.card (FpVec p m)) ^ k :=
  amplification_inductive
    (Fintype.card (FpVec p m))
    p
    (escapingSet W opening).card
    (single_matrix_game_bound hm W opening hCheat)
    k

end VerifiedInference
