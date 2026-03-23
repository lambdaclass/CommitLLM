import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Finset.Basic
import VerifiedInference.Basic
import VerifiedInference.Freivalds

/-!
# Security Game: Adversary Model for VeriLM Model-Identity Guarantee

This module formalises the adversary model used to state VeriLM's
model-identity guarantee: an honest verifier cannot be fooled into
accepting a wrong matrix-vector product with probability greater than 1/p.

## Commitment ordering

The commitment ordering is implicit in the quantifier structure throughout
this file: the adversary's opening (`MatmulOpening`) is a fixed parameter
chosen **before** the random vector `r` is drawn.  The bound in
`escaping_set_bound` therefore holds uniformly over all `r` — the adversary
cannot adaptively choose its opening after seeing `r`.  Concretely,
`escapingSet W opening` is the set of random vectors that would *not* catch
a cheating adversary, and `escaping_set_bound` shows this set occupies at
most a `1/p` fraction of the full randomness space.
-/

namespace VerifiedInference

variable {p : ℕ} [hp : Fact (Nat.Prime p)]

/-! ## Challenge -/

/-- A verifier challenge for a multi-layer model.

    `tokenIndex` identifies which token's inference trace is being checked;
    `openedLayers` is the list of layer indices the verifier asks the prover
    to open. -/
structure Challenge (nLayers : ℕ) where
  /-- Index of the token whose trace is being verified. -/
  tokenIndex : ℕ
  /-- The layers the verifier has asked the prover to open. -/
  openedLayers : List (Fin nLayers)

/-! ## Matmul Opening -/

/-- The adversary's claimed input/output pair for a single matrix multiplication.

    An honest prover would supply `x` and `z = W · x`; a cheating prover may
    supply any `z` it likes.  The security game asks: for how many random
    vectors `r` would such a cheating claim go undetected by Freivalds? -/
structure MatmulOpening (p : ℕ) (m n : ℕ) where
  /-- The claimed input vector. -/
  x : FpVec p n
  /-- The claimed output vector (may differ from `W · x` if the prover cheats). -/
  z : FpVec p m

/-! ## Cheating Predicate -/

/-- `cheatedAtMatrix W opening` holds when the adversary's claimed output
    differs from the honest matrix-vector product `W · opening.x`. -/
def cheatedAtMatrix {m n : ℕ} (W : FpMatrix p m n) (opening : MatmulOpening p m n) : Prop :=
  opening.z ≠ W.mulVec opening.x

/-! ## Escaping Set -/

/-- The set of random vectors `r` that would **not** detect the adversary's
    cheating claim.  A vector `r` "escapes" detection if the Freivalds check
    passes despite the prover having cheated, i.e.
    `(Wᵀ r) · opening.x = r · opening.z`. -/
def escapingSet {m n : ℕ} (W : FpMatrix p m n) (opening : MatmulOpening p m n) :
    Finset (FpVec p m) :=
  Finset.univ.filter fun r =>
    dotProduct (W.transpose.mulVec r) opening.x = dotProduct r opening.z

/-! ## Security Bound -/

/-- **Security bound**: if the adversary cheats (i.e. `cheatedAtMatrix W opening`),
    the escaping set occupies at most a `1/p` fraction of the full randomness space.

    Formally: `(escapingSet W opening).card * p ≤ Fintype.card (FpVec p m)`.

    This follows directly from `freivalds_sound` by unfolding the definitions. -/
theorem escaping_set_bound {m n : ℕ} (W : FpMatrix p m n) (opening : MatmulOpening p m n)
    (hcheat : cheatedAtMatrix W opening) (hm : 0 < m) :
    (escapingSet W opening).card * p ≤ Fintype.card (FpVec p m) :=
  freivalds_sound W opening.x opening.z hcheat hm

end VerifiedInference
