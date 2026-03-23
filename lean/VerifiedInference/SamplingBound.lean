import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

import VerifiedInference.ReadmeKVProvenance

/-!
# KV Sampling Catch Probability Bound

This module formalises the hypergeometric tail bound that underlies the
README's sampling-based detection argument.

When an adversary tampers with `m` out of `n` KV prefix positions and the
verifier samples `k` positions **without replacement** (hypergeometric
sampling), the exact probability of missing all tampered positions is

  missProb m n k = C(n-m, k) / C(n, k)

The continuous relaxation `(1 - m/n)^k` (binomial / sampling-with-replacement)
is an upper bound for this quantity, so

  missProb m n k ≤ (1 - m/n)^k

and therefore

  1 - (1 - m/n)^k  ≤  1 - missProb m n k  =  catchProbability (exact).

That is, the README formula *understates* the true detection probability —
a conservative, safe bound.
-/

namespace VerifiedInference

/-- Exact probability of missing all `m` tampered positions when sampling
`k` out of `n` positions without replacement (hypergeometric model). -/
noncomputable def missProb (m n k : ℕ) : ℝ :=
  (Nat.choose (n - m) k : ℝ) / (Nat.choose n k : ℝ)

-- TODO: Prove `miss_prob_le_exp` by induction on k.
--
-- Proof strategy:
--   Base case k = 0: missProb m n 0 = 1 and (1 - m/n)^0 = 1, so ≤ holds trivially.
--
--   Inductive step k → k+1:
--     Write missProb m n (k+1) = missProb m n k * (n - m - k) / (n - k).
--     The induction hypothesis gives missProb m n k ≤ (1 - m/n)^k.
--     It therefore suffices to show
--       (n - m - k) / (n - k)  ≤  (n - m) / n  =  1 - m/n,
--     i.e.  n·(n-m-k) ≤ (n-k)·(n-m),
--     which expands to  n²-n·m-n·k ≤ n²-n·m-k·n+k·m,  i.e.  0 ≤ k·m — true
--     since k, m ≥ 0.
--
--     Combining:
--       missProb m n (k+1) = missProb m n k · (n-m-k)/(n-k)
--                          ≤ (1-m/n)^k · (1-m/n)
--                          = (1-m/n)^(k+1).
--
--     Real-arithmetic side conditions needed: 0 < n, 0 < n-k (i.e. k < n),
--     and the Nat subtraction identities for converting ℕ to ℝ.

/-- The hypergeometric miss probability is bounded above by the
binomial approximation `(1 - m/n)^k`. -/
theorem miss_prob_le_exp (m n k : ℕ)
    (hm : m ≤ n) (hn : 0 < n) (hk : k ≤ n - m) :
    missProb m n k ≤ (1 - (m : ℝ) / (n : ℝ)) ^ k := by
  sorry

/-- The README catch-probability formula `1 - (1 - m/n)^k` is a lower bound
for the true (hypergeometric) detection probability `1 - missProb m n k`. -/
theorem catch_probability_lower_bound (m n k : ℕ)
    (hm : m ≤ n) (hn : 0 < n) (hk : k ≤ n - m) :
    catchProbability m n k ≤ 1 - missProb m n k := by
  unfold catchProbability
  have h := miss_prob_le_exp m n k hm hn hk
  linarith

end VerifiedInference
