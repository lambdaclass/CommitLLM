import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Field.Basic

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

/-- Auxiliary: `(n-m-k) * n ≤ (n-k) * (n-m)` in ℕ, given `m + k ≤ n`.
    Equivalently `0 ≤ k * m`, which is always true. -/
private lemma nat_sub_mul_le (m n k : ℕ) (_hmk : m + k ≤ n) :
    (n - m - k) * n ≤ (n - k) * (n - m) := by
  have h2 : n - m ≤ n := Nat.sub_le n m
  rw [Nat.sub_mul (n - m) k n, Nat.sub_mul n k (n - m), Nat.mul_comm (n - m) n]
  exact Nat.sub_le_sub_left (Nat.mul_le_mul_left k h2) (n * (n - m))

/-- Natural number form of the hypergeometric bound:
    `C(n-m, k) * n^k ≤ C(n, k) * (n-m)^k`.
    This avoids real-valued division. Proved by induction on `k` using
    `Nat.choose_succ_right_eq` to relate `C(a, k+1)` to `C(a, k)`. -/
private theorem choose_mul_pow_le (m n k : ℕ) (hm : m ≤ n) (hk : k ≤ n - m) :
    Nat.choose (n - m) k * n ^ k ≤ Nat.choose n k * (n - m) ^ k := by
  induction k with
  | zero => simp
  | succ k ih =>
    have hk' : k ≤ n - m := Nat.le_of_succ_le hk
    have hmk : m + k ≤ n := by omega
    have ih' := ih hk'
    have key := nat_sub_mul_le m n k hmk
    -- Prove the (k+1)-scaled version using choose_succ_right_eq, then cancel (k+1).
    -- choose_succ_right_eq: C(a, k+1) * (k+1) = C(a, k) * (a - k)
    have step1 : Nat.choose (n - m) (k + 1) * (k + 1) * n ^ (k + 1) ≤
        Nat.choose n (k + 1) * (k + 1) * (n - m) ^ (k + 1) := by
      rw [Nat.choose_succ_right_eq (n - m) k, Nat.choose_succ_right_eq n k,
          pow_succ, pow_succ]
      calc Nat.choose (n - m) k * (n - m - k) * (n ^ k * n)
          = (Nat.choose (n - m) k * n ^ k) * ((n - m - k) * n) := by ring
        _ ≤ (Nat.choose n k * (n - m) ^ k) * ((n - k) * (n - m)) :=
            Nat.mul_le_mul ih' key
        _ = Nat.choose n k * (n - k) * ((n - m) ^ k * (n - m)) := by ring
    -- Rearrange to `X * (k+1) ≤ Y * (k+1)` and cancel
    have step2 : Nat.choose (n - m) (k + 1) * n ^ (k + 1) * (k + 1) ≤
        Nat.choose n (k + 1) * (n - m) ^ (k + 1) * (k + 1) := by linarith
    exact Nat.le_of_mul_le_mul_right step2 (Nat.zero_lt_succ k)

/-- The hypergeometric miss probability is bounded above by the
binomial approximation `(1 - m/n)^k`. -/
theorem miss_prob_le_exp (m n k : ℕ)
    (hm : m ≤ n) (hn : 0 < n) (hk : k ≤ n - m) :
    missProb m n k ≤ (1 - (m : ℝ) / (n : ℝ)) ^ k := by
  unfold missProb
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hcn_pos : (0 : ℝ) < Nat.choose n k := by
    exact_mod_cast Nat.choose_pos (by omega : k ≤ n)
  -- Rewrite `1 - m/n = (n - m)/n`, then expand `((n-m)/n)^k = (n-m)^k / n^k`
  rw [one_sub_div hn', div_pow, div_le_div_iff₀ hcn_pos (pow_pos hn_pos k)]
  -- Replace `(↑n - ↑m : ℝ)` with `↑(n - m)`
  rw [show (↑n : ℝ) - (↑m : ℝ) = ↑(n - m) from (Nat.cast_sub hm).symm]
  -- Lift the ℕ inequality to ℝ
  have h : (↑(Nat.choose (n - m) k * n ^ k) : ℝ) ≤ ↑(Nat.choose n k * (n - m) ^ k) := by
    exact_mod_cast choose_mul_pow_le m n k hm hk
  simp only [Nat.cast_mul, Nat.cast_pow] at h
  linarith

/-- The README catch-probability formula `1 - (1 - m/n)^k` is a lower bound
for the true (hypergeometric) detection probability `1 - missProb m n k`. -/
theorem catch_probability_lower_bound (m n k : ℕ)
    (hm : m ≤ n) (hn : 0 < n) (hk : k ≤ n - m) :
    catchProbability m n k ≤ 1 - missProb m n k := by
  unfold catchProbability
  have h := miss_prob_le_exp m n k hm hn hk
  linarith

end VerifiedInference
