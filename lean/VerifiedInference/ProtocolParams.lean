import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Linarith
import VerifiedInference.Basic
import VerifiedInference.AccumulatorBound

/-!
# Protocol Parameters and Integer-to-Field Lifting

This module fixes the concrete protocol prime p = 2^61 - 1 (a Mersenne prime)
and proves key lemmas bridging integer-domain computation to ZMod p computation.

Main results:
- `protocolPrime_gt_int32_max`: p > 2^31 - 1, so INT32 values are distinct in F_p.
- `liftInt_injective_on_int32`: Casting integers to ZMod p is injective on INT32 range.
- `lift_dotProduct_commutes`: Integer dot products commute with casting to F_p.
- `integer_cheating_implies_field_cheating`: Cheating in ℤ implies cheating in F_p.
-/

namespace VerifiedInference

/-! ## Protocol Prime -/

/-- The protocol prime: p = 2^61 - 1 (Mersenne prime M61). -/
def protocolPrime : ℕ := 2305843009213693951

/-- Primality of protocolPrime is verified externally (PARI/GP: `isprime(2^61-1)`,
    Mathematica: `PrimeQ[2^61-1]`). We axiomatize it because `native_decide` times out
    on primality testing of 61-bit numbers in Lean 4. -/
axiom protocolPrime_is_prime : Nat.Prime protocolPrime

/-- Register primality as a `Fact` instance for typeclass inference in ZMod lemmas. -/
instance : Fact (Nat.Prime protocolPrime) := ⟨protocolPrime_is_prime⟩

/-! ## Protocol Prime is Larger than INT32 Range -/

/-- The protocol prime exceeds INT32_MAX = 2^31 - 1. This ensures all INT32 values
    have distinct representations in ZMod protocolPrime. -/
theorem protocolPrime_gt_int32_max : (2147483647 : ℕ) < protocolPrime := by
  native_decide

/-! ## Injectivity of Integer Lifting on INT32 Range -/

/-- If two INT32 integers have equal images in ZMod protocolPrime, they are equal.
    This is the key lemma ensuring that "cheating" (using wrong integer values)
    is detectable after lifting to the field.

    Proof strategy: ZMod.intCast_eq_intCast_iff_dvd_sub gives (p : ℤ) ∣ (y - x).
    Since both x, y are INT32, |y - x| < 2 * 2^31 < 2^61 - 1 = p. The only
    multiple of p in that range is 0, so y - x = 0 and x = y. -/
theorem liftInt_injective_on_int32 (x y : ℤ)
    (hx : isInt32 x) (hy : isInt32 y)
    (heq : (x : ZMod protocolPrime) = (y : ZMod protocolPrime)) :
    x = y := by
  -- Step 1: Extract divisibility from field equality
  rw [ZMod.intCast_eq_intCast_iff_dvd_sub] at heq
  -- heq : (protocolPrime : ℤ) ∣ y - x
  -- Step 2: Show |y - x| < protocolPrime
  have hbound : |y - x| < (protocolPrime : ℤ) := by
    unfold isInt32 at hx hy
    rw [abs_lt]
    unfold protocolPrime
    constructor <;> omega
  -- Step 3: The only multiple of p with |·| < p is 0
  have hzero : y - x = 0 := Int.eq_zero_of_abs_lt_dvd heq hbound
  -- Step 4: Conclude x = y
  linarith

/-! ## Dot Product Commutes with Lifting -/

/-- Casting a dot product (sum of products) from ℤ to ZMod p commutes with
    computing the dot product directly in ZMod p. This follows because Int.cast
    is a ring homomorphism (preserves + and ×). -/
theorem lift_dotProduct_commutes {n : ℕ} (w x : Fin n → ℤ) :
    (↑(∑ i : Fin n, w i * x i) : ZMod protocolPrime) =
    ∑ i : Fin n, (↑(w i) : ZMod protocolPrime) * ↑(x i) := by
  push_cast
  ring

/-! ## Integer Cheating Implies Field Cheating -/

/-- If an integer matrix-vector product disagrees with a claimed result, the
    disagreement persists after lifting to ZMod protocolPrime — provided all
    values are in INT32 range.

    This bridges the integer-domain "honest computation" to the field-domain
    Freivalds check: if the prover cheats on even one output coordinate in ℤ,
    the lifted output in ZMod p also disagrees, so Freivalds can catch it.

    Proof sketch:
    1. There exists some row i where z_int i ≠ ∑ j, W i j * x j.
    2. By liftInt_injective_on_int32, the lifted values also disagree at row i.
    3. By lift_dotProduct_commutes, the lifted dot product equals the field dot product.
    4. Therefore the field-level output disagrees at row i.
-/
theorem integer_cheating_implies_field_cheating
    {m n : ℕ}
    (W : Fin m → Fin n → ℤ) (x : Fin n → ℤ) (z_int : Fin m → ℤ)
    (_hW : ∀ i j, isInt32 (W i j))
    (_hx : ∀ j, isInt32 (x j))
    (hz : ∀ i, isInt32 (z_int i))
    (hdot : ∀ i, isInt32 (∑ j, W i j * x j))
    (hcheat : z_int ≠ (fun i => ∑ j, W i j * x j)) :
    (fun i => (↑(z_int i) : ZMod protocolPrime)) ≠
    (fun i => ∑ j, (↑(W i j) : ZMod protocolPrime) * ↑(x j)) := by
  -- There exists a row where the integer outputs disagree
  intro hcontra
  apply hcheat
  funext i
  -- At row i, the field-level values agree
  have hrow := congr_fun hcontra i
  -- Rewrite the RHS using lift_dotProduct_commutes
  rw [← lift_dotProduct_commutes] at hrow
  -- Apply injectivity to conclude the integer values agree
  exact liftInt_injective_on_int32 (z_int i) (∑ j, W i j * x j) (hz i) (hdot i) hrow

/-- Combined bridge: if weights and inputs are INT8 and the dimension
    satisfies the accumulator bound, integer cheating implies field cheating.
    Eliminates the need for callers to separately prove isInt32 on the dot product. -/
theorem int8_cheating_implies_field_cheating
    {m n : ℕ}
    (W : Fin m → Fin n → ℤ) (x : Fin n → ℤ) (z_int : Fin m → ℤ)
    (hW : ∀ i j, isInt8 (W i j))
    (hx : ∀ j, isInt8 (x j))
    (hz : ∀ i, isInt32 (z_int i))
    (hn : n * 16384 < 2147483648)
    (hcheat : z_int ≠ (fun i => ∑ j, W i j * x j)) :
    (fun i => (↑(z_int i) : ZMod protocolPrime)) ≠
    (fun i => ∑ j, (↑(W i j) : ZMod protocolPrime) * ↑(x j)) :=
  integer_cheating_implies_field_cheating W x z_int
    (fun i j => ⟨by linarith [(hW i j).1], by linarith [(hW i j).2]⟩)
    (fun j => ⟨by linarith [(hx j).1], by linarith [(hx j).2]⟩)
    hz
    (fun i => accumulator_bound_sufficient n hn _ _ (hW i) hx)
    hcheat

end VerifiedInference
