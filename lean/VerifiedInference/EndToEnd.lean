import VerifiedInference.Basic
import VerifiedInference.ProtocolParams
import VerifiedInference.AccumulatorBound
import VerifiedInference.SecurityGame
import VerifiedInference.GameToFreivalds
import VerifiedInference.ReadmeShell

/-!
# End-to-End Chained Model Identity Theorem

This module contains the top-level composition theorem `verilm_end_to_end` that
chains the full VeriLM pipeline from integer-level cheating to amplified
Freivalds bounds, in a single statement.

## Pipeline

The proof chains:

1. **Accumulator bounds** (`accumulator_bound_sufficient`): INT8 x INT8 dot
   products fit in INT32 when the dimension satisfies `n * 16384 < 2^31`.

2. **Integer-to-field lifting** (`integer_cheating_implies_field_cheating`):
   if the integer-level output disagrees with the honest matmul, the
   disagreement persists after lifting to ZMod protocolPrime.

3. **Security game** (`cheatedAtMatrix`, `escaping_set_bound`): the field-level
   disagreement triggers the Freivalds escaping-set bound.

4. **Amplification** (`cheating_amplified_bound`): the single-matrix 1/p bound
   lifts to (1/p)^k over k = exactShellMatmulCount nOpenedLayers independent
   Freivalds checks.

## Key bridge

The non-trivial step is connecting `integer_cheating_implies_field_cheating`
(which produces disagreement of plain functions `Fin m -> ZMod p`) to
`cheatedAtMatrix` (which requires `opening.z /= W.mulVec opening.x`). Since
`Matrix.mulVec` for a function-defined matrix unfolds to `fun i => sum j, W i j * x j`
via `Matrix.mulVec` and `dotProduct`, the two sides agree and the bridge closes
with `simp [Matrix.mulVec, dotProduct]`.
-/

namespace VerifiedInference

/-! ## Helper: INT8 values are INT32 values -/

/-- Every INT8 value is also a valid INT32 value (the range [-128, 127] is
    contained in [-2^31, 2^31 - 1]). -/
theorem isInt8_implies_isInt32 (x : ℤ) (h : isInt8 x) : isInt32 x := by
  unfold isInt8 at h; unfold isInt32; omega

/-! ## Bridge: function-level disagreement implies cheatedAtMatrix -/

/-- If the field-level output functions disagree, then `cheatedAtMatrix` holds
    for the corresponding matrix and opening. This bridges the function-level
    conclusion of `integer_cheating_implies_field_cheating` to the
    `cheatedAtMatrix` predicate used by the security game.

    The proof unfolds `Matrix.mulVec` and `dotProduct` to see that
    `W.mulVec x` is definitionally `fun i => ∑ j, W i j * x j`. -/
private theorem field_cheat_implies_cheatedAtMatrix
    {m n : ℕ}
    (W_int : Fin m → Fin n → ℤ) (x_int : Fin n → ℤ) (z_int : Fin m → ℤ)
    (hFieldCheat :
      (fun i => (↑(z_int i) : ZMod protocolPrime)) ≠
      (fun i => ∑ j, (↑(W_int i j) : ZMod protocolPrime) * ↑(x_int j))) :
    cheatedAtMatrix
      ((fun i j => (↑(W_int i j) : ZMod protocolPrime)) : FpMatrix protocolPrime m n)
      (⟨fun j => (↑(x_int j) : ZMod protocolPrime),
        fun i => (↑(z_int i) : ZMod protocolPrime)⟩ :
        MatmulOpening protocolPrime m n) := by
  intro heq
  apply hFieldCheat
  funext i
  have hi := congr_fun heq i
  -- hi : ↑(z_int i) = (Matrix.mulVec ... ...) i
  -- Unfold Matrix.mulVec (which is fun i => dotProduct (M i) v)
  -- and dotProduct (which is fun v w => ∑ i, v i * w i)
  -- to expose: ↑(z_int i) = ∑ j, ↑(W_int i j) * ↑(x_int j)
  simp only [Matrix.mulVec, dotProduct] at hi
  exact hi

/-! ## Main end-to-end theorem -/

/-- **VeriLM End-to-End Theorem**: from integer-level cheating to amplified
    Freivalds bounds, in a single statement.

    Given:
    - An INT8 weight matrix `W_int` and INT8 input vector `x_int`
    - A claimed INT32 output `z_int` that disagrees with the honest matmul
    - A dimension bound ensuring accumulators fit in INT32
    - A positive output dimension

    We conclude (conjunction of three facts):
    1. The field-level opening cheats (lifted z and Wx disagree in ZMod p)
    2. Per-matrix Freivalds bound: |escapingSet| * p <= |total|
    3. Amplified bound: |escapingSet|^k * p^k <= |total|^k
       where k = exactShellMatmulCount nOpenedLayers -/
theorem verilm_end_to_end
    {m n : ℕ}
    (W_int : Fin m → Fin n → ℤ)
    (x_int : Fin n → ℤ)
    (z_int : Fin m → ℤ)
    (hW : ∀ i j, isInt8 (W_int i j))
    (hx : ∀ j, isInt8 (x_int j))
    (hz : ∀ i, isInt32 (z_int i))
    (hdim : n * 16384 < 2147483648)
    (hcheat : z_int ≠ fun i => ∑ j, W_int i j * x_int j)
    (hm : 0 < m)
    (nOpenedLayers : ℕ) :
    let W_fp : FpMatrix protocolPrime m n :=
      fun i j => (↑(W_int i j) : ZMod protocolPrime)
    let x_fp : FpVec protocolPrime n :=
      fun j => (↑(x_int j) : ZMod protocolPrime)
    let z_fp : FpVec protocolPrime m :=
      fun i => (↑(z_int i) : ZMod protocolPrime)
    let fpOpening : MatmulOpening protocolPrime m n := ⟨x_fp, z_fp⟩
    let k := exactShellMatmulCount nOpenedLayers
    cheatedAtMatrix W_fp fpOpening ∧
    (escapingSet W_fp fpOpening).card * protocolPrime ≤
      Fintype.card (FpVec protocolPrime m) ∧
    (escapingSet W_fp fpOpening).card ^ k * protocolPrime ^ k ≤
      (Fintype.card (FpVec protocolPrime m)) ^ k := by
  -- Step 1: Accumulator bounds — each row's dot product fits in INT32
  have hW32 : ∀ i j, isInt32 (W_int i j) :=
    fun i j => isInt8_implies_isInt32 _ (hW i j)
  have hx32 : ∀ j, isInt32 (x_int j) :=
    fun j => isInt8_implies_isInt32 _ (hx j)
  have hdot : ∀ i, isInt32 (∑ j, W_int i j * x_int j) :=
    fun i => accumulator_bound_sufficient n hdim (W_int i) x_int (fun j => hW i j) hx
  -- Step 2: Integer cheating implies field cheating (on plain functions)
  have hFieldCheat :
      (fun i => (↑(z_int i) : ZMod protocolPrime)) ≠
      (fun i => ∑ j, (↑(W_int i j) : ZMod protocolPrime) * ↑(x_int j)) :=
    integer_cheating_implies_field_cheating W_int x_int z_int hW32 hx32 hz hdot hcheat
  -- Step 3: Bridge to cheatedAtMatrix via the helper lemma
  have hCheated := field_cheat_implies_cheatedAtMatrix W_int x_int z_int hFieldCheat
  -- Step 4: Assemble the three-part conjunction
  -- Part 1: field-level cheating (from Step 3)
  -- Part 2: per-matrix Freivalds bound (from escaping_set_bound)
  -- Part 3: amplified bound (from cheating_amplified_bound)
  exact ⟨hCheated,
         escaping_set_bound _ _ hCheated hm,
         cheating_amplified_bound hm _ _ hCheated (exactShellMatmulCount nOpenedLayers)⟩

end VerifiedInference
