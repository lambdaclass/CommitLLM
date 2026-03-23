import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.DotProduct

import VerifiedInference.Basic

/-!
# Theorem 2: Precomputed Freivalds Equivalence

**Statement**: r^T (W x) = (r^T W) x

This is the key insight enabling efficient verification: the verifier
precomputes v = r^T W once at keygen, then checks v · x = r · z online.

Mirrors: `crates/vi-core/src/freivalds.rs` — `precompute_v()` and `check()`
-/

namespace VerifiedInference

variable {p m n : ℕ}

/-- **Theorem 2 (Precomputed Freivalds Equivalence)**:

    For any random vector r ∈ F_p^m, weight matrix W ∈ F_p^{m×n}, and
    input vector x ∈ F_p^n:

      r · (W x) = (W^T r) · x

    This allows the verifier to precompute v = W^T r and check v · x = r · z.
-/
theorem precomputed_equiv
    (r : Fin m → ZMod p) (W : Matrix (Fin m) (Fin n) (ZMod p))
    (x : Fin n → ZMod p) :
    dotProduct r (W.mulVec x) =
    dotProduct (W.transpose.mulVec r) x := by
  simp only [dotProduct, Matrix.mulVec, Matrix.transpose_apply]
  simp only [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  congr 1; ext j
  congr 1; ext i
  ring

/-- Corollary: The Freivalds check v · x = r · z is equivalent to
    checking r · (W x) = r · z, i.e., r · (W x - z) = 0. -/
theorem freivalds_check_equiv
    (r : Fin m → ZMod p) (W : Matrix (Fin m) (Fin n) (ZMod p))
    (x : Fin n → ZMod p) (z : Fin m → ZMod p)
    (v : Fin n → ZMod p) (hv : v = W.transpose.mulVec r) :
    dotProduct v x = dotProduct r z ↔
    dotProduct r (W.mulVec x) = dotProduct r z := by
  subst hv
  rw [← precomputed_equiv]

end VerifiedInference
