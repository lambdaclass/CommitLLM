import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct

/-!
# Basic README-Native Types

This module contains the small generic objects used throughout the current
README-native Lean formalization: finite-field vectors for Freivalds, the
basic exact shell value ranges, and the one-line predicates reused by later
README-specific modules.
-/

namespace VerifiedInference

/-! ## Field and Vector Types -/

/-- A vector in F_p of dimension n. -/
abbrev FpVec (p : ℕ) (n : ℕ) := Fin n → ZMod p

/-- A matrix over F_p with m rows and n columns. -/
abbrev FpMatrix (p : ℕ) (m n : ℕ) := Matrix (Fin m) (Fin n) (ZMod p)

/-! ## Lifting functions: INT8/INT32 → F_p -/

/-- Lift an integer into F_p. -/
def liftInt (p : ℕ) (x : ℤ) : ZMod p := (x : ZMod p)

/-! ## Freivalds Check Predicate -/

/-- The Freivalds check for a single matrix multiplication:
    v · x = r · z, where v = r^T W is precomputed. -/
def freivaldsAccepts {p : ℕ} {n m : ℕ}
    (v : FpVec p n) (x : FpVec p n) (r : FpVec p m) (z : FpVec p m) : Prop :=
  dotProduct v x = dotProduct r z

/-- Requantization: clamp an i32 value to the INT8 range [-128, 127]. -/
def clampI8 (z : ℤ) : ℤ := max (-128) (min 127 z)

/-! ## Integer Range Predicates -/

/-- An INT8 value is in the range [-128, 127]. -/
def isInt8 (x : ℤ) : Prop := -128 ≤ x ∧ x ≤ 127

/-- An i32 value is in the range [-2³¹, 2³¹ - 1]. -/
def isInt32 (x : ℤ) : Prop := -2147483648 ≤ x ∧ x ≤ 2147483647

end VerifiedInference
