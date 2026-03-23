import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Basic

/-!
# Basic Types and Definitions for Verified Inference Protocol

This module defines the core types mirroring the Rust implementation in
`crates/vi-core/src/types.rs` and `crates/vi-core/src/constants.rs`.

The protocol verifies that an inference provider used correct model weights
by checking matrix multiplications via Freivalds' algorithm over F_p.
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

/-! ## Protocol Parameters -/

/-- Parameters for the verification protocol. -/
structure Params where
  /-- Prime modulus for Freivalds checks -/
  p : ℕ
  /-- Proof that p is prime -/
  hp : Nat.Prime p
  /-- Number of transformer layers -/
  nLayers : ℕ
  /-- Number of matrix types per layer (7: Wq, Wk, Wv, Wo, Wg, Wu, Wd) -/
  nMatrixTypes : ℕ := 7

/-! ## Matrix Types (mirrors Rust MatrixType enum) -/

/-- The 7 weight matrix types in a transformer layer. -/
inductive MatrixType
  | Wq | Wk | Wv | Wo | Wg | Wu | Wd
  deriving DecidableEq, Repr

/-! ## Neural Network Model -/

/-- A neural network's weight matrices, indexed by layer and matrix type.
    Each weight matrix W_{j}^{(i)} maps F_p^{n_j} → F_p^{m_j}. -/
structure NeuralNetwork (p : ℕ) (nLayers : ℕ) where
  /-- Weight matrix for layer i, matrix type mt. -/
  weight : Fin nLayers → MatrixType → Σ (m n : ℕ), FpMatrix p m n

/-! ## Verifier Key (mirrors Rust VerifierKey) -/

/-- The verifier's secret key material.
    Contains random vectors r_j and precomputed v_j = r_j^T W_j. -/
structure VerifierKey (p : ℕ) (nLayers : ℕ) where
  /-- Random vector for each matrix type (verifier-secret). -/
  r : MatrixType → Σ (m : ℕ), FpVec p m
  /-- Precomputed v_j^{(i)} = r_j^T W_j^{(i)} for each layer and matrix type. -/
  v : Fin nLayers → MatrixType → Σ (n : ℕ), FpVec p n

/-! ## Claimed Trace -/

/-- A claimed trace for one layer: input vector x and output vector z = W·x. -/
structure LayerMatmulClaim (p : ℕ) where
  /-- Input dimension -/
  n : ℕ
  /-- Output dimension -/
  m : ℕ
  /-- Input vector x (INT8 values lifted to F_p) -/
  x : FpVec p n
  /-- Claimed output z = W·x -/
  z : FpVec p m

/-- A complete claimed trace for one token across all layers. -/
structure ClaimedTrace (p : ℕ) (nLayers : ℕ) where
  /-- Per-layer, per-matrix-type matmul claims -/
  matmulOutputs : Fin nLayers → MatrixType → LayerMatmulClaim p

/-! ## Honest Computation -/

/-- The honest matrix-vector product for a specific layer and matrix type. -/
noncomputable def honestMatmul {p : ℕ} {nLayers : ℕ}
    (net : NeuralNetwork p nLayers) (layer : Fin nLayers) (mt : MatrixType)
    (x : FpVec p (net.weight layer mt).2.1) : FpVec p (net.weight layer mt).1 :=
  (net.weight layer mt).2.2.mulVec x

/-! ## Freivalds Check Predicate -/

/-- The Freivalds check for a single matrix multiplication:
    v · x = r · z, where v = r^T W is precomputed. -/
def freivaldsAccepts {p : ℕ} {n m : ℕ}
    (v : FpVec p n) (x : FpVec p n) (r : FpVec p m) (z : FpVec p m) : Prop :=
  dotProduct v x = dotProduct r z

/-! ## Deterministic Check Predicates -/

/-- SiLU lookup table: a total function on Fin 256 (the INT8 range). -/
def SiLUTable := Fin 256 → ℤ

/-- The SiLU check passes when the claimed output matches the LUT computation. -/
def siluCheckPasses (lut : SiLUTable) {dim : ℕ} (g u h : Fin dim → ℤ) : Prop :=
  ∀ i : Fin dim, h i = lut ⟨(g i).toNat % 256, Nat.mod_lt _ (by omega)⟩ * u i

/-- Requantization: clamp an i32 value to the INT8 range [-128, 127]. -/
def clampI8 (z : ℤ) : ℤ := max (-128) (min 127 z)

/-- The chain check: next layer's input = clamp(previous layer's output). -/
def chainCheckPasses {dim : ℕ} (output : Fin dim → ℤ) (nextInput : Fin dim → ℤ) : Prop :=
  ∀ i : Fin dim, nextInput i = clampI8 (output i)

/-! ## Integer Range Predicates -/

/-- An INT8 value is in the range [-128, 127]. -/
def isInt8 (x : ℤ) : Prop := -128 ≤ x ∧ x ≤ 127

/-- An i32 value is in the range [-2³¹, 2³¹ - 1]. -/
def isInt32 (x : ℤ) : Prop := -2147483648 ≤ x ∧ x ≤ 2147483647

end VerifiedInference
