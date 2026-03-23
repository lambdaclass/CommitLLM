//! Precomputed Freivalds verification.
//!
//! Key insight: r_j^T (W_j x) = (r_j^T W_j) x = v_j . x
//!
//! Keygen (verifier-side): compute v_j^(i) = r_j^T W_j^(i) once per matrix per layer.
//! Verification: check v_j^(i) . x == r_j . z where z is the claimed output.
//!
//! # Security model
//!
//! The random vectors r_j are **verifier-secret**. The prover sends full
//! output vectors z in the trace; the verifier computes r · z locally.
//! If the prover knew r, it could forge z satisfying r · z = v · x for
//! any input x without actually computing Wx. Since v = r^T W and the
//! prover knows W (open weights), leaking v also leaks r.

use crate::field::{Fp, Fp64, Fp128, P64};

/// Precompute v = r^T W in F_p.
///
/// W is stored row-major: W[row * cols + col], shape (rows, cols).
/// r has length `rows` (output dimension m_j).
/// Returns v of length `cols` (input dimension n_j).
pub fn precompute_v(r: &[Fp], weight: &[i8], rows: usize, cols: usize) -> Vec<Fp> {
    assert_eq!(r.len(), rows, "r length must match output dimension");
    assert_eq!(weight.len(), rows * cols, "weight size must be rows * cols");

    let mut v = vec![Fp::ZERO; cols];
    for col in 0..cols {
        // v[col] = sum_i r[i] * W[i][col]
        let mut acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for row in 0..rows {
            let w_val = weight[row * cols + col] as i16;
            let r_val = r[row].0 as u128;
            if w_val >= 0 {
                acc += r_val * w_val as u128;
            } else {
                neg_acc += r_val * (-w_val) as u128;
            }
        }
        let pos = (acc % Fp::P_U128) as u64;
        let neg = (neg_acc % Fp::P_U128) as u64;
        v[col] = if pos >= neg {
            Fp((pos - neg) as u32)
        } else {
            Fp((pos + crate::field::P - neg) as u32)
        };
    }
    v
}

/// Verify a single matrix multiplication: does v . x == r . z?
///
/// v = precomputed r^T W (length = input_dim)
/// x = input vector (INT8, length = input_dim)
/// r = random vector (length = output_dim)
/// z = claimed output W*x (i32 accumulators, length = output_dim)
///
/// The check operates in F_p: both sides are computed mod p.
/// The prover sends full i32 accumulators (not requantized INT8)
/// because requantization is lossy and would break the check.
pub fn check(v: &[Fp], x: &[i8], r: &[Fp], z: &[i32]) -> bool {
    let lhs = Fp::dot_fp_i8(v, x);
    let rhs = Fp::dot_fp_i32(r, z);
    lhs == rhs
}

impl Fp {
    pub const P_U128: u128 = crate::field::P as u128;
}

// ===========================================================================
// Q8_0 block-aware Freivalds
// ===========================================================================
//
// For a Q8_0 matrix W of shape (M, N) with B = N/32 blocks per row:
//
//   W_b = columns b*32..(b+1)*32 of W  (shape M × 32)
//   x_b = elements b*32..(b+1)*32 of x (length 32)
//   sumi_b[row] = Σ_j W_b[row,j] · x_b[j]  (exact i32)
//
// Phase A: Batched block Freivalds
//
//   Given random r (length M), precomputed v = r^T W (length N),
//   and random batching coefficients c_b (length B):
//
//   Check: Σ_b c_b · (v_b · x_b)  ==  r · z'
//   where  v_b = v[b*32..(b+1)*32]
//   and    z'[row] = Σ_b c_b · sumi_b[row]
//
//   The prover sends the per-block sumi arrays. The verifier computes
//   both sides and checks equality in F_p.
//
// Phase B: f32 assembly
//
//   output_f32[row] = Σ_b (d_w[row,b] · d_x[b] · sumi_b[row])
//
//   Canonical accumulation: left-to-right over blocks, f32 arithmetic.
//   d_w and d_x are f32 (converted from f16 at load time).

use crate::constants::Q8_0_BLOCK_SIZE;

/// Derive random batching coefficients for block Freivalds (Fiat-Shamir).
///
/// Coefficients are derived from the Merkle root commitment, layer index,
/// and matrix type, so both prover and verifier can compute them post-commit.
pub fn derive_block_coefficients(
    merkle_root: &[u8; 32],
    layer: usize,
    matrix_idx: usize,
    n_blocks: usize,
) -> Vec<Fp> {
    use sha2::{Sha256, Digest};

    let mut coeffs = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let mut hasher = Sha256::new();
        hasher.update(b"block_coeff");
        hasher.update(merkle_root);
        hasher.update((layer as u32).to_le_bytes());
        hasher.update((matrix_idx as u32).to_le_bytes());
        hasher.update((b as u32).to_le_bytes());
        let hash = hasher.finalize();
        let val = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
        coeffs.push(Fp::new(val));
    }
    coeffs
}

/// Phase A: Batched block Freivalds check for Q8_0.
///
/// Verifies that all per-block accumulators are correct in a single check:
///
///   Σ_b c_b · dot(v_b, x_b) == r · (Σ_b c_b · sumi_b)
///
/// where v_b = v[b*32..(b+1)*32] and x_b = x[b*32..(b+1)*32].
///
/// Arguments:
/// - `v`: precomputed r^T W (length = input_dim = n_blocks * 32)
/// - `x`: input vector (i8, length = input_dim)
/// - `r`: random vector (length = output_dim)
/// - `sumi`: per-block accumulators, sumi[b] has length output_dim
/// - `c`: random batching coefficients (length = n_blocks)
///
/// Returns true if the check passes.
pub fn check_q8_blocks(
    v: &[Fp],
    x: &[i8],
    r: &[Fp],
    sumi: &[Vec<i32>],
    c: &[Fp],
) -> bool {
    let n_blocks = sumi.len();
    assert_eq!(c.len(), n_blocks);
    assert_eq!(v.len(), n_blocks * Q8_0_BLOCK_SIZE);
    assert_eq!(x.len(), n_blocks * Q8_0_BLOCK_SIZE);

    let output_dim = r.len();
    for s in sumi {
        assert_eq!(s.len(), output_dim);
    }

    // LHS: Σ_b c_b · dot(v_b, x_b)
    let mut lhs = Fp::ZERO;
    for b in 0..n_blocks {
        let start = b * Q8_0_BLOCK_SIZE;
        let end = start + Q8_0_BLOCK_SIZE;
        let dot_b = Fp::dot_fp_i8(&v[start..end], &x[start..end]);
        lhs = lhs.add(c[b].mul(dot_b));
    }

    // RHS: r · z' where z'[row] = Σ_b c_b · sumi_b[row]
    // Compute z' element-wise, then dot with r.
    let mut z_prime = vec![Fp::ZERO; output_dim];
    for b in 0..n_blocks {
        for row in 0..output_dim {
            let term = c[b].mul(Fp::from_i32(sumi[b][row]));
            z_prime[row] = z_prime[row].add(term);
        }
    }
    let rhs = Fp::dot(r, &z_prime);

    lhs == rhs
}

/// Phase B: Verify f32 assembly from verified block accumulators and public scales.
///
/// Checks that:
///   claimed_output[row] == Σ_b (d_w[row * n_blocks + b] · d_x[b] · sumi_b[row])
///
/// Uses canonical left-to-right accumulation in f32.
///
/// Arguments:
/// - `sumi`: per-block accumulators, sumi[b] has length output_dim
/// - `d_w`: weight block scales, row-major: d_w[row * n_blocks + b]
/// - `d_x`: input block scales, length n_blocks
/// - `claimed_output`: the f32 output claimed by the prover
/// - `tolerance`: maximum allowed absolute difference per element (0.0 for exact)
pub fn check_q8_assembly(
    sumi: &[Vec<i32>],
    d_w: &[f32],
    d_x: &[f32],
    claimed_output: &[f32],
    tolerance: f32,
) -> bool {
    let n_blocks = sumi.len();
    if n_blocks == 0 {
        return claimed_output.is_empty();
    }
    let output_dim = sumi[0].len();
    assert_eq!(d_w.len(), output_dim * n_blocks);
    assert_eq!(d_x.len(), n_blocks);
    assert_eq!(claimed_output.len(), output_dim);

    for row in 0..output_dim {
        let mut acc: f32 = 0.0;
        for b in 0..n_blocks {
            acc += d_w[row * n_blocks + b] * d_x[b] * (sumi[b][row] as f32);
        }
        let diff = (acc - claimed_output[row]).abs();
        if diff > tolerance {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Fp64 variants
// ---------------------------------------------------------------------------

/// Precompute v = r^T W in F_p (64-bit Mersenne prime).
pub fn precompute_v_64(r: &[Fp64], weight: &[i8], rows: usize, cols: usize) -> Vec<Fp64> {
    assert_eq!(r.len(), rows, "r length must match output dimension");
    assert_eq!(weight.len(), rows * cols, "weight size must be rows * cols");

    let mut v = vec![Fp64::ZERO; cols];
    for col in 0..cols {
        let mut pos_acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for row in 0..rows {
            let w_val = weight[row * cols + col] as i16;
            let r_val = r[row].0 as u128;
            if w_val >= 0 {
                pos_acc += r_val * w_val as u128;
            } else {
                neg_acc += r_val * (-w_val) as u128;
            }
        }
        let pos = Fp64::reduce(pos_acc);
        let neg = Fp64::reduce(neg_acc);
        v[col] = if pos >= neg {
            Fp64(pos - neg)
        } else {
            Fp64(pos + P64 - neg)
        };
    }
    v
}

/// Verify a single matrix multiplication using Fp64.
pub fn check_64(v: &[Fp64], x: &[i8], r: &[Fp64], z: &[i32]) -> bool {
    let lhs = Fp64::dot_fp_i8(v, x);
    let rhs = Fp64::dot_fp_i32(r, z);
    lhs == rhs
}

// ---------------------------------------------------------------------------
// Fp128 variants
// ---------------------------------------------------------------------------

/// Precompute v = r^T W in F_p (128-bit Mersenne prime).
pub fn precompute_v_128(r: &[Fp128], weight: &[i8], rows: usize, cols: usize) -> Vec<Fp128> {
    assert_eq!(r.len(), rows, "r length must match output dimension");
    assert_eq!(weight.len(), rows * cols, "weight size must be rows * cols");

    let mut v = vec![Fp128::ZERO; cols];
    for col in 0..cols {
        // Products: r[i].0 < 2^127, |w| <= 128 < 2^8, so product < 2^135.
        // We accumulate in u128 pairs (pos/neg). Since each product can exceed u128,
        // we use the simple approach: reduce r[i] * |w| which fits in u128
        // because r[i].0 < 2^127 and |w| < 2^8, product < 2^135 > u128.
        // Actually 2^135 > 2^128 so we need care. But r[i].0 < 2^127 and |w| <= 128,
        // worst case is (2^127 - 1) * 128 which is ~2^134. Doesn't fit u128.
        // So we reduce per-element for safety.
        let mut acc = Fp128::ZERO;
        for row in 0..rows {
            let w_val = weight[row * cols + col];
            let w_fp = Fp128::from_i8(w_val);
            acc = acc.add(r[row].mul(w_fp));
        }
        v[col] = acc;
    }
    v
}

/// Verify a single matrix multiplication using Fp128.
pub fn check_128(v: &[Fp128], x: &[i8], r: &[Fp128], z: &[i32]) -> bool {
    let lhs = Fp128::dot_fp_i8(v, x);
    let rhs = Fp128::dot_fp_i32(r, z);
    lhs == rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precompute_and_check_correct() {
        // 3x2 matrix W, r of length 3, x of length 2
        // W = [[1, 2], [3, 4], [5, 6]]
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let r = vec![Fp(10), Fp(20), Fp(30)];
        let x: Vec<i8> = vec![7, 8];

        // z = W * x = [1*7+2*8, 3*7+4*8, 5*7+6*8] = [23, 53, 83]
        let z: Vec<i32> = vec![23, 53, 83];

        let v = precompute_v(&r, &w, 3, 2);
        assert!(check(&v, &x, &r, &z));
    }

    #[test]
    fn test_precompute_and_check_wrong_output() {
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let r = vec![Fp(10), Fp(20), Fp(30)];
        let x: Vec<i8> = vec![7, 8];

        let z: Vec<i32> = vec![23, 53, 84]; // 84 != 83

        let v = precompute_v(&r, &w, 3, 2);
        assert!(!check(&v, &x, &r, &z));
    }

    #[test]
    fn test_with_negative_weights() {
        // W = [[-1, 2], [3, -4]]
        let w: Vec<i8> = vec![-1, 2, 3, -4];
        let r = vec![Fp(5), Fp(10)];
        let x: Vec<i8> = vec![3, 7];

        // z = W * x = [-1*3+2*7, 3*3+(-4)*7] = [11, -19]
        let z: Vec<i32> = vec![11, -19];

        let v = precompute_v(&r, &w, 2, 2);
        assert!(check(&v, &x, &r, &z));
    }

    #[test]
    fn test_identity_freivalds() {
        let w: Vec<i8> = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let r = vec![Fp(42), Fp(99), Fp(7)];
        let x: Vec<i8> = vec![10, 20, 30];
        let z: Vec<i32> = vec![10, 20, 30]; // I*x = x

        let v = precompute_v(&r, &w, 3, 3);
        assert!(check(&v, &x, &r, &z));
    }

    // -------------------------------------------------------------------
    // Q8_0 block-aware Freivalds tests
    // -------------------------------------------------------------------

    /// Helper: compute per-block sumi for a matrix W (row-major, M×N) and input x.
    fn compute_block_sumi(w: &[i8], x: &[i8], rows: usize, cols: usize) -> Vec<Vec<i32>> {
        let bs = crate::constants::Q8_0_BLOCK_SIZE;
        let n_blocks = cols / bs;
        let mut sumi = vec![vec![0i32; rows]; n_blocks];
        for b in 0..n_blocks {
            for row in 0..rows {
                let mut acc = 0i32;
                for j in 0..bs {
                    acc += w[row * cols + b * bs + j] as i32 * x[b * bs + j] as i32;
                }
                sumi[b][row] = acc;
            }
        }
        sumi
    }

    #[test]
    fn test_q8_block_check_correct() {
        // 2x64 matrix (2 blocks of 32), output_dim=2
        let rows = 2;
        let cols = 64; // 2 blocks
        let mut w = vec![0i8; rows * cols];
        let mut x = vec![0i8; cols];

        // Fill with simple pattern
        for i in 0..rows * cols {
            w[i] = ((i % 7) as i8) - 3;
        }
        for i in 0..cols {
            x[i] = ((i % 5) as i8) - 2;
        }

        let r = vec![Fp(42), Fp(99)];
        let v = precompute_v(&r, &w, rows, cols);
        let sumi = compute_block_sumi(&w, &x, rows, cols);
        let c = vec![Fp(7), Fp(13)]; // arbitrary coefficients

        assert!(check_q8_blocks(&v, &x, &r, &sumi, &c));
    }

    #[test]
    fn test_q8_block_check_wrong_sumi() {
        let rows = 2;
        let cols = 64;
        let mut w = vec![1i8; rows * cols];
        let x = vec![1i8; cols];

        w[0] = 5; // make it non-trivial

        let r = vec![Fp(42), Fp(99)];
        let v = precompute_v(&r, &w, rows, cols);
        let mut sumi = compute_block_sumi(&w, &x, rows, cols);
        let c = vec![Fp(7), Fp(13)];

        // Corrupt block 0's accumulator
        sumi[0][0] += 1;

        assert!(!check_q8_blocks(&v, &x, &r, &sumi, &c));
    }

    #[test]
    fn test_q8_assembly_correct() {
        // 2 blocks, output_dim=3
        let sumi = vec![
            vec![100, 200, 300], // block 0
            vec![400, 500, 600], // block 1
        ];
        let d_w = vec![
            // row 0: scale_b0, scale_b1
            1.0, 2.0,
            // row 1
            0.5, 1.5,
            // row 2
            1.0, 1.0,
        ];
        let d_x = vec![1.0, 0.5];

        // Expected:
        // row 0: 1.0*1.0*100 + 2.0*0.5*400 = 100 + 400 = 500
        // row 1: 0.5*1.0*200 + 1.5*0.5*500 = 100 + 375 = 475
        // row 2: 1.0*1.0*300 + 1.0*0.5*600 = 300 + 300 = 600
        let output = vec![500.0, 475.0, 600.0];

        assert!(check_q8_assembly(&sumi, &d_w, &d_x, &output, 0.0));
    }

    #[test]
    fn test_q8_assembly_wrong() {
        let sumi = vec![vec![100, 200], vec![300, 400]];
        let d_w = vec![1.0, 1.0, 1.0, 1.0];
        let d_x = vec![1.0, 1.0];

        // Correct: row 0 = 100+300=400, row 1 = 200+400=600
        let wrong_output = vec![401.0, 600.0];

        assert!(!check_q8_assembly(&sumi, &d_w, &d_x, &wrong_output, 0.0));
        // But passes with tolerance
        assert!(check_q8_assembly(&sumi, &d_w, &d_x, &wrong_output, 1.5));
    }

    #[test]
    fn test_q8_block_agrees_with_flat() {
        // When all block coefficients are 1 and there's 1 block,
        // block check should agree with flat check.
        let rows = 3;
        let cols = 32; // exactly 1 block
        let w: Vec<i8> = (0..rows * cols).map(|i| ((i % 11) as i8) - 5).collect();
        let x: Vec<i8> = (0..cols).map(|i| ((i % 7) as i8) - 3).collect();

        let r = vec![Fp(10), Fp(20), Fp(30)];
        let v = precompute_v(&r, &w, rows, cols);

        // Flat check
        let z: Vec<i32> = (0..rows)
            .map(|row| {
                (0..cols).map(|col| w[row * cols + col] as i32 * x[col] as i32).sum()
            })
            .collect();
        assert!(check(&v, &x, &r, &z));

        // Block check (1 block, c=[1])
        let sumi = compute_block_sumi(&w, &x, rows, cols);
        assert_eq!(sumi.len(), 1);
        assert_eq!(sumi[0], z); // single block sumi == flat output
        let c = vec![Fp(1)];
        assert!(check_q8_blocks(&v, &x, &r, &sumi, &c));
    }

    // -------------------------------------------------------------------
    // Fp64 Freivalds tests
    // -------------------------------------------------------------------

    #[test]
    fn test_precompute_and_check_correct_64() {
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let r = vec![Fp64(10), Fp64(20), Fp64(30)];
        let x: Vec<i8> = vec![7, 8];
        let z: Vec<i32> = vec![23, 53, 83];

        let v = precompute_v_64(&r, &w, 3, 2);
        assert!(check_64(&v, &x, &r, &z));
    }

    #[test]
    fn test_precompute_and_check_wrong_output_64() {
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let r = vec![Fp64(10), Fp64(20), Fp64(30)];
        let x: Vec<i8> = vec![7, 8];
        let z: Vec<i32> = vec![23, 53, 84]; // wrong

        let v = precompute_v_64(&r, &w, 3, 2);
        assert!(!check_64(&v, &x, &r, &z));
    }

    #[test]
    fn test_with_negative_weights_64() {
        let w: Vec<i8> = vec![-1, 2, 3, -4];
        let r = vec![Fp64(5), Fp64(10)];
        let x: Vec<i8> = vec![3, 7];
        let z: Vec<i32> = vec![11, -19];

        let v = precompute_v_64(&r, &w, 2, 2);
        assert!(check_64(&v, &x, &r, &z));
    }

    #[test]
    fn test_identity_freivalds_64() {
        let w: Vec<i8> = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let r = vec![Fp64(42), Fp64(99), Fp64(7)];
        let x: Vec<i8> = vec![10, 20, 30];
        let z: Vec<i32> = vec![10, 20, 30];

        let v = precompute_v_64(&r, &w, 3, 3);
        assert!(check_64(&v, &x, &r, &z));
    }

    // -------------------------------------------------------------------
    // Fp128 Freivalds tests
    // -------------------------------------------------------------------

    #[test]
    fn test_precompute_and_check_correct_128() {
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let r = vec![Fp128(10), Fp128(20), Fp128(30)];
        let x: Vec<i8> = vec![7, 8];
        let z: Vec<i32> = vec![23, 53, 83];

        let v = precompute_v_128(&r, &w, 3, 2);
        assert!(check_128(&v, &x, &r, &z));
    }

    #[test]
    fn test_precompute_and_check_wrong_output_128() {
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let r = vec![Fp128(10), Fp128(20), Fp128(30)];
        let x: Vec<i8> = vec![7, 8];
        let z: Vec<i32> = vec![23, 53, 84]; // wrong

        let v = precompute_v_128(&r, &w, 3, 2);
        assert!(!check_128(&v, &x, &r, &z));
    }

    #[test]
    fn test_with_negative_weights_128() {
        let w: Vec<i8> = vec![-1, 2, 3, -4];
        let r = vec![Fp128(5), Fp128(10)];
        let x: Vec<i8> = vec![3, 7];
        let z: Vec<i32> = vec![11, -19];

        let v = precompute_v_128(&r, &w, 2, 2);
        assert!(check_128(&v, &x, &r, &z));
    }

    #[test]
    fn test_identity_freivalds_128() {
        let w: Vec<i8> = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let r = vec![Fp128(42), Fp128(99), Fp128(7)];
        let x: Vec<i8> = vec![10, 20, 30];
        let z: Vec<i32> = vec![10, 20, 30];

        let v = precompute_v_128(&r, &w, 3, 3);
        assert!(check_128(&v, &x, &r, &z));
    }
}
