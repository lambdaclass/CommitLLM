//! Weight matrix norm utilities.
//!
//! # Norms
//!
//! All norms in this module are the **L-infinity (max absolute row sum)** norm:
//!
//! ```text
//! ||W||_inf = max_i Σ_j |W[i,j]|
//! ```
//!
//! For INT8 weights with per-tensor scale `s`, the scaled norm is:
//!
//! ```text
//! ||W||_inf = s * max_i Σ_j |W_i8[i,j]|
//! ```

/// Compute the L-infinity operator norm of a weight matrix.
///
/// ```text
/// ||W||_inf = scale * max_i Σ_j |W_i8[i,j]|
/// ```
///
/// This is the max absolute row sum, scaled by the per-tensor quantization
/// scale. For the toy model (unit scale), pass `scale = 1.0`.
pub fn compute_matrix_linf_norm(weights: &[i8], rows: usize, cols: usize, scale: f32) -> f32 {
    assert_eq!(weights.len(), rows * cols);
    let mut max_row_sum: f32 = 0.0;
    for row in 0..rows {
        let row_sum: f32 = (0..cols)
            .map(|col| (weights[row * cols + col] as f32).abs())
            .sum();
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }
    max_row_sum * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_linf_norm() {
        // [[1, -2, 3], [4, -5, 6]]
        // Row sums: 6, 15. Inf-norm = 15.
        let w: Vec<i8> = vec![1, -2, 3, 4, -5, 6];
        assert_eq!(compute_matrix_linf_norm(&w, 2, 3, 1.0), 15.0);
    }

    #[test]
    fn test_matrix_linf_norm_with_scale() {
        let w: Vec<i8> = vec![1, -2, 3, 4, -5, 6];
        assert_eq!(compute_matrix_linf_norm(&w, 2, 3, 0.5), 7.5);
    }
}
