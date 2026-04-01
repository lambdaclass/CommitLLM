//! INT8 matrix-vector multiply returning exact i32 accumulators.
//!
//! Used by the verifier for shell replay from retained state + public weights.
//! The same operation the prover runs during inference, replayed verifier-side
//! so the verifier doesn't trust any prover-supplied intermediates.

/// Compute `W @ x` where W is (rows × cols) row-major INT8 and x is INT8.
///
/// Returns exact i32 accumulators (no requantization). This is the same
/// integer matmul used in inference — the verifier replays it from public
/// weights to reconstruct the computation shell.
pub fn matmul_i32(w: &[i8], x: &[i8], rows: usize, cols: usize) -> Vec<i32> {
    assert_eq!(w.len(), rows * cols, "weight matrix size mismatch");
    assert_eq!(x.len(), cols, "input vector size mismatch");
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| w[r * cols + c] as i32 * x[c] as i32)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matmul() {
        // 2×2 identity matrix times [3, -5]
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let x: Vec<i8> = vec![3, -5];
        let result = matmul_i32(&w, &x, 2, 2);
        assert_eq!(result, vec![3, -5]);
    }

    #[test]
    fn test_matmul_accumulation() {
        // [[1, 2], [3, 4]] @ [5, 6] = [17, 39]
        let w: Vec<i8> = vec![1, 2, 3, 4];
        let x: Vec<i8> = vec![5, 6];
        let result = matmul_i32(&w, &x, 2, 2);
        assert_eq!(result, vec![17, 39]);
    }
}
