//! Canonical RMSNorm recomputation for verification.
//!
//! RMSNorm is the normalization layer between transformer blocks:
//!   output[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
//!
//! The verifier recomputes RMSNorm from the opened residual stream
//! values to verify the requantization bridge between layers.
//! All arithmetic is f64 for deterministic cross-platform results.

/// Default epsilon for RMSNorm (Llama-family models).
pub const DEFAULT_EPS: f64 = 1e-5;

/// Canonical RMSNorm in f64 for deterministic verification.
///
/// Given INT8 input x (the residual stream after accumulation),
/// RMSNorm weight vector, and epsilon:
///   rms = sqrt(mean(x_f64^2) + eps)
///   output[i] = x_f64[i] / rms * weight[i]
///
/// Returns f64 output. Caller requantizes to i8 using the next layer's
/// activation scale.
pub fn rmsnorm_f64(x_i8: &[i8], weights: &[f32], eps: f64) -> Vec<f64> {
    assert_eq!(
        x_i8.len(),
        weights.len(),
        "RMSNorm: x and weights must have same length"
    );
    let n = x_i8.len() as f64;
    let sum_sq: f64 = x_i8.iter().map(|&v| {
        let f = v as f64;
        f * f
    }).sum();
    let rms = (sum_sq / n + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x_i8.iter()
        .zip(weights.iter())
        .map(|(&x, &w)| (x as f64) * inv_rms * (w as f64))
        .collect()
}

/// RMSNorm on f64 input (for chained verification where the input
/// is already a dequantized f64 residual stream).
pub fn rmsnorm_f64_input(x: &[f64], weights: &[f32], eps: f64) -> Vec<f64> {
    assert_eq!(
        x.len(),
        weights.len(),
        "RMSNorm: x and weights must have same length"
    );
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .zip(weights.iter())
        .map(|(&x_val, &w)| x_val * inv_rms * (w as f64))
        .collect()
}

/// Requantize f64 values to INT8 using a per-tensor activation scale.
///
/// output_i8[i] = clamp(round(x_f64[i] / scale), -128, 127)
///
/// This is the production quantization step (dynamic per-tensor).
/// The scale is typically max(abs(x)) / 127.
pub fn quantize_f64_to_i8(x: &[f64], scale: f64) -> Vec<i8> {
    let inv_scale = 1.0 / scale;
    x.iter()
        .map(|&v| {
            let scaled = v * inv_scale;
            scaled.round().clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Dequantize i32 accumulator and add to residual in place.
///
/// Computes: `residual[i] += acc_i32[i] * scale_w * scale_x`
pub fn dequant_add_residual(acc_i32: &[i32], scale_w: f32, scale_x: f32, residual: &mut [f64]) {
    let dequant_scale = (scale_w as f64) * (scale_x as f64);
    for (res, &acc) in residual.iter_mut().zip(acc_i32.iter()) {
        *res += (acc as f64) * dequant_scale;
    }
}

/// **Canonical bridge**: dequant → residual += → RMSNorm → quantize to i8.
///
/// Updates residual **in place** and returns the requantized i8 output.
/// This is the only correct bridge for production W8A8 models:
///   1. Dequantize: `f64 = acc_i32 * scale_w * scale_x`
///   2. Update residual: `residual += dequantized`
///   3. RMSNorm: `f64 = RMSNorm(residual, weights, eps)`
///   4. Quantize: `i8 = round(f64 / scale_next)` clamped to [-128, 127]
///
/// Both prover (`compute_shell_opening`) and verifier (`verify_shell_opening`,
/// `replay_token_shell`) use this function when BridgeParams / initial_residual
/// is present. The simplified `bridge_requantize()` fallback exists only for
/// toy-model tests.
pub fn bridge_residual_rmsnorm(
    acc_i32: &[i32],
    scale_w: f32,
    scale_x: f32,
    residual: &mut [f64],
    rmsnorm_weights: &[f32],
    eps: f64,
    scale_next: f32,
) -> Vec<i8> {
    dequant_add_residual(acc_i32, scale_w, scale_x, residual);
    let normed = rmsnorm_f64_input(residual, rmsnorm_weights, eps);
    quantize_f64_to_i8(&normed, scale_next as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_zeros() {
        // RMSNorm of zeros: 0 / sqrt(eps) * w = 0
        let x = vec![0i8; 4];
        let w = vec![1.0f32; 4];
        let out = rmsnorm_f64(&x, &w, 1e-5);
        for &v in &out {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_rmsnorm_uniform() {
        // Uniform input: RMSNorm(x) = x / |x| * w (when all elements equal)
        let x = vec![10i8; 4];
        let w = vec![1.0f32; 4];
        let out = rmsnorm_f64(&x, &w, 0.0);
        // RMS of [10,10,10,10] = 10, so output = 10/10 * 1 = 1.0
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rmsnorm_with_weights() {
        let x = vec![10i8; 4];
        let w = vec![2.0f32; 4];
        let out = rmsnorm_f64(&x, &w, 0.0);
        // RMS = 10, output = 10/10 * 2 = 2.0
        for &v in &out {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_quantize_roundtrip() {
        let x = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let scale = 1.0 / 127.0;
        let q = quantize_f64_to_i8(&x, scale as f64);
        assert_eq!(q, vec![127, -127, 64, -64, 0]);
    }

    #[test]
    fn test_quantize_clamp() {
        let x = vec![200.0, -200.0];
        let scale = 1.0;
        let q = quantize_f64_to_i8(&x, scale);
        assert_eq!(q, vec![127, -128]);
    }

    #[test]
    fn test_bridge_residual_rmsnorm_roundtrip() {
        // Canonical bridge: unit scales, zero residual, identity RMSNorm weights
        let acc = vec![10, 20, -10, 0];
        let mut residual = vec![0.0f64; 4];
        let weights = vec![1.0f32; 4];
        let result = bridge_residual_rmsnorm(&acc, 1.0, 1.0, &mut residual, &weights, 0.0, 1.0);
        // RMS of [10,20,-10,0] = sqrt((100+400+100+0)/4) = sqrt(150) ≈ 12.247
        // Output: [10/12.247, 20/12.247, -10/12.247, 0] ≈ [0.816, 1.633, -0.816, 0]
        // Quantized with scale=1.0: round to [1, 2, -1, 0]
        assert_eq!(result, vec![1, 2, -1, 0]);
        // Residual was updated in place
        assert!((residual[0] - 10.0).abs() < 1e-10);
        assert!((residual[1] - 20.0).abs() < 1e-10);
    }
}
