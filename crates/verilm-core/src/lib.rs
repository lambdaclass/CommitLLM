pub mod attention;
pub mod constants;
pub mod field;
pub mod freivalds;
pub mod margin;
pub mod matmul;
pub mod merkle;
pub mod rmsnorm;
pub mod rope;
pub mod sampling;
pub mod serialize;
pub mod silu;
pub mod streaming;
pub mod types;

/// Clamp i32 accumulators to INT8 range (simplified requantize).
///
/// This is the toy-model requantization: saturating clamp to [-128, 127].
/// For production W8A8 models, use [`bridge_requantize`] which accounts
/// for weight and activation quantization scales.
pub fn requantize(acc: &[i32]) -> Vec<i8> {
    acc.iter().map(|&v| v.clamp(-128, 127) as i8).collect()
}

/// Scale-aware simplified requantization bridge for V4 verification.
///
/// Converts i32 matmul accumulators to i8 using quantization scales:
///   output = round(acc * scale_w * scale_x / scale_out).clamp(-128, 127)
///
/// For native INT8 (scale_w == 0.0): falls back to toy-model clamp behavior.
/// This is a simplified bridge — it does not include the residual connection
/// or RMSNorm. For the full bridge, see [`rmsnorm::requantize_bridge`].
pub fn bridge_requantize(acc: &[i32], scale_w: f32, scale_x: f32, scale_out: f32) -> Vec<i8> {
    if scale_w == 0.0 {
        return requantize(acc);
    }
    let combined = (scale_w as f64) * (scale_x as f64) / (scale_out as f64);
    acc.iter()
        .map(|&v| (v as f64 * combined).round().clamp(-128.0, 127.0) as i8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_requantize_fallback() {
        // scale_w == 0.0 → same as requantize (clamp)
        let acc = vec![50, -200, 127, 300];
        assert_eq!(bridge_requantize(&acc, 0.0, 1.0, 1.0), requantize(&acc));
    }

    #[test]
    fn test_bridge_requantize_identity() {
        // scale_w * scale_x / scale_out == 1.0 → same as round + clamp
        let acc = vec![10, -20, 100, -128];
        let result = bridge_requantize(&acc, 1.0, 1.0, 1.0);
        assert_eq!(result, vec![10, -20, 100, -128]);
    }

    #[test]
    fn test_bridge_requantize_scaling() {
        // acc=1000, scale_w=0.01, scale_x=0.5, scale_out=0.1
        // output = round(1000 * 0.01 * 0.5 / 0.1) = round(50) = 50
        let result = bridge_requantize(&[1000], 0.01, 0.5, 0.1);
        assert_eq!(result, vec![50]);
    }

    #[test]
    fn test_bridge_requantize_clamps_large() {
        // Result > 127 should clamp
        let result = bridge_requantize(&[100000], 1.0, 1.0, 1.0);
        assert_eq!(result, vec![127]);
        // Result < -128 should clamp
        let result = bridge_requantize(&[-100000], 1.0, 1.0, 1.0);
        assert_eq!(result, vec![-128]);
    }
}
