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
/// **Toy-model only.** Saturating clamp to [-128, 127] with no scale awareness.
/// Production W8A8 models must use [`rmsnorm::bridge_residual_rmsnorm`] which
/// implements the full dequant → residual += → RMSNorm → quantize chain.
pub fn requantize(acc: &[i32]) -> Vec<i8> {
    acc.iter().map(|&v| v.clamp(-128, 127) as i8).collect()
}

/// Scale-aware simplified requantization (toy / last-layer fallback only).
///
/// Converts i32 matmul accumulators to i8 using quantization scales:
///   output = round(acc * scale_w * scale_x / scale_out).clamp(-128, 127)
///
/// **Not the canonical bridge.** This omits the residual connection and
/// RMSNorm that real W8A8 models require. The canonical bridge is
/// [`rmsnorm::bridge_residual_rmsnorm`]. This function is only used:
/// - In toy-model tests (no BridgeParams / no residual tracking)
/// - At the last transformer layer (no subsequent RMSNorm to apply)
///
/// For native INT8 (scale_w == 0.0): falls back to toy-model clamp behavior.
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
