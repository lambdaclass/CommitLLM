pub mod attention;
pub mod constants;
pub mod field;
pub mod freivalds;
pub mod margin;
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
/// For production W8A8 models, use [`rmsnorm::requantize_bridge`] which
/// accounts for activation scales, residual connections, and RMSNorm.
pub fn requantize(acc: &[i32]) -> Vec<i8> {
    acc.iter().map(|&v| v.clamp(-128, 127) as i8).collect()
}
