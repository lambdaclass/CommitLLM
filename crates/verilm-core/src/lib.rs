pub mod attention;
pub mod constants;
pub mod field;
pub mod freivalds;
pub mod margin;
pub mod merkle;
pub mod sampling;
pub mod serialize;
pub mod silu;
pub mod streaming;
pub mod types;

/// Clamp i32 accumulators to INT8 range (requantize).
pub fn requantize(acc: &[i32]) -> Vec<i8> {
    acc.iter().map(|&v| v.clamp(-128, 127) as i8).collect()
}
