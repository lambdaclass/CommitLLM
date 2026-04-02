//! SiLU (Swish) verification for both toy and production INT8 paths.
//!
//! In the toy/unit-scale path, the gate input is first clamped to INT8:
//!   h[i] = requant(SiLU(dequant(g_i8[i])) * dequant(u_i8[i]))
//!
//! Since `g_i8` has only 256 possible values, this path admits a 256-entry LUT.
//!
//! In the production W8A8 path, the verifier instead uses the opened i32
//! accumulators together with the recorded scales:
//!   g_f[i] = g_i32[i] * scale_w_g * scale_x_ffn
//!   u_f[i] = u_i32[i] * scale_w_u * scale_x_ffn
//!   h_i8[i] = requant(SiLU(g_f[i]) * u_f[i])
//!
//! So the LUT is only for the toy path; the production verifier canonically
//! recomputes the scaled floating-point bridge and checks the requantized i8
//! output.

/// Build the SiLU LUT for a given quantization scale.
/// Maps each INT8 value g in -128..127 to SiLU(g * scale).
/// Returns the dequantized SiLU values as f32 (to be used in verification).
pub fn build_silu_lut(scale: f32) -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for i in 0..256u16 {
        let g = i as i8; // wraps: 0..127 -> 0..127, 128..255 -> -128..-1
        let x = g as f32 * scale;
        lut[i as usize] = silu(x);
    }
    lut
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Compute h = SiLU(requant(g)) * requant(u), requantized to i8.
///
/// Takes i32 accumulators from the gate and up projections. Requantizes
/// them to i8 (clamp), looks up SiLU from a 256-entry LUT, multiplies
/// by the up value, and clamps the result to i8.
///
/// Uses unit quantization scale (scale=1.0, zero=0). This is the
/// canonical computation for the toy model and any path where
/// quantization scales are not tracked.
pub fn compute_h_unit_scale(g_acc: &[i32], u_acc: &[i32]) -> Vec<i8> {
    assert_eq!(g_acc.len(), u_acc.len());
    let lut = build_silu_lut(1.0);
    g_acc
        .iter()
        .zip(u_acc.iter())
        .map(|(&g, &u)| {
            let g_i8 = g.clamp(-128, 127) as i8;
            let u_i8 = u.clamp(-128, 127) as i8;
            let silu_g = lut[g_i8 as u8 as usize];
            let product = silu_g * u_i8 as f32;
            product.round().clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Scale-aware SiLU bridge: dequantize g,u accumulators, apply SiLU, requantize.
///
/// For W8A8 models where the gate/up accumulators need proper dequantization:
///   g_f = g_i32 * scale_w_g * scale_x_ffn
///   u_f = u_i32 * scale_w_u * scale_x_ffn
///   h_f = SiLU(g_f) * u_f
///   h_i8 = round(h_f / scale_h).clamp(-128, 127)
///
/// For native INT8 (scale_w_g == 0.0): falls back to `compute_h_unit_scale`.
pub fn compute_h_scaled(
    g_acc: &[i32],
    u_acc: &[i32],
    scale_w_g: f32,
    scale_w_u: f32,
    scale_x_ffn: f32,
    scale_h: f32,
) -> Vec<i8> {
    if scale_w_g == 0.0 {
        return compute_h_unit_scale(g_acc, u_acc);
    }
    assert_eq!(g_acc.len(), u_acc.len());
    let dequant_g = (scale_w_g as f64) * (scale_x_ffn as f64);
    let dequant_u = (scale_w_u as f64) * (scale_x_ffn as f64);
    let inv_scale_h = 1.0 / (scale_h as f64);
    g_acc
        .iter()
        .zip(u_acc.iter())
        .map(|(&g, &u)| {
            let g_f = g as f64 * dequant_g;
            let u_f = u as f64 * dequant_u;
            let h_f = silu_f64(g_f) * u_f;
            (h_f * inv_scale_h).round().clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Per-channel SiLU bridge for W8A8 models with per-channel weight scales.
///
/// Same computation as `compute_h_scaled` but uses per-channel weight scales:
///   g_f[i] = g_i32[i] * scale_w_g[i] * scale_x_ffn
///   u_f[i] = u_i32[i] * scale_w_u[i] * scale_x_ffn
///   h_f[i] = SiLU(g_f[i]) * u_f[i]
///   h_i8[i] = round(h_f[i] / scale_h).clamp(-128, 127)
pub fn compute_h_per_channel(
    g_acc: &[i32],
    u_acc: &[i32],
    scale_w_g: &[f32],
    scale_w_u: &[f32],
    scale_x_ffn: f32,
    scale_h: f32,
) -> Vec<i8> {
    assert_eq!(g_acc.len(), u_acc.len());
    assert_eq!(g_acc.len(), scale_w_g.len());
    assert_eq!(u_acc.len(), scale_w_u.len());
    let sx = scale_x_ffn as f64;
    let inv_scale_h = 1.0 / (scale_h as f64);
    g_acc
        .iter()
        .zip(u_acc.iter())
        .zip(scale_w_g.iter().zip(scale_w_u.iter()))
        .map(|((&g, &u), (&swg, &swu))| {
            let g_f = g as f64 * (swg as f64) * sx;
            let u_f = u as f64 * (swu as f64) * sx;
            let h_f = silu_f64(g_f) * u_f;
            (h_f * inv_scale_h).round().clamp(-128.0, 127.0) as i8
        })
        .collect()
}

fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Verify the SiLU + elementwise multiply step.
///
/// Given g (gate), u (up), h (output), and quantization parameters,
/// check that h[i] = requant(SiLU(dequant(g[i])) * dequant(u[i])).
///
/// Returns true if all elements match within requantization tolerance (+-1).
pub fn check_silu(
    g: &[i8],
    u: &[i8],
    h: &[i8],
    lut: &[f32; 256],
    u_scale: f32,
    h_scale: f32,
    h_zero: i8,
) -> bool {
    assert_eq!(g.len(), u.len());
    assert_eq!(g.len(), h.len());

    for i in 0..g.len() {
        let g_idx = g[i] as u8 as usize;
        let silu_g = lut[g_idx];
        let u_val = u[i] as f32 * u_scale;
        let product = silu_g * u_val;

        // Requantize to INT8
        let expected = (product / h_scale).round() as i32 + h_zero as i32;
        let expected_clamped = expected.clamp(-128, 127) as i8;
        let actual = h[i];

        // Allow +-1 for rounding differences
        let diff = (actual as i32 - expected_clamped as i32).abs();
        if diff > 1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // SiLU(1) = 1 / (1 + e^-1) ≈ 0.7311
        assert!((silu(1.0) - 0.7311).abs() < 0.001);
        // SiLU(-3) ≈ -3 * sigmoid(-3) ≈ -3 * 0.0474 ≈ -0.1423
        assert!((silu(-3.0) - (-0.1423)).abs() < 0.001);
    }

    #[test]
    fn test_build_lut() {
        let lut = build_silu_lut(0.1);
        // g=0 -> SiLU(0) = 0
        assert!((lut[0] - 0.0).abs() < 1e-6);
        // g=10 -> SiLU(1.0) ≈ 0.7311
        assert!((lut[10] - 0.7311).abs() < 0.001);
    }

    #[test]
    fn test_check_silu_correct() {
        let scale = 0.1f32;
        let lut = build_silu_lut(scale);

        let g = vec![10i8, 20, -5];
        let u = vec![5i8, 10, -3];
        let u_scale = 0.1f32;
        let h_scale = 0.01f32;
        let h_zero = 0i8;

        // Compute expected h
        let h: Vec<i8> = g
            .iter()
            .zip(u.iter())
            .map(|(&gi, &ui)| {
                let silu_g = lut[gi as u8 as usize];
                let u_val = ui as f32 * u_scale;
                let product = silu_g * u_val;
                (product / h_scale).round().clamp(-128.0, 127.0) as i8
            })
            .collect();

        assert!(check_silu(&g, &u, &h, &lut, u_scale, h_scale, h_zero));
    }

    #[test]
    fn test_compute_h_scaled_fallback() {
        // With scale_w_g == 0.0, should produce same result as compute_h_unit_scale.
        let g_acc = vec![10, -5, 127, -128, 200];
        let u_acc = vec![20, 30, -10, 50, -60];
        let unit = compute_h_unit_scale(&g_acc, &u_acc);
        let scaled = compute_h_scaled(&g_acc, &u_acc, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(
            unit, scaled,
            "zero weight scale should fall back to unit-scale"
        );
    }

    #[test]
    fn test_compute_h_scaled_known_values() {
        // Manual calculation with known scales.
        // g_acc = [100], u_acc = [50]
        // scale_w_g = 0.01, scale_w_u = 0.02, scale_x_ffn = 0.5, scale_h = 0.1
        // g_f = 100 * 0.01 * 0.5 = 0.5
        // u_f = 50 * 0.02 * 0.5 = 0.5
        // SiLU(0.5) = 0.5 / (1 + e^-0.5) ≈ 0.5 * 0.6225 = 0.31122
        // h_f = 0.31122 * 0.5 = 0.15561
        // h_i8 = round(0.15561 / 0.1) = round(1.5561) = 2
        let result = compute_h_scaled(&[100], &[50], 0.01, 0.02, 0.5, 0.1);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_compute_h_scaled_clamps() {
        // Large values should clamp to i8 range.
        let result = compute_h_scaled(&[10000], &[10000], 1.0, 1.0, 1.0, 0.001);
        assert_eq!(result, vec![127]);
    }
}
