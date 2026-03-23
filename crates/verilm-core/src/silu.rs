//! SiLU (Swish) verification via INT8 lookup table.
//!
//! In INT8 inference, SiLU is applied to the gate output g:
//!   h[i] = requant(SiLU(dequant(g[i])) * dequant(u[i]))
//!
//! Since g is INT8, SiLU(dequant(g)) has only 256 possible values.
//! We precompute a LUT at keygen time and verify against it.

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
}
