//! Canonical RoPE (Rotary Position Embedding) recomputation for verification.
//!
//! RoPE applies a position-dependent rotation to each pair of elements
//! in the Q and K vectors after the QKV projection:
//!
//!   For each head, for each pair (q[2i], q[2i+1]):
//!     theta_i = position * base^(-2i/d_head)
//!     q_rope[2i]   = q[2i]   * cos(theta_i) - q[2i+1] * sin(theta_i)
//!     q_rope[2i+1] = q[2i]   * sin(theta_i) + q[2i+1] * cos(theta_i)
//!
//! The verifier recomputes RoPE from the position index and model config
//! to verify that K values in the KV cache are correctly derived from
//! the raw matmul accumulators.
//!
//! All arithmetic is f64 for deterministic cross-platform results.

use crate::constants::ModelConfig;

/// Apply RoPE to a single vector (Q or K for one head).
///
/// `head_vec` has length `d_head`. RoPE rotates pairs of elements
/// using position-dependent frequencies.
///
/// Uses the **half-rotary** convention (GPT-NeoX / LLaMA / Qwen style):
/// pair `(head_vec[i], head_vec[i + half])` is rotated by
/// `position * theta^(-2i / d_head)`.
///
/// Returns the rotated vector in f64.
pub fn apply_rope_head(head_vec: &[f64], position: usize, d_head: usize, theta: f64) -> Vec<f64> {
    // Standard (unscaled) frequencies
    let half = d_head / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|k| 1.0 / theta.powf((2 * k) as f64 / d_head as f64))
        .collect();
    apply_rope_head_with_inv_freq(head_vec, position, d_head, &inv_freq)
}

/// Apply RoPE using precomputed (possibly scaled) inverse frequencies.
///
/// `inv_freq` has length `d_head / 2`. Each entry is the inverse frequency
/// for that dimension pair. For standard RoPE this is `1 / theta^(2k/d)`.
/// For Llama3-scaled RoPE, the frequencies are modified per the scaling config.
pub fn apply_rope_head_with_inv_freq(
    head_vec: &[f64],
    position: usize,
    d_head: usize,
    inv_freq: &[f64],
) -> Vec<f64> {
    assert_eq!(head_vec.len(), d_head);
    let half = d_head / 2;
    assert_eq!(inv_freq.len(), half);
    let mut out = vec![0.0f64; d_head];

    for i in 0..half {
        let angle = (position as f64) * inv_freq[i];
        let cos_f = angle.cos();
        let sin_f = angle.sin();
        out[i] = head_vec[i] * cos_f - head_vec[i + half] * sin_f;
        out[i + half] = head_vec[i + half] * cos_f + head_vec[i] * sin_f;
    }

    out
}

/// Apply RoPE to a full Q vector (all heads concatenated).
///
/// `q_f64` has length `n_heads * d_head`. Each head is rotated independently
/// using the same position-dependent frequencies. When `cfg.rope_scaling` is
/// set, uses the scaled inverse frequencies (e.g. Llama3 extended context).
pub fn apply_rope_q(q_f64: &[f64], position: usize, cfg: &ModelConfig) -> Vec<f64> {
    let d = cfg.d_head;
    let n_heads = cfg.n_q_heads;
    assert_eq!(q_f64.len(), n_heads * d);

    let inv_freq = cfg.scaled_inv_freq();
    let mut out = Vec::with_capacity(q_f64.len());
    for h in 0..n_heads {
        let head = &q_f64[h * d..(h + 1) * d];
        out.extend(apply_rope_head_with_inv_freq(head, position, d, &inv_freq));
    }
    out
}

/// Apply RoPE to a full K vector (all KV heads concatenated).
///
/// `k_f64` has length `n_kv_heads * d_head`. Uses scaled inverse frequencies
/// when `cfg.rope_scaling` is set.
pub fn apply_rope_k(k_f64: &[f64], position: usize, cfg: &ModelConfig) -> Vec<f64> {
    let d = cfg.d_head;
    let n_kv_heads = cfg.n_kv_heads;
    assert_eq!(k_f64.len(), n_kv_heads * d);

    let inv_freq = cfg.scaled_inv_freq();
    let mut out = Vec::with_capacity(k_f64.len());
    for h in 0..n_kv_heads {
        let head = &k_f64[h * d..(h + 1) * d];
        out.extend(apply_rope_head_with_inv_freq(head, position, d, &inv_freq));
    }
    out
}

/// Dequantize i32 accumulators to f64 using activation and weight scales.
///
/// `f64[i] = acc[i] * scale_w * scale_x`
///
/// When scales are None (toy model), uses unit scale (identity).
pub fn dequantize_acc(acc: &[i32], scale_w: Option<f32>, scale_x: Option<f32>) -> Vec<f64> {
    let sw = scale_w.unwrap_or(1.0) as f64;
    let sx = scale_x.unwrap_or(1.0) as f64;
    let scale = sw * sx;
    acc.iter().map(|&v| (v as f64) * scale).collect()
}

/// Dequantize i32 accumulators using per-channel weight scales.
///
/// `f64[i] = acc[i] * scale_w[i] * scale_x`
///
/// Used when the model has per-channel weight quantization (e.g., W8A8
/// with `weight_scale` shape `[out_features]`). The VerifierKey's
/// per-tensor scalar scale is incorrect for these models — it stores
/// `0.0` for native INT8 weights, zeroing all dequantized values.
///
/// Measurement-only — not part of the verification protocol.
pub fn dequantize_acc_per_channel(acc: &[i32], scale_w: &[f32], scale_x: f32) -> Vec<f64> {
    assert_eq!(
        acc.len(),
        scale_w.len(),
        "dequantize_acc_per_channel: acc len {} != scale_w len {}",
        acc.len(),
        scale_w.len()
    );
    let sx = scale_x as f64;
    acc.iter()
        .zip(scale_w.iter())
        .map(|(&a, &sw)| (a as f64) * (sw as f64) * sx)
        .collect()
}

/// Verify that a KV cache K entry matches the RoPE'd and requantized K projection.
///
/// Steps:
///   1. Dequantize k_i32 to f64 using scales
///   2. Apply RoPE at the given position
///   3. Requantize to i8
///   4. Compare against kv_cache_k entry
///
/// Returns None if match, Some(description) if mismatch.
pub fn verify_k_rope(
    k_i32: &[i32],
    kv_cache_k_entry: &[i8],
    position: usize,
    cfg: &ModelConfig,
    scale_w: Option<f32>,
    scale_x: Option<f32>,
    scale_next: Option<f32>,
) -> Option<String> {
    let k_f64 = dequantize_acc(k_i32, scale_w, scale_x);
    let k_roped = apply_rope_k(&k_f64, position, cfg);

    // Requantize to i8
    let expected: Vec<i8> = if let Some(s) = scale_next {
        crate::rmsnorm::quantize_f64_to_i8(&k_roped, s as f64)
    } else {
        // Toy model: simple clamp
        k_roped
            .iter()
            .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
            .collect()
    };

    if expected.len() != kv_cache_k_entry.len() {
        return Some(format!(
            "K length mismatch: expected {} got {}",
            expected.len(),
            kv_cache_k_entry.len()
        ));
    }

    let max_diff = expected
        .iter()
        .zip(kv_cache_k_entry.iter())
        .map(|(&e, &a)| (e as i16 - a as i16).abs())
        .max()
        .unwrap_or(0);

    if max_diff > 0 {
        Some(format!("K RoPE mismatch: max_diff={}", max_diff))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_position_zero() {
        // At position 0, all frequencies are 0, cos(0)=1, sin(0)=0
        // So RoPE is identity.
        let head = vec![1.0, 2.0, 3.0, 4.0];
        let out = apply_rope_head(&head, 0, 4, 10000.0);
        for (a, b) in out.iter().zip(head.iter()) {
            assert!((a - b).abs() < 1e-10, "position 0 should be identity");
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation — it preserves the L2 norm of each pair.
        // Half convention: pairs are (0, 2) and (1, 3) for d_head=4.
        let head = vec![3.0, 4.0, 1.0, 2.0];
        let out = apply_rope_head(&head, 5, 4, 10000.0);

        let norm_before_0 = (head[0] * head[0] + head[2] * head[2]).sqrt();
        let norm_after_0 = (out[0] * out[0] + out[2] * out[2]).sqrt();
        assert!(
            (norm_before_0 - norm_after_0).abs() < 1e-10,
            "RoPE should preserve pair norm"
        );

        let norm_before_1 = (head[1] * head[1] + head[3] * head[3]).sqrt();
        let norm_after_1 = (out[1] * out[1] + out[3] * out[3]).sqrt();
        assert!(
            (norm_before_1 - norm_after_1).abs() < 1e-10,
            "RoPE should preserve pair norm"
        );
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let head = vec![1.0, 0.0, 1.0, 0.0];
        let out_p1 = apply_rope_head(&head, 1, 4, 10000.0);
        let out_p2 = apply_rope_head(&head, 2, 4, 10000.0);
        // Different positions should give different results
        assert!(
            (out_p1[0] - out_p2[0]).abs() > 1e-6,
            "different positions should produce different outputs"
        );
    }

    #[test]
    fn test_apply_rope_q_length() {
        let cfg = ModelConfig::toy();
        let q = vec![1.0f64; cfg.n_q_heads * cfg.d_head];
        let out = apply_rope_q(&q, 0, &cfg);
        assert_eq!(out.len(), q.len());
    }

    #[test]
    fn test_dequantize_unit_scale() {
        let acc = vec![10, -20, 30];
        let out = dequantize_acc(&acc, None, None);
        assert_eq!(out, vec![10.0, -20.0, 30.0]);
    }

    #[test]
    fn test_dequantize_with_scales() {
        let acc = vec![100, -200];
        let out = dequantize_acc(&acc, Some(0.5), Some(0.1));
        // 100 * 0.5 * 0.1 = 5.0, -200 * 0.5 * 0.1 = -10.0
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_per_channel() {
        let acc = vec![100, -200, 50];
        let scale_w = vec![0.5, 0.1, 0.3];
        let scale_x = 0.2;
        let out = dequantize_acc_per_channel(&acc, &scale_w, scale_x);
        // 100 * 0.5 * 0.2 = 10.0
        // -200 * 0.1 * 0.2 = -4.0
        // 50 * 0.3 * 0.2 = 3.0
        assert!((out[0] - 10.0).abs() < 1e-6);
        assert!((out[1] - (-4.0)).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "acc len 2 != scale_w len 1")]
    fn test_dequantize_per_channel_length_mismatch() {
        let acc = vec![100, -200];
        let scale_w = vec![0.5]; // wrong length
        dequantize_acc_per_channel(&acc, &scale_w, 1.0);
    }

    // ── Llama3 RoPE scaling tests ──

    #[test]
    fn test_scaled_inv_freq_no_scaling() {
        // Without rope_scaling, scaled_inv_freq should match standard formula
        let cfg = ModelConfig::toy(); // d_head=2, theta=10000.0
        let inv_freq = cfg.scaled_inv_freq();
        assert_eq!(inv_freq.len(), 1); // d_head/2 = 1
        let expected = 1.0 / 10000.0_f64.powf(0.0 / 2.0); // = 1.0
        assert!((inv_freq[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_scaled_inv_freq_llama3_bands() {
        use crate::constants::RopeScaling;

        // Llama 3.1 8B config: d_head=128, theta=500000, factor=8
        let cfg = ModelConfig {
            name: "test-llama3".into(),
            hidden_dim: 4096,
            kv_dim: 1024,
            ffn_dim: 14336,
            d_head: 128,
            n_layers: 32,
            n_q_heads: 32,
            n_kv_heads: 8,
            vocab_size: 128256,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                rope_type: "llama3".into(),
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            }),
        };

        let scaled = cfg.scaled_inv_freq();
        assert_eq!(scaled.len(), 64); // 128/2

        // Compute unscaled for comparison
        let mut unscaled_cfg = cfg.clone();
        unscaled_cfg.rope_scaling = None;
        let unscaled = unscaled_cfg.scaled_inv_freq();

        let low_freq_wavelen = 8192.0 / 1.0;
        let high_freq_wavelen = 8192.0 / 4.0;

        for k in 0..64 {
            let wavelen = 2.0 * std::f64::consts::PI / unscaled[k];
            if wavelen < high_freq_wavelen {
                // High freq: unchanged
                assert!(
                    (scaled[k] - unscaled[k]).abs() < 1e-15,
                    "dim {} (wavelen={:.0}) should be unchanged",
                    k,
                    wavelen
                );
            } else if wavelen > low_freq_wavelen {
                // Low freq: divided by factor
                assert!(
                    (scaled[k] - unscaled[k] / 8.0).abs() < 1e-15,
                    "dim {} (wavelen={:.0}) should be /8",
                    k,
                    wavelen
                );
            } else {
                // Medium: interpolated — should be between scaled and unscaled
                assert!(
                    scaled[k] >= unscaled[k] / 8.0 - 1e-15 && scaled[k] <= unscaled[k] + 1e-15,
                    "dim {} (wavelen={:.0}) should be interpolated: got {}, range [{}, {}]",
                    k,
                    wavelen,
                    scaled[k],
                    unscaled[k] / 8.0,
                    unscaled[k]
                );
            }
        }
    }

    #[test]
    fn test_rope_with_scaling_differs_from_unscaled() {
        use crate::constants::RopeScaling;

        let cfg_scaled = ModelConfig {
            name: "scaled".into(),
            hidden_dim: 256,
            kv_dim: 64,
            ffn_dim: 512,
            d_head: 64,
            n_layers: 1,
            n_q_heads: 4,
            n_kv_heads: 1,
            vocab_size: 100,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                rope_type: "llama3".into(),
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            }),
        };
        let mut cfg_unscaled = cfg_scaled.clone();
        cfg_unscaled.rope_scaling = None;

        // At position 0, both should be identity
        let head = vec![1.0f64; 64];
        let out_scaled_0 = apply_rope_q(&head.repeat(4), 0, &cfg_scaled);
        let out_unscaled_0 = apply_rope_q(&head.repeat(4), 0, &cfg_unscaled);
        let diff_0: f64 = out_scaled_0
            .iter()
            .zip(out_unscaled_0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(diff_0 < 1e-10, "position 0 should be identical");

        // At a long position (e.g. 4000), they should differ
        let q = vec![1.0f64; 256];
        let out_scaled = apply_rope_q(&q, 4000, &cfg_scaled);
        let out_unscaled = apply_rope_q(&q, 4000, &cfg_unscaled);
        let diff: f64 = out_scaled
            .iter()
            .zip(out_unscaled.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            diff > 0.01,
            "at position 4000, scaled and unscaled should differ materially, got diff={}",
            diff
        );
    }

    #[test]
    fn test_rope_scaling_preserves_norm() {
        use crate::constants::RopeScaling;

        let cfg = ModelConfig {
            name: "norm-test".into(),
            hidden_dim: 128,
            kv_dim: 128,
            ffn_dim: 256,
            d_head: 128,
            n_layers: 1,
            n_q_heads: 1,
            n_kv_heads: 1,
            vocab_size: 100,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                rope_type: "llama3".into(),
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            }),
        };

        // RoPE is still a rotation even with scaling — must preserve L2 norm
        let q: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let norm_before: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt();

        let out = apply_rope_q(&q, 5000, &cfg);
        let norm_after: f64 = out.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert!(
            (norm_before - norm_after).abs() < 1e-10,
            "scaled RoPE must preserve L2 norm: before={}, after={}",
            norm_before,
            norm_after
        );
    }
}
