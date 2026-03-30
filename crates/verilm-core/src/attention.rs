//! Attention replay for Level C verification.
//!
//! The verifier replays GQA attention from the trace's requantized Q, K, V
//! vectors and compares against the claimed attention output `a`. Level A/B
//! trust `a` blindly; Level C replays it from the provided KV snapshot.
//!
//! The KV cache is partially bound to prior-token computation via prefix-KV
//! chain binding: each opened token's own KV cache entry is verified against
//! its requantized K/V projection (self-consistency, always checkable), and
//! when multiple tokens are co-opened, cross-token entries are verified against
//! the prior token's requantized K/V projection. Unopened tokens' KV cache
//! entries cannot be cross-checked — this is inherent to sampled opening.
//!
//! # Architecture
//!
//! The module separates **replay** from **tolerance policy**:
//!
//! - [`replay_attention_reference`] — canonical f64 reference implementation.
//!   This is the ground-truth replay used for INT8 toy models and as the
//!   oracle for differential testing. A production FP16/BF16 replay would
//!   be a separate function with matching arithmetic.
//!
//! - [`compare_attention_output`] — tolerance-aware comparison between the
//!   claimed `a` vector and the replayed output. Exact for the toy model
//!   (tolerance=0), tolerant for mixed-precision production paths.

use crate::constants::ModelConfig;

/// Tolerance configuration for attention output comparison.
///
/// Controls how [`compare_attention_output`] judges agreement between
/// claimed and replayed attention vectors.
#[derive(Debug, Clone, Copy)]
pub struct AttentionToleranceConfig {
    /// Maximum allowed L-infinity difference between claimed and replayed
    /// attention output `a` (per element, in i8 space).
    /// 0 = exact match (appropriate for deterministic INT8 reference path).
    /// 1-2 = appropriate for FP16/BF16 hardware nondeterminism.
    pub max_abs_diff: u8,
}

impl Default for AttentionToleranceConfig {
    fn default() -> Self {
        AttentionToleranceConfig { max_abs_diff: 0 }
    }
}

/// Canonical f64 reference implementation of GQA attention replay.
///
/// Replays `softmax(QK^T / sqrt(d)) V` using f64 arithmetic, which is
/// exact for the INT8 toy model. This is the **reference oracle** — any
/// production replay (FP16, BF16, INT8-only) should be differential-tested
/// against this function.
///
/// For production mixed-precision replay, add a separate
/// `replay_attention_fp16(...)` function and use a nonzero tolerance
/// in [`compare_attention_output`].
///
/// # Arguments
/// - `q_i8` — requantized Q vector, length `hidden_dim`
/// - `kv_cache_k` — K vectors for all seq positions, each length `kv_dim`
/// - `kv_cache_v` — V vectors for all seq positions, each length `kv_dim`
/// - `cfg` — model config (d_head, n_q_heads, n_kv_heads)
///
/// # Returns
/// Expected attention output vector, length `hidden_dim`, as i8.
pub fn replay_attention_reference(
    q_i8: &[i8],
    kv_cache_k: &[Vec<i8>],
    kv_cache_v: &[Vec<i8>],
    cfg: &ModelConfig,
) -> Vec<i8> {
    let d_head = cfg.d_head;
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let inv_sqrt_d = 1.0 / (d_head as f64).sqrt();
    let seq_len = kv_cache_k.len();

    let mut a = vec![0i8; cfg.hidden_dim];

    for qh in 0..cfg.n_q_heads {
        let kv_head = qh / heads_per_kv;

        // Extract Q head
        let q_head: Vec<f64> = (0..d_head)
            .map(|i| q_i8[qh * d_head + i] as f64)
            .collect();

        // Compute attention scores: score[t] = q · k_t / sqrt(d)
        let scores: Vec<f64> = (0..seq_len)
            .map(|t| {
                let k_t = &kv_cache_k[t];
                let dot: f64 = (0..d_head)
                    .map(|i| q_head[i] * k_t[kv_head * d_head + i] as f64)
                    .sum();
                dot * inv_sqrt_d
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum of V cache
        let mut head_out = vec![0.0f64; d_head];
        for t in 0..seq_len {
            let v_t = &kv_cache_v[t];
            for i in 0..d_head {
                head_out[i] += weights[t] * v_t[kv_head * d_head + i] as f64;
            }
        }

        // Requantize head output to i8
        for i in 0..d_head {
            a[qh * d_head + i] = head_out[i].round().clamp(-128.0, 127.0) as i8;
        }
    }

    a
}

/// RoPE-aware f64 attention replay for production models.
///
/// Takes post-RoPE Q and K in f64, dequantized V in f64, and requantizes
/// the output to i8 using `scale_a`. This is the production counterpart to
/// [`replay_attention_reference`] which uses raw i8 inputs (toy model).
///
/// The caller is responsible for:
/// - Dequantizing i32 accumulators via [`crate::rope::dequantize_acc`]
/// - Applying RoPE via [`crate::rope::apply_rope_q`] / [`crate::rope::apply_rope_k`]
/// - Passing the correct `scale_a` from `RetainedLayerState`
///
/// # Arguments
/// - `q_roped` — post-RoPE Q, length `hidden_dim` (f64)
/// - `kv_cache_k_roped` — post-RoPE K entries per seq position, each length `kv_dim`
/// - `kv_cache_v_deq` — dequantized V entries (no RoPE), each length `kv_dim`
/// - `scale_a` — quantization scale for the output `a_i8 = round(a_f64 / scale_a)`
/// - `cfg` — model config
pub fn replay_attention_roped(
    q_roped: &[f64],
    kv_cache_k_roped: &[Vec<f64>],
    kv_cache_v_deq: &[Vec<f64>],
    scale_a: f64,
    cfg: &ModelConfig,
) -> Vec<i8> {
    let d_head = cfg.d_head;
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let inv_sqrt_d = 1.0 / (d_head as f64).sqrt();
    let seq_len = kv_cache_k_roped.len();
    let inv_scale = if scale_a.abs() > 1e-30 { 1.0 / scale_a } else { 1.0 };

    let mut a = vec![0i8; cfg.hidden_dim];

    for qh in 0..cfg.n_q_heads {
        let kv_head = qh / heads_per_kv;

        let q_head: Vec<f64> = (0..d_head)
            .map(|i| q_roped[qh * d_head + i])
            .collect();

        // Attention scores: q · k_t / sqrt(d)
        let scores: Vec<f64> = (0..seq_len)
            .map(|t| {
                let k_t = &kv_cache_k_roped[t];
                let dot: f64 = (0..d_head)
                    .map(|i| q_head[i] * k_t[kv_head * d_head + i])
                    .sum();
                dot * inv_sqrt_d
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum of V
        let mut head_out = vec![0.0f64; d_head];
        for t in 0..seq_len {
            let v_t = &kv_cache_v_deq[t];
            for i in 0..d_head {
                head_out[i] += weights[t] * v_t[kv_head * d_head + i];
            }
        }

        // Requantize: a_i8 = round(a_f64 / scale_a)
        for i in 0..d_head {
            a[qh * d_head + i] = (head_out[i] * inv_scale)
                .round()
                .clamp(-128.0, 127.0) as i8;
        }
    }

    a
}

/// Like [`replay_attention_roped`] but also returns the pre-quantization f64 output.
///
/// Returns `(a_i8, a_f64)` where `a_f64` is the raw attention output before
/// requantization. Used by corridor diagnostics to separate V-dequant bugs
/// from output-quantization bugs.
pub fn replay_attention_roped_raw(
    q_roped: &[f64],
    kv_cache_k_roped: &[Vec<f64>],
    kv_cache_v_deq: &[Vec<f64>],
    scale_a: f64,
    cfg: &ModelConfig,
) -> (Vec<i8>, Vec<f64>) {
    let d_head = cfg.d_head;
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let inv_sqrt_d = 1.0 / (d_head as f64).sqrt();
    let seq_len = kv_cache_k_roped.len();
    let inv_scale = if scale_a.abs() > 1e-30 { 1.0 / scale_a } else { 1.0 };

    let mut a_i8 = vec![0i8; cfg.hidden_dim];
    let mut a_f64 = vec![0.0f64; cfg.hidden_dim];

    for qh in 0..cfg.n_q_heads {
        let kv_head = qh / heads_per_kv;

        let q_head: Vec<f64> = (0..d_head)
            .map(|i| q_roped[qh * d_head + i])
            .collect();

        let scores: Vec<f64> = (0..seq_len)
            .map(|t| {
                let k_t = &kv_cache_k_roped[t];
                let dot: f64 = (0..d_head)
                    .map(|i| q_head[i] * k_t[kv_head * d_head + i])
                    .sum();
                dot * inv_sqrt_d
            })
            .collect();

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        let mut head_out = vec![0.0f64; d_head];
        for t in 0..seq_len {
            let v_t = &kv_cache_v_deq[t];
            for i in 0..d_head {
                head_out[i] += weights[t] * v_t[kv_head * d_head + i];
            }
        }

        for i in 0..d_head {
            let idx = qh * d_head + i;
            a_f64[idx] = head_out[i];
            a_i8[idx] = (head_out[i] * inv_scale)
                .round()
                .clamp(-128.0, 127.0) as i8;
        }
    }

    (a_i8, a_f64)
}

/// Per-element diff statistics between claimed and replayed attention outputs.
///
/// Used by corridor measurement to characterize the FP16-vs-f64 divergence
/// without a pass/fail threshold.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AttentionDiffStats {
    pub layer: usize,
    pub token_position: usize,
    /// Max |claimed - replayed| across all elements.
    pub linf: i16,
    /// Mean of |claimed - replayed|.
    pub mean_abs: f64,
    /// Root mean square of element-wise diffs.
    pub rms: f64,
    /// 95th percentile of |claimed - replayed|.
    pub p95_abs_diff: i16,
    /// Fraction of elements with diff == 0.
    pub frac_eq: f64,
    /// Fraction of elements with diff <= 1.
    pub frac_le_1: f64,
    /// Fraction of elements with diff <= 2.
    pub frac_le_2: f64,
    /// Histogram: count of elements with diff = 0, 1, 2, 3, 4, 5+.
    pub histogram: [usize; 6],
    pub n_elements: usize,
}

/// Compute element-wise diff statistics between claimed and replayed attention outputs.
///
/// Pure comparison — no replay logic. Sits next to [`compare_attention_output`].
/// Returns `Err` if lengths differ.
pub fn measure_attention_diff(
    claimed: &[i8],
    replayed: &[i8],
    layer: usize,
    token_position: usize,
) -> Result<AttentionDiffStats, String> {
    if claimed.len() != replayed.len() {
        return Err(format!(
            "length mismatch: claimed={} replayed={}",
            claimed.len(),
            replayed.len()
        ));
    }
    let n = claimed.len();
    let mut linf: i16 = 0;
    let mut sum_abs: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut histogram = [0usize; 6];
    let mut all_diffs: Vec<i16> = Vec::with_capacity(n);

    for (&c, &r) in claimed.iter().zip(replayed.iter()) {
        let diff = (c as i16 - r as i16).abs();
        if diff > linf {
            linf = diff;
        }
        sum_abs += diff as f64;
        sum_sq += (diff as f64) * (diff as f64);
        let bucket = (diff as usize).min(5);
        histogram[bucket] += 1;
        all_diffs.push(diff);
    }

    // p95: sort and pick the 95th percentile element
    all_diffs.sort_unstable();
    let p95_abs_diff = if n > 0 {
        let idx = ((n as f64 * 0.95).ceil() as usize).min(n) - 1;
        all_diffs[idx]
    } else {
        0
    };

    let n_f = n as f64;
    let frac_eq = if n > 0 { histogram[0] as f64 / n_f } else { 0.0 };
    let frac_le_1 = if n > 0 { (histogram[0] + histogram[1]) as f64 / n_f } else { 0.0 };
    let frac_le_2 = if n > 0 {
        (histogram[0] + histogram[1] + histogram[2]) as f64 / n_f
    } else {
        0.0
    };

    Ok(AttentionDiffStats {
        layer,
        token_position,
        linf,
        mean_abs: if n > 0 { sum_abs / n_f } else { 0.0 },
        rms: if n > 0 { (sum_sq / n_f).sqrt() } else { 0.0 },
        p95_abs_diff,
        frac_eq,
        frac_le_1,
        frac_le_2,
        histogram,
        n_elements: n,
    })
}

/// Compare claimed vs replayed attention output with tolerance policy.
///
/// Returns `None` if the vectors have equal length and the L-infinity difference
/// is within `tolerance.max_abs_diff`. Returns `Some(i16::MAX)` if lengths differ
/// (malformed input), or `Some(max_diff)` if the tolerance is exceeded.
pub fn compare_attention_output(claimed: &[i8], replayed: &[i8], tolerance: &AttentionToleranceConfig) -> Option<i16> {
    if claimed.len() != replayed.len() {
        return Some(i16::MAX);
    }

    let max_diff = claimed
        .iter()
        .zip(replayed.iter())
        .map(|(&c, &r)| (c as i16 - r as i16).abs())
        .max()
        .unwrap_or(0);

    if max_diff > tolerance.max_abs_diff as i16 {
        Some(max_diff)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_cfg() -> ModelConfig {
        ModelConfig::toy()
    }

    #[test]
    fn test_single_token_single_attention() {
        // With one token, softmax([score]) = [1.0], so output = V.
        let cfg = toy_cfg();
        let q_i8 = vec![10i8; cfg.hidden_dim];
        let k = vec![5i8; cfg.kv_dim];
        let v = vec![3i8; cfg.kv_dim];

        let result = replay_attention_reference(&q_i8, &[k], &[v.clone()], &cfg);

        // Each query head maps to a KV head. With one token, softmax = [1.0],
        // so output for each query head = V values from its KV head.
        let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            for i in 0..cfg.d_head {
                assert_eq!(
                    result[qh * cfg.d_head + i],
                    v[kv_head * cfg.d_head + i],
                    "head {} dim {}: expected V value",
                    qh,
                    i
                );
            }
        }
    }

    #[test]
    fn test_two_tokens_uniform_scores() {
        // Q = zeros → all scores = 0 → softmax = uniform → output = mean(V)
        let cfg = toy_cfg();
        let q_i8 = vec![0i8; cfg.hidden_dim];
        let k0 = vec![1i8; cfg.kv_dim];
        let k1 = vec![2i8; cfg.kv_dim];
        let v0 = vec![10i8; cfg.kv_dim];
        let v1 = vec![20i8; cfg.kv_dim];

        let result = replay_attention_reference(&q_i8, &[k0, k1], &[v0, v1], &cfg);

        // With zero Q, all dot products are 0, softmax is uniform [0.5, 0.5].
        // Output per KV head dim = (10 + 20) / 2 = 15.0 → 15i8
        let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            for i in 0..cfg.d_head {
                let expected = 15i8; // (10 + 20) / 2
                assert_eq!(
                    result[qh * cfg.d_head + i],
                    expected,
                    "head {} dim {} kv_head {}: expected mean of V",
                    qh,
                    i,
                    kv_head
                );
            }
        }
    }

    #[test]
    fn test_compare_attention_exact_match() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![1i8, 2, 3, 4];
        let tol = AttentionToleranceConfig { max_abs_diff: 0 };
        assert_eq!(compare_attention_output(&a, &b, &tol), None);
    }

    #[test]
    fn test_compare_attention_within_tolerance() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![2i8, 2, 3, 3];
        let tol = AttentionToleranceConfig { max_abs_diff: 1 };
        assert_eq!(compare_attention_output(&a, &b, &tol), None);
    }

    #[test]
    fn test_compare_attention_exceeds_tolerance() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![4i8, 2, 3, 4]; // diff=3 at index 0
        let tol = AttentionToleranceConfig { max_abs_diff: 2 };
        assert_eq!(compare_attention_output(&a, &b, &tol), Some(3));
    }

    #[test]
    fn test_compare_attention_length_mismatch_rejected() {
        // Truncated claimed vector must not silently pass via zip
        let claimed = vec![1i8, 2, 3];
        let replayed = vec![1i8, 2, 3, 4];
        let tol = AttentionToleranceConfig { max_abs_diff: 0 };
        assert_eq!(compare_attention_output(&claimed, &replayed, &tol), Some(i16::MAX));

        // Extended claimed vector is also rejected
        let claimed2 = vec![1i8, 2, 3, 4, 5];
        assert_eq!(compare_attention_output(&claimed2, &replayed, &tol), Some(i16::MAX));
    }

    #[test]
    fn test_measure_diff_exact_match() {
        let a = vec![1i8, 2, 3, -128, 127];
        let b = vec![1i8, 2, 3, -128, 127];
        let stats = measure_attention_diff(&a, &b, 0, 5).unwrap();
        assert_eq!(stats.linf, 0);
        assert_eq!(stats.mean_abs, 0.0);
        assert_eq!(stats.rms, 0.0);
        assert_eq!(stats.p95_abs_diff, 0);
        assert_eq!(stats.frac_eq, 1.0);
        assert_eq!(stats.frac_le_1, 1.0);
        assert_eq!(stats.frac_le_2, 1.0);
        assert_eq!(stats.histogram, [5, 0, 0, 0, 0, 0]);
        assert_eq!(stats.n_elements, 5);
        assert_eq!(stats.layer, 0);
        assert_eq!(stats.token_position, 5);
    }

    #[test]
    fn test_measure_diff_known_values() {
        // diffs: 0, 1, 2, 3, 5, 7
        let claimed  = vec![10i8, 20, 30, 40, 50, 60];
        let replayed = vec![10i8, 19, 28, 37, 45, 53];
        let stats = measure_attention_diff(&claimed, &replayed, 3, 10).unwrap();
        assert_eq!(stats.linf, 7);
        assert_eq!(stats.n_elements, 6);
        // histogram: diff=0 -> 1, diff=1 -> 1, diff=2 -> 1, diff=3 -> 1, diff=4 -> 0, diff=5+ -> 2
        assert_eq!(stats.histogram, [1, 1, 1, 1, 0, 2]);
        // mean_abs = (0+1+2+3+5+7)/6 = 18/6 = 3.0
        assert!((stats.mean_abs - 3.0).abs() < 1e-10);
        // rms = sqrt((0+1+4+9+25+49)/6) = sqrt(88/6) = sqrt(14.666...)
        let expected_rms = (88.0f64 / 6.0).sqrt();
        assert!((stats.rms - expected_rms).abs() < 1e-10);
        // frac_eq = 1/6, frac_le_1 = 2/6, frac_le_2 = 3/6
        assert!((stats.frac_eq - 1.0 / 6.0).abs() < 1e-10);
        assert!((stats.frac_le_1 - 2.0 / 6.0).abs() < 1e-10);
        assert!((stats.frac_le_2 - 3.0 / 6.0).abs() < 1e-10);
        // p95: sorted diffs = [0,1,2,3,5,7], idx = ceil(6*0.95)-1 = ceil(5.7)-1 = 5, diffs[5]=7
        assert_eq!(stats.p95_abs_diff, 7);
    }

    #[test]
    fn test_measure_diff_length_mismatch() {
        let a = vec![1i8, 2, 3];
        let b = vec![1i8, 2];
        assert!(measure_attention_diff(&a, &b, 0, 0).is_err());
    }
}
