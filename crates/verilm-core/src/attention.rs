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
//!
//! Important protocol rule: replayed attention is a **comparison reference**,
//! not a replacement state. If the claimed/opened `a` passes tolerance against
//! the verifier's replay, downstream verification should continue from the
//! claimed/opened value so approximation error does not compound across layers.

use crate::constants::ModelConfig;
use half::f16;

/// Replay precision for attention computation.
///
/// Controls the arithmetic precision used in the score/softmax/V-sum loop.
/// The experiment ladder:
///   F64 — current baseline (over-precise vs GPU)
///   F32 — tests whether f32 accumulation alone closes the gap
///   Fp16InputsF32Accum — fp16 input truncation + f32 accumulation (closest to GPU SDPA)
///   Bf16InputsF32Accum — bf16 input truncation + f32 accumulation (for bf16 models)
///
/// Inputs (Q, K, V) always arrive as f64 from dequant+RoPE. The precision
/// enum controls how they are narrowed before the attention inner loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReplayPrecision {
    /// All arithmetic in f64. Current verifier default.
    F64,
    /// Inputs truncated to f32, scores/softmax/V-sum in f32.
    /// Tests whether over-precise accumulation is the main gap source.
    F32,
    /// Inputs truncated to fp16 (via `half` crate), then promoted back to f32.
    /// Scores/softmax/V-sum in f32. Closest match to GPU SDPA with fp16 tensors.
    Fp16InputsF32Accum,
    /// Inputs truncated to bf16 (via `half` crate), then promoted back to f32.
    /// Scores/softmax/V-sum in f32. For models with `torch_dtype=bfloat16`.
    Bf16InputsF32Accum,
}

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
        let q_head: Vec<f64> = (0..d_head).map(|i| q_i8[qh * d_head + i] as f64).collect();

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

/// Production attention replay using dequantized+RoPE'd f64 inputs.
///
/// Thin wrapper around [`replay_attention_roped_precision`] with
/// [`ReplayPrecision::F64`]. This is the canonical reference replay.
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
    replay_attention_roped_precision(
        q_roped,
        kv_cache_k_roped,
        kv_cache_v_deq,
        scale_a,
        cfg,
        ReplayPrecision::F64,
    )
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
    let inv_scale = if scale_a.abs() > 1e-30 {
        1.0 / scale_a
    } else {
        1.0
    };

    let mut a_i8 = vec![0i8; cfg.hidden_dim];
    let mut a_f64 = vec![0.0f64; cfg.hidden_dim];

    for qh in 0..cfg.n_q_heads {
        let kv_head = qh / heads_per_kv;

        let q_head: Vec<f64> = (0..d_head).map(|i| q_roped[qh * d_head + i]).collect();

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
            a_i8[idx] = (head_out[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (a_i8, a_f64)
}

/// Thin wrapper: f32 replay. See [`replay_attention_roped_precision`].
pub fn replay_attention_roped_f32(
    q_roped: &[f64],
    kv_cache_k_roped: &[Vec<f64>],
    kv_cache_v_deq: &[Vec<f64>],
    scale_a: f64,
    cfg: &ModelConfig,
) -> Vec<i8> {
    replay_attention_roped_precision(
        q_roped,
        kv_cache_k_roped,
        kv_cache_v_deq,
        scale_a,
        cfg,
        ReplayPrecision::F32,
    )
}

/// Narrow an f64 value to the target input precision.
///
/// - F32: f64 → f32
/// - Fp16InputsF32Accum: f64 → f32 → f16 → f32 (real IEEE fp16 round-trip)
/// - Bf16InputsF32Accum: f64 → f32 → bf16 → f32
#[inline]
fn narrow(x: f64, precision: ReplayPrecision) -> f32 {
    match precision {
        ReplayPrecision::F64 => unreachable!("narrow() not used for F64 path"),
        ReplayPrecision::F32 => x as f32,
        ReplayPrecision::Fp16InputsF32Accum => f16::from_f64(x).to_f32(),
        ReplayPrecision::Bf16InputsF32Accum => half::bf16::from_f64(x).to_f32(),
    }
}

/// Parameterized attention replay with selectable precision.
///
/// Inputs (Q, K, V) always arrive as f64 from dequant+RoPE. The `precision`
/// parameter controls how they are narrowed before the attention inner loop:
///
/// - `F64`: all arithmetic in f64. Current verifier default.
/// - `F32`: inputs truncated to f32, accumulation in f32.
///   Tests whether over-precise accumulation is the main gap source.
///   Note: f64→f32 is NOT the same as GPU fp16→f32 — this only tests
///   whether f32 accumulation alone closes some of the gap.
/// - `Fp16InputsF32Accum`: inputs truncated to IEEE fp16 via `half` crate,
///   then promoted to f32 for accumulation. Closest match to GPU SDPA with
///   fp16 tensors + fp32 internal accumulation.
/// - `Bf16InputsF32Accum`: same but bf16 input truncation, for bf16 models.
pub fn replay_attention_roped_precision(
    q_roped: &[f64],
    kv_cache_k_roped: &[Vec<f64>],
    kv_cache_v_deq: &[Vec<f64>],
    scale_a: f64,
    cfg: &ModelConfig,
    precision: ReplayPrecision,
) -> Vec<i8> {
    if precision == ReplayPrecision::F64 {
        return replay_attention_f64(q_roped, kv_cache_k_roped, kv_cache_v_deq, scale_a, cfg);
    }
    replay_attention_narrowed(q_roped, kv_cache_k_roped, kv_cache_v_deq, scale_a, cfg, precision)
}

/// F64 path (original implementation).
fn replay_attention_f64(
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
        let q_head: Vec<f64> = (0..d_head).map(|i| q_roped[qh * d_head + i]).collect();

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
            a[qh * d_head + i] = (head_out[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    a
}

/// Narrowed-precision path (f32, fp16+f32, bf16+f32).
///
/// Inputs are narrowed via [`narrow`] before the inner loop. All
/// accumulation happens in f32. Final requantization goes through f64.
fn replay_attention_narrowed(
    q_roped: &[f64],
    kv_cache_k_roped: &[Vec<f64>],
    kv_cache_v_deq: &[Vec<f64>],
    scale_a: f64,
    cfg: &ModelConfig,
    precision: ReplayPrecision,
) -> Vec<i8> {
    let d_head = cfg.d_head;
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let inv_sqrt_d: f32 = 1.0 / (d_head as f32).sqrt();
    let seq_len = kv_cache_k_roped.len();
    let inv_scale = if scale_a.abs() > 1e-30 { 1.0 / scale_a } else { 1.0 };

    let mut a = vec![0i8; cfg.hidden_dim];

    for qh in 0..cfg.n_q_heads {
        let kv_head = qh / heads_per_kv;

        let q_head: Vec<f32> = (0..d_head)
            .map(|i| narrow(q_roped[qh * d_head + i], precision))
            .collect();

        let scores: Vec<f32> = (0..seq_len)
            .map(|t| {
                let k_t = &kv_cache_k_roped[t];
                let dot: f32 = (0..d_head)
                    .map(|i| q_head[i] * narrow(k_t[kv_head * d_head + i], precision))
                    .sum();
                dot * inv_sqrt_d
            })
            .collect();

        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        let mut head_out = vec![0.0f32; d_head];
        for t in 0..seq_len {
            let v_t = &kv_cache_v_deq[t];
            for i in 0..d_head {
                head_out[i] += weights[t] * narrow(v_t[kv_head * d_head + i], precision);
            }
        }

        for i in 0..d_head {
            let val = (head_out[i] as f64) * inv_scale;
            a[qh * d_head + i] = val.round().clamp(-128.0, 127.0) as i8;
        }
    }

    a
}

/// Replay attention using witnessed pre-softmax scores from the GPU.
///
/// Instead of computing Q·K^T/√d, takes the GPU's actual scores as input.
/// Applies softmax in f64, then aggregates with committed V in f64.
/// Returns `(a_i8, a_f64)` like [`replay_attention_roped_raw`].
///
/// # Arguments
/// - `witnessed_scores` — flat f32 scores, row-major `[qh * seq_len + t]`
/// - `n_q_heads` — number of query heads (may differ from `cfg.n_q_heads`
///   only for partial-head subsets; normally equal)
/// - `seq_len` — number of KV positions the scores cover
/// - `kv_cache_v_deq` — dequantized V entries per position, each length kv_dim
/// - `scale_a` — quantization scale for output requantization
/// - `cfg` — model config (for `d_head`, `n_kv_heads`, GQA ratio)
pub fn replay_attention_witnessed_scores(
    witnessed_scores: &[f32],
    n_q_heads: usize,
    seq_len: usize,
    kv_cache_v_deq: &[Vec<f64>],
    scale_a: f64,
    cfg: &ModelConfig,
) -> (Vec<i8>, Vec<f64>) {
    let d_head = cfg.d_head;
    let heads_per_kv = n_q_heads / cfg.n_kv_heads;
    let inv_scale = if scale_a.abs() > 1e-30 {
        1.0 / scale_a
    } else {
        1.0
    };

    assert_eq!(
        witnessed_scores.len(),
        n_q_heads * seq_len,
        "witnessed_scores length mismatch: expected {} ({}×{}), got {}",
        n_q_heads * seq_len,
        n_q_heads,
        seq_len,
        witnessed_scores.len()
    );
    assert_eq!(
        kv_cache_v_deq.len(),
        seq_len,
        "kv_cache_v_deq length {} != seq_len {}",
        kv_cache_v_deq.len(),
        seq_len
    );

    let hidden_dim = n_q_heads * d_head;
    let mut a_i8 = vec![0i8; hidden_dim];
    let mut a_f64 = vec![0.0f64; hidden_dim];

    for qh in 0..n_q_heads {
        let kv_head = qh / heads_per_kv;

        // Extract this head's scores and convert to f64.
        let scores_f64: Vec<f64> = (0..seq_len)
            .map(|t| witnessed_scores[qh * seq_len + t] as f64)
            .collect();

        // Softmax in f64 (same stabilized path as replay_attention_roped_raw).
        let max_score = scores_f64
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores_f64.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted V aggregation in f64.
        let mut head_out = vec![0.0f64; d_head];
        for t in 0..seq_len {
            let v_t = &kv_cache_v_deq[t];
            for i in 0..d_head {
                head_out[i] += weights[t] * v_t[kv_head * d_head + i];
            }
        }

        // Requantize to i8.
        for i in 0..d_head {
            let idx = qh * d_head + i;
            a_f64[idx] = head_out[i];
            a_i8[idx] = (head_out[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (a_i8, a_f64)
}

/// LSE-conditioned replay: use externally-captured log-sum-exp values from the
/// GPU kernel instead of computing softmax normalization from scratch.
///
/// Given witnessed scores S and captured LSE (one f32 per query head):
///   P_i = exp(S_i - LSE)    (exact GPU normalization)
///   attn_out = P @ V
///
/// Also computes CPU LSE from the witnessed scores for agreement measurement.
///
/// Returns `(a_i8, a_f64, lse_report)` where `lse_report` contains per-head
/// CPU vs GPU LSE comparison.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LseReport {
    /// Per-head |CPU_LSE - GPU_LSE| for this layer.
    pub per_head_gap: Vec<f32>,
    /// max |CPU_LSE - GPU_LSE| across heads.
    pub max_gap: f32,
    /// mean |CPU_LSE - GPU_LSE| across heads.
    pub mean_gap: f32,
    /// CPU-computed LSE per head.
    pub cpu_lse: Vec<f32>,
}

pub fn replay_attention_with_captured_lse(
    witnessed_scores: &[f32],
    n_q_heads: usize,
    seq_len: usize,
    kv_cache_v_deq: &[Vec<f64>],
    scale_a: f64,
    cfg: &ModelConfig,
    // One f32 per query head: the GPU kernel's log-sum-exp.
    gpu_lse: &[f32],
) -> (Vec<i8>, Vec<f64>, LseReport) {
    let d_head = cfg.d_head;
    let heads_per_kv = n_q_heads / cfg.n_kv_heads;
    let inv_scale = if scale_a.abs() > 1e-30 {
        1.0 / scale_a
    } else {
        1.0
    };

    assert_eq!(
        witnessed_scores.len(),
        n_q_heads * seq_len,
        "witnessed_scores length mismatch"
    );
    assert_eq!(kv_cache_v_deq.len(), seq_len);
    assert!(
        gpu_lse.len() >= n_q_heads,
        "gpu_lse length {} < n_q_heads {}",
        gpu_lse.len(),
        n_q_heads
    );

    let hidden_dim = n_q_heads * d_head;
    let mut a_i8 = vec![0i8; hidden_dim];
    let mut a_f64 = vec![0.0f64; hidden_dim];
    let mut per_head_gap = Vec::with_capacity(n_q_heads);
    let mut cpu_lse_vec = Vec::with_capacity(n_q_heads);

    for qh in 0..n_q_heads {
        let kv_head = qh / heads_per_kv;
        let head_lse = gpu_lse[qh];

        // Compute CPU LSE for agreement measurement.
        let scores_f32: Vec<f32> = (0..seq_len)
            .map(|t| witnessed_scores[qh * seq_len + t])
            .collect();
        let max_s = scores_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = scores_f32.iter().map(|&s| (s - max_s).exp()).collect::<Vec<f32>>().iter().sum();
        let cpu_lse = max_s + sum_exp.ln();
        cpu_lse_vec.push(cpu_lse);
        per_head_gap.push((cpu_lse - head_lse).abs());

        // Use GPU LSE for softmax weights: P_i = exp(S_i - LSE_gpu).
        // Compute weights in f32 (matching GPU precision), then aggregate P@V in f64.
        let weights_f32: Vec<f32> = scores_f32.iter().map(|&s| (s - head_lse).exp()).collect();

        // Weighted V aggregation in f64 (same path as global replay).
        let mut head_out = vec![0.0f64; d_head];
        for t in 0..seq_len {
            let w = weights_f32[t] as f64;
            let v_t = &kv_cache_v_deq[t];
            for i in 0..d_head {
                head_out[i] += w * v_t[kv_head * d_head + i];
            }
        }

        // Requantize to i8.
        for i in 0..d_head {
            let idx = qh * d_head + i;
            a_f64[idx] = head_out[i];
            a_i8[idx] = (head_out[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    let max_gap = per_head_gap.iter().cloned().fold(0.0f32, f32::max);
    let mean_gap = if per_head_gap.is_empty() {
        0.0
    } else {
        per_head_gap.iter().sum::<f32>() / per_head_gap.len() as f32
    };

    let report = LseReport {
        per_head_gap,
        max_gap,
        mean_gap,
        cpu_lse: cpu_lse_vec,
    };

    (a_i8, a_f64, report)
}

/// Tiled online-softmax replay matching FlashAttention-2's decode recurrence.
///
/// Instead of materialising global softmax weights then aggregating P@V,
/// this processes witnessed scores in tiles of `block_n` and maintains
/// running (m, l, O) state in f32 — mirroring FA2's online-softmax with
/// fused O accumulation.
///
/// The kernel uses `exp2f(score * scale_softmax_log2)` via MUFU.EX2.
/// Witnessed scores are already pre-scaled (Q·K^T / √d), so we compute:
///   p = exp2f((s - m_new) * LOG2_E)
/// which equals `exp(s - m_new)` in exact math but follows the GPU's
/// exp2f rounding path.
///
/// # Arguments
/// Same as [`replay_attention_witnessed_scores`], plus:
/// - `block_n` — KV tile size matching the FA2 kernel (128 for split-KV
///   decode on A100 head_dim=128, 64 for standard kernel).
///
/// # Returns
/// `(Vec<i8>, Vec<f64>)` — requantised i8 output and f64 pre-quant output,
/// same layout as the non-tiled version.
pub fn replay_attention_witnessed_scores_tiled(
    witnessed_scores: &[f32],
    n_q_heads: usize,
    seq_len: usize,
    kv_cache_v_deq: &[Vec<f64>],
    scale_a: f64,
    cfg: &ModelConfig,
    block_n: usize,
) -> (Vec<i8>, Vec<f64>) {
    let d_head = cfg.d_head;
    let heads_per_kv = n_q_heads / cfg.n_kv_heads;
    let inv_scale = if scale_a.abs() > 1e-30 {
        1.0 / scale_a
    } else {
        1.0
    };

    assert_eq!(
        witnessed_scores.len(),
        n_q_heads * seq_len,
        "witnessed_scores length mismatch: expected {} ({}×{}), got {}",
        n_q_heads * seq_len,
        n_q_heads,
        seq_len,
        witnessed_scores.len()
    );
    assert_eq!(
        kv_cache_v_deq.len(),
        seq_len,
        "kv_cache_v_deq length {} != seq_len {}",
        kv_cache_v_deq.len(),
        seq_len
    );
    assert!(block_n > 0, "block_n must be > 0");

    // LOG2_E = log2(e) ≈ 1.4426950408889634
    const LOG2_E: f32 = std::f32::consts::LOG2_E;

    let hidden_dim = n_q_heads * d_head;
    let mut a_i8 = vec![0i8; hidden_dim];
    let mut a_f64 = vec![0.0f64; hidden_dim];

    for qh in 0..n_q_heads {
        let kv_head = qh / heads_per_kv;

        // Running state: m (max), l (sum of exp), O (output accumulator).
        // All in f32 to match GPU kernel precision.
        let mut m: f32 = f32::NEG_INFINITY;
        let mut l: f32 = 0.0_f32;
        let mut o_acc = vec![0.0_f32; d_head];

        // Process tiles of block_n.
        let n_tiles = (seq_len + block_n - 1) / block_n;
        for tile in 0..n_tiles {
            let t_start = tile * block_n;
            let t_end = (t_start + block_n).min(seq_len);

            // Tile max over witnessed scores (already scaled by 1/√d on GPU).
            let mut m_tile: f32 = f32::NEG_INFINITY;
            for t in t_start..t_end {
                let s = witnessed_scores[qh * seq_len + t];
                if s > m_tile {
                    m_tile = s;
                }
            }

            let m_new = m.max(m_tile);

            // Rescale previous accumulator: alpha = exp2f((m_old - m_new) * LOG2_E)
            let alpha: f32 = ((m - m_new) * LOG2_E).exp2();
            l *= alpha;
            for i in 0..d_head {
                o_acc[i] *= alpha;
            }

            // Accumulate this tile: fused softmax weights + P@V.
            for t in t_start..t_end {
                let s = witnessed_scores[qh * seq_len + t];
                let p: f32 = ((s - m_new) * LOG2_E).exp2();
                l += p;

                // Fused P@V: O += p * V[t]
                let v_t = &kv_cache_v_deq[t];
                for i in 0..d_head {
                    o_acc[i] += p * v_t[kv_head * d_head + i] as f32;
                }
            }

            m = m_new;
        }

        // Final normalisation: O = O * (1/l)  (reciprocal multiply, matching GPU).
        let inv_l: f32 = 1.0 / l;
        let mut head_out = vec![0.0_f64; d_head];
        for i in 0..d_head {
            head_out[i] = (o_acc[i] * inv_l) as f64;
        }

        // Requantize to i8.
        for i in 0..d_head {
            let idx = qh * d_head + i;
            a_f64[idx] = head_out[i];
            a_i8[idx] = (head_out[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (a_i8, a_f64)
}

/// Compute canonical Q·K^T/√d scores in f64 from shell Q and committed K.
///
/// Returns flat f64 scores in row-major order: `scores[qh * seq_len + t]`.
/// These are the verifier's independent reconstruction of pre-softmax
/// attention scores — used to anchor witnessed scores.
pub fn compute_canonical_scores(
    q_roped: &[f64],
    kv_cache_k_roped: &[Vec<f64>],
    cfg: &ModelConfig,
) -> Vec<f64> {
    let d_head = cfg.d_head;
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let inv_sqrt_d = 1.0 / (d_head as f64).sqrt();
    let seq_len = kv_cache_k_roped.len();

    let mut scores = vec![0.0f64; cfg.n_q_heads * seq_len];

    for qh in 0..cfg.n_q_heads {
        let kv_head = qh / heads_per_kv;
        for t in 0..seq_len {
            let k_t = &kv_cache_k_roped[t];
            let dot: f64 = (0..d_head)
                .map(|i| q_roped[qh * d_head + i] * k_t[kv_head * d_head + i])
                .sum();
            scores[qh * seq_len + t] = dot * inv_sqrt_d;
        }
    }

    scores
}

/// Compute canonical scores in GPU-like precision (f32 with fp16 truncation).
///
/// Mimics the GPU serving path:
/// 1. Q/K dequantized from i32 accumulators with per-channel scales (f32)
/// 2. Bias added in f32
/// 3. Truncated to fp16 (matching cutlass_scaled_mm output precision)
/// 4. Cast back to f32 for RoPE and dot product
///
/// This produces canonical scores that closely match what the GPU computed,
/// unlike the f64 path which diverges due to precision differences.
pub fn compute_canonical_scores_gpu_like(
    q_roped_f32: &[f32],
    kv_cache_k_roped_f32: &[Vec<f32>],
    cfg: &ModelConfig,
) -> Vec<f32> {
    let d_head = cfg.d_head;
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let inv_sqrt_d = 1.0f32 / (d_head as f32).sqrt();
    let seq_len = kv_cache_k_roped_f32.len();

    let mut scores = vec![0.0f32; cfg.n_q_heads * seq_len];

    for qh in 0..cfg.n_q_heads {
        let kv_head = qh / heads_per_kv;
        for t in 0..seq_len {
            let k_t = &kv_cache_k_roped_f32[t];
            let dot: f32 = (0..d_head)
                .map(|i| q_roped_f32[qh * d_head + i] * k_t[kv_head * d_head + i])
                .sum();
            scores[qh * seq_len + t] = dot * inv_sqrt_d;
        }
    }

    scores
}

/// Dequantize i32 accumulator to f32 with per-channel scales, add bias,
/// then truncate to fp16 precision (mimicking cutlass_scaled_mm output).
///
/// Returns f32 values that match the GPU's fp16 output precision.
pub fn dequant_bias_fp16(
    acc: &[i32],
    per_channel_scales: &[f32],
    scale_x: f32,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out: Vec<f32> = acc
        .iter()
        .zip(per_channel_scales.iter())
        .map(|(&a, &sw)| (a as f32) * sw * scale_x)
        .collect();

    if let Some(b) = bias {
        for (x, &bv) in out.iter_mut().zip(b.iter()) {
            *x += bv;
        }
    }

    // Truncate to fp16 precision: this is the key step that matches
    // the GPU's cutlass_scaled_mm output, which is fp16.
    for x in out.iter_mut() {
        *x = half::f16::from_f32(*x).to_f32();
    }

    out
}

/// Dequantize i32 accumulators → f32, add bias, truncate through **bf16**.
///
/// Same as `dequant_bias_fp16` but uses bf16 narrowing to match models that
/// run in bfloat16 (e.g. Qwen on vLLM with `dtype=torch.bfloat16`).
/// The GPU's cutlass output is bf16, so narrowing through bf16 matches the
/// actual precision the GPU produces.
pub fn dequant_bias_bf16(
    acc: &[i32],
    per_channel_scales: &[f32],
    scale_x: f32,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out: Vec<f32> = acc
        .iter()
        .zip(per_channel_scales.iter())
        .map(|(&a, &sw)| (a as f32) * sw * scale_x)
        .collect();

    if let Some(b) = bias {
        for (x, &bv) in out.iter_mut().zip(b.iter()) {
            *x += bv;
        }
    }

    // Truncate to bf16 precision: matches the GPU's bfloat16 output dtype.
    for x in out.iter_mut() {
        *x = half::bf16::from_f32(*x).to_f32();
    }

    out
}

/// Deterministic epilogue matching vLLM's CUTLASS `_scaled_mm` exactly.
///
/// Reproduces the exact computation order from CUTLASS's `ScaledEpilogue` /
/// `ScaledEpilogueBias` EVT (Epilogue Visitor Tree):
///
/// Without bias:
///   `bf16_rne(scale_a * (f32(acc) * scale_b))`
///
/// With bias (CUTLASS uses `homogeneous_multiply_add` → compiler emits FMA):
///   `bf16_rne(fma(scale_a, f32(acc) * scale_b, f32(bias)))`
///
/// Where:
/// - `scale_b` = per-channel weight scale (`RowOrScalarLoad`, shape `[1,N]`)
/// - `scale_a` = per-token activation scale (scalar for the given token)
/// - `bias` = projection bias stored as bf16 in model, loaded as f32 (bf16-precision)
/// - `fma` = fused multiply-add with single rounding (matches GPU `__fma_rn`)
/// - `bf16_rne` = bf16 cast with round-to-nearest-even
///
/// The Rust `f32::mul_add(a, b, c)` maps to hardware FMA on x86 (FMA3),
/// matching the GPU's FMA instruction.
pub fn cutlass_epilogue_bf16(
    acc: &[i32],
    per_channel_scales: &[f32],
    scale_a: f32,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };

    match bias {
        Some(b) => acc
            .iter()
            .zip(per_channel_scales.iter())
            .zip(b.iter())
            .map(|((&a, &sw), &bv)| {
                let temp = (a as f32) * sw; // f32(acc) * scale_b
                bf16(scale_a.mul_add(temp, bv)) // fma(scale_a, temp, bias) → bf16
            })
            .collect(),
        None => acc
            .iter()
            .zip(per_channel_scales.iter())
            .map(|(&a, &sw)| {
                let temp = (a as f32) * sw; // f32(acc) * scale_b
                bf16(temp * scale_a) // scale_a * temp → bf16
            })
            .collect(),
    }
}

/// Dequant+bias with multiple bf16 cast-order strategies for diagnostics.
///
/// Returns a map of strategy name → f32 output vector. Each strategy represents
/// a plausible interpretation of what CUTLASS `scaled_mm` does internally:
///
/// - `"bf16_final"`: `bf16(f32_dequant + f32_bias)` — single bf16 cast at the end
/// - `"bf16_before_bias"`: `bf16(f32_dequant) + bf16(bias)` — cast dequant to bf16, add bias in bf16
/// - `"bf16_dequant_f32_bias"`: `bf16(bf16(f32_dequant) + f32_bias)` — cast dequant to bf16, add bias in f32, recast
/// - `"f32_all"`: `f32_dequant + f32_bias` — no bf16 narrowing (full precision baseline)
/// - `"bf16_per_mult"`: `bf16(bf16(a * sw) * sx) + bf16(bias)` — bf16 after each multiply
pub fn dequant_bias_cast_variants(
    acc: &[i32],
    per_channel_scales: &[f32],
    scale_x: f32,
    bias: Option<&[f32]>,
) -> Vec<(&'static str, Vec<f32>)> {
    let bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };

    // Strategy A: bf16(f32_dequant + f32_bias) — current dequant_bias_bf16
    let a = {
        let mut out: Vec<f32> = acc.iter().zip(per_channel_scales.iter())
            .map(|(&a, &sw)| (a as f32) * sw * scale_x)
            .collect();
        if let Some(b) = bias {
            for (x, &bv) in out.iter_mut().zip(b.iter()) { *x += bv; }
        }
        for x in out.iter_mut() { *x = bf16(*x); }
        out
    };

    // Strategy B: bf16(f32_dequant) + bf16(bias) — truncate dequant, truncate bias, add in bf16
    let b = {
        let mut out: Vec<f32> = acc.iter().zip(per_channel_scales.iter())
            .map(|(&a, &sw)| bf16((a as f32) * sw * scale_x))
            .collect();
        if let Some(bi) = bias {
            for (x, &bv) in out.iter_mut().zip(bi.iter()) {
                *x = bf16(*x + bf16(bv));
            }
        }
        out
    };

    // Strategy C: bf16(bf16(f32_dequant) + f32_bias) — truncate dequant, add bias in f32, truncate again
    let c = {
        let mut out: Vec<f32> = acc.iter().zip(per_channel_scales.iter())
            .map(|(&a, &sw)| bf16((a as f32) * sw * scale_x))
            .collect();
        if let Some(bi) = bias {
            for (x, &bv) in out.iter_mut().zip(bi.iter()) {
                *x = bf16(*x + bv);
            }
        }
        out
    };

    // Strategy D: f32 all — no bf16, full precision baseline
    let d = {
        let mut out: Vec<f32> = acc.iter().zip(per_channel_scales.iter())
            .map(|(&a, &sw)| (a as f32) * sw * scale_x)
            .collect();
        if let Some(bi) = bias {
            for (x, &bv) in out.iter_mut().zip(bi.iter()) { *x += bv; }
        }
        out
    };

    // Strategy E: bf16 after each multiply — bf16(bf16(a * sw) * sx) + bf16(bias)
    let e = {
        let mut out: Vec<f32> = acc.iter().zip(per_channel_scales.iter())
            .map(|(&a, &sw)| bf16(bf16((a as f32) * sw) * scale_x))
            .collect();
        if let Some(bi) = bias {
            for (x, &bv) in out.iter_mut().zip(bi.iter()) {
                *x = bf16(*x + bf16(bv));
            }
        }
        out
    };

    vec![
        ("bf16_final", a),
        ("bf16_before_bias", b),
        ("bf16_dequant_f32_bias", c),
        ("f32_all", d),
        ("bf16_per_mult", e),
    ]
}

/// Apply RoPE to a vector in f32 precision (matching GPU score witness path).
///
/// `vec_f32` has length `n_heads * d_head`. Returns RoPE'd vector in f32.
pub fn apply_rope_f32(
    vec_f32: &[f32],
    n_heads: usize,
    position: usize,
    cfg: &ModelConfig,
) -> Vec<f32> {
    let d_head = cfg.d_head;
    let half = d_head / 2;
    assert_eq!(vec_f32.len(), n_heads * d_head);

    // Compute inv_freq in f32 (matching GPU capture.py's compute_witnessed_scores).
    let inv_freq: Vec<f32> = cfg
        .scaled_inv_freq()
        .iter()
        .map(|&f| f as f32)
        .collect();

    let mut out = vec![0.0f32; vec_f32.len()];
    for h in 0..n_heads {
        let head = &vec_f32[h * d_head..(h + 1) * d_head];
        let base = h * d_head;
        for i in 0..half {
            let angle = (position as f32) * inv_freq[i];
            let cos_f = angle.cos();
            let sin_f = angle.sin();
            out[base + i] = head[i] * cos_f - head[i + half] * sin_f;
            out[base + i + half] = head[i + half] * cos_f + head[i] * sin_f;
        }
    }

    out
}

/// Apply RoPE in f32, then truncate each output element to bf16.
///
/// Matches the GPU's KV cache storage: after RoPE, K is stored as bf16 in the
/// cache. The RoPE arithmetic (multiply/add) happens in f32 (promoted from bf16
/// inputs), but the result is written back as bf16.
pub fn apply_rope_bf16(
    vec_f32: &[f32],
    n_heads: usize,
    position: usize,
    cfg: &ModelConfig,
) -> Vec<f32> {
    let d_head = cfg.d_head;
    let half = d_head / 2;
    assert_eq!(vec_f32.len(), n_heads * d_head);

    let inv_freq: Vec<f32> = cfg
        .scaled_inv_freq()
        .iter()
        .map(|&f| f as f32)
        .collect();

    let bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };

    let mut out = vec![0.0f32; vec_f32.len()];
    for h in 0..n_heads {
        let head = &vec_f32[h * d_head..(h + 1) * d_head];
        let base = h * d_head;
        for i in 0..half {
            let angle = (position as f32) * inv_freq[i];
            let cos_f = angle.cos();
            let sin_f = angle.sin();
            // RoPE in f32, then truncate to bf16 (matching KV cache storage).
            out[base + i] = bf16(head[i] * cos_f - head[i + half] * sin_f);
            out[base + i + half] = bf16(head[i + half] * cos_f + head[i] * sin_f);
        }
    }

    out
}

/// Per-layer score anchoring result.
///
/// Compares witnessed pre-softmax scores against canonical verifier-reconstructed
/// scores (from shell Q and committed K). A max gap exceeding the threshold
/// indicates the witnessed scores may have been tampered with.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScoreAnchorStats {
    pub layer: usize,
    /// Max |witnessed - canonical| across all (head, position) entries.
    pub max_gap: f64,
    /// Mean |witnessed - canonical|.
    pub mean_gap: f64,
    /// Number of score elements compared.
    pub n_elements: usize,
    /// Per-head max gap.
    pub per_head_max_gap: Vec<f64>,
}

/// Compare witnessed scores against canonical scores for one layer.
///
/// Returns anchoring statistics. The caller decides the threshold.
pub fn anchor_witnessed_scores(
    witnessed: &[f32],
    canonical: &[f64],
    n_q_heads: usize,
    seq_len: usize,
    layer: usize,
) -> ScoreAnchorStats {
    assert_eq!(witnessed.len(), n_q_heads * seq_len);
    assert_eq!(canonical.len(), n_q_heads * seq_len);

    let mut max_gap = 0.0f64;
    let mut sum_gap = 0.0f64;
    let mut per_head_max = vec![0.0f64; n_q_heads];

    for qh in 0..n_q_heads {
        for t in 0..seq_len {
            let idx = qh * seq_len + t;
            let gap = (witnessed[idx] as f64 - canonical[idx]).abs();
            if gap > max_gap {
                max_gap = gap;
            }
            sum_gap += gap;
            if gap > per_head_max[qh] {
                per_head_max[qh] = gap;
            }
        }
    }

    let n = n_q_heads * seq_len;
    ScoreAnchorStats {
        layer,
        max_gap,
        mean_gap: if n > 0 { sum_gap / n as f64 } else { 0.0 },
        n_elements: n,
        per_head_max_gap: per_head_max,
    }
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
    let frac_eq = if n > 0 {
        histogram[0] as f64 / n_f
    } else {
        0.0
    };
    let frac_le_1 = if n > 0 {
        (histogram[0] + histogram[1]) as f64 / n_f
    } else {
        0.0
    };
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
pub fn compare_attention_output(
    claimed: &[i8],
    replayed: &[i8],
    tolerance: &AttentionToleranceConfig,
) -> Option<i16> {
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

/// Per-head attention evidence curve: tail bound as a function of k.
///
/// For each head, we precompute the sorted softmax weights and max|V_tail|
/// so the certifier can adaptively pick the smallest k that closes the bound.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HeadEvidenceCurve {
    /// Sorted softmax weights (descending). weights[0] is the largest.
    pub sorted_weights: Vec<f32>,
    /// Prefix mass at each k: prefix_mass[k] = sum of top-(k+1) weights.
    pub prefix_mass: Vec<f32>,
    /// Tail bound at each k: tail_bound[k] = (1 - prefix_mass[k]) * max|V_tail(k)|.
    /// tail_bound[k] bounds the worst-case perturbation to any attention output
    /// dimension from ignoring positions outside the top-(k+1).
    pub tail_bounds: Vec<f64>,
    /// max|V| across ALL positions (used when k=0, the full tail).
    pub max_v_abs: f64,
}

/// Per-layer attention evidence for stock-compatible certification.
///
/// Contains per-head evidence curves, allowing the certifier to adaptively
/// pick k per head to minimize total certified bound.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AttentionEvidence {
    pub layer: usize,
    /// Per-head evidence curves (one per query head).
    pub head_curves: Vec<HeadEvidenceCurve>,
    /// Legacy fields for backward compatibility with reports.
    /// Minimum top-k mass across all heads at the default k.
    pub min_top_k_mass: f32,
    /// Maximum tail bound across all heads at the default k.
    pub max_tail_bound: f64,
    /// Top-k value used for legacy fields.
    pub top_k: usize,
}

/// Compute attention evidence curves for one layer.
///
/// For each query head, computes the full sorted-weight curve and tail bounds
/// at every possible k, enabling adaptive k selection by the certifier.
///
/// Given witnessed scores and committed V, computes:
/// 1. Softmax weights from witnessed scores (f32)
/// 2. Sorted weights (descending) per head
/// 3. Prefix mass curve: prefix_mass[k] = sum of top-(k+1) weights
/// 4. Tail bound curve: tail_bound[k] = tail_mass(k) × max|V_tail(k)|
///
/// `default_k` is used only to populate the legacy summary fields.
pub fn compute_attention_evidence(
    witnessed_scores: &[f32],
    n_q_heads: usize,
    seq_len: usize,
    kv_cache_v_deq: &[Vec<f64>],
    cfg: &ModelConfig,
    default_k: usize,
) -> AttentionEvidence {
    let d_head = cfg.d_head;
    let heads_per_kv = n_q_heads / cfg.n_kv_heads;

    let mut head_curves = Vec::with_capacity(n_q_heads);

    for qh in 0..n_q_heads {
        let kv_head = qh / heads_per_kv;

        // 1. Extract scores for this head.
        let head_scores: Vec<f32> = (0..seq_len)
            .map(|t| witnessed_scores[qh * seq_len + t])
            .collect();

        // 2. Compute softmax weights in f32 (numerically stable).
        let max_score = head_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = head_scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // 3. Sort by weight descending, keeping original indices.
        let mut indexed_weights: Vec<(usize, f32)> =
            weights.iter().cloned().enumerate().collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_weights: Vec<f32> = indexed_weights.iter().map(|&(_, w)| w).collect();

        // 4. Precompute per-position max|V| for the sorted order.
        //    v_abs_sorted[i] = max over d_head dims of |V[pos_i][kv_head*d_head+d]|
        let v_abs_sorted: Vec<f64> = indexed_weights
            .iter()
            .map(|&(pos, _)| {
                let v_t = &kv_cache_v_deq[pos];
                (0..d_head)
                    .map(|d| v_t[kv_head * d_head + d].abs())
                    .fold(0.0f64, f64::max)
            })
            .collect();

        let max_v_abs = v_abs_sorted
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        // 5. Build prefix mass curve and tail bound curve.
        let mut prefix_mass = Vec::with_capacity(seq_len);
        let mut tail_bounds = Vec::with_capacity(seq_len);
        let mut running_mass: f32 = 0.0;
        // max|V| in the tail (positions k+1..seq_len), starts as global max
        // We track this by maintaining a running suffix max of v_abs_sorted.
        // Precompute suffix max: suffix_max_v[k] = max(v_abs_sorted[k..])
        let mut suffix_max_v = vec![0.0f64; seq_len + 1];
        for i in (0..seq_len).rev() {
            suffix_max_v[i] = f64::max(v_abs_sorted[i], suffix_max_v[i + 1]);
        }

        for k in 0..seq_len {
            running_mass += sorted_weights[k];
            prefix_mass.push(running_mass);
            let tail_mass = 1.0f32 - running_mass;
            // max|V| in tail = suffix_max_v[k+1] (positions after the top-(k+1))
            let max_v_tail = suffix_max_v[k + 1];
            tail_bounds.push((tail_mass as f64) * max_v_tail);
        }

        head_curves.push(HeadEvidenceCurve {
            sorted_weights,
            prefix_mass,
            tail_bounds,
            max_v_abs,
        });
    }

    // Populate legacy summary fields at default_k.
    let effective_k = default_k.min(seq_len);
    let min_top_k_mass = head_curves
        .iter()
        .map(|c| if effective_k > 0 { c.prefix_mass[effective_k - 1] } else { 0.0 })
        .fold(f32::INFINITY, f32::min);
    let max_tail_bound = head_curves
        .iter()
        .map(|c| if effective_k > 0 { c.tail_bounds[effective_k - 1] } else { c.max_v_abs })
        .fold(0.0f64, f64::max);

    AttentionEvidence {
        layer: 0, // caller should set this
        head_curves,
        min_top_k_mass,
        max_tail_bound,
        top_k: effective_k,
    }
}

/// Certification outcome — why the certifier accepted or rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificationOutcome {
    /// Bound closed: estimated_logit_bound < logit_margin / 2.
    Certified,
    /// Budget spent but bound did not close.
    BudgetExhausted,
    /// Logit margin too small (≤ 0): top token not distinct.
    MarginTooSmall,
}

/// Certification budget — hard caps on how much work the certifier can do.
#[derive(Debug, Clone, Copy)]
pub struct CertificationBudget {
    /// Maximum k for any single head.
    pub max_k_per_head: usize,
    /// Maximum total k summed across all (layer, head) pairs.
    pub max_total_k: usize,
}

impl Default for CertificationBudget {
    fn default() -> Self {
        Self {
            max_k_per_head: 128,
            max_total_k: 8192,
        }
    }
}

/// Token-level certification result combining attention evidence with logit margin.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenCertification {
    /// Certification outcome.
    pub outcome: CertificationOutcome,
    /// Logit margin: gap between top-1 and top-2 final logits.
    pub logit_margin: f32,
    /// Estimated worst-case final-logit perturbation bound from attention uncertainty.
    /// Certification requires: estimated_logit_bound < logit_margin / 2.
    pub estimated_logit_bound: f64,
    /// Per-layer, per-head chosen k values.
    pub chosen_k_per_layer: Vec<Vec<usize>>,
    /// Total k budget spent.
    pub total_k_spent: usize,
    /// Human-readable certification reason.
    pub reason: String,
}

impl TokenCertification {
    pub fn certified(&self) -> bool {
        self.outcome == CertificationOutcome::Certified
    }
}

/// K steps in the schedule. Each step represents a k value to try.
/// The cost of promoting a head from k_prev to k_next is (k_next - k_prev).
const ADAPTIVE_K_SCHEDULE: &[usize] = &[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

/// Fallback downstream gain factor used when alpha[l,h] is not available.
const FALLBACK_DOWNSTREAM_GAIN: f64 = 10.0;

/// Certify a token using budgeted greedy allocation with optional alpha.
///
/// Instead of picking k independently per head, this allocator:
/// 1. Starts every head at k=0 (full tail bound).
/// 2. Builds a priority queue of (layer, head, next_k_step) entries.
/// 3. Greedily promotes the head with the best marginal_bound_reduction / marginal_cost.
/// 4. Stops when the total bound closes (< margin/2) or budget is exhausted.
///
/// The bound at any point is:
///   estimated_logit_bound = sum over (layer, head) of alpha[l,h] * eps_attn[l,h,current_k]
///
/// `o_proj_alpha`: per-layer, per-head sensitivity from o_proj weights.
/// `budget`: hard caps on per-head k and total k across all heads.
pub fn certify_token_budgeted(
    evidence: &[AttentionEvidence],
    logit_margin: f32,
    o_proj_alpha: Option<&[Vec<f64>]>,
    budget: &CertificationBudget,
) -> TokenCertification {
    if logit_margin <= 0.0 {
        return TokenCertification {
            outcome: CertificationOutcome::MarginTooSmall,
            logit_margin,
            estimated_logit_bound: f64::INFINITY,
            chosen_k_per_layer: vec![],
            total_k_spent: 0,
            reason: format!(
                "margin_too_small: logit margin {:.4} <= 0",
                logit_margin
            ),
        };
    }

    let target = (logit_margin as f64) / 2.0;
    let has_alpha = o_proj_alpha.is_some();

    // --- Phase 1: Build per-head metadata ---
    // For each (layer, head), precompute:
    //   - alpha coefficient
    //   - seq_len (number of positions in curve)
    //   - effective max k for this head
    //   - current resid bound (at k=0 / schedule step 0)
    //   - schedule of (k, resid_bound_at_k) pairs

    struct HeadState {
        layer_ev_idx: usize,
        head_idx: usize,
        alpha: f64,
        /// Index into ADAPTIVE_K_SCHEDULE of current step (0 = not yet promoted).
        schedule_idx: usize,
        /// Residual-space bound at current k.
        current_resid_bound: f64,
        /// Precomputed (k, attn_bound) for each schedule step that fits within max_k.
        schedule: Vec<(usize, f64)>,
    }

    let mut heads: Vec<HeadState> = Vec::new();
    let mut total_resid_bound: f64 = 0.0;

    // Initialize chosen k per layer (all zeros).
    let mut chosen_k_per_layer: Vec<Vec<usize>> = evidence
        .iter()
        .map(|ev| vec![0; ev.head_curves.len()])
        .collect();

    for (ev_idx, layer_ev) in evidence.iter().enumerate() {
        let layer_alpha = o_proj_alpha.and_then(|a| a.get(layer_ev.layer));

        for (head_idx, curve) in layer_ev.head_curves.iter().enumerate() {
            let seq_len = curve.tail_bounds.len();
            if seq_len == 0 {
                continue;
            }

            let alpha = layer_alpha
                .and_then(|la| la.get(head_idx))
                .copied()
                .unwrap_or(FALLBACK_DOWNSTREAM_GAIN);

            let effective_max_k = budget.max_k_per_head.min(seq_len.saturating_sub(1));

            // Build schedule: (k, attn_bound) for each applicable step.
            let mut schedule: Vec<(usize, f64)> = Vec::new();
            for &k in ADAPTIVE_K_SCHEDULE {
                if k > effective_max_k {
                    break;
                }
                let idx = k - 1;
                if idx < seq_len {
                    schedule.push((k, curve.tail_bounds[idx]));
                }
            }

            // Initial bound: k=0, full tail.
            let initial_attn_bound = curve.max_v_abs;
            let initial_resid_bound = alpha * initial_attn_bound;
            total_resid_bound += initial_resid_bound;

            heads.push(HeadState {
                layer_ev_idx: ev_idx,
                head_idx,
                alpha,
                schedule_idx: 0, // not yet promoted
                current_resid_bound: initial_resid_bound,
                schedule,
            });
        }
    }

    // --- Phase 2: Greedy budget allocation ---
    // Repeatedly find the head+step with best marginal_reduction / marginal_cost,
    // promote it, and update the total bound.

    let mut total_k_spent: usize = 0;

    loop {
        // Check if bound already closes.
        if total_resid_bound < target {
            break;
        }

        // Find best promotion: highest (marginal_reduction / marginal_cost).
        let mut best_head_idx: Option<usize> = None;
        let mut best_ratio: f64 = 0.0;
        let mut best_cost: usize = 0;

        for (hi, head) in heads.iter().enumerate() {
            if head.schedule_idx >= head.schedule.len() {
                continue; // fully promoted
            }

            let (next_k, next_attn_bound) = head.schedule[head.schedule_idx];
            let new_resid_bound = head.alpha * next_attn_bound;
            let reduction = head.current_resid_bound - new_resid_bound;
            if reduction <= 0.0 {
                continue; // no improvement
            }

            // Cost = delta k (new positions to check).
            let prev_k = if head.schedule_idx == 0 {
                0
            } else {
                head.schedule[head.schedule_idx - 1].0
            };
            let cost = next_k - prev_k;
            if cost == 0 {
                continue;
            }

            // Check budget.
            if total_k_spent + cost > budget.max_total_k {
                continue; // would exceed total budget
            }

            let ratio = reduction / (cost as f64);
            if ratio > best_ratio {
                best_ratio = ratio;
                best_cost = cost;
                best_head_idx = Some(hi);
            }
        }

        match best_head_idx {
            Some(hi) => {
                let head = &mut heads[hi];
                let (next_k, next_attn_bound) = head.schedule[head.schedule_idx];
                let new_resid_bound = head.alpha * next_attn_bound;

                total_resid_bound -= head.current_resid_bound;
                total_resid_bound += new_resid_bound;
                head.current_resid_bound = new_resid_bound;
                head.schedule_idx += 1;
                total_k_spent += best_cost;

                chosen_k_per_layer[head.layer_ev_idx][head.head_idx] = next_k;
            }
            None => break, // no promotions available
        }
    }

    // --- Phase 3: Produce result ---
    let estimated_logit_bound = total_resid_bound;
    let outcome = if estimated_logit_bound < target {
        CertificationOutcome::Certified
    } else {
        CertificationOutcome::BudgetExhausted
    };

    let gain_info = if has_alpha { "alpha" } else { "fallback" };
    let reason = match outcome {
        CertificationOutcome::Certified => format!(
            "certified: bound {:.4} < margin/2 {:.4} (margin {:.4}, k_spent={}, gain={})",
            estimated_logit_bound, target, logit_margin, total_k_spent, gain_info
        ),
        CertificationOutcome::BudgetExhausted => format!(
            "budget_exhausted: bound {:.4} >= margin/2 {:.4} (margin {:.4}, k_spent={}, gain={})",
            estimated_logit_bound, target, logit_margin, total_k_spent, gain_info
        ),
        CertificationOutcome::MarginTooSmall => unreachable!(),
    };

    TokenCertification {
        outcome,
        logit_margin,
        estimated_logit_bound,
        chosen_k_per_layer,
        total_k_spent,
        reason,
    }
}

/// Convenience wrapper: budgeted greedy with default budget and no alpha.
pub fn certify_token_adaptive(
    evidence: &[AttentionEvidence],
    logit_margin: f32,
    max_k: usize,
) -> TokenCertification {
    let budget = CertificationBudget {
        max_k_per_head: max_k,
        max_total_k: max_k * 64, // reasonable default
    };
    certify_token_budgeted(evidence, logit_margin, None, &budget)
}

/// Convenience wrapper: budgeted greedy with alpha and explicit budget.
pub fn certify_token_adaptive_with_alpha(
    evidence: &[AttentionEvidence],
    logit_margin: f32,
    max_k: usize,
    o_proj_alpha: Option<&[Vec<f64>]>,
) -> TokenCertification {
    let budget = CertificationBudget {
        max_k_per_head: max_k,
        max_total_k: max_k * 64,
    };
    certify_token_budgeted(evidence, logit_margin, o_proj_alpha, &budget)
}

/// Legacy fixed-threshold certifier. Delegates to budgeted greedy.
pub fn certify_token(
    evidence: &[AttentionEvidence],
    logit_margin: f32,
    _concentration_threshold: f32,
) -> TokenCertification {
    certify_token_adaptive(evidence, logit_margin, 128)
}

// ---------------------------------------------------------------------------
// Frozen parallel reduction tree — deterministic attention v2 primitive
// ---------------------------------------------------------------------------
//
// CONTRACT (protocol-level — changing this changes the protocol):
//
// Both `tree_reduce_sum_f32` and `tree_reduce_max_f32` implement a binary
// reduction tree over an f32 slice. The tree shape is fully defined by the
// input length and must be identical on CPU (Rust) and GPU (CUDA).
//
// 1. PADDING: The logical length is rounded up to the next power of two.
//    Padding elements use the identity for the operation:
//      - sum:  0.0 (positive zero, 0x00000000)
//      - max:  -inf (0xFF800000)
//    Padding is appended to the END of the input.
//
// 2. TREE STRUCTURE: At each level, adjacent pairs are combined:
//      buf[i] = op(buf[i], buf[i + stride])
//    where stride = padded_len / 2, padded_len / 4, ..., 1.
//    Left operand always has the lower index.
//
// 3. PAIR ORDER: Within each pair, the operation is always:
//      result = left OP right
//    For sum: result = left + right   (f32 add, round-to-nearest-even)
//    For max: result = if left >= right { left } else { right }
//            (IEEE 754 comparison; on equal values, left operand wins.
//             For signed zeros: -0.0 >= +0.0 is true, so left wins on ties.)
//
// 4. NaN POLICY: Inputs must not contain NaN. Behavior on NaN input is
//    undefined. Callers are responsible for NaN-free inputs.
//
// 5. SIGNED ZERO: For sum, 0.0 + (-0.0) = 0.0 in IEEE 754 (positive zero).
//    For max, +0.0 >= -0.0 is true, so +0.0 is preferred over -0.0.
//    Both sides (CPU/GPU) use IEEE 754 default rounding mode.
//
// 6. EMPTY INPUT: Length 0 returns the identity element (0.0 for sum, -inf
//    for max).
//
// 7. LENGTH 1: Returns the single element unchanged.
//
// 8. WARP/BLOCK STAGING: The GPU kernel may use warp shuffles for the last
//    5 levels (stride <= 16) as an optimization, but the LOGICAL tree
//    structure is the same as the CPU reference. Warp shuffle respects
//    lane ordering which matches the array index ordering.

/// Fixed binary tree reduction: f32 sum.
///
/// Deterministic across CPU and GPU when both follow the frozen tree contract.
/// See the contract comment above for the full specification.
pub fn tree_reduce_sum_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0_f32;
    }
    if values.len() == 1 {
        return values[0];
    }
    let n = values.len().next_power_of_two();
    let mut buf = Vec::with_capacity(n);
    buf.extend_from_slice(values);
    buf.resize(n, 0.0_f32); // identity for sum
    let mut stride = n / 2;
    while stride >= 1 {
        for i in 0..stride {
            buf[i] = buf[i] + buf[i + stride];
        }
        stride /= 2;
    }
    buf[0]
}

/// Fixed binary tree reduction: f32 max.
///
/// Deterministic across CPU and GPU when both follow the frozen tree contract.
/// See the contract comment above for the full specification.
pub fn tree_reduce_max_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NEG_INFINITY;
    }
    if values.len() == 1 {
        return values[0];
    }
    let n = values.len().next_power_of_two();
    let mut buf = Vec::with_capacity(n);
    buf.extend_from_slice(values);
    buf.resize(n, f32::NEG_INFINITY); // identity for max
    let mut stride = n / 2;
    while stride >= 1 {
        for i in 0..stride {
            buf[i] = if buf[i] >= buf[i + stride] {
                buf[i]
            } else {
                buf[i + stride]
            };
        }
        stride /= 2;
    }
    buf[0]
}

// ---------------------------------------------------------------------------
// Deterministic attention: bit-exact CPU reference for verified-attention mode
// ---------------------------------------------------------------------------

/// Bit-level ldexp: p * 2^n without any library call.
///
/// Adds n to the IEEE 754 biased exponent. Assumes p is a normal positive
/// float and |n| < 127. Identical logic on CPU (Rust) and GPU (CUDA).
#[inline]
fn ldexp_bitwise(p: f32, n: i32) -> f32 {
    let bits = p.to_bits();
    let biased_exp = ((bits >> 23) & 0xFF) as i32 + n;
    if biased_exp <= 0 {
        return 0.0_f32;
    }
    if biased_exp >= 255 {
        // +inf with same sign
        return f32::from_bits((bits & 0x80000000) | 0x7F800000);
    }
    f32::from_bits((bits & 0x807FFFFF) | ((biased_exp as u32) << 23))
}

/// Round-to-nearest-even, matching CUDA `rintf()`.
///
/// Rust's `f32::round()` uses round-half-away-from-zero (0.5 → 1.0),
/// but CUDA's `rintf()` uses round-to-nearest-even (0.5 → 0.0, 1.5 → 2.0).
/// This difference caused 3/10000 parity failures in stress testing.
#[inline]
fn rintf(x: f32) -> f32 {
    // nearbyint/rint semantics: round to nearest, ties to even.
    // On x86, this maps to the SSE4.1 ROUNDSS instruction (or frndint).
    // Rust's f32::round() is the wrong function (round-half-away-from-zero).
    //
    // The correct Rust function is f32::round_ties_even() (stable since 1.77).
    // Fallback for older compilers: (x + 0.5).floor() has edge cases,
    // so we use the explicit bit-level algorithm if round_ties_even is unavailable.
    x.round_ties_even()
}

/// Canonical exp function for deterministic softmax.
///
/// Computes exp(x) via exp2(x * LOG2_E) using a frozen minimax polynomial
/// for 2^f on [-0.5, 0.5]. Both prover (GPU) and verifier (CPU) must
/// evaluate this identical polynomial in Horner form.
///
/// All library-dependent math is eliminated:
/// - Uses bit-level ldexp (no libm scalbn/ldexp)
/// - Uses round-to-nearest-even (matching CUDA rintf, not Rust round)
/// - No FMA (Rust default for f32)
/// - Polynomial coefficients are protocol constants
#[inline]
fn exp_canonical(x: f32) -> f32 {
    // exp(x) = 2^(x * log2(e))
    let t = x * 1.4426950216293335_f32; // log2(e) rounded to f32
    let n = rintf(t);
    let f = t - n;
    // Horner form (inside-out): p = c4*f + c3, then *f + c2, etc.
    let mut p = f * 0.009618129_f32;
    p = p + 0.055504109_f32;
    p = p * f;
    p = p + 0.240226507_f32;
    p = p * f;
    p = p + 0.693147182_f32;
    p = p * f;
    p = p + 1.0_f32;
    // p * 2^n via bit manipulation (no library ldexp)
    ldexp_bitwise(p, n as i32)
}

/// Result of deterministic attention replay.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeterministicAttentionResult {
    /// Attention output: `n_q_heads * d_head` f32 values, row-major.
    pub output_f32: Vec<f32>,
    /// Per-head softmax weights (for diagnostics): `n_q_heads * seq_len`.
    pub softmax_weights: Vec<f32>,
}

/// Compute the canonical inv_sqrt_d for a given d_head.
///
/// This is the single source of truth for the softmax scale factor.
/// Both CPU and GPU must use the **same f32 bits** produced by this function.
/// The harness calls this once and passes the result to both sides.
pub fn canonical_inv_sqrt_d(d_head: usize) -> f32 {
    1.0_f32 / (d_head as f32).sqrt()
}

/// Deterministic attention: bit-exact CPU reference implementation (v2).
///
/// Computes single-query decode attention with fully specified arithmetic:
/// - Score: parallel multiply + tree_reduce_sum (v2 — replaces v1 serial)
/// - Softmax: canonical exp polynomial + sequential f32 sum (unchanged from v1)
/// - V aggregation: sequential f32 weighted sum (unchanged from v1)
///
/// # Arguments
///
/// * `q_bf16` — Query vector, bf16-encoded as u16, shape `[n_q_heads * d_head]`
/// * `k_bf16` — Key cache, bf16-encoded as u16, shape `[seq_len][n_kv_heads * d_head]`
/// * `v_bf16` — Value cache, bf16-encoded as u16, shape `[seq_len][n_kv_heads * d_head]`
/// * `cfg` — Model config (head counts, dimensions)
/// * `inv_sqrt_d` — Precomputed 1/sqrt(d_head), same f32 bits as GPU receives
///
/// # Returns
///
/// Attention output in f32, shape `[n_q_heads * d_head]`, plus diagnostic softmax weights.
pub fn replay_attention_deterministic(
    q_bf16: &[u16],
    k_bf16: &[Vec<u16>],
    v_bf16: &[Vec<u16>],
    cfg: &ModelConfig,
    inv_sqrt_d: f32,
) -> DeterministicAttentionResult {
    let n_q_heads = cfg.n_q_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let d_head = cfg.d_head;
    let seq_len = k_bf16.len();
    let heads_per_kv = n_q_heads / n_kv_heads;

    assert_eq!(q_bf16.len(), n_q_heads * d_head);
    assert!(seq_len > 0);

    let mut output = vec![0.0_f32; n_q_heads * d_head];
    let mut all_weights = vec![0.0_f32; n_q_heads * seq_len];

    // Reusable buffer for per-element products (avoids allocation per dot product).
    let mut products = vec![0.0_f32; d_head];

    for qh in 0..n_q_heads {
        let kv_group = qh / heads_per_kv;

        // --- Step 1 (v2): Scores via parallel multiply + tree reduce ---
        let mut scores = vec![0.0_f32; seq_len];

        for t in 0..seq_len {
            // Phase A: element-wise multiply (embarrassingly parallel on GPU)
            for i in 0..d_head {
                let q_val = bf16_to_f32(q_bf16[qh * d_head + i]);
                let k_val = bf16_to_f32(k_bf16[t][kv_group * d_head + i]);
                products[i] = q_val * k_val;
            }
            // Phase B: fixed binary tree reduction
            let dot = tree_reduce_sum_f32(&products);
            scores[t] = dot * inv_sqrt_d;
        }

        // --- Step 2 (v3): Softmax — parallel max, exp, tree sum, normalize ---

        // 2a. Max via tree reduce (parallel on GPU)
        let max_score = tree_reduce_max_f32(&scores);

        // 2b. Exp (embarrassingly parallel on GPU)
        let mut exp_scores = vec![0.0_f32; seq_len];
        for t in 0..seq_len {
            exp_scores[t] = exp_canonical(scores[t] - max_score);
        }

        // 2c. Sum via tree reduce (parallel on GPU)
        let sum_exp = tree_reduce_sum_f32(&exp_scores);

        // 2d. Normalize (embarrassingly parallel on GPU)
        for t in 0..seq_len {
            let w = exp_scores[t] / sum_exp;
            all_weights[qh * seq_len + t] = w;
        }

        // --- Step 3: V aggregation (v4 — tiled partials + tree merge) ---
        const TILE_SIZE: usize = 128;
        for i in 0..d_head {
            let mut tile_partials: Vec<f32> = Vec::new();
            let mut tile_start = 0;
            while tile_start < seq_len {
                let tile_end = (tile_start + TILE_SIZE).min(seq_len);
                let mut partial: f32 = 0.0;
                for t in tile_start..tile_end {
                    let v_val = bf16_to_f32(v_bf16[t][kv_group * d_head + i]);
                    let w = all_weights[qh * seq_len + t];
                    let prod = w * v_val;
                    partial = partial + prod;
                }
                tile_partials.push(partial);
                tile_start += TILE_SIZE;
            }
            output[qh * d_head + i] = tree_reduce_sum_f32(&tile_partials);
        }
    }

    DeterministicAttentionResult {
        output_f32: output,
        softmax_weights: all_weights,
    }
}

/// Convert bf16 (stored as u16) to f32. Lossless.
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bf16 (round-to-nearest-even). For output requantization.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round-to-nearest-even: add 0x7FFF + bit 16 (the "round" bit)
    let round_bit = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + round_bit);
    (rounded >> 16) as u16
}

/// Deterministic attention with f32 inputs (v2 — tree-reduced scores).
pub fn replay_attention_deterministic_f32(
    q_f32: &[f32],
    k_f32: &[Vec<f32>],
    v_f32: &[Vec<f32>],
    cfg: &ModelConfig,
    inv_sqrt_d: f32,
) -> DeterministicAttentionResult {
    let n_q_heads = cfg.n_q_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let d_head = cfg.d_head;
    let seq_len = k_f32.len();
    let heads_per_kv = n_q_heads / n_kv_heads;

    assert_eq!(q_f32.len(), n_q_heads * d_head);
    assert!(seq_len > 0);

    let mut output = vec![0.0_f32; n_q_heads * d_head];
    let mut all_weights = vec![0.0_f32; n_q_heads * seq_len];
    let mut products = vec![0.0_f32; d_head];

    for qh in 0..n_q_heads {
        let kv_group = qh / heads_per_kv;

        // Step 1 (v2): Scores via parallel multiply + tree reduce
        let mut scores = vec![0.0_f32; seq_len];

        for t in 0..seq_len {
            for i in 0..d_head {
                products[i] = q_f32[qh * d_head + i] * k_f32[t][kv_group * d_head + i];
            }
            let dot = tree_reduce_sum_f32(&products);
            scores[t] = dot * inv_sqrt_d;
        }

        // Step 2 (v3): Softmax — parallel max, exp, tree sum, normalize
        let max_score = tree_reduce_max_f32(&scores);
        let mut exp_scores = vec![0.0_f32; seq_len];
        for t in 0..seq_len {
            exp_scores[t] = exp_canonical(scores[t] - max_score);
        }
        let sum_exp = tree_reduce_sum_f32(&exp_scores);
        for t in 0..seq_len {
            all_weights[qh * seq_len + t] = exp_scores[t] / sum_exp;
        }

        // Step 3: V aggregation (v4 — tiled partials + tree merge)
        const TILE_SIZE: usize = 128;
        for i in 0..d_head {
            let mut tile_partials: Vec<f32> = Vec::new();
            let mut tile_start = 0;
            while tile_start < seq_len {
                let tile_end = (tile_start + TILE_SIZE).min(seq_len);
                let mut partial: f32 = 0.0;
                for t in tile_start..tile_end {
                    let w = all_weights[qh * seq_len + t];
                    let prod = w * v_f32[t][kv_group * d_head + i];
                    partial = partial + prod;
                }
                tile_partials.push(partial);
                tile_start += TILE_SIZE;
            }
            output[qh * d_head + i] = tree_reduce_sum_f32(&tile_partials);
        }
    }

    DeterministicAttentionResult {
        output_f32: output,
        softmax_weights: all_weights,
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
        assert_eq!(
            compare_attention_output(&claimed, &replayed, &tol),
            Some(i16::MAX)
        );

        // Extended claimed vector is also rejected
        let claimed2 = vec![1i8, 2, 3, 4, 5];
        assert_eq!(
            compare_attention_output(&claimed2, &replayed, &tol),
            Some(i16::MAX)
        );
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
        let claimed = vec![10i8, 20, 30, 40, 50, 60];
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

    // ──── Witnessed score replay tests ────

    #[test]
    fn test_witnessed_single_token_equals_v() {
        // With one token, softmax([any_score]) = [1.0], so output = V.
        let cfg = toy_cfg();
        let seq_len = 1;
        // Any scores — with one position, softmax is always [1.0].
        let scores: Vec<f32> = vec![42.0; cfg.n_q_heads * seq_len];
        // V values: distinct per KV-head dim
        let v = vec![
            3.0f64, 7.0, // kv_head 0: d_head=2
            -2.0, 5.0, // kv_head 1: d_head=2
        ];
        let scale_a = 1.0;

        let (a_i8, a_f64) =
            replay_attention_witnessed_scores(&scores, cfg.n_q_heads, seq_len, &[v.clone()], scale_a, &cfg);

        assert_eq!(a_i8.len(), cfg.hidden_dim);
        let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            for i in 0..cfg.d_head {
                let expected = v[kv_head * cfg.d_head + i];
                assert!(
                    (a_f64[qh * cfg.d_head + i] - expected).abs() < 1e-10,
                    "qh={} i={}: expected {}, got {}",
                    qh,
                    i,
                    expected,
                    a_f64[qh * cfg.d_head + i]
                );
            }
        }
    }

    #[test]
    fn test_witnessed_matches_full_replay_exact_scores() {
        // When witnessed scores are computed from exact f64 Q·K^T/√d,
        // the witnessed-score replay must produce identical a_i8 as full replay.
        let cfg = toy_cfg();
        let d_head = cfg.d_head;
        let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
        let inv_sqrt_d = 1.0 / (d_head as f64).sqrt();

        // Build synthetic Q (post-RoPE, f64) and KV cache
        let q_roped: Vec<f64> = (0..cfg.hidden_dim)
            .map(|i| (i as f64 * 0.1) - 0.5)
            .collect();
        let kv_k: Vec<Vec<f64>> = (0..3)
            .map(|t| {
                (0..cfg.kv_dim)
                    .map(|i| ((t * cfg.kv_dim + i) as f64 * 0.05) - 0.3)
                    .collect()
            })
            .collect();
        let kv_v: Vec<Vec<f64>> = (0..3)
            .map(|t| {
                (0..cfg.kv_dim)
                    .map(|i| ((t * cfg.kv_dim + i) as f64 * 0.2) - 1.0)
                    .collect()
            })
            .collect();
        let scale_a = 0.05;

        // Compute exact f64 scores per head
        let seq_len = kv_k.len();
        let mut exact_scores = vec![0.0f32; cfg.n_q_heads * seq_len];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            for t in 0..seq_len {
                let dot: f64 = (0..d_head)
                    .map(|i| q_roped[qh * d_head + i] * kv_k[t][kv_head * d_head + i])
                    .sum();
                exact_scores[qh * seq_len + t] = (dot * inv_sqrt_d) as f32;
            }
        }

        // Full replay (reference)
        let (ref_i8, _) = replay_attention_roped_raw(
            &q_roped, &kv_k, &kv_v, scale_a, &cfg,
        );

        // Witnessed replay with exact scores
        let (wit_i8, _) = replay_attention_witnessed_scores(
            &exact_scores, cfg.n_q_heads, seq_len, &kv_v, scale_a, &cfg,
        );

        // They should match exactly when scores are exact f64→f32→f64 roundtrip.
        // Small diffs from f32 truncation are acceptable (≤1).
        let max_diff: i16 = ref_i8
            .iter()
            .zip(wit_i8.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 1,
            "witnessed vs full replay max diff = {} (expected ≤1 from f32 truncation)",
            max_diff
        );
    }

    #[test]
    fn test_canonical_scores_matches_replay_scores() {
        // Verify compute_canonical_scores produces the same scores as
        // replay_attention_roped_raw's internal Q·K^T/√d computation.
        let cfg = ModelConfig::toy(); // 8 Q heads, 2 KV heads, d_head=2
        let q_roped = vec![1.0f64, 0.5, -0.3, 0.7, 0.2, 0.9, -0.1, 0.4,
                           0.6, -0.2, 0.3, 0.8, -0.5, 0.1, 0.7, -0.3];
        let kv_k = vec![
            vec![0.5, 0.3, -0.1, 0.6],  // pos 0, 2 KV heads × d_head=2
            vec![-0.2, 0.8, 0.4, -0.3],  // pos 1
        ];

        let canonical = compute_canonical_scores(&q_roped, &kv_k, &cfg);
        assert_eq!(canonical.len(), cfg.n_q_heads * 2);

        // Check a few values manually.
        // Head 0 (KV head 0): Q=[1.0, 0.5], K0=[0.5, 0.3], K1=[-0.2, 0.8]
        let inv_sqrt_d = 1.0 / (2.0f64).sqrt();
        let expected_h0_t0 = (1.0 * 0.5 + 0.5 * 0.3) * inv_sqrt_d;
        let expected_h0_t1 = (1.0 * -0.2 + 0.5 * 0.8) * inv_sqrt_d;
        assert!((canonical[0] - expected_h0_t0).abs() < 1e-12);
        assert!((canonical[1] - expected_h0_t1).abs() < 1e-12);
    }

    #[test]
    fn test_anchor_honest_scores_accepted() {
        // Honest witnessed scores = canonical scores truncated to f32.
        // The gap should be tiny (f64→f32 rounding only).
        let cfg = ModelConfig::toy();
        let q_roped = vec![1.0f64, 0.5, -0.3, 0.7, 0.2, 0.9, -0.1, 0.4,
                           0.6, -0.2, 0.3, 0.8, -0.5, 0.1, 0.7, -0.3];
        let kv_k = vec![
            vec![0.5, 0.3, -0.1, 0.6],
            vec![-0.2, 0.8, 0.4, -0.3],
            vec![0.1, -0.5, 0.7, 0.2],
        ];

        let canonical = compute_canonical_scores(&q_roped, &kv_k, &cfg);
        // "Honest" witnessed = canonical truncated to f32 (simulates GPU fp16→f32 path)
        let witnessed: Vec<f32> = canonical.iter().map(|&v| v as f32).collect();

        let stats = anchor_witnessed_scores(&witnessed, &canonical, cfg.n_q_heads, 3, 0);
        // f64→f32 gap should be negligible (< 1e-6 for small values)
        assert!(
            stats.max_gap < 1e-6,
            "honest scores gap {} too large (expected < 1e-6)",
            stats.max_gap
        );
    }

    #[test]
    fn test_anchor_tampered_scores_rejected() {
        // Tampered: add 1.0 to all scores (shifts softmax weights).
        let cfg = ModelConfig::toy();
        let q_roped = vec![1.0f64, 0.5, -0.3, 0.7, 0.2, 0.9, -0.1, 0.4,
                           0.6, -0.2, 0.3, 0.8, -0.5, 0.1, 0.7, -0.3];
        let kv_k = vec![
            vec![0.5, 0.3, -0.1, 0.6],
            vec![-0.2, 0.8, 0.4, -0.3],
        ];

        let canonical = compute_canonical_scores(&q_roped, &kv_k, &cfg);
        // Tampered: offset by 1.0
        let tampered: Vec<f32> = canonical.iter().map(|&v| (v + 1.0) as f32).collect();

        let stats = anchor_witnessed_scores(&tampered, &canonical, cfg.n_q_heads, 2, 0);
        assert!(
            stats.max_gap > 0.9,
            "tampered scores gap {} too small (expected > 0.9)",
            stats.max_gap
        );
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_anchor_wrong_length_panics() {
        let cfg = ModelConfig::toy();
        let canonical = vec![0.0f64; cfg.n_q_heads * 3]; // 3 positions
        let wrong_len = vec![0.0f32; cfg.n_q_heads * 2]; // 2 positions — mismatch
        anchor_witnessed_scores(&wrong_len, &canonical, cfg.n_q_heads, 3, 0);
    }

    #[test]
    fn test_dequant_bias_fp16_truncates() {
        // Verify fp16 truncation actually reduces precision.
        let acc = vec![12345i32, -67890, 100000];
        let scales = vec![0.00123f32, 0.00456, 0.00789];
        let scale_x = 0.5f32;
        let bias = vec![100.0f32, -200.0, 300.0];

        let result = dequant_bias_fp16(&acc, &scales, scale_x, Some(&bias));
        assert_eq!(result.len(), 3);

        // Each result should be representable in fp16 (no extra f32 precision bits).
        for &v in &result {
            let roundtripped = half::f16::from_f32(v).to_f32();
            assert_eq!(v, roundtripped, "value {v} should be fp16-exact after truncation");
        }
    }

    #[test]
    fn test_cutlass_epilogue_bf16_no_bias() {
        // Without bias: bf16(scale_a * (f32(acc) * scale_b))
        let acc = vec![12345i32, -67890, 100000];
        let scales = vec![0.00123f32, 0.00456, 0.00789];
        let scale_a = 0.5f32;

        let result = cutlass_epilogue_bf16(&acc, &scales, scale_a, None);
        assert_eq!(result.len(), 3);

        // Each result should be bf16-exact.
        for &v in &result {
            let roundtripped = half::bf16::from_f32(v).to_f32();
            assert_eq!(v, roundtripped, "value {v} should be bf16-exact");
        }

        // Verify the computation order: scale_b first, then scale_a.
        let bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };
        let expected_0 = bf16((12345.0f32 * 0.00123) * 0.5);
        assert_eq!(result[0], expected_0);
    }

    #[test]
    fn test_cutlass_epilogue_bf16_with_bias_fma() {
        // With bias: bf16(fma(scale_a, f32(acc) * scale_b, bias))
        let acc = vec![12345i32, -67890];
        let scales = vec![0.00123f32, 0.00456];
        let scale_a = 0.5f32;
        let bias = vec![1.5f32, -2.0];

        let result = cutlass_epilogue_bf16(&acc, &scales, scale_a, Some(&bias));

        let bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };

        // FMA: scale_a * temp + bias (single rounding)
        let temp_0 = 12345.0f32 * 0.00123;
        let expected_0 = bf16(0.5f32.mul_add(temp_0, 1.5));
        assert_eq!(result[0], expected_0);

        // Verify FMA differs from separate multiply+add for some inputs.
        // (This may or may not differ depending on values, but the code path is correct.)
        let temp_1 = (-67890.0f32) * 0.00456;
        let expected_1 = bf16(0.5f32.mul_add(temp_1, -2.0));
        assert_eq!(result[1], expected_1);
    }

    #[test]
    fn test_cutlass_epilogue_vs_old_dequant_no_bias() {
        // Without bias, cutlass_epilogue and dequant_bias_bf16 should be identical.
        // Both compute bf16(f32(acc) * scale_b * scale_a) — same order, no FMA.
        let acc: Vec<i32> = (0..128).map(|i| (i * 137 - 8000) as i32).collect();
        let scales: Vec<f32> = (0..128).map(|i| 0.001 + (i as f32) * 0.0001).collect();
        let scale_a = 0.42f32;

        let old = dequant_bias_bf16(&acc, &scales, scale_a, None);
        let new = cutlass_epilogue_bf16(&acc, &scales, scale_a, None);

        assert_eq!(old.len(), new.len());
        for (i, (&o, &n)) in old.iter().zip(new.iter()).enumerate() {
            assert_eq!(o, n, "mismatch at index {i}: old={o}, new={n}");
        }
    }

    #[test]
    fn test_apply_rope_f32_matches_f64_closely() {
        let cfg = ModelConfig::toy();
        let d = cfg.d_head;
        let n_heads = cfg.n_q_heads;
        let q_f64: Vec<f64> = (0..n_heads * d).map(|i| (i as f64) * 0.01).collect();
        let q_f32: Vec<f32> = q_f64.iter().map(|&x| x as f32).collect();

        let roped_f64 = crate::rope::apply_rope_q(&q_f64, 42, &cfg);
        let roped_f32 = apply_rope_f32(&q_f32, n_heads, 42, &cfg);

        assert_eq!(roped_f64.len(), roped_f32.len());
        let max_diff: f64 = roped_f64
            .iter()
            .zip(roped_f32.iter())
            .map(|(&a, &b)| (a - b as f64).abs())
            .fold(0.0, f64::max);
        // f32 RoPE should be very close to f64 RoPE for small values.
        assert!(
            max_diff < 0.01,
            "f32 vs f64 RoPE max diff {max_diff} too large"
        );
    }

    #[test]
    fn test_bf16_anchor_pipeline_low_gap() {
        // Regression test: the full bf16 anchoring pipeline should produce
        // a near-zero anchor gap when both sides use the same data.
        //
        // Pipeline: cutlass_epilogue_bf16 → apply_rope_f32 → compute_canonical_scores_gpu_like
        //           → anchor_witnessed_scores (gap should be < 0.25, matching Llama strong tier).
        //
        // This catches: bf16 truncation bugs, RoPE off-by-one, FMA mismatches.
        let cfg = ModelConfig::toy();
        let d = cfg.d_head;
        let half = d / 2;
        let seq_len = 5;

        // Simulate INT32 accumulators (realistic magnitude range for W8A8).
        let q_acc: Vec<i32> = (0..cfg.n_q_heads * d)
            .map(|i| ((i as i32) * 37 - 3000))
            .collect();
        let k_accs: Vec<Vec<i32>> = (0..seq_len)
            .map(|t| {
                (0..cfg.n_kv_heads * d)
                    .map(|i| ((t * 1000 + i) as i32 * 23 - 2000))
                    .collect()
            })
            .collect();

        // Per-channel scales and per-token scale (realistic W8A8 values).
        let q_scales: Vec<f32> = (0..cfg.n_q_heads * d)
            .map(|i| 0.001 + (i as f32) * 0.0001)
            .collect();
        let k_scales: Vec<f32> = (0..cfg.n_kv_heads * d)
            .map(|i| 0.002 + (i as f32) * 0.00015)
            .collect();
        let scale_a = 0.05f32;

        // Reconstruct Q: epilogue → RoPE in f32 (no bf16 truncation).
        let q_f32 = cutlass_epilogue_bf16(&q_acc, &q_scales, scale_a, None);
        let q_roped = apply_rope_f32(&q_f32, cfg.n_q_heads, seq_len - 1, &cfg);

        // Reconstruct K for each position: epilogue → RoPE in f32.
        let kv_k: Vec<Vec<f32>> = k_accs
            .iter()
            .enumerate()
            .map(|(t, k_acc)| {
                let k_f32 = cutlass_epilogue_bf16(k_acc, &k_scales, scale_a, None);
                apply_rope_f32(&k_f32, cfg.n_kv_heads, t, &cfg)
            })
            .collect();

        // Canonical scores in f32.
        let canonical_f32 = compute_canonical_scores_gpu_like(&q_roped, &kv_k, &cfg);

        // "Witnessed" scores = same canonical scores (simulates honest prover).
        let witnessed: Vec<f32> = canonical_f32.clone();
        let canonical_f64: Vec<f64> = canonical_f32.iter().map(|&x| x as f64).collect();

        let stats = anchor_witnessed_scores(
            &witnessed, &canonical_f64, cfg.n_q_heads, seq_len, 0,
        );

        // When both sides compute identically, the gap should be exactly 0
        // (or at most f32→f64 rounding, which is < 1e-6).
        assert!(
            stats.max_gap < 1e-6,
            "bf16 anchor pipeline gap {} too large (expected < 1e-6)",
            stats.max_gap
        );
    }

    #[test]
    fn test_bf16_anchor_pipeline_rejects_wrong_position() {
        // If the verifier uses a wrong RoPE position, the gap should be large.
        // This catches the off-by-one regression (j+1 vs j).
        let cfg = ModelConfig::toy();
        let d = cfg.d_head;
        let seq_len = 5;

        let k_acc: Vec<i32> = (0..cfg.n_kv_heads * d)
            .map(|i| (i as i32 * 23 - 2000))
            .collect();
        let k_scales: Vec<f32> = (0..cfg.n_kv_heads * d)
            .map(|i| 0.002 + (i as f32) * 0.00015)
            .collect();
        let scale_a = 0.05f32;

        let k_f32 = cutlass_epilogue_bf16(&k_acc, &k_scales, scale_a, None);

        // Position 3 vs position 4 should give different RoPE outputs.
        let k_pos3 = apply_rope_f32(&k_f32, cfg.n_kv_heads, 3, &cfg);
        let k_pos4 = apply_rope_f32(&k_f32, cfg.n_kv_heads, 4, &cfg);

        let max_diff: f32 = k_pos3
            .iter()
            .zip(k_pos4.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Different positions must give measurably different K vectors.
        assert!(
            max_diff > 0.01,
            "RoPE position 3 vs 4 diff {} too small — off-by-one would not be caught",
            max_diff
        );
    }

    #[test]
    fn test_gpu_like_scores_match_f64_scores_closely() {
        let cfg = ModelConfig::toy();
        let d = cfg.d_head;
        let seq_len = 5;

        // Build Q and K in both precisions.
        let q_f64: Vec<f64> = (0..cfg.n_q_heads * d).map(|i| (i as f64) * 0.1).collect();
        let q_f32: Vec<f32> = q_f64.iter().map(|&x| x as f32).collect();
        let kv_f64: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| (0..cfg.n_kv_heads * d).map(|i| ((t * 100 + i) as f64) * 0.1).collect())
            .collect();
        let kv_f32: Vec<Vec<f32>> = kv_f64
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();

        let scores_f64 = compute_canonical_scores(&q_f64, &kv_f64, &cfg);
        let scores_f32 = compute_canonical_scores_gpu_like(&q_f32, &kv_f32, &cfg);

        assert_eq!(scores_f64.len(), scores_f32.len());
        let max_diff: f64 = scores_f64
            .iter()
            .zip(scores_f32.iter())
            .map(|(&a, &b)| (a - b as f64).abs())
            .fold(0.0, f64::max);
        // Scores should be close (same data, different precision).
        assert!(
            max_diff < 0.1,
            "f64 vs f32 scores max diff {max_diff} too large"
        );
    }

    #[test]
    fn test_tiled_single_tile_matches_global() {
        // When block_n >= seq_len, tiled should behave like single-pass softmax.
        let cfg = toy_cfg();
        let n_q_heads = cfg.n_q_heads;
        let seq_len = 4;

        // Synthetic witnessed scores (already scaled by 1/sqrt(d)).
        let mut scores = vec![0.0_f32; n_q_heads * seq_len];
        for qh in 0..n_q_heads {
            for t in 0..seq_len {
                scores[qh * seq_len + t] = ((qh as f32) * 0.3 + (t as f32) * 0.5) - 1.0;
            }
        }

        // V cache: distinct values per position.
        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let v_deq: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| {
                (0..kv_dim)
                    .map(|i| (t as f64) * 10.0 + (i as f64))
                    .collect()
            })
            .collect();

        let scale_a = 0.05;

        let (_i8_global, f64_global) = replay_attention_witnessed_scores(
            &scores, n_q_heads, seq_len, &v_deq, scale_a, &cfg,
        );

        // block_n = seq_len => single tile, should be very close.
        let (_i8_tiled, f64_tiled) = replay_attention_witnessed_scores_tiled(
            &scores, n_q_heads, seq_len, &v_deq, scale_a, &cfg, seq_len,
        );

        let max_diff: f64 = f64_global
            .iter()
            .zip(f64_tiled.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);

        // Tiled f32 vs global f64 will have some precision gap, but small.
        assert!(
            max_diff < 0.5,
            "single-tile tiled vs global max diff {max_diff} too large"
        );
    }

    #[test]
    fn test_tiled_multi_tile_reasonable() {
        // Multiple tiles: verify the online recurrence produces reasonable output.
        let cfg = toy_cfg();
        let n_q_heads = cfg.n_q_heads;
        let seq_len = 10;
        let block_n = 3; // Forces 4 tiles: [0..3], [3..6], [6..9], [9..10]

        let mut scores = vec![0.0_f32; n_q_heads * seq_len];
        for qh in 0..n_q_heads {
            for t in 0..seq_len {
                scores[qh * seq_len + t] = ((qh as f32) * 0.1 + (t as f32) * 0.2) - 1.0;
            }
        }

        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let v_deq: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| {
                (0..kv_dim)
                    .map(|i| (t as f64) * 5.0 + (i as f64) * 0.1)
                    .collect()
            })
            .collect();

        let scale_a = 0.1;

        let (_i8_global, f64_global) = replay_attention_witnessed_scores(
            &scores, n_q_heads, seq_len, &v_deq, scale_a, &cfg,
        );
        let (_i8_tiled, f64_tiled) = replay_attention_witnessed_scores_tiled(
            &scores, n_q_heads, seq_len, &v_deq, scale_a, &cfg, block_n,
        );

        let max_diff: f64 = f64_global
            .iter()
            .zip(f64_tiled.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);

        // f32 tiled online-softmax vs f64 global: expect small gap from precision.
        // This is the diagnostic: if max_diff is small (~<1.0), the tiling itself
        // is not the gap source; if large, tiling order matters.
        assert!(
            max_diff < 1.0,
            "multi-tile tiled vs global max diff {max_diff} too large"
        );
    }

    #[test]
    fn test_tiled_block_n_1_matches() {
        // Extreme: block_n=1 means each KV position is its own tile.
        // Maximum number of rescaling steps — tests numerical stability.
        let cfg = toy_cfg();
        let n_q_heads = cfg.n_q_heads;
        let seq_len = 8;

        let mut scores = vec![0.0_f32; n_q_heads * seq_len];
        for qh in 0..n_q_heads {
            for t in 0..seq_len {
                scores[qh * seq_len + t] = (t as f32) * 0.3 - 0.5;
            }
        }

        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let v_deq: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| (0..kv_dim).map(|i| (t * 3 + i) as f64).collect())
            .collect();

        let scale_a = 0.08;

        let (_i8_global, f64_global) = replay_attention_witnessed_scores(
            &scores, n_q_heads, seq_len, &v_deq, scale_a, &cfg,
        );
        let (i8_tiled, f64_tiled) = replay_attention_witnessed_scores_tiled(
            &scores, n_q_heads, seq_len, &v_deq, scale_a, &cfg, 1,
        );

        let max_diff: f64 = f64_global
            .iter()
            .zip(f64_tiled.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff < 1.0,
            "block_n=1 tiled vs global max diff {max_diff} too large"
        );

        // i8 should be very close too (at most ±1 from rounding).
        let max_i8_diff: i16 = _i8_global
            .iter()
            .zip(i8_tiled.iter())
            .map(|(&a, &b)| ((a as i16) - (b as i16)).abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_i8_diff <= 1,
            "block_n=1 i8 max diff {max_i8_diff} too large"
        );
    }

    // ──── Attention evidence + certification tests ────

    #[test]
    fn test_attention_evidence_concentrated() {
        // Create scores where position 0 has a very high score (dominates softmax).
        // With a huge score at position 0 and small/zero scores elsewhere,
        // softmax mass concentrates on position 0.
        let cfg = toy_cfg();
        let n_q_heads = cfg.n_q_heads;
        let seq_len = 10;
        let top_k = 2;

        // All heads: position 0 gets score 100.0, rest get 0.0.
        let mut scores = vec![0.0f32; n_q_heads * seq_len];
        for qh in 0..n_q_heads {
            scores[qh * seq_len] = 100.0;
        }

        // V cache with nonzero values.
        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let v_deq: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| {
                (0..kv_dim)
                    .map(|i| ((t * 10 + i) as f64) * 0.5)
                    .collect()
            })
            .collect();

        let evidence =
            compute_attention_evidence(&scores, n_q_heads, seq_len, &v_deq, &cfg, top_k);

        // All mass should be in top-k (essentially all in position 0).
        assert!(
            evidence.min_top_k_mass > 0.999,
            "concentrated: min_top_k_mass {:.6} should be > 0.999",
            evidence.min_top_k_mass
        );
        // Tail bound should be near 0 (tail_mass ~ 0).
        assert!(
            evidence.max_tail_bound < 0.01,
            "concentrated: max_tail_bound {:.6} should be < 0.01",
            evidence.max_tail_bound
        );
        assert_eq!(evidence.top_k, top_k);
        assert_eq!(evidence.head_curves.len(), n_q_heads);
        // Check curves: prefix mass at k=1 should be ~1.0 (all mass in position 0).
        for curve in &evidence.head_curves {
            assert!(curve.prefix_mass[0] > 0.999, "prefix_mass[0] should be ~1.0");
            assert!(curve.tail_bounds[0] < 0.01, "tail_bounds[0] should be ~0");
        }
    }

    #[test]
    fn test_attention_evidence_uniform() {
        // Create uniform scores (all positions equal).
        // Softmax of uniform scores is uniform: each weight = 1/seq_len.
        // top-k mass = k/seq_len.
        let cfg = toy_cfg();
        let n_q_heads = cfg.n_q_heads;
        let seq_len = 10;
        let top_k = 3;

        // All scores equal.
        let scores = vec![1.0f32; n_q_heads * seq_len];

        // V cache with some magnitude.
        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let v_deq: Vec<Vec<f64>> = (0..seq_len)
            .map(|_| vec![5.0f64; kv_dim])
            .collect();

        let evidence =
            compute_attention_evidence(&scores, n_q_heads, seq_len, &v_deq, &cfg, top_k);

        // Expected top-k mass at k=3: 3/10 = 0.3
        let expected_mass = top_k as f32 / seq_len as f32;
        for curve in &evidence.head_curves {
            let mass_at_k = curve.prefix_mass[top_k - 1];
            assert!(
                (mass_at_k - expected_mass).abs() < 0.01,
                "uniform: prefix_mass[{}] = {:.4} should be close to {:.4}",
                top_k - 1, mass_at_k, expected_mass
            );
        }

        // Tail bound at k=3: tail_mass = 0.7, max|V| = 5.0 → bound = 3.5
        let expected_tail_bound = (1.0 - expected_mass) as f64 * 5.0;
        assert!(
            (evidence.max_tail_bound - expected_tail_bound).abs() < 0.1,
            "uniform: max_tail_bound {:.4} should be close to {:.4}",
            evidence.max_tail_bound,
            expected_tail_bound
        );
    }

    #[test]
    fn test_attention_evidence_top_k_exceeds_seq_len() {
        // When top_k > seq_len, effective_k = seq_len.
        // All mass is captured at full seq_len.
        let cfg = toy_cfg();
        let n_q_heads = cfg.n_q_heads;
        let seq_len = 3;
        let top_k = 100; // much larger than seq_len

        let scores = vec![0.5f32; n_q_heads * seq_len];
        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let v_deq: Vec<Vec<f64>> = (0..seq_len)
            .map(|_| vec![10.0f64; kv_dim])
            .collect();

        let evidence =
            compute_attention_evidence(&scores, n_q_heads, seq_len, &v_deq, &cfg, top_k);

        assert_eq!(evidence.top_k, seq_len);
        // All mass captured at last position in curve.
        for curve in &evidence.head_curves {
            let final_mass = *curve.prefix_mass.last().unwrap();
            assert!(
                (final_mass - 1.0).abs() < 1e-5,
                "top_k > seq_len: final prefix_mass {:.6} should be ~1.0",
                final_mass
            );
            // Last tail bound = 0 (no tail positions).
            let final_bound = *curve.tail_bounds.last().unwrap();
            assert!(
                final_bound < 1e-10,
                "top_k > seq_len: final tail_bound {:.6e} should be ~0",
                final_bound
            );
        }
        assert!(
            evidence.max_tail_bound < 1e-10,
            "top_k > seq_len: max_tail_bound {:.6e} should be ~0",
            evidence.max_tail_bound
        );
    }

    // Helper: create an AttentionEvidence with one curve per head.
    // `best_tail_bound` is the minimum achievable tail bound for that head
    // (reached at the largest k in the schedule that fits within seq_len).
    fn make_evidence(layer: usize, best_tail_bounds: &[f64]) -> AttentionEvidence {
        // Create a 256-position curve so k=4..128 all have meaningful entries.
        let seq_len = 256;
        let head_curves: Vec<HeadEvidenceCurve> = best_tail_bounds
            .iter()
            .map(|&tb| {
                // Uniform-ish weights with a long tail.
                let w = 1.0 / seq_len as f32;
                let sorted_weights = vec![w; seq_len];
                let prefix_mass: Vec<f32> = (1..=seq_len)
                    .map(|k| (k as f32) * w)
                    .collect();
                // Tail bound decreases linearly; at full k it's 0, at k=0 it's max.
                // Scale so that the best achievable (at k=128) is `tb`.
                // tail_bound[k] = tb * (seq_len - k - 1) / (seq_len - 129)
                // when seq_len=256, k=127: tb * 128/127 ≈ tb
                let scale = if seq_len > 129 { (seq_len - 129) as f64 } else { 1.0 };
                let tail_bounds: Vec<f64> = (0..seq_len)
                    .map(|k| {
                        let remaining = (seq_len - k - 1) as f64;
                        tb * remaining / scale
                    })
                    .collect();
                let max_v_abs = tb * (seq_len as f64) / scale;
                HeadEvidenceCurve {
                    sorted_weights,
                    prefix_mass,
                    tail_bounds,
                    max_v_abs,
                }
            })
            .collect();
        let max_tail_bound = best_tail_bounds
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        AttentionEvidence {
            layer,
            head_curves,
            min_top_k_mass: 0.5, // not used by adaptive certifier
            max_tail_bound,
            top_k: 16,
        }
    }

    #[test]
    fn test_certify_token_high_margin() {
        // Small tail bounds + high logit margin → certified.
        // 8 heads × ~0.001 attn bound each = ~0.008 attn total.
        // With 10x gain: ~0.08 logit bound. Margin = 5.0, target = 2.5 → certified.
        let evidence = vec![make_evidence(0, &[0.001; 8])];
        let cert = certify_token_adaptive(&evidence, 5.0, 128);
        assert!(cert.certified(), "should be certified: {}", cert.reason);
        assert!((cert.logit_margin - 5.0).abs() < 1e-6);
        assert!(
            cert.estimated_logit_bound < 2.5,
            "bound {:.4} should be < 2.5",
            cert.estimated_logit_bound
        );
    }

    #[test]
    fn test_certify_token_large_bound() {
        // Large tail bounds → not certified.
        // 8 heads × ~2.0 attn each = ~16 attn. With 10x gain: ~160 logit bound.
        // Margin = 10.0, target = 5.0. 160 >> 5.0 → not certified.
        let evidence = vec![make_evidence(0, &[2.0; 8])];
        let cert = certify_token_adaptive(&evidence, 10.0, 128);
        assert!(!cert.certified(), "should NOT be certified: {}", cert.reason);
        assert!(cert.reason.contains("budget_exhausted"));
    }

    #[test]
    fn test_certify_token_zero_margin() {
        // logit_margin = 0 → not certified (top token not distinct).
        let evidence = vec![make_evidence(0, &[0.001; 8])];
        let cert = certify_token_adaptive(&evidence, 0.0, 128);
        assert!(!cert.certified(), "should NOT be certified: {}", cert.reason);
        assert!(cert.reason.contains("margin_too_small"));
    }

    #[test]
    fn test_certify_token_multi_layer() {
        // Multi-layer: bounds are additive across layers.
        // Layer 0: 8 heads × 0.001 attn = 0.008 attn
        // Layer 1: 8 heads × 0.01 attn = 0.08 attn
        // Sum = 0.088 attn. With 10x gain: 0.88 logit bound.
        // Margin = 5.0, target = 2.5 → certified.
        let layer0 = make_evidence(0, &[0.001; 8]);
        let layer1_ok = make_evidence(1, &[0.01; 8]);

        let cert = certify_token_adaptive(&[layer0.clone(), layer1_ok], 5.0, 128);
        assert!(cert.certified(), "multi-layer should certify: {} (bound={:.4})", cert.reason, cert.estimated_logit_bound);

        // With much larger per-head bounds: 8 × 2.0 = 16 attn. 10x = 160. >> 2.5.
        let layer1_big = make_evidence(1, &[2.0; 8]);
        let cert2 = certify_token_adaptive(&[layer0, layer1_big], 5.0, 128);
        assert!(!cert2.certified(), "multi-layer should fail: {}", cert2.reason);
    }

    // ── Tree reduction primitive tests ─────────────────────────────

    #[test]
    fn test_tree_reduce_sum_empty() {
        assert_eq!(tree_reduce_sum_f32(&[]).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_single() {
        assert_eq!(tree_reduce_sum_f32(&[42.0]).to_bits(), 42.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_power_of_two() {
        // len=2: one level, buf[0] = 1.0 + 2.0 = 3.0
        assert_eq!(tree_reduce_sum_f32(&[1.0, 2.0]).to_bits(), 3.0_f32.to_bits());
        // len=4: two levels
        let r = tree_reduce_sum_f32(&[1.0, 2.0, 3.0, 4.0]);
        // Level 1 (stride=2): buf = [1+3, 2+4] = [4.0, 6.0]
        // Level 2 (stride=1): buf = [4+6] = [10.0]
        assert_eq!(r.to_bits(), 10.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_non_power_of_two() {
        // len=3: padded to 4 with 0.0
        // buf = [1.0, 2.0, 3.0, 0.0]
        // Level 1 (stride=2): [1+3, 2+0] = [4.0, 2.0]
        // Level 2 (stride=1): [4+2] = [6.0]
        assert_eq!(tree_reduce_sum_f32(&[1.0, 2.0, 3.0]).to_bits(), 6.0_f32.to_bits());

        // len=5: padded to 8
        let r = tree_reduce_sum_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // buf = [1,2,3,4,5,0,0,0]
        // stride=4: [1+5, 2+0, 3+0, 4+0] = [6,2,3,4]
        // stride=2: [6+3, 2+4] = [9,6]
        // stride=1: [9+6] = [15]
        assert_eq!(r.to_bits(), 15.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_7() {
        // Padded to 8: [1,2,3,4,5,6,7,0]
        // stride=4: [1+5, 2+6, 3+7, 4+0] = [6,8,10,4]
        // stride=2: [6+10, 8+4] = [16,12]
        // stride=1: [16+12] = [28]
        assert_eq!(tree_reduce_sum_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).to_bits(), 28.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_15() {
        let v: Vec<f32> = (1..=15).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        // Sum 1..15 = 120, but tree reduction may differ from sequential due
        // to f32 non-associativity. For these small integers, should be exact.
        assert_eq!(r.to_bits(), 120.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_16() {
        let v: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        assert_eq!(r.to_bits(), 136.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_31() {
        let v: Vec<f32> = (1..=31).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        // 31*32/2 = 496
        assert_eq!(r.to_bits(), 496.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_32() {
        let v: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        assert_eq!(r.to_bits(), 528.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_33() {
        let v: Vec<f32> = (1..=33).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        // 33*34/2 = 561
        assert_eq!(r.to_bits(), 561.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_127() {
        let v: Vec<f32> = (1..=127).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        // 127*128/2 = 8128
        assert_eq!(r.to_bits(), 8128.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_128() {
        let v: Vec<f32> = (1..=128).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        assert_eq!(r.to_bits(), 8256.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_129() {
        let v: Vec<f32> = (1..=129).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        // 129*130/2 = 8385
        assert_eq!(r.to_bits(), 8385.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_len_1024() {
        let v: Vec<f32> = (1..=1024).map(|i| i as f32).collect();
        let r = tree_reduce_sum_f32(&v);
        // 1024*1025/2 = 524800
        assert_eq!(r.to_bits(), 524800.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_cancellation() {
        // Large positive + large negative that nearly cancel.
        // Tests that tree reduction handles cancellation deterministically.
        let v = vec![1e8_f32, -1e8, 1.0, 2.0, 3.0];
        let r = tree_reduce_sum_f32(&v);
        // Padded: [1e8, -1e8, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        // stride=4: [1e8+3, -1e8+0, 1+0, 2+0] = [1e8+3, -1e8, 1, 2]
        // stride=2: [(1e8+3)+1, -1e8+2] = [1e8+4, -1e8+2]  (but 1e8+3=1e8, 1e8+1=1e8 in f32...)
        // Actually 1e8 + 3.0 = 1e8 in f32 (lost precision). So the tree result
        // may differ from the sequential sum. That's fine — what matters is
        // CPU and GPU agree on the SAME tree result. Just check determinism.
        let r2 = tree_reduce_sum_f32(&v);
        assert_eq!(r.to_bits(), r2.to_bits(), "must be deterministic");
    }

    #[test]
    fn test_tree_reduce_sum_all_zeros() {
        let v = vec![0.0_f32; 16];
        assert_eq!(tree_reduce_sum_f32(&v).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_sum_signed_zeros() {
        // IEEE 754: 0.0 + (-0.0) = 0.0 (positive zero)
        let v = vec![0.0_f32, -0.0, 0.0, -0.0];
        let r = tree_reduce_sum_f32(&v);
        // stride=2: [0.0+0.0, -0.0+(-0.0)] = [0.0, -0.0]
        // stride=1: [0.0+(-0.0)] = [0.0]
        assert_eq!(r.to_bits(), 0.0_f32.to_bits(), "sum of signed zeros should be +0.0");
    }

    #[test]
    fn test_tree_reduce_sum_very_small() {
        let v = vec![1e-38_f32; 8];
        let r = tree_reduce_sum_f32(&v);
        let r2 = tree_reduce_sum_f32(&v);
        assert_eq!(r.to_bits(), r2.to_bits());
        assert!(r > 0.0);
    }

    #[test]
    fn test_tree_reduce_sum_very_large() {
        let v = vec![1e38_f32; 4];
        let r = tree_reduce_sum_f32(&v);
        // 4 * 1e38 = 4e38 > f32::MAX ≈ 3.4e38, so this overflows to inf
        assert!(r.is_infinite() && r > 0.0);
    }

    // ── tree_reduce_max tests ──

    #[test]
    fn test_tree_reduce_max_empty() {
        assert_eq!(tree_reduce_max_f32(&[]).to_bits(), f32::NEG_INFINITY.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_single() {
        assert_eq!(tree_reduce_max_f32(&[42.0]).to_bits(), 42.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_power_of_two() {
        assert_eq!(tree_reduce_max_f32(&[1.0, 3.0]).to_bits(), 3.0_f32.to_bits());
        assert_eq!(tree_reduce_max_f32(&[3.0, 1.0]).to_bits(), 3.0_f32.to_bits());
        assert_eq!(tree_reduce_max_f32(&[1.0, 4.0, 2.0, 3.0]).to_bits(), 4.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_non_power_of_two() {
        // len=3, padded to 4 with -inf
        assert_eq!(tree_reduce_max_f32(&[1.0, 3.0, 2.0]).to_bits(), 3.0_f32.to_bits());
        // len=5, padded to 8 with -inf
        assert_eq!(tree_reduce_max_f32(&[1.0, 5.0, 3.0, 2.0, 4.0]).to_bits(), 5.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_len_7() {
        let v = vec![1.0, 7.0, 3.0, 5.0, 2.0, 6.0, 4.0];
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 7.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_len_15() {
        let v: Vec<f32> = (1..=15).map(|i| i as f32).collect();
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 15.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_len_33() {
        let v: Vec<f32> = (1..=33).map(|i| i as f32).collect();
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 33.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_len_127() {
        let v: Vec<f32> = (1..=127).map(|i| i as f32).collect();
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 127.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_len_128() {
        let v: Vec<f32> = (1..=128).map(|i| i as f32).collect();
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 128.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_len_129() {
        let v: Vec<f32> = (1..=129).map(|i| i as f32).collect();
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 129.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_len_1024() {
        let v: Vec<f32> = (1..=1024).map(|i| i as f32).collect();
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 1024.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_all_negative() {
        let v = vec![-10.0, -5.0, -20.0, -1.0];
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), (-1.0_f32).to_bits());
    }

    #[test]
    fn test_tree_reduce_max_repeated_maxima() {
        // Multiple elements equal to the max
        let v = vec![1.0, 5.0, 3.0, 5.0, 2.0, 5.0, 4.0, 5.0];
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 5.0_f32.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_signed_zeros() {
        // IEEE 754: -0.0 >= +0.0 is true (they compare equal).
        // Our contract: left operand wins on ties.
        // buf = [-0.0, +0.0, -0.0, +0.0]
        // stride=2: [-0.0 vs -0.0 → -0.0, +0.0 vs +0.0 → +0.0]
        // stride=1: [-0.0 vs +0.0 → -0.0 (left wins)]
        let v = vec![-0.0_f32, 0.0, -0.0, 0.0];
        let r = tree_reduce_max_f32(&v);
        assert_eq!(r.to_bits(), (-0.0_f32).to_bits(), "left wins on ties (signed zeros)");

        // Reversed: [+0.0, -0.0, +0.0, -0.0]
        // stride=2: [+0.0 vs +0.0 → +0.0, -0.0 vs -0.0 → -0.0]
        // stride=1: [+0.0 vs -0.0 → +0.0 (left wins)]
        let v2 = vec![0.0_f32, -0.0, 0.0, -0.0];
        let r2 = tree_reduce_max_f32(&v2);
        assert_eq!(r2.to_bits(), 0.0_f32.to_bits(), "left wins on ties (reversed)");
    }

    #[test]
    fn test_tree_reduce_max_neg_inf_preserved() {
        let v = vec![f32::NEG_INFINITY, f32::NEG_INFINITY];
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), f32::NEG_INFINITY.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_very_large() {
        let v = vec![f32::MAX, 1.0, f32::MAX, -1.0];
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), f32::MAX.to_bits());
    }

    #[test]
    fn test_tree_reduce_max_very_small() {
        let v = vec![1e-38_f32, 1e-37, 1e-39, 1e-36];
        assert_eq!(tree_reduce_max_f32(&v).to_bits(), 1e-36_f32.to_bits());
    }

    // ── Cross-check: tree reduce is deterministic ──

    #[test]
    fn test_tree_reduce_deterministic_random_like() {
        // Pseudo-random-ish values from a simple PRNG-like sequence.
        let mut v = Vec::with_capacity(128);
        let mut x = 0.123456789_f32;
        for _ in 0..128 {
            x = (x * 1234.5678 + 0.987654).fract() * 200.0 - 100.0;
            v.push(x);
        }
        let r1 = tree_reduce_sum_f32(&v);
        let r2 = tree_reduce_sum_f32(&v);
        assert_eq!(r1.to_bits(), r2.to_bits(), "sum must be deterministic");

        let m1 = tree_reduce_max_f32(&v);
        let m2 = tree_reduce_max_f32(&v);
        assert_eq!(m1.to_bits(), m2.to_bits(), "max must be deterministic");
    }

    // ── Deterministic attention tests ──────────────────────────────

    #[test]
    fn test_exp_canonical_at_zero() {
        let result = exp_canonical(0.0);
        // exp(0) = 1.0 exactly
        assert!(
            (result - 1.0).abs() < 1e-5,
            "exp_canonical(0) = {}, expected ~1.0",
            result
        );
    }

    #[test]
    fn test_exp_canonical_at_one() {
        let result = exp_canonical(1.0);
        let expected = std::f32::consts::E;
        let rel_err = (result - expected).abs() / expected;
        assert!(
            rel_err < 1e-4,
            "exp_canonical(1) = {}, expected ~{}, rel_err={}",
            result,
            expected,
            rel_err
        );
    }

    #[test]
    fn test_exp_canonical_negative() {
        let result = exp_canonical(-5.0);
        let expected = (-5.0_f64).exp() as f32;
        let rel_err = (result - expected).abs() / expected;
        assert!(
            rel_err < 1e-3,
            "exp_canonical(-5) = {}, expected ~{}, rel_err={}",
            result,
            expected,
            rel_err
        );
    }

    #[test]
    fn test_exp_canonical_large_negative() {
        // Should not produce NaN or negative
        let result = exp_canonical(-100.0);
        assert!(result >= 0.0, "exp_canonical(-100) should be >= 0, got {}", result);
        assert!(!result.is_nan());
    }

    #[test]
    fn test_exp_canonical_deterministic() {
        // Same input must always produce same output (trivial but foundational)
        let a = exp_canonical(1.5);
        let b = exp_canonical(1.5);
        assert_eq!(a.to_bits(), b.to_bits(), "exp_canonical must be deterministic");
    }

    #[test]
    fn test_rintf_matches_cuda_rintf() {
        // rintf must use round-to-nearest-even (banker's rounding),
        // matching CUDA's rintf(). This is NOT Rust's f32::round()
        // which uses round-half-away-from-zero.
        assert_eq!(rintf(0.5), 0.0); // 0.5 → even (0), NOT 1
        assert_eq!(rintf(1.5), 2.0); // 1.5 → even (2)
        assert_eq!(rintf(2.5), 2.0); // 2.5 → even (2), NOT 3
        assert_eq!(rintf(3.5), 4.0); // 3.5 → even (4)
        assert_eq!(rintf(-0.5), 0.0); // -0.5 → even (0), NOT -1
        assert_eq!(rintf(-1.5), -2.0); // -1.5 → even (-2)
        // Non-ties: same as regular round
        assert_eq!(rintf(0.3), 0.0);
        assert_eq!(rintf(0.7), 1.0);
        assert_eq!(rintf(-0.3), 0.0);
        assert_eq!(rintf(-0.7), -1.0);
    }

    #[test]
    fn test_bf16_roundtrip() {
        // bf16 → f32 → bf16 should roundtrip for normal bf16 values
        let original: u16 = 0x3F80; // 1.0 in bf16
        let f = bf16_to_f32(original);
        assert_eq!(f, 1.0_f32);
        let back = f32_to_bf16(f);
        assert_eq!(back, original);
    }

    #[test]
    fn test_bf16_to_f32_zero() {
        assert_eq!(bf16_to_f32(0x0000), 0.0_f32);
    }

    #[test]
    fn test_bf16_to_f32_neg_one() {
        let bits: u16 = 0xBF80; // -1.0 in bf16
        assert_eq!(bf16_to_f32(bits), -1.0_f32);
    }

    fn make_test_cfg(n_q_heads: usize, n_kv_heads: usize, d_head: usize) -> ModelConfig {
        ModelConfig {
            name: "test".into(),
            hidden_dim: n_q_heads * d_head,
            kv_dim: n_kv_heads * d_head,
            d_head,
            n_q_heads,
            n_kv_heads,
            ffn_dim: 1,
            n_layers: 1,
            vocab_size: 1,
            rope_theta: 10000.0,
            rope_scaling: None,
        }
    }

    #[test]
    fn test_deterministic_attention_single_head_single_pos() {
        // 1 query head, 1 KV head, d_head=4, seq_len=1
        // With only 1 position, softmax weight = 1.0, output = V[0]
        let cfg = make_test_cfg(1, 1, 4);

        // Q = [1.0, 0.0, 0.0, 0.0] in bf16
        let q_bf16 = vec![0x3F80, 0x0000, 0x0000, 0x0000];
        // K = [1.0, 0.0, 0.0, 0.0] in bf16
        let k_bf16 = vec![vec![0x3F80, 0x0000, 0x0000, 0x0000]];
        // V = [0.5, -0.5, 1.0, -1.0] in bf16
        let v_bf16 = vec![vec![0x3F00, 0xBF00, 0x3F80, 0xBF80]];

        let result = replay_attention_deterministic(&q_bf16, &k_bf16, &v_bf16, &cfg, canonical_inv_sqrt_d(cfg.d_head));

        // With single position, softmax = 1.0, output = V[0]
        assert_eq!(result.softmax_weights.len(), 1);
        assert!((result.softmax_weights[0] - 1.0).abs() < 1e-5);
        assert!((result.output_f32[0] - 0.5).abs() < 1e-3);
        assert!((result.output_f32[1] - (-0.5)).abs() < 1e-3);
        assert!((result.output_f32[2] - 1.0).abs() < 1e-3);
        assert!((result.output_f32[3] - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_deterministic_attention_two_positions() {
        // 1 head, d_head=2, seq_len=2
        // Q = [1.0, 0.0], K0 = [1.0, 0.0], K1 = [0.0, 1.0]
        // Score0 = 1.0 / sqrt(2), Score1 = 0.0 / sqrt(2)
        // Softmax: exp(1/√2) / (exp(1/√2) + exp(0)) vs exp(0) / (...)
        let cfg = make_test_cfg(1, 1, 2);

        let q = vec![0x3F80, 0x0000]; // [1.0, 0.0]
        let k = vec![
            vec![0x3F80, 0x0000], // [1.0, 0.0]
            vec![0x0000, 0x3F80], // [0.0, 1.0]
        ];
        let v = vec![
            vec![0x3F80, 0x0000], // [1.0, 0.0]
            vec![0x0000, 0x3F80], // [0.0, 1.0]
        ];

        let result = replay_attention_deterministic(&q, &k, &v, &cfg, canonical_inv_sqrt_d(cfg.d_head));

        // Scores: [1.0/sqrt(2), 0.0/sqrt(2)] = [0.7071, 0.0]
        // max = 0.7071
        // exp(0) = 1.0, exp(-0.7071) ≈ 0.4932
        // softmax ≈ [1.0/1.4932, 0.4932/1.4932] ≈ [0.6697, 0.3303]
        assert!(result.softmax_weights[0] > result.softmax_weights[1]);
        let w0 = result.softmax_weights[0];
        let w1 = result.softmax_weights[1];
        assert!((w0 + w1 - 1.0).abs() < 1e-5, "weights should sum to 1");

        // Output = w0 * V0 + w1 * V1 = [w0, w1]
        assert!((result.output_f32[0] - w0).abs() < 1e-5);
        assert!((result.output_f32[1] - w1).abs() < 1e-5);
    }

    #[test]
    fn test_deterministic_attention_gqa() {
        // 2 query heads, 1 KV head (GQA ratio = 2), d_head=2, seq_len=1
        let cfg = make_test_cfg(2, 1, 2);

        // Q has 2 heads: head0=[1,0], head1=[0,1]
        let q = vec![0x3F80, 0x0000, 0x0000, 0x3F80];
        // K: 1 KV head, shape [1][1*2]
        let k = vec![vec![0x3F80, 0x3F80]]; // [1.0, 1.0]
        let v = vec![vec![0x3F00, 0xBF00]]; // [0.5, -0.5]

        let result = replay_attention_deterministic(&q, &k, &v, &cfg, canonical_inv_sqrt_d(cfg.d_head));

        // Both heads use the same KV group (group 0)
        // Single position → softmax = 1.0 for both heads → output = V
        assert_eq!(result.output_f32.len(), 4); // 2 heads * d_head=2
        // Head 0: output = V[0] = [0.5, -0.5]
        assert!((result.output_f32[0] - 0.5).abs() < 1e-3);
        assert!((result.output_f32[1] - (-0.5)).abs() < 1e-3);
        // Head 1: same V (same KV group)
        assert!((result.output_f32[2] - 0.5).abs() < 1e-3);
        assert!((result.output_f32[3] - (-0.5)).abs() < 1e-3);
    }

    #[test]
    fn test_deterministic_attention_self_consistent() {
        // Run twice with same inputs, must produce identical bits
        let cfg = make_test_cfg(2, 1, 4);
        let q = vec![0x3F80, 0x3F00, 0xBF00, 0x3E80, 0x3F00, 0x3F80, 0x3E80, 0xBF00];
        let k = vec![
            vec![0x3F80, 0x3F00, 0xBF00, 0x3E80],
            vec![0x3E80, 0xBF00, 0x3F80, 0x3F00],
            vec![0xBF00, 0x3E80, 0x3F00, 0xBF80],
        ];
        let v = vec![
            vec![0x3F80, 0x0000, 0xBF80, 0x3F00],
            vec![0x0000, 0x3F80, 0x3F00, 0xBF00],
            vec![0xBF00, 0xBF80, 0x0000, 0x3F80],
        ];

        let r1 = replay_attention_deterministic(&q, &k, &v, &cfg, canonical_inv_sqrt_d(cfg.d_head));
        let r2 = replay_attention_deterministic(&q, &k, &v, &cfg, canonical_inv_sqrt_d(cfg.d_head));

        // Bit-exact comparison
        for (i, (a, b)) in r1.output_f32.iter().zip(r2.output_f32.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "output[{}] not bit-exact: {} vs {}",
                i,
                a,
                b
            );
        }
        for (i, (a, b)) in r1
            .softmax_weights
            .iter()
            .zip(r2.softmax_weights.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "weight[{}] not bit-exact: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_deterministic_f32_matches_bf16_on_exact_bf16_values() {
        // When f32 inputs are exact bf16 values, both paths should agree
        let cfg = make_test_cfg(1, 1, 2);

        let q_bf16 = vec![0x3F80, 0x3F00]; // [1.0, 0.5]
        let k_bf16 = vec![vec![0x3F00, 0x3F80]]; // [0.5, 1.0]
        let v_bf16 = vec![vec![0x3F80, 0xBF80]]; // [1.0, -1.0]

        let q_f32: Vec<f32> = q_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        let k_f32: Vec<Vec<f32>> = k_bf16
            .iter()
            .map(|row| row.iter().map(|&b| bf16_to_f32(b)).collect())
            .collect();
        let v_f32: Vec<Vec<f32>> = v_bf16
            .iter()
            .map(|row| row.iter().map(|&b| bf16_to_f32(b)).collect())
            .collect();

        let r_bf16 = replay_attention_deterministic(&q_bf16, &k_bf16, &v_bf16, &cfg, canonical_inv_sqrt_d(cfg.d_head));
        let r_f32 = replay_attention_deterministic_f32(&q_f32, &k_f32, &v_f32, &cfg, canonical_inv_sqrt_d(cfg.d_head));

        for (i, (a, b)) in r_bf16
            .output_f32
            .iter()
            .zip(r_f32.output_f32.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "output[{}] bf16 path {} != f32 path {}",
                i,
                a,
                b
            );
        }
    }
}
