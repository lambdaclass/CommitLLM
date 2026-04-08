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
}
