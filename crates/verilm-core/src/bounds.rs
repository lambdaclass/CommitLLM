//! Formal worst-case bounds for the single-step attention corridor.
//!
//! # Problem Statement
//!
//! The GPU computes attention in FP16/BF16 (with FP32 accumulators in Flash
//! Attention). The verifier replays from the same INT32 QKV accumulators in
//! f64. Both quantize the output to INT8 with `scale_a`. The corridor is
//! `|a_i8_gpu - a_i8_verifier|`.
//!
//! Roadmap #4 theorem target: derive a sufficient condition for
//! `max_abs_diff ≤ 1` in i8 space.
//!
//! # Key Result
//!
//! **From QKV accumulators alone, the formal worst-case bound is ~25 INT8
//! steps.** Committed intermediates are necessary to bring it to ≤ 1.
//!
//! The bound achieves ≤ 1 when:
//! - Q, K, V are all committed (only FP32 accumulation error remains), or
//! - Pre-softmax scores are committed (eliminates softmax amplification), or
//! - `attn_out_i8` is committed (trivially zero corridor for this step).
//!
//! # Derivation
//!
//! ## Notation
//!
//! - `d` = d_head (per-head dimension, typically 128)
//! - `u` = unit roundoff of the GPU arithmetic format
//!   - FP16: u = 2^-11 ≈ 4.88e-4
//!   - BF16: u = 2^-8 ≈ 3.91e-3
//!   - FP32: u = 2^-24 ≈ 5.96e-8
//! - `ε_qk` = total relative error per Q/K element after dequant + RoPE
//! - `ε_v` = total relative error per V element after dequant (no RoPE)
//! - `B_v` = max |v[j,m]| over all sequence positions j, dimensions m
//! - `S_max` = max |score_j| (pre-softmax score magnitude)
//! - `scale_a` = dynamic INT8 quantization scale for attention output
//! - `n` = sequence length (number of KV cache entries)
//!
//! ## Step 1: Input Perturbation (Dequant + RoPE)
//!
//! GPU dequant (CUTLASS epilogue, FP32 arithmetic, FP16 output):
//!   `q_fp16 = fp16(f32(acc_i32) * sw_f32 * sx_f32)`
//!
//! Single FP16 cast: relative error ≤ u_fp16. RoPE (2 muls, 1 add/sub,
//! trig lookup) adds ≈ 4 · u_fp16 per element. Total:
//!
//!   `ε_qk = C_rope · u_fp16` where `C_rope ≈ 5`
//!   `ε_v  = u_fp16` (V has no RoPE)
//!
//! Per-element absolute error: `|δq_i| ≤ ε_qk · |q_i|`,
//! `|δk_{j,i}| ≤ ε_qk · |k_{j,i}|`, `|δv_{j,i}| ≤ ε_v · |v_{j,i}|`.
//!
//! ## Step 2: Score Error
//!
//! score_j = (q · k_j) / √d.  GPU: FP16 inputs, FP32 dot-product
//! accumulator (Flash Attention). Verifier: f64.
//!
//! Error bound (bilinear form, with δk=0 if K committed):
//!
//! When both Q and K have error (no committed K):
//!   `|Δs_j| ≤ 2 · ε_qk · ||q||₂ · ||k_j||₂ / √d`
//!          `= 2 · ε_qk · S_max_j`
//!
//! When only Q has error (K committed):
//!   `|Δs_j| ≤ ε_qk · ||q||₂ · ||k_j||₂ / √d = ε_qk · S_max_j`
//!
//! FP32 accumulation adds d · u_f32 (negligible: 128 · 6e-8 ≈ 8e-6).
//!
//! ## Step 3: Softmax Perturbation
//!
//! Jacobian operator norms:
//!   - `||Δα||_∞ ≤ ½ · ||Δs||_∞`  (∞→∞ norm ≤ 1/2)
//!   - `||Δα||₁  ≤ 2 · ||Δs||_∞`  (∞→1 norm ≤ 2)
//!
//! The ∞→1 bound `||Δα||₁ ≤ 2 · ||Δs||_∞` is the bottleneck: it
//! assumes all n softmax probabilities shift by the maximum amount.
//! In practice, most Δα_j ≪ max, making this very loose.
//!
//! FP32 softmax error: ε_softmax ≈ O(u_f32), negligible.
//!
//! ## Step 4: Weighted Sum (Attention Output)
//!
//! o_m = Σ_j α_j · v[j,m].  Error:
//!
//!   `|Δo_m| ≤ B_v · ||Δα||₁ + ε_v · B_v`
//!
//! The first term (softmax shift × V magnitude) dominates.
//!
//! ## Step 5: Requantization
//!
//! `a_i8 = round(o / scale_a)`.  Two values within distance `scale_a`
//! can differ by at most 1 after rounding:
//!
//!   `|Δo_m| < scale_a  ⟹  |Δa_i8| ≤ 1`
//!
//! ## Full Chain (No Committed Intermediates)
//!
//! Composing steps 1-5:
//!
//!   `|Δo_m| ≤ B_v · (4 · ε_qk · S_max + ε_v)`
//!
//!   `|Δa_i8| ≤ floor(|Δo_m| / scale_a) + 1`
//!
//! Since attention output is a convex combination of V values,
//! `max|o_m| ≤ B_v`, so `scale_a ≈ B_v / 127`. The condition for
//! `|Δa_i8| ≤ 1` becomes:
//!
//!   **127 · (4 · ε_qk · S_max + ε_v) < 1**
//!
//! ## Sources of Looseness
//!
//! 1. **Softmax ∞→1**: Assumes all `n` weights shift maximally and
//!    contribute constructively. Real perturbations are correlated
//!    (proportional to score magnitude) and partially cancel.
//!
//! 2. **B_v / scale_a ≈ 127**: Assumes attention output saturates
//!    the full INT8 range by concentrating on a single position with
//!    extreme V value. Realistic attention spreads across positions.
//!
//! 3. **No cancellation**: All per-dimension errors assumed to add.
//!    Real errors have mixed signs.
//!
//! 4. **S_max is worst-case**: Most pre-softmax scores are much
//!    smaller than the maximum. Only a few positions are highly attended.

use crate::constants::ModelConfig;

/// Arithmetic precision of the GPU computation path.
#[derive(Debug, Clone, Copy)]
pub enum GpuArithmetic {
    /// FP16 (IEEE 754 half): u = 2^-11.
    Fp16,
    /// BF16 (bfloat16): u = 2^-8.
    Bf16,
}

impl GpuArithmetic {
    /// Unit roundoff (half the machine epsilon).
    pub fn unit_roundoff(self) -> f64 {
        match self {
            GpuArithmetic::Fp16 => 2.0_f64.powi(-11), // ≈ 4.88e-4
            GpuArithmetic::Bf16 => 2.0_f64.powi(-8),  // ≈ 3.91e-3
        }
    }
}

/// What the GPU commits as trusted intermediates.
#[derive(Debug, Clone, Copy)]
pub enum CommittedIntermediates {
    /// Only INT32 QKV accumulators (shell-verified). Q, K, V all have
    /// dequant+RoPE error.
    QkvAccumulatorsOnly,
    /// K and V are committed (from KV cache). Only Q has dequant+RoPE error.
    CommittedKV,
    /// Q, K, V all committed. Only FP32 accumulation error remains.
    CommittedQKV,
    /// Pre-softmax scores committed. Eliminates softmax amplification;
    /// only V-dequant error remains.
    CommittedScores,
}

/// Parameters for the worst-case attention corridor bound.
#[derive(Debug, Clone)]
pub struct CorridorBoundParams {
    /// Per-head dimension (e.g. 128).
    pub d_head: usize,
    /// GPU arithmetic format (FP16 or BF16).
    pub gpu_arithmetic: GpuArithmetic,
    /// Number of FP operations in dequant+RoPE per element (typically 5:
    /// 1 FP16 cast from dequant + 4 for RoPE multiply/trig/add).
    pub c_rope: f64,
    /// Maximum pre-softmax score magnitude. Typical: 10-30 for real
    /// models. Conservative worst-case: 30.
    pub s_max: f64,
    /// Ratio B_v / scale_a: how many INT8 steps span the V value range.
    /// Worst case: 127 (output saturates INT8 range). Typical: 50-127.
    pub bv_over_scale_a: f64,
    /// Sequence length (for FP32 accumulation error term, usually negligible).
    pub seq_len: usize,
}

/// Result of the worst-case corridor bound computation.
#[derive(Debug, Clone)]
pub struct CorridorBound {
    /// Total relative error per Q/K element after dequant+RoPE.
    pub eps_qk: f64,
    /// Total relative error per V element after dequant.
    pub eps_v: f64,
    /// Worst-case float-level output error |Δo_m| / scale_a.
    pub delta_o_over_scale_a: f64,
    /// Worst-case INT8 corridor: max |Δa_i8|.
    pub max_abs_diff_i8: u32,
    /// Whether the bound achieves ≤ 1.
    pub achieves_leq_1: bool,
    /// Breakdown of the error terms (for analysis).
    pub softmax_term: f64,
    pub v_dequant_term: f64,
    pub fp32_accum_term: f64,
}

/// Compute the formal worst-case attention corridor bound.
///
/// Returns the worst-case `|Δa_i8|` under the given assumptions about
/// GPU arithmetic, committed intermediates, and model parameters.
pub fn compute_corridor_bound(
    params: &CorridorBoundParams,
    committed: CommittedIntermediates,
) -> CorridorBound {
    let u = params.gpu_arithmetic.unit_roundoff();
    let u_f32: f64 = 2.0_f64.powi(-24);

    let (eps_qk, eps_v, softmax_term, v_dequant_term, fp32_accum_term) = match committed {
        CommittedIntermediates::QkvAccumulatorsOnly => {
            // Both Q and K have dequant+RoPE error; V has dequant error.
            let eps_qk = params.c_rope * u;
            let eps_v = u;
            // Score error: |Δs_j| ≤ 2·ε_qk·S_max (both Q,K perturbed)
            // Softmax: ||Δα||₁ ≤ 2·||Δs||_∞
            // → softmax contribution to |Δo|/scale_a:
            //   (B_v/scale_a) · 4·ε_qk·S_max
            let softmax = params.bv_over_scale_a * 4.0 * eps_qk * params.s_max;
            let v_deq = params.bv_over_scale_a * eps_v;
            let fp32 = params.bv_over_scale_a * params.seq_len as f64 * u_f32;
            (eps_qk, eps_v, softmax, v_deq, fp32)
        }
        CommittedIntermediates::CommittedKV => {
            // K,V committed (exact). Only Q has dequant+RoPE error.
            let eps_qk = params.c_rope * u;
            let eps_v = 0.0; // V is committed
                             // Score error: |Δs_j| ≤ ε_qk·S_max (only Q perturbed)
            let softmax = params.bv_over_scale_a * 2.0 * eps_qk * params.s_max;
            let v_deq = 0.0;
            let fp32 = params.bv_over_scale_a * params.seq_len as f64 * u_f32;
            (eps_qk, eps_v, softmax, v_deq, fp32)
        }
        CommittedIntermediates::CommittedQKV => {
            // Q, K, V all committed. Only FP32 accumulation error.
            let eps_qk = 0.0;
            let eps_v = 0.0;
            // No score error from inputs. FP32 softmax + accumulation:
            // ||Δα||₁ ≤ 2·n·u_f32 (FP32 softmax error)
            // |Δo_m| ≤ B_v · 2·n·u_f32 + n·u_f32·B_v
            let softmax = params.bv_over_scale_a * 2.0 * params.seq_len as f64 * u_f32;
            let v_deq = 0.0;
            let fp32 = params.bv_over_scale_a * params.seq_len as f64 * u_f32;
            (eps_qk, eps_v, softmax, v_deq, fp32)
        }
        CommittedIntermediates::CommittedScores => {
            // Scores committed → α computed deterministically from committed scores.
            // Only V dequant error remains.
            let eps_qk = 0.0;
            let eps_v = u;
            let softmax = 0.0;
            let v_deq = params.bv_over_scale_a * eps_v;
            let fp32 = params.bv_over_scale_a * params.seq_len as f64 * u_f32;
            (eps_qk, eps_v, softmax, v_deq, fp32)
        }
    };

    let delta_o_over_scale_a = softmax_term + v_dequant_term + fp32_accum_term;
    let max_abs_diff_i8 = if delta_o_over_scale_a < 1.0 {
        1
    } else {
        delta_o_over_scale_a.floor() as u32 + 1
    };

    CorridorBound {
        eps_qk,
        eps_v,
        delta_o_over_scale_a,
        max_abs_diff_i8,
        achieves_leq_1: delta_o_over_scale_a < 1.0,
        softmax_term,
        v_dequant_term,
        fp32_accum_term,
    }
}

/// Compute corridor bounds for a specific model config.
///
/// Uses typical worst-case assumptions: S_max=20, B_v/scale_a=127.
pub fn corridor_bound_for_model(
    _cfg: &ModelConfig,
    gpu: GpuArithmetic,
    committed: CommittedIntermediates,
    s_max: f64,
    seq_len: usize,
) -> CorridorBound {
    let params = CorridorBoundParams {
        d_head: _cfg.d_head,
        gpu_arithmetic: gpu,
        c_rope: 5.0,
        s_max,
        bv_over_scale_a: 127.0,
        seq_len,
    };
    compute_corridor_bound(&params, committed)
}

/// Find the maximum S_max that still achieves |Δa_i8| ≤ 1
/// for a given commitment level and arithmetic.
pub fn max_s_for_leq_1(
    gpu: GpuArithmetic,
    committed: CommittedIntermediates,
    bv_over_scale_a: f64,
) -> f64 {
    let u = gpu.unit_roundoff();
    let eps_qk = 5.0 * u; // C_rope = 5

    match committed {
        CommittedIntermediates::QkvAccumulatorsOnly => {
            // Need: bv_over_scale_a * (4·ε_qk·S + ε_v) < 1
            let eps_v = u;
            let budget = 1.0 / bv_over_scale_a - eps_v;
            if budget <= 0.0 {
                return 0.0;
            }
            budget / (4.0 * eps_qk)
        }
        CommittedIntermediates::CommittedKV => {
            // Need: bv_over_scale_a * 2·ε_qk·S < 1
            let budget = 1.0 / bv_over_scale_a;
            budget / (2.0 * eps_qk)
        }
        CommittedIntermediates::CommittedQKV | CommittedIntermediates::CommittedScores => {
            // No dependence on S_max
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the headline result: QKV-only replay gives ~25 for Qwen/Llama-class models.
    #[test]
    fn test_qkv_only_fp16_s20() {
        let params = CorridorBoundParams {
            d_head: 128,
            gpu_arithmetic: GpuArithmetic::Fp16,
            c_rope: 5.0,
            s_max: 20.0,
            bv_over_scale_a: 127.0,
            seq_len: 1024,
        };
        let b = compute_corridor_bound(&params, CommittedIntermediates::QkvAccumulatorsOnly);

        // ε_qk = 5 · 2^-11 ≈ 0.00244
        assert!((b.eps_qk - 5.0 * 2.0_f64.powi(-11)).abs() < 1e-10);

        // softmax_term = 127 · 4 · 0.00244 · 20 ≈ 24.8
        assert!(
            b.softmax_term > 24.0 && b.softmax_term < 26.0,
            "softmax_term={}",
            b.softmax_term
        );

        // Total ≈ 24.8 + 0.06 + negligible ≈ 24.9
        assert!(
            b.delta_o_over_scale_a > 24.0,
            "expected ~25, got {}",
            b.delta_o_over_scale_a
        );
        assert!(!b.achieves_leq_1);
        assert!(b.max_abs_diff_i8 >= 25);
    }

    /// Committed K,V with S_max=20: bound ≈ 2.5, still > 1.
    #[test]
    fn test_committed_kv_fp16_s20() {
        let params = CorridorBoundParams {
            d_head: 128,
            gpu_arithmetic: GpuArithmetic::Fp16,
            c_rope: 5.0,
            s_max: 20.0,
            bv_over_scale_a: 127.0,
            seq_len: 1024,
        };
        let b = compute_corridor_bound(&params, CommittedIntermediates::CommittedKV);

        // softmax_term = 127 · 2 · 0.00244 · 20 ≈ 12.4
        assert!(
            b.softmax_term > 12.0 && b.softmax_term < 13.0,
            "softmax_term={}",
            b.softmax_term
        );
        assert!(!b.achieves_leq_1);
    }

    /// Committed K,V with S_max=8: bound ≈ 1.0, marginal.
    #[test]
    fn test_committed_kv_fp16_s8() {
        let params = CorridorBoundParams {
            d_head: 128,
            gpu_arithmetic: GpuArithmetic::Fp16,
            c_rope: 5.0,
            s_max: 8.0,
            bv_over_scale_a: 127.0,
            seq_len: 1024,
        };
        let b = compute_corridor_bound(&params, CommittedIntermediates::CommittedKV);

        // 127 · 2 · 0.00244 · 8 ≈ 4.96
        assert!(
            b.softmax_term > 4.5 && b.softmax_term < 5.5,
            "softmax_term={}",
            b.softmax_term
        );
        // Still > 1
        assert!(!b.achieves_leq_1);
    }

    /// Committed Q,K,V: only FP32 accumulation error. Achieves ≤ 1.
    #[test]
    fn test_committed_qkv_fp16() {
        let params = CorridorBoundParams {
            d_head: 128,
            gpu_arithmetic: GpuArithmetic::Fp16,
            c_rope: 5.0,
            s_max: 20.0,
            bv_over_scale_a: 127.0,
            seq_len: 4096,
        };
        let b = compute_corridor_bound(&params, CommittedIntermediates::CommittedQKV);

        // 127 · 3 · 4096 · 2^-24 ≈ 0.093
        assert!(
            b.delta_o_over_scale_a < 0.1,
            "expected < 0.1, got {}",
            b.delta_o_over_scale_a
        );
        assert!(b.achieves_leq_1);
    }

    /// Committed scores: only V-dequant error. Achieves ≤ 1.
    #[test]
    fn test_committed_scores_fp16() {
        let params = CorridorBoundParams {
            d_head: 128,
            gpu_arithmetic: GpuArithmetic::Fp16,
            c_rope: 5.0,
            s_max: 20.0,
            bv_over_scale_a: 127.0,
            seq_len: 4096,
        };
        let b = compute_corridor_bound(&params, CommittedIntermediates::CommittedScores);

        // v_dequant = 127 · 2^-11 ≈ 0.062
        assert!(
            b.v_dequant_term > 0.06 && b.v_dequant_term < 0.07,
            "v_dequant_term={}",
            b.v_dequant_term
        );
        assert!(b.delta_o_over_scale_a < 0.1);
        assert!(b.achieves_leq_1);
    }

    /// BF16 makes everything worse: QKV-only bound ≈ 200.
    #[test]
    fn test_qkv_only_bf16() {
        let params = CorridorBoundParams {
            d_head: 128,
            gpu_arithmetic: GpuArithmetic::Bf16,
            c_rope: 5.0,
            s_max: 20.0,
            bv_over_scale_a: 127.0,
            seq_len: 1024,
        };
        let b = compute_corridor_bound(&params, CommittedIntermediates::QkvAccumulatorsOnly);

        // ε_qk = 5 · 2^-8 ≈ 0.0195
        // softmax_term = 127 · 4 · 0.0195 · 20 ≈ 199
        assert!(b.softmax_term > 190.0, "softmax_term={}", b.softmax_term);
        assert!(b.max_abs_diff_i8 >= 195);
    }

    /// max_s_for_leq_1: what S_max stays within ≤ 1 for each commitment level.
    #[test]
    fn test_max_s_thresholds() {
        // QKV-only, FP16: S_max must be < 0.76
        let s = max_s_for_leq_1(
            GpuArithmetic::Fp16,
            CommittedIntermediates::QkvAccumulatorsOnly,
            127.0,
        );
        assert!(s > 0.7 && s < 0.8, "QKV-only max_s={}", s);

        // Committed KV, FP16: S_max must be < 1.61
        let s = max_s_for_leq_1(
            GpuArithmetic::Fp16,
            CommittedIntermediates::CommittedKV,
            127.0,
        );
        assert!(s > 1.5 && s < 1.7, "committed-KV max_s={}", s);

        // Committed QKV: no dependence on S
        let s = max_s_for_leq_1(
            GpuArithmetic::Fp16,
            CommittedIntermediates::CommittedQKV,
            127.0,
        );
        assert!(s.is_infinite());
    }

    /// Verify with Llama-8B config (structurally identical bound).
    #[test]
    fn test_llama_8b_qkv_only() {
        let cfg = ModelConfig::llama_8b();
        let b = corridor_bound_for_model(
            &cfg,
            GpuArithmetic::Fp16,
            CommittedIntermediates::QkvAccumulatorsOnly,
            20.0,
            1024,
        );
        // Same d_head=128, same bound
        assert!(b.max_abs_diff_i8 >= 25);
        assert!(!b.achieves_leq_1);
    }

    /// Committed scores on Llama-8B: achieves ≤ 1.
    #[test]
    fn test_llama_8b_committed_scores() {
        let cfg = ModelConfig::llama_8b();
        let b = corridor_bound_for_model(
            &cfg,
            GpuArithmetic::Fp16,
            CommittedIntermediates::CommittedScores,
            20.0,
            4096,
        );
        assert!(b.achieves_leq_1);
        assert!(b.delta_o_over_scale_a < 0.1);
    }
}
