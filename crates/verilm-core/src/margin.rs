//! Margin certificates for Level B token-generation integrity.
//!
//! A margin certificate proves that the selected token is stable under
//! attention perturbations bounded by epsilon. If the gap between the
//! top-1 and top-2 logits exceeds twice the perturbation bound, no
//! floating-point nondeterminism in attention/softmax/layernorm can
//! flip the argmax.
//!
//! # Perturbation bound
//!
//! ```text
//! B(epsilon, n_layers, max_v_norm, wo_norm)
//!   = n_layers * epsilon * max_v_norm * wo_norm
//! ```
//!
//! where:
//! - `epsilon`: per-entry softmax tolerance across hardware implementations
//! - `n_layers`: number of transformer layers (perturbation accumulates)
//! - `max_v_norm`: max L-infinity norm of V head vectors across all attention heads
//! - `wo_norm`: L-infinity operator norm of W_o (= max absolute row sum)
//!
//! The certificate is valid when: `delta > 2 * B`
//!
//! # Norms
//!
//! All norms in this module are the **L-infinity (max absolute row sum)** norm:
//!
//! ```text
//! ||W||_inf = max_i Σ_j |W[i,j]|
//! ```
//!
//! For INT8 weights with per-tensor scale `s`, the scaled norm is:
//!
//! ```text
//! ||W||_inf = s * max_i Σ_j |W_i8[i,j]|
//! ```

use serde::{Deserialize, Serialize};

/// Default per-entry attention softmax tolerance.
/// Conservative starting point; tighten based on empirical hardware measurements.
pub const DEFAULT_ATTENTION_EPSILON: f32 = 1e-3;

/// A margin certificate for one generated token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginCertificate {
    /// Index of this token in the sequence.
    pub token_index: u32,

    /// The full logit vector over the vocabulary (f32).
    pub logits: Vec<f32>,

    /// Token ID selected (argmax for greedy decoding).
    pub selected_token_id: u32,

    /// Top-1 logit value.
    pub top1_logit: f32,
    /// Token ID of the top-1.
    pub top1_token_id: u32,

    /// Runner-up logit value.
    pub top2_logit: f32,
    /// Token ID of the runner-up.
    pub top2_token_id: u32,

    /// Observed margin: top1_logit - top2_logit.
    pub delta: f32,
}

/// Result of verifying a margin certificate.
#[derive(Debug, Clone)]
pub struct MarginVerifyResult {
    /// Whether the certificate proves argmax stability.
    pub certified: bool,
    /// The computed perturbation bound B.
    pub perturbation_bound: f32,
    /// The observed margin delta.
    pub delta: f32,
    /// Structural failures found during verification (empty = structurally valid).
    pub failures: Vec<String>,
}

/// Extract top-1 and top-2 logit values and their token IDs.
///
/// Returns `(top1_logit, top1_id, top2_logit, top2_id)`.
/// Returns `None` if the logit vector has fewer than 2 elements.
pub fn extract_top2(logits: &[f32]) -> Option<(f32, u32, f32, u32)> {
    if logits.len() < 2 {
        return None;
    }

    let mut top1_id: u32 = 0;
    let mut top1_val = logits[0];
    let mut top2_id: u32 = 1;
    let mut top2_val = logits[1];

    if top2_val > top1_val {
        std::mem::swap(&mut top1_val, &mut top2_val);
        std::mem::swap(&mut top1_id, &mut top2_id);
    }

    for (i, &val) in logits.iter().enumerate().skip(2) {
        if val > top1_val {
            top2_val = top1_val;
            top2_id = top1_id;
            top1_val = val;
            top1_id = i as u32;
        } else if val > top2_val {
            top2_val = val;
            top2_id = i as u32;
        }
    }

    Some((top1_val, top1_id, top2_val, top2_id))
}

/// Compute the perturbation bound B.
///
/// ```text
/// B = n_layers * epsilon * max_v_norm * wo_norm
/// ```
///
/// A token's argmax is certified stable when `delta > 2 * B`.
pub fn compute_perturbation_bound(
    epsilon: f32,
    n_layers: usize,
    max_v_norm: f32,
    wo_norm: f32,
) -> f32 {
    n_layers as f32 * epsilon * max_v_norm * wo_norm
}

/// Check a margin certificate's structural integrity.
///
/// Pure verification function — checks that:
/// 1. top1/top2 extraction matches the logit vector
/// 2. delta is correctly computed
/// 3. selected_token_id matches top1
///
/// Does NOT check certification (caller computes the bound separately).
/// Returns a list of failure descriptions (empty = structurally valid).
pub fn check_margin_certificate(cert: &MarginCertificate) -> Vec<String> {
    let mut failures = Vec::new();

    if cert.logits.len() < 2 {
        failures.push("logit vector has fewer than 2 entries".into());
        return failures;
    }

    let (expected_top1, expected_top1_id, expected_top2, expected_top2_id) =
        extract_top2(&cert.logits).unwrap();

    if (cert.top1_logit - expected_top1).abs() > f32::EPSILON {
        failures.push(format!(
            "top1_logit mismatch: claimed {}, actual {}",
            cert.top1_logit, expected_top1
        ));
    }
    if cert.top1_token_id != expected_top1_id {
        failures.push(format!(
            "top1_token_id mismatch: claimed {}, actual {}",
            cert.top1_token_id, expected_top1_id
        ));
    }
    if (cert.top2_logit - expected_top2).abs() > f32::EPSILON {
        failures.push(format!(
            "top2_logit mismatch: claimed {}, actual {}",
            cert.top2_logit, expected_top2
        ));
    }
    if cert.top2_token_id != expected_top2_id {
        failures.push(format!(
            "top2_token_id mismatch: claimed {}, actual {}",
            cert.top2_token_id, expected_top2_id
        ));
    }

    let expected_delta = expected_top1 - expected_top2;
    if (cert.delta - expected_delta).abs() > f32::EPSILON * 10.0 {
        failures.push(format!(
            "delta mismatch: claimed {}, actual {}",
            cert.delta, expected_delta
        ));
    }

    if cert.selected_token_id != expected_top1_id {
        failures.push(format!(
            "selected_token_id {} != top1_token_id {}",
            cert.selected_token_id, expected_top1_id
        ));
    }

    failures
}

/// Verify a margin certificate: structural check + certification decision.
///
/// Combines `check_margin_certificate` (structural) with the perturbation
/// bound computation to produce a full verification result.
pub fn verify_margin_certificate(
    cert: &MarginCertificate,
    epsilon: f32,
    n_layers: usize,
    max_v_norm: f32,
    wo_norm: f32,
) -> MarginVerifyResult {
    let failures = check_margin_certificate(cert);

    let bound = compute_perturbation_bound(epsilon, n_layers, max_v_norm, wo_norm);

    let delta = if cert.logits.len() >= 2 {
        let (top1, _, top2, _) = extract_top2(&cert.logits).unwrap();
        top1 - top2
    } else {
        cert.delta
    };

    let certified = failures.is_empty() && delta > 2.0 * bound;

    MarginVerifyResult {
        certified,
        perturbation_bound: bound,
        delta,
        failures,
    }
}

/// Build a margin certificate from a logit vector.
///
/// For greedy decoding, `selected_token_id = argmax(logits)`.
pub fn build_margin_certificate(token_index: u32, logits: Vec<f32>) -> Option<MarginCertificate> {
    let (top1_logit, top1_id, top2_logit, top2_id) = extract_top2(&logits)?;
    let delta = top1_logit - top2_logit;

    Some(MarginCertificate {
        token_index,
        logits,
        selected_token_id: top1_id,
        top1_logit,
        top1_token_id: top1_id,
        top2_logit,
        top2_token_id: top2_id,
        delta,
    })
}

/// Compute the L-infinity operator norm of a weight matrix.
///
/// ```text
/// ||W||_inf = scale * max_i Σ_j |W_i8[i,j]|
/// ```
///
/// This is the max absolute row sum, scaled by the per-tensor quantization
/// scale. For the toy model (unit scale), pass `scale = 1.0`.
pub fn compute_matrix_linf_norm(weights: &[i8], rows: usize, cols: usize, scale: f32) -> f32 {
    assert_eq!(weights.len(), rows * cols);
    let mut max_row_sum: f32 = 0.0;
    for row in 0..rows {
        let row_sum: f32 = (0..cols)
            .map(|col| (weights[row * cols + col] as f32).abs())
            .sum();
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }
    max_row_sum * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_top2_basic() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let (top1, id1, top2, id2) = extract_top2(&logits).unwrap();
        assert_eq!(top1, 5.0);
        assert_eq!(id1, 1);
        assert_eq!(top2, 4.0);
        assert_eq!(id2, 4);
    }

    #[test]
    fn test_extract_top2_ties() {
        let logits = vec![3.0, 3.0, 1.0];
        let (top1, id1, top2, id2) = extract_top2(&logits).unwrap();
        assert_eq!(top1, 3.0);
        assert_eq!(id1, 0);
        assert_eq!(top2, 3.0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_extract_top2_negative_logits() {
        let logits = vec![-5.0, -1.0, -3.0, -2.0];
        let (top1, id1, top2, id2) = extract_top2(&logits).unwrap();
        assert_eq!(top1, -1.0);
        assert_eq!(id1, 1);
        assert_eq!(top2, -2.0);
        assert_eq!(id2, 3);
    }

    #[test]
    fn test_extract_top2_two_elements() {
        let logits = vec![10.0, 20.0];
        let (top1, id1, top2, id2) = extract_top2(&logits).unwrap();
        assert_eq!(top1, 20.0);
        assert_eq!(id1, 1);
        assert_eq!(top2, 10.0);
        assert_eq!(id2, 0);
    }

    #[test]
    fn test_extract_top2_too_small() {
        assert!(extract_top2(&[1.0]).is_none());
        assert!(extract_top2(&[]).is_none());
    }

    #[test]
    fn test_perturbation_bound_grows_with_layers() {
        let b1 = compute_perturbation_bound(1e-3, 10, 100.0, 50.0);
        let b2 = compute_perturbation_bound(1e-3, 20, 100.0, 50.0);
        assert!(b2 > b1);
    }

    #[test]
    fn test_perturbation_bound_grows_with_epsilon() {
        let b1 = compute_perturbation_bound(1e-4, 10, 100.0, 50.0);
        let b2 = compute_perturbation_bound(1e-3, 10, 100.0, 50.0);
        assert!(b2 > b1);
    }

    #[test]
    fn test_perturbation_bound_zero_epsilon() {
        assert_eq!(compute_perturbation_bound(0.0, 10, 100.0, 50.0), 0.0);
    }

    #[test]
    fn test_check_certificate_valid() {
        let logits = vec![10.0, 5.0, 3.0];
        let cert = build_margin_certificate(0, logits).unwrap();
        assert!(check_margin_certificate(&cert).is_empty());
    }

    #[test]
    fn test_check_certificate_wrong_top1() {
        let logits = vec![10.0, 5.0, 3.0];
        let mut cert = build_margin_certificate(0, logits).unwrap();
        cert.top1_token_id = 2;
        assert!(!check_margin_certificate(&cert).is_empty());
    }

    #[test]
    fn test_check_certificate_wrong_delta() {
        let logits = vec![10.0, 5.0, 3.0];
        let mut cert = build_margin_certificate(0, logits).unwrap();
        cert.delta = 100.0;
        assert!(!check_margin_certificate(&cert).is_empty());
    }

    #[test]
    fn test_check_certificate_wrong_selected() {
        let logits = vec![10.0, 5.0, 3.0];
        let mut cert = build_margin_certificate(0, logits).unwrap();
        cert.selected_token_id = 1;
        assert!(!check_margin_certificate(&cert).is_empty());
    }

    #[test]
    fn test_verify_certified_large_gap() {
        let logits = vec![10.0, 0.1, 0.05, 0.01];
        let cert = build_margin_certificate(0, logits).unwrap();
        let result = verify_margin_certificate(&cert, 1e-3, 2, 1.0, 1.0);
        assert!(result.failures.is_empty());
        assert!(result.certified);
    }

    #[test]
    fn test_verify_uncertified_small_gap() {
        let logits = vec![1.001, 1.000, 0.5, 0.1];
        let cert = build_margin_certificate(0, logits).unwrap();
        let result = verify_margin_certificate(&cert, 1e-3, 80, 1000.0, 500.0);
        assert!(result.failures.is_empty());
        assert!(!result.certified);
    }

    #[test]
    fn test_build_margin_certificate() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let cert = build_margin_certificate(7, logits).unwrap();
        assert_eq!(cert.token_index, 7);
        assert_eq!(cert.selected_token_id, 1);
        assert_eq!(cert.top1_logit, 5.0);
        assert_eq!(cert.top1_token_id, 1);
        assert_eq!(cert.top2_logit, 3.0);
        assert_eq!(cert.top2_token_id, 2);
        assert_eq!(cert.delta, 2.0);
    }

    #[test]
    fn test_matrix_linf_norm() {
        // [[1, -2, 3], [4, -5, 6]]
        // Row sums: 6, 15. Inf-norm = 15.
        let w: Vec<i8> = vec![1, -2, 3, 4, -5, 6];
        assert_eq!(compute_matrix_linf_norm(&w, 2, 3, 1.0), 15.0);
    }

    #[test]
    fn test_matrix_linf_norm_with_scale() {
        let w: Vec<i8> = vec![1, -2, 3, 4, -5, 6];
        assert_eq!(compute_matrix_linf_norm(&w, 2, 3, 0.5), 7.5);
    }
}
