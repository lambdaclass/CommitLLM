/// Level B integration tests: margin certificates for token-generation integrity.
///
/// Tests cover:
/// - Honest certificate passes verification
/// - Missing certificate fails Level B
/// - Tampered logits detected
/// - Wrong top-1 claim detected
/// - Forged delta detected
/// - Narrow margin correctly uncertified
/// - End-to-end with toy model + lm_head

use verilm_core::constants::ModelConfig;
use verilm_core::margin::{
    build_margin_certificate, compute_perturbation_bound, verify_margin_certificate,
    DEFAULT_ATTENTION_EPSILON,
};
use verilm_test_vectors::*;
use verilm_verify::{verify_margin, verify_trace_at_level, Verdict, VerificationLevel};

fn toy_cfg() -> ModelConfig {
    ModelConfig::toy()
}

/// Helper: run forward pass, compute logits, build margin cert, attach to trace.
/// Includes final_hidden binding for Level B logit verification.
fn build_level_b_trace(
    cfg: &ModelConfig,
    model: &ToyModel,
    input: &[i8],
    token_index: u32,
) -> verilm_core::types::TokenTrace {
    let layers = forward_pass(cfg, &model.layers, input);
    let last_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &last_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(token_index, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, token_index, 10, Some(last_hidden));
    trace.margin_cert = Some(cert);
    trace
}

#[test]
fn test_honest_margin_cert_passes() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let trace = build_level_b_trace(&cfg, &model, &input, 0);

    // Level A should pass (it ignores margin cert)
    let report_a = verify_trace_at_level(&key, &trace, VerificationLevel::A, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report_a.verdict, Verdict::Pass, "Level A: {:?}", report_a.failures);

    // Level B: if delta > 2*B, should be certified
    let report_b = verify_trace_at_level(&key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON);
    // The toy model has random weights, so the margin may or may not be large enough.
    // But structural checks must pass.
    let structural_failures: Vec<_> = report_b
        .failures
        .iter()
        .filter(|f| !f.contains("margin not certified"))
        .collect();
    assert!(
        structural_failures.is_empty(),
        "structural failures: {:?}",
        structural_failures
    );
}

#[test]
fn test_missing_margin_cert_fails_level_b() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let layers = forward_pass(&cfg, &model.layers, &input);
    let trace = build_trace(layers, 0, 10);
    // No margin_cert attached

    let report = verify_trace_at_level(&key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("missing margin certificate")),
        "expected missing cert failure, got: {:?}",
        report.failures
    );
}

#[test]
fn test_tampered_logits_detected() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let mut trace = build_level_b_trace(&cfg, &model, &input, 0);

    // Tamper the logit vector: swap two values to break top-1/top-2 extraction
    if let Some(ref mut cert) = trace.margin_cert {
        if cert.logits.len() > 3 {
            cert.logits[0] = cert.top1_logit + 100.0; // make index 0 the new top
        }
    }

    let report = verify_trace_at_level(&key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("margin") || f.contains("logit mismatch")),
        "expected margin or logit mismatch failure, got: {:?}",
        report.failures
    );
}

#[test]
fn test_wrong_top1_claim_detected() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let mut trace = build_level_b_trace(&cfg, &model, &input, 0);

    // Lie about which token is top-1
    if let Some(ref mut cert) = trace.margin_cert {
        cert.top1_token_id = cert.top2_token_id;
        cert.selected_token_id = cert.top2_token_id;
    }

    let report = verify_trace_at_level(&key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("top1_token_id mismatch")),
        "expected top1 mismatch, got: {:?}",
        report.failures
    );
}

#[test]
fn test_forged_delta_detected() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let mut trace = build_level_b_trace(&cfg, &model, &input, 0);

    // Forge a large delta to try to pass certification
    if let Some(ref mut cert) = trace.margin_cert {
        cert.delta = 99999.0;
    }

    let report = verify_trace_at_level(&key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("delta mismatch")),
        "expected delta mismatch, got: {:?}",
        report.failures
    );
}

#[test]
fn test_narrow_margin_uncertified() {
    // Build a logit vector with a very narrow margin
    let mut logits = vec![0.0f32; 64];
    logits[0] = 1.001;
    logits[1] = 1.000;

    let cert = build_margin_certificate(0, logits).unwrap();

    // With huge model params, bound will be large, margin won't certify
    let result = verify_margin_certificate(&cert, 1e-3, 80, 1000.0, 500.0);
    assert!(result.failures.is_empty(), "structural should pass");
    assert!(!result.certified, "narrow margin should not certify");
}

#[test]
fn test_wide_margin_certified() {
    // Build a logit vector with a very wide margin
    let mut logits = vec![0.0f32; 64];
    logits[0] = 100.0;
    logits[1] = 0.1;

    let cert = build_margin_certificate(0, logits).unwrap();

    // With tiny model params, bound will be small, margin certifies
    let result = verify_margin_certificate(&cert, 1e-3, 2, 1.0, 1.0);
    assert!(result.failures.is_empty(), "structural should pass");
    assert!(result.certified, "wide margin should certify: delta={}, 2*B={}", result.delta, 2.0 * result.perturbation_bound);
}

#[test]
fn test_level_a_ignores_margin_cert() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let mut trace = build_level_b_trace(&cfg, &model, &input, 0);

    // Corrupt the margin cert — Level A should still pass
    if let Some(ref mut cert) = trace.margin_cert {
        cert.delta = 99999.0;
        cert.top1_token_id = 99999;
    }

    let report = verify_trace_at_level(&key, &trace, VerificationLevel::A, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, Verdict::Pass, "Level A should ignore margin: {:?}", report.failures);
}

// -----------------------------------------------------------------------
// Bug fix: certification outcome must change when bound parameters change
// -----------------------------------------------------------------------

#[test]
fn test_larger_wo_norms_flips_certification() {
    // A wide-margin cert that certifies with small norms...
    let mut logits = vec![0.0f32; 64];
    logits[0] = 10.0;
    logits[1] = 0.1;
    let cert = build_margin_certificate(0, logits).unwrap();
    let delta = cert.delta;

    // ...certifies with small bound
    let small_bound = compute_perturbation_bound(1e-3, 2, 1.0, 1.0);
    assert!(delta > 2.0 * small_bound, "should certify with small norms");

    // ...but NOT with large wo_norm that pushes 2*B above delta
    let large_bound = compute_perturbation_bound(1e-3, 2, 1.0, 10000.0);
    assert!(delta < 2.0 * large_bound, "should NOT certify with large wo_norm");

    // Verify through the full verify_margin_certificate path
    let result_small = verify_margin_certificate(&cert, 1e-3, 2, 1.0, 1.0);
    assert!(result_small.certified, "small norms should certify");

    let result_large = verify_margin_certificate(&cert, 1e-3, 2, 1.0, 10000.0);
    assert!(!result_large.certified, "large wo_norm should uncertify");
}

#[test]
fn test_larger_v_norm_flips_certification() {
    // Same cert, but max_v_norm changes the outcome
    let mut logits = vec![0.0f32; 64];
    logits[0] = 10.0;
    logits[1] = 0.1;
    let cert = build_margin_certificate(0, logits).unwrap();

    let result_small = verify_margin_certificate(&cert, 1e-3, 2, 1.0, 1.0);
    assert!(result_small.certified, "small v_norm should certify");

    let result_large = verify_margin_certificate(&cert, 1e-3, 2, 10000.0, 1.0);
    assert!(!result_large.certified, "large v_norm should uncertify");
}

#[test]
fn test_verify_margin_uses_key_v_norms() {
    // Bug: verify_margin hardcoded max_v_norm=1.0. This test ensures the
    // key's max_v_norm is actually used in the bound computation.
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let mut key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let trace = build_level_b_trace(&cfg, &model, &input, 0);

    // With real key norms, get a baseline result
    let baseline = verify_margin(&key, &trace, DEFAULT_ATTENTION_EPSILON);

    // Now inflate max_v_norm in the key to something huge — should make
    // the bound so large that certification fails (if it was passing)
    // or stays failing (but the bound should be different)
    key.max_v_norm = 1e6;

    let inflated = verify_margin(&key, &trace, DEFAULT_ATTENTION_EPSILON);

    // With a huge max_v_norm, the bound is enormous, so certification must fail
    assert!(
        inflated.iter().any(|f| f.contains("margin not certified")),
        "inflated max_v_norm should prevent certification, got: {:?} (baseline: {:?})",
        inflated,
        baseline
    );
}

#[test]
fn test_missing_wo_norms_fails_level_b() {
    // A key without wo_norms should fail Level B verification
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key(&cfg, &model.layers, [1u8; 32]); // Level A key, no wo_norms
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let trace = build_level_b_trace(&cfg, &model, &input, 0);

    let report = verify_trace_at_level(&key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("wo_norms")),
        "expected wo_norms failure, got: {:?}",
        report.failures
    );
}
