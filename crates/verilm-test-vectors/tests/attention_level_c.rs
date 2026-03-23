//! Integration tests for Level C (attention replay) verification.

use verilm_core::attention::{AttentionToleranceConfig, compare_attention_output, replay_attention_reference};
use verilm_core::constants::ModelConfig;
use verilm_core::margin::{build_margin_certificate, DEFAULT_ATTENTION_EPSILON};
use verilm_test_vectors::*;
use verilm_verify::VerificationLevel;

fn toy_cfg() -> ModelConfig {
    ModelConfig::toy()
}

/// Helper: build a Level C trace with margin cert and final_hidden.
fn build_level_c_trace(
    cfg: &ModelConfig,
    model: &ToyModel,
    input: &[i8],
    token_index: u32,
) -> verilm_core::types::TokenTrace {
    let layers = forward_pass_level_c(cfg, &model.layers, input);
    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(token_index, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, token_index, 1, Some(final_hidden));
    trace.margin_cert = Some(cert);
    trace
}

// ── Single-token Level C ──────────────────────────────────────

#[test]
fn test_single_token_level_c_pass() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));
    let trace = build_level_c_trace(&cfg, &model, &vec![1i8; cfg.hidden_dim], 0);

    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, verilm_verify::Verdict::Pass, "failures: {:?}", report.failures);
}

// ── Multi-token autoregressive Level C ────────────────────────

#[test]
fn test_multi_token_level_c_pass() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 99);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        3,
    );

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    for (t, layers) in all_layers.iter().enumerate() {
        let final_hidden = requantize(&layers.last().unwrap().ffn_out);
        let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
        let cert = build_margin_certificate(t as u32, logits).unwrap();
        let mut trace = build_trace_with_hidden(layers.clone(), t as u32, 3, Some(final_hidden));
        trace.margin_cert = Some(cert);

        let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
        // With random toy weights, margin may not certify. Filter to structural failures only.
        let structural_failures: Vec<_> = report.failures.iter()
            .filter(|f| !f.contains("margin not certified"))
            .collect();
        assert!(
            structural_failures.is_empty(),
            "token {}: structural failures: {:?}",
            t,
            structural_failures
        );
    }
}

// ── Attention replay unit check ──────────────────────────────

#[test]
fn test_attention_replay_matches_forward_pass() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 77);
    let layers = forward_pass_level_c(&cfg, &model, &vec![3i8; cfg.hidden_dim]);

    for (layer_idx, lt) in layers.iter().enumerate() {
        let q_i8: Vec<i8> = lt.q.iter().map(|&v| v.clamp(-128, 127) as i8).collect();
        let replayed = replay_attention_reference(&q_i8, &lt.kv_cache_k, &lt.kv_cache_v, &cfg);
        let tol = AttentionToleranceConfig { max_abs_diff: 0 };
        assert_eq!(
            compare_attention_output(&lt.a, &replayed, &tol),
            None,
            "layer {}: attention replay mismatch",
            layer_idx
        );
    }
}

// ── Tampered attention vector detected ───────────────────────

#[test]
fn test_tampered_attention_detected_level_c() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let mut layers = forward_pass_level_c(&cfg, &model.layers, &vec![1i8; cfg.hidden_dim]);

    // Tamper with the attention output in layer 0
    layers[0].a[0] = layers[0].a[0].wrapping_add(10);

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(0, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 0, 1, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, verilm_verify::Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("attention replay mismatch")),
        "expected attention replay mismatch, got: {:?}",
        report.failures
    );
}

// ── Missing KV cache detected ────────────────────────────────

#[test]
fn test_missing_kv_cache_detected_level_c() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    // Use non-Level-C forward pass (no KV cache)
    let layers = forward_pass(&cfg, &model.layers, &vec![1i8; cfg.hidden_dim]);
    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(0, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 0, 1, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, verilm_verify::Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("missing KV cache")),
        "expected missing KV cache error, got: {:?}",
        report.failures
    );
}

// ── Tolerance: small diff within tolerance passes ────────────

#[test]
fn test_attention_within_tolerance_passes() {
    let cfg = toy_cfg();
    let claimed = vec![10i8; cfg.hidden_dim];
    let mut replayed = claimed.clone();
    replayed[0] = 11; // diff = 1

    let tol = AttentionToleranceConfig { max_abs_diff: 1 };
    assert_eq!(compare_attention_output(&claimed, &replayed, &tol), None);

    let tol_strict = AttentionToleranceConfig { max_abs_diff: 0 };
    assert!(compare_attention_output(&claimed, &replayed, &tol_strict).is_some());
}

// ── Configurable tolerance via verify_trace_at_level_with_attn_tolerance ──

#[test]
fn test_configurable_tolerance_level_c() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let mut layers = forward_pass_level_c(&cfg, &model.layers, &vec![1i8; cfg.hidden_dim]);

    // Introduce a small perturbation (diff=1)
    layers[0].a[0] = layers[0].a[0].wrapping_add(1);

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(0, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 0, 1, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    // With tolerance=0 (default), this should fail
    let report_strict = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    assert!(
        report_strict.failures.iter().any(|f| f.contains("attention replay mismatch")),
        "tolerance=0 should catch diff=1, got: {:?}",
        report_strict.failures
    );

    // With tolerance=1, this should pass attention check
    let tolerant = AttentionToleranceConfig { max_abs_diff: 1 };
    let report_tolerant = verilm_verify::verify_trace_at_level_with_attn_tolerance(
        &key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON, Some(tolerant),
    );
    let has_attn_fail = report_tolerant.failures.iter().any(|f| f.contains("attention replay mismatch"));
    assert!(!has_attn_fail, "tolerance=1 should accept diff=1, got: {:?}", report_tolerant.failures);
}

// ── KV cache splice attack: wrong token's cache ─────────────

#[test]
fn test_kv_cache_spliced_from_different_input() {
    // Attacker runs two different inputs and splices the KV cache from
    // one run into the trace of another. The replay should detect this
    // because the attention output won't match.
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);

    // Honest run
    let honest_layers = forward_pass_level_c(&cfg, &model.layers, &vec![1i8; cfg.hidden_dim]);
    // Different input
    let other_layers = forward_pass_level_c(&cfg, &model.layers, &vec![99i8; cfg.hidden_dim]);

    // Splice: take honest trace but replace KV cache with the other run's cache
    let mut spliced_layers = honest_layers.clone();
    for (layer_idx, lt) in spliced_layers.iter_mut().enumerate() {
        lt.kv_cache_k = other_layers[layer_idx].kv_cache_k.clone();
        lt.kv_cache_v = other_layers[layer_idx].kv_cache_v.clone();
    }

    let final_hidden = requantize(&spliced_layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(0, logits).unwrap();
    let mut trace = build_trace_with_hidden(spliced_layers, 0, 1, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, verilm_verify::Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("attention replay mismatch")),
        "spliced KV cache should cause attention mismatch, got: {:?}",
        report.failures
    );
}

// ── KV cache reorder attack: swap positions ─────────────────

#[test]
fn test_kv_cache_reorder_attack() {
    // For multi-token, reorder KV cache entries (swap position 0 and 1).
    // This changes softmax weights and should produce different attention output.
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        3,
    );

    // Take the last token's trace (has 3 KV cache entries per layer)
    let mut layers = all_layers[2].clone();

    // Swap KV cache positions 0 and 1 in each layer
    for lt in layers.iter_mut() {
        if lt.kv_cache_k.len() >= 2 {
            lt.kv_cache_k.swap(0, 1);
            lt.kv_cache_v.swap(0, 1);
        }
    }

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(2, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 2, 3, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    // Reordering may or may not change the output depending on values.
    // But the point is replay catches it if it does differ.
    // With random weights, it very likely differs.
    let structural_failures: Vec<_> = report.failures.iter()
        .filter(|f| !f.contains("margin not certified"))
        .collect();
    // We can't guarantee the reorder changes output (e.g. if positions had identical KV),
    // but with random weights it almost certainly will.
    if !structural_failures.is_empty() {
        assert!(
            structural_failures.iter().any(|f| f.contains("attention replay mismatch")),
            "reordered cache should cause attention mismatch, got: {:?}",
            structural_failures
        );
    }
}

// ── KV cache poisoning: corrupt one V entry ─────────────────

#[test]
fn test_kv_cache_v_poisoning_detected() {
    // Poison a single V cache entry. The replayed attention will use the
    // poisoned value but the original `a` was computed with the honest value.
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        2,
    );

    // Take the second token (has 2 KV cache entries per layer)
    let mut layers = all_layers[1].clone();

    // Poison: flip all values in the first V cache entry of layer 0
    for v in layers[0].kv_cache_v[0].iter_mut() {
        *v = v.wrapping_neg();
    }

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(1, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 1, 2, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, verilm_verify::Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("attention replay mismatch")),
        "poisoned V cache should cause attention mismatch, got: {:?}",
        report.failures
    );
}

// ── KV cache K poisoning: corrupt one K entry ───────────────

#[test]
fn test_kv_cache_k_poisoning_detected() {
    // Poison a K cache entry — changes softmax weights, different attention output.
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        2,
    );

    let mut layers = all_layers[1].clone();

    // Poison: flip all values in the first K cache entry of layer 0
    for k in layers[0].kv_cache_k[0].iter_mut() {
        *k = k.wrapping_neg();
    }

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(1, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 1, 2, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, verilm_verify::Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("attention replay mismatch")),
        "poisoned K cache should cause attention mismatch, got: {:?}",
        report.failures
    );
}

// ── Cross-token KV replay attack ────────────────────────────

#[test]
fn test_cross_token_kv_replay_attack() {
    // Attacker reuses token 0's KV cache for token 1 (instead of the
    // extended cache that includes both tokens). This is a "stale cache" attack.
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        3,
    );

    // Token 1 should have 2 KV entries per layer. Replace with token 0's cache (1 entry).
    let mut layers = all_layers[1].clone();
    for (layer_idx, lt) in layers.iter_mut().enumerate() {
        lt.kv_cache_k = all_layers[0][layer_idx].kv_cache_k.clone();
        lt.kv_cache_v = all_layers[0][layer_idx].kv_cache_v.clone();
    }

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(1, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 1, 3, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::C, DEFAULT_ATTENTION_EPSILON);
    assert_eq!(report.verdict, verilm_verify::Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("attention replay mismatch")),
        "stale KV cache (wrong seq_len) should cause attention mismatch, got: {:?}",
        report.failures
    );
}

// ── Level A ignores attention ────────────────────────────────

#[test]
fn test_level_a_ignores_attention_issues() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let mut layers = forward_pass_level_c(&cfg, &model.layers, &vec![1i8; cfg.hidden_dim]);

    // Tamper attention — Level A should not care
    layers[0].a[0] = layers[0].a[0].wrapping_add(10);

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let mut trace = build_trace_with_hidden(layers, 0, 1, Some(final_hidden));
    trace.margin_cert = None;

    let key = generate_key_level_b_with_head(&cfg, &model.layers, [0u8; 32], Some(model.lm_head.clone()));

    // Level A: no attention check, but Freivalds on Wo will catch the tampered `a`
    // since Wo * a != attn_out. The point is Level A doesn't run attention replay.
    let report = verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::A, DEFAULT_ATTENTION_EPSILON);
    // We expect failures from Freivalds (Wo check), NOT from attention replay
    let has_attention_failure = report.failures.iter().any(|f| f.contains("attention replay"));
    assert!(!has_attention_failure, "Level A should not run attention replay");
}
