//! Integration tests for prefix-KV chain binding (Level C).
//!
//! Tests self-consistency (each token's KV cache entry matches its own
//! requantized K/V projection) and cross-token consistency (when co-opened
//! tokens share KV cache entries, those entries match the source token's
//! requantized projections).

use std::collections::BTreeMap;

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

// ── Self-consistency: honest single-token passes ─────────────

#[test]
fn test_honest_single_token_kv_self_consistency() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(
        &cfg,
        &model.layers,
        [0u8; 32],
        Some(model.lm_head.clone()),
    );
    let trace = build_level_c_trace(&cfg, &model, &vec![1i8; cfg.hidden_dim], 0);

    // Self-consistency via verify_attention (includes self-check now)
    let attn_tol = verilm_core::attention::AttentionToleranceConfig { max_abs_diff: 0 };
    let attn_failures = verilm_verify::verify_attention(&key, &trace, &attn_tol);
    let self_consistency_failures: Vec<_> = attn_failures
        .iter()
        .filter(|f| f.contains("KV self-consistency"))
        .collect();
    assert!(
        self_consistency_failures.is_empty(),
        "honest trace should pass self-consistency: {:?}",
        self_consistency_failures
    );

    // Also via verify_kv_chain
    let mut opened = BTreeMap::new();
    opened.insert(trace.token_index, &trace);
    let kv_failures = verilm_verify::verify_kv_chain(&opened);
    assert!(
        kv_failures.is_empty(),
        "honest single-token KV chain should pass: {:?}",
        kv_failures
    );
}

// ── Self-consistency + cross-token: honest multi-token passes ─

#[test]
fn test_honest_multi_token_kv_chain_passes() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 99);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        3,
    );

    let traces: Vec<verilm_core::types::TokenTrace> = all_layers
        .iter()
        .enumerate()
        .map(|(t, layers)| {
            let final_hidden = requantize(&layers.last().unwrap().ffn_out);
            let logits =
                compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
            let cert = build_margin_certificate(t as u32, logits).unwrap();
            let mut trace =
                build_trace_with_hidden(layers.clone(), t as u32, 3, Some(final_hidden));
            trace.margin_cert = Some(cert);
            trace
        })
        .collect();

    let opened: BTreeMap<u32, &verilm_core::types::TokenTrace> =
        traces.iter().map(|t| (t.token_index, t)).collect();
    let kv_failures = verilm_verify::verify_kv_chain(&opened);
    assert!(
        kv_failures.is_empty(),
        "honest multi-token KV chain should pass: {:?}",
        kv_failures
    );
}

// ── Tampered self KV K detected ──────────────────────────────

#[test]
fn test_tampered_self_kv_k_detected() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let mut layers = forward_pass_level_c(&cfg, &model.layers, &vec![1i8; cfg.hidden_dim]);

    layers[0].kv_cache_k[0][0] = layers[0].kv_cache_k[0][0].wrapping_add(10);

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(0, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 0, 1, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(
        &cfg,
        &model.layers,
        [0u8; 32],
        Some(model.lm_head.clone()),
    );

    let attn_tol = verilm_core::attention::AttentionToleranceConfig { max_abs_diff: 0 };
    let failures = verilm_verify::verify_attention(&key, &trace, &attn_tol);
    assert!(
        failures.iter().any(|f| f.contains("KV self-consistency") && f.contains("kv_cache_k")),
        "tampered self K should be detected: {:?}",
        failures
    );
}

// ── Tampered self KV V detected ──────────────────────────────

#[test]
fn test_tampered_self_kv_v_detected() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let mut layers = forward_pass_level_c(&cfg, &model.layers, &vec![1i8; cfg.hidden_dim]);

    layers[0].kv_cache_v[0][0] = layers[0].kv_cache_v[0][0].wrapping_add(10);

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(0, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, 0, 1, Some(final_hidden));
    trace.margin_cert = Some(cert);

    let key = generate_key_level_b_with_head(
        &cfg,
        &model.layers,
        [0u8; 32],
        Some(model.lm_head.clone()),
    );

    let attn_tol = verilm_core::attention::AttentionToleranceConfig { max_abs_diff: 0 };
    let failures = verilm_verify::verify_attention(&key, &trace, &attn_tol);
    assert!(
        failures.iter().any(|f| f.contains("KV self-consistency") && f.contains("kv_cache_v")),
        "tampered self V should be detected: {:?}",
        failures
    );
}

// ── Tampered prior KV K detected (cross-token) ──────────────

#[test]
fn test_tampered_prior_kv_k_detected() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 99);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        2,
    );

    let layers_t0 = all_layers[0].clone();
    let mut layers_t1 = all_layers[1].clone();

    layers_t1[0].kv_cache_k[0][0] = layers_t1[0].kv_cache_k[0][0].wrapping_add(10);

    let fh0 = requantize(&layers_t0.last().unwrap().ffn_out);
    let fh1 = requantize(&layers_t1.last().unwrap().ffn_out);
    let logits0 = compute_logits(&model.lm_head, &fh0, cfg.vocab_size, cfg.hidden_dim);
    let logits1 = compute_logits(&model.lm_head, &fh1, cfg.vocab_size, cfg.hidden_dim);
    let cert0 = build_margin_certificate(0, logits0).unwrap();
    let cert1 = build_margin_certificate(1, logits1).unwrap();

    let mut trace0 = build_trace_with_hidden(layers_t0, 0, 2, Some(fh0));
    trace0.margin_cert = Some(cert0);
    let mut trace1 = build_trace_with_hidden(layers_t1, 1, 2, Some(fh1));
    trace1.margin_cert = Some(cert1);

    let mut opened = BTreeMap::new();
    opened.insert(0u32, &trace0);
    opened.insert(1u32, &trace1);
    let failures = verilm_verify::verify_kv_chain(&opened);
    assert!(
        failures.iter().any(|f| f.contains("cross-token KV mismatch") && f.contains("kv_cache_k")),
        "tampered prior K should be detected: {:?}",
        failures
    );
}

// ── Tampered prior KV V detected (cross-token) ──────────────

#[test]
fn test_tampered_prior_kv_v_detected() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 99);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        2,
    );

    let layers_t0 = all_layers[0].clone();
    let mut layers_t1 = all_layers[1].clone();

    layers_t1[0].kv_cache_v[0][0] = layers_t1[0].kv_cache_v[0][0].wrapping_add(10);

    let fh0 = requantize(&layers_t0.last().unwrap().ffn_out);
    let fh1 = requantize(&layers_t1.last().unwrap().ffn_out);
    let logits0 = compute_logits(&model.lm_head, &fh0, cfg.vocab_size, cfg.hidden_dim);
    let logits1 = compute_logits(&model.lm_head, &fh1, cfg.vocab_size, cfg.hidden_dim);
    let cert0 = build_margin_certificate(0, logits0).unwrap();
    let cert1 = build_margin_certificate(1, logits1).unwrap();

    let mut trace0 = build_trace_with_hidden(layers_t0, 0, 2, Some(fh0));
    trace0.margin_cert = Some(cert0);
    let mut trace1 = build_trace_with_hidden(layers_t1, 1, 2, Some(fh1));
    trace1.margin_cert = Some(cert1);

    let mut opened = BTreeMap::new();
    opened.insert(0u32, &trace0);
    opened.insert(1u32, &trace1);
    let failures = verilm_verify::verify_kv_chain(&opened);
    assert!(
        failures.iter().any(|f| f.contains("cross-token KV mismatch") && f.contains("kv_cache_v")),
        "tampered prior V should be detected: {:?}",
        failures
    );
}

// ── Fabricated KV from a different input ─────────────────────

#[test]
fn test_fabricated_kv_from_different_input() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 99);

    let all_layers_a = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        2,
    );

    let input_b: Vec<i8> = (0..cfg.hidden_dim).map(|i| if i % 2 == 0 { 50i8 } else { -50i8 }).collect();
    let all_layers_b = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &input_b,
        2,
    );

    let k_a = verilm_test_vectors::requantize(&all_layers_a[0][0].k);
    let k_b = verilm_test_vectors::requantize(&all_layers_b[0][0].k);
    assert_ne!(k_a, k_b, "precondition: runs A and B must produce different K projections for token 0");

    let layers_t0 = all_layers_a[0].clone();
    let mut layers_t1 = all_layers_a[1].clone();
    for (layer_idx, lt) in layers_t1.iter_mut().enumerate() {
        lt.kv_cache_k[0] = all_layers_b[0][layer_idx].kv_cache_k[0].clone();
        lt.kv_cache_v[0] = all_layers_b[0][layer_idx].kv_cache_v[0].clone();
    }

    let fh0 = requantize(&layers_t0.last().unwrap().ffn_out);
    let fh1 = requantize(&layers_t1.last().unwrap().ffn_out);
    let logits0 = compute_logits(&model.lm_head, &fh0, cfg.vocab_size, cfg.hidden_dim);
    let logits1 = compute_logits(&model.lm_head, &fh1, cfg.vocab_size, cfg.hidden_dim);
    let cert0 = build_margin_certificate(0, logits0).unwrap();
    let cert1 = build_margin_certificate(1, logits1).unwrap();

    let mut trace0 = build_trace_with_hidden(layers_t0, 0, 2, Some(fh0));
    trace0.margin_cert = Some(cert0);
    let mut trace1 = build_trace_with_hidden(layers_t1, 1, 2, Some(fh1));
    trace1.margin_cert = Some(cert1);

    let mut opened = BTreeMap::new();
    opened.insert(0u32, &trace0);
    opened.insert(1u32, &trace1);
    let failures = verilm_verify::verify_kv_chain(&opened);
    assert!(
        !failures.is_empty(),
        "fabricated KV from different input should be detected"
    );
    assert!(
        failures.iter().any(|f| f.contains("cross-token KV mismatch")),
        "should report cross-token mismatch: {:?}",
        failures
    );
}

// ── Only opened tokens are checked (no spurious failures) ────

#[test]
fn test_kv_chain_only_checks_opened_tokens() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 99);
    let all_layers = forward_pass_autoregressive_level_c(
        &cfg,
        &model.layers,
        &vec![5i8; cfg.hidden_dim],
        3,
    );

    let fh0 = requantize(&all_layers[0].last().unwrap().ffn_out);
    let fh2 = requantize(&all_layers[2].last().unwrap().ffn_out);
    let logits0 = compute_logits(&model.lm_head, &fh0, cfg.vocab_size, cfg.hidden_dim);
    let logits2 = compute_logits(&model.lm_head, &fh2, cfg.vocab_size, cfg.hidden_dim);
    let cert0 = build_margin_certificate(0, logits0).unwrap();
    let cert2 = build_margin_certificate(2, logits2).unwrap();

    let mut trace0 =
        build_trace_with_hidden(all_layers[0].clone(), 0, 3, Some(fh0));
    trace0.margin_cert = Some(cert0);
    let mut trace2 =
        build_trace_with_hidden(all_layers[2].clone(), 2, 3, Some(fh2));
    trace2.margin_cert = Some(cert2);

    let mut opened = BTreeMap::new();
    opened.insert(0u32, &trace0);
    opened.insert(2u32, &trace2);
    let failures = verilm_verify::verify_kv_chain(&opened);

    assert!(
        failures.is_empty(),
        "opening tokens 0 and 2 (not 1) should not produce spurious failures: {:?}",
        failures
    );
}

// ── Level A ignores KV chain check ───────────────────────────

#[test]
fn test_level_a_ignores_kv_chain() {
    let cfg = toy_cfg();
    let model = generate_model_with_head(&cfg, 42);
    let mut layers = forward_pass_level_c(&cfg, &model.layers, &vec![1i8; cfg.hidden_dim]);

    layers[0].kv_cache_k[0][0] = layers[0].kv_cache_k[0][0].wrapping_add(10);

    let final_hidden = requantize(&layers.last().unwrap().ffn_out);
    let mut trace = build_trace_with_hidden(layers, 0, 1, Some(final_hidden));
    trace.margin_cert = None;

    let key = generate_key_level_b_with_head(
        &cfg,
        &model.layers,
        [0u8; 32],
        Some(model.lm_head.clone()),
    );

    let report =
        verilm_verify::verify_trace_at_level(&key, &trace, VerificationLevel::A, DEFAULT_ATTENTION_EPSILON);
    let has_kv_failure = report
        .failures
        .iter()
        .any(|f| f.contains("KV self-consistency") || f.contains("cross-token KV"));
    assert!(
        !has_kv_failure,
        "Level A should not run KV chain checks, got: {:?}",
        report.failures
    );
}
