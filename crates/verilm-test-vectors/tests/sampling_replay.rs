//! Tests for sampling replay verification.
//!
//! Verifies that the canonical sampler produces deterministic results
//! and that the verifier detects dishonest sampling: wrong seed, wrong
//! decode params, wrong token_id, and boundary/tie cases.

use verilm_core::constants::ModelConfig;
use verilm_core::sampling::{self, DecodeParams};
use verilm_test_vectors::*;
use verilm_verify;

fn setup_level_b() -> (
    ModelConfig,
    ToyModel,
    verilm_core::types::VerifierKey,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model_with_head(&cfg, 42);
    let key = generate_key_level_b_with_head(
        &cfg,
        &model.layers,
        [7u8; 32],
        Some(model.lm_head.clone()),
    );
    (cfg, model, key)
}

/// Build a trace with final_hidden and token_id from honest sampling.
fn honest_trace_with_sampling(
    cfg: &ModelConfig,
    model: &ToyModel,
    input: &[i8],
    params: &DecodeParams,
    batch_seed: &[u8; 32],
    token_index: u32,
) -> verilm_core::types::TokenTrace {
    let layers = forward_pass(cfg, &model.layers, input);
    let final_hidden: Vec<i8> = requantize(&layers.last().unwrap().ffn_out);

    // Compute the honest token_id via the canonical sampler
    let token_id = sampling::replay_sampling(
        &model.lm_head,
        &final_hidden,
        cfg.vocab_size,
        cfg.hidden_dim,
        params,
        batch_seed,
        token_index,
    );

    let mut trace = build_trace_with_hidden(
        layers,
        token_index,
        token_index as usize + 1,
        Some(final_hidden),
    );
    trace.token_id = Some(token_id);
    trace
}

// -----------------------------------------------------------------------
// Honest path
// -----------------------------------------------------------------------

#[test]
fn test_honest_sampling_replay_passes() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let batch_seed = [42u8; 32];

    let trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &batch_seed, 0);

    let failures = verilm_verify::verify_sampling(&key, &trace, &params, &batch_seed);
    assert!(failures.is_empty(), "honest sampling should pass: {:?}", failures);
}

#[test]
fn test_honest_sampling_with_temperature() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 0.7, top_k: 0, top_p: 1.0 };
    let batch_seed = [99u8; 32];

    let trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &batch_seed, 0);

    let failures = verilm_verify::verify_sampling(&key, &trace, &params, &batch_seed);
    assert!(failures.is_empty(), "honest sampling with temp should pass: {:?}", failures);
}

#[test]
fn test_honest_sampling_with_top_k() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 1.0, top_k: 5, top_p: 1.0 };
    let batch_seed = [55u8; 32];

    let trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &batch_seed, 0);

    let failures = verilm_verify::verify_sampling(&key, &trace, &params, &batch_seed);
    assert!(failures.is_empty(), "honest sampling with top_k should pass: {:?}", failures);
}

#[test]
fn test_honest_sampling_with_top_p() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 1.0, top_k: 0, top_p: 0.9 };
    let batch_seed = [77u8; 32];

    let trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &batch_seed, 0);

    let failures = verilm_verify::verify_sampling(&key, &trace, &params, &batch_seed);
    assert!(failures.is_empty(), "honest sampling with top_p should pass: {:?}", failures);
}

#[test]
fn test_honest_sampling_multiple_tokens() {
    let (cfg, model, key) = setup_level_b();
    let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let batch_seed = [42u8; 32];

    // Verify multiple tokens with different inputs
    for t in 0..5u32 {
        let input: Vec<i8> = (0..cfg.hidden_dim)
            .map(|i| ((i + t as usize * 7) % 256) as i8)
            .collect();
        let trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &batch_seed, t);
        let failures = verilm_verify::verify_sampling(&key, &trace, &params, &batch_seed);
        assert!(failures.is_empty(), "token {} should pass: {:?}", t, failures);
    }
}

// -----------------------------------------------------------------------
// Wrong seed detected
// -----------------------------------------------------------------------

#[test]
fn test_wrong_seed_detected() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
    let honest_seed = [42u8; 32];

    let trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &honest_seed, 0);

    // Verify with wrong seed — should detect mismatch
    // (unless both seeds happen to produce the same token, which is unlikely
    // with temperature=1.0 and the toy model's logit distribution)
    let wrong_seed = [99u8; 32];

    // First check they actually produce different tokens
    let layers = forward_pass(&cfg, &model.layers, &input);
    let fh = requantize(&layers.last().unwrap().ffn_out);
    let honest_tid = sampling::replay_sampling(
        &model.lm_head, &fh, cfg.vocab_size, cfg.hidden_dim,
        &params, &honest_seed, 0,
    );
    let wrong_tid = sampling::replay_sampling(
        &model.lm_head, &fh, cfg.vocab_size, cfg.hidden_dim,
        &params, &wrong_seed, 0,
    );

    if honest_tid != wrong_tid {
        let failures = verilm_verify::verify_sampling(&key, &trace, &params, &wrong_seed);
        assert!(!failures.is_empty(), "wrong seed should be detected");
        assert!(failures.iter().any(|f| f.contains("sampling replay mismatch")));
    }
    // If they happen to match (unlikely), this test is vacuously true
}

// -----------------------------------------------------------------------
// Wrong decode params detected
// -----------------------------------------------------------------------

#[test]
fn test_wrong_temperature_detected() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let honest_params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let batch_seed = [42u8; 32];

    let trace = honest_trace_with_sampling(&cfg, &model, &input, &honest_params, &batch_seed, 0);

    // Greedy picked some token. Verify with temperature=1.0 — may pick different.
    let wrong_params = DecodeParams { temperature: 1.0, top_k: 0, top_p: 1.0 };

    // Check if the two params produce different results
    let layers = forward_pass(&cfg, &model.layers, &input);
    let fh = requantize(&layers.last().unwrap().ffn_out);
    let honest_tid = sampling::replay_sampling(
        &model.lm_head, &fh, cfg.vocab_size, cfg.hidden_dim,
        &honest_params, &batch_seed, 0,
    );
    let wrong_tid = sampling::replay_sampling(
        &model.lm_head, &fh, cfg.vocab_size, cfg.hidden_dim,
        &wrong_params, &batch_seed, 0,
    );

    if honest_tid != wrong_tid {
        let failures = verilm_verify::verify_sampling(&key, &trace, &wrong_params, &batch_seed);
        assert!(!failures.is_empty(), "wrong temperature should be detected");
    }
}

// -----------------------------------------------------------------------
// Wrong token under honest logits
// -----------------------------------------------------------------------

#[test]
fn test_wrong_token_id_detected() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let batch_seed = [42u8; 32];

    let mut trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &batch_seed, 0);
    let honest_tid = trace.token_id.unwrap();

    // ATTACK: claim a different token
    let fake_tid = if honest_tid == 0 { 1 } else { 0 };
    trace.token_id = Some(fake_tid);

    let failures = verilm_verify::verify_sampling(&key, &trace, &params, &batch_seed);
    assert!(!failures.is_empty(), "wrong token_id must be detected");
    assert!(
        failures.iter().any(|f| f.contains("sampling replay mismatch")),
        "should report mismatch: {:?}", failures
    );
}

// -----------------------------------------------------------------------
// Missing fields detected
// -----------------------------------------------------------------------

#[test]
fn test_missing_token_id_detected() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let batch_seed = [42u8; 32];

    let mut trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &batch_seed, 0);
    trace.token_id = None;

    let failures = verilm_verify::verify_sampling(&key, &trace, &params, &batch_seed);
    assert!(!failures.is_empty());
    assert!(failures.iter().any(|f| f.contains("missing token_id")));
}

#[test]
fn test_missing_final_hidden_detected() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let batch_seed = [42u8; 32];

    let mut trace = honest_trace_with_sampling(&cfg, &model, &input, &params, &batch_seed, 0);
    trace.final_hidden = None;

    let failures = verilm_verify::verify_sampling(&key, &trace, &params, &batch_seed);
    assert!(!failures.is_empty());
    assert!(failures.iter().any(|f| f.contains("missing final_hidden")));
}

#[test]
fn test_missing_lm_head_detected() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42); // no lm_head
    let key = generate_key(&cfg, &model, [7u8; 32]); // key without lm_head

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let layers = forward_pass(&cfg, &model, &input);
    let mut trace = build_trace_with_hidden(layers, 0, 1, Some(vec![0i8; cfg.hidden_dim]));
    trace.token_id = Some(0);

    let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let failures = verilm_verify::verify_sampling(&key, &trace, &params, &[0u8; 32]);
    assert!(!failures.is_empty());
    assert!(failures.iter().any(|f| f.contains("missing lm_head")));
}

// -----------------------------------------------------------------------
// Per-token seed isolation
// -----------------------------------------------------------------------

#[test]
fn test_per_token_seed_isolation() {
    // Two tokens with same batch seed but different indices get different seeds
    let batch_seed = [42u8; 32];
    let s0 = sampling::derive_token_seed(&batch_seed, 0);
    let s1 = sampling::derive_token_seed(&batch_seed, 1);
    assert_ne!(s0, s1, "per-token seeds must differ by index");
}

// -----------------------------------------------------------------------
// Replay under different manifest (via different DecodeParams)
// -----------------------------------------------------------------------

#[test]
fn test_replay_with_different_manifest_params() {
    let (cfg, model, key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let batch_seed = [42u8; 32];

    // Commit with greedy params
    let honest_params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let trace = honest_trace_with_sampling(&cfg, &model, &input, &honest_params, &batch_seed, 0);

    // Honest passes
    let failures = verilm_verify::verify_sampling(&key, &trace, &honest_params, &batch_seed);
    assert!(failures.is_empty(), "honest should pass: {:?}", failures);

    // Different manifest params (top_k=1 still greedy-like, so may not differ)
    // Use top_k=2 + temperature=1.0 to maximize chance of different result
    let alt_params = DecodeParams { temperature: 1.0, top_k: 2, top_p: 1.0 };

    let layers = forward_pass(&cfg, &model.layers, &input);
    let fh = requantize(&layers.last().unwrap().ffn_out);
    let alt_tid = sampling::replay_sampling(
        &model.lm_head, &fh, cfg.vocab_size, cfg.hidden_dim,
        &alt_params, &batch_seed, 0,
    );
    let honest_tid = trace.token_id.unwrap();

    if alt_tid != honest_tid {
        let failures = verilm_verify::verify_sampling(&key, &trace, &alt_params, &batch_seed);
        assert!(!failures.is_empty(), "different manifest params should detect mismatch");
    }
}

// -----------------------------------------------------------------------
// Determinism: same inputs always produce same result
// -----------------------------------------------------------------------

#[test]
fn test_sampling_is_deterministic() {
    let (cfg, model, _key) = setup_level_b();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let params = DecodeParams { temperature: 0.8, top_k: 10, top_p: 0.95 };
    let batch_seed = [123u8; 32];

    let layers = forward_pass(&cfg, &model.layers, &input);
    let fh = requantize(&layers.last().unwrap().ffn_out);

    let results: Vec<u32> = (0..10).map(|_| {
        sampling::replay_sampling(
            &model.lm_head, &fh, cfg.vocab_size, cfg.hidden_dim,
            &params, &batch_seed, 0,
        )
    }).collect();

    assert!(results.windows(2).all(|w| w[0] == w[1]),
        "same inputs must always produce same token: {:?}", results);
}
