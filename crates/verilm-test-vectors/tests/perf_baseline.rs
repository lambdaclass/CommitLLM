//! Performance regression tests.
//!
//! These are NOT benchmarks — they verify that core operations complete within
//! generous time bounds to catch catastrophic regressions (10x slowdowns)
//! without being flaky on CI.

use std::time::Instant;

use verilm_core::constants::ModelConfig;
use verilm_core::serialize;
use verilm_core::types::CompactBatchProof;
use verilm_test_vectors::*;

/// Shared setup: toy config, model, random input, and verifier key.
fn setup() -> (ModelConfig, Vec<LayerWeights>, Vec<i8>, verilm_core::types::VerifierKey) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let key = generate_key(&cfg, &model, [7u8; 32]);
    (cfg, model, input, key)
}

#[test]
fn forward_pass_under_100ms() {
    let (cfg, model, input, _key) = setup();

    let start = Instant::now();
    let _traces = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 100,
        "forward_pass_autoregressive(10 tokens) took {}ms, expected < 100ms",
        elapsed.as_millis()
    );
}

#[test]
fn commit_under_50ms() {
    let (cfg, model, input, _key) = setup();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let token_ids: Vec<u32> = (0..10).collect();

    let start = Instant::now();
    let _result = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"perf test prompt",
            sampling_seed: [1u8; 32],
            manifest: None,
        },
        None,
    );
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 50,
        "commit_with_full_binding(10 tokens) took {}ms, expected < 50ms",
        elapsed.as_millis()
    );
}

#[test]
fn open_all_under_50ms() {
    let (cfg, model, input, _key) = setup();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let token_ids: Vec<u32> = (0..10).collect();
    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"perf test prompt",
            sampling_seed: [1u8; 32],
            manifest: None,
        },
        None,
    );
    let all_indices: Vec<u32> = (0..10).collect();

    let start = Instant::now();
    let _proof = open(&state, &all_indices);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 50,
        "open(10 tokens) took {}ms, expected < 50ms",
        elapsed.as_millis()
    );
}

#[test]
fn verify_batch_under_100ms() {
    let (cfg, model, input, key) = setup();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let token_ids: Vec<u32> = (0..10).collect();
    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"perf test prompt",
            sampling_seed: [1u8; 32],
            manifest: None,
        },
        None,
    );
    let all_indices: Vec<u32> = (0..10).collect();
    let proof = open(&state, &all_indices);

    let start = Instant::now();
    let (passed, failures) = verify_batch(&key, &proof, &all_indices);
    let elapsed = start.elapsed();

    assert!(passed, "verification failed: {:?}", failures);
    assert!(
        elapsed.as_millis() < 100,
        "verify_batch(10 tokens) took {}ms, expected < 100ms",
        elapsed.as_millis()
    );
}

#[test]
fn full_pipeline_under_300ms() {
    let (cfg, model, input, key) = setup();
    let token_ids: Vec<u32> = (0..10).collect();
    let all_indices: Vec<u32> = (0..10).collect();

    let start = Instant::now();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"perf test prompt",
            sampling_seed: [1u8; 32],
            manifest: None,
        },
        None,
    );
    let proof = open(&state, &all_indices);
    let (passed, failures) = verify_batch(&key, &proof, &all_indices);

    let elapsed = start.elapsed();

    assert!(passed, "verification failed: {:?}", failures);
    assert!(
        elapsed.as_millis() < 300,
        "full pipeline(10 tokens) took {}ms, expected < 300ms",
        elapsed.as_millis()
    );
}

#[test]
fn serialization_roundtrip_under_10ms() {
    let (cfg, model, input, _key) = setup();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let token_ids: Vec<u32> = (0..10).collect();
    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"perf test prompt",
            sampling_seed: [1u8; 32],
            manifest: None,
        },
        None,
    );
    let all_indices: Vec<u32> = (0..10).collect();
    let proof = open(&state, &all_indices);

    let start = Instant::now();
    let bytes = serialize::serialize_batch(&proof);
    let roundtrip = serialize::deserialize_batch(&bytes).expect("deserialize failed");
    let elapsed = start.elapsed();

    assert_eq!(roundtrip.traces.len(), proof.traces.len());
    assert!(
        elapsed.as_millis() < 10,
        "serialization roundtrip took {}ms, expected < 10ms",
        elapsed.as_millis()
    );
}

#[test]
fn compact_conversion_under_10ms() {
    let (cfg, model, input, _key) = setup();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let token_ids: Vec<u32> = (0..10).collect();
    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"perf test prompt",
            sampling_seed: [1u8; 32],
            manifest: None,
        },
        None,
    );
    let all_indices: Vec<u32> = (0..10).collect();
    let proof = open(&state, &all_indices);

    let start = Instant::now();
    let compact = CompactBatchProof::from_full(&proof);
    let restored = compact.to_full().expect("compact to_full failed");
    let elapsed = start.elapsed();

    assert_eq!(restored.traces.len(), proof.traces.len());
    assert!(
        elapsed.as_millis() < 10,
        "compact conversion roundtrip took {}ms, expected < 10ms",
        elapsed.as_millis()
    );
}

#[test]
fn key_generation_under_50ms() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);

    let start = Instant::now();
    let _key = generate_key(&cfg, &model, [7u8; 32]);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 50,
        "generate_key(toy) took {}ms, expected < 50ms",
        elapsed.as_millis()
    );
}

#[test]
fn single_token_verify_under_20ms() {
    let (cfg, model, input, key) = setup();
    let layers = forward_pass(&cfg, &model, &input);
    let trace = build_trace(layers, 0, 1);

    let start = Instant::now();
    let (passed, failures) = verify_trace(&key, &trace);
    let elapsed = start.elapsed();

    assert!(passed, "verification failed: {:?}", failures);
    assert!(
        elapsed.as_millis() < 20,
        "verify_trace(1 token) took {}ms, expected < 20ms",
        elapsed.as_millis()
    );
}

#[test]
fn twenty_token_pipeline_under_1s() {
    let (cfg, model, input, key) = setup();
    let token_ids: Vec<u32> = (0..20).collect();
    let all_indices: Vec<u32> = (0..20).collect();

    let start = Instant::now();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 20);
    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"perf test prompt",
            sampling_seed: [1u8; 32],
            manifest: None,
        },
        None,
    );
    let proof = open(&state, &all_indices);
    let (passed, failures) = verify_batch(&key, &proof, &all_indices);

    let elapsed = start.elapsed();

    assert!(passed, "verification failed: {:?}", failures);
    assert!(
        elapsed.as_millis() < 1000,
        "full pipeline(20 tokens) took {}ms, expected < 1000ms",
        elapsed.as_millis()
    );
}
