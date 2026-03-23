//! End-to-end tests for V4 retained-state verification.
//!
//! Flow: toy model forward pass → extract retained state → commit_minimal →
//! open_v4 → verify_v4, exercising both structural checks and shell replay.

use verilm_core::constants::ModelConfig;
use verilm_core::types::{RetainedLayerState, RetainedTokenState};
use verilm_prover::{commit_minimal, open_v4, FullBindingParams};
use verilm_test_vectors::{forward_pass, generate_key, generate_model};
use verilm_verify::{verify_v4, Verdict};

fn setup() -> (
    ModelConfig,
    Vec<verilm_test_vectors::LayerWeights>,
    verilm_core::types::VerifierKey,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    (cfg, model, key)
}

/// Extract RetainedTokenState from a full LayerTrace (toy model: unit scales).
fn retained_from_traces(traces: &[verilm_core::types::LayerTrace]) -> RetainedTokenState {
    RetainedTokenState {
        layers: traces
            .iter()
            .map(|lt| RetainedLayerState {
                a: lt.a.clone(),
                scale_a: lt.scale_a.unwrap_or(1.0),
                scale_x_attn: lt.scale_x_attn.unwrap_or(1.0),
                scale_x_ffn: lt.scale_x_ffn.unwrap_or(1.0),
                scale_h: lt.scale_h.unwrap_or(1.0),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Structural-only verification (no replayed_layers)
// ---------------------------------------------------------------------------

#[test]
fn v4_structural_single_token_pass() {
    let (cfg, model, _key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0);

    // Structural-only: no replayed_layers
    assert!(response.replayed_layers.is_none());

    let key = generate_key(&cfg, &model, [1u8; 32]);
    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    assert!(report.checks_run >= 5, "expected at least 5 structural checks");
}

#[test]
fn v4_structural_multi_token_pass() {
    let (cfg, model, key) = setup();

    // 3 tokens with different inputs
    let inputs: Vec<Vec<i8>> = (0..3)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();

    let all_retained: Vec<RetainedTokenState> = inputs
        .iter()
        .map(|inp| retained_from_traces(&forward_pass(&cfg, &model, inp)))
        .collect();

    let params = FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"multi token test",
        sampling_seed: [99u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params);

    // Challenge token 2 (has 2 prefix tokens)
    let response = open_v4(&state, 2);
    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // 7 structural + 2 prefix Merkle proofs
    assert!(report.checks_run >= 9);
}

#[test]
fn v4_structural_token_zero_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[1],
        prompt: b"token zero",
        sampling_seed: [3u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    assert_eq!(response.prefix_retained.len(), 0);
}

// ---------------------------------------------------------------------------
// Full verification with shell replay (replayed_layers)
// ---------------------------------------------------------------------------

#[test]
fn v4_full_single_token_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let mut response = open_v4(&state, 0);

    // Attach replayed layers (the honest computation)
    response.replayed_layers = Some(traces);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Structural (7+) + per-layer (1 binding + 7 Freivalds + 1 SiLU + 1 chain + cross-layer) × n_layers
    let expected_min = 7 + cfg.n_layers * 10;
    assert!(
        report.checks_run >= expected_min,
        "expected >= {} checks, got {}",
        expected_min,
        report.checks_run
    );
}

#[test]
fn v4_full_multi_token_pass() {
    let (cfg, model, key) = setup();

    let inputs: Vec<Vec<i8>> = (0..3)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();

    let all_traces: Vec<Vec<verilm_core::types::LayerTrace>> = inputs
        .iter()
        .map(|inp| forward_pass(&cfg, &model, inp))
        .collect();

    let all_retained: Vec<RetainedTokenState> = all_traces
        .iter()
        .map(|t| retained_from_traces(t))
        .collect();

    let params = FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"multi token test",
        sampling_seed: [99u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params);

    // Challenge token 1 (middle), attach its replayed layers
    let mut response = open_v4(&state, 1);
    response.replayed_layers = Some(all_traces[1].clone());

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Tamper detection
// ---------------------------------------------------------------------------

#[test]
fn v4_tampered_retained_a_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let mut retained = retained_from_traces(&traces);

    // Tamper: flip a byte in retained a
    retained.layers[0].a[0] ^= 0x7f;

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let mut response = open_v4(&state, 0);
    response.replayed_layers = Some(traces);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    // Should catch: retained binding mismatch (replayed a != retained a)
    assert!(
        report.failures.iter().any(|f| f.contains("retained a")),
        "expected retained a mismatch, got: {:?}",
        report.failures
    );
}

#[test]
fn v4_tampered_replayed_freivalds_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let mut response = open_v4(&state, 0);

    // Tamper: corrupt an accumulator in the replayed layers
    let mut tampered_traces = traces;
    tampered_traces[0].attn_out[0] += 1;
    response.replayed_layers = Some(tampered_traces);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("Freivalds") || f.contains("chain")),
        "expected Freivalds or chain failure, got: {:?}",
        report.failures
    );
}

#[test]
fn v4_tampered_io_chain_detected() {
    let (cfg, model, key) = setup();

    let inputs: Vec<Vec<i8>> = (0..2)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();

    let all_retained: Vec<RetainedTokenState> = inputs
        .iter()
        .map(|inp| retained_from_traces(&forward_pass(&cfg, &model, inp)))
        .collect();

    let params = FullBindingParams {
        token_ids: &[10, 20],
        prompt: b"io chain test",
        sampling_seed: [55u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params);

    let mut response = open_v4(&state, 1);
    // Tamper: change prev_io_hash
    response.prev_io_hash[0] ^= 0xff;

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("prev_io_hash") || f.contains("IO chain")),
        "expected IO chain failure, got: {:?}",
        report.failures
    );
}

#[test]
fn v4_wrong_seed_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let mut response = open_v4(&state, 0);

    // Tamper: change revealed seed
    response.revealed_seed[0] ^= 0xff;

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("seed")),
        "expected seed failure, got: {:?}",
        report.failures
    );
}
