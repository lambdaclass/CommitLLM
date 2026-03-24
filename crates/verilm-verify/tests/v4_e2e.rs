//! End-to-end tests for V4 retained-state verification.
//!
//! Protocol path: prover commits retained state, opens with shell intermediates,
//! verifier checks with key-only Freivalds + bridge checks. No weights needed.
//!
//! Debug/oracle path: verifier independently replays from public weights.

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{RetainedLayerState, RetainedTokenState, ShellWeights};
use verilm_prover::{commit_minimal, open_v4, open_v4_structural, FullBindingParams};
use verilm_test_vectors::{forward_pass, generate_key, generate_model, LayerWeights};
use verilm_verify::{verify_v4, verify_v4_with_weights, Verdict};

/// Adapter: exposes toy model LayerWeights as ShellWeights.
struct ToyWeights<'a>(&'a [LayerWeights]);

impl ShellWeights for ToyWeights<'_> {
    fn weight(&self, layer: usize, mt: MatrixType) -> &[i8] {
        let lw = &self.0[layer];
        match mt {
            MatrixType::Wq => &lw.wq,
            MatrixType::Wk => &lw.wk,
            MatrixType::Wv => &lw.wv,
            MatrixType::Wo => &lw.wo,
            MatrixType::Wg => &lw.wg,
            MatrixType::Wu => &lw.wu,
            MatrixType::Wd => &lw.wd,
        }
    }
}

fn setup() -> (ModelConfig, Vec<LayerWeights>, verilm_core::types::VerifierKey) {
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
// Protocol path: key-only Freivalds on prover-supplied shell openings
// ---------------------------------------------------------------------------

#[test]
fn v4_protocol_single_token_pass() {
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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[]);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Structural (~8) + Freivalds per layer:
    //   Layer 0: 4 (Wo, Wg, Wu, Wd) — no QKV (embedding unknown)
    //   Layer 1+: 7 (Wq, Wk, Wv, Wo, Wg, Wu, Wd) — cross-layer chain known
    assert!(report.checks_run >= 8 + cfg.n_layers * 4);
}

#[test]
fn v4_protocol_multi_token_pass() {
    let (cfg, model, key) = setup();

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

    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &[]);
    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Structural + shell Freivalds for challenged token only
    assert!(report.checks_run >= 8 + cfg.n_layers * 4);
}

#[test]
fn v4_protocol_token_zero_pass() {
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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[]);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    assert_eq!(response.prefix_retained.len(), 0);
}

// ---------------------------------------------------------------------------
// Tamper detection (protocol path)
// ---------------------------------------------------------------------------

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

    let mut response = open_v4(&state, 1, &ToyWeights(&model), &cfg, &[]);
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
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[]);
    response.revealed_seed[0] ^= 0xff;

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("seed")),
        "expected seed failure, got: {:?}",
        report.failures
    );
}

#[test]
fn v4_wrong_shell_opening_detected() {
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

    // Prover opens with WRONG weights — shell intermediates are inconsistent
    // with the keygen weights. Freivalds catches this.
    let wrong_model = generate_model(&cfg, 99999);
    let response = open_v4(&state, 0, &ToyWeights(&wrong_model), &cfg, &[]);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("Freivalds")),
        "expected Freivalds failure on shell opening, got: {:?}",
        report.failures
    );
}

#[test]
fn v4_missing_shell_opening_detected() {
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
    // Structural-only open: no shell opening
    let response = open_v4_structural(&state, 0);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("shell_opening")),
        "expected missing shell_opening failure, got: {:?}",
        report.failures
    );
}

// ---------------------------------------------------------------------------
// Debug/oracle: verifier-side weight-backed replay
// ---------------------------------------------------------------------------

#[test]
fn v4_weights_single_token_pass() {
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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[]);

    let report = verify_v4_with_weights(&key, &response, &ToyWeights(&model));
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Structural + debug replay per layer:
    //   Layer 0: 4 Freivalds (Wo, Wg, Wu, Wd) — no QKV (embedding unknown)
    //   Layer 1+: 7 Freivalds (Wq, Wk, Wv, Wo, Wg, Wu, Wd)
    assert!(report.checks_run >= 8 + cfg.n_layers * 4);
}

#[test]
fn v4_weights_multi_token_pass() {
    let (cfg, model, key) = setup();

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

    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &[]);
    let report = verify_v4_with_weights(&key, &response, &ToyWeights(&model));
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // 3 independent token replays (debug path replays all prefix + challenged)
    assert!(report.checks_run > 20);
}

#[test]
fn v4_weights_wrong_weights_detected() {
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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[]);

    // Debug verifier replays with WRONG weights — Freivalds catches mismatch.
    let wrong_model = generate_model(&cfg, 99999);
    let report = verify_v4_with_weights(&key, &response, &ToyWeights(&wrong_model));
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.contains("Freivalds")),
        "expected Freivalds weight-binding failure, got: {:?}",
        report.failures
    );
}

// ---------------------------------------------------------------------------
// Scale-aware bridge: prover and verifier agree with non-trivial scales
// ---------------------------------------------------------------------------

/// Build a key with synthetic weight_scales to exercise scale-aware bridges.
fn setup_with_scales() -> (
    ModelConfig,
    Vec<LayerWeights>,
    verilm_core::types::VerifierKey,
    Vec<Vec<f32>>,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);

    // Synthetic weight scales: non-zero → triggers scale-aware path.
    let n_mt = verilm_core::constants::MatrixType::ALL.len();
    let weight_scales: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..n_mt).map(|m| 0.01 + 0.001 * (l * n_mt + m) as f32).collect())
        .collect();

    key.weight_scales = weight_scales.clone();
    (cfg, model, key, weight_scales)
}

/// Build RetainedTokenState with non-trivial activation scales.
fn retained_with_scales(traces: &[verilm_core::types::LayerTrace]) -> RetainedTokenState {
    RetainedTokenState {
        layers: traces
            .iter()
            .enumerate()
            .map(|(i, lt)| RetainedLayerState {
                a: lt.a.clone(),
                scale_a: 0.5 + 0.1 * i as f32,
                scale_x_attn: 0.3 + 0.05 * i as f32,
                scale_x_ffn: 0.4 + 0.07 * i as f32,
                scale_h: 0.6 + 0.03 * i as f32,
            })
            .collect(),
    }
}

#[test]
fn v4_scale_aware_single_token_pass() {
    let (cfg, model, key, ws) = setup_with_scales();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_with_scales(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_scale_aware_multi_token_pass() {
    let (cfg, model, key, ws) = setup_with_scales();
    let inputs: Vec<Vec<i8>> = (0..3)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();

    let all_retained: Vec<RetainedTokenState> = inputs
        .iter()
        .map(|inp| retained_with_scales(&forward_pass(&cfg, &model, inp)))
        .collect();

    let params = FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"multi token scaled",
        sampling_seed: [99u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params);
    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &ws);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_scale_mismatch_detected() {
    // Prover uses different weight_scales than verifier → Freivalds still
    // binds to the key's precomputed v, but the bridge-derived intermediates
    // (x_ffn, h) won't match → Freivalds on W_g/W_u/W_d should fail.
    let (cfg, model, key, _ws) = setup_with_scales();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_with_scales(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);

    // Prover uses WRONG scales (all 1.0) while verifier has non-trivial scales.
    let wrong_ws: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![1.0; verilm_core::constants::MatrixType::ALL.len()])
        .collect();
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &wrong_ws);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "scale mismatch should cause failure");
    assert!(
        report.failures.iter().any(|f| f.contains("Freivalds")),
        "expected Freivalds failure from scale mismatch, got: {:?}",
        report.failures
    );
}
