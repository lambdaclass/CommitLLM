//! End-to-end tests for V4 retained-state verification.
//!
//! Protocol path: prover commits retained state, opens with shell intermediates,
//! verifier checks with key-only Freivalds + bridge checks. No weights needed.
//!
//! Debug/oracle path: verifier independently replays from public weights.

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{BridgeParams, DeploymentManifest, EmbeddingLookup, RetainedLayerState, RetainedTokenState, ShellWeights};
use verilm_prover::{commit_minimal, open_v4, open_v4_structural, CapturedLayerScales, FullBindingParams};
use verilm_test_vectors::{forward_pass, generate_key, generate_model, LayerWeights};
use verilm_core::types::InputSpec;
use verilm_verify::{verify_v4_legacy, verify_v4_with_weights, Detokenizer, PromptTokenizer, Verdict};

/// Thin wrapper: routes through the legacy verifier while the canonical path
/// is validated. Will be removed once test vectors are upgraded to canonical-grade.
fn verify_v4(
    key: &verilm_core::types::VerifierKey,
    response: &verilm_core::types::V4AuditResponse,
    expected_prompt_token_ids: Option<&[u32]>,
) -> verilm_verify::V4VerifyReport {
    verify_v4_legacy(key, response, expected_prompt_token_ids, None, None)
}

fn verify_v4_full(
    key: &verilm_core::types::VerifierKey,
    response: &verilm_core::types::V4AuditResponse,
    expected_prompt_token_ids: Option<&[u32]>,
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> verilm_verify::V4VerifyReport {
    verify_v4_legacy(key, response, expected_prompt_token_ids, tokenizer, detokenizer)
}

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
            MatrixType::LmHead => panic!("ToyWeights: LmHead is not a per-layer weight"),
        }
    }
}

/// Attach prover's logits_i32 claim to a V4 response (toy model, no RMSNorm).
///
/// In the real protocol, `open_v4` computes this via `TailParams`. For toy models
/// (no final_norm), we compute `lm_head @ clamp(last_ffn_out)` directly.
fn attach_toy_logits(
    response: &mut verilm_core::types::V4AuditResponse,
    lm_head: &[i8],
    cfg: &ModelConfig,
) {
    let shell = response.shell_opening.as_mut().unwrap();
    let last_layer = shell.layers.last().unwrap();
    let fh: Vec<i8> = last_layer.ffn_out.iter().map(|&v| v.clamp(-128, 127) as i8).collect();
    let logits_i32: Vec<i32> = (0..cfg.vocab_size)
        .map(|row| {
            (0..cfg.hidden_dim)
                .map(|c| lm_head[row * cfg.hidden_dim + c] as i32 * fh[c] as i32)
                .sum::<i32>()
        })
        .collect();
    shell.logits_i32 = Some(logits_i32);
}

/// Toy-model setup: no bridge, no residual, no RMSNorm.
/// Only valid for toy-model tests — NOT for production W8A8 verification.
/// For canonical bridge tests, use `setup_full_bridge()`.
fn setup() -> (ModelConfig, Vec<LayerWeights>, verilm_core::types::VerifierKey) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    (cfg, model, key)
}

/// Extract RetainedTokenState from a full LayerTrace (toy model: unit scales).
/// Bridge replay scales (scale_x_attn, scale_x_ffn, scale_h) are no longer
/// part of RetainedLayerState — use `scales_from_traces` for those.
fn retained_from_traces(traces: &[verilm_core::types::LayerTrace]) -> RetainedTokenState {
    RetainedTokenState {
        layers: traces
            .iter()
            .map(|lt| RetainedLayerState {
                a: lt.a.clone(),
                scale_a: lt.scale_a.unwrap_or(1.0),
                x_attn_i8: None,
                scale_x_attn: None,
            })
            .collect(),
    }
}

/// Extract per-layer bridge replay scales from traces (companion to retained_from_traces).
#[allow(dead_code)]
fn scales_from_traces(traces: &[verilm_core::types::LayerTrace]) -> Vec<CapturedLayerScales> {
    traces
        .iter()
        .map(|lt| CapturedLayerScales {
            scale_x_attn: lt.scale_x_attn.unwrap_or(1.0),
            scale_x_ffn: lt.scale_x_ffn.unwrap_or(1.0),
            scale_h: lt.scale_h.unwrap_or(1.0),
        })
        .collect()
}

/// Unit scales for toy model tests (all 1.0).
fn unit_scales(n_layers: usize) -> Vec<CapturedLayerScales> {
    vec![CapturedLayerScales { scale_x_attn: 1.0, scale_x_ffn: 1.0, scale_h: 1.0 }; n_layers]
}

/// Wrapper around `commit_minimal` that automatically provides unit scales
/// for every token. Use only in toy-model tests where all scales are 1.0.
fn commit_toy(
    all_retained: Vec<RetainedTokenState>,
    params: &FullBindingParams,
    final_residuals: Option<Vec<Vec<f32>>>,
    n_layers: usize,
) -> (verilm_core::types::BatchCommitment, verilm_prover::MinimalBatchState) {
    let n_tokens = all_retained.len();
    let scales = vec![unit_scales(n_layers); n_tokens];
    commit_minimal(all_retained, params, final_residuals, scales, None)
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(all_retained, &params, None, cfg.n_layers);

    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let report = verify_v4(&key, &response, None);
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    assert_eq!(response.prefix_leaf_hashes.len(), 0);
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(all_retained, &params, None, cfg.n_layers);

    let mut response = open_v4(&state, 1, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    response.prev_io_hash[0] ^= 0xff;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.message.contains("prev_io_hash") || f.message.contains("IO chain")),
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    response.revealed_seed[0] ^= 0xff;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.message.contains("seed")),
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);

    // Prover opens with WRONG weights — shell intermediates are inconsistent
    // with the keygen weights. Freivalds catches this.
    let wrong_model = generate_model(&cfg, 99999);
    let response = open_v4(&state, 0, &ToyWeights(&wrong_model), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.message.contains("Freivalds")),
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    // Structural-only open: no shell opening
    let response = open_v4_structural(&state, 0);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.message.contains("shell_opening")),
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(all_retained, &params, None, cfg.n_layers);

    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let report = verify_v4_with_weights(&key, &response, &ToyWeights(&model));
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Structural checks + challenged token replay only (prefix tokens are
    // verified via compact leaf hashes, not full weight-backed replay).
    assert!(report.checks_run > 10);
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
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Debug verifier replays with WRONG weights — Freivalds catches mismatch.
    let wrong_model = generate_model(&cfg, 99999);
    let report = verify_v4_with_weights(&key, &response, &ToyWeights(&wrong_model));
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.message.contains("Freivalds")),
        "expected Freivalds weight-binding failure, got: {:?}",
        report.failures
    );
}

// ---------------------------------------------------------------------------
// Scale-aware bridge: prover and verifier agree with non-trivial scales
// ---------------------------------------------------------------------------

/// Toy-model setup with non-trivial scales but no residual/RMSNorm.
/// Exercises the simplified `bridge_requantize()` path with scale awareness.
/// For canonical bridge tests, use `setup_full_bridge()`.
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
    let n_mt = verilm_core::constants::MatrixType::PER_LAYER.len();
    let weight_scales: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..n_mt).map(|m| 0.01 + 0.001 * (l * n_mt + m) as f32).collect())
        .collect();

    key.weight_scales = weight_scales.clone();
    (cfg, model, key, weight_scales)
}

/// Build RetainedTokenState with non-trivial activation scales.
/// Returns (retained_state, captured_scales) since bridge scales are no longer
/// part of RetainedLayerState.
fn retained_with_scales(traces: &[verilm_core::types::LayerTrace]) -> (RetainedTokenState, Vec<CapturedLayerScales>) {
    let retained = RetainedTokenState {
        layers: traces
            .iter()
            .enumerate()
            .map(|(i, lt)| RetainedLayerState {
                a: lt.a.clone(),
                scale_a: 0.5 + 0.1 * i as f32,
                x_attn_i8: None,
                scale_x_attn: None,
            })
            .collect(),
    };
    let scales = traces
        .iter()
        .enumerate()
        .map(|(i, _lt)| CapturedLayerScales {
            scale_x_attn: 0.3 + 0.05 * i as f32,
            scale_x_ffn: 0.4 + 0.07 * i as f32,
            scale_h: 0.6 + 0.03 * i as f32,
        })
        .collect();
    (retained, scales)
}

#[test]
fn v4_scale_aware_single_token_pass() {
    let (cfg, model, key, ws) = setup_with_scales();
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let (retained, rscales) = retained_with_scales(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![rscales], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_scale_aware_multi_token_pass() {
    let (cfg, model, key, ws) = setup_with_scales();
    let inputs: Vec<Vec<i8>> = (0..3)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();

    let (all_retained, all_scales): (Vec<_>, Vec<_>) = inputs
        .iter()
        .map(|inp| retained_with_scales(&forward_pass(&cfg, &model, inp)))
        .unzip();

    let params = FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"multi token scaled",
        sampling_seed: [99u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);
    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &ws, None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
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
    let (retained, rscales) = retained_with_scales(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![rscales], None);

    // Prover uses WRONG scales (all 1.0) while verifier has non-trivial scales.
    let wrong_ws: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![1.0; verilm_core::constants::MatrixType::PER_LAYER.len()])
        .collect();
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &wrong_ws, None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "scale mismatch should cause failure");
    assert!(
        report.failures.iter().any(|f| f.message.contains("Freivalds")),
        "expected Freivalds failure from scale mismatch, got: {:?}",
        report.failures
    );
}

// ---------------------------------------------------------------------------
// Canonical bridge: dequant → residual += → RMSNorm → quantize
// This is the only valid path for production W8A8 verification.
// ---------------------------------------------------------------------------

/// Build synthetic RMSNorm weights and initial residual for full-bridge tests.
fn setup_full_bridge() -> (
    ModelConfig,
    Vec<LayerWeights>,
    verilm_core::types::VerifierKey,
    Vec<Vec<f32>>, // weight_scales
    Vec<Vec<f32>>, // rmsnorm_attn_weights
    Vec<Vec<f32>>, // rmsnorm_ffn_weights
    Vec<f32>,      // initial_residual
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);

    let n_mt = MatrixType::PER_LAYER.len();
    let weight_scales: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..n_mt).map(|m| 0.01 + 0.001 * (l * n_mt + m) as f32).collect())
        .collect();

    // Synthetic RMSNorm weights (all positive, like real models)
    let rmsnorm_attn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..cfg.hidden_dim).map(|i| 0.5 + 0.01 * ((l * cfg.hidden_dim + i) % 100) as f32).collect())
        .collect();
    let rmsnorm_ffn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..cfg.hidden_dim).map(|i| 0.6 + 0.01 * ((l * cfg.hidden_dim + i + 37) % 100) as f32).collect())
        .collect();

    // Synthetic initial residual (embedding-like)
    let initial_residual: Vec<f32> = (0..cfg.hidden_dim)
        .map(|i| 0.1 * (i as f32 - cfg.hidden_dim as f32 / 2.0))
        .collect();

    key.weight_scales = weight_scales.clone();
    key.rmsnorm_attn_weights = rmsnorm_attn.clone();
    key.rmsnorm_ffn_weights = rmsnorm_ffn.clone();
    key.rmsnorm_eps = 1e-5;

    (cfg, model, key, weight_scales, rmsnorm_attn, rmsnorm_ffn, initial_residual)
}

/// Run a full-bridge forward pass to produce RetainedTokenState consistent
/// with the bridge computation. This replicates the paper's residual stream.
/// Returns (retained_state, captured_scales) since bridge scales are no longer
/// part of RetainedLayerState.
fn full_bridge_forward(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_residual: &[f32],
    rmsnorm_attn: &[Vec<f32>],
    rmsnorm_ffn: &[Vec<f32>],
    weight_scales: &[Vec<f32>],
    scales: &[(f32, f32, f32, f32)], // per layer: (scale_x_attn, scale_a, scale_x_ffn, scale_h)
    eps: f64,
) -> (RetainedTokenState, Vec<CapturedLayerScales>) {
    use verilm_core::matmul::matmul_i32;
    use verilm_core::rmsnorm::{bridge_residual_rmsnorm, dequant_add_residual, rmsnorm_f64_input, quantize_f64_to_i8};

    let mut residual: Vec<f64> = initial_residual.iter().map(|&v| v as f64).collect();
    let mut layers = Vec::new();
    let mut captured_scales = Vec::new();
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    for (l, lw) in model.iter().enumerate() {
        let (scale_x_attn, scale_a, scale_x_ffn, scale_h) = scales[l];

        let ws = |mt: MatrixType| -> f32 {
            let idx = MatrixType::PER_LAYER.iter().position(|&m| m == mt).unwrap();
            weight_scales[l][idx]
        };

        // x_attn = quantize(RMSNorm_attn(residual), scale_x_attn)
        let normed = rmsnorm_f64_input(&residual, &rmsnorm_attn[l], eps);
        let x_attn = quantize_f64_to_i8(&normed, scale_x_attn as f64);

        // Single-token attention: a = expand(requantize(V))
        let v_acc = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);
        let v_i8 = verilm_core::requantize(&v_acc);
        let mut a = vec![0i8; cfg.hidden_dim];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            let src = kv_head * cfg.d_head;
            let dst = qh * cfg.d_head;
            a[dst..dst + cfg.d_head].copy_from_slice(&v_i8[src..src + cfg.d_head]);
        }

        // Replicate bridge to advance residual for next layer
        let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
        let x_ffn = bridge_residual_rmsnorm(
            &attn_out, ws(MatrixType::Wo), scale_a,
            &mut residual, &rmsnorm_ffn[l], eps, scale_x_ffn,
        );

        let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let h = verilm_core::silu::compute_h_scaled(
            &g, &u, ws(MatrixType::Wg), ws(MatrixType::Wu), scale_x_ffn, scale_h,
        );
        let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);

        if l + 1 < rmsnorm_attn.len() {
            let next_scale = scales.get(l + 1).map(|s| s.0).unwrap_or(1.0);
            bridge_residual_rmsnorm(
                &ffn_out, ws(MatrixType::Wd), scale_h,
                &mut residual, &rmsnorm_attn[l + 1], eps, next_scale,
            );
        } else {
            dequant_add_residual(&ffn_out, ws(MatrixType::Wd), scale_h, &mut residual);
        }

        layers.push(RetainedLayerState { a, scale_a, x_attn_i8: None, scale_x_attn: None });
        captured_scales.push(CapturedLayerScales { scale_x_attn, scale_x_ffn, scale_h });
    }

    (RetainedTokenState { layers }, captured_scales)
}

/// Synthetic per-layer activation scales for full-bridge tests.
fn bridge_scales(cfg: &ModelConfig) -> Vec<(f32, f32, f32, f32)> {
    (0..cfg.n_layers)
        .map(|l| (
            0.3 + 0.05 * l as f32,  // scale_x_attn
            0.5 + 0.1 * l as f32,   // scale_a
            0.4 + 0.07 * l as f32,  // scale_x_ffn
            0.6 + 0.03 * l as f32,  // scale_h
        ))
        .collect()
}

#[test]
fn v4_full_bridge_single_token_pass() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    let proof = verilm_core::merkle::prove(&tree, token_id as usize);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &initial_residual,
        embedding_proof: Some(proof),
    };

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"full bridge",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // 1 embedding check + structural (7 for token 0) + 7 Freivalds per layer
    assert!(report.checks_run >= 8 + cfg.n_layers * 7,
        "expected at least {} checks, got {}",
        8 + cfg.n_layers * 7, report.checks_run);
}

#[test]
fn v4_full_bridge_cross_layer_chain() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);

    // Build embedding table with rows for token_ids 10, 20, 30
    let token_ids = [10u32, 20, 30];
    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    // Build a tree that contains all 3 residuals at their token_id indices
    let n_vocab = 128;
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if let Some(pos) = token_ids.iter().position(|&tid| tid as usize == i) {
            leaves.push(verilm_core::merkle::hash_embedding_row(&residuals[pos]));
        } else {
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001).collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    key.embedding_merkle_root = Some(tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"multi token full bridge",
        sampling_seed: [99u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    // Open token 2 (token_id=30) — its bridge must match
    let proof = verilm_core::merkle::prove(&tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_full_bridge_wrong_residual_detected() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    // Prover computes shell with CORRECT initial_residual + valid proof
    let proof = verilm_core::merkle::prove(&tree, token_id as usize);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &initial_residual,
        embedding_proof: Some(proof),
    };

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"wrong residual",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales.clone()], None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    // Tamper: change initial_residual in the shell opening after prover built it.
    // The embedding proof was computed for the original residual, so hash won't match.
    if let Some(ref mut shell) = response.shell_opening {
        if let Some(ref mut ir) = shell.initial_residual {
            for v in ir.iter_mut() {
                *v += 100.0; // grossly wrong
            }
        }
    }

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "wrong residual should be detected");
    assert!(
        report.failures.iter().any(|f| f.message.contains("embedding Merkle proof")),
        "expected embedding proof failure from tampered residual, got: {:?}",
        report.failures
    );
}

#[test]
fn v4_full_bridge_qkv_layer0() {
    // Verify that full bridge enables QKV checking at layer 0
    // (toy model skips layer 0 QKV because x_attn is unknown)
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    let proof = verilm_core::merkle::prove(&tree, token_id as usize);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &initial_residual,
        embedding_proof: Some(proof),
    };

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"qkv layer0",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales.clone()], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    // Verify shell has QKV at layer 0
    let shell = response.shell_opening.as_ref().unwrap();
    assert!(shell.layers[0].q.is_some(), "full bridge should produce QKV at layer 0");
    assert!(shell.layers[0].k.is_some());
    assert!(shell.layers[0].v.is_some());

    // Verification passes with all 7 checks per layer (QKV at layer 0 included)
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);

    // Full bridge: 1 embedding + 7 structural + 7 Freivalds per layer
    assert!(report.checks_run >= 8 + cfg.n_layers * 7,
        "full bridge should have 7 checks/layer + embedding (got {} total, expected >= {})",
        report.checks_run, 8 + cfg.n_layers * 7);
}

/// Build a toy "embedding table" and Merkle tree for embedding-proof tests.
fn setup_embedding_tree(initial_residual: &[f32], token_id: u32, n_vocab: usize) -> (verilm_core::merkle::MerkleTree, [u8; 32]) {
    // Build embedding table: random rows except row[token_id] = initial_residual
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if i == token_id as usize {
            leaves.push(verilm_core::merkle::hash_embedding_row(initial_residual));
        } else {
            // Dummy rows with deterministic content
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    let root = tree.root;
    (tree, root)
}

#[test]
fn v4_embedding_proof_pass() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    // Build embedding tree and set root in key
    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    let proof = verilm_core::merkle::prove(&tree, token_id as usize);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &initial_residual,
        embedding_proof: Some(proof),
    };

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"embedding proof",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales.clone()], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Should have embedding proof check + structural + bridge checks
    assert!(report.checks_run >= 8 + cfg.n_layers * 7,
        "expected at least {} checks (incl embedding), got {}",
        8 + cfg.n_layers * 7, report.checks_run);
}

#[test]
fn v4_embedding_proof_tampered_residual_detected() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    // Tamper: use wrong initial_residual but correct proof (for the real one)
    let tampered: Vec<f32> = initial_residual.iter().map(|&v| v + 1.0).collect();
    let proof = verilm_core::merkle::prove(&tree, token_id as usize);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &tampered, // wrong residual
        embedding_proof: Some(proof), // proof is for the real residual
    };

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"tampered embedding",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales.clone()], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "tampered residual should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("embedding Merkle proof")),
        "should fail on embedding proof, failures: {:?}", report.failures);
}

#[test]
fn v4_embedding_proof_missing_when_root_present() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);

    let (_tree, root) = setup_embedding_tree(&initial_residual, 42, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    // Bridge with initial_residual but NO embedding_proof
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &initial_residual,
        embedding_proof: None, // missing!
    };

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"missing proof",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales.clone()], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "missing proof should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("missing embedding_proof")),
        "should fail on missing proof, failures: {:?}", report.failures);
}

#[test]
fn v4_embedding_proof_wrong_token_id_detected() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    // Proof for wrong token index (43 instead of 42)
    let wrong_proof = verilm_core::merkle::prove(&tree, 43);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &initial_residual,
        embedding_proof: Some(wrong_proof),
    };

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"wrong token proof",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales.clone()], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "wrong token_id should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("leaf_index") && f.message.contains("token_id")),
        "should fail on token_id mismatch, failures: {:?}", report.failures);
}

#[test]
fn v4_downgrade_omit_initial_residual_detected() {
    // Key has embedding_merkle_root → prover MUST provide initial_residual.
    // Omitting it is a downgrade attack to the simplified bridge path.
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (_, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    // Deliberately pass None for bridge → no initial_residual in the shell opening
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"downgrade",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales.clone()], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "omitted initial_residual should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("initial_residual")),
        "should fail on missing initial_residual, failures: {:?}", report.failures);
}

#[test]
fn v4_unbound_initial_residual_rejected() {
    // Key has RMSNorm weights but NO embedding_merkle_root.
    // Prover supplies initial_residual anyway → must be rejected (unbound vector).
    let (cfg, model, key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    assert!(key.embedding_merkle_root.is_none(), "key should have no embedding root");

    let (retained, captured_scales) = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &initial_residual,
        embedding_proof: None,
    };

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"unbound residual",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales.clone()], None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "unbound initial_residual should be rejected");
    assert!(report.failures.iter().any(|f| f.message.contains("embedding_merkle_root")),
        "should fail on missing embedding root, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// LM head verification (greedy argmax)
// ---------------------------------------------------------------------------

/// Setup with lm_head + Freivalds r/v: generate model with head, derive correct token_id.
fn setup_lm_head() -> (
    ModelConfig,
    Vec<LayerWeights>,
    verilm_core::types::VerifierKey,
    Vec<i8>,  // lm_head weights
    Vec<i8>,  // input
    u32,      // correct token_id (argmax of logits)
) {
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 12345);
    // Use generate_key_level_b_with_head so r_lm_head / v_lm_head are populated.
    let key = verilm_test_vectors::generate_key_level_b_with_head(
        &cfg, &toy.layers, [1u8; 32], Some(toy.lm_head.clone()),
    );

    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let token_id = logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    (cfg, toy.layers, key, toy.lm_head, input, token_id)
}

#[test]
fn v4_lm_head_greedy_pass() {
    let (cfg, model, key, lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"lm_head test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    assert!(report.failures.is_empty());
}

#[test]
fn v4_lm_head_wrong_token_detected() {
    let (cfg, model, key, lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    // Use a wrong token_id (offset by 1)
    let wrong_token = (token_id + 1) % cfg.vocab_size as u32;

    let params = FullBindingParams {
        token_ids: &[wrong_token],
        prompt: b"lm_head tamper test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "wrong token should be detected");
    assert!(report.failures.iter().any(|f| f.message.contains("lm_head")),
        "should fail on lm_head check, failures: {:?}", report.failures);
}

#[test]
fn v4_lm_head_multi_token_pass() {
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 12345);
    let key = verilm_test_vectors::generate_key_level_b_with_head(
        &cfg, &toy.layers, [1u8; 32], Some(toy.lm_head.clone()),
    );

    // Generate 3 tokens, each with correct argmax token_id
    let inputs: Vec<Vec<i8>> = (0..3)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();

    let mut all_retained = Vec::new();
    let mut token_ids = Vec::new();
    for inp in &inputs {
        let traces = forward_pass(&cfg, &toy.layers, inp);
        let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);
        let logits = verilm_core::sampling::recompute_logits(
            &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
        );
        let tid = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        all_retained.push(retained_from_traces(&traces));
        token_ids.push(tid);
    }

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"lm_head multi",
        sampling_seed: [99u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(all_retained, &params, None, cfg.n_layers);

    // Verify each token
    for i in 0..3 {
        let mut response = open_v4(&state, i, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);
        attach_toy_logits(&mut response, &toy.lm_head, &cfg);
        let report = verify_v4(&key, &response, None);
        assert_eq!(report.verdict, Verdict::Pass,
            "token {}: failures: {:?}", i, report.failures);
    }
}

#[test]
fn v4_lm_head_freivalds_catches_tampered_logits() {
    // The prover provides logits_i32. If the prover lies about logits
    // (e.g. computed from a different lm_head), Freivalds catches it.
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 12345);
    let key = verilm_test_vectors::generate_key_level_b_with_head(
        &cfg, &toy.layers, [1u8; 32], Some(toy.lm_head.clone()),
    );

    assert!(!key.r_for(MatrixType::LmHead).is_empty(), "r for LmHead should be generated");
    assert!(key.v_lm_head.is_some(), "v_lm_head should be generated");

    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let token_id = logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    let retained = retained_from_traces(&traces);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"freivalds lm_head test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);

    // Attach honest logits — should pass.
    attach_toy_logits(&mut response, &toy.lm_head, &cfg);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "honest logits should pass: {:?}", report.failures);

    // Now tamper: flip some logits values.
    let shell = response.shell_opening.as_mut().unwrap();
    let logits = shell.logits_i32.as_mut().unwrap();
    logits[0] = logits[0].wrapping_add(1000);
    logits[1] = logits[1].wrapping_sub(500);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "tampered logits should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("Freivalds")),
        "should fail on Freivalds check, got: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Manifest binding + sampling replay
// ---------------------------------------------------------------------------

fn make_manifest(temperature: f32, top_k: u32, top_p: f32) -> DeploymentManifest {
    DeploymentManifest {
        tokenizer_hash: [0u8; 32],
        temperature,
        top_k,
        top_p,
        eos_policy: "stop".into(),
        weight_hash: None,
        quant_hash: None,
        system_prompt_hash: None,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        logit_bias: vec![],
        bad_word_ids: vec![],
        guided_decoding: String::new(),
        stop_sequences: vec![],
        max_tokens: 0,
        chat_template_hash: None,
        rope_config_hash: None,
        rmsnorm_eps: None,
        sampler_version: None,
        bos_eos_policy: None,
        truncation_policy: None,
        special_token_policy: None,
        adapter_hash: None,
        n_layers: None,
        hidden_dim: None,
        vocab_size: None,
        embedding_merkle_root: None,
        quant_family: None,
        scale_derivation: None,
        quant_block_size: None,
        kv_dim: None,
        ffn_dim: None,
        d_head: None,
        n_q_heads: None,
        n_kv_heads: None,
        rope_theta: None,
        min_tokens: 0,
        ignore_eos: false,
        detokenization_policy: None,
        eos_token_id: None,
        padding_policy: None,
        decode_mode: None,
    }
}

#[test]
fn v4_manifest_greedy_sampling_replay_pass() {
    // Greedy manifest (temp=0): sampling replay should agree with argmax.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"manifest greedy",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_manifest_sampled_replay_pass() {
    // Non-greedy: use temperature=1.0, produce correct sampled token.
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [2u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i * 3 % 256) as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);

    let manifest = make_manifest(1.0, 0, 1.0);
    let sampling_seed = [42u8; 32];

    // Compute the correct sampled token using the canonical sampler.
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let dp = verilm_core::sampling::DecodeParams {
        temperature: manifest.temperature,
        top_k: manifest.top_k,
        top_p: manifest.top_p,
    };
    let token_seed = verilm_core::sampling::derive_token_seed(&sampling_seed, 0);
    let token_id = verilm_core::sampling::sample(&logits, &dp, &token_seed);

    let retained = retained_from_traces(&traces);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"manifest sampled",
        sampling_seed,
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_manifest_wrong_sampled_token_detected() {
    // Non-greedy with wrong token → sampling replay should catch it.
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let key = verilm_test_vectors::generate_key_level_b_with_head(
        &cfg, &toy.layers, [2u8; 32], Some(toy.lm_head.clone()),
    );

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i * 3 % 256) as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);

    let manifest = make_manifest(1.0, 0, 1.0);
    let sampling_seed = [42u8; 32];

    // Compute the correct token, then use a different one.
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let dp = verilm_core::sampling::DecodeParams {
        temperature: manifest.temperature,
        top_k: manifest.top_k,
        top_p: manifest.top_p,
    };
    let token_seed = verilm_core::sampling::derive_token_seed(&sampling_seed, 0);
    let correct_token = verilm_core::sampling::sample(&logits, &dp, &token_seed);
    let wrong_token = (correct_token + 1) % cfg.vocab_size as u32;

    let retained = retained_from_traces(&traces);
    let params = FullBindingParams {
        token_ids: &[wrong_token],
        prompt: b"manifest wrong token",
        sampling_seed,
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &toy.lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("lm_head")),
        "should fail on lm_head sampling check, failures: {:?}", report.failures);
}

#[test]
fn v4_manifest_hash_mismatch_detected() {
    // Manifest hash in commitment doesn't match the manifest in response.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"manifest mismatch",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Tamper with the manifest in the response (different temperature).
    response.manifest = Some(make_manifest(1.0, 0, 1.0));

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("manifest hash")),
        "should fail on manifest hash, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Architecture field cross-checks (n_layers, hidden_dim, vocab_size, embedding_merkle_root)
// ---------------------------------------------------------------------------

/// Helper: commit + open a manifest-bound response for architecture cross-check tests.
fn setup_manifest_crosscheck() -> (
    verilm_core::constants::ModelConfig,
    Vec<LayerWeights>,
    verilm_core::types::VerifierKey,
    DeploymentManifest,
    verilm_core::types::V4AuditResponse,
) {
    let cfg = ModelConfig::toy();
    let model = verilm_test_vectors::generate_model(&cfg, 12345);
    let key = verilm_test_vectors::generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    (cfg, model, key, manifest, response)
}

#[test]
fn v4_manifest_n_layers_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    // Set wrong n_layers in manifest
    manifest.n_layers = Some(999);
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("n_layers mismatch")),
        "expected n_layers mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_hidden_dim_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.hidden_dim = Some(9999);
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("hidden_dim mismatch")),
        "expected hidden_dim mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_vocab_size_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.vocab_size = Some(99999);
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("vocab_size mismatch")),
        "expected vocab_size mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_embedding_root_mismatch_rejected() {
    let (_cfg, _model, mut key, mut manifest, _) = setup_manifest_crosscheck();
    // Both sides must have embedding_merkle_root for the cross-check to fire.
    key.embedding_merkle_root = Some([0xAA; 32]);
    manifest.embedding_merkle_root = Some([0xBB; 32]);
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("embedding_merkle_root mismatch")),
        "expected embedding_merkle_root mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_architecture_fields_pass() {
    let (cfg, model, key, mut manifest, _) = setup_manifest_crosscheck();
    // Set correct architecture values matching the key config.
    manifest.n_layers = Some(cfg.n_layers as u32);
    manifest.hidden_dim = Some(cfg.hidden_dim as u32);
    manifest.vocab_size = Some(cfg.vocab_size as u32);
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Should have at least the 3 extra architecture checks (n_layers, hidden_dim, vocab_size).
    assert!(report.checks_run >= 11, "expected extra architecture checks, got {}", report.checks_run);
}

// ---------------------------------------------------------------------------
// Decode mode cross-checks (#9)
// ---------------------------------------------------------------------------

#[test]
fn v4_decode_mode_greedy_with_nonzero_temp_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.decode_mode = Some("greedy".into());
    manifest.temperature = 1.0; // inconsistent: greedy but temp != 0
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"decode mode check",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("decode_mode='greedy'")),
        "expected decode_mode/temperature mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_decode_mode_sampled_with_zero_temp_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.decode_mode = Some("sampled".into());
    manifest.temperature = 0.0; // inconsistent: sampled but temp = 0
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"decode mode check",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("decode_mode='sampled'")),
        "expected decode_mode/temperature mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_decode_mode_unknown_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.decode_mode = Some("nucleus".into()); // unsupported mode
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"decode mode check",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("unsupported decode_mode")),
        "expected unsupported decode_mode, got: {:?}", report.failures);
}

#[test]
fn v4_decode_mode_greedy_consistent_pass() {
    let (cfg, model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.decode_mode = Some("greedy".into());
    manifest.temperature = 0.0; // consistent
    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"decode mode check",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Quantization cross-checks (#5-7: quant_family, scale_derivation, quant_block_size)
// ---------------------------------------------------------------------------

#[test]
fn v4_manifest_quant_family_mismatch_rejected() {
    let (_cfg, _model, mut key, mut manifest, _) = setup_manifest_crosscheck();
    key.quant_family = Some("W8A8".into());
    manifest.quant_family = Some("GPTQ".into());
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"quant crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("quant_family mismatch")),
        "expected quant_family mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_scale_derivation_mismatch_rejected() {
    let (_cfg, _model, mut key, mut manifest, _) = setup_manifest_crosscheck();
    key.scale_derivation = Some("absmax".into());
    manifest.scale_derivation = Some("zeropoint".into());
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"quant crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("scale_derivation mismatch")),
        "expected scale_derivation mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_quant_block_size_mismatch_rejected() {
    let (_cfg, _model, mut key, mut manifest, _) = setup_manifest_crosscheck();
    key.quant_block_size = Some(32);
    manifest.quant_block_size = Some(128);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"quant crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("quant_block_size mismatch")),
        "expected quant_block_size mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_quant_fields_consistent_pass() {
    let (_cfg, _model, mut key, mut manifest, _) = setup_manifest_crosscheck();
    // Both sides agree on quant identity.
    key.quant_family = Some("W8A8".into());
    key.scale_derivation = Some("absmax".into());
    key.quant_block_size = Some(32);
    manifest.quant_family = Some("W8A8".into());
    manifest.scale_derivation = Some("absmax".into());
    manifest.quant_block_size = Some(32);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"quant crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Extended architecture cross-checks (#8: kv_dim, ffn_dim, d_head, n_q/kv_heads, rope_theta)
// ---------------------------------------------------------------------------

#[test]
fn v4_manifest_kv_dim_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.kv_dim = Some(9999);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("kv_dim mismatch")),
        "expected kv_dim mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_ffn_dim_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.ffn_dim = Some(9999);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("ffn_dim mismatch")),
        "expected ffn_dim mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_d_head_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.d_head = Some(9999);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("d_head mismatch")),
        "expected d_head mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_n_q_heads_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.n_q_heads = Some(9999);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("n_q_heads mismatch")),
        "expected n_q_heads mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_n_kv_heads_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.n_kv_heads = Some(9999);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("n_kv_heads mismatch")),
        "expected n_kv_heads mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rope_theta_mismatch_rejected() {
    let (_cfg, _model, key, mut manifest, _) = setup_manifest_crosscheck();
    manifest.rope_theta = Some(99999.0);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0.._cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&_cfg, &_model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, _cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&_model), &_cfg, &[], None, None, None, None, false);
    response.manifest = Some(manifest);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("rope_theta mismatch")),
        "expected rope_theta mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_full_architecture_pass() {
    let (cfg, model, key, mut manifest, _) = setup_manifest_crosscheck();
    // Set all architecture fields to correct values.
    manifest.n_layers = Some(cfg.n_layers as u32);
    manifest.hidden_dim = Some(cfg.hidden_dim as u32);
    manifest.vocab_size = Some(cfg.vocab_size as u32);
    manifest.kv_dim = Some(cfg.kv_dim as u32);
    manifest.ffn_dim = Some(cfg.ffn_dim as u32);
    manifest.d_head = Some(cfg.d_head as u32);
    manifest.n_q_heads = Some(cfg.n_q_heads as u32);
    manifest.n_kv_heads = Some(cfg.n_kv_heads as u32);
    manifest.rope_theta = Some(cfg.rope_theta);
    let params = FullBindingParams {
        token_ids: &[42], prompt: b"arch crosscheck",
        sampling_seed: [7u8; 32], manifest: Some(&manifest), n_prompt_tokens: Some(1),
    };
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // 3 original + 4 arch + 6 new arch + 1 manifest + 1 sampler + 1 decode params = many checks
    assert!(report.checks_run >= 17, "expected full architecture checks, got {}", report.checks_run);
}

// ---------------------------------------------------------------------------
// Captured final_residual: exact LM-head verification from GPU boundary state
// ---------------------------------------------------------------------------

/// Setup for captured final_residual tests: model with lm_head + final_norm_weights.
/// Returns the correct token_id derived from the captured residual (not shell replay).
/// Attach prover's logits_i32 computed via the RMSNorm tail path.
fn attach_tail_logits(
    response: &mut verilm_core::types::V4AuditResponse,
    lm_head: &[i8],
    final_norm_weights: &[f32],
    cfg: &ModelConfig,
) {
    let shell = response.shell_opening.as_mut().unwrap();
    let fr = shell.final_residual.as_ref().expect("final_residual required for tail logits");
    let res_f64: Vec<f64> = fr.iter().map(|&v| v as f64).collect();
    let normed = verilm_core::rmsnorm::rmsnorm_f64_input(&res_f64, final_norm_weights, 1e-5);
    let fh: Vec<i8> = normed.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect();
    let logits_i32: Vec<i32> = (0..cfg.vocab_size)
        .map(|row| {
            (0..cfg.hidden_dim)
                .map(|c| lm_head[row * cfg.hidden_dim + c] as i32 * fh[c] as i32)
                .sum::<i32>()
        })
        .collect();
    shell.logits_i32 = Some(logits_i32);
}

fn setup_final_residual() -> (
    ModelConfig,
    Vec<LayerWeights>,
    verilm_core::types::VerifierKey,
    Vec<f32>,  // final_residual (pre-final-norm)
    Vec<i8>,   // input
    u32,       // correct token_id from captured path
) {
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let mut key = verilm_test_vectors::generate_key_level_b_with_head(
        &cfg, &toy.layers, [2u8; 32], Some(toy.lm_head.clone()),
    );

    // Set final_norm_weights (all ones = identity-ish RMSNorm for simplicity).
    let final_norm_weights: Vec<f32> = vec![1.0; cfg.hidden_dim];
    key.final_norm_weights = Some(final_norm_weights.clone());

    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);

    // Compute the "captured" final_residual: in the toy model with unit scales,
    // this is just requantize(ffn_out) cast to f32 (no residual tracking).
    let final_residual: Vec<f32> = verilm_core::requantize(&traces.last().unwrap().ffn_out)
        .iter().map(|&v| v as f32).collect();

    // Derive correct token_id via the captured path:
    // RMSNorm(final_residual) → quantize → lm_head matmul → argmax
    let res_f64: Vec<f64> = final_residual.iter().map(|&v| v as f64).collect();
    let normed = verilm_core::rmsnorm::rmsnorm_f64_input(&res_f64, &final_norm_weights, 1e-5);
    let final_hidden: Vec<i8> = normed.iter()
        .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
        .collect();
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let token_id = logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    (cfg, toy.layers, key, final_residual, input, token_id)
}

#[test]
fn v4_captured_final_residual_pass() {
    let (cfg, model, key, final_residual, input, token_id) = setup_final_residual();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"captured residual test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    // Commit with final_residuals so open_v4 attaches it to the shell.
    let (_commitment, state) = commit_toy(
        vec![retained], &params,
        Some(vec![final_residual]),
        cfg.n_layers,
    );
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Verify: lm_head check uses captured final_residual, not shell replay.
    assert!(response.shell_opening.as_ref().unwrap().final_residual.is_some(),
        "final_residual should be set on shell opening");
    let fnw = key.final_norm_weights.as_ref().unwrap();
    let lm = key.lm_head.as_ref().unwrap();
    attach_tail_logits(&mut response, lm, fnw, &cfg);
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_captured_final_residual_wrong_token_detected() {
    let (cfg, model, key, final_residual, input, token_id) = setup_final_residual();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    // Use wrong token_id.
    let wrong_token = (token_id + 1) % cfg.vocab_size as u32;

    let params = FullBindingParams {
        token_ids: &[wrong_token],
        prompt: b"captured residual tamper",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(
        vec![retained], &params,
        Some(vec![final_residual]),
        cfg.n_layers,
    );
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let fnw = key.final_norm_weights.as_ref().unwrap();
    let lm = key.lm_head.as_ref().unwrap();
    attach_tail_logits(&mut response, lm, fnw, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "wrong token should be detected");
    assert!(report.failures.iter().any(|f| f.message.contains("lm_head")),
        "should fail on lm_head check, failures: {:?}", report.failures);
}

#[test]
fn v4_captured_final_residual_tampered_residual_detected() {
    let (cfg, model, key, mut final_residual, input, token_id) = setup_final_residual();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    // Tamper: zero out most elements, spike one. This forces a completely different
    // final_hidden pattern after RMSNorm → different argmax.
    for v in final_residual.iter_mut() {
        *v = 0.0;
    }
    final_residual[0] = 127.0;

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"captured residual tampered",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(
        vec![retained], &params,
        Some(vec![final_residual]),
        cfg.n_layers,
    );
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let fnw = key.final_norm_weights.as_ref().unwrap();
    let lm = key.lm_head.as_ref().unwrap();
    attach_tail_logits(&mut response, lm, fnw, &cfg);

    let report = verify_v4(&key, &response, None);
    // Tampered residual → different final_hidden → different argmax → lm_head mismatch.
    assert_eq!(report.verdict, Verdict::Fail,
        "tampered residual should be detected, failures: {:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.message.contains("lm_head")),
        "should fail on lm_head check, failures: {:?}", report.failures);
}

#[test]
fn v4_final_residual_fail_closed_missing() {
    // When key has lm_head + final_norm_weights, missing final_residual must reject.
    let (cfg, model, key, _final_residual, input, token_id) = setup_final_residual();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"fail closed test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    // Commit WITHOUT final_residuals — shell.final_residual will be None.
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    assert!(response.shell_opening.as_ref().unwrap().final_residual.is_none());
    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail,
        "should fail when final_residual missing but key requires it, failures: {:?}",
        report.failures);
    assert!(report.failures.iter().any(|f| f.message.contains("final_residual")),
        "should mention final_residual in failure, failures: {:?}", report.failures);
}

#[test]
fn v4_lm_head_fail_closed_missing_logits() {
    // When key has LmHead Freivalds vectors, missing logits_i32 must reject.
    let (cfg, model, key, final_residual, input, token_id) = setup_final_residual();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"fail closed logits",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(
        vec![retained], &params,
        Some(vec![final_residual]),
        cfg.n_layers,
    );
    // open_v4 without tail → logits_i32 = None
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    assert!(response.shell_opening.as_ref().unwrap().logits_i32.is_none());

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail,
        "should fail when logits_i32 missing but key has LmHead Freivalds, failures: {:?}",
        report.failures);
    assert!(report.failures.iter().any(|f| f.message.contains("logits_i32")),
        "should mention logits_i32 in failure, failures: {:?}", report.failures);
}

#[test]
fn v4_final_residual_commitment_binding() {
    // Swapping final_residual after commitment must break the Merkle proof.
    let (cfg, model, key, final_residual, input, token_id) = setup_final_residual();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"binding test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(
        vec![retained], &params,
        Some(vec![final_residual]),
        cfg.n_layers,
    );
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Tamper: swap the final_residual in the response AFTER commitment.
    // This should break the Merkle proof because the leaf hash changed.
    let mut tampered = response.shell_opening.as_ref().unwrap().final_residual.clone().unwrap();
    for v in &mut tampered { *v = 0.0; }
    tampered[0] = 127.0;
    response.shell_opening.as_mut().unwrap().final_residual = Some(tampered);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail,
        "post-commitment swap of final_residual must be detected, failures: {:?}",
        report.failures);
    assert!(report.failures.iter().any(|f| f.message.contains("Merkle proof failed")),
        "should fail on Merkle proof, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Manifest: verifier rejects unsupported logit-modifying parameters
// ---------------------------------------------------------------------------

/// Helper: commit + open with a manifest, return the verification report.
fn verify_with_manifest(manifest: DeploymentManifest) -> verilm_verify::V4VerifyReport {
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [2u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i * 3 % 256) as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);

    // Compute the correct greedy token so the only failure is the manifest field.
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let token_id = logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0 as u32;

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"manifest rejection test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained_from_traces(&traces)], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);

    verify_v4(&key, &response, None)
}

#[test]
fn v4_manifest_rejects_repetition_penalty() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.repetition_penalty = 1.2;
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("repetition_penalty")),
        "expected repetition_penalty rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_frequency_penalty() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.frequency_penalty = 0.5;
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("frequency_penalty")),
        "expected frequency_penalty rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_presence_penalty() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.presence_penalty = 0.3;
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("presence_penalty")),
        "expected presence_penalty rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_logit_bias() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.logit_bias = vec![(42, 5.0), (100, -10.0)];
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("logit_bias")),
        "expected logit_bias rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_bad_word_ids() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.bad_word_ids = vec![42, 100, 200];
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("bad_word_ids")),
        "expected bad_word_ids rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_guided_decoding() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.guided_decoding = "json_schema".into();
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("guided_decoding")),
        "expected guided_decoding rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_stop_sequences() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.stop_sequences = vec!["<|end|>".into(), "STOP".into()];
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("stop_sequences")),
        "expected stop_sequences rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_exceeded_max_tokens() {
    // Commit 2 tokens, open token_index 1 with max_tokens=1.
    // token_index 1 >= max_tokens 1 → should fail.
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [2u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i * 3 % 256) as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let retained = retained_from_traces(&traces);

    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let token_id = logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0 as u32;

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.max_tokens = 1; // only token_index 0 is valid

    let params = FullBindingParams {
        token_ids: &[token_id, token_id], // 2 tokens
        prompt: b"max_tokens test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(
        vec![retained.clone(), retained], &params, None,
        cfg.n_layers,
    );
    // Open token_index 1 — exceeds max_tokens=1
    let response = open_v4(&state, 1, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("max_tokens")),
        "expected max_tokens rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_overlong_transcript() {
    // Commit 3 tokens with max_tokens=2, but open token_index 0 (valid index).
    // The per-token check passes (0 < 2), but committed n_tokens=3 > max_tokens=2.
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [2u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i * 3 % 256) as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let retained = retained_from_traces(&traces);

    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let token_id = logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0 as u32;

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.max_tokens = 2; // allow 2 generated tokens

    let params = FullBindingParams {
        // 4 committed token_ids, n_prompt=1 → n_generated = 4 - (1-1) = 4 > max_tokens=2
        token_ids: &[token_id, token_id, token_id, token_id],
        prompt: b"overlong transcript",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(
        vec![retained.clone(), retained.clone(), retained.clone(), retained], &params, None,
        cfg.n_layers,
    );
    // Open token_index 0 — valid per-token, but generation count exceeds max_tokens
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("generated") && f.message.contains("max_tokens")),
        "expected generation-length max_tokens rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_max_tokens_zero_means_unlimited() {
    // max_tokens=0 means no limit — should pass regardless of token_index.
    let m = make_manifest(0.0, 0, 1.0);
    assert_eq!(m.max_tokens, 0);
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Pass,
        "max_tokens=0 (unlimited) should pass, failures: {:?}", report.failures);
}

#[test]
fn v4_manifest_accepts_all_defaults() {
    // All logit-modifying fields at their canonical defaults → should pass.
    let m = make_manifest(0.0, 0, 1.0);
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Pass,
        "canonical defaults should pass, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Four-spec fail-closed: verifier rejects if spec hashes are missing
// ---------------------------------------------------------------------------

#[test]
fn v4_manifest_rejects_missing_spec_hashes() {
    // Verify that the verifier fails when commitment spec hashes are absent.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"missing spec hashes",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    // Wipe all four spec hashes from the commitment — verifier should fail-closed.
    response.commitment.input_spec_hash = None;
    response.commitment.model_spec_hash = None;
    response.commitment.decode_spec_hash = None;
    response.commitment.output_spec_hash = None;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("missing input_spec_hash")),
        "should fail on missing input_spec_hash, failures: {:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.message.contains("missing model_spec_hash")),
        "should fail on missing model_spec_hash, failures: {:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.message.contains("missing decode_spec_hash")),
        "should fail on missing decode_spec_hash, failures: {:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.message.contains("missing output_spec_hash")),
        "should fail on missing output_spec_hash, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Spec-vs-key consistency: rmsnorm_eps, rope_config_hash, sampler_version
// ---------------------------------------------------------------------------

#[test]
fn v4_rmsnorm_eps_mismatch_detected() {
    // Manifest commits rmsnorm_eps=1e-6, key has rmsnorm_eps=1e-5 → mismatch.
    let mut m = make_manifest(0.0, 0, 1.0);
    m.rmsnorm_eps = Some(1e-6); // key will have 1e-5

    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("rmsnorm_eps mismatch")),
        "expected rmsnorm_eps mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_rmsnorm_eps_matching_passes() {
    // Manifest commits rmsnorm_eps=1e-5, key has rmsnorm_eps=1e-5 → pass.
    let mut m = make_manifest(0.0, 0, 1.0);
    m.rmsnorm_eps = Some(1e-5);

    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Pass,
        "matching rmsnorm_eps should pass: {:?}", report.failures);
}

#[test]
fn v4_rope_config_hash_mismatch_detected() {
    // Key has rope_config_hash X, manifest has Y → mismatch.
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [2u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());
    key.rope_config_hash = Some([1u8; 32]);

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i * 3 % 256) as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let token_id = logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0 as u32;

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.rope_config_hash = Some([2u8; 32]); // different from key's [1u8; 32]

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"rope mismatch",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained_from_traces(&traces)], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("rope_config_hash mismatch")),
        "expected rope_config_hash mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_weight_hash_mismatch_detected() {
    // Key has weight_hash X, manifest has Y → mismatch.
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [2u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());
    key.weight_hash = Some([1u8; 32]);

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i * 3 % 256) as i8).collect();
    let traces = forward_pass(&cfg, &toy.layers, &input);
    let final_hidden = verilm_core::requantize(&traces.last().unwrap().ffn_out);
    let logits = verilm_core::sampling::recompute_logits(
        &toy.lm_head, &final_hidden, cfg.vocab_size, cfg.hidden_dim,
    );
    let token_id = logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0 as u32;

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.weight_hash = Some([2u8; 32]); // different from key's [1u8; 32]

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"weight hash mismatch",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained_from_traces(&traces)], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("weight_hash mismatch")),
        "expected weight_hash mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_sampler_version_unknown_rejected() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.sampler_version = Some("unknown-sampler-v99".into());

    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("unsupported sampler_version")),
        "expected sampler_version rejection, got: {:?}", report.failures);
}

#[test]
fn v4_sampler_version_canonical_passes() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.sampler_version = Some("chacha20-vi-sample-v1".into());

    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Pass,
        "canonical sampler_version should pass: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Request → token verification
// ---------------------------------------------------------------------------

#[test]
fn v4_prompt_hash_binding() {
    // verify_v4 checks hash(prompt) == commitment.prompt_hash
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"the real prompt",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(2),
    };
    let (_commitment, state) = commit_toy(vec![retained.clone()], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Response should carry prompt and n_prompt_tokens.
    assert_eq!(response.prompt.as_deref(), Some(b"the real prompt".as_slice()));
    assert_eq!(response.n_prompt_tokens, Some(2));

    // Prompt hash should verify.
    let report = verify_v4(&key, &response, None);
    assert!(!report.failures.iter().any(|f| f.message.contains("prompt_hash")),
        "prompt hash should verify, failures: {:?}", report.failures);
}

#[test]
fn v4_prompt_hash_tamper_detected() {
    // Tamper the prompt in the response → hash mismatch.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"original prompt",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Tamper the prompt.
    response.prompt = Some(b"different prompt".to_vec());

    let report = verify_v4(&key, &response, None);
    assert!(report.failures.iter().any(|f| f.message.contains("prompt_hash mismatch")),
        "should detect prompt tampering, failures: {:?}", report.failures);
}

#[test]
fn v4_verify_input_tokenization_pass() {
    // Correct prompt token IDs match the committed chain.
    let (cfg, model, _key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    // Simulate: prompt has 2 tokens [100, 42], then token_id is the output.
    // committed token_ids = [42] (prompt tokens after first) + gen tokens
    // But in our toy model, token_ids only has [token_id].
    // So: expected_prompt = [100, token_id] means prompt[1:] = [token_id] should match chain[0].
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(2),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // expected_prompt_token_ids = [100, token_id]: first is embedding input, second in chain
    let expected = vec![100, token_id];
    let failures = verilm_verify::verify_input_tokenization(&response, &expected);
    assert!(failures.is_empty(), "should pass: {:?}", failures);
}

#[test]
fn v4_verify_input_tokenization_mismatch() {
    let (cfg, model, _key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(2),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Wrong second token → mismatch.
    let wrong = vec![100, token_id + 1];
    let failures = verilm_verify::verify_input_tokenization(&response, &wrong);
    assert!(!failures.is_empty(), "should detect mismatch");
    assert!(failures[0].message.contains("prompt token mismatch"), "failure: {}", failures[0]);
}

// ---------------------------------------------------------------------------
// Cross-request splice attacks (V4 sampled serving)
// ---------------------------------------------------------------------------

#[test]
fn v4_cross_request_splice_shell_opening() {
    // Attack: two honest requests (different prompts, different seeds).
    // Attacker splices run B's shell opening into run A's response.
    // The Merkle proof from B won't verify against A's tree.
    let (cfg, model, key) = setup();

    let input_a: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let input_b: Vec<i8> = (0..cfg.hidden_dim).map(|i| ((i * 7 + 100) % 256) as i8).collect();

    let traces_a = forward_pass(&cfg, &model, &input_a);
    let traces_b = forward_pass(&cfg, &model, &input_b);
    let retained_a = retained_from_traces(&traces_a);
    let retained_b = retained_from_traces(&traces_b);

    // Sanity: the two retained states must actually differ for this test to work.
    let hash_a = verilm_core::merkle::hash_retained_with_residual(&retained_a, None);
    let hash_b = verilm_core::merkle::hash_retained_with_residual(&retained_b, None);
    assert_ne!(hash_a, hash_b,
        "test precondition: retained states from different inputs must hash differently");

    let params_a = FullBindingParams {
        token_ids: &[42],
        prompt: b"request A",
        sampling_seed: [1u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let params_b = FullBindingParams {
        token_ids: &[99],
        prompt: b"request B",
        sampling_seed: [2u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };

    let (_commit_a, state_a) = commit_toy(vec![retained_a], &params_a, None, cfg.n_layers);
    let (_commit_b, state_b) = commit_toy(vec![retained_b], &params_b, None, cfg.n_layers);

    let mut response_a = open_v4(&state_a, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let response_b = open_v4(&state_b, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // SPLICE: graft B's shell opening + retained state into A's response,
    // keeping A's commitment (merkle root, io root, seed, prompt hash).
    response_a.shell_opening = response_b.shell_opening.clone();
    response_a.retained = response_b.retained.clone();

    let report = verify_v4(&key, &response_a, None);
    assert_eq!(report.verdict, Verdict::Fail,
        "cross-request splice must be detected, got: {:?}", report.failures);
    // The retained leaf hash from B won't match A's Merkle tree
    assert!(report.failures.iter().any(|f| f.message.contains("Merkle proof")),
        "expected Merkle proof failure from spliced retained state, got: {:?}", report.failures);
}

#[test]
fn v4_cross_request_splice_with_manifest() {
    // Same attack but with sampled decoding manifests.
    // Run A uses greedy (temp=0), run B uses sampled (temp=1.0).
    // Even if the attacker fixes up the retained state, the IO chain
    // and manifest hash binding catch the splice.
    let (cfg, model, key) = setup();

    let input_a: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let input_b: Vec<i8> = (0..cfg.hidden_dim).map(|i| ((i * 7 + 100) % 256) as i8).collect();

    let traces_a = forward_pass(&cfg, &model, &input_a);
    let traces_b = forward_pass(&cfg, &model, &input_b);
    let retained_a = retained_from_traces(&traces_a);
    let retained_b = retained_from_traces(&traces_b);

    let manifest_a = make_manifest(0.0, 0, 1.0); // greedy
    let manifest_b = make_manifest(1.0, 0, 1.0); // sampled

    let params_a = FullBindingParams {
        token_ids: &[42],
        prompt: b"greedy request",
        sampling_seed: [1u8; 32],
        manifest: Some(&manifest_a),
        n_prompt_tokens: Some(1),
    };
    let params_b = FullBindingParams {
        token_ids: &[99],
        prompt: b"sampled request",
        sampling_seed: [2u8; 32],
        manifest: Some(&manifest_b),
        n_prompt_tokens: Some(1),
    };

    let (_commit_a, state_a) = commit_toy(vec![retained_a], &params_a, None, cfg.n_layers);
    let (_commit_b, state_b) = commit_toy(vec![retained_b], &params_b, None, cfg.n_layers);

    let mut response_a = open_v4(&state_a, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let response_b = open_v4(&state_b, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // SPLICE: graft B's shell + retained + manifest into A's response,
    // keeping A's commitment.
    response_a.shell_opening = response_b.shell_opening.clone();
    response_a.retained = response_b.retained.clone();
    response_a.manifest = response_b.manifest.clone();

    let report = verify_v4(&key, &response_a, None);
    assert_eq!(report.verdict, Verdict::Fail,
        "cross-request splice with manifest must be detected, got: {:?}", report.failures);
    // Manifest from B doesn't match A's commitment hash.
    assert!(report.failures.iter().any(|f| f.message.contains("manifest hash")),
        "expected manifest hash mismatch, got: {:?}", report.failures);
    // Retained state from B doesn't match A's Merkle tree.
    assert!(report.failures.iter().any(|f| f.message.contains("Merkle proof")),
        "expected Merkle proof failure, got: {:?}", report.failures);
}

#[test]
fn v4_cross_request_splice_token_id_swap() {
    // Attack: two multi-token requests. Attacker takes token 1 from run B
    // and presents it as token 1 in run A. The IO chain catches this because
    // prev_io_hash for position 1 depends on position 0's leaf hash + token_id.
    let (cfg, model, key) = setup();

    let inputs_a: Vec<Vec<i8>> = (0..3)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();
    let inputs_b: Vec<Vec<i8>> = (0..3)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i * 11 + t * 31 + 80) % 256) as i8).collect())
        .collect();

    let all_retained_a: Vec<RetainedTokenState> = inputs_a.iter()
        .map(|inp| retained_from_traces(&forward_pass(&cfg, &model, inp)))
        .collect();
    let all_retained_b: Vec<RetainedTokenState> = inputs_b.iter()
        .map(|inp| retained_from_traces(&forward_pass(&cfg, &model, inp)))
        .collect();

    let params_a = FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"multi A",
        sampling_seed: [1u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let params_b = FullBindingParams {
        token_ids: &[40, 50, 60],
        prompt: b"multi B",
        sampling_seed: [2u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };

    let (_commit_a, state_a) = commit_toy(all_retained_a, &params_a, None, cfg.n_layers);
    let (_commit_b, state_b) = commit_toy(all_retained_b, &params_b, None, cfg.n_layers);

    let mut response_a = open_v4(&state_a, 1, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let response_b = open_v4(&state_b, 1, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // SPLICE: replace token 1's retained state and shell from B into A
    response_a.retained = response_b.retained.clone();
    response_a.shell_opening = response_b.shell_opening.clone();
    response_a.token_id = response_b.token_id;

    let report = verify_v4(&key, &response_a, None);
    assert_eq!(report.verdict, Verdict::Fail,
        "cross-request token splice must be detected, got: {:?}", report.failures);
    assert!(report.failures.iter().any(|f|
        f.message.contains("Merkle proof") || f.message.contains("IO") || f.message.contains("chain")),
        "expected Merkle or IO chain failure, got: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// verify_v4 integrated tokenization check
// ---------------------------------------------------------------------------

#[test]
fn v4_integrated_tokenization_pass() {
    // Correct prompt token IDs passed through verify_v4's new parameter.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(2),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    // expected: [100, token_id] — first is embedding input, second is in the chain.
    let expected = vec![100u32, token_id];
    let report = verify_v4(&key, &response, Some(&expected));
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_integrated_tokenization_mismatch_detected() {
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(2),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    // Wrong token ID in expected prompt.
    let wrong = vec![100u32, token_id + 1];
    let report = verify_v4(&key, &response, Some(&wrong));
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("prompt token mismatch")),
        "should detect mismatch, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Output policy enforcement: min_tokens + ignore_eos
// ---------------------------------------------------------------------------

#[test]
fn v4_min_tokens_pass_when_respected() {
    // Token at generation position >= min_tokens and non-EOS → pass.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.min_tokens = 1;
    manifest.eos_token_id = Some(999); // EOS is 999, our token is not 999

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"min_tokens pass",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_min_tokens_rejects_early_eos() {
    // Token at generation position 0 is EOS but min_tokens=5 → fail.
    let (cfg, model, key, _lm_head, input, _token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let eos_id = 42u32; // Pick a specific EOS token
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.min_tokens = 5;
    manifest.eos_token_id = Some(eos_id);

    // Force token_id to be EOS.
    let params = FullBindingParams {
        token_ids: &[eos_id],
        prompt: b"early eos",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("min_tokens")),
        "should reject early EOS, failures: {:?}", report.failures);
}

#[test]
fn v4_min_tokens_rejects_short_generation() {
    // Committed n_tokens < min_tokens → reject.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.min_tokens = 10; // Way more than our 1 generated token
    manifest.eos_token_id = Some(999);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"short gen",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("min_tokens")),
        "should reject short generation, failures: {:?}", report.failures);
}

#[test]
fn v4_ignore_eos_rejects_eos_token() {
    // ignore_eos=true but the challenged token IS EOS → fail.
    let (cfg, model, key, _lm_head, input, _token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let eos_id = 42u32;
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.ignore_eos = true;
    manifest.eos_token_id = Some(eos_id);

    let params = FullBindingParams {
        token_ids: &[eos_id],
        prompt: b"ignore eos",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("ignore_eos")),
        "should reject EOS when ignore_eos=true, failures: {:?}", report.failures);
}

#[test]
fn v4_output_policy_fails_closed_without_eos_token_id() {
    // min_tokens > 0 but eos_token_id missing → fail-closed.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.min_tokens = 5;
    // eos_token_id left as None (from make_manifest default)

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"no eos id",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("eos_token_id is missing")),
        "should fail-closed without eos_token_id, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// eos_policy enforcement
// ---------------------------------------------------------------------------

#[test]
fn v4_eos_policy_stop_allows_eos_as_last_token() {
    // eos_policy="stop", EOS is the last committed token → pass.
    let (cfg, model, key, _lm_head, input, _token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let eos_id = 42u32;
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.eos_token_id = Some(eos_id);
    // eos_policy is already "stop" from make_manifest

    // Single token committed, token_index=0, n_tokens=1 → it IS the last token.
    let params = FullBindingParams {
        token_ids: &[eos_id],
        prompt: b"eos last",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    // May fail on lm_head (token_id forced to eos_id) but should NOT fail on eos_policy.
    assert!(!report.failures.iter().any(|f| f.message.contains("eos_policy")),
        "eos_policy should not fail when EOS is last token, failures: {:?}", report.failures);
}

#[test]
fn v4_eos_policy_stop_rejects_eos_mid_sequence() {
    // eos_policy="stop", EOS at index 0 but n_tokens=2 → generation continued past EOS.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let input2: Vec<i8> = (0..cfg.hidden_dim).map(|i| ((i * 3 + 50) % 256) as i8).collect();
    let traces0 = forward_pass(&cfg, &model, &input);
    let traces1 = forward_pass(&cfg, &model, &input2);
    let retained0 = retained_from_traces(&traces0);
    let retained1 = retained_from_traces(&traces1);

    let eos_id = token_id; // Make the first token be EOS
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.eos_token_id = Some(eos_id);

    // Two tokens committed: [eos_id, something]. Opening token 0 → EOS but not last.
    let params = FullBindingParams {
        token_ids: &[eos_id, 99],
        prompt: b"eos mid",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained0, retained1], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("eos_policy")),
        "should reject EOS mid-sequence with stop policy, failures: {:?}", report.failures);
}

#[test]
fn v4_unknown_eos_policy_rejected() {
    // eos_policy="custom_thing" → rejected as unknown.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.eos_policy = "custom_thing".into();

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"unknown policy",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("unknown eos_policy")),
        "should reject unknown eos_policy, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// PromptTokenizer trait: canonical reconstruction via verify_v4_full
// ---------------------------------------------------------------------------

/// Test tokenizer that returns fixed token IDs based on prompt content.
struct FixedTokenizer {
    token_ids: Vec<u32>,
}

impl PromptTokenizer for FixedTokenizer {
    fn tokenize(&self, _prompt: &[u8], _input_spec: &InputSpec) -> Result<Vec<u32>, String> {
        Ok(self.token_ids.clone())
    }
}

#[test]
fn v4_tokenizer_trait_reconstruction_pass() {
    // verify_v4_full with a PromptTokenizer that returns the correct token IDs.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"tokenizer test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(2),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    // Tokenizer returns [100, token_id]: first is embedding input, second in chain.
    let tokenizer = FixedTokenizer { token_ids: vec![100, token_id] };
    let report = verify_v4_full(&key, &response, None, Some(&tokenizer as &dyn PromptTokenizer), None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_tokenizer_trait_reconstruction_mismatch() {
    // verify_v4_full with a PromptTokenizer that returns wrong token IDs.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"tokenizer mismatch",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(2),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    // Tokenizer returns wrong second token.
    let tokenizer = FixedTokenizer { token_ids: vec![100, token_id + 1] };
    let report = verify_v4_full(&key, &response, None, Some(&tokenizer as &dyn PromptTokenizer), None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("prompt token mismatch")),
        "should detect mismatch from reconstructed tokens, failures: {:?}", report.failures);
}

#[test]
fn v4_tokenizer_trait_error_reported() {
    // Tokenizer returns an error → should appear in failures.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"tokenizer error",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    struct FailingTokenizer;
    impl PromptTokenizer for FailingTokenizer {
        fn tokenize(&self, _: &[u8], _: &InputSpec) -> Result<Vec<u32>, String> {
            Err("tokenizer not available".into())
        }
    }

    let tokenizer = FailingTokenizer;
    let report = verify_v4_full(&key, &response, None, Some(&tokenizer as &dyn PromptTokenizer), None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("tokenizer reconstruction failed")),
        "should report tokenizer error, failures: {:?}", report.failures);
}

#[test]
fn v4_tokenizer_fallback_to_caller_supplied() {
    // No manifest in response → tokenizer can't reconstruct → falls back to caller-supplied IDs.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"no manifest",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(2),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    // Tokenizer provided but no manifest → falls back to caller-supplied IDs.
    let tokenizer = FixedTokenizer { token_ids: vec![999, 888] }; // would mismatch
    let correct_ids = vec![100u32, token_id];
    let report = verify_v4_full(&key, &response, Some(&correct_ids), Some(&tokenizer as &dyn PromptTokenizer), None);
    assert_eq!(report.verdict, Verdict::Pass, "should fall back to caller-supplied IDs, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Detokenizer trait: output text verification via verify_v4_full
// ---------------------------------------------------------------------------

/// Test detokenizer that returns a fixed string regardless of token IDs.
struct FixedDetokenizer {
    text: String,
}

impl Detokenizer for FixedDetokenizer {
    fn decode(&self, _token_ids: &[u32], _policy: Option<&str>) -> Result<String, String> {
        Ok(self.text.clone())
    }
}

struct FailingDetokenizer;

impl Detokenizer for FailingDetokenizer {
    fn decode(&self, _token_ids: &[u32], _policy: Option<&str>) -> Result<String, String> {
        Err("detokenizer exploded".into())
    }
}

#[test]
fn v4_detokenization_pass_last_token() {
    // When challenged token is the last token and detokenizer output matches
    // the claimed output_text, verification should pass.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.detokenization_policy = Some("default".into());
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"detok test",
        sampling_seed: [8u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);
    response.output_text = Some("hello world".into());

    let detok = FixedDetokenizer { text: "hello world".into() };
    let report = verify_v4_full(
        &key, &response, None, None,
        Some(&detok as &dyn Detokenizer),
    );
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_detokenization_mismatch_detected() {
    // When the detokenizer output doesn't match claimed output_text, fail.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.detokenization_policy = Some("default".into());
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"detok test",
        sampling_seed: [9u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);
    response.output_text = Some("claimed text".into());

    let detok = FixedDetokenizer { text: "actual decoded text".into() };
    let report = verify_v4_full(
        &key, &response, None, None,
        Some(&detok as &dyn Detokenizer),
    );
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("detokenization mismatch")),
        "expected detokenization mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_detokenization_error_reported() {
    // When the detokenizer callback errors, the failure is reported.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.detokenization_policy = Some("default".into());
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"detok test",
        sampling_seed: [10u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);
    response.output_text = Some("some text".into());

    let detok = FailingDetokenizer;
    let report = verify_v4_full(
        &key, &response, None, None,
        Some(&detok as &dyn Detokenizer),
    );
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("detokenization failed")),
        "expected detokenization failure, got: {:?}", report.failures);
}

#[test]
fn v4_detokenization_fails_closed_without_output_text() {
    // Fail-closed: when detokenizer is provided but output_text is None, reject.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"detok test",
        sampling_seed: [11u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);
    // response.output_text is None — fail-closed should reject.

    let detok = FixedDetokenizer { text: "anything".into() };
    let report = verify_v4_full(
        &key, &response, None, None,
        Some(&detok as &dyn Detokenizer),
    );
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("missing output_text")),
        "expected fail-closed for missing output_text, got: {:?}", report.failures);
}

#[test]
fn v4_detokenization_not_checked_without_detokenizer() {
    // When no detokenizer is provided, output_text is ignored — no failure.
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"detok test",
        sampling_seed: [12u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    attach_toy_logits(&mut response, &_lm_head, &cfg);
    response.output_text = Some("some text".into());

    // No detokenizer → check not run, pass.
    let no_detok: Option<&dyn Detokenizer> = None;
    let report = verify_v4_full(
        &key, &response, None, None,
        no_detok,
    );
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Rich prefix: embedding binding for all prefix tokens
// ---------------------------------------------------------------------------

/// Toy EmbeddingLookup backed by a Vec of rows + a prebuilt Merkle tree.
struct ToyEmbeddingLookup {
    rows: Vec<Vec<f32>>,
    tree: verilm_core::merkle::MerkleTree,
}

impl EmbeddingLookup for ToyEmbeddingLookup {
    fn embedding_row_and_proof(&self, token_id: u32) -> Option<(Vec<f32>, Option<verilm_core::merkle::MerkleProof>)> {
        let idx = token_id as usize;
        if idx >= self.rows.len() {
            return None;
        }
        let proof = verilm_core::merkle::prove(&self.tree, idx);
        Some((self.rows[idx].clone(), Some(proof)))
    }
}

/// Build a ToyEmbeddingLookup with `n_vocab` rows, where specific token_ids
/// have their rows set to the given residuals. All other rows are deterministic filler.
fn build_embedding_lookup_for_tokens(
    n_vocab: usize,
    hidden_dim: usize,
    token_residuals: &[(u32, &[f32])],
) -> ToyEmbeddingLookup {
    let mut rows: Vec<Vec<f32>> = (0..n_vocab)
        .map(|i| (0..hidden_dim).map(|j| (i * 1000 + j) as f32 * 0.001).collect())
        .collect();
    for &(tid, residual) in token_residuals {
        rows[tid as usize] = residual.to_vec();
    }
    let leaves: Vec<[u8; 32]> = rows.iter()
        .map(|r| verilm_core::merkle::hash_embedding_row(r))
        .collect();
    let tree = verilm_core::merkle::build_tree(&leaves);
    ToyEmbeddingLookup { rows, tree }
}

#[test]
fn v4_rich_prefix_embedding_pass() {
    // Commit 3 tokens with full bridge, challenge token 2 → prefix has tokens 0 and 1.
    // With embedding_lookup, response includes embedding rows + proofs for prefix tokens.
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];
    let n_vocab = 128;

    // Each token gets a slightly different initial residual
    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    let lookup = build_embedding_lookup_for_tokens(n_vocab, cfg.hidden_dim, &[
        (10, &residuals[0]), (20, &residuals[1]), (30, &residuals[2]),
    ]);
    key.embedding_merkle_root = Some(lookup.tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"rich prefix test",
        sampling_seed: [42u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    // Open token 2 with full bridge + embedding_lookup for prefix
    let proof = verilm_core::merkle::prove(&lookup.tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None,
        Some(&lookup as &dyn EmbeddingLookup), false,
    );

    // Verify prefix embedding data is populated
    assert!(response.prefix_embedding_rows.is_some(), "should have prefix embedding rows");
    assert!(response.prefix_embedding_proofs.is_some(), "should have prefix embedding proofs");
    let rows = response.prefix_embedding_rows.as_ref().unwrap();
    let proofs = response.prefix_embedding_proofs.as_ref().unwrap();
    assert_eq!(rows.len(), 2, "prefix has 2 tokens (0 and 1)");
    assert_eq!(proofs.len(), 2);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);

    // Should include 2 prefix embedding checks + embedding proof + structural + Freivalds
    assert!(report.checks_run >= 10 + cfg.n_layers * 7,
        "expected >= {} checks (incl 2 prefix embedding), got {}",
        10 + cfg.n_layers * 7, report.checks_run);
}

#[test]
fn v4_rich_prefix_tampered_embedding_detected() {
    // Same setup as pass test, but tamper one prefix embedding row after opening.
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];
    let n_vocab = 128;

    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    let lookup = build_embedding_lookup_for_tokens(n_vocab, cfg.hidden_dim, &[
        (10, &residuals[0]), (20, &residuals[1]), (30, &residuals[2]),
    ]);
    key.embedding_merkle_root = Some(lookup.tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"tampered prefix",
        sampling_seed: [42u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    let proof = verilm_core::merkle::prove(&lookup.tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let mut response = open_v4(
        &state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None,
        Some(&lookup as &dyn EmbeddingLookup), false,
    );

    // Tamper: corrupt the first prefix embedding row
    let rows = response.prefix_embedding_rows.as_mut().unwrap();
    rows[0][0] += 999.0;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "tampered prefix embedding should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("prefix token 0") && f.message.contains("Merkle proof")),
        "should fail on prefix embedding Merkle proof, got: {:?}", report.failures);
}

#[test]
fn v4_compact_prefix_still_works() {
    // Full bridge with embedding root but no embedding_lookup → no prefix embedding data.
    // Verification passes (compact mode; prefix embedding check is opt-in).
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];
    let n_vocab = 128;

    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    let lookup = build_embedding_lookup_for_tokens(n_vocab, cfg.hidden_dim, &[
        (10, &residuals[0]), (20, &residuals[1]), (30, &residuals[2]),
    ]);
    key.embedding_merkle_root = Some(lookup.tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"compact prefix test",
        sampling_seed: [42u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    // Open token 2 with full bridge but NO embedding_lookup → compact prefix
    let proof = verilm_core::merkle::prove(&lookup.tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None,
        None, false, // no embedding_lookup → compact
    );

    assert!(response.prefix_embedding_rows.is_none(), "compact mode should not have prefix rows");
    assert!(response.prefix_embedding_proofs.is_none(), "compact mode should not have prefix proofs");

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "compact prefix should still pass: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Deep prefix (Tier B): full retained state + shell Freivalds for prefix tokens
// ---------------------------------------------------------------------------

#[test]
fn v4_deep_prefix_pass() {
    // Commit 3 tokens with full bridge, open token 2 with deep_prefix=true.
    // Verifier runs hash check + shell Freivalds on both prefix tokens.
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];
    let n_vocab = 128;

    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    let lookup = build_embedding_lookup_for_tokens(n_vocab, cfg.hidden_dim, &[
        (10, &residuals[0]), (20, &residuals[1]), (30, &residuals[2]),
    ]);
    key.embedding_merkle_root = Some(lookup.tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"deep prefix test",
        sampling_seed: [42u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    let proof = verilm_core::merkle::prove(&lookup.tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None,
        Some(&lookup as &dyn EmbeddingLookup), true, // deep_prefix
    );

    // Verify deep prefix data is populated
    assert!(response.prefix_retained.is_some(), "should have prefix retained");
    assert!(response.prefix_shell_openings.is_some(), "should have prefix shell openings");
    assert_eq!(response.prefix_retained.as_ref().unwrap().len(), 2);
    assert_eq!(response.prefix_shell_openings.as_ref().unwrap().len(), 2);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);

    // Should include: structural (~8) + challenged-token Freivalds (7/layer) +
    // embedding proof (1) + 2 prefix embedding checks + 2 prefix hash checks +
    // 2 * prefix shell Freivalds (7/layer each)
    let min_expected = 10 + cfg.n_layers * 7  // challenged token
        + 2                                     // prefix embedding checks
        + 2                                     // prefix hash checks
        + 2 * cfg.n_layers * 7;                 // prefix shell Freivalds
    assert!(report.checks_run >= min_expected,
        "expected >= {} checks for deep prefix, got {}", min_expected, report.checks_run);
}

#[test]
fn v4_deep_prefix_tampered_retained_detected() {
    // Corrupt prefix_retained[0] after opening — hash check should fail.
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];
    let n_vocab = 128;

    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    let lookup = build_embedding_lookup_for_tokens(n_vocab, cfg.hidden_dim, &[
        (10, &residuals[0]), (20, &residuals[1]), (30, &residuals[2]),
    ]);
    key.embedding_merkle_root = Some(lookup.tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"deep prefix tampered retained",
        sampling_seed: [42u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    let proof = verilm_core::merkle::prove(&lookup.tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let mut response = open_v4(
        &state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None,
        Some(&lookup as &dyn EmbeddingLookup), true,
    );

    // Tamper: corrupt prefix retained state (changes hash)
    let prefix_ret = response.prefix_retained.as_mut().unwrap();
    prefix_ret[0].layers[0].a[0] = prefix_ret[0].layers[0].a[0].wrapping_add(100);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "tampered retained should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("prefix token 0") && f.message.contains("retained hash")),
        "should fail on retained hash mismatch, got: {:?}", report.failures);
}

#[test]
fn v4_deep_prefix_tampered_shell_detected() {
    // Corrupt a shell accumulator after opening — hash passes but Freivalds fails.
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];
    let n_vocab = 128;

    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    let lookup = build_embedding_lookup_for_tokens(n_vocab, cfg.hidden_dim, &[
        (10, &residuals[0]), (20, &residuals[1]), (30, &residuals[2]),
    ]);
    key.embedding_merkle_root = Some(lookup.tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"deep prefix tampered shell",
        sampling_seed: [42u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    let proof = verilm_core::merkle::prove(&lookup.tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let mut response = open_v4(
        &state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None,
        Some(&lookup as &dyn EmbeddingLookup), true,
    );

    // Tamper shell accumulator — retained stays correct, so hash passes
    let prefix_shells = response.prefix_shell_openings.as_mut().unwrap();
    prefix_shells[0].layers[0].attn_out[0] += 9999;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "tampered shell accumulator should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("prefix token 0") && f.message.contains("Freivalds")),
        "should fail on Freivalds for prefix token, got: {:?}", report.failures);
}

#[test]
fn v4_deep_prefix_compact_mode_unaffected() {
    // deep_prefix=false with embedding_lookup → Tier A (embedding rows) but not Tier B.
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];
    let n_vocab = 128;

    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    let lookup = build_embedding_lookup_for_tokens(n_vocab, cfg.hidden_dim, &[
        (10, &residuals[0]), (20, &residuals[1]), (30, &residuals[2]),
    ]);
    key.embedding_merkle_root = Some(lookup.tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"compact deep test",
        sampling_seed: [42u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    let proof = verilm_core::merkle::prove(&lookup.tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None,
        Some(&lookup as &dyn EmbeddingLookup), false, // NOT deep
    );

    // Tier A data present, Tier B absent
    assert!(response.prefix_embedding_rows.is_some(), "Tier A should be present");
    assert!(response.prefix_retained.is_none(), "Tier B should be absent");
    assert!(response.prefix_shell_openings.is_none(), "Tier B should be absent");

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_deep_prefix_count_mismatch_rejected() {
    // Manually set prefix_retained to wrong length → verifier rejects.
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];
    let n_vocab = 128;

    let residuals: Vec<Vec<f32>> = (0..3).map(|t| {
        initial_residual.iter().map(|&v| v + 0.05 * t as f32).collect()
    }).collect();

    let lookup = build_embedding_lookup_for_tokens(n_vocab, cfg.hidden_dim, &[
        (10, &residuals[0]), (20, &residuals[1]), (30, &residuals[2]),
    ]);
    key.embedding_merkle_root = Some(lookup.tree.root);

    let (all_retained, all_scales): (Vec<RetainedTokenState>, Vec<Vec<CapturedLayerScales>>) = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).unzip();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"count mismatch test",
        sampling_seed: [42u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None);

    let proof = verilm_core::merkle::prove(&lookup.tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let mut response = open_v4(
        &state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None,
        Some(&lookup as &dyn EmbeddingLookup), true,
    );

    // Sabotage: truncate prefix_retained to 1 entry (should be 2)
    let prefix_ret = response.prefix_retained.as_mut().unwrap();
    prefix_ret.truncate(1);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail, "count mismatch should be caught");
    assert!(report.failures.iter().any(|f| f.message.contains("deep prefix count mismatch")),
        "should fail on count mismatch, got: {:?}", report.failures);
}

// ===========================================================================
// Partial-audit semantics (coverage reporting) — roadmap #12
// ===========================================================================

#[test]
fn v4_full_audit_reports_full_coverage() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    match &report.coverage {
        verilm_verify::AuditCoverage::Full { layers_checked } => {
            assert_eq!(*layers_checked, cfg.n_layers,
                "full audit should check all {} layers", cfg.n_layers);
        }
        other => panic!("expected Full coverage, got {:?}", other),
    }
}

#[test]
fn v4_routine_audit_reports_routine_coverage() {
    // Create a full-audit response, then trim it to a contiguous prefix to simulate routine audit.
    let (cfg, model, key) = setup();
    assert!(cfg.n_layers >= 2, "need at least 2 layers for routine-audit test");

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Trim shell to first layer only (routine-audit prefix).
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers.truncate(1);
    shell.layer_indices = Some(vec![0]);

    let report = verify_v4(&key, &response, None);
    // Routine audit with only layer 0 will still pass the checks it runs.
    match &report.coverage {
        verilm_verify::AuditCoverage::Routine { layers_checked, layers_total } => {
            assert_eq!(*layers_checked, 1);
            assert_eq!(*layers_total, cfg.n_layers);
        }
        other => panic!("expected Routine coverage, got {:?}", other),
    }
}

#[test]
fn v4_non_contiguous_layer_indices_rejected() {
    let (cfg, model, key) = setup();
    assert!(cfg.n_layers >= 2, "need at least 2 layers");

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Keep 2 layers but claim non-contiguous indices [0, 2] (gap at 1).
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers.truncate(2);
    shell.layer_indices = Some(vec![0, 2]);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f|
        f.code == verilm_verify::FailureCode::NonContiguousLayerIndices),
        "should reject non-contiguous layer_indices, got: {:?}", report.failures);
}

#[test]
fn v4_no_shell_opening_reports_unknown_coverage() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    // Remove shell opening entirely.
    response.shell_opening = None;

    let report = verify_v4(&key, &response, None);
    // Should fail (missing shell) but report Unknown coverage.
    assert_eq!(report.verdict, Verdict::Fail);
    assert_eq!(report.coverage, verilm_verify::AuditCoverage::Unknown);
}

#[test]
fn v4_coverage_display_format() {
    let full = verilm_verify::AuditCoverage::Full { layers_checked: 32 };
    assert_eq!(format!("{}", full), "full (32/32 layers)");

    let routine = verilm_verify::AuditCoverage::Routine { layers_checked: 4, layers_total: 32 };
    assert_eq!(format!("{}", routine), "routine (4/32 layers)");

    let unknown = verilm_verify::AuditCoverage::Unknown;
    assert_eq!(format!("{}", unknown), "unknown");
}

#[test]
fn v4_pass_display_includes_coverage() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);

    let report = verify_v4(&key, &response, None);
    let display = format!("{}", report);
    assert!(display.contains("coverage: full"), "display should include coverage, got: {}", display);
}

#[test]
fn v4_routine_audit_not_mistaken_for_full() {
    // The critical semantic property: consumers can programmatically distinguish
    // routine-audit from full-audit passes.
    let (cfg, model, key) = setup();
    assert!(cfg.n_layers >= 2);

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"hello",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_toy(vec![retained], &params, None, cfg.n_layers);

    // Full audit
    let response_full = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let report_full = verify_v4(&key, &response_full, None);

    // Routine audit (trim to 1 layer)
    let mut response_routine = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false);
    let shell = response_routine.shell_opening.as_mut().unwrap();
    shell.layers.truncate(1);
    shell.layer_indices = Some(vec![0]);
    let report_routine = verify_v4(&key, &response_routine, None);

    // Both may pass, but they MUST report different coverage levels.
    assert_ne!(report_full.coverage, report_routine.coverage,
        "full and routine audits must be distinguishable");

    // Serialize coverage to JSON — consumer can parse level from the tag.
    let full_json = serde_json::to_string(&report_full.coverage).unwrap();
    let routine_json = serde_json::to_string(&report_routine.coverage).unwrap();
    assert!(full_json.contains("\"full\""), "full JSON: {}", full_json);
    assert!(routine_json.contains("\"routine\""), "routine JSON: {}", routine_json);
}
