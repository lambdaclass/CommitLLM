//! End-to-end tests for V4 retained-state verification.
//!
//! Protocol path: prover commits retained state, opens with shell intermediates,
//! verifier checks with key-only Freivalds + bridge checks. No weights needed.
//!
//! Debug/oracle path: verifier independently replays from public weights.

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{BridgeParams, DeploymentManifest, RetainedLayerState, RetainedTokenState, ShellWeights};
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None);

    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &[], None, None, None);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    let report = verify_v4(&key, &response);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None);

    let mut response = open_v4(&state, 1, &ToyWeights(&model), &cfg, &[], None, None, None);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);

    // Prover opens with WRONG weights — shell intermediates are inconsistent
    // with the keygen weights. Freivalds catches this.
    let wrong_model = generate_model(&cfg, 99999);
    let response = open_v4(&state, 0, &ToyWeights(&wrong_model), &cfg, &[], None, None, None);

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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None);

    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &[], None, None, None);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

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
    let n_mt = verilm_core::constants::MatrixType::PER_LAYER.len();
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, None, None, None);

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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None);
    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &ws, None, None, None);

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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);

    // Prover uses WRONG scales (all 1.0) while verifier has non-trivial scales.
    let wrong_ws: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![1.0; verilm_core::constants::MatrixType::PER_LAYER.len()])
        .collect();
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &wrong_ws, None, None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "scale mismatch should cause failure");
    assert!(
        report.failures.iter().any(|f| f.contains("Freivalds")),
        "expected Freivalds failure from scale mismatch, got: {:?}",
        report.failures
    );
}

// ---------------------------------------------------------------------------
// Full bridge: dequant → residual → RMSNorm → quantize
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
fn full_bridge_forward(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_residual: &[f32],
    rmsnorm_attn: &[Vec<f32>],
    rmsnorm_ffn: &[Vec<f32>],
    weight_scales: &[Vec<f32>],
    scales: &[(f32, f32, f32, f32)], // per layer: (scale_x_attn, scale_a, scale_x_ffn, scale_h)
    eps: f64,
) -> RetainedTokenState {
    use verilm_core::matmul::matmul_i32;
    use verilm_core::rmsnorm::{bridge_residual_rmsnorm, dequant_add_residual, rmsnorm_f64_input, quantize_f64_to_i8};

    let mut residual: Vec<f64> = initial_residual.iter().map(|&v| v as f64).collect();
    let mut layers = Vec::new();
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

        layers.push(RetainedLayerState { a, scale_a, scale_x_attn, scale_x_ffn, scale_h });
    }

    RetainedTokenState { layers }
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

    let retained = full_bridge_forward(
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    let report = verify_v4(&key, &response);
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

    let all_retained: Vec<RetainedTokenState> = residuals.iter().map(|ir| {
        full_bridge_forward(&cfg, &model, ir, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5)
    }).collect();

    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"multi token full bridge",
        sampling_seed: [99u8; 32],
        manifest: None,
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None);

    // Open token 2 (token_id=30) — its bridge must match
    let proof = verilm_core::merkle::prove(&tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_full_bridge_wrong_residual_detected() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let retained = full_bridge_forward(
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    // Tamper: change initial_residual in the shell opening after prover built it.
    // The embedding proof was computed for the original residual, so hash won't match.
    if let Some(ref mut shell) = response.shell_opening {
        if let Some(ref mut ir) = shell.initial_residual {
            for v in ir.iter_mut() {
                *v += 100.0; // grossly wrong
            }
        }
    }

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "wrong residual should be detected");
    assert!(
        report.failures.iter().any(|f| f.contains("embedding Merkle proof")),
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

    let retained = full_bridge_forward(
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    // Verify shell has QKV at layer 0
    let shell = response.shell_opening.as_ref().unwrap();
    assert!(shell.layers[0].q.is_some(), "full bridge should produce QKV at layer 0");
    assert!(shell.layers[0].k.is_some());
    assert!(shell.layers[0].v.is_some());

    // Verification passes with all 7 checks per layer (QKV at layer 0 included)
    let report = verify_v4(&key, &response);
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

    let retained = full_bridge_forward(
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    let report = verify_v4(&key, &response);
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

    let retained = full_bridge_forward(
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "tampered residual should be caught");
    assert!(report.failures.iter().any(|f| f.contains("embedding Merkle proof")),
        "should fail on embedding proof, failures: {:?}", report.failures);
}

#[test]
fn v4_embedding_proof_missing_when_root_present() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);

    let (_tree, root) = setup_embedding_tree(&initial_residual, 42, 128);
    key.embedding_merkle_root = Some(root);

    let retained = full_bridge_forward(
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "missing proof should be caught");
    assert!(report.failures.iter().any(|f| f.contains("missing embedding_proof")),
        "should fail on missing proof, failures: {:?}", report.failures);
}

#[test]
fn v4_embedding_proof_wrong_token_id_detected() {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let retained = full_bridge_forward(
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "wrong token_id should be caught");
    assert!(report.failures.iter().any(|f| f.contains("leaf_index") && f.contains("token_id")),
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

    let retained = full_bridge_forward(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn, &ws, &scales, 1e-5,
    );

    // Deliberately pass None for bridge → no initial_residual in the shell opening
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"downgrade",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, None, None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "omitted initial_residual should be caught");
    assert!(report.failures.iter().any(|f| f.contains("initial_residual")),
        "should fail on missing initial_residual, failures: {:?}", report.failures);
}

#[test]
fn v4_unbound_initial_residual_rejected() {
    // Key has RMSNorm weights but NO embedding_merkle_root.
    // Prover supplies initial_residual anyway → must be rejected (unbound vector).
    let (cfg, model, key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) = setup_full_bridge();
    let scales = bridge_scales(&cfg);
    assert!(key.embedding_merkle_root.is_none(), "key should have no embedding root");

    let retained = full_bridge_forward(
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "unbound initial_residual should be rejected");
    assert!(report.failures.iter().any(|f| f.contains("embedding_merkle_root")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    attach_toy_logits(&mut response, &lm_head, &cfg);

    let report = verify_v4(&key, &response);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    attach_toy_logits(&mut response, &lm_head, &cfg);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "wrong token should be detected");
    assert!(report.failures.iter().any(|f| f.contains("lm_head")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None);

    // Verify each token
    for i in 0..3 {
        let mut response = open_v4(&state, i, &ToyWeights(&toy.layers), &cfg, &[], None, None, None);
        attach_toy_logits(&mut response, &toy.lm_head, &cfg);
        let report = verify_v4(&key, &response);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None);

    // Attach honest logits — should pass.
    attach_toy_logits(&mut response, &toy.lm_head, &cfg);
    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "honest logits should pass: {:?}", report.failures);

    // Now tamper: flip some logits values.
    let shell = response.shell_opening.as_mut().unwrap();
    let logits = shell.logits_i32.as_mut().unwrap();
    logits[0] = logits[0].wrapping_add(1000);
    logits[1] = logits[1].wrapping_sub(500);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "tampered logits should be caught");
    assert!(report.failures.iter().any(|f| f.contains("Freivalds")),
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
        min_tokens: 0,
        ignore_eos: false,
        detokenization_policy: None,
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    let report = verify_v4(&key, &response);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None);

    let report = verify_v4(&key, &response);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None);
    attach_toy_logits(&mut response, &toy.lm_head, &cfg);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("lm_head")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    // Tamper with the manifest in the response (different temperature).
    response.manifest = Some(make_manifest(1.0, 0, 1.0));

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("manifest hash")),
        "should fail on manifest hash, failures: {:?}", report.failures);
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
        n_prompt_tokens: None,
    };
    // Commit with final_residuals so open_v4 attaches it to the shell.
    let (_commitment, state) = commit_minimal(
        vec![retained], &params,
        Some(vec![final_residual]),
    );
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    // Verify: lm_head check uses captured final_residual, not shell replay.
    assert!(response.shell_opening.as_ref().unwrap().final_residual.is_some(),
        "final_residual should be set on shell opening");
    let fnw = key.final_norm_weights.as_ref().unwrap();
    let lm = key.lm_head.as_ref().unwrap();
    attach_tail_logits(&mut response, lm, fnw, &cfg);
    let report = verify_v4(&key, &response);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(
        vec![retained], &params,
        Some(vec![final_residual]),
    );
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    let fnw = key.final_norm_weights.as_ref().unwrap();
    let lm = key.lm_head.as_ref().unwrap();
    attach_tail_logits(&mut response, lm, fnw, &cfg);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "wrong token should be detected");
    assert!(report.failures.iter().any(|f| f.contains("lm_head")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(
        vec![retained], &params,
        Some(vec![final_residual]),
    );
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    let fnw = key.final_norm_weights.as_ref().unwrap();
    let lm = key.lm_head.as_ref().unwrap();
    attach_tail_logits(&mut response, lm, fnw, &cfg);

    let report = verify_v4(&key, &response);
    // Tampered residual → different final_hidden → different argmax → lm_head mismatch.
    assert_eq!(report.verdict, Verdict::Fail,
        "tampered residual should be detected, failures: {:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.contains("lm_head")),
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
        n_prompt_tokens: None,
    };
    // Commit WITHOUT final_residuals — shell.final_residual will be None.
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    assert!(response.shell_opening.as_ref().unwrap().final_residual.is_none());
    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail,
        "should fail when final_residual missing but key requires it, failures: {:?}",
        report.failures);
    assert!(report.failures.iter().any(|f| f.contains("final_residual")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(
        vec![retained], &params,
        Some(vec![final_residual]),
    );
    // open_v4 without tail → logits_i32 = None
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    assert!(response.shell_opening.as_ref().unwrap().logits_i32.is_none());

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail,
        "should fail when logits_i32 missing but key has LmHead Freivalds, failures: {:?}",
        report.failures);
    assert!(report.failures.iter().any(|f| f.contains("logits_i32")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(
        vec![retained], &params,
        Some(vec![final_residual]),
    );
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    // Tamper: swap the final_residual in the response AFTER commitment.
    // This should break the Merkle proof because the leaf hash changed.
    let mut tampered = response.shell_opening.as_ref().unwrap().final_residual.clone().unwrap();
    for v in &mut tampered { *v = 0.0; }
    tampered[0] = 127.0;
    response.shell_opening.as_mut().unwrap().final_residual = Some(tampered);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail,
        "post-commitment swap of final_residual must be detected, failures: {:?}",
        report.failures);
    assert!(report.failures.iter().any(|f| f.contains("Merkle proof failed")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained_from_traces(&traces)], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None);

    verify_v4(&key, &response)
}

#[test]
fn v4_manifest_rejects_repetition_penalty() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.repetition_penalty = 1.2;
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("repetition_penalty")),
        "expected repetition_penalty rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_frequency_penalty() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.frequency_penalty = 0.5;
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("frequency_penalty")),
        "expected frequency_penalty rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_presence_penalty() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.presence_penalty = 0.3;
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("presence_penalty")),
        "expected presence_penalty rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_logit_bias() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.logit_bias = vec![(42, 5.0), (100, -10.0)];
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("logit_bias")),
        "expected logit_bias rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_guided_decoding() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.guided_decoding = "json_schema".into();
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("guided_decoding")),
        "expected guided_decoding rejection, got: {:?}", report.failures);
}

#[test]
fn v4_manifest_rejects_stop_sequences() {
    let mut m = make_manifest(0.0, 0, 1.0);
    m.stop_sequences = vec!["<|end|>".into(), "STOP".into()];
    let report = verify_with_manifest(m);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("stop_sequences")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(
        vec![retained.clone(), retained], &params, None,
    );
    // Open token_index 1 — exceeds max_tokens=1
    let response = open_v4(&state, 1, &ToyWeights(&toy.layers), &cfg, &[], None, None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("max_tokens")),
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
    manifest.max_tokens = 2; // allow indices 0,1

    let params = FullBindingParams {
        token_ids: &[token_id, token_id, token_id], // 3 tokens committed
        prompt: b"overlong transcript",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(
        vec![retained.clone(), retained.clone(), retained], &params, None,
    );
    // Open token_index 0 — valid per-token, but transcript is overlong
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("committed n_tokens") && f.contains("max_tokens")),
        "expected transcript-level max_tokens rejection, got: {:?}", report.failures);
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    attach_toy_logits(&mut response, &_lm_head, &cfg);

    // Wipe all four spec hashes from the commitment — verifier should fail-closed.
    response.commitment.input_spec_hash = None;
    response.commitment.model_spec_hash = None;
    response.commitment.decode_spec_hash = None;
    response.commitment.output_spec_hash = None;

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("missing input_spec_hash")),
        "should fail on missing input_spec_hash, failures: {:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.contains("missing model_spec_hash")),
        "should fail on missing model_spec_hash, failures: {:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.contains("missing decode_spec_hash")),
        "should fail on missing decode_spec_hash, failures: {:?}", report.failures);
    assert!(report.failures.iter().any(|f| f.contains("missing output_spec_hash")),
        "should fail on missing output_spec_hash, failures: {:?}", report.failures);
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
    let (_commitment, state) = commit_minimal(vec![retained.clone()], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    // Response should carry prompt and n_prompt_tokens.
    assert_eq!(response.prompt.as_deref(), Some(b"the real prompt".as_slice()));
    assert_eq!(response.n_prompt_tokens, Some(2));

    // Prompt hash should verify.
    let report = verify_v4(&key, &response);
    assert!(!report.failures.iter().any(|f| f.contains("prompt_hash")),
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
        n_prompt_tokens: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    // Tamper the prompt.
    response.prompt = Some(b"different prompt".to_vec());

    let report = verify_v4(&key, &response);
    assert!(report.failures.iter().any(|f| f.contains("prompt_hash mismatch")),
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
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

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
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    // Wrong second token → mismatch.
    let wrong = vec![100, token_id + 1];
    let failures = verilm_verify::verify_input_tokenization(&response, &wrong);
    assert!(!failures.is_empty(), "should detect mismatch");
    assert!(failures[0].contains("prompt token mismatch"), "failure: {}", failures[0]);
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
        n_prompt_tokens: None,
    };
    let params_b = FullBindingParams {
        token_ids: &[99],
        prompt: b"request B",
        sampling_seed: [2u8; 32],
        manifest: None,
        n_prompt_tokens: None,
    };

    let (_commit_a, state_a) = commit_minimal(vec![retained_a], &params_a, None);
    let (_commit_b, state_b) = commit_minimal(vec![retained_b], &params_b, None);

    let mut response_a = open_v4(&state_a, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    let response_b = open_v4(&state_b, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    // SPLICE: graft B's shell opening + retained state into A's response,
    // keeping A's commitment (merkle root, io root, seed, prompt hash).
    response_a.shell_opening = response_b.shell_opening.clone();
    response_a.retained = response_b.retained.clone();

    let report = verify_v4(&key, &response_a);
    assert_eq!(report.verdict, Verdict::Fail,
        "cross-request splice must be detected, got: {:?}", report.failures);
    // The retained leaf hash from B won't match A's Merkle tree
    assert!(report.failures.iter().any(|f| f.contains("Merkle proof")),
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
        n_prompt_tokens: None,
    };
    let params_b = FullBindingParams {
        token_ids: &[99],
        prompt: b"sampled request",
        sampling_seed: [2u8; 32],
        manifest: Some(&manifest_b),
        n_prompt_tokens: None,
    };

    let (_commit_a, state_a) = commit_minimal(vec![retained_a], &params_a, None);
    let (_commit_b, state_b) = commit_minimal(vec![retained_b], &params_b, None);

    let mut response_a = open_v4(&state_a, 0, &ToyWeights(&model), &cfg, &[], None, None, None);
    let response_b = open_v4(&state_b, 0, &ToyWeights(&model), &cfg, &[], None, None, None);

    // SPLICE: graft B's shell + retained + manifest into A's response,
    // keeping A's commitment.
    response_a.shell_opening = response_b.shell_opening.clone();
    response_a.retained = response_b.retained.clone();
    response_a.manifest = response_b.manifest.clone();

    let report = verify_v4(&key, &response_a);
    assert_eq!(report.verdict, Verdict::Fail,
        "cross-request splice with manifest must be detected, got: {:?}", report.failures);
    // Manifest from B doesn't match A's commitment hash.
    assert!(report.failures.iter().any(|f| f.contains("manifest hash")),
        "expected manifest hash mismatch, got: {:?}", report.failures);
    // Retained state from B doesn't match A's Merkle tree.
    assert!(report.failures.iter().any(|f| f.contains("Merkle proof")),
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
        n_prompt_tokens: None,
    };
    let params_b = FullBindingParams {
        token_ids: &[40, 50, 60],
        prompt: b"multi B",
        sampling_seed: [2u8; 32],
        manifest: None,
        n_prompt_tokens: None,
    };

    let (_commit_a, state_a) = commit_minimal(all_retained_a, &params_a, None);
    let (_commit_b, state_b) = commit_minimal(all_retained_b, &params_b, None);

    let mut response_a = open_v4(&state_a, 1, &ToyWeights(&model), &cfg, &[], None, None, None);
    let response_b = open_v4(&state_b, 1, &ToyWeights(&model), &cfg, &[], None, None, None);

    // SPLICE: replace token 1's retained state and shell from B into A
    response_a.retained = response_b.retained.clone();
    response_a.shell_opening = response_b.shell_opening.clone();
    response_a.token_id = response_b.token_id;

    let report = verify_v4(&key, &response_a);
    assert_eq!(report.verdict, Verdict::Fail,
        "cross-request token splice must be detected, got: {:?}", report.failures);
    assert!(report.failures.iter().any(|f|
        f.contains("Merkle proof") || f.contains("IO") || f.contains("chain")),
        "expected Merkle or IO chain failure, got: {:?}", report.failures);
}
