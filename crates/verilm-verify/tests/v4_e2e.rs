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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);

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

    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &[], None, None);
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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);

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
    };
    let (_commitment, state) = commit_minimal(all_retained, &params);

    let mut response = open_v4(&state, 1, &ToyWeights(&model), &cfg, &[], None, None);
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
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);
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
    let response = open_v4(&state, 0, &ToyWeights(&wrong_model), &cfg, &[], None, None);

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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);

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

    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &[], None, None);
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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);

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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, None, None);

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
    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &ws, None, None);

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
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &wrong_ws, None, None);

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

    let n_mt = MatrixType::ALL.len();
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
            let idx = MatrixType::ALL.iter().position(|&m| m == mt).unwrap();
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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

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
    };
    let (_commitment, state) = commit_minimal(all_retained, &params);

    // Open token 2 (token_id=30) — its bridge must match
    let proof = verilm_core::merkle::prove(&tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(&state, 2, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, None, None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &ws, Some(&bridge), None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "unbound initial_residual should be rejected");
    assert!(report.failures.iter().any(|f| f.contains("embedding_merkle_root")),
        "should fail on missing embedding root, failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// LM head verification (greedy argmax)
// ---------------------------------------------------------------------------

/// Setup with lm_head: generate model with head, derive correct token_id via argmax.
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
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [1u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());

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
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"lm_head test",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Should have the standard checks PLUS one lm_head check
    assert!(report.failures.is_empty());
}

#[test]
fn v4_lm_head_wrong_token_detected() {
    let (cfg, model, key, _lm_head, input, token_id) = setup_lm_head();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    // Use a wrong token_id (offset by 1)
    let wrong_token = (token_id + 1) % cfg.vocab_size as u32;

    let params = FullBindingParams {
        token_ids: &[wrong_token],
        prompt: b"lm_head tamper test",
        sampling_seed: [7u8; 32],
        manifest: None,
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail, "wrong token should be detected");
    assert!(report.failures.iter().any(|f| f.contains("lm_head")),
        "should fail on lm_head check, failures: {:?}", report.failures);
}

#[test]
fn v4_lm_head_multi_token_pass() {
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 12345);
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [1u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());

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
    };
    let (_commitment, state) = commit_minimal(all_retained, &params);

    // Verify each token
    for i in 0..3 {
        let response = open_v4(&state, i, &ToyWeights(&toy.layers), &cfg, &[], None, None);
        let report = verify_v4(&key, &response);
        assert_eq!(report.verdict, Verdict::Pass,
            "token {}: failures: {:?}", i, report.failures);
    }
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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None);

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn v4_manifest_wrong_sampled_token_detected() {
    // Non-greedy with wrong token → sampling replay should catch it.
    let cfg = ModelConfig::toy();
    let toy = verilm_test_vectors::generate_model_with_head(&cfg, 54321);
    let mut key = verilm_test_vectors::generate_key(&cfg, &toy.layers, [2u8; 32]);
    key.lm_head = Some(toy.lm_head.clone());

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let response = open_v4(&state, 0, &ToyWeights(&toy.layers), &cfg, &[], None, None);

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
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params);
    let mut response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], None, None);

    // Tamper with the manifest in the response (different temperature).
    response.manifest = Some(make_manifest(1.0, 0, 1.0));

    let report = verify_v4(&key, &response);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("manifest hash")),
        "should fail on manifest hash, failures: {:?}", report.failures);
}
