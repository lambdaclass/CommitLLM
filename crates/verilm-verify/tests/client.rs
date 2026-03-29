//! Tests for the client-side verification wrapper (challenge binding).

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{
    AuditChallenge, AuditTier, BridgeParams, DeploymentManifest, RetainedLayerState,
    RetainedTokenState, ShellWeights,
};
use verilm_prover::{commit_minimal, open_v4, CapturedLayerScales, FullBindingParams};
use verilm_test_vectors::{generate_key, generate_model, LayerWeights};
use verilm_verify::canonical::verify_binary;
use verilm_verify::client::{verify_challenged_binary, verify_challenged_response};
use verilm_verify::{FailureCode, Verdict};

// ---------------------------------------------------------------------------
// Shared helpers (same as canonical.rs tests)
// ---------------------------------------------------------------------------

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
            MatrixType::LmHead => panic!("ToyWeights: no LmHead"),
        }
    }
}

fn setup_full_bridge() -> (
    ModelConfig,
    Vec<LayerWeights>,
    verilm_core::types::VerifierKey,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<f32>,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);

    let n_mt = MatrixType::PER_LAYER.len();
    let weight_scales: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..n_mt).map(|m| 0.01 + 0.001 * (l * n_mt + m) as f32).collect())
        .collect();

    let rmsnorm_attn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| {
            (0..cfg.hidden_dim)
                .map(|i| 0.5 + 0.01 * ((l * cfg.hidden_dim + i) % 100) as f32)
                .collect()
        })
        .collect();
    let rmsnorm_ffn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| {
            (0..cfg.hidden_dim)
                .map(|i| 0.6 + 0.01 * ((l * cfg.hidden_dim + i + 37) % 100) as f32)
                .collect()
        })
        .collect();

    let initial_residual: Vec<f32> = (0..cfg.hidden_dim)
        .map(|i| 0.1 * (i as f32 - cfg.hidden_dim as f32 / 2.0))
        .collect();

    key.weight_scales = weight_scales.clone();
    key.rmsnorm_attn_weights = rmsnorm_attn.clone();
    key.rmsnorm_ffn_weights = rmsnorm_ffn.clone();
    key.rmsnorm_eps = 1e-5;

    (cfg, model, key, weight_scales, rmsnorm_attn, rmsnorm_ffn, initial_residual)
}

fn bridge_scales(cfg: &ModelConfig) -> Vec<(f32, f32, f32, f32)> {
    (0..cfg.n_layers)
        .map(|l| (
            0.3 + 0.05 * l as f32,
            0.5 + 0.1 * l as f32,
            0.4 + 0.07 * l as f32,
            0.6 + 0.03 * l as f32,
        ))
        .collect()
}

fn full_bridge_forward(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_residual: &[f32],
    rmsnorm_attn: &[Vec<f32>],
    rmsnorm_ffn: &[Vec<f32>],
    weight_scales: &[Vec<f32>],
    scales: &[(f32, f32, f32, f32)],
    eps: f64,
) -> (RetainedTokenState, Vec<CapturedLayerScales>) {
    use verilm_core::matmul::matmul_i32;
    use verilm_core::rmsnorm::{
        bridge_residual_rmsnorm, dequant_add_residual, quantize_f64_to_i8, rmsnorm_f64_input,
    };

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

        let normed = rmsnorm_f64_input(&residual, &rmsnorm_attn[l], eps);
        let x_attn = quantize_f64_to_i8(&normed, scale_x_attn as f64);

        let v_acc = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);
        let v_i8 = verilm_core::requantize(&v_acc);
        let mut a = vec![0i8; cfg.hidden_dim];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            let src = kv_head * cfg.d_head;
            let dst = qh * cfg.d_head;
            a[dst..dst + cfg.d_head].copy_from_slice(&v_i8[src..src + cfg.d_head]);
        }

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

        layers.push(RetainedLayerState { a, scale_a });
        captured_scales.push(CapturedLayerScales { scale_x_attn, scale_x_ffn, scale_h });
    }

    (RetainedTokenState { layers }, captured_scales)
}

fn setup_embedding_tree(
    initial_residual: &[f32],
    token_id: u32,
    n_vocab: usize,
) -> (verilm_core::merkle::MerkleTree, [u8; 32]) {
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if i == token_id as usize {
            leaves.push(verilm_core::merkle::hash_embedding_row(initial_residual));
        } else {
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

/// Build a canonical-grade test vector, returning (key, binary, response).
fn build_audit() -> (
    verilm_core::types::VerifierKey,
    Vec<u8>,
    verilm_core::types::V4AuditResponse,
) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
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

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"client test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![captured_scales]);
    let response = open_v4(
        &state, 0, &ToyWeights(&model), &cfg, &ws,
        Some(&bridge), None, None, None, false,
    );

    let binary = verilm_core::serialize::serialize_v4_audit(&response);
    (key, binary, response)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn client_challenge_match_pass() {
    let (key, binary, _) = build_audit();
    // Test audit: token_index=0, all layers opened (toy cfg has 2 layers).
    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices: vec![0, 1],
        tier: AuditTier::Full,
    };
    let report = verify_challenged_binary(&challenge, &key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Should have 2 extra checks (token_index + layer_indices) beyond canonical baseline.
    let baseline = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.checks_run, baseline.checks_run + 2);
}

#[test]
fn client_wrong_token_index_rejected() {
    let (key, binary, _) = build_audit();
    let challenge = AuditChallenge {
        token_index: 99, // response has token_index=0
        layer_indices: vec![0, 1],
        tier: AuditTier::Full,
    };
    let report = verify_challenged_binary(&challenge, &key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.code == FailureCode::ChallengeTokenMismatch),
        "should have ChallengeTokenMismatch: {:?}",
        report.failures,
    );
}

#[test]
fn client_wrong_layer_indices_rejected() {
    let (key, binary, _) = build_audit();
    // Response opens all layers (shell.layer_indices == None → [0,1]),
    // but challenge says only layer 0 should be opened.
    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices: vec![0],
        tier: AuditTier::Routine,
    };
    let report = verify_challenged_binary(&challenge, &key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report.failures.iter().any(|f| f.code == FailureCode::ChallengeLayerMismatch),
        "should have ChallengeLayerMismatch: {:?}",
        report.failures,
    );
}

#[test]
fn client_canonical_failure_propagated() {
    // If the canonical verifier fails AND challenge matches, we still get Fail.
    let (key, _, mut response) = build_audit();
    response.commitment.seed_commitment = None; // break canonical check
    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices: vec![0, 1],
        tier: AuditTier::Full,
    };
    let report = verify_challenged_response(&challenge, &key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    // Canonical seed failure present, no challenge failures.
    assert!(report.failures.iter().any(|f| f.code == FailureCode::MissingSeedCommitment));
    assert!(!report.failures.iter().any(|f|
        f.code == FailureCode::ChallengeTokenMismatch || f.code == FailureCode::ChallengeLayerMismatch
    ));
}

#[test]
fn client_both_canonical_and_challenge_failures() {
    // Break both canonical (missing seed) and challenge (wrong token_index).
    let (key, _, mut response) = build_audit();
    response.commitment.seed_commitment = None;
    let challenge = AuditChallenge {
        token_index: 99,
        layer_indices: vec![0, 1],
        tier: AuditTier::Full,
    };
    let report = verify_challenged_response(&challenge, &key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.code == FailureCode::MissingSeedCommitment));
    assert!(report.failures.iter().any(|f| f.code == FailureCode::ChallengeTokenMismatch));
}

#[test]
fn client_without_wrapper_has_no_challenge_checks() {
    // Calling canonical::verify_binary directly doesn't do challenge checks.
    let (key, binary, _) = build_audit();
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Pass);
    assert!(!report.failures.iter().any(|f|
        f.code == FailureCode::ChallengeTokenMismatch || f.code == FailureCode::ChallengeLayerMismatch
    ));
}
