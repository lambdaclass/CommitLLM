//! Tests for the canonical verifier (binary-only, key-only, full-bridge-only).

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{
    BridgeParams, DeploymentManifest, EmbeddingLookup, RetainedLayerState, RetainedTokenState,
    ShellWeights,
};
use verilm_prover::{commit_minimal, open_v4, CapturedLayerScales, FullBindingParams};
use verilm_test_vectors::{generate_key, generate_model, LayerWeights};
use verilm_verify::canonical::{verify_binary, verify_response};
use verilm_verify::{verify_v4_legacy, Detokenizer, FailureCode, PromptTokenizer, Verdict};

// ---------------------------------------------------------------------------
// Shared helpers (same as v4_e2e.rs)
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
        .map(|l| {
            (0..n_mt)
                .map(|m| 0.01 + 0.001 * (l * n_mt + m) as f32)
                .collect()
        })
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

    (
        cfg,
        model,
        key,
        weight_scales,
        rmsnorm_attn,
        rmsnorm_ffn,
        initial_residual,
    )
}

fn bridge_scales(cfg: &ModelConfig) -> Vec<(f32, f32, f32, f32)> {
    (0..cfg.n_layers)
        .map(|l| {
            (
                0.3 + 0.05 * l as f32,
                0.5 + 0.1 * l as f32,
                0.4 + 0.07 * l as f32,
                0.6 + 0.03 * l as f32,
            )
        })
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
            &attn_out,
            ws(MatrixType::Wo),
            scale_a,
            &mut residual,
            &rmsnorm_ffn[l],
            eps,
            scale_x_ffn,
        );

        let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let h = verilm_core::silu::compute_h_scaled(
            &g,
            &u,
            ws(MatrixType::Wg),
            ws(MatrixType::Wu),
            scale_x_ffn,
            scale_h,
        );
        let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);

        if l + 1 < rmsnorm_attn.len() {
            let next_scale = scales.get(l + 1).map(|s| s.0).unwrap_or(1.0);
            bridge_residual_rmsnorm(
                &ffn_out,
                ws(MatrixType::Wd),
                scale_h,
                &mut residual,
                &rmsnorm_attn[l + 1],
                eps,
                next_scale,
            );
        } else {
            dequant_add_residual(&ffn_out, ws(MatrixType::Wd), scale_h, &mut residual);
        }

        layers.push(RetainedLayerState {
            a,
            scale_a,
            x_attn_i8: Some(x_attn),
            scale_x_attn: Some(scale_x_attn),
        });
        captured_scales.push(CapturedLayerScales {
            scale_x_attn,
            scale_x_ffn,
            scale_h,
        });
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
        attn_backend: None,
        attn_dtype: None,
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

/// Build a canonical-grade test vector: full bridge + embedding proof + manifest.
/// Serializes to binary and returns (key, binary_audit).
fn build_canonical_audit(
    manifest: Option<&DeploymentManifest>,
) -> (verilm_core::types::VerifierKey, Vec<u8>) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg,
        &model,
        &initial_residual,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
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
        prompt: b"canonical test",
        sampling_seed: [7u8; 32],
        manifest,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(
        vec![retained],
        &params,
        None,
        vec![captured_scales],
        None,
        None,
        None,
    );
    let response = open_v4(
        &state,
        0,
        &ToyWeights(&model),
        &cfg,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        None,
        false,
        false,
    );

    let binary = verilm_core::serialize::serialize_v4_audit(&response);
    (key, binary)
}

/// Build a canonical audit, returning the response too (for mutation tests).
fn build_canonical_audit_with_response(
    manifest: Option<&DeploymentManifest>,
) -> (
    verilm_core::types::VerifierKey,
    verilm_core::types::V4AuditResponse,
) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg,
        &model,
        &initial_residual,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
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
        prompt: b"canonical test",
        sampling_seed: [7u8; 32],
        manifest,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(
        vec![retained],
        &params,
        None,
        vec![captured_scales],
        None,
        None,
        None,
    );
    let response = open_v4(
        &state,
        0,
        &ToyWeights(&model),
        &cfg,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        None,
        false,
        false,
    );

    (key, response)
}

fn build_canonical_audit_with_response_n_prompt(
    manifest: Option<&DeploymentManifest>,
    n_prompt_tokens: u32,
) -> (
    verilm_core::types::VerifierKey,
    verilm_core::types::V4AuditResponse,
) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key.embedding_merkle_root = Some(root);

    let (retained, captured_scales) = full_bridge_forward(
        &cfg,
        &model,
        &initial_residual,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
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
        prompt: b"canonical test",
        sampling_seed: [7u8; 32],
        manifest,
        n_prompt_tokens: Some(n_prompt_tokens),
    };
    let (_commitment, state) = commit_minimal(
        vec![retained],
        &params,
        None,
        vec![captured_scales],
        None,
        None,
        None,
    );
    let response = open_v4(
        &state,
        0,
        &ToyWeights(&model),
        &cfg,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        None,
        false,
        false,
    );

    (key, response)
}

fn to_binary(r: &verilm_core::types::V4AuditResponse) -> Vec<u8> {
    verilm_core::serialize::serialize_v4_audit(r)
}

// ---------------------------------------------------------------------------
// Pass tests
// ---------------------------------------------------------------------------

#[test]
fn canonical_full_bridge_pass() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "failures: {:?}",
        report.failures
    );
    // Full bridge: structural + embedding + spec hashes + per-layer Freivalds
    assert!(
        report.checks_run >= 12,
        "too few checks: {}",
        report.checks_run
    );
}

#[test]
fn canonical_missing_manifest_rejected() {
    let (key, binary) = build_canonical_audit(None);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.message.contains("manifest")));
}

#[test]
fn canonical_full_bridge_with_manifest_pass() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "failures: {:?}",
        report.failures
    );
    // Should include spec hash checks
    assert!(
        report.checks_run >= 12,
        "too few checks with manifest: {}",
        report.checks_run
    );
}

#[test]
fn canonical_binary_magic_validated() {
    let (key, _) = build_canonical_audit(None);
    // Bad magic
    let err = verify_binary(&key, b"XXXX1234567890", None, None);
    assert!(err.is_err(), "should reject bad magic");
    assert!(
        err.unwrap_err().contains("magic"),
        "error should mention magic"
    );
}

#[test]
fn canonical_empty_input_rejected() {
    let (key, _) = build_canonical_audit(None);
    let err = verify_binary(&key, &[], None, None);
    assert!(err.is_err());
}

#[test]
fn canonical_truncated_payload_rejected() {
    let (key, binary) = build_canonical_audit(None);
    // Just the magic
    let err = verify_binary(&key, &binary[..4], None, None);
    assert!(err.is_err());
}

// ---------------------------------------------------------------------------
// Fail-closed tests
// ---------------------------------------------------------------------------

#[test]
fn canonical_missing_shell_opening_rejected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    response.shell_opening = None;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.message.contains("shell_opening")));
}

#[test]
fn canonical_missing_initial_residual_rejected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    response.shell_opening.as_mut().unwrap().initial_residual = None;
    // Must recompute leaf hash since it includes final_residual but not initial_residual
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.message.contains("initial_residual")));
}

#[test]
fn canonical_missing_embedding_proof_rejected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    response.shell_opening.as_mut().unwrap().embedding_proof = None;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.message.contains("embedding_proof")));
}

// ---------------------------------------------------------------------------
// Tamper detection tests
// ---------------------------------------------------------------------------

#[test]
fn canonical_tampered_attn_out_detected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers[0].attn_out[0] ^= 0x7FFF_FFFF;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.message.contains("Freivalds")));
}

#[test]
fn canonical_tampered_ffn_out_detected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers[0].ffn_out[0] ^= 0x7FFF_FFFF;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.message.contains("Freivalds")));
}

#[test]
fn canonical_tampered_seed_detected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    response.revealed_seed[0] ^= 0xFF;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| { f.code == verilm_verify::FailureCode::SeedMismatch }));
}

#[test]
fn canonical_tampered_embedding_detected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.initial_residual.as_mut().unwrap()[0] += 1.0;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| { f.code == verilm_verify::FailureCode::EmbeddingProofFailed }));
}

// ---------------------------------------------------------------------------
// Revealed-scale tamper tests (#2 security: wrong scale → wrong quantization
// → bridge/Freivalds mismatch → rejection)
// ---------------------------------------------------------------------------

#[test]
fn canonical_tampered_revealed_scale_x_attn_rejected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers[0].scale_x_attn *= 2.0; // wrong QKV input quantization
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Fail,
        "tampered scale_x_attn must cause rejection via bridge equations"
    );
    // With check-and-gate, BridgeScaleMismatch catches the tampered shell scale.
    // Freivalds may still pass since the gated x_attn uses the committed value.
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.message.contains("Freivalds")
                || matches!(
                    f.code,
                    FailureCode::BridgeScaleMismatch | FailureCode::BridgeXAttnMismatch
                )),
        "rejection should come from Freivalds or bridge mismatch, got: {:?}",
        report.failures
    );
}

#[test]
fn canonical_tampered_revealed_scale_x_ffn_rejected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers[0].scale_x_ffn *= 2.0; // wrong gate_up input quantization
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Fail,
        "tampered scale_x_ffn must cause rejection via bridge equations"
    );
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.message.contains("Freivalds")),
        "rejection should come from Freivalds mismatch, got: {:?}",
        report.failures
    );
}

#[test]
fn canonical_tampered_revealed_scale_h_rejected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers[0].scale_h *= 2.0; // wrong down projection input quantization
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Fail,
        "tampered scale_h must cause rejection via bridge equations"
    );
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.message.contains("Freivalds")),
        "rejection should come from Freivalds mismatch, got: {:?}",
        report.failures
    );
}

#[test]
fn canonical_correct_revealed_scales_pass() {
    // Confirm the clean path passes with scales in the shell opening.
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "correct revealed scales must pass: {:?}",
        report.failures
    );
}

// ---------------------------------------------------------------------------
// Spec binding tests
// ---------------------------------------------------------------------------

#[test]
fn canonical_manifest_spec_hash_verified() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "failures: {:?}",
        report.failures
    );
}

#[test]
fn canonical_unsupported_sampler_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.sampler_version = Some("custom-sampler-v99".into());
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| { f.code == verilm_verify::FailureCode::UnsupportedSamplerVersion }));
}

#[test]
fn canonical_unsupported_repetition_penalty_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.repetition_penalty = 1.5;
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| { f.code == verilm_verify::FailureCode::UnsupportedDecodeFeature }));
}

#[test]
fn canonical_decode_mode_inconsistency_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.decode_mode = Some("sampled".into()); // temp=0 but mode=sampled
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| { f.code == verilm_verify::FailureCode::DecodeModeTempInconsistent }));
}

// ---------------------------------------------------------------------------
// Attention backend binding
// ---------------------------------------------------------------------------

#[test]
fn canonical_attn_backend_mismatch_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.attn_backend = Some("eager".into());
    let (mut key, binary) = build_canonical_audit(Some(&manifest));
    key.attn_backend = Some("sdpa".into());
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.code == verilm_verify::FailureCode::SpecFieldMismatch
            && f.context.field.as_deref() == Some("attn_backend")));
}

#[test]
fn canonical_w8a8_eager_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.quant_family = Some("W8A8".into());
    manifest.attn_backend = Some("eager".into());
    let (mut key, binary) = build_canonical_audit(Some(&manifest));
    // Key also says W8A8 + eager — the fail-closed policy should reject regardless
    key.quant_family = Some("W8A8".into());
    key.attn_backend = Some("eager".into());
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.code == verilm_verify::FailureCode::SpecFieldMismatch
            && f.context.field.as_deref() == Some("attn_backend")));
}

#[test]
fn canonical_w8a8_sdpa_accepted() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.quant_family = Some("W8A8".into());
    manifest.attn_backend = Some("sdpa".into());
    let (mut key, binary) = build_canonical_audit(Some(&manifest));
    key.quant_family = Some("W8A8".into());
    key.attn_backend = Some("sdpa".into());
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "W8A8+sdpa should be accepted, failures: {:?}",
        report.failures
    );
}

#[test]
fn canonical_w8a8_flash_attention_accepted() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.quant_family = Some("W8A8".into());
    manifest.attn_backend = Some("flash_attention_2".into());
    let (mut key, binary) = build_canonical_audit(Some(&manifest));
    key.quant_family = Some("W8A8".into());
    key.attn_backend = None; // key doesn't pin backend; verifier validates the allow-list
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "W8A8+flash_attention_2 should be accepted, failures: {:?}",
        report.failures
    );
}

#[test]
fn canonical_attn_backend_match_accepted() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.attn_backend = Some("sdpa".into());
    let (mut key, binary) = build_canonical_audit(Some(&manifest));
    key.attn_backend = Some("sdpa".into());
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "matching attn_backend should pass, failures: {:?}",
        report.failures
    );
}

#[test]
fn canonical_w8a8_missing_backend_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.quant_family = Some("W8A8".into());
    manifest.attn_backend = None; // missing
    let (mut key, binary) = build_canonical_audit(Some(&manifest));
    key.quant_family = Some("W8A8".into());
    key.attn_backend = None;
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.code == verilm_verify::FailureCode::SpecFieldMismatch
            && f.context.field.as_deref() == Some("attn_backend")));
}

#[test]
fn canonical_w8a8_unknown_backend_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.quant_family = Some("W8A8".into());
    manifest.attn_backend = Some("custom_kernel".into());
    let (mut key, binary) = build_canonical_audit(Some(&manifest));
    key.quant_family = Some("W8A8".into());
    key.attn_backend = Some("custom_kernel".into());
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report
        .failures
        .iter()
        .any(|f| f.code == verilm_verify::FailureCode::SpecFieldMismatch
            && f.context.field.as_deref() == Some("attn_backend")));
}

// ---------------------------------------------------------------------------
// Coverage reporting
// ---------------------------------------------------------------------------

#[test]
fn canonical_reports_full_coverage() {
    let (key, binary) = build_canonical_audit(None);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    match &report.coverage {
        verilm_verify::AuditCoverage::Full { layers_checked } => {
            assert!(*layers_checked > 0);
        }
        other => panic!("expected Full coverage, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Frozen fixture: golden binary audit through canonical path
// ---------------------------------------------------------------------------

#[test]
fn canonical_frozen_audit_deserializes() {
    // Legacy frozen fixtures are toy-model (no initial_residual) — canonical will reject
    // them on bridge requirements, but must deserialize and produce a report.
    let audit_path = std::path::Path::new("tests/fixtures/v4_audit_canonical.bin");
    let key_path = std::path::Path::new("tests/fixtures/v4_key_canonical.bin");
    if !audit_path.exists() || !key_path.exists() {
        return;
    }

    let audit_data = std::fs::read(audit_path).unwrap();
    let key_data = std::fs::read(key_path).unwrap();
    let key = verilm_core::serialize::deserialize_key(&key_data).unwrap();

    let report = verify_binary(&key, &audit_data, None, None).unwrap();
    // Must not panic; verdict is either Pass or Fail.
    assert!(report.verdict == Verdict::Pass || report.verdict == Verdict::Fail);
}

#[test]
fn canonical_frozen_fullbridge_passes() {
    let audit_path = std::path::Path::new("tests/fixtures/v4_audit_fullbridge.bin");
    let key_path = std::path::Path::new("tests/fixtures/v4_key_fullbridge.bin");

    let audit_data = std::fs::read(audit_path)
        .unwrap_or_else(|e| panic!("fixture not found: {} — run gen_fixtures first", e));
    let key_data = std::fs::read(key_path)
        .unwrap_or_else(|e| panic!("fixture not found: {} — run gen_fixtures first", e));
    let key = verilm_core::serialize::deserialize_key(&key_data).unwrap();

    let report = verify_binary(&key, &audit_data, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "canonical frozen fullbridge fixture must pass: {:?}",
        report.failures
    );
    assert!(
        report.checks_run >= 20,
        "expected >= 20 checks, got {}",
        report.checks_run
    );
}

#[test]
fn corridor_precision_f64_matches_legacy_committed_kv_on_toy_kv_response() {
    let (key, response) = kv_toy_response(3);

    let legacy = verilm_verify::corridor::measure_corridor_committed_kv(&key, &response, None)
        .expect("legacy committed-KV corridor must succeed on toy KV response");
    let precision = verilm_verify::corridor::measure_corridor_committed_kv_precision(
        &key,
        &response,
        None,
        verilm_core::attention::ReplayPrecision::F64,
    )
    .expect("precision-dispatch committed-KV corridor must succeed on toy KV response");

    let legacy_json = serde_json::to_value(&legacy).unwrap();
    let precision_json = serde_json::to_value(&precision).unwrap();
    assert_eq!(
        legacy_json, precision_json,
        "precision-dispatch F64 path must exactly match legacy committed-KV corridor output"
    );
}

// ===========================================================================
// Parity tests: legacy verify_v4_full vs canonical::verify_binary
//
// For canonical-grade inputs (full bridge + manifest + embedding proof),
// both verifiers must produce the same verdict and same failure codes.
// ===========================================================================

/// Run both verifiers on the same response/binary and assert parity.
fn assert_parity(
    key: &verilm_core::types::VerifierKey,
    response: &verilm_core::types::V4AuditResponse,
) {
    let legacy = verify_v4_legacy(key, response, None, None, None);
    let binary = to_binary(response);
    let canonical = verify_binary(key, &binary, None, None).unwrap();

    assert_eq!(
        legacy.verdict, canonical.verdict,
        "verdict mismatch: legacy={:?} canonical={:?}\nlegacy failures: {:?}\ncanonical failures: {:?}",
        legacy.verdict, canonical.verdict, legacy.failures, canonical.failures
    );

    // Check-and-gate changes which codes fire (e.g. BridgeXAttnMismatch replaces
    // FreivaldsFailed when committed x_attn is used). Require same verdict + both
    // non-empty when failing, but allow different codes.
    if legacy.verdict == Verdict::Fail {
        assert!(
            !canonical.failures.is_empty(),
            "legacy failed but canonical has no failures\n  legacy: {:?}\n  canonical: {:?}",
            legacy.failures,
            canonical.failures,
        );
    }
}

#[test]
fn parity_clean_pass() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    assert_parity(&key, &response);
}

#[test]
fn parity_tampered_attn_out() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.shell_opening.as_mut().unwrap().layers[0].attn_out[0] ^= 0x7FFF_FFFF;
    assert_parity(&key, &response);
}

#[test]
fn parity_tampered_ffn_out() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.shell_opening.as_mut().unwrap().layers[0].ffn_out[0] ^= 0x7FFF_FFFF;
    assert_parity(&key, &response);
}

#[test]
fn parity_tampered_seed() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.revealed_seed[0] ^= 0xFF;
    assert_parity(&key, &response);
}

#[test]
fn parity_tampered_embedding() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response
        .shell_opening
        .as_mut()
        .unwrap()
        .initial_residual
        .as_mut()
        .unwrap()[0] += 1.0;
    assert_parity(&key, &response);
}

#[test]
fn parity_missing_shell_opening() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.shell_opening = None;
    assert_parity(&key, &response);
}

#[test]
fn parity_unsupported_sampler() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.sampler_version = Some("custom-sampler-v99".into());
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    assert_parity(&key, &response);
}

#[test]
fn parity_unsupported_repetition_penalty() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.repetition_penalty = 1.5;
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    assert_parity(&key, &response);
}

#[test]
fn parity_decode_mode_inconsistency() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.decode_mode = Some("sampled".into());
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    assert_parity(&key, &response);
}

#[test]
fn parity_wrong_prompt_bytes() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.prompt = Some(b"tampered prompt".to_vec());
    assert_parity(&key, &response);
}

#[test]
fn parity_tampered_merkle_root() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.commitment.merkle_root[0] ^= 0xFF;
    assert_parity(&key, &response);
}

#[test]
fn parity_tampered_io_root() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.commitment.io_root[0] ^= 0xFF;
    assert_parity(&key, &response);
}

// ---------------------------------------------------------------------------
// Phase-targeted tests
//
// Each test exercises a specific check within a named phase by tampering
// exactly one field of a valid audit, then asserting the expected FailureCode.
// ---------------------------------------------------------------------------

// Phase 1: Structural — IO chain replay

#[test]
fn phase1_io_chain_tampered_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.prev_io_hash[0] ^= 0xFF;
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::IoChainMismatch),
        "should have IoChainMismatch: {:?}",
        report.failures,
    );
}

// Phase 1: Structural — n_prompt_tokens mismatch

#[test]
fn phase1_n_prompt_tokens_mismatch_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    // Set response n_prompt_tokens to differ from commitment
    response.n_prompt_tokens = Some(response.commitment.n_prompt_tokens.unwrap_or(1) + 99);
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::NPromptTokensMismatch),
        "should have NPromptTokensMismatch: {:?}",
        report.failures,
    );
}

// Phase 1: Structural — missing seed commitment

#[test]
fn phase1_missing_seed_commitment_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.commitment.seed_commitment = None;
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::MissingSeedCommitment),
        "should have MissingSeedCommitment: {:?}",
        report.failures,
    );
}

// Phase 1: Structural — prompt hash binding

#[test]
fn phase1_prompt_hash_mismatch_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.prompt = Some(b"tampered".to_vec());
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::PromptHashMismatch),
        "should have PromptHashMismatch: {:?}",
        report.failures,
    );
}

// Phase 1: Structural — prefix count

#[test]
fn phase1_prefix_count_mismatch_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    // token_index=0 expects 0 prefix entries; inject a spurious set so
    // check_io_chain can iterate without panic, but count mismatches.
    response.prefix_leaf_hashes.push([0u8; 32]);
    response.prefix_token_ids.push(0);
    response
        .prefix_merkle_proofs
        .push(verilm_core::merkle::MerkleProof {
            leaf_index: 0,
            siblings: vec![],
        });
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::PrefixTokenCountMismatch),
        "should have PrefixTokenCountMismatch: {:?}",
        report.failures,
    );
}

// Phase 1: Structural — Merkle proof tampered

#[test]
fn phase1_merkle_proof_tampered_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.commitment.merkle_root[0] ^= 0xFF;
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::MerkleProofFailed),
        "should have MerkleProofFailed: {:?}",
        report.failures,
    );
}

// Phase 2: Embedding — tampered initial_residual

#[test]
fn phase2_embedding_leaf_mismatch_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    if let Some(ref mut shell) = response.shell_opening {
        if let Some(ref mut ir) = shell.initial_residual {
            ir[0] += 999.0; // corrupt embedding
        }
    }
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::EmbeddingLeafMismatch
                || f.code == FailureCode::EmbeddingProofFailed),
        "should have embedding failure: {:?}",
        report.failures,
    );
}

// Phase 3: Specs — manifest hash mismatch

#[test]
fn phase3_manifest_hash_mismatch_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.commitment.manifest_hash = Some([0xFFu8; 32]);
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::ManifestHashMismatch),
        "should have ManifestHashMismatch: {:?}",
        report.failures,
    );
}

// Phase 3: Specs — spec hash missing from commitment

#[test]
fn phase3_missing_spec_hash_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.commitment.input_spec_hash = None;
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::MissingSpecHash),
        "should have MissingSpecHash: {:?}",
        report.failures,
    );
}

// Phase 5: Bridge — Freivalds on Wq (QKV subcheck)

#[test]
fn phase5_freivalds_wq_tampered_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    if let Some(ref mut shell) = response.shell_opening {
        if let Some(ref mut q) = shell.layers[0].q {
            q[0] = q[0].wrapping_add(9999);
        }
    }
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::FreivaldsFailed
                && f.context.matrix.as_deref() == Some("Wq")),
        "should have FreivaldsFailed on Wq: {:?}",
        report.failures,
    );
}

// Phase 5: Bridge — residual chain (attn_out tamper breaks bridge)

#[test]
fn phase5_bridge_residual_chain_broken() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    if let Some(ref mut shell) = response.shell_opening {
        shell.layers[0].attn_out[0] = shell.layers[0].attn_out[0].wrapping_add(9999);
    }
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    // Tampered attn_out breaks the residual chain, causing downstream Freivalds failures
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::FreivaldsFailed),
        "should have FreivaldsFailed from broken residual chain: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Phase 5b: opened-token attention boundary
// ---------------------------------------------------------------------------

#[test]
fn phase5_first_traced_prompt_token_passes_without_attention_failure() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, response) = build_canonical_audit_with_response_n_prompt(Some(&manifest), 2);
    let report = verify_response(&key, &response, None, None);
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "failures: {:?}",
        report.failures
    );
    assert!(
        !report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::AttentionReplayMismatch),
        "first traced prompt token should not trigger impossible self-attention replay: {:?}",
        report.failures,
    );
}

#[test]
fn phase5_tampered_a_is_caught_by_exact_shell_checks() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    // Flip a byte in layer 0's attention output.
    response.retained.layers[0].a[0] ^= 0x7F;
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::FreivaldsFailed),
        "tampering retained.a should still be caught by exact shell checks: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Phase 4: Output policy
// ---------------------------------------------------------------------------

#[test]
fn phase4_ignore_eos_violated() {
    // ignore_eos=true but token_id=42 matches eos_token_id → IgnoreEosViolated
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.ignore_eos = true;
    manifest.eos_token_id = Some(42); // matches the test token_id
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::IgnoreEosViolated),
        "should have IgnoreEosViolated: {:?}",
        report.failures,
    );
}

#[test]
fn phase4_unknown_eos_policy() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.eos_policy = "badpolicy".into();
    manifest.eos_token_id = Some(42);
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::UnknownEosPolicy),
        "should have UnknownEosPolicy: {:?}",
        report.failures,
    );
}

#[test]
fn phase4_missing_eos_token_id() {
    // min_tokens > 0 but eos_token_id is None → MissingEosTokenId
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.min_tokens = 5;
    // eos_token_id stays None
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::MissingEosTokenId),
        "should have MissingEosTokenId: {:?}",
        report.failures,
    );
}

#[test]
fn phase4_min_tokens_violated() {
    // min_tokens=100 but n_generated=1 → MinTokensViolated
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.min_tokens = 100;
    manifest.eos_token_id = Some(42);
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::MinTokensViolated),
        "should have MinTokensViolated: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Phase 6: LM-head (structural guards)
// ---------------------------------------------------------------------------

#[test]
fn phase6_missing_logits_rejected() {
    // Key advertises lm_head Freivalds capability but shell has no logits.
    let manifest = make_manifest(0.0, 0, 1.0);
    let (mut key, response) = build_canonical_audit_with_response(Some(&manifest));
    // Populate key's LmHead Freivalds fields so phase 6 runs.
    // LmHead is index 7 in MatrixType::ALL.
    let hidden_dim = key.config.hidden_dim;
    let vocab_size = key.config.vocab_size;
    key.r_vectors[7] = vec![verilm_core::field::Fp(1); vocab_size];
    key.v_lm_head = Some(vec![verilm_core::field::Fp(1); hidden_dim]);
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::MissingLogits),
        "should have MissingLogits: {:?}",
        report.failures,
    );
}

#[test]
fn phase6_missing_final_hidden_rejected() {
    // Shell has (fake) logits but bridge can't produce final_hidden because
    // final_residual is missing → MissingFinalHidden.
    let manifest = make_manifest(0.0, 0, 1.0);
    let (mut key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    let hidden_dim = key.config.hidden_dim;
    let vocab_size = key.config.vocab_size;
    // Enable LmHead Freivalds so phase 6 doesn't early-return.
    key.r_vectors[7] = vec![verilm_core::field::Fp(1); vocab_size];
    key.v_lm_head = Some(vec![verilm_core::field::Fp(1); hidden_dim]);
    // Remove final_residual so bridge produces final_hidden=None.
    if let Some(ref mut shell) = response.shell_opening {
        shell.final_residual = None;
        // Add fake logits so shell has logits_i32.
        shell.logits_i32 = Some(vec![0i32; vocab_size]);
    }
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::MissingFinalHidden),
        "should have MissingFinalHidden: {:?}",
        report.failures,
    );
}

#[test]
fn phase6_incompatible_key_dimensions_fail_closed_without_panic() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (mut key, response) = build_canonical_audit_with_response(Some(&manifest));

    key.config.hidden_dim += 512;
    key.rmsnorm_attn_weights = vec![vec![1.0; key.config.hidden_dim]; key.config.n_layers];
    key.rmsnorm_ffn_weights = vec![vec![1.0; key.config.hidden_dim]; key.config.n_layers];
    key.final_norm_weights = Some(vec![1.0; key.config.hidden_dim]);

    let result = std::panic::catch_unwind(|| verify_response(&key, &response, None, None));
    assert!(result.is_ok(), "dimension-mismatched key must not panic");

    let report = result.unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::SpecFieldMismatch),
        "should fail closed with SpecFieldMismatch: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Phase 7: Deep prefix
// ---------------------------------------------------------------------------

#[test]
fn phase7_deep_prefix_count_mismatch() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    // Inject mismatched deep prefix arrays.
    // prefix_leaf_hashes has 0 entries; prefix_retained has 1 → mismatch.
    response.prefix_retained = Some(vec![RetainedTokenState {
        layers: vec![RetainedLayerState {
            a: vec![0i8; key.config.hidden_dim],
            scale_a: 1.0,
            x_attn_i8: None,
            scale_x_attn: None,
        }],
    }]);
    response.prefix_shell_openings = Some(vec![verilm_core::types::ShellTokenOpening {
        layers: vec![],
        layer_indices: None,
        initial_residual: None,
        embedding_proof: None,
        final_residual: None,
        logits_i32: None,
        lp_hidden_bf16: None,
    }]);
    // prefix_leaf_hashes still has 0 entries → count mismatch
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::PrefixCountMismatch),
        "should have PrefixCountMismatch: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Phase 8: Tokenization
// ---------------------------------------------------------------------------

struct MockTokenizer {
    result: Result<Vec<u32>, String>,
}

impl PromptTokenizer for MockTokenizer {
    fn tokenize(
        &self,
        _prompt: &[u8],
        _input_spec: &verilm_core::types::InputSpec,
    ) -> Result<Vec<u32>, String> {
        self.result.clone()
    }
}

#[test]
fn phase8_tokenizer_count_mismatch() {
    // Tokenizer returns 3 tokens but n_prompt_tokens=1 → PromptTokenCountMismatch
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    let tokenizer = MockTokenizer {
        result: Ok(vec![10, 20, 30]),
    };
    let report = verify_response(&key, &response, Some(&tokenizer), None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::PromptTokenCountMismatch),
        "should have PromptTokenCountMismatch: {:?}",
        report.failures,
    );
}

#[test]
fn phase8_tokenizer_error() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    let tokenizer = MockTokenizer {
        result: Err("mock tokenizer failure".into()),
    };
    let report = verify_response(&key, &response, Some(&tokenizer), None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::TokenizerError),
        "should have TokenizerError: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Phase 9: Detokenization
// ---------------------------------------------------------------------------

struct MockDetokenizer {
    result: Result<String, String>,
}

impl Detokenizer for MockDetokenizer {
    fn decode(&self, _token_ids: &[u32], _policy: Option<&str>) -> Result<String, String> {
        self.result.clone()
    }
}

#[test]
fn phase9_missing_output_text() {
    // Detokenizer provided but response has no output_text → MissingOutputText
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.output_text = None; // explicitly no output text
    let detokenizer = MockDetokenizer {
        result: Ok("decoded".into()),
    };
    let report = verify_response(&key, &response, None, Some(&detokenizer));
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::MissingOutputText),
        "should have MissingOutputText: {:?}",
        report.failures,
    );
}

#[test]
fn phase9_detokenization_mismatch() {
    // Detokenizer returns "decoded" but claimed output_text is "wrong" → mismatch
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.output_text = Some("wrong output".into());
    let detokenizer = MockDetokenizer {
        result: Ok("decoded".into()),
    };
    let report = verify_response(&key, &response, None, Some(&detokenizer));
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::DetokenizationMismatch),
        "should have DetokenizationMismatch: {:?}",
        report.failures,
    );
}

#[test]
fn phase9_detokenizer_error() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.output_text = Some("claimed".into());
    let detokenizer = MockDetokenizer {
        result: Err("mock detokenizer failure".into()),
    };
    let report = verify_response(&key, &response, None, Some(&detokenizer));
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::DetokenizerError),
        "should have DetokenizerError: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Phase 7b: Deep-prefix attention replay
//
// Multi-token test infrastructure that computes actual attention (with QK
// scoring and softmax) so that deep-prefix attention replay can be exercised.
// ---------------------------------------------------------------------------

/// Forward pass with proper multi-token attention (no RoPE, matching toy model).
///
/// Processes `n_tokens` sequentially, maintaining a per-layer KV cache.
/// Each token's attention output `a` is computed via `replay_attention_reference`
/// with the accumulated KV cache, giving correct multi-token behavior.
fn multi_token_forward_with_attention(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_residuals: &[Vec<f32>],
    rmsnorm_attn: &[Vec<f32>],
    rmsnorm_ffn: &[Vec<f32>],
    weight_scales: &[Vec<f32>],
    scales: &[(f32, f32, f32, f32)],
    eps: f64,
) -> Vec<(RetainedTokenState, Vec<CapturedLayerScales>)> {
    use verilm_core::attention::replay_attention_reference;
    use verilm_core::matmul::matmul_i32;
    use verilm_core::rmsnorm::{
        bridge_residual_rmsnorm, dequant_add_residual, quantize_f64_to_i8, rmsnorm_f64_input,
    };

    let n_tokens = initial_residuals.len();
    // Per-layer KV cache: kv_caches[layer] = (all_k_i8, all_v_i8)
    let mut kv_caches: Vec<(Vec<Vec<i8>>, Vec<Vec<i8>>)> = (0..cfg.n_layers)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    let mut results = Vec::with_capacity(n_tokens);

    for t in 0..n_tokens {
        let mut residual: Vec<f64> = initial_residuals[t].iter().map(|&v| v as f64).collect();
        let mut layers = Vec::new();
        let mut captured = Vec::new();

        for (l, lw) in model.iter().enumerate() {
            let (scale_x_attn, scale_a, scale_x_ffn, scale_h) = scales[l];
            let ws = |mt: MatrixType| -> f32 {
                let idx = MatrixType::PER_LAYER.iter().position(|&m| m == mt).unwrap();
                weight_scales[l][idx]
            };

            let normed = rmsnorm_f64_input(&residual, &rmsnorm_attn[l], eps);
            let x_attn = quantize_f64_to_i8(&normed, scale_x_attn as f64);

            // QKV projections
            let q_i8 = verilm_core::requantize(&matmul_i32(
                &lw.wq,
                &x_attn,
                cfg.hidden_dim,
                cfg.hidden_dim,
            ));
            let k_i8 =
                verilm_core::requantize(&matmul_i32(&lw.wk, &x_attn, cfg.kv_dim, cfg.hidden_dim));
            let v_i8 =
                verilm_core::requantize(&matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim));

            kv_caches[l].0.push(k_i8);
            kv_caches[l].1.push(v_i8);

            // Attention with full KV cache (seq_len = t+1)
            let a = replay_attention_reference(&q_i8, &kv_caches[l].0, &kv_caches[l].1, cfg);

            let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
            let x_ffn = bridge_residual_rmsnorm(
                &attn_out,
                ws(MatrixType::Wo),
                scale_a,
                &mut residual,
                &rmsnorm_ffn[l],
                eps,
                scale_x_ffn,
            );

            let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
            let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
            let h = verilm_core::silu::compute_h_scaled(
                &g,
                &u,
                ws(MatrixType::Wg),
                ws(MatrixType::Wu),
                scale_x_ffn,
                scale_h,
            );
            let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);

            if l + 1 < rmsnorm_attn.len() {
                let next_scale = scales.get(l + 1).map(|s| s.0).unwrap_or(1.0);
                bridge_residual_rmsnorm(
                    &ffn_out,
                    ws(MatrixType::Wd),
                    scale_h,
                    &mut residual,
                    &rmsnorm_attn[l + 1],
                    eps,
                    next_scale,
                );
            } else {
                dequant_add_residual(&ffn_out, ws(MatrixType::Wd), scale_h, &mut residual);
            }

            layers.push(RetainedLayerState {
                a,
                scale_a,
                x_attn_i8: None,
                scale_x_attn: None,
            });
            captured.push(CapturedLayerScales {
                scale_x_attn,
                scale_x_ffn,
                scale_h,
            });
        }

        results.push((RetainedTokenState { layers }, captured));
    }

    results
}

/// Simple embedding lookup for test vectors.
struct TestEmbeddingLookup {
    /// Map from token_id → (embedding_row, merkle_proof)
    entries: std::collections::HashMap<u32, (Vec<f32>, verilm_core::merkle::MerkleProof)>,
}

impl EmbeddingLookup for TestEmbeddingLookup {
    fn embedding_row_and_proof(
        &self,
        token_id: u32,
    ) -> Option<(Vec<f32>, Option<verilm_core::merkle::MerkleProof>)> {
        self.entries
            .get(&token_id)
            .map(|(row, proof)| (row.clone(), Some(proof.clone())))
    }
}

/// Multi-token forward with RoPE-aware attention (production path).
///
/// Like `multi_token_forward_with_attention` but uses dequantize→RoPE→f64 replay
/// and stores `a_i8 = round(a_f64 / scale_a)`.
fn multi_token_forward_with_rope(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_residuals: &[Vec<f32>],
    rmsnorm_attn: &[Vec<f32>],
    rmsnorm_ffn: &[Vec<f32>],
    weight_scales: &[Vec<f32>],
    scales: &[(f32, f32, f32, f32)],
    eps: f64,
) -> Vec<(RetainedTokenState, Vec<CapturedLayerScales>)> {
    use verilm_core::attention::replay_attention_roped;
    use verilm_core::matmul::matmul_i32;
    use verilm_core::rmsnorm::{
        bridge_residual_rmsnorm, dequant_add_residual, quantize_f64_to_i8, rmsnorm_f64_input,
    };
    use verilm_core::rope::{apply_rope_k, apply_rope_q, dequantize_acc};

    let n_tokens = initial_residuals.len();
    // Per-layer KV cache in f64 (post-RoPE K, dequantized V)
    let mut kv_k_cache: Vec<Vec<Vec<f64>>> = (0..cfg.n_layers).map(|_| Vec::new()).collect();
    let mut kv_v_cache: Vec<Vec<Vec<f64>>> = (0..cfg.n_layers).map(|_| Vec::new()).collect();
    let mut results = Vec::with_capacity(n_tokens);

    for t in 0..n_tokens {
        let mut residual: Vec<f64> = initial_residuals[t].iter().map(|&v| v as f64).collect();
        let mut layers = Vec::new();
        let mut captured = Vec::new();

        for (l, lw) in model.iter().enumerate() {
            let (scale_x_attn, scale_a, scale_x_ffn, scale_h) = scales[l];
            let ws = |mt: MatrixType| -> f32 {
                let idx = MatrixType::PER_LAYER.iter().position(|&m| m == mt).unwrap();
                weight_scales[l][idx]
            };

            let normed = rmsnorm_f64_input(&residual, &rmsnorm_attn[l], eps);
            let x_attn = quantize_f64_to_i8(&normed, scale_x_attn as f64);

            // QKV projections (i32 accumulators)
            let q_acc = matmul_i32(&lw.wq, &x_attn, cfg.hidden_dim, cfg.hidden_dim);
            let k_acc = matmul_i32(&lw.wk, &x_attn, cfg.kv_dim, cfg.hidden_dim);
            let v_acc = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);

            // Dequantize + RoPE
            let sx = Some(scale_x_attn);
            let q_f64 = dequantize_acc(&q_acc, Some(ws(MatrixType::Wq)), sx);
            let k_f64 = dequantize_acc(&k_acc, Some(ws(MatrixType::Wk)), sx);
            let v_f64 = dequantize_acc(&v_acc, Some(ws(MatrixType::Wv)), sx);

            let q_roped = apply_rope_q(&q_f64, t, cfg);
            let k_roped = apply_rope_k(&k_f64, t, cfg);

            kv_k_cache[l].push(k_roped);
            kv_v_cache[l].push(v_f64);

            // Attention with full KV cache (seq_len = t+1), requantized with scale_a
            let a = replay_attention_roped(
                &q_roped,
                &kv_k_cache[l],
                &kv_v_cache[l],
                scale_a as f64,
                cfg,
            );

            let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
            let x_ffn = bridge_residual_rmsnorm(
                &attn_out,
                ws(MatrixType::Wo),
                scale_a,
                &mut residual,
                &rmsnorm_ffn[l],
                eps,
                scale_x_ffn,
            );

            let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
            let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
            let h = verilm_core::silu::compute_h_scaled(
                &g,
                &u,
                ws(MatrixType::Wg),
                ws(MatrixType::Wu),
                scale_x_ffn,
                scale_h,
            );
            let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);

            if l + 1 < rmsnorm_attn.len() {
                let next_scale = scales.get(l + 1).map(|s| s.0).unwrap_or(1.0);
                bridge_residual_rmsnorm(
                    &ffn_out,
                    ws(MatrixType::Wd),
                    scale_h,
                    &mut residual,
                    &rmsnorm_attn[l + 1],
                    eps,
                    next_scale,
                );
            } else {
                dequant_add_residual(&ffn_out, ws(MatrixType::Wd), scale_h, &mut residual);
            }

            layers.push(RetainedLayerState {
                a,
                scale_a,
                x_attn_i8: None,
                scale_x_attn: None,
            });
            captured.push(CapturedLayerScales {
                scale_x_attn,
                scale_x_ffn,
                scale_h,
            });
        }

        results.push((RetainedTokenState { layers }, captured));
    }

    results
}

/// Build a 3-token deep-prefix audit, opening token 2.
/// Returns (key, response) where response has prefix_retained and prefix_shell_openings.
fn build_deep_prefix_audit() -> (
    verilm_core::types::VerifierKey,
    verilm_core::types::V4AuditResponse,
) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];

    // Embedding rows per token (offset from base)
    let residuals: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            initial_residual
                .iter()
                .map(|&v| v + 0.05 * t as f32)
                .collect()
        })
        .collect();

    // Build embedding Merkle tree
    let n_vocab = 128;
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if let Some(pos) = token_ids.iter().position(|&tid| tid as usize == i) {
            leaves.push(verilm_core::merkle::hash_embedding_row(&residuals[pos]));
        } else {
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    key.embedding_merkle_root = Some(tree.root);

    // Multi-token forward with proper attention
    let all_results = multi_token_forward_with_attention(
        &cfg,
        &model,
        &residuals,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
    );
    let (all_retained, all_scales): (Vec<_>, Vec<_>) = all_results.into_iter().unzip();

    // Embedding lookup for deep prefix
    let mut lookup_entries = std::collections::HashMap::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        let proof = verilm_core::merkle::prove(&tree, tid as usize);
        lookup_entries.insert(tid, (residuals[i].clone(), proof));
    }
    let lookup = TestEmbeddingLookup {
        entries: lookup_entries,
    };

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"deep prefix test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None, None);

    // Open token 2 with deep_prefix=true
    let proof = verilm_core::merkle::prove(&tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state,
        2,
        &ToyWeights(&model),
        &cfg,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        Some(&lookup),
        true,  // deep_prefix
        false, // use_captured_x_attn
    );

    (key, response)
}

#[test]
fn phase7b_deep_prefix_attention_replay_pass() {
    let (key, response) = build_deep_prefix_audit();
    // Verify prefix data is present
    assert!(
        response.prefix_retained.is_some(),
        "should have prefix_retained"
    );
    assert!(
        response.prefix_shell_openings.is_some(),
        "should have prefix_shell_openings"
    );
    let prefix_len = response.prefix_retained.as_ref().unwrap().len();
    assert_eq!(prefix_len, 2, "token 2 should have 2 prefix tokens");

    let report = verify_response(&key, &response, None, None);
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "deep prefix audit should pass: {:?}",
        report.failures
    );
}

#[test]
fn phase7b_deep_prefix_tampered_attention_rejected() {
    let (key, mut response) = build_deep_prefix_audit();
    // Tamper prefix token 1's retained attention output (layer 0).
    // This should be caught by the deep-prefix attention replay.
    if let Some(ref mut prefix_ret) = response.prefix_retained {
        prefix_ret[1].layers[0].a[0] ^= 0x7F;
    }
    let report = verify_response(&key, &response, None, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::AttentionReplayMismatch),
        "should have AttentionReplayMismatch from tampered prefix attention: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Known-gap test: routine audit with self-consistent fake attention
// ---------------------------------------------------------------------------

/// Build a 2-token audit where token 1 has a self-consistent fake attention
/// vector (single-token shortcut instead of proper multi-token attention).
///
/// Returns (key, response) where the response is a routine audit (no deep
/// prefix) for token 1.
///
/// The attack: the prover computes correct QKV projections but derives `a`
/// using a single-token attention shortcut (just GQA-expand V, ignoring the
/// KV cache from token 0). Everything downstream — Wo·a, bridge, FFN — is
/// recomputed consistently from the fake `a`. The routine audit passes because
/// `bridge_layers` checks `Wo·a == attn_out` (self-consistent) but does NOT
/// verify that `a` was correctly derived from QKV via multi-token attention.
fn build_routine_audit_with_fake_a() -> (
    verilm_core::types::VerifierKey,
    verilm_core::types::V4AuditResponse,
    Vec<i8>, // honest_a_layer0 for assertion
    Vec<i8>, // fake_a_layer0 for assertion
) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [42u32, 43];

    // Different embedding per token
    let residuals: Vec<Vec<f32>> = (0..2)
        .map(|t| {
            initial_residual
                .iter()
                .map(|&v| v + 0.05 * t as f32)
                .collect()
        })
        .collect();

    // Embedding Merkle tree
    let n_vocab = 128;
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if let Some(pos) = token_ids.iter().position(|&tid| tid as usize == i) {
            leaves.push(verilm_core::merkle::hash_embedding_row(&residuals[pos]));
        } else {
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    key.embedding_merkle_root = Some(tree.root);

    // Honest multi-token forward (proper attention with KV cache)
    let honest_results = multi_token_forward_with_attention(
        &cfg,
        &model,
        &residuals,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
    );
    let honest_a_layer0 = honest_results[1].0.layers[0].a.clone();

    // Fake: single-token forward for token 1 (ignores KV cache from token 0)
    let (fake_retained_1, fake_scales_1) = full_bridge_forward(
        &cfg,
        &model,
        &residuals[1],
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
    );
    let fake_a_layer0 = fake_retained_1.layers[0].a.clone();

    // Commit with: honest token 0, fake token 1
    let all_retained = vec![honest_results[0].0.clone(), fake_retained_1];
    let all_scales = vec![honest_results[0].1.clone(), fake_scales_1];

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"routine audit gap test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None, None);

    // Open token 1 as routine audit (NOT deep prefix)
    let proof = verilm_core::merkle::prove(&tree, 43);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[1],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state,
        1,
        &ToyWeights(&model),
        &cfg,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        None,
        false,
        false,
    );

    (key, response, honest_a_layer0, fake_a_layer0)
}

/// Documents the known routine-audit limitation: a prover can substitute a
/// self-consistent fake attention vector for token_index > 0 and pass all
/// verifier checks.
///
/// The fake path: compute correct QKV (Freivalds-verifiable), but derive `a`
/// from single-token attention (ignoring KV cache from prior tokens). Then
/// honestly compute Wo·a, bridge, and FFN from the fake `a`. The verifier's
/// Freivalds on Wo confirms `Wo·a == attn_out` (self-consistent), but never
/// checks that `a` was correctly derived from `softmax(QK^T/√d)·V` with the
/// full sequence context.
///
/// This gap is closed by:
/// - Token-0 attention replay (seq_len=1, softmax trivial)
/// - Deep-prefix attention replay (all co-opened prefix tokens)
/// - Future: bounded KV-cache commitment for arbitrary-token replay (roadmap #72)
#[test]
fn routine_audit_self_consistent_fake_a_passes() {
    let (key, response, honest_a, fake_a) = build_routine_audit_with_fake_a();

    // Precondition: the fake and honest attention vectors MUST differ.
    // If they don't, the test is vacuous.
    assert_ne!(
        honest_a, fake_a,
        "fake a (single-token shortcut) must differ from honest a (multi-token attention) \
         to prove the gap is real"
    );

    // The routine audit passes despite the wrong attention computation.
    let report = verify_response(&key, &response, None, None);
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "routine audit MUST pass with self-consistent fake attention vector — \
         this documents the known limitation: routine audits for token_index > 0 \
         verify Wo·a consistency but not QKV→a derivation. \
         Failures: {:?}",
        report.failures
    );

    // Verify that the bridge checks ran (the pass isn't vacuous)
    assert!(
        report.checks_run >= 12,
        "expected full bridge checks to run, got only {}",
        report.checks_run
    );
}

// ---------------------------------------------------------------------------
// Deep-prefix: opened-token attention replay
// ---------------------------------------------------------------------------

/// Verify that the existing deep-prefix audit still passes now that the
/// verifier also replays the opened token's attention.
#[test]
fn deep_prefix_opened_token_replay_pass() {
    let (key, response) = build_deep_prefix_audit();
    let report = verify_response(&key, &response, None, None);
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "deep prefix audit (including opened-token replay) should pass: {:?}",
        report.failures
    );
    // The opened-token replay adds n_layers extra checks vs the prefix-only version.
    // With 3 tokens, 2 layers: prefix replay (2 checks for token 1) + opened token
    // replay (2 checks) + token-0 replay (2 checks) + structural/bridge/etc.
    assert!(
        report.checks_run >= 16,
        "expected extra checks from opened-token replay, got {}",
        report.checks_run
    );
}

/// Build a 3-token deep-prefix audit where the opened token (token 2) has a
/// self-consistent fake `a` (single-token shortcut). Prefix tokens 0 and 1
/// are honest. This is the same attack as `routine_audit_self_consistent_fake_a_passes`
/// but in deep-prefix mode — here the verifier CAN detect it.
fn build_deep_prefix_audit_fake_opened_a() -> (
    verilm_core::types::VerifierKey,
    verilm_core::types::V4AuditResponse,
    Vec<i8>, // honest_a_layer0 for assertion
    Vec<i8>, // fake_a_layer0 for assertion
) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];

    let residuals: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            initial_residual
                .iter()
                .map(|&v| v + 0.05 * t as f32)
                .collect()
        })
        .collect();

    let n_vocab = 128;
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if let Some(pos) = token_ids.iter().position(|&tid| tid as usize == i) {
            leaves.push(verilm_core::merkle::hash_embedding_row(&residuals[pos]));
        } else {
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    key.embedding_merkle_root = Some(tree.root);

    // Honest multi-token forward
    let honest_results = multi_token_forward_with_attention(
        &cfg,
        &model,
        &residuals,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
    );
    let honest_a_layer0 = honest_results[2].0.layers[0].a.clone();

    // Fake: single-token forward for token 2 (ignores KV cache from tokens 0,1)
    let (fake_retained_2, fake_scales_2) = full_bridge_forward(
        &cfg,
        &model,
        &residuals[2],
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
    );
    let fake_a_layer0 = fake_retained_2.layers[0].a.clone();

    // Commit: honest tokens 0,1 + fake token 2
    let all_retained = vec![
        honest_results[0].0.clone(),
        honest_results[1].0.clone(),
        fake_retained_2,
    ];
    let all_scales = vec![
        honest_results[0].1.clone(),
        honest_results[1].1.clone(),
        fake_scales_2,
    ];

    let mut lookup_entries = std::collections::HashMap::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        let proof = verilm_core::merkle::prove(&tree, tid as usize);
        lookup_entries.insert(tid, (residuals[i].clone(), proof));
    }
    let lookup = TestEmbeddingLookup {
        entries: lookup_entries,
    };

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"deep prefix opened token gap test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None, None);

    // Open token 2 with deep_prefix=true
    let proof = verilm_core::merkle::prove(&tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state,
        2,
        &ToyWeights(&model),
        &cfg,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        Some(&lookup),
        true,
        false,
    );

    (key, response, honest_a_layer0, fake_a_layer0)
}

// ---------------------------------------------------------------------------
// RoPE-aware deep-prefix tests (production path)
// ---------------------------------------------------------------------------

/// Single-token forward with RoPE — the attacker shortcut (ignores KV cache).
fn full_bridge_forward_with_rope(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_residual: &[f32],
    rmsnorm_attn: &[Vec<f32>],
    rmsnorm_ffn: &[Vec<f32>],
    weight_scales: &[Vec<f32>],
    scales: &[(f32, f32, f32, f32)],
    eps: f64,
    position: usize,
) -> (RetainedTokenState, Vec<CapturedLayerScales>) {
    let results = multi_token_forward_with_rope(
        cfg,
        model,
        &[initial_residual.to_vec()],
        rmsnorm_attn,
        rmsnorm_ffn,
        weight_scales,
        scales,
        eps,
    );
    // The forward ran at t=0 internally, but the real position matters only for
    // RoPE. For a single-token shortcut, the attacker would process position 0.
    // This is correct — the attack is using seq_len=1 instead of the full cache.
    let _ = position;
    results.into_iter().next().unwrap()
}

/// Build a 3-token deep-prefix audit using the RoPE-aware production forward.
fn build_deep_prefix_audit_roped() -> (
    verilm_core::types::VerifierKey,
    verilm_core::types::V4AuditResponse,
) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    key.rope_aware_replay = true;
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];

    let residuals: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            initial_residual
                .iter()
                .map(|&v| v + 0.05 * t as f32)
                .collect()
        })
        .collect();

    let n_vocab = 128;
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if let Some(pos) = token_ids.iter().position(|&tid| tid as usize == i) {
            leaves.push(verilm_core::merkle::hash_embedding_row(&residuals[pos]));
        } else {
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    key.embedding_merkle_root = Some(tree.root);

    let all_results = multi_token_forward_with_rope(
        &cfg,
        &model,
        &residuals,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
    );
    let (all_retained, all_scales): (Vec<_>, Vec<_>) = all_results.into_iter().unzip();

    let mut lookup_entries = std::collections::HashMap::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        let proof = verilm_core::merkle::prove(&tree, tid as usize);
        lookup_entries.insert(tid, (residuals[i].clone(), proof));
    }
    let lookup = TestEmbeddingLookup {
        entries: lookup_entries,
    };

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"rope deep prefix test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None, None);

    let proof = verilm_core::merkle::prove(&tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state,
        2,
        &ToyWeights(&model),
        &cfg,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        Some(&lookup),
        true,
        false,
    );

    (key, response)
}

#[test]
fn rope_deep_prefix_attention_replay_pass() {
    let (key, response) = build_deep_prefix_audit_roped();
    assert!(key.rope_aware_replay);

    let report = verify_response(&key, &response, None, None);
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "RoPE deep-prefix honest audit must pass: {:?}",
        report.failures,
    );
    // Should have attention checks from both prefix and opened token
    assert!(
        report.checks_run >= 20,
        "expected >= 20 checks, got {}",
        report.checks_run
    );
}

/// Build a 3-token audit with RoPE where the opened token uses a single-token
/// shortcut (fake attention). The verifier should detect the mismatch.
fn build_deep_prefix_audit_roped_fake_a() -> (
    verilm_core::types::VerifierKey,
    verilm_core::types::V4AuditResponse,
    Vec<i8>,
    Vec<i8>,
) {
    let (cfg, model, mut key, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    key.rope_aware_replay = true;
    let scales = bridge_scales(&cfg);
    let token_ids = [10u32, 20, 30];

    let residuals: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            initial_residual
                .iter()
                .map(|&v| v + 0.05 * t as f32)
                .collect()
        })
        .collect();

    let n_vocab = 128;
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if let Some(pos) = token_ids.iter().position(|&tid| tid as usize == i) {
            leaves.push(verilm_core::merkle::hash_embedding_row(&residuals[pos]));
        } else {
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    key.embedding_merkle_root = Some(tree.root);

    // Honest multi-token forward with RoPE
    let honest_results = multi_token_forward_with_rope(
        &cfg,
        &model,
        &residuals,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
    );
    let honest_a_layer0 = honest_results[2].0.layers[0].a.clone();

    // Fake: single-token RoPE forward for token 2 (ignores KV cache from tokens 0,1)
    let (fake_retained_2, fake_scales_2) = full_bridge_forward_with_rope(
        &cfg,
        &model,
        &residuals[2],
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
        2,
    );
    let fake_a_layer0 = fake_retained_2.layers[0].a.clone();

    let all_retained = vec![
        honest_results[0].0.clone(),
        honest_results[1].0.clone(),
        fake_retained_2,
    ];
    let all_scales = vec![
        honest_results[0].1.clone(),
        honest_results[1].1.clone(),
        fake_scales_2,
    ];

    let mut lookup_entries = std::collections::HashMap::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        let proof = verilm_core::merkle::prove(&tree, tid as usize);
        lookup_entries.insert(tid, (residuals[i].clone(), proof));
    }
    let lookup = TestEmbeddingLookup {
        entries: lookup_entries,
    };

    let manifest = make_manifest(0.0, 0, 1.0);
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"rope deep prefix fake a test",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None, None);

    let proof = verilm_core::merkle::prove(&tree, 30);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &residuals[2],
        embedding_proof: Some(proof),
    };
    let response = open_v4(
        &state,
        2,
        &ToyWeights(&model),
        &cfg,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        Some(&lookup),
        true,
        false,
    );

    (key, response, honest_a_layer0, fake_a_layer0)
}

#[test]
fn rope_deep_prefix_fake_a_caught() {
    let (key, response, honest_a, fake_a) = build_deep_prefix_audit_roped_fake_a();

    assert_ne!(
        honest_a, fake_a,
        "fake a must differ from honest a to prove the detection is real"
    );

    let report = verify_response(&key, &response, None, None);
    assert_eq!(
        report.verdict,
        Verdict::Fail,
        "RoPE deep-prefix MUST catch fake opened-token attention: {:?}",
        report.failures,
    );
    assert!(
        report.failures.iter().any(|f| {
            f.code == FailureCode::AttentionReplayMismatch && f.message.contains("opened token")
        }),
        "should have AttentionReplayMismatch for opened token: {:?}",
        report.failures,
    );
}

/// Deep-prefix mode catches the same fake-attention attack that routine audit
/// misses. The opened token has fake `a` from single-token shortcut, but the
/// verifier replays attention using the full prefix KV cache and detects the
/// mismatch.
#[test]
fn deep_prefix_opened_token_fake_a_caught() {
    let (key, response, honest_a, fake_a) = build_deep_prefix_audit_fake_opened_a();

    // Precondition: fake and honest attention differ
    assert_ne!(
        honest_a, fake_a,
        "fake a must differ from honest a to prove the detection is real"
    );

    let report = verify_response(&key, &response, None, None);
    assert_eq!(
        report.verdict,
        Verdict::Fail,
        "deep-prefix audit MUST catch fake opened-token attention: {:?}",
        report.failures
    );
    assert!(
        report.failures.iter().any(|f| {
            f.code == FailureCode::AttentionReplayMismatch && f.message.contains("opened token")
        }),
        "should have AttentionReplayMismatch for opened token: {:?}",
        report.failures,
    );
}

// ---------------------------------------------------------------------------
// Corridor measurement tests
// ---------------------------------------------------------------------------

#[test]
fn corridor_toy_model_all_diffs_zero() {
    let (key, response) = build_deep_prefix_audit();
    assert!(
        !key.rope_aware_replay,
        "toy model should not use RoPE replay"
    );

    let report = verilm_verify::corridor::measure_corridor(&key, &response, None).unwrap();
    assert_eq!(
        report.global_linf, 0,
        "toy model corridor should have zero diff everywhere: {:?}",
        report.measurements,
    );
    for m in &report.measurements {
        assert_eq!(
            m.linf, 0,
            "layer {} pos {}: expected zero linf",
            m.layer, m.token_position
        );
        assert_eq!(
            m.frac_eq, 1.0,
            "layer {} pos {}: expected all equal",
            m.layer, m.token_position
        );
        assert_eq!(m.histogram[0], m.n_elements);
    }
    assert!(!report.measurements.is_empty(), "should have measurements");
}

#[test]
fn corridor_roped_toy_model_all_diffs_zero() {
    let (key, response) = build_deep_prefix_audit_roped();
    assert!(key.rope_aware_replay, "roped key should use RoPE replay");

    let report = verilm_verify::corridor::measure_corridor(&key, &response, None).unwrap();
    assert_eq!(
        report.global_linf, 0,
        "roped toy model corridor should have zero diff: {:?}",
        report.measurements,
    );
    for m in &report.measurements {
        assert_eq!(m.linf, 0);
        assert_eq!(m.frac_eq, 1.0);
    }
}

#[test]
fn corridor_missing_prefix_fails_closed() {
    let (key, mut response) = build_deep_prefix_audit();
    response.prefix_retained = None;
    assert!(verilm_verify::corridor::measure_corridor(&key, &response, None).is_err());

    let (key2, mut response2) = build_deep_prefix_audit();
    response2.prefix_shell_openings = None;
    assert!(verilm_verify::corridor::measure_corridor(&key2, &response2, None).is_err());

    let (key3, mut response3) = build_deep_prefix_audit();
    response3.shell_opening = None;
    assert!(verilm_verify::corridor::measure_corridor(&key3, &response3, None).is_err());
}

#[test]
fn corridor_aggregation_with_known_diffs() {
    // Start from a passing toy audit, then tamper specific retained `a` values
    // so we get known non-zero diffs and can check the aggregation fields.
    let (key, mut response) = build_deep_prefix_audit();

    // Sanity: untampered should be all-zero.
    let clean = verilm_verify::corridor::measure_corridor(&key, &response, None).unwrap();
    assert_eq!(clean.global_linf, 0);

    // Tamper prefix token 1, layer 0: shift a[0] by +3
    let prefix_ret = response.prefix_retained.as_mut().unwrap();
    let original = prefix_ret[1].layers[0].a[0];
    prefix_ret[1].layers[0].a[0] = original.wrapping_add(3);

    // Tamper opened token (token 2), layer 1: shift a[0] by +5
    let orig_opened = response.retained.layers[1].a[0];
    response.retained.layers[1].a[0] = orig_opened.wrapping_add(5);

    let report = verilm_verify::corridor::measure_corridor(&key, &response, None).unwrap();

    // global_linf should be 5 (the larger tamper)
    assert_eq!(
        report.global_linf, 5,
        "global_linf: {:?}",
        report.per_layer_max_linf
    );

    // per_layer_max_linf: layer 0 should be 3, layer 1 should be 5
    assert!(report.per_layer_max_linf.len() >= 2);
    assert_eq!(report.per_layer_max_linf[0], 3, "layer 0 max linf");
    assert_eq!(report.per_layer_max_linf[1], 5, "layer 1 max linf");

    // per_position_max_linf: should have entries for pos=1 (diff=3) and
    // pos=2 (diff=5). They're sorted by position.
    assert!(report.per_position_max_linf.len() >= 2);
    // Find the max linf for each position from measurements directly
    let pos1_max = report
        .measurements
        .iter()
        .filter(|m| m.token_position == 1)
        .map(|m| m.linf)
        .max()
        .unwrap_or(0);
    let pos2_max = report
        .measurements
        .iter()
        .filter(|m| m.token_position == 2)
        .map(|m| m.linf)
        .max()
        .unwrap_or(0);
    assert_eq!(pos1_max, 3, "position 1 max linf from measurements");
    assert_eq!(pos2_max, 5, "position 2 max linf from measurements");

    // Verify the per_position_max_linf vector matches the measurements
    for &max_linf in &report.per_position_max_linf {
        assert!(max_linf <= report.global_linf);
    }

    // Check that the tampered measurement for layer 0 / pos 1 has correct frac fields
    let m_l0_p1 = report
        .measurements
        .iter()
        .find(|m| m.layer == 0 && m.token_position == 1)
        .expect("should have measurement for layer 0, pos 1");
    assert_eq!(m_l0_p1.linf, 3);
    // Only 1 element was tampered; the rest are exact. frac_eq should be (n-1)/n.
    let expected_frac_eq = (m_l0_p1.n_elements - 1) as f64 / m_l0_p1.n_elements as f64;
    assert!(
        (m_l0_p1.frac_eq - expected_frac_eq).abs() < 1e-10,
        "frac_eq: got {} expected {}",
        m_l0_p1.frac_eq,
        expected_frac_eq,
    );
    // frac_le_2 should also be (n-1)/n since the one tampered element has diff=3
    assert!(
        (m_l0_p1.frac_le_2 - expected_frac_eq).abs() < 1e-10,
        "frac_le_2: got {} expected {}",
        m_l0_p1.frac_le_2,
        expected_frac_eq,
    );
}

// =========================================================================
// Check-and-gate bridge trust boundary tests
// =========================================================================

/// Verify that committed x_attn matching derived exactly passes verification.
#[test]
fn bridge_check_and_gate_exact_match_passes() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, response) = build_canonical_audit_with_response(Some(&manifest));
    // full_bridge_forward now populates x_attn_i8 + scale_x_attn — should pass.
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Pass,
        "exact-match committed x_attn should pass: {:?}",
        report.failures
    );
    assert!(
        !report.failures.iter().any(|f| matches!(
            f.code,
            FailureCode::BridgeXAttnMismatch | FailureCode::BridgeScaleMismatch
        )),
        "no bridge mismatch expected: {:?}",
        report.failures
    );
}

/// Verify that a tampered committed x_attn produces BridgeXAttnMismatch.
#[test]
fn bridge_check_and_gate_tampered_x_attn_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    // Tamper committed x_attn by a large amount.
    if let Some(ref mut xa) = response.retained.layers[0].x_attn_i8 {
        for v in xa.iter_mut() {
            *v = v.wrapping_add(10);
        }
    }
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::BridgeXAttnMismatch),
        "tampered x_attn should produce BridgeXAttnMismatch: {:?}",
        report.failures
    );
}

/// Verify that a tampered committed scale_x_attn produces BridgeScaleMismatch.
#[test]
fn bridge_check_and_gate_tampered_scale_rejected() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, mut response) = build_canonical_audit_with_response(Some(&manifest));
    response.retained.layers[0].scale_x_attn = Some(999.0); // wrong scale
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.code == FailureCode::BridgeScaleMismatch),
        "tampered scale should produce BridgeScaleMismatch: {:?}",
        report.failures
    );
}

/// Verify that hash domain v3 includes x_attn in the hash.
#[test]
fn bridge_hash_v3_includes_x_attn() {
    use verilm_core::types::{RetainedLayerState, RetainedTokenState};

    let a = vec![1i8, 2, 3, 4];
    let with_xa = RetainedTokenState {
        layers: vec![RetainedLayerState {
            a: a.clone(),
            scale_a: 1.0,
            x_attn_i8: Some(vec![10i8, 20, 30, 40]),
            scale_x_attn: Some(0.5),
        }],
    };
    let without_xa = RetainedTokenState {
        layers: vec![RetainedLayerState {
            a: a.clone(),
            scale_a: 1.0,
            x_attn_i8: None,
            scale_x_attn: None,
        }],
    };
    let h1 = verilm_core::merkle::hash_retained_state_direct(&with_xa);
    let h2 = verilm_core::merkle::hash_retained_state_direct(&without_xa);
    assert_ne!(
        h1, h2,
        "committed vs uncommitted x_attn must produce different hashes"
    );
}

/// Verify that different committed x_attn values produce different hashes.
#[test]
fn bridge_hash_v3_different_x_attn_different_hash() {
    use verilm_core::types::{RetainedLayerState, RetainedTokenState};

    let make = |xa: Vec<i8>| RetainedTokenState {
        layers: vec![RetainedLayerState {
            a: vec![1i8; 4],
            scale_a: 1.0,
            x_attn_i8: Some(xa),
            scale_x_attn: Some(0.5),
        }],
    };
    let h1 = verilm_core::merkle::hash_retained_state_direct(&make(vec![10, 20, 30, 40]));
    let h2 = verilm_core::merkle::hash_retained_state_direct(&make(vec![11, 20, 30, 40]));
    assert_ne!(
        h1, h2,
        "different x_attn values must produce different hashes"
    );
}

// ===========================================================================
// KV transcript Merkle tests (roadmap #3)
// ===========================================================================

/// KV entry hash is deterministic for same inputs.
#[test]
fn kv_hash_deterministic() {
    let k = vec![1.0f64, 2.0, 3.0, 4.0];
    let v = vec![5.0f64, 6.0, 7.0, 8.0];
    let h1 = verilm_core::merkle::hash_kv_entry(0, 0, &k, &v);
    let h2 = verilm_core::merkle::hash_kv_entry(0, 0, &k, &v);
    assert_eq!(h1, h2);
}

/// Different K values produce different hashes.
#[test]
fn kv_hash_different_k() {
    let v = vec![5.0f64, 6.0, 7.0, 8.0];
    let h1 = verilm_core::merkle::hash_kv_entry(0, 0, &[1.0, 2.0, 3.0, 4.0], &v);
    let h2 = verilm_core::merkle::hash_kv_entry(0, 0, &[1.0, 2.0, 3.0, 4.1], &v);
    assert_ne!(h1, h2);
}

/// Different V values produce different hashes.
#[test]
fn kv_hash_different_v() {
    let k = vec![1.0f64, 2.0, 3.0, 4.0];
    let h1 = verilm_core::merkle::hash_kv_entry(0, 0, &k, &[5.0, 6.0, 7.0, 8.0]);
    let h2 = verilm_core::merkle::hash_kv_entry(0, 0, &k, &[5.0, 6.0, 7.0, 8.1]);
    assert_ne!(h1, h2);
}

/// Different layer indices produce different hashes (domain separation).
#[test]
fn kv_hash_different_layer() {
    let k = vec![1.0f64, 2.0];
    let v = vec![3.0f64, 4.0];
    let h1 = verilm_core::merkle::hash_kv_entry(0, 5, &k, &v);
    let h2 = verilm_core::merkle::hash_kv_entry(1, 5, &k, &v);
    assert_ne!(h1, h2);
}

/// Different positions produce different hashes (domain separation).
#[test]
fn kv_hash_different_position() {
    let k = vec![1.0f64, 2.0];
    let v = vec![3.0f64, 4.0];
    let h1 = verilm_core::merkle::hash_kv_entry(0, 0, &k, &v);
    let h2 = verilm_core::merkle::hash_kv_entry(0, 1, &k, &v);
    assert_ne!(h1, h2);
}

/// KV Merkle tree: build tree from entries, verify proof roundtrip.
#[test]
fn kv_merkle_tree_roundtrip() {
    use verilm_core::merkle;

    let entries: Vec<([f64; 2], [f64; 2])> = vec![
        ([1.0, 2.0], [3.0, 4.0]),
        ([5.0, 6.0], [7.0, 8.0]),
        ([9.0, 10.0], [11.0, 12.0]),
    ];

    let layer = 0;
    let leaves: Vec<[u8; 32]> = entries
        .iter()
        .enumerate()
        .map(|(pos, (k, v))| merkle::hash_kv_entry(layer, pos, k, v))
        .collect();

    let tree = merkle::build_tree(&leaves);
    assert_eq!(tree.n_leaves, 3);

    // Prove and verify each leaf.
    for i in 0..3 {
        let proof = merkle::prove(&tree, i);
        assert!(merkle::verify(&tree.root, &leaves[i], &proof));
    }

    // Tampered leaf should fail.
    let mut tampered = leaves[1];
    tampered[0] ^= 0xff;
    let proof1 = merkle::prove(&tree, 1);
    assert!(!merkle::verify(&tree.root, &tampered, &proof1));
}

/// compute_root matches build_tree root for KV leaves.
#[test]
fn kv_compute_root_matches_build_tree() {
    use verilm_core::merkle;

    let leaves: Vec<[u8; 32]> = (0..5)
        .map(|pos| merkle::hash_kv_entry(2, pos, &[pos as f64; 4], &[0.0; 4]))
        .collect();

    let tree = merkle::build_tree(&leaves);
    let root = merkle::compute_root(&leaves);
    assert_eq!(tree.root, root);
}

/// End-to-end: commit KV entries from toy autoregressive forward pass,
/// verify kv_roots are populated and Merkle proofs validate.
#[test]
fn kv_commit_e2e_toy_autoregressive() {
    use verilm_core::merkle;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let initial: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let traces = verilm_test_vectors::forward_pass_autoregressive(&cfg, &model, &initial, 3);
    let kv_entries = verilm_test_vectors::kv_entries_from_traces(&cfg, &traces);

    let all_retained: Vec<_> = traces
        .iter()
        .map(|token_traces| verilm_core::types::RetainedTokenState {
            layers: token_traces
                .iter()
                .map(|lt| RetainedLayerState {
                    a: lt.a.clone(),
                    scale_a: lt.scale_a.unwrap_or(1.0),
                    x_attn_i8: None,
                    scale_x_attn: None,
                })
                .collect(),
        })
        .collect();

    let params = verilm_prover::FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"kv commit test",
        sampling_seed: [9u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let scales = vec![
        vec![
            verilm_prover::CapturedLayerScales {
                scale_x_attn: 1.0,
                scale_x_ffn: 1.0,
                scale_h: 1.0
            };
            cfg.n_layers
        ];
        3
    ];

    let (commitment, _state) = commit_minimal(
        all_retained,
        &params,
        None,
        scales,
        None,
        Some(kv_entries.clone()),
        None,
    );

    // kv_roots must have one root per layer.
    assert_eq!(commitment.kv_roots.len(), cfg.n_layers);

    // Each root must match independently computed root.
    for (layer_idx, layer_entries) in kv_entries.iter().enumerate() {
        let leaves: Vec<[u8; 32]> = layer_entries
            .iter()
            .enumerate()
            .map(|(pos, e)| merkle::hash_kv_entry(layer_idx, pos, &e.k_roped, &e.v_deq))
            .collect();
        let expected_root = merkle::compute_root(&leaves);
        assert_eq!(
            commitment.kv_roots[layer_idx], expected_root,
            "kv_root mismatch at layer {}",
            layer_idx
        );
    }

    // Each layer should have 3 entries (one per token).
    for layer_entries in &kv_entries {
        assert_eq!(layer_entries.len(), 3);
    }
}

/// kv_roots field in BatchCommitment defaults to empty via serde.
#[test]
fn kv_roots_default_empty() {
    use verilm_core::types::BatchCommitment;

    // JSON without kv_roots should deserialize with empty vec.
    let json = r#"{
        "merkle_root": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "io_root": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "n_tokens": 1,
        "version": "V4"
    }"#;
    let c: BatchCommitment = serde_json::from_str(json).unwrap();
    assert!(c.kv_roots.is_empty());
}

/// End-to-end: commit + open produces valid KV Merkle proofs in audit response.
#[test]
fn kv_open_produces_valid_merkle_proofs() {
    use verilm_core::merkle;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let _key = generate_key(&cfg, &model, [1u8; 32]);
    let initial: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let traces = verilm_test_vectors::forward_pass_autoregressive(&cfg, &model, &initial, 3);
    let kv_entries = verilm_test_vectors::kv_entries_from_traces(&cfg, &traces);

    let all_retained: Vec<_> = traces
        .iter()
        .map(|token_traces| verilm_core::types::RetainedTokenState {
            layers: token_traces
                .iter()
                .map(|lt| RetainedLayerState {
                    a: lt.a.clone(),
                    scale_a: lt.scale_a.unwrap_or(1.0),
                    x_attn_i8: None,
                    scale_x_attn: None,
                })
                .collect(),
        })
        .collect();

    let params = verilm_prover::FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"kv open test",
        sampling_seed: [9u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let scales = vec![
        vec![
            verilm_prover::CapturedLayerScales {
                scale_x_attn: 1.0,
                scale_x_ffn: 1.0,
                scale_h: 1.0
            };
            cfg.n_layers
        ];
        3
    ];

    let (_commitment, state) =
        commit_minimal(all_retained, &params, None, scales, None, Some(kv_entries), None);

    // Open audit for token 2 (last token) — should include KV for positions 0..=2.
    let response = verilm_prover::open_v4(
        &state,
        2,
        &ToyWeights(&model),
        &cfg,
        &[],
        &[],
        None,
        None,
        None,
        None,
        false,
        false,
    );

    // Verify KV entries and proofs are present.
    let kv_entries = response
        .kv_entries
        .as_ref()
        .expect("kv_entries must be present");
    let kv_proofs = response
        .kv_proofs
        .as_ref()
        .expect("kv_proofs must be present");
    assert_eq!(
        kv_entries.len(),
        cfg.n_layers,
        "one set of entries per layer"
    );
    assert_eq!(kv_proofs.len(), cfg.n_layers, "one set of proofs per layer");

    // Each layer should have 3 entries (positions 0, 1, 2).
    for (layer_idx, (entries, proofs)) in kv_entries.iter().zip(kv_proofs.iter()).enumerate() {
        assert_eq!(
            entries.len(),
            3,
            "layer {} should have 3 KV entries",
            layer_idx
        );
        assert_eq!(
            proofs.len(),
            3,
            "layer {} should have 3 KV proofs",
            layer_idx
        );

        let root = response.commitment.kv_roots[layer_idx];
        for (pos, (entry, proof)) in entries.iter().zip(proofs.iter()).enumerate() {
            let leaf = merkle::hash_kv_entry(layer_idx, pos, &entry.k_roped, &entry.v_deq);
            assert!(
                merkle::verify(&root, &leaf, proof),
                "KV proof invalid at layer {} pos {}",
                layer_idx,
                pos
            );
        }
    }
}

/// Tampered KV entry fails Merkle proof verification.
#[test]
fn kv_tampered_entry_fails_proof() {
    use verilm_core::merkle;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let initial: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let traces = verilm_test_vectors::forward_pass_autoregressive(&cfg, &model, &initial, 2);
    let kv_entries = verilm_test_vectors::kv_entries_from_traces(&cfg, &traces);

    let all_retained: Vec<_> = traces
        .iter()
        .map(|token_traces| verilm_core::types::RetainedTokenState {
            layers: token_traces
                .iter()
                .map(|lt| RetainedLayerState {
                    a: lt.a.clone(),
                    scale_a: lt.scale_a.unwrap_or(1.0),
                    x_attn_i8: None,
                    scale_x_attn: None,
                })
                .collect(),
        })
        .collect();

    let params = verilm_prover::FullBindingParams {
        token_ids: &[10, 20],
        prompt: b"kv tamper test",
        sampling_seed: [9u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let scales = vec![
        vec![
            verilm_prover::CapturedLayerScales {
                scale_x_attn: 1.0,
                scale_x_ffn: 1.0,
                scale_h: 1.0
            };
            cfg.n_layers
        ];
        2
    ];

    let (_commitment, state) =
        commit_minimal(all_retained, &params, None, scales, None, Some(kv_entries), None);

    let mut response = verilm_prover::open_v4(
        &state,
        1,
        &ToyWeights(&model),
        &cfg,
        &[],
        &[],
        None,
        None,
        None,
        None,
        false,
        false,
    );

    // Tamper with KV entry at layer 0, position 0.
    let kv = response.kv_entries.as_mut().unwrap();
    kv[0][0].k_roped[0] += 1.0;

    // Proof should now be invalid.
    let proofs = response.kv_proofs.as_ref().unwrap();
    let tampered_leaf = merkle::hash_kv_entry(0, 0, &kv[0][0].k_roped, &kv[0][0].v_deq);
    let root = response.commitment.kv_roots[0];
    assert!(
        !merkle::verify(&root, &tampered_leaf, &proofs[0][0]),
        "tampered KV entry must fail Merkle proof"
    );
}

// ===========================================================================
// Canonical verifier: phase_kv_transcript
// ===========================================================================

/// Helper: produce a toy audit response with KV entries committed and opened.
fn kv_toy_response(
    n_tokens: usize,
) -> (
    verilm_core::types::VerifierKey,
    verilm_core::types::V4AuditResponse,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let initial: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let traces = verilm_test_vectors::forward_pass_autoregressive(&cfg, &model, &initial, n_tokens);
    let kv_entries = verilm_test_vectors::kv_entries_from_traces(&cfg, &traces);

    let all_retained: Vec<_> = traces
        .iter()
        .map(|token_traces| RetainedTokenState {
            layers: token_traces
                .iter()
                .map(|lt| RetainedLayerState {
                    a: lt.a.clone(),
                    scale_a: lt.scale_a.unwrap_or(1.0),
                    x_attn_i8: None,
                    scale_x_attn: None,
                })
                .collect(),
        })
        .collect();

    let token_ids: Vec<u32> = (0..n_tokens as u32).map(|i| i + 10).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"kv canonical test",
        sampling_seed: [9u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let scales = vec![
        vec![
            CapturedLayerScales {
                scale_x_attn: 1.0,
                scale_x_ffn: 1.0,
                scale_h: 1.0,
            };
            cfg.n_layers
        ];
        n_tokens
    ];

    let (_commitment, state) =
        commit_minimal(all_retained, &params, None, scales, None, Some(kv_entries), None);

    let last_token = n_tokens - 1;
    let response = open_v4(
        &state,
        last_token as u32,
        &ToyWeights(&model),
        &cfg,
        &[],
        &[],
        None,
        None,
        None,
        None,
        false,
        false,
    );

    (key, response)
}

#[test]
fn kv_canonical_valid_proofs_pass() {
    let (key, response) = kv_toy_response(3);
    assert!(!response.commitment.kv_roots.is_empty());
    assert!(response.kv_entries.is_some());
    assert!(response.kv_proofs.is_some());

    let report = verify_response(&key, &response, None, None);
    // No KV-related failures.
    let kv_failures: Vec<_> = report
        .failures
        .iter()
        .filter(|f| {
            matches!(
                f.code,
                FailureCode::KvRootsCountMismatch
                    | FailureCode::KvEntriesCountMismatch
                    | FailureCode::KvProofInvalid
                    | FailureCode::KvProofCountMismatch
            )
        })
        .collect();
    assert!(
        kv_failures.is_empty(),
        "unexpected KV failures: {:?}",
        kv_failures
    );
}

#[test]
fn kv_canonical_tampered_entry_fails() {
    let (key, mut response) = kv_toy_response(3);

    // Tamper with a KV entry — the Merkle proof should fail.
    response.kv_entries.as_mut().unwrap()[0][0].k_roped[0] += 999.0;

    let report = verify_response(&key, &response, None, None);
    let kv_proof_failures: Vec<_> = report
        .failures
        .iter()
        .filter(|f| f.code == FailureCode::KvProofInvalid)
        .collect();
    assert!(
        !kv_proof_failures.is_empty(),
        "tampered KV entry should produce KvProofInvalid failure"
    );
}

#[test]
fn kv_canonical_roots_count_mismatch() {
    let (key, mut response) = kv_toy_response(3);

    // Remove one kv_root to trigger count mismatch.
    response.commitment.kv_roots.pop();

    let report = verify_response(&key, &response, None, None);
    let mismatch: Vec<_> = report
        .failures
        .iter()
        .filter(|f| f.code == FailureCode::KvRootsCountMismatch)
        .collect();
    assert!(
        !mismatch.is_empty(),
        "kv_roots count mismatch should produce KvRootsCountMismatch failure"
    );
}

#[test]
fn kv_canonical_entries_without_proofs_rejected() {
    let (key, mut response) = kv_toy_response(3);

    // Remove proofs but keep entries — structural error.
    response.kv_proofs = None;

    let report = verify_response(&key, &response, None, None);
    let mismatch: Vec<_> = report
        .failures
        .iter()
        .filter(|f| f.code == FailureCode::KvProofCountMismatch)
        .collect();
    assert!(
        !mismatch.is_empty(),
        "entries without proofs should produce KvProofCountMismatch"
    );
}

#[test]
fn kv_roots_without_opened_entries_are_allowed() {
    let (key, mut response) = kv_toy_response(3);

    response.kv_entries = None;
    response.kv_proofs = None;

    let report = verify_response(&key, &response, None, None);
    let kv_failures: Vec<_> = report
        .failures
        .iter()
        .filter(|f| {
            matches!(
                f.code,
                FailureCode::KvRootsCountMismatch
                    | FailureCode::KvEntriesCountMismatch
                    | FailureCode::KvProofInvalid
                    | FailureCode::KvProofCountMismatch
            )
        })
        .collect();
    assert!(
        kv_failures.is_empty(),
        "kv roots without opened entries should remain verifier-acceptable"
    );
}

#[test]
fn kv_canonical_legacy_no_kv_roots_skipped() {
    let (key, mut response) = kv_toy_response(3);

    // Clear KV data to simulate legacy response.
    response.commitment.kv_roots.clear();
    response.kv_entries = None;
    response.kv_proofs = None;

    let report = verify_response(&key, &response, None, None);
    // No KV-related failures — phase is silently skipped.
    let kv_failures: Vec<_> = report
        .failures
        .iter()
        .filter(|f| {
            matches!(
                f.code,
                FailureCode::KvRootsCountMismatch
                    | FailureCode::KvEntriesCountMismatch
                    | FailureCode::KvProofInvalid
                    | FailureCode::KvProofCountMismatch
            )
        })
        .collect();
    assert!(
        kv_failures.is_empty(),
        "legacy (no kv_roots) should skip KV phase"
    );
}

/// compute_kv_transcript produces the same KV entries as kv_entries_from_traces
/// for the toy model (validates that commit-time derivation from x_attn + weights
/// matches the reference path).
#[test]
fn kv_compute_transcript_matches_traces() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let initial: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let traces = verilm_test_vectors::forward_pass_autoregressive(&cfg, &model, &initial, 3);

    // Reference: KV entries from traces (the test-vectors path).
    let reference_kv = verilm_test_vectors::kv_entries_from_traces(&cfg, &traces);

    // Build x_attn per token/layer from traces (simulates GPU capture).
    let captured_x_attn: Vec<Vec<Vec<i8>>> = traces
        .iter()
        .map(|token_traces| token_traces.iter().map(|lt| lt.x_attn.clone()).collect())
        .collect();

    // Build captured scales (toy model uses unit scales).
    let captured_scales: Vec<Vec<CapturedLayerScales>> = traces
        .iter()
        .map(|token_traces| {
            token_traces
                .iter()
                .map(|_| CapturedLayerScales {
                    scale_x_attn: 1.0,
                    scale_x_ffn: 1.0,
                    scale_h: 1.0,
                })
                .collect()
        })
        .collect();

    // Compute KV transcript from x_attn + weights (the production commit-time path).
    let computed_kv = verilm_prover::compute_kv_transcript(
        &captured_x_attn,
        &ToyWeights(&model),
        &cfg,
        &captured_scales,
        &[], // no per-tensor scales (toy)
        &[], // no per-channel scales (toy)
        &[], // no QKV biases (toy)
    );

    // Both should have n_layers entries.
    assert_eq!(computed_kv.len(), reference_kv.len());

    // Each layer should have n_tokens entries, with identical KV values.
    for (layer_idx, (computed, reference)) in
        computed_kv.iter().zip(reference_kv.iter()).enumerate()
    {
        assert_eq!(
            computed.len(),
            reference.len(),
            "layer {} token count mismatch",
            layer_idx
        );
        for (pos, (c, r)) in computed.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                c.k_roped, r.k_roped,
                "layer {} pos {} k_roped mismatch",
                layer_idx, pos
            );
            assert_eq!(
                c.v_deq, r.v_deq,
                "layer {} pos {} v_deq mismatch",
                layer_idx, pos
            );
        }
    }
}

/// KV transcript derived from x_attn at commit time produces valid Merkle proofs
/// that the canonical verifier accepts.
#[test]
fn kv_derived_transcript_passes_canonical_verification() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let initial: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let traces = verilm_test_vectors::forward_pass_autoregressive(&cfg, &model, &initial, 3);

    // Build x_attn and retained states from traces.
    let captured_x_attn: Vec<Vec<Vec<i8>>> = traces
        .iter()
        .map(|token_traces| token_traces.iter().map(|lt| lt.x_attn.clone()).collect())
        .collect();

    let all_retained: Vec<_> = traces
        .iter()
        .map(|token_traces| RetainedTokenState {
            layers: token_traces
                .iter()
                .map(|lt| RetainedLayerState {
                    a: lt.a.clone(),
                    scale_a: lt.scale_a.unwrap_or(1.0),
                    x_attn_i8: None,
                    scale_x_attn: None,
                })
                .collect(),
        })
        .collect();

    let captured_scales: Vec<Vec<CapturedLayerScales>> = traces
        .iter()
        .map(|token_traces| {
            token_traces
                .iter()
                .map(|_| CapturedLayerScales {
                    scale_x_attn: 1.0,
                    scale_x_ffn: 1.0,
                    scale_h: 1.0,
                })
                .collect()
        })
        .collect();

    // Compute KV transcript from x_attn + weights.
    let kv_transcripts = verilm_prover::compute_kv_transcript(
        &captured_x_attn,
        &ToyWeights(&model),
        &cfg,
        &captured_scales,
        &[],
        &[],
        &[], // no QKV biases (toy)
    );

    let params = FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"kv derived test",
        sampling_seed: [9u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let scales_for_commit = captured_scales;

    let (_commitment, state) = commit_minimal(
        all_retained,
        &params,
        None,
        scales_for_commit,
        Some(captured_x_attn),
        Some(kv_transcripts),
        None,
    );

    // Open audit for last token.
    let response = open_v4(
        &state,
        2,
        &ToyWeights(&model),
        &cfg,
        &[],
        &[],
        None,
        None,
        None,
        None,
        false,
        false,
    );

    // Verify: KV roots present, entries + proofs present.
    assert!(!response.commitment.kv_roots.is_empty());
    assert!(response.kv_entries.is_some());
    assert!(response.kv_proofs.is_some());

    // Canonical verifier should accept.
    let report = verify_response(&key, &response, None, None);
    let kv_failures: Vec<_> = report
        .failures
        .iter()
        .filter(|f| {
            matches!(
                f.code,
                FailureCode::KvRootsCountMismatch
                    | FailureCode::KvEntriesCountMismatch
                    | FailureCode::KvProofInvalid
                    | FailureCode::KvProofCountMismatch
            )
        })
        .collect();
    assert!(
        kv_failures.is_empty(),
        "derived KV transcript should pass canonical verification, failures: {:?}",
        kv_failures
    );
}

/// Reproduce the GPU KV proof bug: hash_kv_entry uses raw f64 bytes,
/// so JSON round-trip must preserve bit-exact f64 values.
#[test]
fn kv_entry_json_round_trip_hash_stability() {
    use verilm_core::merkle;
    use verilm_core::types::KvEntry;

    // Simulate dequant+RoPE values: i32 * f32_scale → f64
    let k: Vec<f64> = (0..512)
        .map(|i| (i as i32 * 17 - 4000) as f64 * 0.0078125_f32 as f64)
        .collect();
    let v: Vec<f64> = (0..512)
        .map(|i| (i as i32 * 23 + 100) as f64 * 0.00390625_f32 as f64)
        .collect();

    let entry = KvEntry {
        k_roped: k,
        v_deq: v,
    };

    let hash_before = merkle::hash_kv_entry(0, 0, &entry.k_roped, &entry.v_deq);

    // JSON round-trip (this is what happens on the audit_v4 → verify_v4 path)
    let json = serde_json::to_string(&entry).unwrap();
    let restored: KvEntry = serde_json::from_str(&json).unwrap();

    // Check bit-exact preservation of every f64
    for (i, (orig, rt)) in entry
        .k_roped
        .iter()
        .zip(restored.k_roped.iter())
        .enumerate()
    {
        assert_eq!(
            orig.to_bits(),
            rt.to_bits(),
            "k_roped[{}]: bits differ: {:016x} vs {:016x}",
            i,
            orig.to_bits(),
            rt.to_bits()
        );
    }
    for (i, (orig, rt)) in entry.v_deq.iter().zip(restored.v_deq.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            rt.to_bits(),
            "v_deq[{}]: bits differ: {:016x} vs {:016x}",
            i,
            orig.to_bits(),
            rt.to_bits()
        );
    }

    let hash_after = merkle::hash_kv_entry(0, 0, &restored.k_roped, &restored.v_deq);
    assert_eq!(
        hash_before, hash_after,
        "KV entry hash must survive JSON round-trip"
    );
}

#[test]
fn kv_entry_json_round_trip_rope_values() {
    // Test with values that come from RoPE (sin/cos) — more realistic than simple integer products.
    use verilm_core::merkle;
    use verilm_core::types::KvEntry;

    // Simulate RoPE-derived K values: int_accum * scale * cos(theta) + rotated * scale * sin(theta)
    let k: Vec<f64> = (0..512)
        .map(|i| {
            let theta = (i as f64) * 0.001 / (10000.0_f64).powf((2.0 * (i % 64) as f64) / 128.0);
            let accum = (i as i32 * 37 - 9000) as f64;
            let scale = 0.00390625_f64; // typical w8a8 scale
            accum * scale * theta.cos() + (accum * 0.5) * scale * theta.sin()
        })
        .collect();
    let v: Vec<f64> = (0..512)
        .map(|i| ((i as i32 * 23 + 100) as f64) * 0.0078125_f64)
        .collect();

    let entry = KvEntry {
        k_roped: k,
        v_deq: v,
    };

    let hash_before = merkle::hash_kv_entry(0, 0, &entry.k_roped, &entry.v_deq);

    let json = serde_json::to_string(&entry).unwrap();
    let restored: KvEntry = serde_json::from_str(&json).unwrap();

    let mut mismatches = 0;
    for (i, (orig, rt)) in entry.k_roped.iter().zip(restored.k_roped.iter()).enumerate() {
        if orig.to_bits() != rt.to_bits() {
            if mismatches < 5 {
                eprintln!(
                    "k_roped[{}]: orig={:.17e} ({:016x}) rt={:.17e} ({:016x})",
                    i, orig, orig.to_bits(), rt, rt.to_bits()
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{} k_roped values changed after JSON round-trip", mismatches);

    let hash_after = merkle::hash_kv_entry(0, 0, &restored.k_roped, &restored.v_deq);
    assert_eq!(hash_before, hash_after, "KV hash must survive JSON round-trip");
}

#[test]
fn kv_derived_transcript_full_response_json_round_trip_passes() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let initial: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let traces = verilm_test_vectors::forward_pass_autoregressive(&cfg, &model, &initial, 3);

    let captured_x_attn: Vec<Vec<Vec<i8>>> = traces
        .iter()
        .map(|token_traces| token_traces.iter().map(|lt| lt.x_attn.clone()).collect())
        .collect();

    let all_retained: Vec<_> = traces
        .iter()
        .map(|token_traces| RetainedTokenState {
            layers: token_traces
                .iter()
                .map(|lt| RetainedLayerState {
                    a: lt.a.clone(),
                    scale_a: lt.scale_a.unwrap_or(1.0),
                    x_attn_i8: None,
                    scale_x_attn: None,
                })
                .collect(),
        })
        .collect();

    let captured_scales: Vec<Vec<CapturedLayerScales>> = traces
        .iter()
        .map(|token_traces| {
            token_traces
                .iter()
                .map(|_| CapturedLayerScales {
                    scale_x_attn: 1.0,
                    scale_x_ffn: 1.0,
                    scale_h: 1.0,
                })
                .collect()
        })
        .collect();

    let kv_transcripts = verilm_prover::compute_kv_transcript(
        &captured_x_attn,
        &ToyWeights(&model),
        &cfg,
        &captured_scales,
        &[],
        &[],
        &[],
    );

    let params = FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"kv derived json roundtrip",
        sampling_seed: [9u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(
        all_retained,
        &params,
        None,
        captured_scales,
        Some(captured_x_attn),
        Some(kv_transcripts),
        None,
    );

    let response = open_v4(
        &state,
        2,
        &ToyWeights(&model),
        &cfg,
        &[],
        &[],
        None,
        None,
        None,
        None,
        false,
        false,
    );

    let json = serde_json::to_string(&response).unwrap();
    let restored: verilm_core::types::V4AuditResponse = serde_json::from_str(&json).unwrap();
    let report = verify_response(&key, &restored, None, None);
    let kv_failures: Vec<_> = report
        .failures
        .iter()
        .filter(|f| {
            matches!(
                f.code,
                FailureCode::KvRootsCountMismatch
                    | FailureCode::KvEntriesCountMismatch
                    | FailureCode::KvProofInvalid
                    | FailureCode::KvProofCountMismatch
            )
        })
        .collect();
    assert!(
        kv_failures.is_empty(),
        "full-response JSON round-trip should preserve derived KV transcript: {:?}",
        report.failures,
    );
}

#[test]
fn kv_derived_transcript_binary_round_trip_passes() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let initial: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let traces = verilm_test_vectors::forward_pass_autoregressive(&cfg, &model, &initial, 3);

    let captured_x_attn: Vec<Vec<Vec<i8>>> = traces
        .iter()
        .map(|token_traces| token_traces.iter().map(|lt| lt.x_attn.clone()).collect())
        .collect();

    let all_retained: Vec<_> = traces
        .iter()
        .map(|token_traces| RetainedTokenState {
            layers: token_traces
                .iter()
                .map(|lt| RetainedLayerState {
                    a: lt.a.clone(),
                    scale_a: lt.scale_a.unwrap_or(1.0),
                    x_attn_i8: None,
                    scale_x_attn: None,
                })
                .collect(),
        })
        .collect();

    let captured_scales: Vec<Vec<CapturedLayerScales>> = traces
        .iter()
        .map(|token_traces| {
            token_traces
                .iter()
                .map(|_| CapturedLayerScales {
                    scale_x_attn: 1.0,
                    scale_x_ffn: 1.0,
                    scale_h: 1.0,
                })
                .collect()
        })
        .collect();

    let kv_transcripts = verilm_prover::compute_kv_transcript(
        &captured_x_attn,
        &ToyWeights(&model),
        &cfg,
        &captured_scales,
        &[],
        &[],
        &[],
    );

    let params = FullBindingParams {
        token_ids: &[10, 20, 30],
        prompt: b"kv derived binary roundtrip",
        sampling_seed: [9u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(
        all_retained,
        &params,
        None,
        captured_scales,
        Some(captured_x_attn),
        Some(kv_transcripts),
        None,
    );

    let response = open_v4(
        &state,
        2,
        &ToyWeights(&model),
        &cfg,
        &[],
        &[],
        None,
        None,
        None,
        None,
        false,
        false,
    );

    let binary = verilm_core::serialize::serialize_v4_audit(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    let kv_failures: Vec<_> = report
        .failures
        .iter()
        .filter(|f| {
            matches!(
                f.code,
                FailureCode::KvRootsCountMismatch
                    | FailureCode::KvEntriesCountMismatch
                    | FailureCode::KvProofInvalid
                    | FailureCode::KvProofCountMismatch
            )
        })
        .collect();
    assert!(
        kv_failures.is_empty(),
        "binary round-trip should preserve derived KV transcript: {:?}",
        report.failures,
    );
}
