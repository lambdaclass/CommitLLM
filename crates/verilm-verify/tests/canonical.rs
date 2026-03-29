//! Tests for the canonical verifier (binary-only, key-only, full-bridge-only).

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{
    BridgeParams, DeploymentManifest, RetainedLayerState, RetainedTokenState, ShellWeights,
};
use verilm_prover::{commit_minimal, open_v4, FullBindingParams};
use verilm_test_vectors::{generate_key, generate_model, LayerWeights};
use verilm_verify::canonical::verify_binary;
use verilm_verify::{verify_v4_legacy, Verdict};

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
) -> RetainedTokenState {
    use verilm_core::matmul::matmul_i32;
    use verilm_core::rmsnorm::{
        bridge_residual_rmsnorm, dequant_add_residual, quantize_f64_to_i8, rmsnorm_f64_input,
    };

    let mut residual: Vec<f64> = initial_residual.iter().map(|&v| v as f64).collect();
    let mut layers = Vec::new();
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

        layers.push(RetainedLayerState { a, scale_a, scale_x_attn, scale_x_ffn, scale_h });
    }

    RetainedTokenState { layers }
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
        prompt: b"canonical test",
        sampling_seed: [7u8; 32],
        manifest,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(
        &state, 0, &ToyWeights(&model), &cfg, &ws,
        Some(&bridge), None, None, None, false,
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
        prompt: b"canonical test",
        sampling_seed: [7u8; 32],
        manifest,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(
        &state, 0, &ToyWeights(&model), &cfg, &ws,
        Some(&bridge), None, None, None, false,
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
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Full bridge: structural + embedding + spec hashes + per-layer Freivalds
    assert!(report.checks_run >= 12, "too few checks: {}", report.checks_run);
}

#[test]
fn canonical_missing_manifest_rejected() {
    let (key, binary) = build_canonical_audit(None);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("manifest")));
}

#[test]
fn canonical_full_bridge_with_manifest_pass() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    // Should include spec hash checks
    assert!(report.checks_run >= 12, "too few checks with manifest: {}", report.checks_run);
}

#[test]
fn canonical_binary_magic_validated() {
    let (key, _) = build_canonical_audit(None);
    // Bad magic
    let err = verify_binary(&key, b"XXXX1234567890", None, None);
    assert!(err.is_err(), "should reject bad magic");
    assert!(err.unwrap_err().contains("magic"), "error should mention magic");
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
    assert!(report.failures.iter().any(|f| f.message.contains("shell_opening")));
}

#[test]
fn canonical_missing_initial_residual_rejected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    response.shell_opening.as_mut().unwrap().initial_residual = None;
    // Must recompute leaf hash since it includes final_residual but not initial_residual
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("initial_residual")));
}

#[test]
fn canonical_missing_embedding_proof_rejected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    response.shell_opening.as_mut().unwrap().embedding_proof = None;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("embedding_proof")));
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
    assert!(report.failures.iter().any(|f| f.message.contains("Freivalds")));
}

#[test]
fn canonical_tampered_ffn_out_detected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers[0].ffn_out[0] ^= 0x7FFF_FFFF;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.message.contains("Freivalds")));
}

#[test]
fn canonical_tampered_seed_detected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    response.revealed_seed[0] ^= 0xFF;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| {
        f.code == verilm_verify::FailureCode::SeedMismatch
    }));
}

#[test]
fn canonical_tampered_embedding_detected() {
    let (key, mut response) = build_canonical_audit_with_response(None);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.initial_residual.as_mut().unwrap()[0] += 1.0;
    let binary = to_binary(&response);
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| {
        f.code == verilm_verify::FailureCode::EmbeddingProofFailed
    }));
}

// ---------------------------------------------------------------------------
// Spec binding tests
// ---------------------------------------------------------------------------

#[test]
fn canonical_manifest_spec_hash_verified() {
    let manifest = make_manifest(0.0, 0, 1.0);
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn canonical_unsupported_sampler_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.sampler_version = Some("custom-sampler-v99".into());
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| {
        f.code == verilm_verify::FailureCode::UnsupportedSamplerVersion
    }));
}

#[test]
fn canonical_unsupported_repetition_penalty_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.repetition_penalty = 1.5;
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| {
        f.code == verilm_verify::FailureCode::UnsupportedDecodeFeature
    }));
}

#[test]
fn canonical_decode_mode_inconsistency_rejected() {
    let mut manifest = make_manifest(0.0, 0, 1.0);
    manifest.decode_mode = Some("sampled".into()); // temp=0 but mode=sampled
    let (key, binary) = build_canonical_audit(Some(&manifest));
    let report = verify_binary(&key, &binary, None, None).unwrap();
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| {
        f.code == verilm_verify::FailureCode::DecodeModeTempInconsistent
    }));
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
    assert!(report.checks_run >= 20, "expected >= 20 checks, got {}", report.checks_run);
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

    let legacy_codes: std::collections::BTreeSet<_> =
        legacy.failures.iter().map(|f| format!("{:?}", f.code)).collect();
    let canonical_codes: std::collections::BTreeSet<_> =
        canonical.failures.iter().map(|f| format!("{:?}", f.code)).collect();

    assert_eq!(
        legacy_codes, canonical_codes,
        "failure code mismatch:\n  legacy only: {:?}\n  canonical only: {:?}",
        legacy_codes.difference(&canonical_codes).collect::<Vec<_>>(),
        canonical_codes.difference(&legacy_codes).collect::<Vec<_>>(),
    );
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
    response.shell_opening.as_mut().unwrap().initial_residual.as_mut().unwrap()[0] += 1.0;
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
