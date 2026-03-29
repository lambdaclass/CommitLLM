//! Golden conformance vectors for the VeriLM protocol.
//!
//! These tests pin exact byte-level outputs of protocol-critical functions.
//! Any silent change in hashing, challenge derivation, commitment, serialization,
//! or verification will break these tests. That is intentional: conformance
//! vectors are the canonical reference for future consumers and independent
//! verifier implementations.
//!
//! # Coverage
//!
//! - **Challenge derivation**: `build_audit_challenge` token index and layer
//!   indices for known seeds, token counts, and layer counts.
//! - **Manifest hashing**: `hash_manifest` and `hash_model_spec` for a known
//!   deployment manifest with architecture fields.
//! - **End-to-end commitment + verification**: deterministic commit → open →
//!   verify with pinned merkle roots, IO roots, verdict, and checks_run count.
//! - **Binary format**: serialized V4AuditResponse roundtrips, unknown magic
//!   rejected, truncated payloads rejected, version field preserved.

use verilm_core::constants::ModelConfig;
use verilm_core::types::{
    AuditTier, DeploymentManifest, RetainedLayerState, RetainedTokenState, ShellWeights,
};
use verilm_prover::{commit_minimal, open_v4, CapturedLayerScales, FullBindingParams};
use verilm_test_vectors::{forward_pass, generate_key, generate_model, LayerWeights};

// =========================================================================
// Helpers
// =========================================================================

struct ToyWeights<'a>(&'a [LayerWeights]);

impl ShellWeights for ToyWeights<'_> {
    fn weight(&self, layer: usize, mt: verilm_core::constants::MatrixType) -> &[i8] {
        let lw = &self.0[layer];
        match mt {
            verilm_core::constants::MatrixType::Wq => &lw.wq,
            verilm_core::constants::MatrixType::Wk => &lw.wk,
            verilm_core::constants::MatrixType::Wv => &lw.wv,
            verilm_core::constants::MatrixType::Wo => &lw.wo,
            verilm_core::constants::MatrixType::Wg => &lw.wg,
            verilm_core::constants::MatrixType::Wu => &lw.wu,
            verilm_core::constants::MatrixType::Wd => &lw.wd,
            verilm_core::constants::MatrixType::LmHead => panic!("ToyWeights: no LmHead"),
        }
    }
}

fn retained_from_traces(traces: &[verilm_core::types::LayerTrace]) -> RetainedTokenState {
    RetainedTokenState {
        layers: traces
            .iter()
            .map(|lt| RetainedLayerState {
                a: lt.a.clone(),
                scale_a: lt.scale_a.unwrap_or(1.0),
            })
            .collect(),
    }
}

fn unit_scales(n_layers: usize) -> Vec<CapturedLayerScales> {
    vec![CapturedLayerScales { scale_x_attn: 1.0, scale_x_ffn: 1.0, scale_h: 1.0 }; n_layers]
}

// =========================================================================
// 1. Challenge derivation golden vectors (roadmap #29, #30)
// =========================================================================

/// Pin the exact token_index and layer_indices produced by `build_audit_challenge`
/// for known seeds, token counts, layer counts, and tiers.
///
/// Challenge protocol:
///   token_index = u32_le(SHA256("vi-audit-token-v1" || seed)[0..4]) % n_tokens
///   routine layers = 0..=L_max where
///     L_max = max(min_prefix-1, u32_le(SHA256("vi-audit-prefix-v1" || seed || token_index_le32)[0..4]) % n_layers)
///     min_prefix = min(10, n_layers)
///   full layers = 0..n_layers
#[test]
fn golden_challenge_derivation() {
    let cases: &[([u8; 32], u32, usize, u32, usize)] = &[
        // (seed, n_tokens, n_layers, expected_token_index, expected_routine_n_layers)
        ([0x00; 32], 10, 32, 9, 12),
        ([0x2a; 32], 100, 80, 70, 20),
        ([0xff; 32], 1, 2, 0, 2),
        ([0x07; 32], 50, 16, 7, 10),
    ];
    for &(seed, n_tokens, n_layers, exp_token, exp_routine_layers) in cases {
        let routine = verilm_verify::build_audit_challenge(&seed, n_tokens, n_layers, AuditTier::Routine);
        assert_eq!(
            routine.token_index, exp_token,
            "token_index drift: seed=0x{:02x} n_tokens={}", seed[0], n_tokens
        );
        assert_eq!(
            routine.layer_indices.len(), exp_routine_layers,
            "routine layer count drift: seed=0x{:02x}", seed[0]
        );
        // Routine layers must be contiguous 0..=L_max
        for (i, &l) in routine.layer_indices.iter().enumerate() {
            assert_eq!(l, i, "routine layers must be contiguous prefix");
        }

        let full = verilm_verify::build_audit_challenge(&seed, n_tokens, n_layers, AuditTier::Full);
        assert_eq!(full.token_index, exp_token, "full tier must pick same token");
        assert_eq!(full.layer_indices.len(), n_layers, "full tier must include all layers");
    }
}

// =========================================================================
// 2. Manifest and model-spec hash golden vectors (roadmap #29)
// =========================================================================

fn golden_manifest() -> DeploymentManifest {
    DeploymentManifest {
        tokenizer_hash: [0x01; 32],
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        eos_policy: "stop".into(),
        weight_hash: Some([0x02; 32]),
        quant_hash: None,
        system_prompt_hash: None,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        logit_bias: vec![],
        bad_word_ids: vec![],
        guided_decoding: String::new(),
        stop_sequences: vec![],
        max_tokens: 256,
        chat_template_hash: None,
        rope_config_hash: Some([0x03; 32]),
        rmsnorm_eps: Some(1e-5),
        sampler_version: Some("chacha20-vi-sample-v1".into()),
        bos_eos_policy: None,
        truncation_policy: None,
        special_token_policy: None,
        adapter_hash: None,
        n_layers: Some(32),
        hidden_dim: Some(4096),
        vocab_size: Some(32000),
        embedding_merkle_root: Some([0x04; 32]),
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
fn golden_manifest_hash() {
    let manifest = golden_manifest();
    let h = verilm_core::merkle::hash_manifest(&manifest);
    assert_eq!(
        hex::encode(h),
        "79b3b497b0fbe9224b099a0c07b2cf181c68c701187c2fd231d34579c0670f1b",
        "manifest hash drifted — was the four-spec hash format changed?"
    );
}

#[test]
fn golden_model_spec_hash() {
    let manifest = golden_manifest();
    let (_, model_spec, _, _) = manifest.split();
    let h = verilm_core::merkle::hash_model_spec(&model_spec);
    assert_eq!(
        hex::encode(h),
        "4a5a121ff13fedf3d6f6425598ea00346effef40583ce0c519dead3c176d4890",
        "model_spec hash drifted — was the model-spec hash format changed?"
    );
}

// =========================================================================
// 3. End-to-end commitment + verification golden vectors (roadmap #29)
// =========================================================================

/// Deterministic setup: fixed model seed, fixed key seed, fixed input, fixed params.
/// Pin the commitment roots, verification verdict, and checks_run count.
#[test]
fn golden_e2e_verify() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"golden test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (commitment, state) = commit_minimal(vec![retained], &params, None, vec![unit_scales(cfg.n_layers)], None);
    let response = open_v4(
        &state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false,
    );

    // Pin commitment roots
    assert_eq!(
        hex::encode(commitment.merkle_root),
        "e18abb9e62d6267fc6b393da6f6c11e9c7ecc6adc46a7b7bf98ad1d0782d65c1",
        "merkle_root drifted — was retained-state hashing or commitment changed?"
    );
    assert_eq!(
        hex::encode(commitment.io_root),
        "e749d661577510798f38e29a30fc39f365a2797bb2448e16bafe1c6b40072f02",
        "io_root drifted — was IO chain hashing or prompt binding changed?"
    );

    // Pin verification outcome
    let report = verilm_verify::verify_v4_legacy(&key, &response, None, None, None);
    assert_eq!(
        report.verdict,
        verilm_verify::Verdict::Pass,
        "golden e2e verify failed: {:?}",
        report.failures
    );
    assert_eq!(
        report.checks_run, 19,
        "checks_run count drifted — was a check added or removed?"
    );
    assert!(report.failures.is_empty());
}

// =========================================================================
// 4. Binary format tests (roadmap #31)
// =========================================================================

/// V4AuditResponse roundtrips through serialize + deserialize.
#[test]
fn binary_v4_audit_roundtrip() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"binary roundtrip",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![unit_scales(cfg.n_layers)], None);
    let response = open_v4(
        &state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false,
    );

    let binary = verilm_core::serialize::serialize_v4_audit(&response);

    // Magic bytes must be "VV4A"
    assert_eq!(&binary[..4], b"VV4A", "wrong magic bytes");

    // Roundtrip
    let restored = verilm_core::serialize::deserialize_v4_audit(&binary).unwrap();
    assert_eq!(restored.token_index, response.token_index);
    assert_eq!(restored.token_id, response.token_id);
    assert_eq!(restored.retained, response.retained);
    assert_eq!(restored.commitment.merkle_root, response.commitment.merkle_root);
    assert_eq!(restored.commitment.io_root, response.commitment.io_root);
    assert_eq!(restored.prev_io_hash, response.prev_io_hash);

    // Restored response must still verify
    let report = verilm_verify::verify_v4_legacy(&key, &restored, None, None, None);
    assert_eq!(
        report.verdict,
        verilm_verify::Verdict::Pass,
        "roundtripped response failed verification: {:?}",
        report.failures
    );
}

/// Unknown magic bytes are rejected (forward compatibility: fail-closed).
#[test]
fn binary_unknown_magic_rejected() {
    assert!(verilm_core::serialize::deserialize_v4_audit(b"VV5Axyz").is_err());
    assert!(verilm_core::serialize::deserialize_v4_audit(b"BAAD1234567890").is_err());
    assert!(verilm_core::serialize::deserialize_key(b"VKEX1234567890").is_err());
}

/// Truncated payloads are rejected cleanly (no panics).
#[test]
fn binary_truncated_payload_rejected() {
    // Just magic, no body
    assert!(verilm_core::serialize::deserialize_v4_audit(b"VV4A").is_err());
    // Partial body (not valid zstd)
    assert!(verilm_core::serialize::deserialize_v4_audit(b"VV4A\x00\x01\x02").is_err());
    // Empty
    assert!(verilm_core::serialize::deserialize_v4_audit(b"").is_err());
    // Too short for magic
    assert!(verilm_core::serialize::deserialize_v4_audit(b"VV").is_err());
}

/// CommitmentVersion is preserved through serialization.
#[test]
fn binary_commitment_version_preserved() {
    use verilm_core::types::CommitmentVersion;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"version test",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None, vec![unit_scales(cfg.n_layers)], None);
    let response = open_v4(
        &state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false,
    );

    let binary = verilm_core::serialize::serialize_v4_audit(&response);
    let restored = verilm_core::serialize::deserialize_v4_audit(&binary).unwrap();

    // Version field must survive roundtrip
    match restored.commitment.version {
        CommitmentVersion::V4 => {} // expected
    }
}

/// Verifier key roundtrips through binary serialization.
#[test]
fn binary_key_roundtrip() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let binary = verilm_core::serialize::serialize_key(&key);
    assert_eq!(&binary[..4], b"VKEY", "wrong key magic");

    let restored = verilm_core::serialize::deserialize_key(&binary).unwrap();
    assert_eq!(restored.version, key.version);
    assert_eq!(restored.config.n_layers, key.config.n_layers);
    assert_eq!(restored.config.hidden_dim, key.config.hidden_dim);
    assert_eq!(restored.config.vocab_size, key.config.vocab_size);
    assert_eq!(restored.rmsnorm_eps, key.rmsnorm_eps);
    assert_eq!(restored.weight_hash, key.weight_hash);
    assert_eq!(restored.embedding_merkle_root, key.embedding_merkle_root);
}

/// Cross-format rejection: key bytes fed to audit deserializer and vice versa.
#[test]
fn binary_cross_format_rejected() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let key_binary = verilm_core::serialize::serialize_key(&key);
    // Key binary should not deserialize as V4 audit
    assert!(verilm_core::serialize::deserialize_v4_audit(&key_binary).is_err());
}
