//! Boundary-condition fuzzing tests for the V4 verifier (roadmap #8).
//!
//! These exercise edge cases that should fail correctly with specific failure
//! codes/categories. Covers:
//!
//! - Malformed/truncated binary payloads
//! - Unknown versions / bad magic
//! - EOS-shortened requests
//! - Prompt/output length edge cases
//! - Weird token_index / prefix-count / layer_indices combinations
//! - Missing required fields / malformed shapes

use verilm_core::constants::ModelConfig;
use verilm_core::merkle::MerkleProof;
use verilm_core::serialize;
use verilm_core::types::{
    DeploymentManifest, RetainedLayerState, RetainedTokenState, V4AuditResponse,
};
use verilm_prover::{commit_minimal, open_v4, CapturedLayerScales, FullBindingParams};
use verilm_test_vectors::{forward_pass, generate_key, generate_model, LayerWeights};
use verilm_verify::{verify_v4_legacy, AuditCoverage, FailureCategory, FailureCode, Verdict};

/// Thin wrapper: routes through the legacy verifier while canonical is validated.
fn verify_v4(
    key: &verilm_core::types::VerifierKey,
    response: &verilm_core::types::V4AuditResponse,
    expected_prompt_token_ids: Option<&[u32]>,
) -> verilm_verify::V4VerifyReport {
    verify_v4_legacy(key, response, expected_prompt_token_ids, None, None)
}

// ── Helpers ──────────────────────────────────────────────────

struct ToyWeights<'a>(&'a [LayerWeights]);

impl verilm_core::types::ShellWeights for ToyWeights<'_> {
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
            verilm_core::constants::MatrixType::LmHead => panic!("LmHead not per-layer"),
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
                x_attn_i8: None,
                scale_x_attn: None,
            })
            .collect(),
    }
}

fn unit_scales(n_layers: usize) -> Vec<CapturedLayerScales> {
    vec![CapturedLayerScales { scale_x_attn: 1.0, scale_x_ffn: 1.0, scale_h: 1.0 }; n_layers]
}

/// Build a valid V4AuditResponse via the prover, ready for mutation.
fn make_valid_response(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    n_tokens: usize,
) -> (verilm_core::types::VerifierKey, V4AuditResponse) {
    let key = generate_key(cfg, model, [1u8; 32]);
    let inputs: Vec<Vec<i8>> = (0..n_tokens)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();
    let all_retained: Vec<RetainedTokenState> = inputs
        .iter()
        .map(|inp| retained_from_traces(&forward_pass(cfg, model, inp)))
        .collect();
    let token_ids: Vec<u32> = (0..n_tokens as u32).map(|i| 42 + i).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"boundary test prompt",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let n_tok = all_retained.len();
    let all_scales = vec![unit_scales(cfg.n_layers); n_tok];
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None);
    // Open token 0 (has no prefix).
    let response = open_v4(&state, 0, &ToyWeights(model), cfg, &[], &[], None, None, None, None, false, false);
    (key, response)
}

/// Like make_valid_response but opens a specific token_index (for prefix tests).
fn make_valid_response_at(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    n_tokens: usize,
    token_index: u32,
) -> (verilm_core::types::VerifierKey, V4AuditResponse) {
    let key = generate_key(cfg, model, [1u8; 32]);
    let inputs: Vec<Vec<i8>> = (0..n_tokens)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();
    let all_retained: Vec<RetainedTokenState> = inputs
        .iter()
        .map(|inp| retained_from_traces(&forward_pass(cfg, model, inp)))
        .collect();
    let token_ids: Vec<u32> = (0..n_tokens as u32).map(|i| 42 + i).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"boundary test prompt",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let n_tok = all_retained.len();
    let all_scales = vec![unit_scales(cfg.n_layers); n_tok];
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None);
    let response = open_v4(&state, token_index, &ToyWeights(model), cfg, &[], &[], None, None, None, None, false, false);
    (key, response)
}

fn make_manifest(temperature: f32) -> DeploymentManifest {
    DeploymentManifest {
        tokenizer_hash: [0u8; 32],
        temperature,
        top_k: 0,
        top_p: 1.0,
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

fn has_code(report: &verilm_verify::V4VerifyReport, code: FailureCode) -> bool {
    report.failures.iter().any(|f| f.code == code)
}

fn has_category(report: &verilm_verify::V4VerifyReport, cat: FailureCategory) -> bool {
    report.failures.iter().any(|f| f.category == cat)
}

// ===========================================================================
// 1. Malformed / truncated binary payloads
// ===========================================================================

#[test]
fn binary_empty_input_rejected() {
    assert!(serialize::deserialize_v4_audit(&[]).is_err());
}

#[test]
fn binary_just_magic_rejected() {
    assert!(serialize::deserialize_v4_audit(b"VV4A").is_err());
}

#[test]
fn binary_magic_plus_one_byte_rejected() {
    assert!(serialize::deserialize_v4_audit(b"VV4A\x00").is_err());
}

#[test]
fn binary_wrong_magic_rejected() {
    let mut data = b"VV4A\x00\x00\x00\x00".to_vec();
    data[0] = b'X'; // corrupt first byte
    assert!(serialize::deserialize_v4_audit(&data).is_err());
}

#[test]
fn binary_key_magic_as_audit_rejected() {
    // VKEY magic fed to audit deserializer.
    assert!(serialize::deserialize_v4_audit(b"VKEY\x00\x00\x00\x00").is_err());
}

#[test]
fn binary_truncated_at_every_offset() {
    // Serialize a valid response, then try every possible truncation.
    let (cfg, model) = {
        let cfg = ModelConfig::toy();
        let model = generate_model(&cfg, 12345);
        (cfg, model)
    };
    let (_key, response) = make_valid_response(&cfg, &model, 1);
    let binary = serialize::serialize_v4_audit(&response);

    // Truncate at every offset from 0 to len-1. None should panic.
    for cutoff in 0..binary.len() {
        let result = serialize::deserialize_v4_audit(&binary[..cutoff]);
        assert!(result.is_err(), "truncation at byte {} should fail", cutoff);
    }
    // Full payload must succeed.
    assert!(serialize::deserialize_v4_audit(&binary).is_ok());
}

#[test]
fn binary_random_garbage_after_magic() {
    // Valid magic + random bytes that aren't valid zstd.
    let mut data = b"VV4A".to_vec();
    data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x42, 0x42, 0x42]);
    assert!(serialize::deserialize_v4_audit(&data).is_err());
}

#[test]
fn binary_valid_zstd_wrong_bincode() {
    // Valid magic + valid zstd of garbage bincode content.
    let garbage = vec![0xABu8; 64];
    let compressed = zstd::encode_all(garbage.as_slice(), 3).unwrap();
    let mut data = b"VV4A".to_vec();
    data.extend_from_slice(&compressed);
    assert!(serialize::deserialize_v4_audit(&data).is_err());
}

#[test]
fn binary_single_bit_flip_in_payload() {
    let (cfg, model) = {
        let cfg = ModelConfig::toy();
        let model = generate_model(&cfg, 12345);
        (cfg, model)
    };
    let (_key, response) = make_valid_response(&cfg, &model, 1);
    let binary = serialize::serialize_v4_audit(&response);

    // Flip one bit in the compressed payload region. Most flips should break
    // either zstd decompression or bincode deserialization.
    let mut failures = 0;
    for byte_idx in 4..binary.len() {
        let mut corrupted = binary.clone();
        corrupted[byte_idx] ^= 0x01;
        if serialize::deserialize_v4_audit(&corrupted).is_err() {
            failures += 1;
        }
    }
    // At least 25% of single-bit flips should break deserialization.
    // (zstd's frame structure absorbs some flips that produce valid-but-wrong
    // decompressed output, which may still partially parse as bincode.)
    let payload_len = binary.len() - 4;
    assert!(
        failures > payload_len / 4,
        "only {}/{} bit flips caused failure — too fragile",
        failures,
        payload_len
    );
}

// ===========================================================================
// 2. Commitment version / structural boundaries
// ===========================================================================

#[test]
fn structural_missing_seed_commitment() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.commitment.seed_commitment = None;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MissingSeedCommitment));
}

#[test]
fn structural_missing_prompt_hash() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.commitment.prompt_hash = None;
    response.prompt = None;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MissingPromptHash));
}

#[test]
fn structural_prompt_bytes_without_committed_hash() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.commitment.prompt_hash = None;
    // prompt bytes present but no committed hash.

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::UncommittedPrompt));
}

#[test]
fn structural_committed_hash_without_prompt_bytes() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    // Keep prompt_hash but strip prompt bytes.
    response.prompt = None;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MissingPromptBytes));
}

#[test]
fn structural_missing_n_prompt_tokens_in_commitment() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.commitment.n_prompt_tokens = None;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MissingNPromptTokens));
}

#[test]
fn structural_missing_n_prompt_tokens_in_response() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.n_prompt_tokens = None;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MissingNPromptTokens));
}

#[test]
fn structural_n_prompt_tokens_mismatch() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    // Commitment says 1, response says 2.
    response.n_prompt_tokens = Some(2);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::NPromptTokensMismatch));
}

#[test]
fn structural_n_prompt_tokens_exceeds_n_tokens() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    // n_tokens=1, set n_prompt_tokens to 100 (way exceeds n_tokens+1).
    response.commitment.n_prompt_tokens = Some(100);
    response.n_prompt_tokens = Some(100);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::NPromptTokensBound));
}

// ===========================================================================
// 3. Prefix count / token_index boundary conditions
// ===========================================================================

#[test]
fn prefix_count_mismatch_too_few() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response_at(&cfg, &model, 3, 2);
    // token_index=2 expects 2 prefix leaves. Remove one.
    response.prefix_leaf_hashes.pop();
    response.prefix_merkle_proofs.pop();
    response.prefix_token_ids.pop();

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::PrefixTokenCountMismatch));
}

#[test]
fn prefix_count_mismatch_too_many() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response_at(&cfg, &model, 3, 1);
    // token_index=1 expects 1 prefix leaf. Add an extra.
    response.prefix_leaf_hashes.push([0xAA; 32]);
    response.prefix_merkle_proofs.push(MerkleProof { leaf_index: 99, siblings: vec![] });
    response.prefix_token_ids.push(999);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::PrefixTokenCountMismatch));
}

#[test]
fn token_index_zero_no_prefix() {
    // token_index=0 should require exactly 0 prefix leaves and still pass.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, response) = make_valid_response(&cfg, &model, 1);
    assert_eq!(response.token_index, 0);
    assert!(response.prefix_leaf_hashes.is_empty());

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

// ===========================================================================
// 4. Layer_indices boundary conditions
// ===========================================================================

#[test]
fn layer_indices_gap_rejected() {
    // layer_indices [0, 2] with 2 layers — gap at 1.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers.truncate(2);
    shell.layer_indices = Some(vec![0, 2]);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::NonContiguousLayerIndices));
}

#[test]
fn layer_indices_reversed_rejected() {
    // layer_indices [1, 0] — not ascending.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers.truncate(2);
    shell.layer_indices = Some(vec![1, 0]);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::NonContiguousLayerIndices));
}

#[test]
fn layer_indices_empty_is_contiguous() {
    // layer_indices = Some(vec![]) with 0 layers — degenerate but contiguous.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers.clear();
    shell.layer_indices = Some(vec![]);

    let report = verify_v4(&key, &response, None);
    // This should report routine coverage (0 of N layers) and no contiguity error.
    assert!(!has_code(&report, FailureCode::NonContiguousLayerIndices),
        "empty layer_indices should be trivially contiguous");
    match &report.coverage {
        AuditCoverage::Routine { layers_checked, .. } => assert_eq!(*layers_checked, 0),
        other => panic!("expected Routine(0/N), got {:?}", other),
    }
}

#[test]
fn layer_indices_count_mismatch_rejected() {
    // layer_indices says 1 layer but shell has 2 layer openings.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let shell = response.shell_opening.as_mut().unwrap();
    // Keep all layers but claim only index 0.
    shell.layer_indices = Some(vec![0]);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::ShellLayerCountMismatch));
}

// ===========================================================================
// 5. EOS-shortened requests and output policy edge cases
// ===========================================================================

#[test]
fn eos_stop_policy_eos_not_at_end_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response_at(&cfg, &model, 3, 1);
    let mut manifest = make_manifest(0.0);
    manifest.eos_policy = "stop".into();
    manifest.eos_token_id = Some(response.token_id); // challenged token IS eos
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);
    // token_index=1 but n_tokens=3 → eos at non-last position.

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::EosPolicyViolated));
}

#[test]
fn eos_stop_policy_eos_at_end_passes_policy_check() {
    // EOS at the last position under "stop" policy should not trigger EosPolicyViolated.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response_at(&cfg, &model, 3, 2);
    let mut manifest = make_manifest(0.0);
    manifest.eos_policy = "stop".into();
    manifest.eos_token_id = Some(response.token_id);
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);
    // token_index=2, n_tokens=3 → last token. EOS here is OK.

    let report = verify_v4(&key, &response, None);
    assert!(!has_code(&report, FailureCode::EosPolicyViolated),
        "EOS at last position should not violate stop policy, got: {:?}", report.failures);
}

#[test]
fn eos_unknown_policy_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let mut manifest = make_manifest(0.0);
    manifest.eos_policy = "yolo".into();
    manifest.eos_token_id = Some(999);
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::UnknownEosPolicy));
}

#[test]
fn eos_ignore_eos_with_eos_token_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let mut manifest = make_manifest(0.0);
    manifest.ignore_eos = true;
    manifest.eos_token_id = Some(response.token_id); // challenged token IS eos
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::IgnoreEosViolated));
}

#[test]
fn eos_min_tokens_without_eos_token_id_rejected() {
    // min_tokens > 0 but no eos_token_id → fail-closed.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let mut manifest = make_manifest(0.0);
    manifest.min_tokens = 5;
    manifest.eos_token_id = None; // can't enforce without knowing what EOS is
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MissingEosTokenId));
}

#[test]
fn eos_ignore_eos_without_eos_token_id_rejected() {
    // ignore_eos=true but no eos_token_id → fail-closed.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let mut manifest = make_manifest(0.0);
    manifest.ignore_eos = true;
    manifest.eos_token_id = None;
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MissingEosTokenId));
}

#[test]
fn eos_min_tokens_eos_in_window_rejected() {
    // min_tokens=10 but challenged token is EOS within the window.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response_at(&cfg, &model, 3, 1);
    let eos_id = response.token_id; // make this token the EOS
    let mut manifest = make_manifest(0.0);
    manifest.min_tokens = 10; // window extends beyond all committed tokens
    manifest.eos_token_id = Some(eos_id);
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MinTokensViolated));
}

// ===========================================================================
// 6. Missing shell opening fields
// ===========================================================================

#[test]
fn missing_shell_opening_reports_structural_failure() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.shell_opening = None;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MissingShellOpening));
    assert_eq!(report.coverage, AuditCoverage::Unknown);
}

// ===========================================================================
// 7. Seed / hash tampering (cryptographic binding boundaries)
// ===========================================================================

#[test]
fn wrong_revealed_seed_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.revealed_seed[0] ^= 0xFF; // flip one byte

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::SeedMismatch));
    assert!(has_category(&report, FailureCategory::CryptographicBinding));
}

#[test]
fn wrong_prompt_bytes_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    // Change prompt bytes without updating the committed hash.
    response.prompt = Some(b"TAMPERED PROMPT".to_vec());

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::PromptHashMismatch));
}

#[test]
fn tampered_merkle_root_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.commitment.merkle_root[0] ^= 0xFF;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::MerkleProofFailed));
}

#[test]
fn tampered_io_root_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    response.commitment.io_root[0] ^= 0xFF;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::IoChainProofFailed));
}

#[test]
fn tampered_prev_io_hash_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response_at(&cfg, &model, 3, 2);
    response.prev_io_hash[0] ^= 0xFF;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::IoChainMismatch));
}

#[test]
fn tampered_prefix_leaf_hash_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response_at(&cfg, &model, 3, 2);
    response.prefix_leaf_hashes[0][0] ^= 0xFF;

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    // Should fail on at least one of: merkle proof, IO chain.
    assert!(has_category(&report, FailureCategory::CryptographicBinding));
}

// ===========================================================================
// 8. Decode-mode / manifest field edge cases
// ===========================================================================

#[test]
fn decode_mode_greedy_with_nonzero_temp_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let mut manifest = make_manifest(0.5); // temp=0.5 (sampled)
    manifest.decode_mode = Some("greedy".into()); // but claims greedy
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::DecodeModeTempInconsistent));
}

#[test]
fn decode_mode_sampled_with_zero_temp_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let mut manifest = make_manifest(0.0); // temp=0.0 (greedy)
    manifest.decode_mode = Some("sampled".into()); // but claims sampled
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::DecodeModeTempInconsistent));
}

#[test]
fn unsupported_decode_features_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let mut manifest = make_manifest(0.0);
    manifest.repetition_penalty = 1.5; // non-default
    manifest.frequency_penalty = 0.1;
    manifest.presence_penalty = 0.2;
    manifest.logit_bias = vec![(42, 1.0)];
    manifest.bad_word_ids = vec![1, 2];
    manifest.guided_decoding = "json".into();
    manifest.stop_sequences = vec!["<stop>".into()];
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    // All 7 unsupported features should be individually flagged.
    let unsupported_count = report.failures.iter()
        .filter(|f| f.code == FailureCode::UnsupportedDecodeFeature)
        .count();
    assert!(unsupported_count >= 7,
        "expected >= 7 UnsupportedDecodeFeature, got {}: {:?}",
        unsupported_count, report.failures);
}

// ===========================================================================
// 9. Manifest hash binding boundaries
// ===========================================================================

#[test]
fn manifest_hash_mismatch_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let manifest = make_manifest(0.0);
    response.commitment.manifest_hash = Some([0xAA; 32]); // wrong hash
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::ManifestHashMismatch));
}

// ===========================================================================
// 10. Multi-token edge: single-token commitment (minimal case)
// ===========================================================================

#[test]
fn single_token_commitment_pass() {
    // The minimal possible commitment: 1 token, token_index=0, no prefix.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, response) = make_valid_response(&cfg, &model, 1);
    assert_eq!(response.commitment.n_tokens, 1);
    assert_eq!(response.token_index, 0);
    assert!(response.prefix_leaf_hashes.is_empty());

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn last_token_in_batch_pass() {
    // Open the last token in a 4-token batch.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, response) = make_valid_response_at(&cfg, &model, 4, 3);
    assert_eq!(response.token_index, 3);
    assert_eq!(response.prefix_leaf_hashes.len(), 3);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

// ===========================================================================
// 11. Retained state tampering with correct failure categories
// ===========================================================================

#[test]
fn tampered_retained_a_vector_rejected_with_category() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    // Tamper the retained state — should break Merkle proof.
    response.retained.layers[0].a[0] = !response.retained.layers[0].a[0];

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_category(&report, FailureCategory::CryptographicBinding));
}

// ===========================================================================
// 12. Coverage semantics across boundary cases
// ===========================================================================

#[test]
fn coverage_single_layer_routine() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    assert!(cfg.n_layers >= 2);
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let shell = response.shell_opening.as_mut().unwrap();
    shell.layers.truncate(1);
    shell.layer_indices = Some(vec![0]);

    let report = verify_v4(&key, &response, None);
    match &report.coverage {
        AuditCoverage::Routine { layers_checked, layers_total } => {
            assert_eq!(*layers_checked, 1);
            assert_eq!(*layers_total, cfg.n_layers);
        }
        other => panic!("expected Routine, got {:?}", other),
    }
}

#[test]
fn coverage_all_layers_full() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, response) = make_valid_response(&cfg, &model, 1);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    match &report.coverage {
        AuditCoverage::Full { layers_checked } => {
            assert_eq!(*layers_checked, cfg.n_layers);
        }
        other => panic!("expected Full, got {:?}", other),
    }
}

// ===========================================================================
// 13. Failure context carries correct metadata
// ===========================================================================

#[test]
fn failure_context_carries_token_index() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response_at(&cfg, &model, 3, 2);
    response.commitment.merkle_root[0] ^= 0xFF; // break Merkle proof

    let report = verify_v4(&key, &response, None);
    let merkle_failure = report.failures.iter()
        .find(|f| f.code == FailureCode::MerkleProofFailed)
        .expect("should have MerkleProofFailed");
    assert_eq!(merkle_failure.context.token_index, Some(2),
        "failure context should carry the challenged token_index");
}

// ===========================================================================
// 14. Sampler version edge cases
// ===========================================================================

#[test]
fn unknown_sampler_version_rejected() {
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (key, mut response) = make_valid_response(&cfg, &model, 1);
    let mut manifest = make_manifest(0.5);
    manifest.sampler_version = Some("futuristic-sampler-v99".into());
    let manifest_hash = verilm_core::merkle::hash_manifest(&manifest);
    response.commitment.manifest_hash = Some(manifest_hash);
    response.manifest = Some(manifest);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(has_code(&report, FailureCode::UnsupportedSamplerVersion));
}

// ===========================================================================
// 15. Unknown protocol version in binary format
// ===========================================================================

#[test]
fn binary_unknown_audit_magic_vv5a_rejected() {
    // Future version magic "VV5A" must be rejected by the current deserializer.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let (_key, response) = make_valid_response(&cfg, &model, 1);
    let mut binary = serialize::serialize_v4_audit(&response);
    // Replace VV4A magic with VV5A.
    binary[2] = b'5';
    let result = serialize::deserialize_v4_audit(&binary);
    assert!(result.is_err(), "VV5A magic should be rejected");
    assert!(result.unwrap_err().contains("magic"), "error should mention magic");
}

#[test]
fn binary_unknown_audit_magic_vv9z_rejected() {
    // Arbitrary future magic.
    let result = serialize::deserialize_v4_audit(b"VV9Z\x00\x01\x02\x03");
    assert!(result.is_err());
}

#[test]
fn binary_unknown_key_magic_rejected_as_audit() {
    // VKEY bytes (valid key format) must not parse as an audit response.
    let result = serialize::deserialize_v4_audit(b"VKEY\x00\x01\x02\x03\x04\x05");
    assert!(result.is_err());
}

// ===========================================================================
// 16. Long/short prompt-output boundary combinations
// ===========================================================================

#[test]
fn long_prompt_short_output_pass() {
    // Many prompt tokens (high n_prompt_tokens), minimal generation (1 token).
    // Simulated: commit 5 tokens, declare 4 are prompt, 1 generated.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let n_tokens = 5usize;
    let n_prompt = 4u32; // 4 prompt tokens → 1 generated token

    let inputs: Vec<Vec<i8>> = (0..n_tokens)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();
    let all_retained: Vec<RetainedTokenState> = inputs
        .iter()
        .map(|inp| retained_from_traces(&forward_pass(&cfg, &model, inp)))
        .collect();
    let token_ids: Vec<u32> = (0..n_tokens as u32).map(|i| 42 + i).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"this is a long prompt with many tokens for boundary testing",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(n_prompt),
    };
    let n_tok = all_retained.len();
    let all_scales = vec![unit_scales(cfg.n_layers); n_tok];
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None);
    // Open the last token (the single generated token).
    let response = open_v4(&state, (n_tokens - 1) as u32, &ToyWeights(&model), &cfg, &[], &[], None, None, None, None, false, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass,
        "long prompt + short output should pass, failures: {:?}", report.failures);
    assert_eq!(response.commitment.n_prompt_tokens, Some(n_prompt));
    assert_eq!(response.prefix_leaf_hashes.len(), n_tokens - 1);
}

#[test]
fn short_prompt_long_output_pass() {
    // Minimal prompt (1 token), many generated tokens.
    // Simulated: commit 8 tokens, declare 1 is prompt, 7 generated.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let n_tokens = 8usize;
    let n_prompt = 1u32;

    let inputs: Vec<Vec<i8>> = (0..n_tokens)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();
    let all_retained: Vec<RetainedTokenState> = inputs
        .iter()
        .map(|inp| retained_from_traces(&forward_pass(&cfg, &model, inp)))
        .collect();
    let token_ids: Vec<u32> = (0..n_tokens as u32).map(|i| 42 + i).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"hi",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(n_prompt),
    };
    let n_tok = all_retained.len();
    let all_scales = vec![unit_scales(cfg.n_layers); n_tok];
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None);
    // Open a token in the middle of the generated range.
    let mid = (n_tokens / 2) as u32;
    let response = open_v4(&state, mid, &ToyWeights(&model), &cfg, &[], &[], None, None, None, None, false, false);

    let report = verify_v4(&key, &response, None);
    assert_eq!(report.verdict, Verdict::Pass,
        "short prompt + long output should pass, failures: {:?}", report.failures);
    assert_eq!(response.commitment.n_prompt_tokens, Some(n_prompt));
    assert_eq!(response.prefix_leaf_hashes.len(), mid as usize);
}

#[test]
fn all_prompt_no_generation_boundary() {
    // Edge: n_prompt_tokens == n_tokens + 1 (all tokens are prompt, zero generated).
    // This is the maximum allowed by the bound check.
    let (cfg, model) = (ModelConfig::toy(), generate_model(&ModelConfig::toy(), 12345));
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let n_tokens = 3usize;
    // n_prompt_tokens = n_tokens + 1 = 4 (the +1 is the embedding input token).
    let n_prompt = (n_tokens as u32) + 1;

    let inputs: Vec<Vec<i8>> = (0..n_tokens)
        .map(|t| (0..cfg.hidden_dim).map(|i| ((i + t * 7) % 256) as i8).collect())
        .collect();
    let all_retained: Vec<RetainedTokenState> = inputs
        .iter()
        .map(|inp| retained_from_traces(&forward_pass(&cfg, &model, inp)))
        .collect();
    let token_ids: Vec<u32> = (0..n_tokens as u32).map(|i| 42 + i).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"all prompt tokens no generation",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(n_prompt),
    };
    let n_tok = all_retained.len();
    let all_scales = vec![unit_scales(cfg.n_layers); n_tok];
    let (_commitment, state) = commit_minimal(all_retained, &params, None, all_scales, None, None);
    let response = open_v4(&state, 0, &ToyWeights(&model), &cfg, &[], &[], None, None, None, None, false, false);

    // Should pass structural checks — n_prompt_tokens == n_tokens + 1 is the allowed maximum.
    let report = verify_v4(&key, &response, None);
    assert!(!has_code(&report, FailureCode::NPromptTokensBound),
        "n_prompt_tokens == n_tokens + 1 should be within bounds, got: {:?}", report.failures);
}
