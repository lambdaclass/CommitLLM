//! Tests for deployment manifest binding (Item 1 of 3 protocol improvements).

use verilm_core::merkle;
use verilm_core::types::DeploymentManifest;
use verilm_test_vectors::*;

fn toy_config() -> verilm_core::constants::ModelConfig {
    verilm_core::constants::ModelConfig {
        name: "manifest-test".into(),
        hidden_dim: 8,
        ffn_dim: 16,
        kv_dim: 8,
        n_layers: 2,
        n_q_heads: 2,
        n_kv_heads: 2,
        d_head: 4,
        vocab_size: 32,
    }
}

fn sample_manifest() -> DeploymentManifest {
    DeploymentManifest {
        tokenizer_hash: [0xAB; 32],
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        eos_policy: "stop".into(),
    }
}

/// Helper: generate a toy model, run forward pass, and return all_layers.
fn toy_all_layers() -> Vec<Vec<verilm_core::types::LayerTrace>> {
    let cfg = toy_config();
    let model = generate_model(&cfg, 42);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let layers = forward_pass(&cfg, &model, &input);
    vec![layers]
}

#[test]
fn test_honest_manifest_binding_passes() {
    let manifest = sample_manifest();
    let all_layers = toy_all_layers();
    let (commitment, _state) = commit_with_manifest(all_layers, &manifest);

    let failures = verilm_verify::verify_manifest(&commitment, &manifest);
    assert!(failures.is_empty(), "honest manifest should pass: {:?}", failures);
}

#[test]
fn test_wrong_temperature_detected() {
    let manifest = sample_manifest();
    let all_layers = toy_all_layers();
    let (commitment, _state) = commit_with_manifest(all_layers, &manifest);

    let mut wrong = sample_manifest();
    wrong.temperature = 0.0;
    let failures = verilm_verify::verify_manifest(&commitment, &wrong);
    assert!(!failures.is_empty(), "wrong temperature should fail");
    assert!(failures[0].contains("mismatch"), "expected mismatch: {:?}", failures);
}

#[test]
fn test_wrong_tokenizer_detected() {
    let manifest = sample_manifest();
    let all_layers = toy_all_layers();
    let (commitment, _state) = commit_with_manifest(all_layers, &manifest);

    let mut wrong = sample_manifest();
    wrong.tokenizer_hash = [0xCD; 32];
    let failures = verilm_verify::verify_manifest(&commitment, &wrong);
    assert!(!failures.is_empty(), "wrong tokenizer should fail");
    assert!(failures[0].contains("mismatch"), "expected mismatch: {:?}", failures);
}

#[test]
fn test_wrong_top_k_detected() {
    let manifest = sample_manifest();
    let all_layers = toy_all_layers();
    let (commitment, _state) = commit_with_manifest(all_layers, &manifest);

    let mut wrong = sample_manifest();
    wrong.top_k = 100;
    let failures = verilm_verify::verify_manifest(&commitment, &wrong);
    assert!(!failures.is_empty(), "wrong top_k should fail");
    assert!(failures[0].contains("mismatch"), "expected mismatch: {:?}", failures);
}

#[test]
fn test_wrong_eos_policy_detected() {
    let manifest = sample_manifest();
    let all_layers = toy_all_layers();
    let (commitment, _state) = commit_with_manifest(all_layers, &manifest);

    let mut wrong = sample_manifest();
    wrong.eos_policy = "sample".into();
    let failures = verilm_verify::verify_manifest(&commitment, &wrong);
    assert!(!failures.is_empty(), "wrong eos_policy should fail");
    assert!(failures[0].contains("mismatch"), "expected mismatch: {:?}", failures);
}

#[test]
fn test_missing_manifest_hash_when_expected() {
    // Legacy commit (no manifest)
    let all_layers = toy_all_layers();
    let (commitment, _state) = commit(all_layers);
    assert!(commitment.manifest_hash.is_none());

    let manifest = sample_manifest();
    let failures = verilm_verify::verify_manifest(&commitment, &manifest);
    assert!(!failures.is_empty(), "missing manifest_hash should fail");
    assert!(
        failures[0].contains("missing manifest_hash"),
        "expected 'missing manifest_hash': {:?}",
        failures
    );
}

#[test]
fn test_manifest_present_not_expected_passes() {
    // Commit with manifest, but verifier doesn't call verify_manifest (backward compat).
    let manifest = sample_manifest();
    let all_layers = toy_all_layers();
    let (commitment, _state) = commit_with_manifest(all_layers, &manifest);
    assert!(commitment.manifest_hash.is_some());
}

#[test]
fn test_manifest_hash_deterministic() {
    let manifest = sample_manifest();
    let h1 = merkle::hash_manifest(&manifest);
    let h2 = merkle::hash_manifest(&manifest);
    assert_eq!(h1, h2, "same manifest should hash identically");
}

#[test]
fn test_manifest_hash_sensitive_to_each_field() {
    let base = sample_manifest();
    let base_hash = merkle::hash_manifest(&base);

    let mut m = base.clone();
    m.tokenizer_hash = [0x00; 32];
    assert_ne!(merkle::hash_manifest(&m), base_hash, "tokenizer_hash change should alter hash");

    let mut m = base.clone();
    m.temperature = 1.0;
    assert_ne!(merkle::hash_manifest(&m), base_hash, "temperature change should alter hash");

    let mut m = base.clone();
    m.top_k = 999;
    assert_ne!(merkle::hash_manifest(&m), base_hash, "top_k change should alter hash");

    let mut m = base.clone();
    m.top_p = 0.5;
    assert_ne!(merkle::hash_manifest(&m), base_hash, "top_p change should alter hash");

    let mut m = base.clone();
    m.eos_policy = "sample".into();
    assert_ne!(merkle::hash_manifest(&m), base_hash, "eos_policy change should alter hash");
}
