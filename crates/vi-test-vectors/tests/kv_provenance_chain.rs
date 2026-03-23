//! Adversarial tests for KV provenance chaining (task 29).
//!
//! Validates that the per-token KV chain hash
//!   H("vi-kv-v1" || prev_kv_hash || requantize(k_t) || requantize(v_t) || token_index_le32)
//! catches fabricated, swapped, or modified KV projections when consecutive
//! tokens are opened.

use vi_core::constants::ModelConfig;
use vi_core::merkle;
use vi_test_vectors::*;

fn setup_autoregressive() -> (
    ModelConfig,
    Vec<LayerWeights>,
    vi_core::types::VerifierKey,
    Vec<Vec<vi_core::types::LayerTrace>>,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [7u8; 32]);
    let all_layers = forward_pass_autoregressive(&cfg, &model, &vec![1i8; cfg.hidden_dim], 5);
    (cfg, model, key, all_layers)
}

#[test]
fn honest_kv_chain_passes() {
    let (_cfg, _model, key, all_layers) = setup_autoregressive();
    let token_ids: Vec<u32> = (100..105).collect();

    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"test prompt",
            sampling_seed: [0xAA; 32],
            manifest: None,
        },
    );

    // Open all 5 tokens
    let proof = open(&state, &[0, 1, 2, 3, 4]);
    let (passed, failures) = verify_batch(&key, &proof, &[0, 1, 2, 3, 4]);
    assert!(passed, "Honest KV chain should pass: {:?}", failures);

    // Verify KV chain root is set
    assert!(proof.commitment.kv_chain_root.is_some());

    // Verify all traces have prev_kv_hash
    for trace in &proof.traces {
        assert!(trace.prev_kv_hash.is_some(), "token {} missing prev_kv_hash", trace.token_index);
    }
}

#[test]
fn kv_chain_genesis_is_zero() {
    let (_cfg, _model, _key, all_layers) = setup_autoregressive();
    let token_ids: Vec<u32> = (100..105).collect();

    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"test",
            sampling_seed: [0xBB; 32],
            manifest: None,
        },
    );

    let proof = open(&state, &[0]);
    assert_eq!(
        proof.traces[0].prev_kv_hash,
        Some([0u8; 32]),
        "Token 0 should have genesis zero hash as prev_kv_hash"
    );
}

#[test]
fn tampered_prev_kv_hash_detected_for_consecutive_tokens() {
    let (_cfg, _model, key, all_layers) = setup_autoregressive();
    let token_ids: Vec<u32> = (100..105).collect();

    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"test",
            sampling_seed: [0xCC; 32],
            manifest: None,
        },
    );

    // Open tokens 1 and 2 (consecutive)
    let mut proof = open(&state, &[1, 2]);

    // Tamper: replace token 2's prev_kv_hash with garbage
    proof.traces[1].prev_kv_hash = Some([0xFF; 32]);

    let (passed, failures) = verify_batch(&key, &proof, &[1, 2]);
    assert!(!passed, "Tampered prev_kv_hash should be detected");
    assert!(
        failures.iter().any(|f| f.contains("prev_kv_hash mismatch")),
        "Should report prev_kv_hash mismatch, got: {:?}",
        failures
    );
}

#[test]
fn fabricated_kv_projections_detected() {
    let (_cfg, _model, key, all_layers) = setup_autoregressive();
    let token_ids: Vec<u32> = (100..105).collect();

    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"test",
            sampling_seed: [0xDD; 32],
            manifest: None,
        },
    );

    // Open tokens 2 and 3 (consecutive)
    let mut proof = open(&state, &[2, 3]);

    // Fabricate: zero out token 3's K projections at every layer.
    // This corrupts the trace data, which should be caught by the Merkle
    // proof (leaf hash mismatch) and/or Freivalds check.
    for lt in &mut proof.traces[1].layers {
        for k in lt.k.iter_mut() {
            *k = 0;
        }
    }

    let (passed, failures) = verify_batch(&key, &proof, &[2, 3]);
    assert!(!passed, "Fabricated KV projections should be detected: {:?}", failures);
    // The Merkle proof should fail because the leaf hash changed
    assert!(
        failures.iter().any(|f| f.contains("Merkle") || f.contains("Freivalds")),
        "Should detect via Merkle or Freivalds, got: {:?}",
        failures
    );
}

#[test]
fn kv_chain_hash_deterministic() {
    // Same inputs produce same hash
    let prev = [0u8; 32];
    let k = vec![vec![1i8, 2, 3], vec![4, 5, 6]];
    let v = vec![vec![7i8, 8, 9], vec![10, 11, 12]];

    let h1 = merkle::kv_chain_hash(&prev, &k, &v, 0);
    let h2 = merkle::kv_chain_hash(&prev, &k, &v, 0);
    assert_eq!(h1, h2, "KV chain hash should be deterministic");
}

#[test]
fn kv_chain_hash_changes_with_token_index() {
    let prev = [0u8; 32];
    let k = vec![vec![1i8, 2, 3]];
    let v = vec![vec![4i8, 5, 6]];

    let h0 = merkle::kv_chain_hash(&prev, &k, &v, 0);
    let h1 = merkle::kv_chain_hash(&prev, &k, &v, 1);
    assert_ne!(h0, h1, "Different token indices should produce different hashes");
}

#[test]
fn kv_chain_hash_changes_with_prev() {
    let k = vec![vec![1i8, 2, 3]];
    let v = vec![vec![4i8, 5, 6]];

    let h_zero = merkle::kv_chain_hash(&[0u8; 32], &k, &v, 0);
    let h_nonzero = merkle::kv_chain_hash(&[1u8; 32], &k, &v, 0);
    assert_ne!(h_zero, h_nonzero, "Different prev_kv_hash should produce different hashes");
}

#[test]
fn non_consecutive_tokens_no_cross_check_no_false_positive() {
    // When non-consecutive tokens are opened, the verifier cannot cross-check
    // prev_kv_hash (inherent limitation of sampled opening). This test verifies
    // no false positives occur.
    let (_cfg, _model, key, all_layers) = setup_autoregressive();
    let token_ids: Vec<u32> = (100..105).collect();

    let (_commitment, state) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: b"test",
            sampling_seed: [0xEE; 32],
            manifest: None,
        },
    );

    // Open tokens 0 and 3 (non-consecutive — gap at 1, 2)
    let proof = open(&state, &[0, 3]);
    let (passed, failures) = verify_batch(&key, &proof, &[0, 3]);
    assert!(passed, "Non-consecutive honest tokens should pass: {:?}", failures);
}
