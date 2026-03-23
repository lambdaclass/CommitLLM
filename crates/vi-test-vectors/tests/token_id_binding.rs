//! Tests for token-ID binding in the IO commitment.
//!
//! The core gap: a dishonest provider can commit honest computation
//! (correct logits from the real model) but return a different token
//! than what was actually sampled. Without binding the emitted token ID
//! into the commitment, verification passes despite the swap.

use vi_core::constants::ModelConfig;
use vi_test_vectors::*;

fn setup() -> (ModelConfig, Vec<LayerWeights>, vi_core::types::VerifierKey) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [7u8; 32]);
    (cfg, model, key)
}

// -----------------------------------------------------------------------
// Token swap is detected when token_ids are bound
// -----------------------------------------------------------------------

/// A dishonest provider runs honest inference, commits the trace with
/// token_ids bound (V2 IO hash), but then claims a different token_id
/// in the opened trace. The IO proof must fail.
#[test]
fn test_token_swap_detected_with_v2_commit() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    // Honest token_ids (simulated — in production these come from sampling)
    let token_ids: Vec<u32> = vec![100, 200, 300, 400];

    // Commit with token_ids bound into V2 IO hash
    let (_commitment, state) = commit_with_token_ids(all_layers, &token_ids);

    // Challenge token 2
    let challenges = vec![2u32];
    let mut proof = open(&state, &challenges);

    // Verify the honest open passes first
    {
        let (passed, failures) = verify_batch(&key, &proof, &challenges);
        assert!(passed, "honest V2 open should pass: {:?}", failures);
    }

    // ATTACK: swap token_id on the opened trace
    assert_eq!(proof.traces[0].token_id, Some(300));
    proof.traces[0].token_id = Some(999);

    // Verification MUST fail — the committed IO hash used token_id=300,
    // but the trace now claims 999.
    let (passed, _failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        !passed,
        "verification must fail when token_id is swapped on V2 commit"
    );
}

// -----------------------------------------------------------------------
// Legacy: token swap is undetected without token_id binding (V1)
// -----------------------------------------------------------------------

/// Without token_id binding (legacy V1 commit), swapping a token_id
/// field on the trace is invisible to verification. This test documents
/// the gap that V2 closes.
#[test]
fn test_token_swap_invisible_on_legacy_v1_commit() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    // Legacy commit (no token_ids)
    let (_commitment, state) = commit_legacy(all_layers);

    let challenges = vec![2u32];
    let mut proof = open(&state, &challenges);

    // token_id is None on legacy traces
    assert_eq!(proof.traces[0].token_id, None);

    // Even if someone sets a token_id on a V1-committed trace,
    // verification fails because the IO hash doesn't match.
    proof.traces[0].token_id = Some(9999);
    let (passed, _failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        !passed,
        "setting token_id on a V1-committed trace should fail IO check \
         because V2 hash != V1 hash"
    );
}

// -----------------------------------------------------------------------
// Honest path: correct token_ids pass
// -----------------------------------------------------------------------

#[test]
fn test_honest_token_ids_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids: Vec<u32> = vec![10, 20, 30, 40];

    let (_commitment, state) = commit_with_token_ids(all_layers, &token_ids);

    let challenges = vec![0u32, 1, 2, 3];
    let proof = open(&state, &challenges);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "honest batch with token_ids should pass: {:?}", failures);
}

// -----------------------------------------------------------------------
// Backward compat: V1 traces without token_id still verify
// -----------------------------------------------------------------------

#[test]
fn test_legacy_traces_without_token_id_still_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let (_commitment, state) = commit_legacy(all_layers);

    let challenges = vec![1u32, 3];
    let proof = open(&state, &challenges);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "legacy traces without token_id should still pass: {:?}", failures);
}

// -----------------------------------------------------------------------
// V1 and V2 IO hashes are distinct (domain separation)
// -----------------------------------------------------------------------

#[test]
fn test_v1_v2_io_hashes_differ() {
    use vi_core::merkle;

    let input = vec![1i8, 2, 3];
    let output = vec![4i32, 5, 6];

    let v1 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::Legacy);
    let v2 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::TokenId(42));

    assert_ne!(v1, v2, "V1 and V2 IO hashes must differ (domain separation)");
}

/// Two different token_ids produce different V2 hashes.
#[test]
fn test_different_token_ids_produce_different_hashes() {
    use vi_core::merkle;

    let input = vec![1i8, 2, 3];
    let output = vec![4i32, 5, 6];

    let h1 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::TokenId(100));
    let h2 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::TokenId(101));

    assert_ne!(h1, h2, "different token_ids must produce different IO hashes");
}

// -----------------------------------------------------------------------
// Commitment version field
// -----------------------------------------------------------------------

#[test]
fn test_v2_commit_sets_version_v2() {
    let (cfg, model, _key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![10, 20, 30, 40];

    let (commitment, _state) = commit_with_token_ids(all_layers, &token_ids);
    assert_eq!(
        commitment.version,
        vi_core::types::CommitmentVersion::V2,
        "commit_with_token_ids must produce V2 commitment"
    );
}

#[test]
fn test_legacy_commit_sets_version_v1() {
    let (cfg, model, _key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    let (commitment, _state) = commit_legacy(all_layers);
    assert_eq!(
        commitment.version,
        vi_core::types::CommitmentVersion::V1,
        "legacy commit must produce V1 commitment"
    );
}

// -----------------------------------------------------------------------
// Full batch E2E through serialized format
// -----------------------------------------------------------------------

/// Full round-trip: commit with token_ids → serialize → deserialize
/// → challenge → open → serialize → deserialize → verify.
/// Exercises the entire public API surface including wire format.
#[test]
fn test_full_batch_e2e_serialized_with_token_ids() {
    use vi_core::merkle;
    use vi_core::serialize::{
        serialize_batch, deserialize_batch,
        serialize_compact_batch, deserialize_compact_batch,
    };

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    // 1. Generate tokens
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);
    let token_ids: Vec<u32> = vec![101, 202, 303, 404, 505, 606, 707, 808];

    // 2. Commit with token_ids
    let (commitment, state) = commit_with_token_ids(all_layers, &token_ids);
    assert_eq!(commitment.version, vi_core::types::CommitmentVersion::V2);

    // 3. Derive challenges from commitment (verifier side)
    let seed = [42u8; 32];
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &seed, commitment.n_tokens, 3,
    );

    // 4. Open challenged tokens
    let proof = open(&state, &challenges);

    // Verify token_ids propagated through open
    for trace in &proof.traces {
        let expected_tid = token_ids[trace.token_index as usize];
        assert_eq!(trace.token_id, Some(expected_tid));
    }

    // 5. Serialize → deserialize (full format)
    let wire_bytes = serialize_batch(&proof);
    let proof_rt = deserialize_batch(&wire_bytes)
        .expect("full batch deserialization failed");

    // Verify token_ids survive serialization
    for trace in &proof_rt.traces {
        let expected_tid = token_ids[trace.token_index as usize];
        assert_eq!(trace.token_id, Some(expected_tid),
            "token_id lost in full serialization roundtrip");
    }
    assert_eq!(proof_rt.commitment.version, vi_core::types::CommitmentVersion::V2,
        "commitment version lost in serialization roundtrip");

    // 6. Verify deserialized proof
    let (passed, failures) = verify_batch(&key, &proof_rt, &challenges);
    assert!(passed, "full-format E2E failed: {:?}", failures);

    // 7. Repeat with compact format
    let compact_bytes = serialize_compact_batch(&proof);
    let proof_compact = deserialize_compact_batch(&compact_bytes)
        .expect("compact batch deserialization failed");

    // token_ids survive compact roundtrip
    for trace in &proof_compact.traces {
        let expected_tid = token_ids[trace.token_index as usize];
        assert_eq!(trace.token_id, Some(expected_tid),
            "token_id lost in compact serialization roundtrip");
    }

    let (passed, failures) = verify_batch(&key, &proof_compact, &challenges);
    assert!(passed, "compact-format E2E failed: {:?}", failures);
}

/// Full E2E with token swap through the serialized path.
/// Commit V2, serialize, deserialize, tamper token_id, verify → fail.
#[test]
fn test_serialized_token_swap_detected() {
    use vi_core::serialize::{serialize_batch, deserialize_batch};

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![100, 200, 300, 400];

    let (_commitment, state) = commit_with_token_ids(all_layers, &token_ids);
    let challenges = vec![1u32];
    let proof = open(&state, &challenges);

    // Serialize → deserialize → tamper → verify
    let wire = serialize_batch(&proof);
    let mut proof_rt = deserialize_batch(&wire).unwrap();

    // Swap token_id after deserialization
    assert_eq!(proof_rt.traces[0].token_id, Some(200));
    proof_rt.traces[0].token_id = Some(999);

    let (passed, _) = verify_batch(&key, &proof_rt, &challenges);
    assert!(!passed, "serialized token swap must be detected");
}
