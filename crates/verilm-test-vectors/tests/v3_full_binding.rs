//! Tests for V3 full binding: transcript chaining, seed commit-reveal,
//! and prompt hash binding.

use verilm_core::constants::ModelConfig;
use verilm_core::merkle;
use verilm_core::types::CommitmentVersion;
use verilm_test_vectors::*;

fn setup() -> (ModelConfig, Vec<LayerWeights>, verilm_core::types::VerifierKey) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [7u8; 32]);
    (cfg, model, key)
}

// -----------------------------------------------------------------------
// Transcript chaining — honest path
// -----------------------------------------------------------------------

#[test]
fn test_v3_honest_full_binding_passes() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 6);
    let token_ids: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"What is 2+2?",
        sampling_seed: [42u8; 32],
        manifest: None,
    };

    let (commitment, state) = commit_with_full_binding(all_layers, &params, None);
    assert_eq!(commitment.version, CommitmentVersion::V3);
    assert!(commitment.prompt_hash.is_some());
    assert!(commitment.seed_commitment.is_some());

    // Open all tokens
    let challenges: Vec<u32> = (0..6).collect();
    let proof = open(&state, &challenges);

    assert!(proof.revealed_seed.is_some());
    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "honest V3 full binding should pass: {:?}", failures);
}

#[test]
fn test_v3_subset_challenges_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);
    let token_ids: Vec<u32> = (100..108).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"hello world",
        sampling_seed: [99u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);

    // Challenge only token 0 and 1 (consecutive — chain is reconstructible)
    let challenges = vec![0u32, 1];
    let proof = open(&state, &challenges);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "V3 subset challenges should pass: {:?}", failures);
}

// -----------------------------------------------------------------------
// Transcript chaining — attack detection
// -----------------------------------------------------------------------

#[test]
fn test_v3_token_swap_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![100, 200, 300, 400];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"test prompt",
        sampling_seed: [1u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges = vec![0u32, 1, 2, 3];
    let mut proof = open(&state, &challenges);

    // ATTACK: swap token_id on token 2
    assert_eq!(proof.traces[2].token_id, Some(300));
    proof.traces[2].token_id = Some(999);

    let (passed, _) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "V3 token swap must be detected");
}

#[test]
fn test_v3_token_swap_breaks_chain_for_subsequent_tokens() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![100, 200, 300, 400];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"test",
        sampling_seed: [2u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges = vec![0u32, 1, 2, 3];
    let mut proof = open(&state, &challenges);

    // ATTACK: swap token_id on token 1 — this should break token 1, 2, and 3
    // because the chain propagates: token 2's prev_io_hash depends on token 1's IO hash
    proof.traces[1].token_id = Some(999);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "chain break should be detected");
    // At least token 1 should fail (its IO hash changed)
    let io_failures: Vec<_> = failures.iter().filter(|f| f.contains("IO proof")).collect();
    assert!(!io_failures.is_empty(), "should have IO proof failures from chain break");
}

// -----------------------------------------------------------------------
// Seed commit-reveal
// -----------------------------------------------------------------------

#[test]
fn test_v3_seed_commitment_honest() {
    let (cfg, model, _key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 2);
    let seed = [55u8; 32];
    let params = FullBindingParams {
        token_ids: &[10, 20],
        prompt: b"test",
        sampling_seed: seed,
        manifest: None,
    };

    let (commitment, _state) = commit_with_full_binding(all_layers, &params, None);

    // Verify seed commitment matches hash_seed
    assert_eq!(commitment.seed_commitment, Some(merkle::hash_seed(&seed)));
}

#[test]
fn test_v3_seed_tamper_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let params = FullBindingParams {
        token_ids: &[10, 20, 30, 40],
        prompt: b"test",
        sampling_seed: [77u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges: Vec<u32> = (0..4).collect();
    let mut proof = open(&state, &challenges);

    // ATTACK: reveal a different seed
    proof.revealed_seed = Some([88u8; 32]);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "tampered seed must be detected");
    assert!(
        failures.iter().any(|f| f.contains("seed commitment mismatch")),
        "should report seed mismatch: {:?}", failures
    );
}

#[test]
fn test_v3_missing_revealed_seed_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 2);
    let params = FullBindingParams {
        token_ids: &[10, 20],
        prompt: b"test",
        sampling_seed: [77u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges = vec![0u32, 1];
    let mut proof = open(&state, &challenges);

    // ATTACK: withhold the seed
    proof.revealed_seed = None;

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "missing revealed_seed must be detected");
    assert!(
        failures.iter().any(|f| f.contains("missing revealed_seed")),
        "should report missing seed: {:?}", failures
    );
}

// -----------------------------------------------------------------------
// Prompt hash binding
// -----------------------------------------------------------------------

#[test]
fn test_v3_prompt_hash_present() {
    let (cfg, model, _key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 2);
    let prompt = b"What is the meaning of life?";
    let params = FullBindingParams {
        token_ids: &[10, 20],
        prompt,
        sampling_seed: [1u8; 32],
        manifest: None,
    };

    let (commitment, _state) = commit_with_full_binding(all_layers, &params, None);
    assert_eq!(commitment.prompt_hash, Some(merkle::hash_prompt(prompt)));
}

#[test]
fn test_v3_different_prompts_different_hashes() {
    let h1 = merkle::hash_prompt(b"hello");
    let h2 = merkle::hash_prompt(b"world");
    assert_ne!(h1, h2, "different prompts must produce different hashes");
}

// -----------------------------------------------------------------------
// V3 commitment version field
// -----------------------------------------------------------------------

#[test]
fn test_v3_commit_sets_version() {
    let (cfg, model, _key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 2);
    let params = FullBindingParams {
        token_ids: &[10, 20],
        prompt: b"test",
        sampling_seed: [1u8; 32],
        manifest: None,
    };

    let (commitment, _) = commit_with_full_binding(all_layers, &params, None);
    assert_eq!(commitment.version, CommitmentVersion::V3);
}

// -----------------------------------------------------------------------
// Domain separation: V1, V2, V3 IO hashes are all distinct
// -----------------------------------------------------------------------

#[test]
fn test_v1_v2_v3_io_hashes_all_differ() {
    let input = vec![1i8, 2, 3];
    let output = vec![4i32, 5, 6];

    let v1 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::Legacy);
    let v2 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::TokenId(42));
    let v3 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::Chained {
        token_id: 42,
        prev_io_hash: [0u8; 32],
    });

    assert_ne!(v1, v2, "V1 != V2");
    assert_ne!(v1, v3, "V1 != V3");
    assert_ne!(v2, v3, "V2 != V3");
}

#[test]
fn test_v3_different_prev_io_produces_different_hash() {
    let input = vec![1i8, 2, 3];
    let output = vec![4i32, 5, 6];

    let h1 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::Chained {
        token_id: 42,
        prev_io_hash: [0u8; 32],
    });
    let h2 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::Chained {
        token_id: 42,
        prev_io_hash: [1u8; 32],
    });

    assert_ne!(h1, h2, "different prev_io_hash must produce different IO hashes");
}

// -----------------------------------------------------------------------
// V3 missing token_id on trace is rejected
// -----------------------------------------------------------------------

#[test]
fn test_v3_missing_token_id_rejected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 2);
    let params = FullBindingParams {
        token_ids: &[10, 20],
        prompt: b"test",
        sampling_seed: [1u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges = vec![0u32, 1];
    let mut proof = open(&state, &challenges);

    // Remove token_id from a trace
    proof.traces[0].token_id = None;

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "V3 with missing token_id must fail");
    assert!(
        failures.iter().any(|f| f.contains("requires token_id")),
        "should report missing token_id: {:?}", failures
    );
}

// -----------------------------------------------------------------------
// Full E2E through serialization with V3
// -----------------------------------------------------------------------

#[test]
fn test_v3_full_e2e_serialized() {
    use verilm_core::serialize::{
        serialize_batch, deserialize_batch,
        serialize_compact_batch, deserialize_compact_batch,
    };

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 6);
    let token_ids: Vec<u32> = vec![101, 202, 303, 404, 505, 606];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"serialize test prompt",
        sampling_seed: [123u8; 32],
        manifest: None,
    };

    let (commitment, state) = commit_with_full_binding(all_layers, &params, None);
    assert_eq!(commitment.version, CommitmentVersion::V3);

    let challenges: Vec<u32> = (0..6).collect();
    let proof = open(&state, &challenges);

    // Full format roundtrip
    let wire = serialize_batch(&proof);
    let proof_rt = deserialize_batch(&wire).expect("full deserialization failed");

    assert_eq!(proof_rt.commitment.version, CommitmentVersion::V3);
    assert_eq!(proof_rt.commitment.prompt_hash, commitment.prompt_hash);
    assert_eq!(proof_rt.commitment.seed_commitment, commitment.seed_commitment);
    assert_eq!(proof_rt.revealed_seed, Some([123u8; 32]));

    for trace in &proof_rt.traces {
        let expected_tid = token_ids[trace.token_index as usize];
        assert_eq!(trace.token_id, Some(expected_tid));
    }

    let (passed, failures) = verify_batch(&key, &proof_rt, &challenges);
    assert!(passed, "V3 full-format E2E failed: {:?}", failures);

    // Compact format roundtrip
    let compact_wire = serialize_compact_batch(&proof);
    let proof_compact = deserialize_compact_batch(&compact_wire)
        .expect("compact deserialization failed");

    assert_eq!(proof_compact.commitment.version, CommitmentVersion::V3);
    assert_eq!(proof_compact.revealed_seed, Some([123u8; 32]));

    let (passed, failures) = verify_batch(&key, &proof_compact, &challenges);
    assert!(passed, "V3 compact-format E2E failed: {:?}", failures);
}

// -----------------------------------------------------------------------
// Sparse challenge chain verification (#4)
// -----------------------------------------------------------------------

/// Non-consecutive challenge set (e.g. tokens 0, 3, 7) must still pass
/// because each trace carries prev_io_hash for chain reconstruction.
#[test]
fn test_v3_sparse_challenges_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);
    let token_ids: Vec<u32> = (100..108).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"sparse test",
        sampling_seed: [50u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);

    // Non-consecutive: tokens 0, 3, 7
    let challenges = vec![0u32, 3, 7];
    let proof = open(&state, &challenges);

    // Each trace should have prev_io_hash
    for trace in &proof.traces {
        assert!(trace.prev_io_hash.is_some(),
            "token {} should have prev_io_hash", trace.token_index);
    }

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "sparse V3 challenges should pass: {:?}", failures);
}

/// Tampering prev_io_hash on a non-first trace breaks the IO proof.
#[test]
fn test_v3_tampered_prev_io_hash_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let params = FullBindingParams {
        token_ids: &[10, 20, 30, 40],
        prompt: b"test",
        sampling_seed: [1u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges: Vec<u32> = (0..4).collect();
    let mut proof = open(&state, &challenges);

    // ATTACK: tamper prev_io_hash on token 2
    proof.traces[2].prev_io_hash = Some([0xffu8; 32]);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "tampered prev_io_hash must be detected");
    // Should fail IO proof for token 2 (wrong chained hash) AND
    // should fail prev_io_hash cross-check for token 3 (since token 2's
    // recomputed IO hash changed)
    let io_failures: Vec<_> = failures.iter()
        .filter(|f| f.contains("IO proof") || f.contains("prev_io_hash mismatch"))
        .collect();
    assert!(!io_failures.is_empty(), "should have IO/chain failures: {:?}", failures);
}

/// Cross-check: when consecutive tokens are both opened, prev_io_hash
/// of the later token must match the recomputed IO hash of the earlier.
#[test]
fn test_v3_cross_check_prev_io_hash_consecutive() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let params = FullBindingParams {
        token_ids: &[10, 20, 30, 40],
        prompt: b"test",
        sampling_seed: [1u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    // Open tokens 1 and 2 (consecutive)
    let challenges = vec![1u32, 2];
    let mut proof = open(&state, &challenges);

    // Tamper token 2's prev_io_hash to something wrong
    // (but leave token 1 honest)
    proof.traces[1].prev_io_hash = Some([0xaau8; 32]);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "cross-check must catch mismatched prev_io_hash");
    assert!(
        failures.iter().any(|f| f.contains("prev_io_hash mismatch")),
        "should specifically report prev_io_hash mismatch: {:?}", failures
    );
}

// -----------------------------------------------------------------------
// VerificationPolicy (#1 + #5)
// -----------------------------------------------------------------------

#[test]
fn test_policy_min_version_rejects_v1() {
    use verilm_core::types::VerificationPolicy;

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    // Create a V1 (legacy) commitment
    let (_commitment, state) = commit_legacy(all_layers);
    let challenges = vec![0u32, 1, 2, 3];
    let proof = open(&state, &challenges);

    // With default policy, V1 passes
    let (passed, _) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "V1 should pass with default policy");

    // With min_version = V3, V1 is rejected
    let policy = VerificationPolicy {
        min_version: Some(CommitmentVersion::V3),
        ..Default::default()
    };
    let (passed, failures) = verify_batch_with_policy(&key, &proof, &challenges, &policy);
    assert!(!passed, "V1 should be rejected when min_version=V3");
    assert!(
        failures.iter().any(|f| f.contains("below minimum")),
        "should report version below minimum: {:?}", failures
    );
}

#[test]
fn test_policy_min_version_rejects_v2() {
    use verilm_core::types::VerificationPolicy;

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    let (_commitment, state) = commit_with_token_ids(all_layers, &[10, 20, 30, 40]);
    let challenges = vec![0u32, 1, 2, 3];
    let proof = open(&state, &challenges);

    let policy = VerificationPolicy {
        min_version: Some(CommitmentVersion::V3),
        ..Default::default()
    };
    let (passed, failures) = verify_batch_with_policy(&key, &proof, &challenges, &policy);
    assert!(!passed, "V2 should be rejected when min_version=V3");
    assert!(failures.iter().any(|f| f.contains("below minimum")));
}

#[test]
fn test_policy_expected_prompt_hash_correct() {
    use verilm_core::types::VerificationPolicy;

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    let prompt = b"What is 2+2?";
    let params = FullBindingParams {
        token_ids: &[10, 20, 30, 40],
        prompt,
        sampling_seed: [1u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges: Vec<u32> = (0..4).collect();
    let proof = open(&state, &challenges);

    // Correct prompt hash passes
    let policy = VerificationPolicy {
        expected_prompt_hash: Some(merkle::hash_prompt(prompt)),
        ..Default::default()
    };
    let (passed, failures) = verify_batch_with_policy(&key, &proof, &challenges, &policy);
    assert!(passed, "correct prompt hash should pass: {:?}", failures);
}

#[test]
fn test_policy_expected_prompt_hash_wrong_rejected() {
    use verilm_core::types::VerificationPolicy;

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    let params = FullBindingParams {
        token_ids: &[10, 20, 30, 40],
        prompt: b"What is 2+2?",
        sampling_seed: [1u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges: Vec<u32> = (0..4).collect();
    let proof = open(&state, &challenges);

    // Wrong prompt hash fails
    let policy = VerificationPolicy {
        expected_prompt_hash: Some(merkle::hash_prompt(b"What is 3+3?")),
        ..Default::default()
    };
    let (passed, failures) = verify_batch_with_policy(&key, &proof, &challenges, &policy);
    assert!(!passed, "wrong prompt hash must fail");
    assert!(
        failures.iter().any(|f| f.contains("prompt_hash mismatch")),
        "should report prompt_hash mismatch: {:?}", failures
    );
}

#[test]
fn test_policy_expected_prompt_hash_missing_rejected() {
    use verilm_core::types::VerificationPolicy;

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    // V1 commit has no prompt_hash
    let (_commitment, state) = commit_legacy(all_layers);
    let challenges = vec![0u32, 1, 2, 3];
    let proof = open(&state, &challenges);

    let policy = VerificationPolicy {
        expected_prompt_hash: Some(merkle::hash_prompt(b"anything")),
        ..Default::default()
    };
    let (passed, failures) = verify_batch_with_policy(&key, &proof, &challenges, &policy);
    assert!(!passed, "missing prompt_hash must fail when policy expects it");
    assert!(failures.iter().any(|f| f.contains("policy requires prompt_hash")));
}

// -----------------------------------------------------------------------
// Manifest in V3 path (#3)
// -----------------------------------------------------------------------

#[test]
fn test_v3_with_manifest_binding() {
    use verilm_core::types::{DeploymentManifest, VerificationPolicy};

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    let manifest = DeploymentManifest {
        tokenizer_hash: [0xaa; 32],
        temperature: 0.7,
        top_k: 50,
        top_p: 1.0,
        eos_policy: "stop".into(),
        weight_hash: None,
        quant_hash: None,
        system_prompt_hash: None,
    };

    let params = FullBindingParams {
        token_ids: &[10, 20, 30, 40],
        prompt: b"test with manifest",
        sampling_seed: [1u8; 32],
        manifest: Some(&manifest),
    };

    let (commitment, state) = commit_with_full_binding(all_layers, &params, None);
    assert!(commitment.manifest_hash.is_some(), "V3 with manifest should set manifest_hash");
    assert_eq!(commitment.manifest_hash, Some(merkle::hash_manifest(&manifest)));

    let challenges: Vec<u32> = (0..4).collect();
    let proof = open(&state, &challenges);

    // Passes with correct manifest policy
    let policy = VerificationPolicy {
        expected_manifest_hash: Some(merkle::hash_manifest(&manifest)),
        ..Default::default()
    };
    let (passed, failures) = verify_batch_with_policy(&key, &proof, &challenges, &policy);
    assert!(passed, "V3 with matching manifest should pass: {:?}", failures);
}

#[test]
fn test_v3_wrong_manifest_rejected() {
    use verilm_core::types::{DeploymentManifest, VerificationPolicy};

    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

    let manifest = DeploymentManifest {
        tokenizer_hash: [0xaa; 32],
        temperature: 0.7,
        top_k: 50,
        top_p: 1.0,
        eos_policy: "stop".into(),
        weight_hash: None,
        quant_hash: None,
        system_prompt_hash: None,
    };

    let params = FullBindingParams {
        token_ids: &[10, 20, 30, 40],
        prompt: b"test",
        sampling_seed: [1u8; 32],
        manifest: Some(&manifest),
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params, None);
    let challenges: Vec<u32> = (0..4).collect();
    let proof = open(&state, &challenges);

    // Wrong manifest hash
    let wrong_manifest = DeploymentManifest {
        tokenizer_hash: [0xbb; 32], // different
        temperature: 0.7,
        top_k: 50,
        top_p: 1.0,
        eos_policy: "stop".into(),
        weight_hash: None,
        quant_hash: None,
        system_prompt_hash: None,
    };
    let policy = VerificationPolicy {
        expected_manifest_hash: Some(merkle::hash_manifest(&wrong_manifest)),
        ..Default::default()
    };
    let (passed, failures) = verify_batch_with_policy(&key, &proof, &challenges, &policy);
    assert!(!passed, "wrong manifest must fail");
    assert!(failures.iter().any(|f| f.contains("manifest_hash mismatch")));
}
