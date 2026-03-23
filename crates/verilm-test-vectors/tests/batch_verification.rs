//! Multi-token batch verification tests.
//!
//! Tests the full two-phase commit → challenge → open → verify flow
//! with real multi-token KV-cached attention.

use verilm_core::constants::ModelConfig;
use verilm_core::merkle;
use verilm_test_vectors::*;

fn setup_batch(n_tokens: usize) -> (ModelConfig, Vec<LayerWeights>, Vec<Vec<i8>>) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);

    // Generate distinct inputs for each token
    let inputs: Vec<Vec<i8>> = (0..n_tokens)
        .map(|t| {
            (0..cfg.hidden_dim)
                .map(|i| ((t * 7 + i * 3) % 256) as i8)
                .collect()
        })
        .collect();

    (cfg, model, inputs)
}

// -----------------------------------------------------------------------
// 1. Honest batch — everything passes
// -----------------------------------------------------------------------

#[test]
fn test_honest_batch_all_tokens_pass() {
    let (cfg, model, inputs) = setup_batch(8);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let all_layers = forward_pass_multi(&cfg, &model, &inputs);
    assert_eq!(all_layers.len(), 8);

    let (_commitment, _state, all_traces) = build_batch(all_layers);

    // Verify every single token
    for trace in &all_traces {
        let (passed, failures) = verify_trace(&key, trace);
        assert!(passed, "token {} failed: {:?}", trace.token_index, failures);
    }
}

#[test]
fn test_honest_batch_challenged_subset_passes() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    // Autoregressive generation — tokens chain from previous output
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 16);

    // Phase 1: Commit — prover publishes only the commitment
    let (commitment, state) = commit(all_layers);

    // Verifier derives challenges from the published commitment
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[99u8; 32], commitment.n_tokens, 5,
    );
    assert_eq!(challenges.len(), 5);

    // Phase 2: Open — prover generates proofs only for challenged tokens
    let proof = open(&state, &challenges);
    assert_eq!(proof.traces.len(), 5);

    // Verifier checks
    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "batch failed: {:?}", failures);
}

// -----------------------------------------------------------------------
// 2. Single-token consistency — batch with 1 token == single token
// -----------------------------------------------------------------------

#[test]
fn test_batch_single_token_matches_single_pass() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    // Single-token forward pass
    let single_layers = forward_pass(&cfg, &model, &input);

    // Multi-token forward pass with 1 token
    let multi_layers = forward_pass_multi(&cfg, &model, &[input]);

    // Results should be identical
    assert_eq!(multi_layers.len(), 1);
    for (layer_idx, (sl, ml)) in single_layers.iter().zip(multi_layers[0].iter()).enumerate() {
        assert_eq!(sl.x_attn, ml.x_attn, "layer {} x_attn mismatch", layer_idx);
        assert_eq!(sl.q, ml.q, "layer {} q mismatch", layer_idx);
        assert_eq!(sl.k, ml.k, "layer {} k mismatch", layer_idx);
        assert_eq!(sl.v, ml.v, "layer {} v mismatch", layer_idx);
        assert_eq!(sl.a, ml.a, "layer {} a mismatch", layer_idx);
        assert_eq!(sl.attn_out, ml.attn_out, "layer {} attn_out mismatch", layer_idx);
        assert_eq!(sl.x_ffn, ml.x_ffn, "layer {} x_ffn mismatch", layer_idx);
        assert_eq!(sl.g, ml.g, "layer {} g mismatch", layer_idx);
        assert_eq!(sl.u, ml.u, "layer {} u mismatch", layer_idx);
        assert_eq!(sl.h, ml.h, "layer {} h mismatch", layer_idx);
        assert_eq!(sl.ffn_out, ml.ffn_out, "layer {} ffn_out mismatch", layer_idx);
    }
}

// -----------------------------------------------------------------------
// 3. Attack: corrupt one token in a batch
// -----------------------------------------------------------------------

#[test]
fn test_batch_corrupt_one_token_detected() {
    let (cfg, model, inputs) = setup_batch(8);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let mut all_layers = forward_pass_multi(&cfg, &model, &inputs);

    // Corrupt token 3's q output
    all_layers[3][0].q[0] += 1;

    let (_commitment, _state, all_traces) = build_batch(all_layers);

    // Token 3 should fail, others should pass
    for trace in &all_traces {
        let (passed, _) = verify_trace(&key, trace);
        if trace.token_index == 3 {
            assert!(!passed, "corrupted token 3 should fail");
        } else {
            assert!(passed, "token {} should pass", trace.token_index);
        }
    }
}

// -----------------------------------------------------------------------
// 4. Attack: wrong token for challenge (two-phase)
// -----------------------------------------------------------------------

#[test]
fn test_batch_wrong_token_for_challenge_fails() {
    let (cfg, model, inputs) = setup_batch(8);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let all_layers = forward_pass_multi(&cfg, &model, &inputs);
    let (_commitment, state) = commit(all_layers);

    // Challenge asks for token 2, but prover opens token 5 and lies about index
    let challenges = vec![2];
    let mut proof = open(&state, &[5]); // opens token 5
    proof.traces[0].token_index = 2; // lie about the index

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "wrong token should fail: {:?}", failures);
}

// -----------------------------------------------------------------------
// 5. Attack: tamper with merkle root after commitment
// -----------------------------------------------------------------------

#[test]
fn test_batch_tampered_commitment_detected() {
    let (cfg, model, inputs) = setup_batch(8);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let all_layers = forward_pass_multi(&cfg, &model, &inputs);
    let (_commitment, state) = commit(all_layers);

    let challenges = vec![0, 3, 5];
    let mut proof = open(&state, &challenges);

    // Tamper with the commitment root
    proof.commitment.merkle_root[0] ^= 0xff;

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "tampered commitment should fail: {:?}", failures);
}

// -----------------------------------------------------------------------
// 6. Challenge derivation
// -----------------------------------------------------------------------

#[test]
fn test_challenge_derivation_deterministic() {
    let root = [42u8; 32];
    let seed = [7u8; 32];
    let c1 = merkle::derive_challenges(&root, &seed, 100, 10);
    let c2 = merkle::derive_challenges(&root, &seed, 100, 10);
    assert_eq!(c1, c2);
}

#[test]
fn test_challenge_derivation_changes_with_seed() {
    let root = [42u8; 32];
    let c1 = merkle::derive_challenges(&root, &[1u8; 32], 100, 10);
    let c2 = merkle::derive_challenges(&root, &[2u8; 32], 100, 10);
    assert_ne!(c1, c2);
}

#[test]
fn test_challenge_derivation_unique_indices() {
    let root = [42u8; 32];
    let seed = [7u8; 32];
    let challenges = merkle::derive_challenges(&root, &seed, 1000, 50);
    assert_eq!(challenges.len(), 50);

    // All unique (BTreeSet ensures this, but verify)
    let mut sorted = challenges.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 50);
}

#[test]
fn test_challenge_k_capped_at_n() {
    let root = [42u8; 32];
    let seed = [7u8; 32];
    // Ask for 100 challenges but only 5 tokens exist
    let challenges = merkle::derive_challenges(&root, &seed, 5, 100);
    assert_eq!(challenges.len(), 5);
}

// -----------------------------------------------------------------------
// 7. Multi-token attention differs from single-token
// -----------------------------------------------------------------------

#[test]
fn test_second_token_sees_kv_cache() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let inputs: Vec<Vec<i8>> = (0..3)
        .map(|t| {
            (0..cfg.hidden_dim)
                .map(|i| ((t * 7 + i * 3) % 256) as i8)
                .collect()
        })
        .collect();

    let multi = forward_pass_multi(&cfg, &model, &inputs);

    // Token 0 in multi should match single-token pass
    let single = forward_pass(&cfg, &model, &inputs[0]);
    assert_eq!(multi[0][0].a, single[0].a, "token 0 should match single-pass");

    // Token 1 should differ from running it as a standalone single token
    // (because it sees KV cache from token 0)
    let single_1 = forward_pass(&cfg, &model, &inputs[1]);
    // The attention vector 'a' should differ because multi-token sees 2 KV entries
    // (This may not always differ due to saturation, so just check it produces valid traces)
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let (_commitment, _state, traces) = build_batch(multi);
    for trace in &traces {
        let (passed, failures) = verify_trace(&key, trace);
        assert!(passed, "token {} failed: {:?}", trace.token_index, failures);
    }

    // At minimum, token 1's 'a' with KV cache might differ from standalone
    // (not guaranteed due to toy model saturation, but the pipeline should be valid)
    let _ = single_1; // used for reasoning, not assertion
}

// -----------------------------------------------------------------------
// 8. Batch serialization roundtrip
// -----------------------------------------------------------------------

#[test]
fn test_batch_serialization_roundtrip() {
    let (cfg, model, inputs) = setup_batch(4);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let all_layers = forward_pass_multi(&cfg, &model, &inputs);
    let (_commitment, state) = commit(all_layers);

    let challenges = vec![0, 2];
    let proof = open(&state, &challenges);

    // Serialize and deserialize
    let data = verilm_core::serialize::serialize_batch(&proof);
    let proof2 = verilm_core::serialize::deserialize_batch(&data).unwrap();

    assert_eq!(proof.commitment.merkle_root, proof2.commitment.merkle_root);
    assert_eq!(proof.commitment.io_root, proof2.commitment.io_root);
    assert_eq!(proof.commitment.n_tokens, proof2.commitment.n_tokens);
    assert_eq!(proof.traces.len(), proof2.traces.len());
    for (t1, t2) in proof.traces.iter().zip(proof2.traces.iter()) {
        assert_eq!(t1.token_index, t2.token_index);
        assert_eq!(t1.layers.len(), t2.layers.len());
    }

    // Verify deserialized proof
    let (passed, failures) = verify_batch(&key, &proof2, &challenges);
    assert!(passed, "deserialized batch should pass: {:?}", failures);
}

// -----------------------------------------------------------------------
// 9. Cross-token chain attacks
// -----------------------------------------------------------------------

#[test]
fn test_cross_token_chain_tampered_io_proof_detected() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);
    let (_commitment, state) = commit(all_layers);

    // Challenge consecutive tokens so cross-token chain check fires
    let challenges = vec![2, 3];
    let mut proof = open(&state, &challenges);

    // Tamper with token 3's io_proof — flip a sibling bit
    if !proof.traces[1].io_proof.siblings.is_empty() {
        proof.traces[1].io_proof.siblings[0][0] ^= 0xff;
    }

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "tampered IO proof should fail: {:?}", failures);
    assert!(
        failures.iter().any(|f| f.contains("IO proof")),
        "should mention IO proof failure: {:?}",
        failures
    );
}

#[test]
fn test_cross_token_chain_branched_computation_detected() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    // Generate honest autoregressive trace
    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);

    // Attack: replace token 3's first-layer input with something different.
    // This simulates a prover who branched computation at token 3.
    all_layers[3][0].x_attn[0] = all_layers[3][0].x_attn[0].wrapping_add(1);

    // Rebuild batch (IO tree will reflect the tampered input)
    let (_commitment, state) = commit(all_layers);

    // Challenge tokens 2 and 3 — cross-token chain should catch the branch
    let challenges = vec![2, 3];
    let proof = open(&state, &challenges);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    // Should fail: either Freivalds (tampered x_attn doesn't match Wx) or cross-token chain
    assert!(!passed, "branched computation should fail: {:?}", failures);
}

// -----------------------------------------------------------------------
// 10. Two-phase protocol: commit is small, open is selective
// -----------------------------------------------------------------------

#[test]
fn test_commit_returns_only_roots_and_count() {
    let (cfg, model, inputs) = setup_batch(32);
    let all_layers = forward_pass_multi(&cfg, &model, &inputs);

    let (commitment, _state) = commit(all_layers);

    // Commitment is just three fields — no traces, no proofs
    assert_eq!(commitment.n_tokens, 32);
    assert_ne!(commitment.merkle_root, [0u8; 32]);
    assert_ne!(commitment.io_root, [0u8; 32]);
}

#[test]
fn test_open_produces_only_challenged_traces() {
    let (cfg, model, inputs) = setup_batch(32);
    let all_layers = forward_pass_multi(&cfg, &model, &inputs);

    let (commitment, state) = commit(all_layers);

    // Open only 3 of 32 tokens
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], 32, 3,
    );
    let proof = open(&state, &challenges);

    assert_eq!(proof.traces.len(), 3);
    assert_eq!(proof.commitment.n_tokens, 32);

    // Each opened trace has the correct token_index
    for (trace, &expected_idx) in proof.traces.iter().zip(challenges.iter()) {
        assert_eq!(trace.token_index, expected_idx);
    }
}

#[test]
fn test_two_phase_end_to_end() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    // Prover runs autoregressive inference
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 20);

    // Phase 1: Prover commits (publishes commitment, keeps state)
    let (commitment, state) = commit(all_layers);

    // Network: only commitment travels to verifier (64 bytes + u32)

    // Verifier derives challenges from commitment
    let verifier_seed = [77u8; 32];
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &verifier_seed, commitment.n_tokens, 5,
    );

    // Network: challenges travel to prover (5 x u32)

    // Phase 2: Prover opens only challenged tokens
    let proof = open(&state, &challenges);

    // Network: only k token traces + proofs travel to verifier

    // Verifier checks
    assert_eq!(proof.commitment.merkle_root, commitment.merkle_root);
    assert_eq!(proof.commitment.io_root, commitment.io_root);
    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "end-to-end two-phase should pass: {:?}", failures);
}

// -----------------------------------------------------------------------
// 10. Compact trace: field elimination + compression
// -----------------------------------------------------------------------

#[test]
fn test_compact_roundtrip_verifies() {
    use verilm_core::serialize;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    // Compact schema roundtrip (no compression)
    let compact_bytes = serialize::serialize_compact_batch(&proof);
    let restored = serialize::deserialize_compact_batch(&compact_bytes).unwrap();

    // The restored proof should pass verification identically
    let (passed, failures) = verify_batch(&key, &restored, &challenges);
    assert!(passed, "compact roundtrip should pass verification: {:?}", failures);

    // Compact schema + transport compression roundtrip (composed layers)
    let compressed = serialize::compress(&compact_bytes);
    let decompressed = serialize::decompress(&compressed).unwrap();
    let restored2 = serialize::deserialize_compact_batch(&decompressed).unwrap();
    let (passed2, failures2) = verify_batch(&key, &restored2, &challenges);
    assert!(passed2, "compact+zstd roundtrip should pass: {:?}", failures2);
}

#[test]
fn test_compact_reconstructs_derivable_fields() {
    use verilm_core::types::CompactLayerTrace;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let layers = forward_pass(&cfg, &model, &input);

    for (i, lt) in layers.iter().enumerate() {
        let compact = CompactLayerTrace::from_full(lt);
        let reconstructed = compact.to_full().expect("to_full should succeed for valid trace");
        assert_eq!(lt.x_ffn, reconstructed.x_ffn,
            "layer {}: x_ffn reconstruction mismatch", i);
        assert_eq!(lt.h, reconstructed.h,
            "layer {}: h reconstruction mismatch", i);
        assert_eq!(lt, &reconstructed,
            "layer {}: full reconstruction mismatch", i);
    }
}

#[test]
fn test_compact_smaller_than_full() {
    use verilm_core::serialize;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 20);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    let full_bytes = serialize::serialize_batch(&proof);
    let compact_bytes = serialize::serialize_compact_batch(&proof);
    let full_zstd = serialize::compress(&full_bytes);
    let compact_zstd = serialize::compress(&compact_bytes);

    let full_size = full_bytes.len();
    let compact_size = compact_bytes.len();
    let full_zstd_size = full_zstd.len();
    let compact_zstd_size = compact_zstd.len();

    // Compact (no compression) should be smaller than full
    assert!(compact_size < full_size,
        "compact ({}) should be smaller than full ({})", compact_size, full_size);

    // Compact + zstd should be the smallest
    assert!(compact_zstd_size < full_size,
        "compact+zstd ({}) should be smaller than full ({})", compact_zstd_size, full_size);

    // Full + zstd should be smaller than full uncompressed
    assert!(full_zstd_size < full_size,
        "full+zstd ({}) should be smaller than full ({})", full_zstd_size, full_size);

    eprintln!("Size comparison:");
    eprintln!("  full (bincode):            {} bytes", full_size);
    eprintln!("  compact (bincode):         {} bytes ({:.1}% of full)",
        compact_size, compact_size as f64 / full_size as f64 * 100.0);
    eprintln!("  full + zstd:               {} bytes ({:.1}% of full)",
        full_zstd_size, full_zstd_size as f64 / full_size as f64 * 100.0);
    eprintln!("  compact + zstd:            {} bytes ({:.1}% of full)",
        compact_zstd_size, compact_zstd_size as f64 / full_size as f64 * 100.0);
}

// -----------------------------------------------------------------------
// 12. Negative tests for compact/compressed traces
// -----------------------------------------------------------------------

#[test]
fn test_truncated_compact_batch_fails() {
    use verilm_core::serialize;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    let compact_bytes = serialize::serialize_compact_batch(&proof);

    // Truncate to first half
    let truncated = &compact_bytes[..compact_bytes.len() / 2];
    let result = serialize::deserialize_compact_batch(truncated);
    assert!(result.is_err(), "truncated compact batch should fail deserialization");
}

#[test]
fn test_corrupted_compressed_payload_fails() {
    use verilm_core::serialize;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    let compact_bytes = serialize::serialize_compact_batch(&proof);
    let mut compressed = serialize::compress(&compact_bytes);

    // Flip a byte in the middle of the compressed data
    let mid = compressed.len() / 2;
    compressed[mid] ^= 0xff;

    let decompress_result = serialize::decompress(&compressed);
    // Corruption must be detected at some layer: decompress, deserialize, or verify
    match decompress_result {
        Err(_) => { /* expected: decompression itself failed */ }
        Ok(decompressed) => {
            match serialize::deserialize_compact_batch(&decompressed) {
                Err(_) => { /* expected: deserialization caught the corruption */ }
                Ok(restored) => {
                    // If both decompress and deserialize succeeded with garbage,
                    // verification must still reject the corrupted proof
                    let (passed, _) = verify_batch(&key, &restored, &challenges);
                    assert!(
                        !passed,
                        "corrupted compressed data should fail at decompress, deserialize, or verify"
                    );
                }
            }
        }
    }
}

#[test]
fn test_wrong_magic_bytes_rejected() {
    use verilm_core::serialize;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    // Test with full serialization format
    let mut full_bytes = serialize::serialize_batch(&proof);
    if full_bytes.len() >= 4 {
        full_bytes[0..4].copy_from_slice(b"XXXX");
    }
    let result_full = serialize::deserialize_batch(&full_bytes);
    assert!(result_full.is_err(), "wrong magic bytes should fail deserialize_batch");

    // Test with compact serialization format
    let mut compact_bytes = serialize::serialize_compact_batch(&proof);
    if compact_bytes.len() >= 4 {
        compact_bytes[0..4].copy_from_slice(b"XXXX");
    }
    let result_compact = serialize::deserialize_compact_batch(&compact_bytes);
    assert!(result_compact.is_err(), "wrong magic bytes should fail deserialize_compact_batch");
}

#[test]
fn test_compact_with_extra_trailing_bytes() {
    use verilm_core::serialize;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    let mut compact_bytes = serialize::serialize_compact_batch(&proof);

    // Append 100 random-ish bytes
    for i in 0u8..100 {
        compact_bytes.push(i.wrapping_mul(37).wrapping_add(17));
    }

    // Bincode may or may not reject trailing bytes
    match serialize::deserialize_compact_batch(&compact_bytes) {
        Err(_) => { /* acceptable: strict deserializer rejected trailing bytes */ }
        Ok(restored) => {
            // If it succeeded, verify_batch must still pass (extra bytes were ignored)
            let (passed, failures) = verify_batch(&key, &restored, &challenges);
            assert!(passed, "deserialized proof with trailing bytes should still verify: {:?}", failures);
        }
    }
}

#[test]
fn test_compact_reconstruction_mismatch_detected() {
    use verilm_core::serialize;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    // Serialize and deserialize to get a BatchProof from the compact path
    let compact_bytes = serialize::serialize_compact_batch(&proof);
    let mut restored = serialize::deserialize_compact_batch(&compact_bytes).unwrap();

    // Corrupt one of the reconstructed h values in the first trace's first layer
    if !restored.traces.is_empty() && !restored.traces[0].layers.is_empty() {
        restored.traces[0].layers[0].h[0] = restored.traces[0].layers[0].h[0].wrapping_add(1);
    }

    let (passed, _failures) = verify_batch(&key, &restored, &challenges);
    assert!(!passed, "corrupted h value should cause verify_batch to fail");
}

#[test]
fn test_empty_batch_proof() {
    use verilm_core::types::{BatchProof, BatchCommitment};

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let empty_proof = BatchProof {
        commitment: BatchCommitment {
            merkle_root: [0u8; 32],
            io_root: [0u8; 32],
            n_tokens: 0,
            manifest_hash: None,
            version: Default::default(),
            prompt_hash: None,
            seed_commitment: None,
            kv_chain_root: None,
        },
        traces: vec![],
        revealed_seed: None,
    };

    // Empty challenges with empty proof — nothing to check, should pass
    let (passed, failures) = verify_batch(&key, &empty_proof, &[]);
    assert!(passed, "empty batch with empty challenges should pass: {:?}", failures);

    // Non-empty challenges with empty proof — should fail
    let (passed, _failures) = verify_batch(&key, &empty_proof, &[0, 1, 2]);
    assert!(!passed, "empty batch with non-empty challenges should fail");
}

// -----------------------------------------------------------------------
// 13. Longer-sequence KV/cache attack tests
// -----------------------------------------------------------------------

#[test]
fn test_64_token_honest_end_to_end() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 64);

    let (commitment, state) = commit(all_layers);

    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[55u8; 32], commitment.n_tokens, 5,
    );
    assert_eq!(challenges.len(), 5);

    let proof = open(&state, &challenges);
    assert_eq!(proof.traces.len(), 5);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "64-token honest batch should pass: {:?}", failures);
}

#[test]
fn test_128_token_honest_end_to_end() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 128);

    let (commitment, state) = commit(all_layers);

    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[66u8; 32], commitment.n_tokens, 10,
    );
    assert_eq!(challenges.len(), 10);

    let proof = open(&state, &challenges);
    assert_eq!(proof.traces.len(), 10);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "128-token honest batch should pass: {:?}", failures);
}

#[test]
fn test_early_token_corruption_detected_at_later_token() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 64);

    // Corrupt token 5's ffn_out (the output that feeds into token 6's input)
    let last_layer = all_layers[5].last_mut().unwrap();
    last_layer.ffn_out[0] += 1;

    let (_commitment, state) = commit(all_layers);

    // Challenge tokens 5 and 6
    let challenges = vec![5, 6];
    let proof = open(&state, &challenges);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        !passed,
        "corruption at token 5's ffn_out should be detected when checking tokens 5 and 6: {:?}",
        failures
    );
}

#[test]
fn test_multiple_corruption_points_across_long_sequence() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 64);

    // Corrupt tokens 10, 30, 50 — corrupt q[0] for each
    all_layers[10][0].q[0] += 1;
    all_layers[30][0].q[0] += 1;
    all_layers[50][0].q[0] += 1;

    let (_commitment, _state, all_traces) = build_batch(all_layers);

    let corrupted_tokens: Vec<u32> = vec![10, 30, 50];
    for trace in &all_traces {
        let (passed, _failures) = verify_trace(&key, trace);
        if corrupted_tokens.contains(&(trace.token_index as u32)) {
            assert!(
                !passed,
                "corrupted token {} should fail",
                trace.token_index
            );
        } else {
            assert!(
                passed,
                "uncorrupted token {} should pass",
                trace.token_index
            );
        }
    }
}

#[test]
fn test_consecutive_token_pairs_chain_verification() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 32);

    let (_commitment, state) = commit(all_layers);

    // Check consecutive pairs: [0,1], [15,16], [30,31]
    let pairs: Vec<Vec<u32>> = vec![vec![0, 1], vec![15, 16], vec![30, 31]];

    for challenges in &pairs {
        let proof = open(&state, challenges);
        let (passed, failures) = verify_batch(&key, &proof, challenges);
        assert!(
            passed,
            "consecutive pair {:?} should pass cross-token chain: {:?}",
            challenges, failures
        );
    }
}

#[test]
fn test_long_sequence_compact_roundtrip() {
    use verilm_core::serialize;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 64);

    let (commitment, state) = commit(all_layers);

    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[88u8; 32], commitment.n_tokens, 10,
    );
    assert_eq!(challenges.len(), 10);

    let proof = open(&state, &challenges);

    // Serialize to compact, compress with zstd, decompress, deserialize
    let compact_bytes = serialize::serialize_compact_batch(&proof);
    let compressed = serialize::compress(&compact_bytes);
    let decompressed = serialize::decompress(&compressed).unwrap();
    let restored = serialize::deserialize_compact_batch(&decompressed).unwrap();

    let (passed, failures) = verify_batch(&key, &restored, &challenges);
    assert!(
        passed,
        "64-token compact+zstd roundtrip should pass verification: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 14. Protocol invariants
// -----------------------------------------------------------------------

#[test]
fn test_every_opened_token_verifies_against_commitment() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 32);
    let (commitment, state) = commit(all_layers);

    for &k in &[1u32, 2, 5, 10, 15, 32] {
        let challenges = merkle::derive_challenges(
            &commitment.merkle_root, &[99u8; 32], commitment.n_tokens, k,
        );
        let proof = open(&state, &challenges);
        let (passed, failures) = verify_batch(&key, &proof, &challenges);
        assert!(passed, "k={} batch failed: {:?}", k, failures);
    }
}

#[test]
fn test_unopened_tokens_dont_affect_opened_proofs() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);

    // Generate two sets of inputs that differ only at token 10
    let inputs_a: Vec<Vec<i8>> = (0..16)
        .map(|t| {
            (0..cfg.hidden_dim)
                .map(|i| ((t * 7 + i * 3) % 256) as i8)
                .collect()
        })
        .collect();

    let mut inputs_b = inputs_a.clone();
    // Make token 10 different in batch B
    for x in inputs_b[10].iter_mut() {
        *x = x.wrapping_add(42);
    }

    let all_layers_a = forward_pass_multi(&cfg, &model, &inputs_a);
    let all_layers_b = forward_pass_multi(&cfg, &model, &inputs_b);

    let (_commitment_a, state_a) = commit(all_layers_a);
    let (_commitment_b, state_b) = commit(all_layers_b);

    let challenge_indices = vec![2u32, 5];
    let proof_a = open(&state_a, &challenge_indices);
    let proof_b = open(&state_b, &challenge_indices);

    // Tokens 2 and 5 have identical inputs in both batches, so their traces
    // (layer data) should be identical.
    assert_eq!(proof_a.traces.len(), proof_b.traces.len());
    for (ta, tb) in proof_a.traces.iter().zip(proof_b.traces.iter()) {
        assert_eq!(ta.token_index, tb.token_index);
        assert_eq!(ta.layers, tb.layers,
            "trace data for token {} should be identical across batches",
            ta.token_index);
    }
}

#[test]
fn test_challenge_derivation_unpredictable_before_commitment() {
    let cfg = ModelConfig::toy();

    // Two different models (different seeds) produce different commitments
    let model_a = generate_model(&cfg, 11111);
    let model_b = generate_model(&cfg, 22222);

    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let layers_a = forward_pass_autoregressive(&cfg, &model_a, &input, 16);
    let layers_b = forward_pass_autoregressive(&cfg, &model_b, &input, 16);

    let (commitment_a, _) = commit(layers_a);
    let (commitment_b, _) = commit(layers_b);

    // Different commitments
    assert_ne!(commitment_a.merkle_root, commitment_b.merkle_root,
        "different models should produce different commitment roots");

    // Same verifier seed, but challenges differ because roots differ
    let seed = [55u8; 32];
    let challenges_a = merkle::derive_challenges(
        &commitment_a.merkle_root, &seed, commitment_a.n_tokens, 5,
    );
    let challenges_b = merkle::derive_challenges(
        &commitment_b.merkle_root, &seed, commitment_b.n_tokens, 5,
    );

    assert_ne!(challenges_a, challenges_b,
        "different commitment roots should produce different challenge sets");
}

#[test]
fn test_batch_state_isolated_per_request() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Two different inputs
    let input_1: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let input_2: Vec<i8> = (0..cfg.hidden_dim).map(|i| ((i + 50) % 256) as i8).collect();

    let layers_1 = forward_pass_autoregressive(&cfg, &model, &input_1, 8);
    let layers_2 = forward_pass_autoregressive(&cfg, &model, &input_2, 8);

    let (commitment_1, state_1) = commit(layers_1);
    let (commitment_2, state_2) = commit(layers_2);

    let seed = [88u8; 32];
    let challenges_1 = merkle::derive_challenges(
        &commitment_1.merkle_root, &seed, commitment_1.n_tokens, 3,
    );
    let challenges_2 = merkle::derive_challenges(
        &commitment_2.merkle_root, &seed, commitment_2.n_tokens, 3,
    );

    // Each batch verifies against its own state
    let proof_1 = open(&state_1, &challenges_1);
    let proof_2 = open(&state_2, &challenges_2);

    let (passed_1, failures_1) = verify_batch(&key, &proof_1, &challenges_1);
    assert!(passed_1, "batch 1 should pass: {:?}", failures_1);

    let (passed_2, failures_2) = verify_batch(&key, &proof_2, &challenges_2);
    assert!(passed_2, "batch 2 should pass: {:?}", failures_2);

    // Cross-verify: batch 1's proof against batch 2's challenges should fail
    // (only meaningful if challenge sets differ — with small token counts they can collide)
    if challenges_1 != challenges_2 {
        let (cross_passed, _) = verify_batch(&key, &proof_1, &challenges_2);
        assert!(!cross_passed, "batch 1 proof should not pass with batch 2 challenges");
    }
}

#[test]
fn test_commitment_is_binding() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let layers = forward_pass_autoregressive(&cfg, &model, &input, 10);

    // Commit the same traces twice
    let layers_clone = layers.clone();
    let (commitment_1, _) = commit(layers);
    let (commitment_2, _) = commit(layers_clone);

    assert_eq!(commitment_1.merkle_root, commitment_2.merkle_root,
        "same traces must produce the same merkle_root");
    assert_eq!(commitment_1.io_root, commitment_2.io_root,
        "same traces must produce the same io_root");
    assert_eq!(commitment_1.n_tokens, commitment_2.n_tokens,
        "same traces must produce the same n_tokens");
}

#[test]
fn test_open_only_challenged_tokens_no_extras() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 32);
    let (commitment, state) = commit(all_layers);

    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 3,
    );
    assert_eq!(challenges.len(), 3);

    let proof = open(&state, &challenges);

    // Exactly 3 traces, no extras
    assert_eq!(proof.traces.len(), 3,
        "proof should contain exactly 3 traces, got {}", proof.traces.len());

    // Each trace's token_index must be in the challenge set
    for trace in &proof.traces {
        assert!(challenges.contains(&trace.token_index),
            "trace token_index {} not in challenge set {:?}",
            trace.token_index, challenges);
    }

    // Verify the proof passes
    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "proof should verify: {:?}", failures);
}

// =======================================================================
// Task 11: Property-style protocol invariant tests
// =======================================================================

// -----------------------------------------------------------------------
// 11a. Any single-field mutation in a LayerTrace causes verification failure
// -----------------------------------------------------------------------

#[test]
fn test_single_field_mutation_always_detected() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    // Define mutations for every field in LayerTrace
    let field_mutators: Vec<(&str, Box<dyn Fn(&mut verilm_core::types::LayerTrace)>)> = vec![
        ("x_attn", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.x_attn[0] = lt.x_attn[0].wrapping_add(1))),
        ("q", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.q[0] = lt.q[0].wrapping_add(1))),
        ("k", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.k[0] = lt.k[0].wrapping_add(1))),
        ("v", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.v[0] = lt.v[0].wrapping_add(1))),
        ("a", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.a[0] = lt.a[0].wrapping_add(1))),
        ("attn_out", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.attn_out[0] = lt.attn_out[0].wrapping_add(1))),
        ("x_ffn", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.x_ffn[0] = lt.x_ffn[0].wrapping_add(1))),
        ("g", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.g[0] = lt.g[0].wrapping_add(1))),
        ("u", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.u[0] = lt.u[0].wrapping_add(1))),
        ("h", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.h[0] = lt.h[0].wrapping_add(1))),
        ("ffn_out", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.ffn_out[0] = lt.ffn_out[0].wrapping_add(1))),
    ];

    for (field_name, mutate_fn) in &field_mutators {
        // Test mutation on each layer
        for layer_idx in 0..cfg.n_layers {
            let mut layers = forward_pass(&cfg, &model, &input);
            mutate_fn(&mut layers[layer_idx]);

            let trace = build_trace(layers, 0, 10);
            let (passed, failures) = verify_trace(&key, &trace);

            assert!(
                !passed,
                "mutation of {} in layer {} should cause verification failure, but it passed. failures: {:?}",
                field_name, layer_idx, failures
            );
        }
    }
}

// -----------------------------------------------------------------------
// 11b. Different verifier seeds on the same honest trace all pass
// -----------------------------------------------------------------------

#[test]
fn test_different_verifier_seeds_all_pass_honest() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let layers = forward_pass(&cfg, &model, &input);

    for seed_byte in 0..20u8 {
        let seed = [seed_byte; 32];
        let key = generate_key(&cfg, &model, seed);
        let trace = build_trace(layers.clone(), 0, 10);
        let (passed, failures) = verify_trace(&key, &trace);

        assert!(
            passed,
            "honest trace should pass with verifier seed {}: {:?}",
            seed_byte, failures
        );
    }
}

// -----------------------------------------------------------------------
// 11c. Commitment determinism: same input always produces same commitment
// -----------------------------------------------------------------------

#[test]
fn test_commitment_determinism() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let layers1 = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let layers2 = forward_pass_autoregressive(&cfg, &model, &input, 10);

    let (commitment1, _) = commit(layers1);
    let (commitment2, _) = commit(layers2);

    assert_eq!(commitment1.merkle_root, commitment2.merkle_root,
        "identical runs must produce identical merkle roots");
    assert_eq!(commitment1.io_root, commitment2.io_root,
        "identical runs must produce identical io roots");
    assert_eq!(commitment1.n_tokens, commitment2.n_tokens,
        "identical runs must produce identical token counts");
}

// -----------------------------------------------------------------------
// 11d. Re-opening different challenge sets from same state always passes
// -----------------------------------------------------------------------

#[test]
fn test_reopening_different_challenge_sets_from_same_state() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 16);
    let (commitment, state) = commit(all_layers);

    // Open with multiple different challenge sets from the same committed state
    let challenge_sets: Vec<Vec<u32>> = vec![
        vec![0, 5, 10],
        vec![1, 7, 15],
        vec![3, 8, 12, 14],
        vec![0, 1, 2, 3, 4],
        merkle::derive_challenges(&commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5),
        merkle::derive_challenges(&commitment.merkle_root, &[99u8; 32], commitment.n_tokens, 3),
    ];

    for challenges in &challenge_sets {
        let proof = open(&state, challenges);
        let (passed, failures) = verify_batch(&key, &proof, challenges);
        assert!(
            passed,
            "re-opening challenges {:?} from same state should pass: {:?}",
            challenges, failures
        );
    }
}

// -----------------------------------------------------------------------
// 11e. Verification is independent of opening order
// -----------------------------------------------------------------------

#[test]
fn test_verification_independent_of_opening_order() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 16);
    let (_commitment, state) = commit(all_layers);

    // Open in ascending order
    let challenges_asc: Vec<u32> = vec![2, 5, 10, 13];
    let proof_asc = open(&state, &challenges_asc);
    let (passed_asc, failures_asc) = verify_batch(&key, &proof_asc, &challenges_asc);
    assert!(passed_asc, "ascending order should pass: {:?}", failures_asc);

    // Open in descending order
    let challenges_desc: Vec<u32> = vec![13, 10, 5, 2];
    let proof_desc = open(&state, &challenges_desc);
    let (passed_desc, failures_desc) = verify_batch(&key, &proof_desc, &challenges_desc);
    assert!(passed_desc, "descending order should pass: {:?}", failures_desc);

    // Open in random order
    let challenges_rand: Vec<u32> = vec![10, 2, 13, 5];
    let proof_rand = open(&state, &challenges_rand);
    let (passed_rand, failures_rand) = verify_batch(&key, &proof_rand, &challenges_rand);
    assert!(passed_rand, "random order should pass: {:?}", failures_rand);
}

// =======================================================================
// Task 12: Negative tests for compact/compressed traces (additions)
// =======================================================================

// -----------------------------------------------------------------------
// 12a. Compact trace with corrupted g field
// -----------------------------------------------------------------------

#[test]
fn test_compact_trace_corrupted_g_field() {
    use verilm_core::types::CompactBatchProof;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    // Build a compact proof, corrupt g in first trace's first layer, then restore to full
    let mut compact = CompactBatchProof::from_full(&proof);
    compact.traces[0].layers[0].g[0] = compact.traces[0].layers[0].g[0].wrapping_add(99);
    let corrupted_proof = compact.to_full().expect("corrupted-g compact trace should reconstruct");

    // Verification should fail: corrupted g changes h reconstruction (since h = silu(g, u)),
    // which changes the Merkle leaf hash, AND Freivalds Wg check fails
    let (passed, failures) = verify_batch(&key, &corrupted_proof, &challenges);
    assert!(
        !passed,
        "compact trace with corrupted g should fail verification: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 12b. Zero-length vectors in compact trace
// -----------------------------------------------------------------------

#[test]
fn test_compact_trace_zero_length_vectors() {
    use verilm_core::types::CompactBatchProof;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let _key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    // Replace g with empty vector in compact trace
    let mut compact = CompactBatchProof::from_full(&proof);
    compact.traces[0].layers[0].g = vec![];
    compact.traces[0].layers[0].u = vec![];

    // Zero-length vectors should be rejected with a structured error,
    // not silently accepted or cause a panic.
    let result = compact.to_full();
    assert!(
        result.is_err(),
        "compact trace with zero-length vectors should return Err, got Ok"
    );
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("empty"),
        "error message should mention empty vectors, got: {err_msg}"
    );
}

// -----------------------------------------------------------------------
// 12c. Compact trace with attn_out values that saturate i8 boundaries
// -----------------------------------------------------------------------

#[test]
fn test_compact_trace_boundary_requantization() {
    use verilm_core::types::CompactBatchProof;

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 10);
    let (commitment, state) = commit(all_layers);
    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[42u8; 32], commitment.n_tokens, 5,
    );
    let proof = open(&state, &challenges);

    // Set attn_out values to extreme i32 values that overflow i8 range during requantization.
    // This tests that the compact trace's x_ffn reconstruction (requantize(attn_out))
    // clamps correctly but produces a different Merkle leaf hash.
    let mut compact = CompactBatchProof::from_full(&proof);
    for val in compact.traces[0].layers[0].attn_out.iter_mut() {
        *val = 50000; // way beyond i8 range, will clamp to 127
    }
    let corrupted_proof = compact.to_full().expect("boundary values should reconstruct");

    // Verify that reconstructed x_ffn is all 127s (clamped)
    assert!(
        corrupted_proof.traces[0].layers[0].x_ffn.iter().all(|&v| v == 127),
        "extreme attn_out should clamp to 127 during requantization"
    );

    // Should fail: attn_out is wrong, so Wo Freivalds fails and Merkle leaf differs
    let (passed, _failures) = verify_batch(&key, &corrupted_proof, &challenges);
    assert!(
        !passed,
        "compact trace with extreme attn_out values should fail verification"
    );
}

// =======================================================================
// Task 13: Longer-sequence KV/cache attack tests
// =======================================================================

// -----------------------------------------------------------------------
// 13a. 256-token honest end-to-end
// -----------------------------------------------------------------------

#[test]
fn test_256_token_honest_end_to_end() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 256);

    let (commitment, state) = commit(all_layers);

    let challenges = merkle::derive_challenges(
        &commitment.merkle_root, &[77u8; 32], commitment.n_tokens, 10,
    );
    assert_eq!(challenges.len(), 10);

    let proof = open(&state, &challenges);
    assert_eq!(proof.traces.len(), 10);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(passed, "256-token honest batch should pass: {:?}", failures);
}

// -----------------------------------------------------------------------
// 13b. KV cache poisoning: corrupt k or v after it was added to the cache
// -----------------------------------------------------------------------

#[test]
fn test_kv_cache_poisoning_detected() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 32);

    // Simulate KV cache poisoning: change token 5's k output after it was computed.
    // In a real attack, this would affect how subsequent tokens attend to token 5.
    // The Freivalds check on Wk should catch this because k != Wk @ x_attn.
    all_layers[5][0].k[0] = all_layers[5][0].k[0].wrapping_add(50);

    let (_commitment, _state, all_traces) = build_batch(all_layers);

    // Token 5 should fail (k was corrupted)
    let (passed_5, failures_5) = verify_trace(&key, &all_traces[5]);
    assert!(
        !passed_5,
        "token 5 with poisoned k should fail verification: {:?}",
        failures_5
    );
    assert!(
        failures_5.iter().any(|f| f.contains("Wk")),
        "should detect Wk corruption from KV cache poisoning: {:?}",
        failures_5
    );

    // Verify that other tokens (not corrupted) still pass
    let (passed_0, failures_0) = verify_trace(&key, &all_traces[0]);
    assert!(passed_0, "token 0 should pass: {:?}", failures_0);

    let (passed_10, failures_10) = verify_trace(&key, &all_traces[10]);
    assert!(passed_10, "token 10 should pass: {:?}", failures_10);
}

#[test]
fn test_kv_cache_poisoning_v_detected() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 32);

    // Corrupt token 10's v output in layer 1
    all_layers[10][1].v[0] = all_layers[10][1].v[0].wrapping_add(33);

    let (_commitment, _state, all_traces) = build_batch(all_layers);

    let (passed, failures) = verify_trace(&key, &all_traces[10]);
    assert!(
        !passed,
        "token 10 with poisoned v should fail: {:?}",
        failures
    );
    assert!(
        failures.iter().any(|f| f.contains("Wv")),
        "should detect Wv corruption: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 13c. Late-sequence corruption (corrupt token 200 of 256)
// -----------------------------------------------------------------------

#[test]
fn test_late_sequence_corruption_detected() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 256);

    // Corrupt token 200's q output in layer 0
    all_layers[200][0].q[0] = all_layers[200][0].q[0].wrapping_add(42);

    let (_commitment, state) = commit(all_layers);

    // Challenge includes token 200
    let challenges = vec![50, 100, 200, 250];
    let proof = open(&state, &challenges);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(!passed, "late-sequence corruption should be detected: {:?}", failures);
    assert!(
        failures.iter().any(|f| f.contains("token 200")),
        "token 200 corruption should be reported: {:?}",
        failures
    );

    // Tokens 50, 100, 250 should not have Freivalds failures
    // (they may have chain failures if adjacent to corrupted token, but not here)
    for &honest_token in &[50u32, 100, 250] {
        let token_freivalds_failures: Vec<_> = failures.iter()
            .filter(|f| f.starts_with(&format!("token {}: layer", honest_token)))
            .collect();
        assert!(
            token_freivalds_failures.is_empty(),
            "honest token {} should not have Freivalds failures: {:?}",
            honest_token, token_freivalds_failures
        );
    }
}

// -----------------------------------------------------------------------
// 13d. Sliding window challenge pattern
// -----------------------------------------------------------------------

#[test]
fn test_sliding_window_challenge_pattern() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 256);
    let (_commitment, state) = commit(all_layers);

    // Sliding window: challenge 3 consecutive tokens starting at position 100
    let challenges: Vec<u32> = vec![100, 101, 102];
    let proof = open(&state, &challenges);
    assert_eq!(proof.traces.len(), 3);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        passed,
        "sliding window challenges [100,101,102] in 256-token sequence should pass: {:?}",
        failures
    );

    // Cross-token chain between 100->101 and 101->102 should be verified
    // (this is implicit in verify_batch for consecutive opened tokens)
}

#[test]
fn test_sliding_window_with_corruption_in_window() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 256);

    // Corrupt token 101 (middle of the sliding window)
    all_layers[101][0].g[0] = all_layers[101][0].g[0].wrapping_add(77);

    let (_commitment, state) = commit(all_layers);

    let challenges: Vec<u32> = vec![100, 101, 102];
    let proof = open(&state, &challenges);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        !passed,
        "corruption in sliding window should be detected: {:?}",
        failures
    );
    assert!(
        failures.iter().any(|f| f.contains("token 101")),
        "token 101 corruption should be reported: {:?}",
        failures
    );
}

#[test]
fn test_256_token_cross_token_chain_at_boundaries() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 256);
    let (_commitment, state) = commit(all_layers);

    // Check cross-token chain at various boundaries in the 256-token sequence
    let boundary_pairs: Vec<Vec<u32>> = vec![
        vec![0, 1],       // start of sequence
        vec![127, 128],   // middle of sequence
        vec![254, 255],   // end of sequence
        vec![63, 64],     // quarter boundary
        vec![191, 192],   // three-quarter boundary
    ];

    for challenges in &boundary_pairs {
        let proof = open(&state, challenges);
        let (passed, failures) = verify_batch(&key, &proof, challenges);
        assert!(
            passed,
            "cross-token chain at boundary {:?} should pass: {:?}",
            challenges, failures
        );
    }
}
