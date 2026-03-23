//! Property-based tests for critical protocol invariants.
//!
//! Uses proptest to generate random inputs and verify that:
//! 1. Challenge-order invariance
//! 2. Challenge subset monotonicity
//! 3. Token-ID mutation causes rejection
//! 4. Prompt-hash mutation causes rejection
//! 5. Seed mutation causes rejection
//! 6. KV-chain-hash mutation causes rejection
//! 7. Serialization roundtrip identity
//! 8. Compact format reconstruction correctness

use proptest::prelude::*;
use verilm_core::constants::ModelConfig;
use verilm_core::merkle;
use verilm_core::serialize;
use verilm_core::types::CompactBatchProof;
use verilm_test_vectors::*;

/// Shared setup: generate model, key, and an honest V3 commitment with N tokens.
fn honest_v3_setup(
    n_tokens: usize,
    model_seed: u64,
) -> (
    ModelConfig,
    Vec<LayerWeights>,
    verilm_core::types::VerifierKey,
    verilm_core::types::BatchCommitment,
    BatchState,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, model_seed);
    let key = generate_key(&cfg, &model, [7u8; 32]);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, n_tokens);
    let token_ids: Vec<u32> = (0..n_tokens as u32).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"property test prompt",
        sampling_seed: [42u8; 32],
        manifest: None,
    };
    let (commitment, state) = commit_with_full_binding(all_layers, &params, None);
    (cfg, model, key, commitment, state)
}

// =====================================================================
// 1. Challenge-order invariance
// =====================================================================

mod challenge_order_invariance {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Any permutation of the same challenge indices must produce
        /// identical verification results (both pass or both fail with same errors).
        #[test]
        fn any_permutation_same_result(
            n_tokens in 3usize..=6,
            perm_seed in 0u64..1000,
        ) {
            let (_cfg, _model, key, _commitment, state) = honest_v3_setup(n_tokens, 42);

            // Use all token indices as challenges
            let challenges: Vec<u32> = (0..n_tokens as u32).collect();

            let proof_orig = open(&state, &challenges);
            let (passed_orig, failures_orig) = verify_batch(&key, &proof_orig, &challenges);

            // Generate a deterministic permutation
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut perm = challenges.clone();
            // Fisher-Yates with deterministic seed
            for i in (1..perm.len()).rev() {
                let mut hasher = DefaultHasher::new();
                perm_seed.hash(&mut hasher);
                i.hash(&mut hasher);
                let j = (hasher.finish() as usize) % (i + 1);
                perm.swap(i, j);
            }

            let proof_perm = open(&state, &perm);
            let (passed_perm, failures_perm) = verify_batch(&key, &proof_perm, &perm);

            prop_assert_eq!(
                passed_orig, passed_perm,
                "order invariance violated: orig={:?}, perm={:?}, orig_fails={:?}, perm_fails={:?}",
                challenges, perm, failures_orig, failures_perm
            );
        }
    }
}

// =====================================================================
// 2. Challenge subset monotonicity
// =====================================================================

mod challenge_subset_monotonicity {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Verifying a subset of challenge indices must not fail if the full set passes.
        #[test]
        fn subset_passes_if_full_passes(
            n_tokens in 3usize..=7,
            subset_mask in prop::collection::vec(any::<bool>(), 3..=7),
        ) {
            let n = n_tokens.min(subset_mask.len());
            let (_cfg, _model, key, _commitment, state) = honest_v3_setup(n, 42);

            // Full challenge set
            let full: Vec<u32> = (0..n as u32).collect();
            let proof_full = open(&state, &full);
            let (passed_full, _) = verify_batch(&key, &proof_full, &full);

            if !passed_full {
                // If full set doesn't pass, skip (shouldn't happen with honest setup)
                return Ok(());
            }

            // Build a non-empty subset
            let subset: Vec<u32> = full.iter()
                .enumerate()
                .filter(|(i, _)| *i < subset_mask.len() && subset_mask[*i])
                .map(|(_, &idx)| idx)
                .collect();

            if subset.is_empty() {
                return Ok(());
            }

            let proof_subset = open(&state, &subset);
            let (passed_subset, failures_subset) = verify_batch(&key, &proof_subset, &subset);

            prop_assert!(
                passed_subset,
                "subset {:?} failed but full set {:?} passed: {:?}",
                subset, full, failures_subset
            );
        }
    }
}

// =====================================================================
// 3. Any mutation to committed token_id causes rejection
// =====================================================================

mod token_id_mutation {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(40))]

        /// Mutating any single token_id in the opened trace must cause verification failure.
        #[test]
        fn any_token_id_mutation_rejected(
            n_tokens in 2usize..=6,
            target_idx in 0usize..6,
            new_token_id in 1000u32..2000,
        ) {
            let n = n_tokens;
            let target = target_idx % n;
            let (_cfg, _model, key, _commitment, state) = honest_v3_setup(n, 42);

            let challenges: Vec<u32> = (0..n as u32).collect();
            let mut proof = open(&state, &challenges);

            // Mutate the token_id at target position
            let original = proof.traces[target].token_id;
            proof.traces[target].token_id = Some(new_token_id);

            // Only test if we actually changed the value
            if Some(new_token_id) == original {
                return Ok(());
            }

            let (passed, _failures) = verify_batch(&key, &proof, &challenges);
            prop_assert!(
                !passed,
                "mutating token_id at index {} from {:?} to {} should fail verification",
                target, original, new_token_id
            );
        }
    }
}

// =====================================================================
// 4. Any mutation to prompt_hash causes rejection
// =====================================================================

mod prompt_hash_mutation {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Changing the prompt hash on a V3 commitment must fail verification
        /// when the verifier checks expected prompt hash.
        #[test]
        fn prompt_hash_mutation_rejected(
            n_tokens in 2usize..=5,
            fake_prompt_byte in 0u8..255,
        ) {
            let n = n_tokens;
            let (_cfg, _model, key, _commitment, state) = honest_v3_setup(n, 42);

            let challenges: Vec<u32> = (0..n as u32).collect();
            let proof = open(&state, &challenges);

            // The honest prompt is "property test prompt"
            let honest_hash = merkle::hash_prompt(b"property test prompt");

            // Build a fake prompt hash
            let fake_prompt = vec![fake_prompt_byte; 20];
            let fake_hash = merkle::hash_prompt(&fake_prompt);

            // Skip if collision (astronomically unlikely)
            if fake_hash == honest_hash {
                return Ok(());
            }

            // Verify with the correct expected prompt hash passes
            let policy_correct = verilm_core::types::VerificationPolicy {
                expected_prompt_hash: Some(honest_hash),
                ..Default::default()
            };
            let (passed_correct, _) = verify_batch_with_policy(
                &key, &proof, &challenges, &policy_correct,
            );
            prop_assert!(passed_correct, "honest proof with correct prompt hash should pass");

            // Verify with a wrong expected prompt hash fails
            let policy_wrong = verilm_core::types::VerificationPolicy {
                expected_prompt_hash: Some(fake_hash),
                ..Default::default()
            };
            let (passed_wrong, failures) = verify_batch_with_policy(
                &key, &proof, &challenges, &policy_wrong,
            );
            prop_assert!(
                !passed_wrong,
                "proof with wrong expected prompt hash should fail: {:?}",
                failures
            );
        }
    }
}

// =====================================================================
// 5. Any mutation to seed causes rejection
// =====================================================================

mod seed_mutation {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Changing the revealed_seed on a V3 proof must fail seed-commitment verification.
        #[test]
        fn seed_mutation_rejected(
            n_tokens in 2usize..=5,
            fake_seed_byte in 0u8..255,
        ) {
            let n = n_tokens;
            let (_cfg, _model, key, _commitment, state) = honest_v3_setup(n, 42);

            let challenges: Vec<u32> = (0..n as u32).collect();
            let mut proof = open(&state, &challenges);

            // The honest seed is [42u8; 32]
            let fake_seed = [fake_seed_byte; 32];
            if fake_seed == [42u8; 32] {
                return Ok(());
            }

            proof.revealed_seed = Some(fake_seed);

            let (passed, failures) = verify_batch(&key, &proof, &challenges);
            prop_assert!(
                !passed,
                "mutated seed should fail verification: {:?}",
                failures
            );
            prop_assert!(
                failures.iter().any(|f| f.contains("seed commitment mismatch")),
                "should report seed mismatch, got: {:?}",
                failures
            );
        }
    }
}

// =====================================================================
// 6. Any mutation to kv_chain_hash causes rejection
// =====================================================================

mod kv_chain_mutation {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        /// Changing prev_kv_hash on any opened trace must fail KV chain checks
        /// when consecutive tokens are both opened.
        #[test]
        fn kv_chain_mutation_rejected(
            n_tokens in 3usize..=6,
            target_idx in 1usize..6,
            flip_byte in 0u8..32,
        ) {
            let n = n_tokens;
            let target = target_idx % (n - 1) + 1; // never 0 (which has genesis hash)
            let (_cfg, _model, key, _commitment, state) = honest_v3_setup(n, 42);

            // Open all tokens so cross-checks fire
            let challenges: Vec<u32> = (0..n as u32).collect();
            let mut proof = open(&state, &challenges);

            // Mutate prev_kv_hash on the target trace
            if let Some(ref mut hash) = proof.traces[target].prev_kv_hash {
                hash[flip_byte as usize % 32] ^= 0xff;
            } else {
                // No prev_kv_hash to mutate (shouldn't happen in V3)
                return Ok(());
            }

            let (passed, failures) = verify_batch(&key, &proof, &challenges);
            prop_assert!(
                !passed,
                "mutated prev_kv_hash at token {} should fail: {:?}",
                target, failures
            );
        }
    }
}

// =====================================================================
// 7. Serialization roundtrip identity for any valid proof
// =====================================================================

mod serialization_roundtrip {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        /// For any honestly generated batch proof, deserialize(serialize(proof)) == proof
        /// for both full and compact formats, with and without zstd compression.
        #[test]
        fn roundtrip_identity(
            n_tokens in 2usize..=6,
            model_seed in 0u64..100,
        ) {
            let (_cfg, _model, key, _commitment, state) = honest_v3_setup(n_tokens, model_seed);

            let challenges: Vec<u32> = (0..n_tokens as u32).collect();
            let proof = open(&state, &challenges);

            // 1. Full format roundtrip
            let full_bytes = serialize::serialize_batch(&proof);
            let proof_full_rt = serialize::deserialize_batch(&full_bytes)
                .expect("full deserialization failed");
            let (passed_full, failures_full) = verify_batch(&key, &proof_full_rt, &challenges);
            prop_assert!(passed_full, "full roundtrip should pass: {:?}", failures_full);

            // Check structural equality
            prop_assert_eq!(
                proof.commitment.merkle_root,
                proof_full_rt.commitment.merkle_root
            );
            prop_assert_eq!(
                proof.commitment.io_root,
                proof_full_rt.commitment.io_root
            );
            prop_assert_eq!(
                proof.commitment.n_tokens,
                proof_full_rt.commitment.n_tokens
            );
            prop_assert_eq!(proof.traces.len(), proof_full_rt.traces.len());
            prop_assert_eq!(proof.revealed_seed, proof_full_rt.revealed_seed);
            prop_assert_eq!(
                proof.commitment.prompt_hash,
                proof_full_rt.commitment.prompt_hash
            );
            prop_assert_eq!(
                proof.commitment.seed_commitment,
                proof_full_rt.commitment.seed_commitment
            );

            // 2. Compact format roundtrip
            let compact_bytes = serialize::serialize_compact_batch(&proof);
            let proof_compact_rt = serialize::deserialize_compact_batch(&compact_bytes)
                .expect("compact deserialization failed");
            let (passed_compact, failures_compact) = verify_batch(
                &key, &proof_compact_rt, &challenges,
            );
            prop_assert!(
                passed_compact,
                "compact roundtrip should pass: {:?}",
                failures_compact
            );

            // 3. Full format + zstd compression roundtrip
            let compressed_full = serialize::compress(&full_bytes);
            let decompressed_full = serialize::decompress(&compressed_full)
                .expect("full+zstd decompress failed");
            let proof_full_zstd = serialize::deserialize_batch(&decompressed_full)
                .expect("full+zstd deserialize failed");
            let (passed_fz, failures_fz) = verify_batch(&key, &proof_full_zstd, &challenges);
            prop_assert!(passed_fz, "full+zstd roundtrip should pass: {:?}", failures_fz);

            // 4. Compact format + zstd compression roundtrip
            let compressed_compact = serialize::compress(&compact_bytes);
            let decompressed_compact = serialize::decompress(&compressed_compact)
                .expect("compact+zstd decompress failed");
            let proof_compact_zstd = serialize::deserialize_compact_batch(&decompressed_compact)
                .expect("compact+zstd deserialize failed");
            let (passed_cz, failures_cz) = verify_batch(
                &key, &proof_compact_zstd, &challenges,
            );
            prop_assert!(
                passed_cz,
                "compact+zstd roundtrip should pass: {:?}",
                failures_cz
            );
        }
    }
}

// =====================================================================
// 8. Compact format reconstruction correctness
// =====================================================================

mod compact_reconstruction {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        /// For any honest proof, CompactBatchProof::from_full(proof).to_full()
        /// produces traces where x_ffn and h match the originals.
        #[test]
        fn compact_roundtrip_preserves_derived_fields(
            n_tokens in 2usize..=6,
            model_seed in 0u64..50,
        ) {
            let (_cfg, _model, key, _commitment, state) = honest_v3_setup(n_tokens, model_seed);

            let challenges: Vec<u32> = (0..n_tokens as u32).collect();
            let proof = open(&state, &challenges);

            let compact = CompactBatchProof::from_full(&proof);
            let reconstructed = compact.to_full()
                .expect("compact to_full should succeed for honest proof");

            // Check that every trace's x_ffn and h match
            for (i, (orig_trace, recon_trace)) in
                proof.traces.iter().zip(reconstructed.traces.iter()).enumerate()
            {
                for (j, (orig_layer, recon_layer)) in
                    orig_trace.layers.iter().zip(recon_trace.layers.iter()).enumerate()
                {
                    prop_assert_eq!(
                        &orig_layer.x_ffn, &recon_layer.x_ffn,
                        "trace {} layer {}: x_ffn mismatch after compact roundtrip",
                        i, j
                    );
                    prop_assert_eq!(
                        &orig_layer.h, &recon_layer.h,
                        "trace {} layer {}: h mismatch after compact roundtrip",
                        i, j
                    );
                }
            }

            // The reconstructed proof must also pass verification
            let (passed, failures) = verify_batch(&key, &reconstructed, &challenges);
            prop_assert!(
                passed,
                "compact-reconstructed proof should pass verification: {:?}",
                failures
            );
        }
    }
}
