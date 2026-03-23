//! Differential tests: compute the same thing two different ways and compare.
//!
//! Validates consistency across code paths in the verified-inference pipeline:
//! forward pass determinism, commit determinism, verify_batch vs verify_one,
//! matmul correctness, requantize semantics, Merkle reconstruction, IO hash
//! determinism, compact-vs-full proof equivalence, key generation determinism,
//! and cross-token chain consistency.

use proptest::prelude::*;
use verilm_core::constants::ModelConfig;
use verilm_core::merkle;
use verilm_core::types::CompactBatchProof;
use verilm_test_vectors::*;

/// Shared tiny config + model + key for deterministic tests.
fn setup() -> (ModelConfig, Vec<LayerWeights>, verilm_core::types::VerifierKey) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 0xDEAD);
    let key = generate_key(&cfg, &model, [3u8; 32]);
    (cfg, model, key)
}

/// Shared input vector for the toy config.
fn toy_input(cfg: &ModelConfig) -> Vec<i8> {
    (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect()
}

// =========================================================================
// 1. forward_pass_determinism
// =========================================================================

#[test]
fn forward_pass_determinism() {
    let (cfg, model, _) = setup();
    let input = toy_input(&cfg);

    let run1 = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let run2 = forward_pass_autoregressive(&cfg, &model, &input, 4);

    assert_eq!(run1.len(), run2.len());
    for (t, (layers1, layers2)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(layers1.len(), layers2.len(), "token {}: layer count mismatch", t);
        for (l, (lt1, lt2)) in layers1.iter().zip(layers2.iter()).enumerate() {
            assert_eq!(lt1.x_attn, lt2.x_attn, "token {} layer {}: x_attn", t, l);
            assert_eq!(lt1.q, lt2.q, "token {} layer {}: q", t, l);
            assert_eq!(lt1.k, lt2.k, "token {} layer {}: k", t, l);
            assert_eq!(lt1.v, lt2.v, "token {} layer {}: v", t, l);
            assert_eq!(lt1.a, lt2.a, "token {} layer {}: a", t, l);
            assert_eq!(lt1.attn_out, lt2.attn_out, "token {} layer {}: attn_out", t, l);
            assert_eq!(lt1.x_ffn, lt2.x_ffn, "token {} layer {}: x_ffn", t, l);
            assert_eq!(lt1.g, lt2.g, "token {} layer {}: g", t, l);
            assert_eq!(lt1.u, lt2.u, "token {} layer {}: u", t, l);
            assert_eq!(lt1.h, lt2.h, "token {} layer {}: h", t, l);
            assert_eq!(lt1.ffn_out, lt2.ffn_out, "token {} layer {}: ffn_out", t, l);
        }
    }
}

// =========================================================================
// 2. commit_determinism
// =========================================================================

#[test]
fn commit_determinism() {
    let (cfg, model, _) = setup();
    let input = toy_input(&cfg);
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 6);

    // Clone the layers so we commit identical data twice.
    let copy1 = all_layers.clone();
    let copy2 = all_layers;

    let (c1, _) = commit(copy1);
    let (c2, _) = commit(copy2);

    assert_eq!(c1.merkle_root, c2.merkle_root, "merkle_root mismatch");
    assert_eq!(c1.io_root, c2.io_root, "io_root mismatch");
    assert_eq!(c1.n_tokens, c2.n_tokens, "n_tokens mismatch");
    assert_eq!(c1.manifest_hash, c2.manifest_hash, "manifest_hash mismatch");
}

// =========================================================================
// 3. verify_batch_vs_verify_one
// =========================================================================

#[test]
fn verify_batch_vs_verify_one() {
    let (cfg, model, key) = setup();
    let input = toy_input(&cfg);
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);

    let n = all_layers.len() as u32;
    let (_, state) = commit(all_layers);
    let all_indices: Vec<u32> = (0..n).collect();
    let proof = open(&state, &all_indices);

    // verify_batch (from verilm_test_vectors)
    let (batch_passed, batch_failures) = verify_batch(&key, &proof, &all_indices);

    // verify_one (from verilm_verify) called per trace
    let mut individual_failures: Vec<String> = Vec::new();
    for trace in &proof.traces {
        let one_failures = verilm_verify::verify_one(&key, trace);
        for f in one_failures {
            individual_failures.push(format!("token {}: {}", trace.token_index, f));
        }
    }

    // Both must agree: either both pass or both fail with the same issues.
    let individual_passed = individual_failures.is_empty();

    assert_eq!(
        batch_passed, individual_passed,
        "batch passed={} but individual passed={}\nbatch failures: {:?}\nindividual failures: {:?}",
        batch_passed, individual_passed, batch_failures, individual_failures
    );

    // If both pass, nothing else to check. If both fail, the per-token Freivalds/SiLU/chain
    // failures should appear in both lists.
    if !batch_passed {
        for f in &individual_failures {
            assert!(
                batch_failures.iter().any(|bf| bf.contains(f.as_str()) || f.contains(bf.as_str())),
                "individual failure {:?} not reflected in batch failures {:?}",
                f,
                batch_failures
            );
        }
    }
}

// =========================================================================
// 4. matmul_i32_vs_naive (proptest)
// =========================================================================

/// Naive O(n^3) matmul for differential comparison.
fn naive_matmul(w: &[i8], x: &[i8], rows: usize, cols: usize) -> Vec<i32> {
    let mut out = vec![0i32; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r] += w[r * cols + c] as i32 * x[c] as i32;
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn matmul_i32_vs_naive(
        rows in 1usize..=16,
        cols in 1usize..=16,
        seed in any::<u64>(),
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let w: Vec<i8> = (0..rows * cols).map(|_| rng.gen::<i8>()).collect();
        let x: Vec<i8> = (0..cols).map(|_| rng.gen::<i8>()).collect();

        let result = matmul_i32(&w, &x, rows, cols);
        let expected = naive_matmul(&w, &x, rows, cols);

        prop_assert_eq!(result, expected);
    }
}

// =========================================================================
// 5. requantize_clamp_consistency (proptest)
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn requantize_clamp_consistency(values in prop::collection::vec(any::<i32>(), 1..64)) {
        let result = requantize(&values);
        for (i, (&v, &r)) in values.iter().zip(result.iter()).enumerate() {
            let expected = v.clamp(-128, 127) as i8;
            prop_assert_eq!(r, expected, "index {}: requantize({}) = {} but expected {}", i, v, r, expected);
        }
    }
}

// =========================================================================
// 6. merkle_tree_reconstruction
// =========================================================================

#[test]
fn merkle_tree_reconstruction() {
    // Build a tree from 7 leaves, verify every proof, rebuild, compare roots.
    let leaves: Vec<[u8; 32]> = (0..7u32)
        .map(|i| merkle::hash_leaf(&i.to_le_bytes()))
        .collect();

    let tree1 = merkle::build_tree(&leaves);

    // Verify each leaf's proof against the root.
    for (i, leaf) in leaves.iter().enumerate() {
        let proof = merkle::prove(&tree1, i);
        assert!(
            merkle::verify(&tree1.root, leaf, &proof),
            "leaf {} proof failed on tree1",
            i
        );
    }

    // Rebuild from scratch and compare roots.
    let tree2 = merkle::build_tree(&leaves);
    assert_eq!(tree1.root, tree2.root, "reconstructed root differs");

    // Cross-verify: proofs from tree1 validate against tree2's root.
    for (i, leaf) in leaves.iter().enumerate() {
        let proof = merkle::prove(&tree1, i);
        assert!(
            merkle::verify(&tree2.root, leaf, &proof),
            "cross-tree verification failed for leaf {}",
            i
        );
    }
}

// =========================================================================
// 7. io_hash_determinism
// =========================================================================

#[test]
fn io_hash_determinism() {
    let input: Vec<i8> = (0..16).map(|i| i as i8).collect();
    let output: Vec<i32> = (0..16).map(|i| i * 100).collect();

    // Legacy binding
    let h1 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::Legacy);
    let h2 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::Legacy);
    assert_eq!(h1, h2, "legacy io_hash not deterministic");

    // V2 token-id binding
    let h3 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::TokenId(42));
    let h4 = merkle::io_hash(&input, &output, &merkle::IoHashBinding::TokenId(42));
    assert_eq!(h3, h4, "v2 io_hash not deterministic");

    // V3 chained binding
    let prev = [0xABu8; 32];
    let h5 = merkle::io_hash(
        &input,
        &output,
        &merkle::IoHashBinding::Chained {
            token_id: 7,
            prev_io_hash: prev,
        },
    );
    let h6 = merkle::io_hash(
        &input,
        &output,
        &merkle::IoHashBinding::Chained {
            token_id: 7,
            prev_io_hash: prev,
        },
    );
    assert_eq!(h5, h6, "v3 chained io_hash not deterministic");

    // Different bindings must produce different hashes.
    assert_ne!(h1, h3, "legacy vs v2 should differ");
    assert_ne!(h3, h5, "v2 vs v3 should differ");
    assert_ne!(h1, h5, "legacy vs v3 should differ");
}

// =========================================================================
// 8. compact_vs_full_verification
// =========================================================================

#[test]
fn compact_vs_full_verification() {
    let (cfg, model, key) = setup();
    let input = toy_input(&cfg);
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 6);

    let n = all_layers.len() as u32;
    let (_, state) = commit(all_layers);
    let all_indices: Vec<u32> = (0..n).collect();
    let full_proof = open(&state, &all_indices);

    // Convert to compact and back.
    let compact = CompactBatchProof::from_full(&full_proof);
    let reconstructed = compact
        .to_full()
        .expect("compact-to-full reconstruction failed");

    // Verify both give the same result.
    let (full_passed, full_failures) = verify_batch(&key, &full_proof, &all_indices);
    let (recon_passed, recon_failures) = verify_batch(&key, &reconstructed, &all_indices);

    assert_eq!(
        full_passed, recon_passed,
        "full passed={} but reconstructed passed={}\nfull failures: {:?}\nrecon failures: {:?}",
        full_passed, recon_passed, full_failures, recon_failures
    );

    // Commitments must be identical.
    assert_eq!(
        full_proof.commitment.merkle_root,
        reconstructed.commitment.merkle_root,
        "merkle_root mismatch after compact roundtrip"
    );
    assert_eq!(
        full_proof.commitment.io_root,
        reconstructed.commitment.io_root,
        "io_root mismatch after compact roundtrip"
    );
}

// =========================================================================
// 9. key_from_same_model_is_identical
// =========================================================================

#[test]
fn key_from_same_model_is_identical() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 0xCAFE);
    let seed = [55u8; 32];

    let key1 = generate_key(&cfg, &model, seed);
    let key2 = generate_key(&cfg, &model, seed);

    assert_eq!(key1.seed, key2.seed);
    assert_eq!(key1.r_vectors, key2.r_vectors, "r_vectors differ");
    assert_eq!(key1.v_vectors, key2.v_vectors, "v_vectors differ");
    assert_eq!(key1.weight_hash, key2.weight_hash, "weight_hash differs");
    assert_eq!(key1.config.hidden_dim, key2.config.hidden_dim);
    assert_eq!(key1.config.n_layers, key2.config.n_layers);
}

// =========================================================================
// 10. cross_token_chain_matches_requantize
// =========================================================================

#[test]
fn cross_token_chain_matches_requantize() {
    let (cfg, model, _) = setup();
    let input = toy_input(&cfg);
    let traces = forward_pass_autoregressive(&cfg, &model, &input, 5);

    for t in 0..traces.len() - 1 {
        let prev_last_ffn_out = &traces[t].last().unwrap().ffn_out;
        let expected_next_input = requantize(prev_last_ffn_out);
        let actual_next_input = &traces[t + 1][0].x_attn;

        assert_eq!(
            *actual_next_input, expected_next_input,
            "token {}->{}:  x_attn[0] != requantize(prev.ffn_out)",
            t,
            t + 1
        );
    }
}
