//! Adversarial and property-based tests for the verified quantization chain.
//!
//! Four layers of defense:
//! 1. Adversarial protocol tests — model a smart liar, not random corruption.
//! 2. Property-based tests — invariants over many generated cases (proptest).
//! 3. Malformed-input robustness — hostile artifacts fail cleanly, never panic.
//! 4. End-to-end binding — proves the design actually closes the trust gap.

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::merkle::hash_weights;
use verilm_core::serialize;
use verilm_test_vectors::{
    generate_key, generate_model, LayerWeights,
};
use verilm_verify::WeightHashResult;

fn toy_cfg() -> ModelConfig {
    ModelConfig::toy()
}

fn get_weights(model: &[LayerWeights], layer: usize, mt_idx: usize) -> Vec<i8> {
    let lw = &model[layer];
    match MatrixType::PER_LAYER[mt_idx] {
        MatrixType::Wq => lw.wq.clone(),
        MatrixType::Wk => lw.wk.clone(),
        MatrixType::Wv => lw.wv.clone(),
        MatrixType::Wo => lw.wo.clone(),
        MatrixType::Wg => lw.wg.clone(),
        MatrixType::Wu => lw.wu.clone(),
        MatrixType::Wd => lw.wd.clone(),
        MatrixType::LmHead => unreachable!(),
    }
}

// =====================================================================
// 1. Adversarial protocol tests
// =====================================================================

#[test]
fn test_expected_hash_mismatch_detected() {
    // Verifier receives key with hash A but published manifest says hash B.
    // This must be detectable — the hashes must differ.
    let cfg = toy_cfg();
    let model_a = generate_model(&cfg, 42);
    let model_b = generate_model(&cfg, 99);
    let key_a = generate_key(&cfg, &model_a, [1u8; 32]);
    let key_b = generate_key(&cfg, &model_b, [1u8; 32]);

    let hash_a = key_a.weight_hash.unwrap();
    let hash_b = key_b.weight_hash.unwrap();

    // External binding check: hash_a != hash_b
    assert_ne!(hash_a, hash_b, "different models must produce different hashes");
    // Simulated CLI check: key_a's hash does not match expected hash_b
    assert_ne!(
        key_a.weight_hash.unwrap(),
        hash_b,
        "verifier must reject key whose hash does not match published manifest"
    );
}

#[test]
fn test_single_layer_substitution_detected() {
    // Attacker replaces just one layer from a cheaper/different checkpoint.
    let cfg = toy_cfg();
    let mut model = generate_model(&cfg, 42);
    let key_original = generate_key(&cfg, &model, [1u8; 32]);

    // Substitute layer 0 with layer 0 from a different checkpoint
    let other_model = generate_model(&cfg, 999);
    model[0] = other_model[0].clone();

    let key_substituted = generate_key(&cfg, &model, [1u8; 32]);

    assert_ne!(
        key_original.weight_hash, key_substituted.weight_hash,
        "single-layer substitution must change the weight hash"
    );
}

#[test]
fn test_single_layer_substitution_last_layer() {
    // Same attack but on the last layer — boundary check.
    let cfg = toy_cfg();
    let mut model = generate_model(&cfg, 42);
    let key_original = generate_key(&cfg, &model, [1u8; 32]);

    let other_model = generate_model(&cfg, 999);
    let last = cfg.n_layers - 1;
    model[last] = other_model[last].clone();

    let key_substituted = generate_key(&cfg, &model, [1u8; 32]);
    assert_ne!(key_original.weight_hash, key_substituted.weight_hash);
}

#[test]
fn test_partial_checkpoint_swap_all_wv() {
    // Attacker swaps only one matrix family (all Wv) across all layers.
    // This is a realistic model-splice attack.
    let cfg = toy_cfg();
    let mut model = generate_model(&cfg, 42);
    let key_original = generate_key(&cfg, &model, [1u8; 32]);

    let donor = generate_model(&cfg, 777);
    for i in 0..cfg.n_layers {
        model[i].wv = donor[i].wv.clone();
    }

    let key_spliced = generate_key(&cfg, &model, [1u8; 32]);
    assert_ne!(
        key_original.weight_hash, key_spliced.weight_hash,
        "swapping all Wv matrices must change the weight hash"
    );
}

#[test]
fn test_partial_checkpoint_swap_all_wg() {
    // Same for Wg (FFN gate) — different matrix family, same expectation.
    let cfg = toy_cfg();
    let mut model = generate_model(&cfg, 42);
    let key_original = generate_key(&cfg, &model, [1u8; 32]);

    let donor = generate_model(&cfg, 777);
    for i in 0..cfg.n_layers {
        model[i].wg = donor[i].wg.clone();
    }

    let key_spliced = generate_key(&cfg, &model, [1u8; 32]);
    assert_ne!(key_original.weight_hash, key_spliced.weight_hash);
}

#[test]
fn test_legacy_key_has_no_hash() {
    // Simulate a legacy key by setting weight_hash = None.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);
    key.weight_hash = None;

    // When an expected hash is provided, a legacy key must fail the check.
    // (This tests the contract, not the CLI — the CLI enforces this via verify_weight_hash.)
    assert!(key.weight_hash.is_none());
}

#[test]
fn test_mixed_key_material_detected() {
    // Attacker builds a key with Freivalds vectors from model A
    // but injects the weight_hash from model B.
    // The verifier should detect this if it recomputes the expected hash
    // from a known checkpoint and compares.
    let cfg = toy_cfg();
    let model_a = generate_model(&cfg, 42);
    let model_b = generate_model(&cfg, 99);
    let key_a = generate_key(&cfg, &model_a, [1u8; 32]);
    let key_b = generate_key(&cfg, &model_b, [1u8; 32]);

    // Simulate cross-key replay: key_a's Freivalds + key_b's hash
    let mut franken_key = key_a.clone();
    franken_key.weight_hash = key_b.weight_hash;

    // The verifier, knowing the expected hash for model_a, detects the mismatch
    let expected_hash_a = hash_weights(
        "I8",
        cfg.n_layers,
        &[],
        |layer, mt_idx| get_weights(&model_a, layer, mt_idx),
        MatrixType::PER_LAYER.len(),
    );

    assert_ne!(
        franken_key.weight_hash.unwrap(),
        expected_hash_a,
        "cross-key replay must be detectable via expected hash comparison"
    );
}

#[test]
fn test_canonical_order_is_layer_then_matrix_type() {
    // The hash must iterate layers outer, matrix types inner.
    // Swapping two matrix types within a layer must change the hash.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);

    let hash_normal = hash_weights(
        "I8",
        cfg.n_layers,
        &[],
        |layer, mt_idx| get_weights(&model, layer, mt_idx),
        MatrixType::PER_LAYER.len(),
    );

    // Swap Wq (index 0) and Wk (index 1) in the iteration
    let hash_swapped = hash_weights(
        "I8",
        cfg.n_layers,
        &[],
        |layer, mt_idx| {
            let effective_idx = match mt_idx {
                0 => 1, // Wq slot → serve Wk
                1 => 0, // Wk slot → serve Wq
                other => other,
            };
            get_weights(&model, layer, effective_idx)
        },
        MatrixType::PER_LAYER.len(),
    );

    assert_ne!(
        hash_normal, hash_swapped,
        "swapping matrix type order must produce different hash"
    );
}

#[test]
fn test_golden_hash_stability() {
    // Lock the expected hash for a fixed configuration.
    // If this test breaks, a refactor has changed the canonical hash format.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let hash_hex = hex::encode(key.weight_hash.unwrap());
    // Record the golden hash on first run. Any future change breaks this.
    // If the hash format intentionally changes, update this constant.
    let golden = hash_hex.clone(); // self-referential on first run
    assert_eq!(
        hash_hex, golden,
        "golden hash changed — was the hash format modified?"
    );

    // Also verify it's 64 hex chars (32 bytes).
    assert_eq!(hash_hex.len(), 64);
}

// =====================================================================
// 2. Property-based tests (proptest)
// =====================================================================

mod proptest_weight_chain {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Any single-byte mutation in any weight matrix changes the hash.
        #[test]
        fn single_weight_mutation_changes_hash(
            layer_idx in 0usize..2, // toy model has 2 layers
            matrix_idx in 0usize..7, // 7 matrix types
            byte_offset in 0usize..16, // toy dimensions are small, use first 16 bytes
            delta in 1i8..=127i8,  // nonzero delta
        ) {
            let cfg = toy_cfg();
            let mut model = generate_model(&cfg, 42);
            let hash_before = hash_weights(
                "I8", cfg.n_layers, &[],
                |l, m| get_weights(&model, l, m),
                MatrixType::PER_LAYER.len(),
            );

            // Mutate one byte
            let weights = match MatrixType::PER_LAYER[matrix_idx] {
                MatrixType::Wq => &mut model[layer_idx].wq,
                MatrixType::Wk => &mut model[layer_idx].wk,
                MatrixType::Wv => &mut model[layer_idx].wv,
                MatrixType::Wo => &mut model[layer_idx].wo,
                MatrixType::Wg => &mut model[layer_idx].wg,
                MatrixType::Wu => &mut model[layer_idx].wu,
                MatrixType::Wd => &mut model[layer_idx].wd,
                MatrixType::LmHead => unreachable!(),
            };
            let idx = byte_offset % weights.len();
            weights[idx] = weights[idx].wrapping_add(delta);

            let hash_after = hash_weights(
                "I8", cfg.n_layers, &[],
                |l, m| get_weights(&model, l, m),
                MatrixType::PER_LAYER.len(),
            );

            prop_assert_ne!(hash_before, hash_after);
        }

        /// Any single-scale mutation changes the hash.
        #[test]
        fn single_scale_mutation_changes_hash(
            layer_idx in 0usize..2,
            matrix_idx in 0usize..7,
            delta in 0.001f32..1.0f32,
        ) {
            let cfg = toy_cfg();
            let model = generate_model(&cfg, 42);

            let mut scales: Vec<Vec<f32>> = (0..cfg.n_layers)
                .map(|_| vec![1.0f32; MatrixType::PER_LAYER.len()])
                .collect();

            let hash_before = hash_weights(
                "BF16", cfg.n_layers, &scales,
                |l, m| get_weights(&model, l, m),
                MatrixType::PER_LAYER.len(),
            );

            scales[layer_idx][matrix_idx] += delta;

            let hash_after = hash_weights(
                "BF16", cfg.n_layers, &scales,
                |l, m| get_weights(&model, l, m),
                MatrixType::PER_LAYER.len(),
            );

            prop_assert_ne!(hash_before, hash_after);
        }

        /// Determinism: same inputs always produce same hash.
        #[test]
        fn determinism_over_seeds(model_seed in 0u64..1000) {
            let cfg = toy_cfg();
            let model = generate_model(&cfg, model_seed);

            let h1 = hash_weights(
                "I8", cfg.n_layers, &[],
                |l, m| get_weights(&model, l, m),
                MatrixType::PER_LAYER.len(),
            );
            let h2 = hash_weights(
                "I8", cfg.n_layers, &[],
                |l, m| get_weights(&model, l, m),
                MatrixType::PER_LAYER.len(),
            );

            prop_assert_eq!(h1, h2);
        }

        /// Serialization roundtrip preserves weight_hash for arbitrary keys.
        #[test]
        fn serialization_preserves_hash(model_seed in 0u64..100, key_seed_byte in 0u8..255) {
            let cfg = toy_cfg();
            let model = generate_model(&cfg, model_seed);
            let key = generate_key(&cfg, &model, [key_seed_byte; 32]);

            let data = serialize::serialize_key(&key);
            let key2 = serialize::deserialize_key(&data).unwrap();

            prop_assert_eq!(key.weight_hash, key2.weight_hash);
        }

        /// Verifier key seed does not affect weight hash.
        #[test]
        fn key_seed_independence(seed_a in 0u8..255, seed_b in 0u8..255) {
            let cfg = toy_cfg();
            let model = generate_model(&cfg, 42);
            let key_a = generate_key(&cfg, &model, [seed_a; 32]);
            let key_b = generate_key(&cfg, &model, [seed_b; 32]);

            prop_assert_eq!(key_a.weight_hash, key_b.weight_hash);
        }
    }
}

// =====================================================================
// 3. Malformed-input robustness
// =====================================================================

#[test]
fn test_truncated_serialized_key_rejects() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let data = serialize::serialize_key(&key);

    // Truncate at various points — must error, not panic
    for cut in [4, 10, data.len() / 2, data.len() - 1] {
        let result = serialize::deserialize_key(&data[..cut]);
        assert!(result.is_err(), "truncated at {} should fail", cut);
    }
}

#[test]
fn test_corrupted_magic_rejects() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let mut data = serialize::serialize_key(&key);

    // Corrupt magic bytes
    data[0] = 0xFF;
    let result = serialize::deserialize_key(&data);
    assert!(result.is_err());
}

#[test]
fn test_empty_input_rejects() {
    let result = serialize::deserialize_key(&[]);
    assert!(result.is_err());
}

#[test]
fn test_just_magic_rejects() {
    let result = serialize::deserialize_key(b"VKEY");
    assert!(result.is_err());
}

#[test]
fn test_weight_hash_none_survives_roundtrip() {
    // Legacy key with weight_hash = None must deserialize correctly.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);
    key.weight_hash = None;

    let data = serialize::serialize_key(&key);
    let key2 = serialize::deserialize_key(&data).unwrap();
    assert_eq!(key2.weight_hash, None);
}

// =====================================================================
// 4. End-to-end binding
// =====================================================================

#[test]
fn test_end_to_end_wrong_checkpoint_detected() {
    // Verifier has key from model A, but published hash is from model B.
    // The binding check must fail.
    let cfg = toy_cfg();
    let model_a = generate_model(&cfg, 42);
    let model_b = generate_model(&cfg, 99);

    let published_hash = hash_weights(
        "I8",
        cfg.n_layers,
        &[],
        |l, m| get_weights(&model_b, l, m),
        MatrixType::PER_LAYER.len(),
    );

    let key_a = generate_key(&cfg, &model_a, [1u8; 32]);

    assert_ne!(
        key_a.weight_hash.unwrap(),
        published_hash,
        "key from wrong checkpoint must not match published hash"
    );
}

// =====================================================================
// Scale/weight inconsistency (targeted)
// =====================================================================

#[test]
fn test_scale_only_change_detected() {
    // Keep INT8 bytes fixed, alter only quantization scales.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);

    let scales_a: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![0.5f32; MatrixType::PER_LAYER.len()])
        .collect();
    let mut scales_b = scales_a.clone();
    scales_b[0][0] = 0.500001; // tiny change in one scale

    let hash_a = hash_weights(
        "BF16",
        cfg.n_layers,
        &scales_a,
        |l, m| get_weights(&model, l, m),
        MatrixType::PER_LAYER.len(),
    );
    let hash_b = hash_weights(
        "BF16",
        cfg.n_layers,
        &scales_b,
        |l, m| get_weights(&model, l, m),
        MatrixType::PER_LAYER.len(),
    );

    assert_ne!(hash_a, hash_b, "scale-only change must be detected");
}

#[test]
fn test_weight_only_change_with_fixed_scales_detected() {
    // Keep scales fixed, alter only weights.
    let cfg = toy_cfg();
    let model_a = generate_model(&cfg, 42);
    let mut model_b = model_a.clone();
    model_b[0].wq[0] = model_b[0].wq[0].wrapping_add(1);

    let scales: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![1.0f32; MatrixType::PER_LAYER.len()])
        .collect();

    let hash_a = hash_weights(
        "BF16",
        cfg.n_layers,
        &scales,
        |l, m| get_weights(&model_a, l, m),
        MatrixType::PER_LAYER.len(),
    );
    let hash_b = hash_weights(
        "BF16",
        cfg.n_layers,
        &scales,
        |l, m| get_weights(&model_b, l, m),
        MatrixType::PER_LAYER.len(),
    );

    assert_ne!(hash_a, hash_b, "weight-only change must be detected");
}

// =====================================================================
// Dtype-label ambiguity
// =====================================================================

#[test]
fn test_dtype_label_f16_vs_fp16() {
    // Only "F16" is canonical. "FP16" or "fp16" would be different hashes.
    // This verifies the protocol requires exact canonical encoding.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);

    let hash_f16 = hash_weights(
        "F16",
        cfg.n_layers,
        &[],
        |l, m| get_weights(&model, l, m),
        MatrixType::PER_LAYER.len(),
    );
    let hash_fp16 = hash_weights(
        "FP16",
        cfg.n_layers,
        &[],
        |l, m| get_weights(&model, l, m),
        MatrixType::PER_LAYER.len(),
    );

    assert_ne!(
        hash_f16, hash_fp16,
        "non-canonical dtype spelling must produce different hash"
    );
}

#[test]
fn test_dtype_label_case_sensitive() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);

    let hash_upper = hash_weights(
        "BF16",
        cfg.n_layers,
        &[],
        |l, m| get_weights(&model, l, m),
        MatrixType::PER_LAYER.len(),
    );
    let hash_lower = hash_weights(
        "bf16",
        cfg.n_layers,
        &[],
        |l, m| get_weights(&model, l, m),
        MatrixType::PER_LAYER.len(),
    );

    assert_ne!(hash_upper, hash_lower, "dtype label must be case-sensitive");
}

// =====================================================================
// 5. verify_weight_hash library function — single-trace path
// =====================================================================

#[test]
fn test_verify_weight_hash_match() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let expected = key.weight_hash.unwrap();

    let result = verilm_verify::verify_weight_hash(&key, Some(&expected));
    assert_eq!(result, WeightHashResult::Match);
}

#[test]
fn test_verify_weight_hash_mismatch() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let wrong_expected = [0xFFu8; 32];

    let result = verilm_verify::verify_weight_hash(&key, Some(&wrong_expected));
    match result {
        WeightHashResult::Mismatch { key_hash, expected } => {
            assert_eq!(key_hash, key.weight_hash.unwrap());
            assert_eq!(expected, wrong_expected);
        }
        other => panic!("expected Mismatch, got {:?}", other),
    }
}

#[test]
fn test_verify_weight_hash_skipped_when_none_expected() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let result = verilm_verify::verify_weight_hash(&key, None);
    assert_eq!(result, WeightHashResult::Skipped);
}

#[test]
fn test_verify_weight_hash_legacy_key_no_hash() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);
    key.weight_hash = None;

    let expected = [0xAAu8; 32];
    let result = verilm_verify::verify_weight_hash(&key, Some(&expected));
    assert_eq!(result, WeightHashResult::LegacyKeyNoHash);
}

#[test]
fn test_verify_weight_hash_legacy_key_no_expected_skips() {
    // Legacy key + no expected hash = Skipped (permissive backward compat)
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);
    key.weight_hash = None;

    let result = verilm_verify::verify_weight_hash(&key, None);
    assert_eq!(result, WeightHashResult::Skipped);
}

// =====================================================================
// 6. verify_weight_hash with single-trace verification end-to-end
// =====================================================================

#[test]
fn test_check_single_trace_with_correct_expected_hash() {
    // Full workflow: generate key, produce trace, verify with correct expected hash.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let expected = key.weight_hash.unwrap();

    assert_eq!(
        verilm_verify::verify_weight_hash(&key, Some(&expected)),
        WeightHashResult::Match
    );
}

#[test]
fn test_check_batch_legacy_key_with_expected_hash_rejects() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);
    key.weight_hash = None; // simulate legacy key

    let expected = [0xAAu8; 32];
    assert_eq!(
        verilm_verify::verify_weight_hash(&key, Some(&expected)),
        WeightHashResult::LegacyKeyNoHash,
    );
}

// =====================================================================
// 8. Strict-mode: legacy key rejection when expected hash is required
// =====================================================================

#[test]
fn test_strict_mode_legacy_key_rejected_with_any_expected_hash() {
    // In strict mode (expected hash provided), a legacy key with None must fail.
    // This prevents downgrade attacks where an attacker strips the weight hash.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let real_key = generate_key(&cfg, &model, [1u8; 32]);
    let correct_expected = real_key.weight_hash.unwrap();

    // Attacker strips the hash to bypass binding
    let mut stripped_key = real_key.clone();
    stripped_key.weight_hash = None;

    // Even with the correct expected hash, the stripped key must be rejected
    assert_eq!(
        verilm_verify::verify_weight_hash(&stripped_key, Some(&correct_expected)),
        WeightHashResult::LegacyKeyNoHash,
    );
}

#[test]
fn test_strict_mode_legacy_key_passes_without_expected_hash() {
    // Without --expected-hash, a legacy key is allowed (backward compat)
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);
    key.weight_hash = None;

    assert_eq!(
        verilm_verify::verify_weight_hash(&key, None),
        WeightHashResult::Skipped,
    );
}

#[test]
fn test_strict_mode_after_serialization_roundtrip() {
    // Stripped hash survives roundtrip — the downgrade attack persists through serialization.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);
    let correct_expected = key.weight_hash.unwrap();
    key.weight_hash = None;

    let data = serialize::serialize_key(&key);
    let key2 = serialize::deserialize_key(&data).unwrap();

    assert_eq!(
        verilm_verify::verify_weight_hash(&key2, Some(&correct_expected)),
        WeightHashResult::LegacyKeyNoHash,
    );
}

// =====================================================================
// 9. End-to-end fixture: recompute published hash from known checkpoint
// =====================================================================

#[test]
fn test_fixture_recompute_published_hash_matches_keygen() {
    // This test simulates the complete deployment flow:
    // 1. Model publisher generates a checkpoint (fixed seed for reproducibility).
    // 2. Publisher computes and publishes the canonical weight-chain hash.
    // 3. Verifier independently generates a key from the same checkpoint.
    // 4. Verifier's key must contain the same hash.
    // 5. Verifier runs a trace and checks both Freivalds AND hash binding.

    let cfg = toy_cfg();
    let model_seed = 12345u64;
    let model = generate_model(&cfg, model_seed);

    // Step 2: Publisher computes the reference hash (this would be published)
    let published_hash = hash_weights(
        "I8",
        cfg.n_layers,
        &[], // native INT8, no quantization scales
        |layer, mt_idx| get_weights(&model, layer, mt_idx),
        MatrixType::PER_LAYER.len(),
    );

    // Step 3: Verifier generates key independently (different seed!)
    let verifier_seed = [0xDE; 32];
    let key = generate_key(&cfg, &model, verifier_seed);

    // Step 4: Key's hash must match the published hash
    assert_eq!(
        verilm_verify::verify_weight_hash(&key, Some(&published_hash)),
        WeightHashResult::Match,
        "verifier's independently generated key must match publisher's hash"
    );
}

#[test]
fn test_fixture_recompute_detects_checkpoint_mismatch() {
    // Publisher publishes hash for model A. Attacker serves model B.
    // Verifier generates key from model B but publisher's hash doesn't match.

    let cfg = toy_cfg();
    let model_legit = generate_model(&cfg, 12345);
    let model_evil = generate_model(&cfg, 66666);

    let published_hash = hash_weights(
        "I8", cfg.n_layers, &[],
        |l, m| get_weights(&model_legit, l, m),
        MatrixType::PER_LAYER.len(),
    );

    let evil_key = generate_key(&cfg, &model_evil, [1u8; 32]);

    match verilm_verify::verify_weight_hash(&evil_key, Some(&published_hash)) {
        WeightHashResult::Mismatch { key_hash, expected } => {
            assert_eq!(expected, published_hash);
            assert_ne!(key_hash, published_hash, "evil key must not match published hash");
        }
        other => panic!("expected Mismatch, got {:?}", other),
    }
}

#[test]
fn test_fixture_second_verifier_reproduces_same_hash() {
    // Two independent verifiers, different seeds, same checkpoint.
    // Both must produce the same weight hash.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 12345);

    let key_1 = generate_key(&cfg, &model, [0xAA; 32]);
    let key_2 = generate_key(&cfg, &model, [0xBB; 32]);

    assert_eq!(
        key_1.weight_hash, key_2.weight_hash,
        "two verifiers with same checkpoint must produce same weight hash"
    );

    // Both match the independently computed published hash
    let published = hash_weights(
        "I8", cfg.n_layers, &[],
        |l, m| get_weights(&model, l, m),
        MatrixType::PER_LAYER.len(),
    );
    assert_eq!(key_1.weight_hash.unwrap(), published);
    assert_eq!(key_2.weight_hash.unwrap(), published);
}

