//! Tests for Task 19: verified quantization chain (weight hash).
//!
//! Verifies that the weight-chain hash:
//! 1. Is deterministic (same weights → same hash).
//! 2. Detects weight tampering (any bit flip → different hash).
//! 3. Is bound to source_dtype (same weights, different dtype label → different hash).
//! 4. Round-trips through serialization.

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::merkle::hash_weights;
use verilm_test_vectors::{generate_key, generate_model, LayerWeights};

fn toy_cfg() -> ModelConfig {
    ModelConfig::toy()
}

#[test]
fn test_weight_hash_deterministic() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key1 = generate_key(&cfg, &model, [1u8; 32]);
    let key2 = generate_key(&cfg, &model, [1u8; 32]);

    assert!(key1.weight_hash.is_some());
    assert_eq!(key1.weight_hash, key2.weight_hash);
}

#[test]
fn test_weight_hash_different_seed_same_hash() {
    // Weight hash depends on weights, not on the verifier key seed.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key1 = generate_key(&cfg, &model, [1u8; 32]);
    let key2 = generate_key(&cfg, &model, [99u8; 32]);

    assert_eq!(key1.weight_hash, key2.weight_hash);
}

#[test]
fn test_weight_hash_detects_tampered_weight() {
    let cfg = toy_cfg();
    let mut model = generate_model(&cfg, 42);
    let key_original = generate_key(&cfg, &model, [1u8; 32]);

    // Flip one bit in one weight
    model[0].wq[0] = model[0].wq[0].wrapping_add(1);
    let key_tampered = generate_key(&cfg, &model, [1u8; 32]);

    assert_ne!(key_original.weight_hash, key_tampered.weight_hash);
}

#[test]
fn test_weight_hash_detects_different_model() {
    let cfg = toy_cfg();
    let model_a = generate_model(&cfg, 42);
    let model_b = generate_model(&cfg, 99);
    let key_a = generate_key(&cfg, &model_a, [1u8; 32]);
    let key_b = generate_key(&cfg, &model_b, [1u8; 32]);

    assert_ne!(key_a.weight_hash, key_b.weight_hash);
}

#[test]
fn test_weight_hash_bound_to_source_dtype() {
    // The hash_weights function takes source_dtype as input.
    // Same weights with different dtype label must produce different hash.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);

    let hash_i8 = hash_weights(
        "I8",
        cfg.n_layers,
        &[],
        |layer, mt_idx| get_weights(&model, layer, mt_idx),
        MatrixType::PER_LAYER.len(),
    );

    let hash_bf16 = hash_weights(
        "BF16",
        cfg.n_layers,
        &[],
        |layer, mt_idx| get_weights(&model, layer, mt_idx),
        MatrixType::PER_LAYER.len(),
    );

    assert_ne!(hash_i8, hash_bf16);
}

#[test]
fn test_weight_hash_bound_to_quantization_scales() {
    // Same weights but different scales → different hash.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);

    let scales_a: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![1.0f32; MatrixType::PER_LAYER.len()])
        .collect();
    let scales_b: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|_| vec![2.0f32; MatrixType::PER_LAYER.len()])
        .collect();

    let hash_a = hash_weights(
        "BF16",
        cfg.n_layers,
        &scales_a,
        |layer, mt_idx| get_weights(&model, layer, mt_idx),
        MatrixType::PER_LAYER.len(),
    );

    let hash_b = hash_weights(
        "BF16",
        cfg.n_layers,
        &scales_b,
        |layer, mt_idx| get_weights(&model, layer, mt_idx),
        MatrixType::PER_LAYER.len(),
    );

    assert_ne!(hash_a, hash_b);
}

#[test]
fn test_weight_hash_survives_serialization() {
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let data = verilm_core::serialize::serialize_key(&key);
    let key2 = verilm_core::serialize::deserialize_key(&data).unwrap();

    assert_eq!(key.weight_hash, key2.weight_hash);
}

#[test]
fn test_weight_hash_matches_direct_computation() {
    // The hash stored in the key must match a fresh computation from the same weights.
    let cfg = toy_cfg();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let direct_hash = hash_weights(
        "I8",
        cfg.n_layers,
        &[],
        |layer, mt_idx| get_weights(&model, layer, mt_idx),
        MatrixType::PER_LAYER.len(),
    );

    assert_eq!(key.weight_hash.unwrap(), direct_hash);
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
