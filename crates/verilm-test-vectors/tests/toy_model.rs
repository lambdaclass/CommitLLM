/// End-to-end test: generate toy model, run forward pass, create key,
/// build trace, verify. This validates the entire Freivalds pipeline.

use verilm_core::constants::ModelConfig;
use verilm_test_vectors::*;

#[test]
fn test_honest_trace_passes() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let layers = forward_pass(&cfg, &model, &input);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(passed, "honest trace should pass: {:?}", failures);
}

#[test]
fn test_corrupted_output_fails() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let mut layers = forward_pass(&cfg, &model, &input);

    // Corrupt one output: flip a value in layer 0's q output
    layers[0].q[0] = layers[0].q[0].wrapping_add(1);

    let key = generate_key(&cfg, &model, [1u8; 32]);
    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "corrupted trace should fail");
    assert!(
        failures.iter().any(|f| f.contains("Wq")),
        "should detect Wq corruption: {:?}",
        failures
    );
}

#[test]
fn test_wrong_weights_fail() {
    let cfg = ModelConfig::toy();
    let model_real = generate_model(&cfg, 12345);
    let model_fake = generate_model(&cfg, 99999); // different model

    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    // Run inference with fake model
    let layers = forward_pass(&cfg, &model_fake, &input);

    // But verify with key from real model
    let key = generate_key(&cfg, &model_real, [1u8; 32]);
    let trace = build_trace(layers, 0, 10);
    let (passed, _failures) = verify_trace(&key, &trace);

    assert!(!passed, "wrong model should fail verification");
}

#[test]
fn test_merkle_proof_valid() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let layers = forward_pass(&cfg, &model, &input);
    let trace = build_trace(layers.clone(), 3, 10);

    // Verify the Merkle proof
    let data = bincode::serialize(&layers).expect("serialize");
    let leaf = verilm_core::merkle::hash_leaf(&data);

    // Rebuild tree to get root
    let mut leaves = Vec::new();
    for t in 0..10u32 {
        if t == 3 {
            leaves.push(leaf);
        } else {
            leaves.push(verilm_core::merkle::hash_leaf(&t.to_le_bytes()));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);

    assert!(verilm_core::merkle::verify(
        &tree.root,
        &leaf,
        &trace.merkle_proof
    ));
}

#[test]
fn test_key_serialization_roundtrip() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let data = verilm_core::serialize::serialize_key(&key);
    let key2 = verilm_core::serialize::deserialize_key(&data).unwrap();

    assert_eq!(key.version, key2.version);
    assert_eq!(key.r_vectors, key2.r_vectors);
    assert_eq!(key.v_vectors.len(), key2.v_vectors.len());
}
