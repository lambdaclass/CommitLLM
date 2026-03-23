//! Attack scenario tests.
//!
//! Each test simulates a specific adversary strategy and verifies that
//! the protocol detects it. These cover the threat model from the paper.

use verilm_core::constants::ModelConfig;
use verilm_test_vectors::*;

fn setup() -> (ModelConfig, Vec<LayerWeights>, Vec<i8>) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    (cfg, model, input)
}

// -----------------------------------------------------------------------
// 1. Model substitution (cost-cutting adversary)
// -----------------------------------------------------------------------

#[test]
fn test_model_swap_detected_on_every_layer() {
    let (cfg, real_model, input) = setup();
    let fake_model = generate_model(&cfg, 99999);

    let layers = forward_pass(&cfg, &fake_model, &input);
    let key = generate_key(&cfg, &real_model, [1u8; 32]);
    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed);
    // Model swap should fail on EVERY matrix of EVERY layer
    assert_eq!(
        failures.len(),
        7 * cfg.n_layers,
        "model swap should fail all {} checks, got {}: {:?}",
        7 * cfg.n_layers,
        failures.len(),
        failures
    );
}

#[test]
fn test_model_swap_different_inputs() {
    let (cfg, real_model, _) = setup();
    let fake_model = generate_model(&cfg, 77777);
    let key = generate_key(&cfg, &real_model, [1u8; 32]);

    // Try multiple different inputs — swap should be caught on all of them
    for seed in 0..5u8 {
        let input: Vec<i8> = (0..cfg.hidden_dim)
            .map(|i| ((i as u8).wrapping_add(seed * 37)) as i8)
            .collect();
        let layers = forward_pass(&cfg, &fake_model, &input);
        let trace = build_trace(layers, 0, 10);
        let (passed, _) = verify_trace(&key, &trace);
        assert!(!passed, "model swap should fail for input seed {}", seed);
    }
}

// -----------------------------------------------------------------------
// 2. Single matrix corruption (subtle adversary)
// -----------------------------------------------------------------------

#[test]
fn test_corrupt_one_matrix_one_layer_detected() {
    let (cfg, model, input) = setup();
    let layers = forward_pass(&cfg, &model, &input);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Corrupt just the Wq output in layer 1
    let mut bad_layers = layers.clone();
    bad_layers[1].q[0] = bad_layers[1].q[0].wrapping_add(42);

    let trace = build_trace(bad_layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed);
    // Should fail exactly on layer 1 Wq
    assert!(
        failures.iter().any(|f| f.contains("layer 1") && f.contains("Wq")),
        "should detect layer 1 Wq corruption: {:?}",
        failures
    );
}

#[test]
fn test_corrupt_each_matrix_type_individually() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // For each of the 7 matrix types, corrupt its output and verify detection
    let matrix_fields: Vec<(&str, Box<dyn Fn(&mut verilm_core::types::LayerTrace)>)> = vec![
        ("Wq", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.q[0] += 1)),
        ("Wk", Box::new(|lt| lt.k[0] += 1)),
        ("Wv", Box::new(|lt| lt.v[0] += 1)),
        ("Wo", Box::new(|lt| lt.attn_out[0] += 1)),
        ("Wg", Box::new(|lt| lt.g[0] += 1)),
        ("Wu", Box::new(|lt| lt.u[0] += 1)),
        ("Wd", Box::new(|lt| lt.ffn_out[0] += 1)),
    ];

    for (name, corrupt_fn) in &matrix_fields {
        let mut layers = forward_pass(&cfg, &model, &input);
        corrupt_fn(&mut layers[0]);

        let trace = build_trace(layers, 0, 10);
        let (passed, failures) = verify_trace(&key, &trace);

        assert!(
            !passed,
            "{} corruption should be detected",
            name
        );
        assert!(
            failures.iter().any(|f| f.contains(name)),
            "{} should appear in failures: {:?}",
            name,
            failures
        );
    }
}

// -----------------------------------------------------------------------
// 3. Output tampering (manipulative adversary)
// -----------------------------------------------------------------------

#[test]
fn test_tamper_final_layer_output_detected() {
    let (cfg, model, input) = setup();
    let mut layers = forward_pass(&cfg, &model, &input);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Adversary tampers with the final layer's FFN output (changes logits)
    let last = cfg.n_layers - 1;
    for v in layers[last].ffn_out.iter_mut() {
        *v = v.wrapping_add(100);
    }

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed);
    assert!(
        failures.iter().any(|f| f.contains("Wd")),
        "final layer Wd tampering should be detected: {:?}",
        failures
    );
}

#[test]
fn test_tamper_attention_vector_detected() {
    let (cfg, model, input) = setup();
    let mut layers = forward_pass(&cfg, &model, &input);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Adversary changes the attention vector (input to W_o).
    // This should break the W_o Freivalds check because
    // v_o . a_tampered != r_o . attn_out
    layers[0].a[0] = layers[0].a[0].wrapping_add(1);

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed);
    assert!(
        failures.iter().any(|f| f.contains("Wo")),
        "attention vector tampering should break Wo check: {:?}",
        failures
    );
}

#[test]
fn test_tamper_ffn_input_detected() {
    let (cfg, model, input) = setup();
    let mut layers = forward_pass(&cfg, &model, &input);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Adversary changes x_ffn (input to FFN block).
    // Should break Wg and Wu checks.
    layers[0].x_ffn[0] = layers[0].x_ffn[0].wrapping_add(1);

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed);
    // Should break both Wg and Wu since x_ffn feeds both
    assert!(
        failures.iter().any(|f| f.contains("Wg")),
        "x_ffn tampering should break Wg: {:?}",
        failures
    );
    assert!(
        failures.iter().any(|f| f.contains("Wu")),
        "x_ffn tampering should break Wu: {:?}",
        failures
    );
}

#[test]
fn test_tamper_h_vector_detected() {
    let (cfg, model, input) = setup();
    let mut layers = forward_pass(&cfg, &model, &input);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Adversary changes h (SiLU output, input to W_d).
    // Should break the Wd check.
    layers[0].h[0] = layers[0].h[0].wrapping_add(1);

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed);
    assert!(
        failures.iter().any(|f| f.contains("Wd")),
        "h tampering should break Wd check: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 4. Consistency attacks (trying to forge intermediate values)
// -----------------------------------------------------------------------

#[test]
fn test_inconsistent_input_output_chain() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let mut layers = forward_pass(&cfg, &model, &input);

    // Directly tamper with layer 1's x_attn to break the cross-layer chain.
    // The verifier checks: x_attn[1] == requantize(ffn_out[0]).
    // Flipping a value breaks this invariant.
    layers[1].x_attn[0] = layers[1].x_attn[0].wrapping_add(1);

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "tampered chain should fail verification");
    assert!(
        failures.iter().any(|f| f.contains("chain check failed")),
        "should report chain failure: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 5. Bit-flip attacks (minimal corruption)
// -----------------------------------------------------------------------

#[test]
fn test_single_bit_flip_in_output_detected() {
    let (cfg, model, input) = setup();
    let mut layers = forward_pass(&cfg, &model, &input);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Flip just one bit in one i32 accumulator
    layers[0].q[0] ^= 1;

    let trace = build_trace(layers, 0, 10);
    let (passed, _) = verify_trace(&key, &trace);

    assert!(!passed, "single bit flip should be detected");
}

#[test]
fn test_single_bit_flip_in_every_output_field() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // For each i32 output field, flip one bit and verify detection
    let fields: Vec<(&str, Box<dyn Fn(&mut verilm_core::types::LayerTrace)>)> = vec![
        ("q", Box::new(|lt: &mut verilm_core::types::LayerTrace| lt.q[0] ^= 1)),
        ("k", Box::new(|lt| lt.k[0] ^= 1)),
        ("v", Box::new(|lt| lt.v[0] ^= 1)),
        ("attn_out", Box::new(|lt| lt.attn_out[0] ^= 1)),
        ("g", Box::new(|lt| lt.g[0] ^= 1)),
        ("u", Box::new(|lt| lt.u[0] ^= 1)),
        ("ffn_out", Box::new(|lt| lt.ffn_out[0] ^= 1)),
    ];

    for (name, flip_fn) in &fields {
        let mut layers = forward_pass(&cfg, &model, &input);
        flip_fn(&mut layers[0]);

        let trace = build_trace(layers, 0, 10);
        let (passed, _) = verify_trace(&key, &trace);

        assert!(!passed, "bit flip in {} should be detected", name);
    }
}

// -----------------------------------------------------------------------
// 6. Multiple token verification
// -----------------------------------------------------------------------

#[test]
fn test_honest_multiple_inputs_all_pass() {
    let (cfg, model, _) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Verify that honest traces pass for many different inputs
    for seed in 0..20u8 {
        let input: Vec<i8> = (0..cfg.hidden_dim)
            .map(|i| ((i as u8).wrapping_mul(seed.wrapping_add(1))) as i8)
            .collect();
        let layers = forward_pass(&cfg, &model, &input);
        let trace = build_trace(layers, 0, 10);
        let (passed, failures) = verify_trace(&key, &trace);

        assert!(passed, "honest trace {} should pass: {:?}", seed, failures);
    }
}

// -----------------------------------------------------------------------
// 7. Zero/extreme input edge cases
// -----------------------------------------------------------------------

#[test]
fn test_zero_input_honest_passes() {
    let (cfg, model, _) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let input = vec![0i8; cfg.hidden_dim];
    let layers = forward_pass(&cfg, &model, &input);
    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(passed, "zero input honest trace should pass: {:?}", failures);
}

#[test]
fn test_max_input_honest_passes() {
    let (cfg, model, _) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let input = vec![127i8; cfg.hidden_dim];
    let layers = forward_pass(&cfg, &model, &input);
    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(passed, "max input honest trace should pass: {:?}", failures);
}

#[test]
fn test_min_input_honest_passes() {
    let (cfg, model, _) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let input = vec![-128i8; cfg.hidden_dim];
    let layers = forward_pass(&cfg, &model, &input);
    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(passed, "min input honest trace should pass: {:?}", failures);
}

// -----------------------------------------------------------------------
// 8. Independent verifier keys
// -----------------------------------------------------------------------

#[test]
fn test_different_verifier_keys_both_catch_cheating() {
    let (cfg, real_model, input) = setup();
    let fake_model = generate_model(&cfg, 55555);

    let key_alice = generate_key(&cfg, &real_model, [1u8; 32]);
    let key_bob = generate_key(&cfg, &real_model, [2u8; 32]);

    let layers = forward_pass(&cfg, &fake_model, &input);
    let trace = build_trace(layers, 0, 10);

    let (passed_a, _) = verify_trace(&key_alice, &trace);
    let (passed_b, _) = verify_trace(&key_bob, &trace);

    assert!(!passed_a, "Alice should detect model swap");
    assert!(!passed_b, "Bob should detect model swap");
}

#[test]
fn test_key_for_wrong_model_rejects_honest_trace() {
    let (cfg, model_a, input) = setup();
    let model_b = generate_model(&cfg, 77777);

    // Honest trace from model_a, but key from model_b
    let layers = forward_pass(&cfg, &model_a, &input);
    let key = generate_key(&cfg, &model_b, [1u8; 32]);
    let trace = build_trace(layers, 0, 10);
    let (passed, _) = verify_trace(&key, &trace);

    assert!(!passed, "key from wrong model should reject honest trace");
}

// -----------------------------------------------------------------------
// 9. Merkle commitment attacks
// -----------------------------------------------------------------------

#[test]
fn test_tampered_merkle_root_detected() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let layers = forward_pass(&cfg, &model, &input);
    let mut trace = build_trace(layers, 0, 10);

    // Tamper with the committed root
    trace.merkle_root[0] ^= 0xff;

    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "tampered merkle root should be detected");
    assert!(
        failures.iter().any(|f| f.contains("Merkle")),
        "should report Merkle failure: {:?}",
        failures
    );
}

#[test]
fn test_tampered_merkle_proof_detected() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let layers = forward_pass(&cfg, &model, &input);
    let mut trace = build_trace(layers, 0, 10);

    // Tamper with a sibling in the proof
    if !trace.merkle_proof.siblings.is_empty() {
        trace.merkle_proof.siblings[0][0] ^= 0xff;
    }

    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "tampered merkle proof should be detected");
    assert!(
        failures.iter().any(|f| f.contains("Merkle")),
        "should report Merkle failure: {:?}",
        failures
    );
}

#[test]
fn test_layers_changed_after_commitment_detected() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let layers = forward_pass(&cfg, &model, &input);
    let mut trace = build_trace(layers, 0, 10);

    // The root was computed from the original layers.
    // Now change a layer value — the Merkle proof should fail
    // because the leaf hash no longer matches.
    trace.layers[0].q[0] = trace.layers[0].q[0].wrapping_add(1);

    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "post-commitment layer change should be detected");
    // Should fail on both Merkle (leaf changed) and Freivalds (output wrong)
    assert!(
        failures.iter().any(|f| f.contains("Merkle")),
        "should report Merkle failure: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 10. Coordinated multi-attack scenarios
// -----------------------------------------------------------------------

#[test]
fn test_merkle_plus_computation_tampering() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let mut layers = forward_pass(&cfg, &model, &input);

    // Corrupt layer 0's ffn_out (computation tampering)
    layers[0].ffn_out[0] = layers[0].ffn_out[0].wrapping_add(42);

    let mut trace = build_trace(layers, 0, 10);

    // Also tamper with the Merkle root (commitment tampering)
    trace.merkle_root[0] ^= 0xff;

    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "combined merkle + computation attack should be detected");
    // Both Merkle and Freivalds should fail
    assert!(
        failures.iter().any(|f| f.contains("Merkle")),
        "should detect Merkle tampering: {:?}",
        failures
    );
    assert!(
        failures.iter().any(|f| f.contains("Wd")),
        "should detect computation tampering on Wd: {:?}",
        failures
    );
}

#[test]
fn test_cross_token_coordinated_corruption() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Run autoregressive forward pass with 8 tokens
    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);

    // Corrupt token 3's ffn_out at the last layer (boundary between tokens)
    let last_layer = cfg.n_layers - 1;
    all_layers[3][last_layer].ffn_out[0] =
        all_layers[3][last_layer].ffn_out[0].wrapping_add(99);

    // Corrupt token 4's x_attn at layer 0 (boundary input)
    all_layers[4][0].x_attn[0] =
        all_layers[4][0].x_attn[0].wrapping_add(77);

    // Commit and open all tokens
    let (_commitment, _state, traces) = build_batch(all_layers);

    // Build a BatchProof for verify_batch
    let proof = verilm_core::types::BatchProof {
        commitment: _commitment,
        traces,
        revealed_seed: None,
    };
    let challenges: Vec<u32> = (0..8).collect();
    let (passed, failures) = verify_batch(&key, &proof, &challenges);

    assert!(!passed, "coordinated cross-token corruption should be detected");
    // Token 3 should fail (Wd Freivalds on the corrupted ffn_out)
    assert!(
        failures.iter().any(|f| f.contains("token 3")),
        "token 3 corruption should be detected: {:?}",
        failures
    );
    // Token 4 should fail (corrupted x_attn breaks Wq/Wk/Wv checks or chain check)
    assert!(
        failures.iter().any(|f| f.contains("token 4")),
        "token 4 corruption should be detected: {:?}",
        failures
    );
}

#[test]
fn test_chain_preserving_local_forgery() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let mut layers = forward_pass(&cfg, &model, &input);

    // Change g[0] in layer 0
    layers[0].g[0] = layers[0].g[0].wrapping_add(5);

    // Recompute h from the corrupted g to keep g/h "locally consistent"
    let new_h = verilm_core::silu::compute_h_unit_scale(&layers[0].g, &layers[0].u);
    layers[0].h = new_h;

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "chain-preserving local forgery should be detected");
    // The Freivalds check on Wg should catch the g corruption since
    // g = Wg @ x_ffn and x_ffn hasn't changed but g has
    assert!(
        failures.iter().any(|f| f.contains("Wg")),
        "Wg Freivalds check should catch g corruption: {:?}",
        failures
    );
}

#[test]
fn test_mixed_honest_dishonest_batch() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // Run autoregressive forward pass with 8 tokens
    let mut all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);

    // Corrupt token 0: tamper with q
    all_layers[0][0].q[0] = all_layers[0][0].q[0].wrapping_add(10);

    // Corrupt token 3: tamper with attn_out
    all_layers[3][0].attn_out[0] = all_layers[3][0].attn_out[0].wrapping_add(20);

    // Corrupt token 7: tamper with ffn_out
    let last_layer = cfg.n_layers - 1;
    all_layers[7][last_layer].ffn_out[0] =
        all_layers[7][last_layer].ffn_out[0].wrapping_add(30);

    let (_commitment, _state, traces) = build_batch(all_layers);

    let proof = verilm_core::types::BatchProof {
        commitment: _commitment,
        traces,
        revealed_seed: None,
    };
    let challenges: Vec<u32> = (0..8).collect();
    let (passed, failures) = verify_batch(&key, &proof, &challenges);

    assert!(!passed, "mixed batch should fail verification");

    // Exactly tokens 0, 3, 7 should have failures
    let corrupted_tokens: Vec<u32> = vec![0, 3, 7];
    let honest_tokens: Vec<u32> = vec![1, 2, 4, 5, 6];

    for &t in &corrupted_tokens {
        assert!(
            failures.iter().any(|f| f.contains(&format!("token {}", t))),
            "token {} should fail: {:?}",
            t,
            failures
        );
    }

    for &t in &honest_tokens {
        assert!(
            !failures.iter().any(|f| f.starts_with(&format!("token {}:", t))),
            "token {} should pass (honest) but found failure: {:?}",
            t,
            failures
        );
    }
}

// -----------------------------------------------------------------------
// 10a. Adversarial composition: multi-layer multi-matrix simultaneous corruption
// -----------------------------------------------------------------------

#[test]
fn test_multi_layer_multi_matrix_simultaneous_corruption() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let mut layers = forward_pass(&cfg, &model, &input);

    // Corrupt Wq output in layer 0 AND Wg output in layer 1 simultaneously
    layers[0].q[0] = layers[0].q[0].wrapping_add(42);
    layers[1].g[0] = layers[1].g[0].wrapping_add(77);

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "multi-layer multi-matrix corruption should be detected");
    // Both corruptions should be independently detected
    assert!(
        failures.iter().any(|f| f.contains("layer 0") && f.contains("Wq")),
        "layer 0 Wq corruption should be detected: {:?}",
        failures
    );
    assert!(
        failures.iter().any(|f| f.contains("layer 1") && f.contains("Wg")),
        "layer 1 Wg corruption should be detected: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 10b. Adversarial composition: SiLU + chain combined attack
// -----------------------------------------------------------------------

#[test]
fn test_silu_plus_chain_combined_attack() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let mut layers = forward_pass(&cfg, &model, &input);

    // Corrupt g, recompute h to keep SiLU consistent, and also corrupt ffn_out
    // to try to hide the chain inconsistency
    layers[0].g[0] = layers[0].g[0].wrapping_add(5);
    // Recompute h from corrupted g to maintain SiLU consistency
    let forged_h = verilm_core::silu::compute_h_unit_scale(&layers[0].g, &layers[0].u);
    layers[0].h = forged_h;
    // Also corrupt ffn_out to try to mask the downstream impact
    layers[0].ffn_out[0] = layers[0].ffn_out[0].wrapping_add(10);

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "SiLU + chain combined attack should be detected");
    // Should catch Wg (g is wrong) and Wd (ffn_out is wrong because h changed
    // and ffn_out was additionally tampered)
    assert!(
        failures.iter().any(|f| f.contains("Wg")),
        "Wg corruption should be detected: {:?}",
        failures
    );
    assert!(
        failures.iter().any(|f| f.contains("Wd")),
        "Wd corruption should be detected: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 10c. Adversarial composition: attention + FFN combined attack
// -----------------------------------------------------------------------

#[test]
fn test_attention_plus_ffn_combined_attack() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let mut layers = forward_pass(&cfg, &model, &input);

    // Corrupt both the attention vector (a) and FFN input (x_ffn)
    layers[0].a[0] = layers[0].a[0].wrapping_add(1);
    layers[0].x_ffn[0] = layers[0].x_ffn[0].wrapping_add(1);

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "attention + FFN combined attack should be detected");
    // Wo check should fail (a is wrong)
    assert!(
        failures.iter().any(|f| f.contains("Wo")),
        "Wo check should catch a corruption: {:?}",
        failures
    );
    // Wg and/or Wu checks should fail (x_ffn is wrong)
    assert!(
        failures.iter().any(|f| f.contains("Wg") || f.contains("Wu")),
        "FFN matrix checks should catch x_ffn corruption: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 10d. Adversarial composition: partial model swap (subset of layers)
// -----------------------------------------------------------------------

#[test]
fn test_partial_model_swap_detected() {
    let (cfg, real_model, input) = setup();
    let fake_model = generate_model(&cfg, 99999);

    // Use real model for layer 0, fake model for layer 1
    // First, run honest pass through layer 0
    let honest_layers = forward_pass(&cfg, &real_model, &input);
    let fake_layers = forward_pass(&cfg, &fake_model, &input);

    // Construct a mixed trace: layer 0 from real model, layer 1 from fake model
    let mixed_layers = vec![honest_layers[0].clone(), fake_layers[1].clone()];

    let key = generate_key(&cfg, &real_model, [1u8; 32]);
    let trace = build_trace(mixed_layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "partial model swap (fake layer 1 only) should be detected");
    // Layer 1 should fail on matrix checks because it used the wrong model weights
    assert!(
        failures.iter().any(|f| f.contains("layer 1")),
        "layer 1 should have failures from fake model: {:?}",
        failures
    );
    // Layer 0 Freivalds checks should still pass (honest computation)
    // but the chain check layer0->layer1 may also fail since the fake layer 1
    // has a different x_attn than what the honest layer 0 output would produce
    assert!(
        failures.iter().any(|f| f.contains("chain check failed") || f.contains("layer 1")),
        "chain or layer 1 matrix failures expected: {:?}",
        failures
    );
}

#[test]
fn test_silu_forgery_with_consistent_h() {
    let (cfg, model, input) = setup();
    let key = generate_key(&cfg, &model, [1u8; 32]);

    let mut layers = forward_pass(&cfg, &model, &input);

    // First, verify that the honest h is consistent
    let honest_h = verilm_core::silu::compute_h_unit_scale(&layers[0].g, &layers[0].u);
    assert_eq!(layers[0].h, honest_h, "honest h should match");

    // Now corrupt g[0]
    layers[0].g[0] += 1;

    // Recompute h from the corrupted g so h is consistent with the new g
    let forged_h = verilm_core::silu::compute_h_unit_scale(&layers[0].g, &layers[0].u);
    layers[0].h = forged_h;

    let trace = build_trace(layers, 0, 10);
    let (passed, failures) = verify_trace(&key, &trace);

    assert!(!passed, "silu forgery with consistent h should be detected");
    // The Freivalds check on Wg should catch the g corruption
    // because g = Wg @ x_ffn, and x_ffn is unchanged
    assert!(
        failures.iter().any(|f| f.contains("Wg")),
        "detection should include Wg: {:?}",
        failures
    );
}
