//! Tests for streaming transcript surgery attacks against V3 transcript chaining.
//!
//! Each test simulates a specific attack (insertion, deletion, reordering,
//! retroactive replacement, splicing) and verifies that V3's chained IO hash
//! and Merkle binding detect the tampering.

use verilm_core::constants::ModelConfig;
use verilm_test_vectors::*;

fn setup() -> (ModelConfig, Vec<LayerWeights>, verilm_core::types::VerifierKey) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let key = generate_key(&cfg, &model, [7u8; 32]);
    (cfg, model, key)
}

// -----------------------------------------------------------------------
// 1. Insertion between consecutive opened tokens
// -----------------------------------------------------------------------

/// Attack: insert a fake token between consecutive opened tokens by shifting
/// all subsequent traces' token_index values up by one.
///
/// Detection: Merkle proofs are bound to the original leaf_index. After
/// shifting token_index, the leaf_index in the Merkle proof no longer matches
/// the claimed token_index, causing verification to fail. Additionally, the
/// IO chain breaks because the fake token's IO hash was never committed.
#[test]
fn test_insertion_between_consecutive_opened_tokens() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    // Honest 4-token V3 batch
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![100, 200, 300, 400];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"insertion attack test",
        sampling_seed: [10u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params);
    let challenges: Vec<u32> = (0..4).collect();
    let mut proof = open(&state, &challenges);

    // Generate a fake token from an independent forward pass
    let fake_input: Vec<i8> = (0..cfg.hidden_dim).map(|i| ((i + 7) % 256) as i8).collect();
    let fake_layers_all = forward_pass_autoregressive(&cfg, &model, &fake_input, 2);
    let fake_token_layers = fake_layers_all[1].clone();

    // ATTACK: Insert the fake token at position 1 by shifting traces 1,2,3 -> 2,3,4.
    // We modify token_index on existing traces to simulate an expanded sequence.
    // Trace 0 stays at index 0.
    // Insert a new trace at index 1 with the fake layers.
    // Original trace 1 -> index 2, trace 2 -> index 3, trace 3 -> index 4.
    proof.traces[1].token_index = 2;
    proof.traces[2].token_index = 3;
    proof.traces[3].token_index = 4;

    // Create a fake trace for position 1 using the fake token's layers
    // but keeping the commitment's Merkle root (which doesn't include this token)
    let fake_trace = verilm_core::types::TokenTrace {
        token_index: 1,
        layers: fake_token_layers,
        merkle_root: proof.commitment.merkle_root,
        merkle_proof: proof.traces[0].merkle_proof.clone(), // wrong proof
        io_proof: proof.traces[0].io_proof.clone(),         // wrong proof
        margin_cert: None,
        final_hidden: None,
        token_id: Some(150),
        prev_io_hash: Some([0u8; 32]), // bogus
        prev_kv_hash: None,
    };

    // Insert fake trace at position 1
    proof.traces.insert(1, fake_trace);

    // Now we have 5 traces claiming indices 0,1,2,3,4
    let attack_challenges: Vec<u32> = vec![0, 1, 2, 3, 4];

    let (passed, failures) = verify_batch(&key, &proof, &attack_challenges);
    assert!(
        !passed,
        "insertion attack must be detected; failures: {:?}",
        failures
    );
    // Should fail on Merkle proof or IO proof for the shifted/fake tokens
    assert!(
        !failures.is_empty(),
        "should have at least one failure from insertion"
    );
}

// -----------------------------------------------------------------------
// 2. Deletion shifts chain
// -----------------------------------------------------------------------

/// Attack: delete token 2 from a 6-token batch and shift remaining tokens
/// down to hide the gap.
///
/// Detection: Merkle proofs are bound to the original leaf indices. After
/// shifting, the proofs for tokens 3,4,5 (now claiming 2,3,4) fail because
/// their leaf_index no longer matches the claimed token_index.
#[test]
fn test_deletion_shifts_chain() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    // Honest 6-token V3 batch
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 6);
    let token_ids: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"deletion attack test",
        sampling_seed: [20u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params);
    let challenges: Vec<u32> = (0..6).collect();
    let mut proof = open(&state, &challenges);

    // ATTACK: remove token 2 and shift tokens 3,4,5 down by 1
    proof.traces.remove(2); // remove the trace for token_index=2
    // Shift remaining: token 3->2, token 4->3, token 5->4
    proof.traces[2].token_index = 2;
    proof.traces[3].token_index = 3;
    proof.traces[4].token_index = 4;

    // Now we have 5 traces claiming indices 0,1,2,3,4
    let attack_challenges: Vec<u32> = vec![0, 1, 2, 3, 4];

    let (passed, failures) = verify_batch(&key, &proof, &attack_challenges);
    assert!(
        !passed,
        "deletion attack must be detected; failures: {:?}",
        failures
    );
    // The shifted traces should fail because their Merkle proofs have
    // leaf_index bound to the original positions (3,4,5), not the claimed (2,3,4).
    assert!(
        !failures.is_empty(),
        "should report failures from shifted Merkle proofs or IO chain"
    );
}

// -----------------------------------------------------------------------
// 3. Reorder adjacent tokens
// -----------------------------------------------------------------------

/// Attack: swap traces[1] and traces[2] entirely (layers, proofs, everything)
/// and also swap their token_index fields so they claim each other's positions.
///
/// Detection: the cross-token chain check fails because the input of "new
/// token 2" (which was originally token 1) won't match the output of "new
/// token 1" (which was originally token 2). The IO chain also breaks because
/// each token's chained IO hash includes prev_io_hash from the wrong predecessor.
#[test]
fn test_reorder_adjacent_tokens() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    // Honest 4-token V3 batch
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![100, 200, 300, 400];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"reorder attack test",
        sampling_seed: [30u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params);
    let challenges: Vec<u32> = (0..4).collect();
    let mut proof = open(&state, &challenges);

    // ATTACK: swap traces[1] and traces[2] completely
    proof.traces.swap(1, 2);
    // Fix token_index so they claim the swapped positions
    proof.traces[1].token_index = 1;
    proof.traces[2].token_index = 2;

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        !passed,
        "reordering attack must be detected; failures: {:?}",
        failures
    );
    // Should fail on cross-token chain (input/output mismatch) and/or
    // Merkle proof (leaf_index doesn't match claimed position) and/or
    // IO proof (chained hash is wrong for the swapped position).
    assert!(
        !failures.is_empty(),
        "should have failures from reordering"
    );
}

// -----------------------------------------------------------------------
// 4. Reorder swaps token_ids only (layers/proofs untouched)
// -----------------------------------------------------------------------

/// Attack: swap only the token_ids on traces[1] and traces[2], leaving
/// layers, Merkle proofs, and IO proofs untouched.
///
/// Detection: the IO proof fails because the chained IO hash includes the
/// token_id. Changing the token_id changes the recomputed IO hash, which
/// no longer matches the IO tree leaf.
#[test]
fn test_reorder_swaps_token_ids_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![100, 200, 300, 400];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"token_id swap test",
        sampling_seed: [40u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params);
    let challenges: Vec<u32> = (0..4).collect();
    let mut proof = open(&state, &challenges);

    // ATTACK: swap only the token_ids on traces[1] and traces[2]
    let tid1 = proof.traces[1].token_id;
    let tid2 = proof.traces[2].token_id;
    proof.traces[1].token_id = tid2; // was 200, now 300
    proof.traces[2].token_id = tid1; // was 300, now 200

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        !passed,
        "token_id swap must be detected; failures: {:?}",
        failures
    );
    // The IO proof should fail for both swapped tokens because the
    // chained IO hash includes the token_id.
    let io_failures: Vec<_> = failures
        .iter()
        .filter(|f| f.contains("IO proof"))
        .collect();
    assert!(
        !io_failures.is_empty(),
        "should have IO proof failures from token_id swap: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 5. Retroactive replacement of a streamed token
// -----------------------------------------------------------------------

/// Attack: after the batch is committed, replace token 1's token_id with a
/// different value, simulating retroactive editing of a token that was already
/// streamed to the user.
///
/// Detection: the IO proof fails for the modified token because the chained
/// IO hash includes the token_id. The chain propagation also causes failures
/// for all subsequent tokens whose prev_io_hash depends on the tampered token.
#[test]
fn test_retroactive_replacement_of_streamed_token() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![100, 200, 300, 400];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"retroactive edit test",
        sampling_seed: [50u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params);
    let challenges: Vec<u32> = (0..4).collect();
    let mut proof = open(&state, &challenges);

    // ATTACK: retroactively change token 1's emitted token_id
    assert_eq!(proof.traces[1].token_id, Some(200));
    proof.traces[1].token_id = Some(999);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        !passed,
        "retroactive token replacement must be detected; failures: {:?}",
        failures
    );

    // Token 1 should fail with IO proof failure (wrong token_id in hash)
    let io_failures: Vec<_> = failures
        .iter()
        .filter(|f| f.contains("IO proof"))
        .collect();
    assert!(
        !io_failures.is_empty(),
        "should have IO proof failure for tampered token: {:?}",
        failures
    );

    // The chain should propagate: token 2 (and possibly token 3) should also
    // fail because token 2's prev_io_hash cross-check against token 1's
    // recomputed IO hash will mismatch.
    let chain_failures: Vec<_> = failures
        .iter()
        .filter(|f| f.contains("prev_io_hash mismatch"))
        .collect();
    assert!(
        !chain_failures.is_empty(),
        "chain propagation should cause prev_io_hash mismatch for subsequent tokens: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 6. Sparse opening does not detect intermediate manipulation (limitation)
// -----------------------------------------------------------------------

/// Non-attack: open only tokens 0 and 7 from an 8-token batch (sparse).
/// Since no consecutive tokens are opened, there is no cross-token chain
/// check to catch manipulation of intermediate tokens 1-6. Each individually
/// opened token still passes its own Merkle + IO proof.
///
/// This documents an inherent limitation of sparse opening: the verifier
/// can only cross-check tokens that are both opened AND consecutive.
#[test]
fn test_insertion_not_detected_without_consecutive_opening() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    // Honest 8-token V3 batch
    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 8);
    let token_ids: Vec<u32> = (100..108).collect();
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"sparse opening limitation test",
        sampling_seed: [60u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params);

    // Open only tokens 0 and 7 (no consecutive pair)
    let challenges = vec![0u32, 7];
    let proof = open(&state, &challenges);

    // This should PASS — each token individually verifies fine,
    // and without consecutive openings, no cross-check is possible.
    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        passed,
        "sparse honest opening should pass (documents limitation): {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 7. Deletion detected when consecutive tokens are opened
// -----------------------------------------------------------------------

/// Attack: tamper with token 2's prev_io_hash to simulate it being recomputed
/// as if token 1 didn't exist (use genesis zero hash).
///
/// Detection: when tokens 1 and 2 are both opened (consecutive), the verifier
/// cross-checks that token 2's prev_io_hash matches the recomputed IO hash of
/// token 1. The zero genesis hash won't match token 1's actual IO hash, so
/// both the IO proof and the prev_io_hash cross-check fail.
#[test]
fn test_deletion_detected_when_consecutive_opened() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();

    let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);
    let token_ids = vec![100, 200, 300, 400];
    let params = FullBindingParams {
        token_ids: &token_ids,
        prompt: b"consecutive deletion detection test",
        sampling_seed: [70u8; 32],
        manifest: None,
    };

    let (_commitment, state) = commit_with_full_binding(all_layers, &params);

    // Open tokens 1 and 2 (consecutive)
    let challenges = vec![1u32, 2];
    let mut proof = open(&state, &challenges);

    // ATTACK: tamper token 2's prev_io_hash to genesis zero, simulating
    // that token 1 was deleted and token 2 is now the first token.
    proof.traces[1].prev_io_hash = Some([0u8; 32]);

    let (passed, failures) = verify_batch(&key, &proof, &challenges);
    assert!(
        !passed,
        "deletion via prev_io_hash tampering must be detected; failures: {:?}",
        failures
    );

    // Should fail with prev_io_hash mismatch (cross-check against token 1)
    let prev_io_failures: Vec<_> = failures
        .iter()
        .filter(|f| f.contains("prev_io_hash mismatch"))
        .collect();
    assert!(
        !prev_io_failures.is_empty(),
        "should report prev_io_hash mismatch: {:?}",
        failures
    );

    // Should also fail IO proof for token 2 (wrong chained hash)
    let io_failures: Vec<_> = failures
        .iter()
        .filter(|f| f.contains("IO proof"))
        .collect();
    assert!(
        !io_failures.is_empty(),
        "should report IO proof failure for token with tampered prev_io_hash: {:?}",
        failures
    );
}

// -----------------------------------------------------------------------
// 8. Splice token from a different run
// -----------------------------------------------------------------------

/// Attack: replace trace[1] with a trace from a completely different
/// inference run (different seed, different prompt, different computation).
///
/// Detection: the Merkle proof for the spliced token was computed against
/// run B's Merkle tree, so it won't verify against run A's Merkle root.
/// Additionally, the IO proof will fail because the spliced token's
/// input/output doesn't match run A's IO tree.
#[test]
fn test_splice_token_from_different_run() {
    let (cfg, model, key) = setup();
    let input_a: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let input_b: Vec<i8> = (0..cfg.hidden_dim).map(|i| ((i + 37) % 256) as i8).collect();

    // Run A: honest batch
    let all_layers_a = forward_pass_autoregressive(&cfg, &model, &input_a, 4);
    let token_ids_a = vec![100, 200, 300, 400];
    let params_a = FullBindingParams {
        token_ids: &token_ids_a,
        prompt: b"run A prompt",
        sampling_seed: [80u8; 32],
        manifest: None,
    };

    let (_commitment_a, state_a) = commit_with_full_binding(all_layers_a, &params_a);

    // Run B: different honest batch with different input and prompt
    let all_layers_b = forward_pass_autoregressive(&cfg, &model, &input_b, 4);
    let token_ids_b = vec![500, 600, 700, 800];
    let params_b = FullBindingParams {
        token_ids: &token_ids_b,
        prompt: b"run B prompt",
        sampling_seed: [90u8; 32],
        manifest: None,
    };

    let (_commitment_b, state_b) = commit_with_full_binding(all_layers_b, &params_b);

    // Open all tokens from both runs
    let challenges: Vec<u32> = (0..4).collect();
    let mut proof_a = open(&state_a, &challenges);
    let proof_b = open(&state_b, &challenges);

    // ATTACK: replace run A's trace[1] with run B's trace[1],
    // but keep run A's commitment (Merkle root, IO root, etc.)
    proof_a.traces[1] = proof_b.traces[1].clone();
    // Fix the token_index to match the expected position in run A
    proof_a.traces[1].token_index = 1;
    // Keep run A's merkle_root on the spliced trace (the attacker would try this)
    proof_a.traces[1].merkle_root = proof_a.commitment.merkle_root;

    let (passed, failures) = verify_batch(&key, &proof_a, &challenges);
    assert!(
        !passed,
        "splicing a token from a different run must be detected; failures: {:?}",
        failures
    );
    // The Merkle proof from run B won't verify against run A's tree
    // and/or the IO proof will fail
    assert!(
        !failures.is_empty(),
        "should have failures from cross-run splice"
    );
}
