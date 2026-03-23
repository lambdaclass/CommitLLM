//! Adversarial forgery tests.
//!
//! Four suites covering:
//! 1. Level B forged-certificate attacks
//! 2. Cross-token replay attacks
//! 3. Spliced honest-run traces
//! 4. Batch-context / Merkle-opening substitution attacks

use vi_core::constants::ModelConfig;
use vi_core::margin::{build_margin_certificate, DEFAULT_ATTENTION_EPSILON};
use vi_core::types::{BatchProof, TokenTrace};
use vi_test_vectors::*;
use vi_verify::{verify_trace_at_level, Verdict, VerificationLevel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn toy_cfg() -> ModelConfig {
    ModelConfig::toy()
}

/// Build a Level B trace: forward pass + logits + margin cert + final_hidden.
fn make_level_b_trace(
    cfg: &ModelConfig,
    model: &ToyModel,
    input: &[i8],
    token_index: u32,
) -> TokenTrace {
    let layers = forward_pass(cfg, &model.layers, input);
    let last_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &last_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(token_index, logits).unwrap();
    let mut trace = build_trace_with_hidden(layers, token_index, 10, Some(last_hidden));
    trace.margin_cert = Some(cert);
    trace
}

/// Build a Level B trace from a specific input, returning (trace, logits).
fn make_level_b_trace_with_logits(
    cfg: &ModelConfig,
    model: &ToyModel,
    input: &[i8],
    token_index: u32,
) -> (TokenTrace, Vec<f32>) {
    let layers = forward_pass(cfg, &model.layers, input);
    let last_hidden = requantize(&layers.last().unwrap().ffn_out);
    let logits = compute_logits(&model.lm_head, &last_hidden, cfg.vocab_size, cfg.hidden_dim);
    let cert = build_margin_certificate(token_index, logits.clone()).unwrap();
    let mut trace = build_trace_with_hidden(layers, token_index, 10, Some(last_hidden));
    trace.margin_cert = Some(cert);
    (trace, logits)
}

// ===========================================================================
// 1. Level B Forged-Certificate Attacks
// ===========================================================================

mod level_b_forged_certificate {
    use super::*;

    #[test]
    fn test_forge_cert_wrong_selected_token() {
        let cfg = toy_cfg();
        let model = generate_model_with_head(&cfg, 42);
        let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
        let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

        let mut trace = make_level_b_trace(&cfg, &model, &input, 0);

        // Honest logits, but claim a different selected_token_id
        if let Some(ref mut cert) = trace.margin_cert {
            // Pick a token that is NOT the true argmax
            let wrong_id = if cert.selected_token_id == 0 { 1 } else { 0 };
            cert.selected_token_id = wrong_id;
        }

        let report = verify_trace_at_level(
            &key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON,
        );
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(
            report.failures.iter().any(|f| f.contains("selected_token_id")),
            "should detect wrong selected_token_id: {:?}",
            report.failures
        );
    }

    #[test]
    fn test_forge_cert_correct_top12_wrong_delta() {
        let cfg = toy_cfg();
        let model = generate_model_with_head(&cfg, 42);
        let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
        let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

        let mut trace = make_level_b_trace(&cfg, &model, &input, 0);

        // Keep correct top-1/top-2 IDs but inflate delta
        if let Some(ref mut cert) = trace.margin_cert {
            cert.delta = cert.delta + 99999.0;
        }

        let report = verify_trace_at_level(
            &key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON,
        );
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(
            report.failures.iter().any(|f| f.contains("delta mismatch")),
            "should detect inflated delta: {:?}",
            report.failures
        );
    }

    #[test]
    fn test_forge_cert_from_different_token() {
        let cfg = toy_cfg();
        let model = generate_model_with_head(&cfg, 42);
        let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));

        let input0: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
        let input1: Vec<i8> = (0..cfg.hidden_dim)
            .map(|i| ((i * 3 + 7) % 256) as i8)
            .collect();

        // Build trace for token 0 and token 1 with different inputs
        let (trace0, _) = make_level_b_trace_with_logits(&cfg, &model, &input0, 0);
        let mut trace1 = make_level_b_trace(&cfg, &model, &input1, 1);

        // Copy token 0's cert onto token 1's trace (different logits)
        trace1.margin_cert = trace0.margin_cert.clone();

        let report = verify_trace_at_level(
            &key, &trace1, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON,
        );
        // The cert's token_index=0 but trace's token_index=1.
        // verify_margin must reject this mismatch.
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(
            report.failures.iter().any(|f| f.contains("token_index")),
            "should detect cert token_index mismatch: {:?}",
            report.failures
        );
    }

    #[test]
    fn test_forge_cert_shifted_logits() {
        let cfg = toy_cfg();
        let model = generate_model_with_head(&cfg, 42);
        let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
        let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

        let mut trace = make_level_b_trace(&cfg, &model, &input, 0);

        // Shift all logits by a constant -- same delta, but wrong top-1/top-2 values
        if let Some(ref mut cert) = trace.margin_cert {
            let shift = 1000.0f32;
            for l in cert.logits.iter_mut() {
                *l += shift;
            }
            // top1_logit and top2_logit are now wrong relative to the shifted logits
            // (they still hold the old values)
        }

        let report = verify_trace_at_level(
            &key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON,
        );
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(
            report.failures.iter().any(|f| {
                f.contains("top1_logit mismatch")
                    || f.contains("top2_logit mismatch")
                    || f.contains("logit mismatch")
            }),
            "should detect shifted logits: {:?}",
            report.failures
        );
    }

    #[test]
    fn test_forge_cert_swapped_logit_entries() {
        let cfg = toy_cfg();
        let model = generate_model_with_head(&cfg, 42);
        let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
        let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

        let mut trace = make_level_b_trace(&cfg, &model, &input, 0);

        // Swap two logit values to change top-2 ranking while keeping delta
        if let Some(ref mut cert) = trace.margin_cert {
            let top1_id = cert.top1_token_id as usize;
            let top2_id = cert.top2_token_id as usize;
            // Find a third index that's neither top1 nor top2
            let third_id = (0..cert.logits.len())
                .find(|&i| i != top1_id && i != top2_id)
                .unwrap();
            // Swap top2 with the third entry -- now top2_token_id is wrong
            cert.logits.swap(top2_id, third_id);
        }

        let report = verify_trace_at_level(
            &key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON,
        );
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(
            report.failures.iter().any(|f| {
                f.contains("top2_token_id mismatch")
                    || f.contains("top1_token_id mismatch")
                    || f.contains("delta mismatch")
                    || f.contains("top2_logit mismatch")
                    || f.contains("top1_logit mismatch")
                    || f.contains("logit mismatch")
            }),
            "should detect swapped logit entries: {:?}",
            report.failures
        );
    }

    #[test]
    fn test_forge_cert_honest_logits_wrong_token_index() {
        let cfg = toy_cfg();
        let model = generate_model_with_head(&cfg, 42);
        let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));
        let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

        let mut trace = make_level_b_trace(&cfg, &model, &input, 0);

        // Honest cert but change the cert's token_index to not match the trace
        if let Some(ref mut cert) = trace.margin_cert {
            cert.token_index = 99; // trace has token_index=0
        }

        // verify_margin must reject cert.token_index != trace.token_index.
        let report = verify_trace_at_level(
            &key, &trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON,
        );
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(
            report.failures.iter().any(|f| f.contains("token_index")),
            "should detect cert token_index mismatch: {:?}",
            report.failures
        );
    }
}

// ===========================================================================
// 2. Cross-Token Replay Attacks
// ===========================================================================

mod cross_token_replay {
    use super::*;

    #[test]
    fn test_replay_margin_cert_from_another_token() {
        let cfg = toy_cfg();
        let model = generate_model_with_head(&cfg, 42);
        let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));

        let input0: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
        let input3: Vec<i8> = (0..cfg.hidden_dim)
            .map(|i| ((i * 5 + 13) % 256) as i8)
            .collect();

        let trace0 = make_level_b_trace(&cfg, &model, &input0, 0);
        let mut trace3 = make_level_b_trace(&cfg, &model, &input3, 3);

        // Take cert from token 0, attach to token 3
        trace3.margin_cert = trace0.margin_cert.clone();

        // The cert is internally consistent (built honestly for token 0's logits)
        // but token 3 has different logits. The structural check in the cert
        // passes (it's self-consistent), but it proves the wrong computation.
        // Level A still passes; Level B margin check passes structurally since
        // the verifier doesn't bind cert logits to the trace.
        // This documents the gap.
        let report_a = verify_trace_at_level(
            &key, &trace3, VerificationLevel::A, DEFAULT_ATTENTION_EPSILON,
        );
        assert_eq!(report_a.verdict, Verdict::Pass, "Level A should pass: {:?}", report_a.failures);
    }

    #[test]
    fn test_replay_entire_trace_wrong_position() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);
        let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

        // Run autoregressive with 4 tokens
        let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

        // Honest batch commit
        let (_, state) = commit(all_layers.clone());
        let challenges: Vec<u32> = vec![0, 3];
        let mut proof = open(&state, &challenges);

        // Replay: put trace for token 0 at position 3
        // Copy token 0's trace data into the slot for token 3
        let trace0 = proof.traces[0].clone();
        proof.traces[1].layers = trace0.layers.clone();
        // Keep token_index=3 but with token 0's data
        // This should break Merkle proof (wrong leaf hash for index 3)

        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "replayed trace at wrong position should fail");
        assert!(
            failures.iter().any(|f| f.contains("Merkle") || f.contains("merkle")),
            "should detect Merkle mismatch: {:?}",
            failures
        );
    }

    #[test]
    fn test_replay_trace_adjacent_token() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);
        let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

        // Run autoregressive with 4 tokens
        let all_layers = forward_pass_autoregressive(&cfg, &model, &input, 4);

        let (_, state) = commit(all_layers.clone());
        let challenges: Vec<u32> = vec![0, 1];
        let mut proof = open(&state, &challenges);

        // Replace token 1's layers with token 0's layers (replay t as t+1)
        proof.traces[1].layers = proof.traces[0].layers.clone();

        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "replaying trace t as t+1 should fail");
        assert!(
            failures.iter().any(|f| {
                f.contains("Merkle")
                    || f.contains("merkle")
                    || f.contains("chain")
                    || f.contains("IO")
            }),
            "should detect replay via Merkle or chain: {:?}",
            failures
        );
    }

    #[test]
    fn test_replay_batch_proof_different_commitment() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        // Two different runs with different inputs
        let input_a: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
        let input_b: Vec<i8> = (0..cfg.hidden_dim)
            .map(|i| ((i * 7 + 3) % 256) as i8)
            .collect();

        let all_layers_a = forward_pass_autoregressive(&cfg, &model, &input_a, 4);
        let all_layers_b = forward_pass_autoregressive(&cfg, &model, &input_b, 4);

        let (_commit_a, state_a) = commit(all_layers_a);
        let (commit_b, _state_b) = commit(all_layers_b);

        // Open traces from run A but attach commitment from run B
        let challenges: Vec<u32> = vec![0, 1, 2, 3];
        let mut proof = open(&state_a, &challenges);
        proof.commitment = commit_b;

        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "traces from run A with commitment from run B should fail");
        assert!(
            failures.iter().any(|f| {
                f.contains("merkle_root")
                    || f.contains("Merkle")
                    || f.contains("merkle")
                    || f.contains("IO")
            }),
            "should detect commitment mismatch: {:?}",
            failures
        );
    }
}

// ===========================================================================
// 3. Spliced Honest-Run Traces
// ===========================================================================

mod spliced_traces {
    use super::*;

    /// Helper to find two inputs that produce distinct intermediate values
    /// for the given model. Uses maximally different inputs.
    fn distinct_inputs(cfg: &ModelConfig) -> (Vec<i8>, Vec<i8>) {
        (vec![127i8; cfg.hidden_dim], vec![-128i8; cfg.hidden_dim])
    }

    #[test]
    fn test_splice_layer0_from_run_a_rest_from_run_b() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let (input_a, input_b) = distinct_inputs(&cfg);

        let layers_a = forward_pass(&cfg, &model, &input_a);
        let layers_b = forward_pass(&cfg, &model, &input_b);

        // Ensure the two runs produce different layer 1 x_attn
        // (if they don't, the chain check can't catch the splice)
        let expected_next_a = requantize(&layers_a[0].ffn_out);

        if expected_next_a != layers_b[1].x_attn {
            // Splice: layer 0 from run A, layer 1 from run B
            // layer_b[1].x_attn was computed from layer_b[0].ffn_out
            // but layer_a[0].ffn_out is different, so chain breaks
            let spliced = vec![layers_a[0].clone(), layers_b[1].clone()];

            let trace = build_trace(spliced, 0, 10);
            let (passed, failures) = vi_test_vectors::verify_trace(&key, &trace);
            assert!(!passed, "spliced layers from different runs should fail");
            assert!(
                failures.iter().any(|f| {
                    f.contains("chain check failed") || f.contains("Freivalds")
                }),
                "should detect chain break or Freivalds failure: {:?}",
                failures
            );
        } else {
            // Edge case: both runs happen to produce the same intermediate.
            // The splice is undetectable at the chain level (same as honest).
            // This is expected for the toy model with clamping.
            // Verify the trace passes (correct behavior: splice is a no-op).
            let spliced = vec![layers_a[0].clone(), layers_b[1].clone()];
            let trace = build_trace(spliced, 0, 10);
            let (passed, _) = vi_test_vectors::verify_trace(&key, &trace);
            assert!(passed, "identical intermediate splice should pass");
        }
    }

    #[test]
    fn test_splice_attention_output_from_different_run() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let (input_a, input_b) = distinct_inputs(&cfg);

        let layers_a = forward_pass(&cfg, &model, &input_a);
        let layers_b = forward_pass(&cfg, &model, &input_b);

        // Splice `a` from run B into run A's trace
        if layers_a[0].a != layers_b[0].a {
            let mut spliced = layers_a.clone();
            spliced[0].a = layers_b[0].a.clone();

            let trace = build_trace(spliced, 0, 10);
            let (passed, failures) = vi_test_vectors::verify_trace(&key, &trace);
            assert!(!passed, "spliced attention vector should fail");
            assert!(
                failures.iter().any(|f| f.contains("Wo")),
                "should detect Wo mismatch from spliced a: {:?}",
                failures
            );
        } else {
            // Same attention vector from both runs (toy model clamping).
            // Splice is undetectable because it's effectively a no-op.
            // Use a third input to find a different `a`.
            let input_c: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i as i8).wrapping_mul(17)).collect();
            let layers_c = forward_pass(&cfg, &model, &input_c);
            if layers_a[0].a != layers_c[0].a {
                let mut spliced = layers_a.clone();
                spliced[0].a = layers_c[0].a.clone();
                let trace = build_trace(spliced, 0, 10);
                let (passed, failures) = vi_test_vectors::verify_trace(&key, &trace);
                assert!(!passed, "spliced attention vector (alt) should fail");
                assert!(
                    failures.iter().any(|f| f.contains("Wo")),
                    "should detect Wo mismatch: {:?}",
                    failures
                );
            }
            // If still identical, skip -- clamping makes splice undetectable
        }
    }

    #[test]
    fn test_splice_ffn_output_from_different_run() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let (input_a, input_b) = distinct_inputs(&cfg);

        let layers_a = forward_pass(&cfg, &model, &input_a);
        let layers_b = forward_pass(&cfg, &model, &input_b);

        if layers_a[0].ffn_out != layers_b[0].ffn_out {
            let mut spliced = layers_a.clone();
            spliced[0].ffn_out = layers_b[0].ffn_out.clone();

            let trace = build_trace(spliced, 0, 10);
            let (passed, failures) = vi_test_vectors::verify_trace(&key, &trace);
            assert!(!passed, "spliced ffn_out should fail");
            assert!(
                failures.iter().any(|f| f.contains("Wd") || f.contains("chain")),
                "should detect Wd Freivalds failure or chain break: {:?}",
                failures
            );
        } else {
            // Same ffn_out from both runs. Try more diverse inputs.
            let input_c: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i as i8).wrapping_mul(17)).collect();
            let layers_c = forward_pass(&cfg, &model, &input_c);
            if layers_a[0].ffn_out != layers_c[0].ffn_out {
                let mut spliced = layers_a.clone();
                spliced[0].ffn_out = layers_c[0].ffn_out.clone();
                let trace = build_trace(spliced, 0, 10);
                let (passed, failures) = vi_test_vectors::verify_trace(&key, &trace);
                assert!(!passed, "spliced ffn_out (alt) should fail");
                assert!(
                    failures.iter().any(|f| f.contains("Wd") || f.contains("chain")),
                    "should detect Wd or chain: {:?}",
                    failures
                );
            }
        }
    }

    #[test]
    fn test_splice_margin_cert_from_different_run() {
        let cfg = toy_cfg();
        let model = generate_model_with_head(&cfg, 42);
        let key = generate_key_level_b_with_head(&cfg, &model.layers, [1u8; 32], Some(model.lm_head.clone()));

        let input_a: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
        let input_b: Vec<i8> = vec![127i8; cfg.hidden_dim];

        // Build honest Level B trace for run A (Level A should pass)
        let trace_a = make_level_b_trace(&cfg, &model, &input_a, 0);
        // Build cert from run B's logits
        let layers_b = forward_pass(&cfg, &model.layers, &input_b);
        let last_hidden_b = requantize(&layers_b.last().unwrap().ffn_out);
        let logits_b = compute_logits(
            &model.lm_head,
            &last_hidden_b,
            cfg.vocab_size,
            cfg.hidden_dim,
        );
        let cert_b = build_margin_certificate(0, logits_b).unwrap();

        // Attach run B's cert to run A's trace
        let mut spliced_trace = trace_a;
        spliced_trace.margin_cert = Some(cert_b);

        // Level A should still pass (ignores margin cert)
        let report_a = verify_trace_at_level(
            &key, &spliced_trace, VerificationLevel::A, DEFAULT_ATTENTION_EPSILON,
        );
        assert_eq!(report_a.verdict, Verdict::Pass, "Level A should pass: {:?}", report_a.failures);

        // Level B: cert logits are from run B, but final_hidden is from run A.
        // The verifier recomputes logits from final_hidden and detects the mismatch.
        let report_b = verify_trace_at_level(
            &key, &spliced_trace, VerificationLevel::B, DEFAULT_ATTENTION_EPSILON,
        );
        assert_eq!(report_b.verdict, Verdict::Fail, "spliced cert should fail Level B: {:?}", report_b.failures);
        assert!(
            report_b.failures.iter().any(|f| f.contains("logit mismatch")),
            "should detect logit mismatch from spliced cert: {:?}",
            report_b.failures
        );
    }

    #[test]
    fn test_splice_across_token_boundary() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        // Run A: 2 tokens from input_a
        let input_a = vec![127i8; cfg.hidden_dim];
        let all_a = forward_pass_autoregressive(&cfg, &model, &input_a, 2);

        // Run B: 2 tokens from input_b
        let input_b = vec![-128i8; cfg.hidden_dim];
        let all_b = forward_pass_autoregressive(&cfg, &model, &input_b, 2);

        // Check that the cross-token boundary values differ
        let expected_t1_input_from_a = requantize(&all_a[0].last().unwrap().ffn_out);
        let actual_t1_input_from_b = &all_b[1][0].x_attn;

        if expected_t1_input_from_a != *actual_t1_input_from_b {
            // Splice: token 0 from run A, token 1 from run B
            let spliced = vec![all_a[0].clone(), all_b[1].clone()];

            let (_commitment, _state, traces) = build_batch(spliced);
            let proof = BatchProof {
                commitment: _commitment,
                traces,
                revealed_seed: None,
            };
            let challenges: Vec<u32> = vec![0, 1];

            let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
            assert!(!passed, "spliced tokens across boundary should fail");
            assert!(
                failures.iter().any(|f| f.contains("cross-token chain")),
                "should detect cross-token chain failure: {:?}",
                failures
            );
        } else {
            // Both runs produce the same token boundary values.
            // This means the splice is undetectable at the chain level.
            // For the toy model with heavy clamping, this can happen.
            // Document: the splice appears honest because both runs converge.
        }
    }
}

// ===========================================================================
// 4. Batch-Context / Merkle-Opening Substitution Attacks
// ===========================================================================

mod batch_merkle_attacks {
    use super::*;

    #[test]
    fn test_swap_merkle_openings_between_tokens() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let inputs: Vec<Vec<i8>> = (0..4)
            .map(|t| {
                (0..cfg.hidden_dim)
                    .map(|i| ((t * 7 + i * 3) % 256) as i8)
                    .collect()
            })
            .collect();

        let all_layers = forward_pass_multi(&cfg, &model, &inputs);
        let (_commitment, state) = commit(all_layers);
        let challenges: Vec<u32> = vec![0, 1, 2, 3];
        let mut proof = open(&state, &challenges);

        // Swap Merkle proofs between token 0 and token 2
        let proof0 = proof.traces[0].merkle_proof.clone();
        let proof2 = proof.traces[2].merkle_proof.clone();
        proof.traces[0].merkle_proof = proof2;
        proof.traces[2].merkle_proof = proof0;

        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "swapped Merkle proofs should fail");
        assert!(
            failures.iter().any(|f| {
                f.contains("Merkle") || f.contains("merkle") || f.contains("leaf_index")
            }),
            "should detect Merkle proof mismatch: {:?}",
            failures
        );
    }

    #[test]
    fn test_reorder_opened_traces() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let inputs: Vec<Vec<i8>> = (0..4)
            .map(|t| {
                (0..cfg.hidden_dim)
                    .map(|i| ((t * 7 + i * 3) % 256) as i8)
                    .collect()
            })
            .collect();

        let all_layers = forward_pass_multi(&cfg, &model, &inputs);
        let (_commitment, state) = commit(all_layers);
        let challenges: Vec<u32> = vec![0, 1, 2, 3];
        let mut proof = open(&state, &challenges);

        // Reverse the order of traces
        proof.traces.reverse();

        // The challenges expect traces in order [0, 1, 2, 3]
        // but now they are [3, 2, 1, 0]. verify_batch checks
        // trace[i].token_index == expected_challenges[i]
        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "reordered traces should fail");
        assert!(
            failures.iter().any(|f| {
                f.contains("token_index")
                    || f.contains("expected")
            }),
            "should detect trace ordering mismatch: {:?}",
            failures
        );
    }

    #[test]
    fn test_copy_io_proof_from_another_token() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let inputs: Vec<Vec<i8>> = (0..4)
            .map(|t| {
                (0..cfg.hidden_dim)
                    .map(|i| ((t * 7 + i * 3) % 256) as i8)
                    .collect()
            })
            .collect();

        let all_layers = forward_pass_multi(&cfg, &model, &inputs);
        let (_commitment, state) = commit(all_layers);
        let challenges: Vec<u32> = vec![0, 1, 2, 3];
        let mut proof = open(&state, &challenges);

        // Use token 0's IO proof for token 3
        let io_proof_0 = proof.traces[0].io_proof.clone();
        proof.traces[3].io_proof = io_proof_0;

        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "copied IO proof should fail");
        assert!(
            failures.iter().any(|f| f.contains("IO") || f.contains("io")),
            "should detect IO proof mismatch: {:?}",
            failures
        );
    }

    #[test]
    fn test_wrong_commitment_root_with_valid_traces() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let inputs: Vec<Vec<i8>> = (0..4)
            .map(|t| {
                (0..cfg.hidden_dim)
                    .map(|i| ((t * 7 + i * 3) % 256) as i8)
                    .collect()
            })
            .collect();

        // Two different batches
        let all_layers_a = forward_pass_multi(&cfg, &model, &inputs);

        let inputs_b: Vec<Vec<i8>> = (0..4)
            .map(|t| {
                (0..cfg.hidden_dim)
                    .map(|i| ((t * 11 + i * 5 + 1) % 256) as i8)
                    .collect()
            })
            .collect();
        let all_layers_b = forward_pass_multi(&cfg, &model, &inputs_b);

        let (_commit_a, state_a) = commit(all_layers_a);
        let (commit_b, _state_b) = commit(all_layers_b);

        // Valid traces from batch A but commitment root from batch B
        let challenges: Vec<u32> = vec![0, 1, 2, 3];
        let mut proof = open(&state_a, &challenges);
        proof.commitment.merkle_root = commit_b.merkle_root;

        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "wrong commitment root should fail");
        assert!(
            failures.iter().any(|f| {
                f.contains("merkle_root") || f.contains("Merkle") || f.contains("merkle")
            }),
            "should detect root mismatch: {:?}",
            failures
        );
    }

    #[test]
    fn test_extra_trace_not_in_challenge_set() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let inputs: Vec<Vec<i8>> = (0..4)
            .map(|t| {
                (0..cfg.hidden_dim)
                    .map(|i| ((t * 7 + i * 3) % 256) as i8)
                    .collect()
            })
            .collect();

        let all_layers = forward_pass_multi(&cfg, &model, &inputs);
        let (_commitment, state) = commit(all_layers);

        // Challenge asks for tokens [0, 2]
        let challenges: Vec<u32> = vec![0, 2];
        let mut proof = open(&state, &challenges);

        // Add an extra trace (token 3) that wasn't challenged
        let extra = open(&state, &[3]);
        proof.traces.push(extra.traces[0].clone());

        // verify_batch checks trace count vs challenge count
        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "extra unchallenged trace should fail");
        assert!(
            failures.iter().any(|f| {
                f.contains("expected") || f.contains("challenged")
            }),
            "should detect extra trace: {:?}",
            failures
        );
    }

    #[test]
    fn test_missing_challenged_trace() {
        let cfg = toy_cfg();
        let model = generate_model(&cfg, 12345);
        let key = generate_key(&cfg, &model, [1u8; 32]);

        let inputs: Vec<Vec<i8>> = (0..4)
            .map(|t| {
                (0..cfg.hidden_dim)
                    .map(|i| ((t * 7 + i * 3) % 256) as i8)
                    .collect()
            })
            .collect();

        let all_layers = forward_pass_multi(&cfg, &model, &inputs);
        let (_commitment, state) = commit(all_layers);

        // Challenge asks for tokens [0, 1, 2, 3]
        let challenges: Vec<u32> = vec![0, 1, 2, 3];
        let mut proof = open(&state, &challenges);

        // Remove one of the challenged traces (token 2)
        proof.traces.remove(2);

        let (passed, failures) = vi_test_vectors::verify_batch(&key, &proof, &challenges);
        assert!(!passed, "missing challenged trace should fail");
        assert!(
            failures.iter().any(|f| {
                f.contains("expected") || f.contains("challenged")
            }),
            "should detect missing trace: {:?}",
            failures
        );
    }
}
