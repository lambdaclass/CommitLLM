//! Tests for the stratified audit verification protocol.
//!
//! Uses the toy model (2 layers, hidden=16, kv=4, ffn=32) to exercise:
//! - Routine audit (layer subsampling — but toy has only 2, so all selected)
//! - Full audit (all layers)
//! - Tampered layer detection
//! - Structural mismatch detection
//! - Per-layer streaming KV commitment binding via Merkle proofs
//! - Multi-token autoregressive audit with real attention replay

use verilm_core::constants::ModelConfig;
use verilm_core::streaming::StreamingKvVerifier;
use verilm_core::types::{AuditChallenge, AuditTier};
use verilm_test_vectors::{
    build_audit_response, build_streaming_kv_verifier, forward_pass_autoregressive_level_c,
    forward_pass_level_c, generate_key, generate_model,
};
use verilm_verify::{build_audit_challenge, derive_audit_layers, verify_audit, Verdict};

fn setup() -> (
    ModelConfig,
    Vec<verilm_test_vectors::LayerWeights>,
    verilm_core::types::VerifierKey,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);
    (cfg, model, key)
}

// ---------------------------------------------------------------------------
// derive_audit_layers
// ---------------------------------------------------------------------------

#[test]
fn derive_layers_full_returns_all() {
    let layers = derive_audit_layers(&[0u8; 32], 0, 80, AuditTier::Full);
    assert_eq!(layers.len(), 80);
    assert_eq!(layers, (0..80).collect::<Vec<_>>());
}

#[test]
fn derive_layers_routine_returns_contiguous_prefix() {
    let layers = derive_audit_layers(&[42u8; 32], 5, 80, AuditTier::Routine);
    // Must be a contiguous prefix 0..=L_max with at least min(10, n_layers) layers
    assert!(layers.len() >= 10, "prefix too short: {}", layers.len());
    assert!(layers.len() <= 80);
    assert_eq!(layers, (0..layers.len()).collect::<Vec<_>>());
}

#[test]
fn derive_layers_routine_capped_at_n_layers() {
    let layers = derive_audit_layers(&[0u8; 32], 0, 2, AuditTier::Routine);
    assert_eq!(layers.len(), 2);
    assert_eq!(layers, vec![0, 1]);
}

#[test]
fn derive_layers_deterministic() {
    let a = derive_audit_layers(&[7u8; 32], 3, 80, AuditTier::Routine);
    let b = derive_audit_layers(&[7u8; 32], 3, 80, AuditTier::Routine);
    assert_eq!(a, b);
}

#[test]
fn derive_layers_different_seed_different_result() {
    let a = derive_audit_layers(&[1u8; 32], 0, 80, AuditTier::Routine);
    let b = derive_audit_layers(&[2u8; 32], 0, 80, AuditTier::Routine);
    assert_ne!(a, b);
}

// ---------------------------------------------------------------------------
// build_audit_challenge
// ---------------------------------------------------------------------------

#[test]
fn build_challenge_routine() {
    let challenge = build_audit_challenge(&[42u8; 32], 100, 80, AuditTier::Routine);
    assert!(challenge.token_index < 100);
    assert!(challenge.layer_indices.len() >= 10);
    assert!(challenge.layer_indices.len() <= 80);
    // Contiguous prefix
    assert_eq!(
        challenge.layer_indices,
        (0..challenge.layer_indices.len()).collect::<Vec<_>>()
    );
    assert_eq!(challenge.tier, AuditTier::Routine);
}

#[test]
fn build_challenge_full() {
    let challenge = build_audit_challenge(&[42u8; 32], 100, 80, AuditTier::Full);
    assert!(challenge.token_index < 100);
    assert_eq!(challenge.layer_indices.len(), 80);
    assert_eq!(challenge.tier, AuditTier::Full);
}

// ---------------------------------------------------------------------------
// Single-token audit — passing cases
// ---------------------------------------------------------------------------

#[test]
fn audit_full_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Full,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    assert_eq!(report.layers_checked, cfg.n_layers);
    assert_eq!(report.tier, AuditTier::Full);
}

#[test]
fn audit_routine_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Routine,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn audit_single_layer_pass() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices: vec![0],
        tier: AuditTier::Routine,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
    assert_eq!(report.layers_checked, 1);
}

// ---------------------------------------------------------------------------
// Single-token audit — failure cases
// ---------------------------------------------------------------------------

#[test]
fn audit_tampered_layer_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Routine,
    };

    let mut response = build_audit_response(&all_traces, &challenge);
    response.partial_layers[0].ffn_out[0] ^= 0x7F;

    let kv_verifier = StreamingKvVerifier::new();
    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(!report.failures.is_empty());
}

#[test]
fn audit_tampered_silu_detected() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Routine,
    };

    let mut response = build_audit_response(&all_traces, &challenge);
    response.partial_layers[0].h[0] ^= 0x7F;

    let kv_verifier = StreamingKvVerifier::new();
    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("SiLU")));
}

#[test]
fn audit_kv_commitment_mismatch() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Routine,
    };

    let response = build_audit_response(&all_traces, &challenge);

    // Wrong KV root
    let mut kv_verifier = StreamingKvVerifier::new();
    kv_verifier.ingest([0xFFu8; 32]);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Fail);
    let has_kv = report.failures.iter().any(|f| f.contains("KV Merkle proof"));
    assert!(has_kv, "expected KV Merkle proof failure, got: {:?}", report.failures);
}

#[test]
fn audit_per_layer_kv_works_for_routine() {
    // Verify that routine audits (subset of layers) can check KV commitments
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];

    // Only open layer 0 (not all layers)
    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices: vec![0],
        tier: AuditTier::Routine,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

// ---------------------------------------------------------------------------
// Structural mismatch detection
// ---------------------------------------------------------------------------

#[test]
fn audit_token_index_mismatch() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Routine,
    };

    let mut response = build_audit_response(&all_traces, &challenge);
    response.token_index = 99;

    let report = verify_audit(&key, &challenge, &response, &StreamingKvVerifier::new(), None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("token_index")));
}

#[test]
fn audit_layer_count_mismatch() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];

    let challenge_build = AuditChallenge {
        token_index: 0,
        layer_indices: vec![0],
        tier: AuditTier::Routine,
    };
    let response = build_audit_response(&all_traces, &challenge_build);

    // Challenge asks for 2 layers but response has 1
    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices: vec![0, 1],
        tier: AuditTier::Routine,
    };

    let report = verify_audit(&key, &challenge, &response, &StreamingKvVerifier::new(), None);
    assert_eq!(report.verdict, Verdict::Fail);
}

#[test]
fn audit_layer_indices_mismatch() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];

    let challenge_build = AuditChallenge {
        token_index: 0,
        layer_indices: vec![0, 1],
        tier: AuditTier::Routine,
    };
    let response = build_audit_response(&all_traces, &challenge_build);

    // Challenge has different order
    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices: vec![1, 0],
        tier: AuditTier::Routine,
    };

    let report = verify_audit(&key, &challenge, &response, &StreamingKvVerifier::new(), None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(report.failures.iter().any(|f| f.contains("layer_indices")));
}

// ---------------------------------------------------------------------------
// Cross-layer chain
// ---------------------------------------------------------------------------

#[test]
fn audit_cross_layer_chain_tampered() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Routine,
    };

    let mut response = build_audit_response(&all_traces, &challenge);
    response.partial_layers[1].x_attn[0] ^= 0x7F;

    let report = verify_audit(&key, &challenge, &response, &StreamingKvVerifier::new(), None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert!(!report.failures.is_empty());
}

// ---------------------------------------------------------------------------
// Multi-token autoregressive audit
// ---------------------------------------------------------------------------

#[test]
fn audit_autoregressive_token0_full() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = forward_pass_autoregressive_level_c(&cfg, &model, &input, 4);
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Full,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn audit_autoregressive_token2_full() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = forward_pass_autoregressive_level_c(&cfg, &model, &input, 4);
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 2,
        layer_indices,
        tier: AuditTier::Full,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn audit_autoregressive_token3_routine() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = forward_pass_autoregressive_level_c(&cfg, &model, &input, 4);

    // Routine: toy has 2 layers → both selected
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();
    let challenge = AuditChallenge {
        token_index: 3,
        layer_indices,
        tier: AuditTier::Routine,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn audit_autoregressive_single_layer() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = forward_pass_autoregressive_level_c(&cfg, &model, &input, 4);

    // Open only layer 1 of token 2
    let challenge = AuditChallenge {
        token_index: 2,
        layer_indices: vec![1],
        tier: AuditTier::Routine,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    assert_eq!(report.verdict, Verdict::Pass, "failures: {:?}", report.failures);
}

#[test]
fn audit_autoregressive_attention_replay_detects_tampered_kv() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = forward_pass_autoregressive_level_c(&cfg, &model, &input, 4);
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 2,
        layer_indices,
        tier: AuditTier::Routine,
    };

    let mut response = build_audit_response(&all_traces, &challenge);
    // Tamper KV prefix for layer 0: corrupt the token's own K entry.
    // This guarantees detection via KV self-consistency check.
    let pos = challenge.token_index as usize;
    if pos < response.kv_k_prefix[0].len() {
        response.kv_k_prefix[0][pos][0] ^= 0x7F;
    }

    // No KV verifier (focus on attention replay / self-consistency detection)
    let report = verify_audit(&key, &challenge, &response, &StreamingKvVerifier::new(), None);
    assert_eq!(report.verdict, Verdict::Fail);
    // Should detect KV self-consistency failure and/or attention replay mismatch
    assert!(!report.failures.is_empty());
}

#[test]
fn audit_autoregressive_kv_commitment_all_tokens() {
    // Verify KV commitments work for every token in a 4-token sequence
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = forward_pass_autoregressive_level_c(&cfg, &model, &input, 4);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    for t in 0..4u32 {
        let challenge = AuditChallenge {
            token_index: t,
            layer_indices: layer_indices.clone(),
            tier: AuditTier::Full,
        };

        let response = build_audit_response(&all_traces, &challenge);
        let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
        assert_eq!(
            report.verdict,
            Verdict::Pass,
            "token {} failed: {:?}",
            t,
            report.failures
        );
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

#[test]
fn audit_report_display() {
    let (cfg, model, key) = setup();
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| i as i8).collect();
    let all_traces = vec![forward_pass_level_c(&cfg, &model, &input)];
    let layer_indices: Vec<usize> = (0..cfg.n_layers).collect();

    let challenge = AuditChallenge {
        token_index: 0,
        layer_indices,
        tier: AuditTier::Full,
    };

    let response = build_audit_response(&all_traces, &challenge);
    let kv_verifier = build_streaming_kv_verifier(&all_traces);

    let report = verify_audit(&key, &challenge, &response, &kv_verifier, None);
    let display = format!("{}", report);
    assert!(display.contains("AUDIT PASS"));
    assert!(display.contains("Full"));
}
