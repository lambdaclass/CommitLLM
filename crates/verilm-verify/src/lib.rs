//! Verification library for the verified-inference protocol.
//!
//! Provides single-trace and batch verification for the reference path
//! (toy model / LUT-based SiLU / exact i32 accumulators). For W8A8
//! production verification, see the W8A8 verification module.
//!
//! # Verification levels
//!
//! - **Level A**: Exact linear verification (Freivalds + SiLU + chain + Merkle).
//! - **Level B**: Level A + margin certificates bound to the opened trace via
//!   final hidden state and verifier-side logit recomputation.
//! - **Level C**: Level B + attention replay verification + prefix-KV chain binding
//!   (self-consistency always; cross-token consistency when co-opened).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use verilm_core::attention::{AttentionToleranceConfig, compare_attention_output, replay_attention_reference};
use verilm_core::constants::MatrixType;
use verilm_core::margin;
use verilm_core::streaming;
use verilm_core::types::{
    AuditChallenge, AuditResponse, AuditTier, BatchProof, DeploymentManifest, TokenTrace,
    VerifierKey,
};
use verilm_core::{freivalds, merkle};

/// Verification level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationLevel {
    /// Exact linear checks only (Freivalds + SiLU + chain + Merkle).
    A,
    /// Level A + margin certificates for token-generation integrity.
    B,
    /// Level B + attention replay from provided KV snapshot.
    C,
}

pub use verilm_core::requantize;

/// Structured verification result.
#[derive(Debug)]
pub struct VerifyReport {
    pub verdict: Verdict,
    pub model_name: String,
    pub n_layers: usize,
    pub checks_run: usize,
    pub checks_passed: usize,
    pub failures: Vec<String>,
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
}

impl std::fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => write!(
                f,
                "PASS: {}/{} checks passed ({} layers, {:.1}ms)",
                self.checks_passed,
                self.checks_run,
                self.n_layers,
                self.duration.as_secs_f64() * 1000.0
            ),
            Verdict::Fail => {
                writeln!(
                    f,
                    "FAIL: {}/{} checks passed ({} failures)",
                    self.checks_passed,
                    self.checks_run,
                    self.failures.len()
                )?;
                for fail in &self.failures {
                    writeln!(f, "  {}", fail)?;
                }
                Ok(())
            }
        }
    }
}

/// Verify a margin certificate on a token trace (Level B check).
///
/// Requires:
/// - `trace.margin_cert` is `Some`
/// - `key.wo_norms` has one entry per layer
///
/// Returns failure descriptions (empty = margin-certified).
pub fn verify_margin(
    key: &VerifierKey,
    trace: &TokenTrace,
    epsilon: f32,
) -> Vec<String> {
    let mut failures = Vec::new();

    let cert = match &trace.margin_cert {
        Some(c) => c,
        None => {
            failures.push(format!(
                "token {}: missing margin certificate",
                trace.token_index
            ));
            return failures;
        }
    };

    if cert.token_index != trace.token_index {
        failures.push(format!(
            "token {}: margin cert token_index={} doesn't match trace token_index={}",
            trace.token_index, cert.token_index, trace.token_index
        ));
        return failures;
    }

    // Logit-to-trace binding: verify final_hidden and recompute logits.
    let final_hidden = match &trace.final_hidden {
        Some(fh) => fh,
        None => {
            failures.push(format!(
                "token {}: missing final_hidden for Level B verification",
                trace.token_index
            ));
            return failures;
        }
    };

    // Verify final_hidden is consistent with the trace's last layer output.
    let expected_fh = requantize(&trace.layers.last().unwrap().ffn_out);
    if *final_hidden != expected_fh {
        failures.push(format!(
            "token {}: final_hidden doesn't match requantize(last_layer.ffn_out)",
            trace.token_index
        ));
        return failures;
    }

    // If key has lm_head, recompute logits and verify against cert.
    if let Some(lm_head) = &key.lm_head {
        let vocab_size = key.config.vocab_size;
        let hidden_dim = key.config.hidden_dim;
        if lm_head.len() != vocab_size * hidden_dim {
            failures.push(format!(
                "token {}: lm_head size {} != vocab_size({}) * hidden_dim({})",
                trace.token_index, lm_head.len(), vocab_size, hidden_dim
            ));
            return failures;
        }
        // Recompute logits: lm_head @ final_hidden
        let recomputed_logits: Vec<f32> = (0..vocab_size)
            .map(|r| {
                (0..hidden_dim)
                    .map(|c| lm_head[r * hidden_dim + c] as i32 * final_hidden[c] as i32)
                    .sum::<i32>() as f32
            })
            .collect();
        // Compare against cert logits
        if cert.logits.len() != recomputed_logits.len() {
            failures.push(format!(
                "token {}: cert logits length {} != recomputed {}",
                trace.token_index, cert.logits.len(), recomputed_logits.len()
            ));
            return failures;
        }
        for (i, (&cert_l, &recomp_l)) in cert.logits.iter().zip(recomputed_logits.iter()).enumerate() {
            if cert_l != recomp_l {
                failures.push(format!(
                    "token {}: logit mismatch at index {}: cert={} recomputed={}",
                    trace.token_index, i, cert_l, recomp_l
                ));
                return failures;
            }
        }
    }

    if key.wo_norms.len() != key.config.n_layers {
        failures.push(format!(
            "token {}: key has {} wo_norms but {} layers",
            trace.token_index,
            key.wo_norms.len(),
            key.config.n_layers
        ));
        return failures;
    }

    // Use max wo_norm across layers (conservative bound).
    let max_wo_norm = key
        .wo_norms
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);

    let max_v_norm = key.max_v_norm;
    if max_v_norm <= 0.0 {
        failures.push(format!(
            "token {}: key has invalid max_v_norm={}",
            trace.token_index, max_v_norm
        ));
        return failures;
    }

    let result = margin::verify_margin_certificate(
        cert,
        epsilon,
        key.config.n_layers,
        max_v_norm,
        max_wo_norm,
    );

    for f in &result.failures {
        failures.push(format!("token {}: margin: {}", trace.token_index, f));
    }

    if !result.certified && result.failures.is_empty() {
        failures.push(format!(
            "token {}: margin not certified (delta={:.4} <= 2*B={:.4})",
            trace.token_index,
            result.delta,
            2.0 * result.perturbation_bound,
        ));
    }

    failures
}

/// Verify sampling honesty for a token trace.
///
/// Recomputes logits from `lm_head @ final_hidden`, replays the canonical
/// deterministic sampler with the given decode parameters and seed, and
/// checks that the result matches `trace.token_id`.
///
/// Requires:
/// - `key.lm_head` is `Some`
/// - `trace.final_hidden` is `Some`
/// - `trace.token_id` is `Some`
///
/// Returns failure descriptions (empty = sampling verified).
pub fn verify_sampling(
    key: &VerifierKey,
    trace: &TokenTrace,
    params: &verilm_core::sampling::DecodeParams,
    batch_seed: &[u8; 32],
) -> Vec<String> {
    let mut failures = Vec::new();

    let token_id = match trace.token_id {
        Some(tid) => tid,
        None => {
            failures.push(format!(
                "token {}: missing token_id for sampling verification",
                trace.token_index
            ));
            return failures;
        }
    };

    let final_hidden = match &trace.final_hidden {
        Some(fh) => fh,
        None => {
            failures.push(format!(
                "token {}: missing final_hidden for sampling verification",
                trace.token_index
            ));
            return failures;
        }
    };

    let lm_head = match &key.lm_head {
        Some(lm) => lm,
        None => {
            failures.push(format!(
                "token {}: verifier key missing lm_head for sampling verification",
                trace.token_index
            ));
            return failures;
        }
    };

    // Verify final_hidden is consistent with the trace's last layer output
    let expected_fh = requantize(&trace.layers.last().unwrap().ffn_out);
    if *final_hidden != expected_fh {
        failures.push(format!(
            "token {}: final_hidden doesn't match requantize(last_layer.ffn_out)",
            trace.token_index
        ));
        return failures;
    }

    let replayed = verilm_core::sampling::replay_sampling(
        lm_head,
        final_hidden,
        key.config.vocab_size,
        key.config.hidden_dim,
        params,
        batch_seed,
        trace.token_index,
    );

    if replayed != token_id {
        failures.push(format!(
            "token {}: sampling replay mismatch: replayed={} but trace claims token_id={}",
            trace.token_index, replayed, token_id
        ));
    }

    failures
}

/// Verify attention replay for a token trace (Level C check).
///
/// Replays GQA attention from the trace's requantized Q and KV cache,
/// compares against the claimed `a` vector per layer.
/// Also performs self-consistency check: the token's own KV cache entry
/// (at position `token_index`) must match `requantize(k)` / `requantize(v)`.
/// Returns failure descriptions (empty = attention verified).
pub fn verify_attention(
    key: &VerifierKey,
    trace: &TokenTrace,
    tolerance: &AttentionToleranceConfig,
) -> Vec<String> {
    let mut failures = Vec::new();
    let pos = trace.token_index as usize;

    for (layer_idx, lt) in trace.layers.iter().enumerate() {
        if lt.kv_cache_k.is_empty() || lt.kv_cache_v.is_empty() {
            failures.push(format!(
                "layer {}: missing KV cache for Level C attention replay",
                layer_idx
            ));
            continue;
        }

        // Self-consistency: own KV cache entry must match requantize(k/v)
        if pos < lt.kv_cache_k.len() {
            let expected_k = requantize(&lt.k);
            if lt.kv_cache_k[pos] != expected_k {
                failures.push(format!(
                    "layer {}: KV self-consistency failed: kv_cache_k[{}] != requantize(k)",
                    layer_idx, pos
                ));
            }
        }
        if pos < lt.kv_cache_v.len() {
            let expected_v = requantize(&lt.v);
            if lt.kv_cache_v[pos] != expected_v {
                failures.push(format!(
                    "layer {}: KV self-consistency failed: kv_cache_v[{}] != requantize(v)",
                    layer_idx, pos
                ));
            }
        }

        // Requantize Q to i8 for the replay
        let q_i8: Vec<i8> = lt.q.iter().map(|&v| v.clamp(-128, 127) as i8).collect();

        let replayed = replay_attention_reference(
            &q_i8,
            &lt.kv_cache_k,
            &lt.kv_cache_v,
            &key.config,
        );

        if let Some(max_diff) = compare_attention_output(&lt.a, &replayed, tolerance) {
            failures.push(format!(
                "layer {}: attention replay mismatch (max_diff={}, tolerance={})",
                layer_idx, max_diff, tolerance.max_abs_diff
            ));
        }
    }

    failures
}

/// Verify that KV cache entries are consistent with requantized K/V projections
/// from opened traces.
///
/// Two levels of checking:
/// - **Self-consistency**: Each opened token's own KV entry (at its position in
///   the cache) must match `requantize(layers[l].k)` / `requantize(layers[l].v)`
///   from its own trace.
/// - **Cross-token consistency**: When token T has a KV cache entry for prior
///   token P, and P is also opened, the entry must match P's requantized K/V.
///
/// Only opened tokens can be cross-checked. Unopened tokens' KV cache entries
/// cannot be verified against their projections — this is inherent to sampled
/// opening, not a soundness gap.
pub fn verify_kv_chain(
    opened_traces: &BTreeMap<u32, &TokenTrace>,
) -> Vec<String> {
    let mut failures = Vec::new();

    for (&pos, &trace) in opened_traces {
        let p = pos as usize;

        for (layer_idx, lt) in trace.layers.iter().enumerate() {
            if lt.kv_cache_k.is_empty() || lt.kv_cache_v.is_empty() {
                continue;
            }

            // Self-consistency
            if p < lt.kv_cache_k.len() {
                let expected_k = requantize(&lt.k);
                if lt.kv_cache_k[p] != expected_k {
                    failures.push(format!(
                        "token {} layer {}: KV self-consistency failed: kv_cache_k[{}] != requantize(k)",
                        pos, layer_idx, p
                    ));
                }
            }
            if p < lt.kv_cache_v.len() {
                let expected_v = requantize(&lt.v);
                if lt.kv_cache_v[p] != expected_v {
                    failures.push(format!(
                        "token {} layer {}: KV self-consistency failed: kv_cache_v[{}] != requantize(v)",
                        pos, layer_idx, p
                    ));
                }
            }

            // Cross-token consistency
            for (&other_pos, &other_trace) in opened_traces {
                if other_pos >= pos {
                    break;
                }
                let q = other_pos as usize;

                if q < lt.kv_cache_k.len() && layer_idx < other_trace.layers.len() {
                    let expected_k = requantize(&other_trace.layers[layer_idx].k);
                    if lt.kv_cache_k[q] != expected_k {
                        failures.push(format!(
                            "token {} layer {}: cross-token KV mismatch: kv_cache_k[{}] != token {}'s requantize(k)",
                            pos, layer_idx, q, other_pos
                        ));
                    }
                }

                if q < lt.kv_cache_v.len() && layer_idx < other_trace.layers.len() {
                    let expected_v = requantize(&other_trace.layers[layer_idx].v);
                    if lt.kv_cache_v[q] != expected_v {
                        failures.push(format!(
                            "token {} layer {}: cross-token KV mismatch: kv_cache_v[{}] != token {}'s requantize(v)",
                            pos, layer_idx, q, other_pos
                        ));
                    }
                }
            }
        }
    }

    failures
}

/// Verify KV provenance chain hashes for opened traces.
///
/// For each opened token that has `prev_kv_hash`, recomputes the KV chain hash
/// from `requantize(k)` and `requantize(v)` across all layers, and checks:
/// 1. The recomputed hash matches what would be committed (self-consistency).
/// 2. For consecutive opened tokens, the later token's `prev_kv_hash` matches
///    the earlier token's recomputed KV chain hash (cross-token chain).
///
/// Returns failure descriptions (empty = pass).
pub fn verify_kv_provenance_chain(
    opened_traces: &BTreeMap<u32, &TokenTrace>,
) -> Vec<String> {
    let mut failures = Vec::new();

    // Recompute KV chain hash for each opened token
    let mut kv_hashes: BTreeMap<u32, [u8; 32]> = BTreeMap::new();

    for (&pos, &trace) in opened_traces {
        if let Some(claimed_prev) = trace.prev_kv_hash {
            let k_per_layer: Vec<Vec<i8>> = trace.layers.iter()
                .map(|lt| requantize(&lt.k))
                .collect();
            let v_per_layer: Vec<Vec<i8>> = trace.layers.iter()
                .map(|lt| requantize(&lt.v))
                .collect();
            let kv = merkle::kv_chain_hash(&claimed_prev, &k_per_layer, &v_per_layer, pos);
            kv_hashes.insert(pos, kv);

            // Cross-check: if previous token is also opened, claimed prev must match
            if pos > 0 {
                if let Some(&computed_prev) = kv_hashes.get(&(pos - 1)) {
                    if claimed_prev != computed_prev {
                        failures.push(format!(
                            "token {}: prev_kv_hash mismatch — claimed != recomputed from token {}",
                            pos, pos - 1
                        ));
                    }
                }
            }
        }
    }

    failures
}

/// Verify that a batch commitment's manifest hash matches the expected deployment manifest.
/// Returns failure descriptions (empty = pass).
pub fn verify_manifest(
    commitment: &verilm_core::types::BatchCommitment,
    expected_manifest: &DeploymentManifest,
) -> Vec<String> {
    let computed = merkle::hash_manifest(expected_manifest);
    match commitment.manifest_hash {
        None => vec!["commitment missing manifest_hash".into()],
        Some(h) if h != computed => vec!["manifest_hash mismatch".into()],
        Some(_) => Vec::new(),
    }
}

/// Result of verifying a weight-chain hash against an expected value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightHashResult {
    /// Key hash matches expected.
    Match,
    /// Key hash does not match expected.
    Mismatch {
        key_hash: [u8; 32],
        expected: [u8; 32],
    },
    /// Key has no weight hash (legacy key) but an expected hash was provided.
    LegacyKeyNoHash,
    /// No expected hash was provided — skipped.
    Skipped,
}

/// Check whether the verifier key's weight hash matches an externally published
/// expected hash. This is the binding that closes the trust gap between
/// "the INT8 computation was correct" and "it used the published checkpoint."
///
/// Returns [`WeightHashResult::Skipped`] if `expected` is `None`.
pub fn verify_weight_hash(key: &VerifierKey, expected: Option<&[u8; 32]>) -> WeightHashResult {
    let Some(expected) = expected else {
        return WeightHashResult::Skipped;
    };

    match key.weight_hash {
        Some(actual) if actual == *expected => WeightHashResult::Match,
        Some(actual) => WeightHashResult::Mismatch {
            key_hash: actual,
            expected: *expected,
        },
        None => WeightHashResult::LegacyKeyNoHash,
    }
}

/// Verify a single token trace against a verifier key.
/// Returns a list of failure descriptions (empty = pass).
pub fn verify_one(key: &VerifierKey, trace: &TokenTrace) -> Vec<String> {
    let mut failures = Vec::new();

    // Step 1: Merkle commitment binding
    let leaf_data = bincode::serialize(&trace.layers).expect("serialize layers");
    let leaf_hash = merkle::trace_leaf_hash(
        &leaf_data,
        trace.final_hidden.as_deref(),
    );
    if !merkle::verify(&trace.merkle_root, &leaf_hash, &trace.merkle_proof) {
        failures.push("Merkle proof verification failed".into());
    }

    for (layer_idx, lt) in trace.layers.iter().enumerate() {
        // Step 2: Freivalds checks for each matrix type
        let checks: [(MatrixType, &[i8], &[i32]); 7] = [
            (MatrixType::Wq, &lt.x_attn, &lt.q),
            (MatrixType::Wk, &lt.x_attn, &lt.k),
            (MatrixType::Wv, &lt.x_attn, &lt.v),
            (MatrixType::Wo, &lt.a, &lt.attn_out),
            (MatrixType::Wg, &lt.x_ffn, &lt.g),
            (MatrixType::Wu, &lt.x_ffn, &lt.u),
            (MatrixType::Wd, &lt.h, &lt.ffn_out),
        ];

        for (mt, input, output) in &checks {
            let v = key.v_for(layer_idx, *mt);
            let r = key.r_for(*mt);
            if !freivalds::check(v, input, r, output) {
                failures.push(format!("layer {} {:?}: Freivalds check failed", layer_idx, mt));
            }
        }

        // Step 3: SiLU (LUT-based, unit scale)
        let expected_h = verilm_core::silu::compute_h_unit_scale(&lt.g, &lt.u);
        if lt.h != expected_h {
            failures.push(format!("layer {}: SiLU check failed", layer_idx));
        }

        // Step 4a: Intra-layer chain
        if lt.x_ffn != requantize(&lt.attn_out) {
            failures.push(format!("layer {}: chain x_ffn mismatch", layer_idx));
        }

        // Step 4b: Cross-layer chain
        if layer_idx + 1 < trace.layers.len() {
            if trace.layers[layer_idx + 1].x_attn != requantize(&lt.ffn_out) {
                failures.push(format!(
                    "layer {}->{}: chain x_attn mismatch",
                    layer_idx,
                    layer_idx + 1
                ));
            }
        }
    }

    failures
}

/// Verify a single token trace, returning a structured report.
pub fn verify_trace(key: &VerifierKey, trace: &TokenTrace) -> VerifyReport {
    let start = Instant::now();
    let failures = verify_one(key, trace);
    let duration = start.elapsed();

    // Each layer has 7 Freivalds + 1 SiLU + 1 chain = 9 checks, plus Merkle = 1
    let checks_run = key.config.n_layers * 9 + 1;
    let checks_passed = checks_run - failures.len();

    VerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        model_name: key.config.name.clone(),
        n_layers: key.config.n_layers,
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

/// Verify a single token trace at a given verification level.
///
/// `epsilon` controls the margin certificate softmax tolerance (Level B/C).
/// `attn_tolerance` controls the attention replay tolerance (Level C only).
/// Pass `None` for `attn_tolerance` to use the default (exact match, tolerance=0).
pub fn verify_trace_at_level(
    key: &VerifierKey,
    trace: &TokenTrace,
    level: VerificationLevel,
    epsilon: f32,
) -> VerifyReport {
    verify_trace_at_level_with_attn_tolerance(key, trace, level, epsilon, None)
}

/// Verify a single token trace with explicit attention tolerance configuration.
pub fn verify_trace_at_level_with_attn_tolerance(
    key: &VerifierKey,
    trace: &TokenTrace,
    level: VerificationLevel,
    epsilon: f32,
    attn_tolerance: Option<AttentionToleranceConfig>,
) -> VerifyReport {
    let start = Instant::now();
    let mut failures = verify_one(key, trace);

    let mut extra_checks = 0;
    if level == VerificationLevel::B || level == VerificationLevel::C {
        extra_checks += 1;
        let margin_failures = verify_margin(key, trace, epsilon);
        failures.extend(margin_failures);
    }
    if level == VerificationLevel::C {
        extra_checks += key.config.n_layers; // one attention check per layer
        let attn_tol = attn_tolerance.unwrap_or_default();
        let attn_failures = verify_attention(key, trace, &attn_tol);
        failures.extend(attn_failures);
    }

    let duration = start.elapsed();
    let checks_run = key.config.n_layers * 9 + 1 + extra_checks;
    let checks_passed = checks_run - failures.len();

    VerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        model_name: key.config.name.clone(),
        n_layers: key.config.n_layers,
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

/// Result from batch verification.
#[derive(Debug)]
pub struct BatchVerifyReport {
    pub verdict: Verdict,
    pub model_name: String,
    pub n_layers: usize,
    pub n_tokens: u32,
    pub n_challenged: usize,
    pub tokens_passed: usize,
    pub challenges: Vec<u32>,
    pub failures: Vec<String>,
    pub duration: Duration,
}

impl std::fmt::Display for BatchVerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => write!(
                f,
                "PASS: {}/{} challenged tokens verified ({} layers each, IO+chain, {:.1}ms)",
                self.tokens_passed,
                self.n_challenged,
                self.n_layers,
                self.duration.as_secs_f64() * 1000.0
            ),
            Verdict::Fail => {
                writeln!(
                    f,
                    "FAIL: {} failures across batch ({}/{} tokens passed)",
                    self.failures.len(),
                    self.tokens_passed,
                    self.n_challenged
                )?;
                for fail in &self.failures {
                    writeln!(f, "  {}", fail)?;
                }
                Ok(())
            }
        }
    }
}

/// Verify a batch proof with challenge derivation and cross-token chain checks.
pub fn verify_batch(
    key: &VerifierKey,
    proof: &BatchProof,
    seed: [u8; 32],
    challenge_k: u32,
) -> BatchVerifyReport {
    verify_batch_with_policy(key, proof, seed, challenge_k, &Default::default())
}

/// Verify a batch proof with explicit verification policy.
///
/// The policy can enforce minimum commitment version, expected prompt hash,
/// and expected manifest hash — closing downgrade, replay, and runtime
/// binding gaps that the default permissive mode leaves open.
pub fn verify_batch_with_policy(
    key: &VerifierKey,
    proof: &BatchProof,
    seed: [u8; 32],
    challenge_k: u32,
    policy: &verilm_core::types::VerificationPolicy,
) -> BatchVerifyReport {
    let start = Instant::now();
    let mut failures = Vec::new();
    let mut tokens_passed = 0;

    let challenges = merkle::derive_challenges(
        &proof.commitment.merkle_root,
        &seed,
        proof.commitment.n_tokens,
        challenge_k,
    );

    let mut opened_by_index: BTreeMap<u32, &TokenTrace> = BTreeMap::new();

    // Policy: minimum version enforcement
    if let Some(min_ver) = policy.min_version {
        if (proof.commitment.version as u32) < (min_ver as u32) {
            failures.push(format!(
                "commitment version {:?} below minimum required {:?}",
                proof.commitment.version, min_ver
            ));
        }
    }

    // Policy: expected prompt hash verification
    if let Some(expected_prompt) = policy.expected_prompt_hash {
        match proof.commitment.prompt_hash {
            Some(actual) if actual != expected_prompt => {
                failures.push("prompt_hash mismatch: commitment prompt_hash != expected".into());
            }
            None => {
                failures.push("policy requires prompt_hash but commitment has None".into());
            }
            _ => {} // matches
        }
    }

    // Policy: expected manifest hash verification
    if let Some(expected_manifest) = policy.expected_manifest_hash {
        match proof.commitment.manifest_hash {
            Some(actual) if actual != expected_manifest => {
                failures.push("manifest_hash mismatch: commitment manifest_hash != expected".into());
            }
            None => {
                failures.push("policy requires manifest_hash but commitment has None".into());
            }
            _ => {} // matches
        }
    }

    // Version consistency: V2/V3 commitments require token_id on every trace.
    if proof.commitment.version == verilm_core::types::CommitmentVersion::V2
        || proof.commitment.version == verilm_core::types::CommitmentVersion::V3
    {
        for trace in &proof.traces {
            if trace.token_id.is_none() {
                failures.push(format!(
                    "token {}: {:?} commitment requires token_id but trace has None",
                    trace.token_index, proof.commitment.version
                ));
            }
        }
    }

    // V3: seed commitment verification — hash(revealed_seed) must equal commitment
    if proof.commitment.version == verilm_core::types::CommitmentVersion::V3 {
        match (proof.commitment.seed_commitment, proof.revealed_seed) {
            (Some(expected), Some(revealed)) => {
                let computed = merkle::hash_seed(&revealed);
                if computed != expected {
                    failures.push("seed commitment mismatch: hash(revealed_seed) != commitment.seed_commitment".into());
                }
            }
            (Some(_), None) => {
                failures.push("V3 commitment has seed_commitment but proof is missing revealed_seed".into());
            }
            (None, _) => {
                failures.push("V3 commitment is missing seed_commitment".into());
            }
        }

        // V3: prompt hash must be present
        if proof.commitment.prompt_hash.is_none() {
            failures.push("V3 commitment is missing prompt_hash".into());
        }

        // V3: all traces must have prev_io_hash for chain reconstruction
        for trace in &proof.traces {
            if trace.prev_io_hash.is_none() {
                failures.push(format!(
                    "token {}: V3 trace is missing prev_io_hash for chain verification",
                    trace.token_index
                ));
            }
        }
    }

    // For V3, build IO hash map using prev_io_hash from each trace.
    // This allows chain verification even for non-consecutive challenge sets.
    // When consecutive tokens are both opened, we cross-check that the
    // later token's prev_io_hash matches the recomputed IO hash of the earlier.
    let mut io_hash_map: BTreeMap<u32, [u8; 32]> = BTreeMap::new();
    if proof.commitment.version == verilm_core::types::CommitmentVersion::V3 {
        let mut sorted: Vec<&TokenTrace> = proof.traces.iter().collect();
        sorted.sort_by_key(|t| t.token_index);

        for trace in &sorted {
            let first_input = &trace.layers[0].x_attn;
            let last_output = &trace.layers.last().unwrap().ffn_out;

            if let (Some(tid), Some(claimed_prev)) = (trace.token_id, trace.prev_io_hash) {
                // Cross-check: if we already computed the previous token's IO hash,
                // the claimed prev_io_hash must match.
                if trace.token_index > 0 {
                    if let Some(&computed_prev) = io_hash_map.get(&(trace.token_index - 1)) {
                        if claimed_prev != computed_prev {
                            failures.push(format!(
                                "token {}: prev_io_hash mismatch — claimed != recomputed from token {}",
                                trace.token_index, trace.token_index - 1
                            ));
                        }
                    }
                }

                let binding = merkle::IoHashBinding::Chained {
                    token_id: tid,
                    prev_io_hash: claimed_prev,
                };
                let io = merkle::io_hash(first_input, last_output, &binding);
                io_hash_map.insert(trace.token_index, io);
            }
        }
    }

    for trace in &proof.traces {
        // Check this trace is in the challenge set
        if !challenges.contains(&trace.token_index) {
            failures.push(format!(
                "token {} not in challenge set",
                trace.token_index
            ));
            continue;
        }

        // Check leaf_index matches token_index
        if trace.merkle_proof.leaf_index != trace.token_index {
            failures.push(format!(
                "token {}: merkle_proof.leaf_index={} mismatch",
                trace.token_index, trace.merkle_proof.leaf_index
            ));
            continue;
        }

        // Check merkle_root matches commitment
        if trace.merkle_root != proof.commitment.merkle_root {
            failures.push(format!(
                "token {}: merkle_root doesn't match commitment",
                trace.token_index
            ));
            continue;
        }

        // IO proof verification
        let first_input = &trace.layers[0].x_attn;
        let last_output = &trace.layers.last().unwrap().ffn_out;

        let io = if proof.commitment.version == verilm_core::types::CommitmentVersion::V3 {
            // V3: use chained hash from pre-computation (uses trace's prev_io_hash)
            if let Some(&precomputed) = io_hash_map.get(&trace.token_index) {
                precomputed
            } else {
                // Fallback: prev_io_hash was missing; use V2-style.
                // This will fail the IO tree proof for a V3 commitment,
                // which is the correct behavior.
                merkle::io_hash(first_input, last_output, &merkle::io_binding(trace.token_id, None))
            }
        } else {
            // V1/V2: straightforward
            merkle::io_hash(first_input, last_output, &merkle::io_binding(trace.token_id, None))
        };

        if !merkle::verify(&proof.commitment.io_root, &io, &trace.io_proof) {
            failures.push(format!(
                "token {}: IO proof verification failed",
                trace.token_index
            ));
        }

        // Full single-token verification
        let token_failures = verify_one(key, trace);
        if token_failures.is_empty() {
            tokens_passed += 1;
        } else {
            for f in token_failures {
                failures.push(format!("token {}: {}", trace.token_index, f));
            }
        }

        opened_by_index.insert(trace.token_index, trace);
    }

    // Cross-token chain: consecutive opened tokens must chain
    for (&idx, &trace) in &opened_by_index {
        if idx == 0 {
            continue;
        }
        if let Some(&prev_trace) = opened_by_index.get(&(idx - 1)) {
            let expected_input = requantize(&prev_trace.layers.last().unwrap().ffn_out);
            if trace.layers[0].x_attn != expected_input {
                failures.push(format!(
                    "token {}->{}: cross-token chain mismatch",
                    idx - 1,
                    idx
                ));
            }
        }
    }

    // KV provenance chain: verify prev_kv_hash cross-checks for consecutive opened tokens
    if proof.commitment.kv_chain_root.is_some() {
        let kv_failures = verify_kv_provenance_chain(&opened_by_index);
        failures.extend(kv_failures);
    }

    let duration = start.elapsed();

    BatchVerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        model_name: key.config.name.clone(),
        n_layers: key.config.n_layers,
        n_tokens: proof.commitment.n_tokens,
        n_challenged: challenges.len(),
        tokens_passed,
        challenges,
        failures,
        duration,
    }
}

// ---------------------------------------------------------------------------
// Stratified audit verification
// ---------------------------------------------------------------------------

/// Derive deterministic layer indices for a routine audit.
///
/// Uses `SHA256(seed || token_index || counter)` to pick `k` unique layer
/// indices from `0..n_layers`. For a full audit, just returns `0..n_layers`.
pub fn derive_audit_layers(
    seed: &[u8; 32],
    token_index: u32,
    n_layers: usize,
    tier: AuditTier,
) -> Vec<usize> {
    match tier {
        AuditTier::Full => (0..n_layers).collect(),
        AuditTier::Routine => {
            use sha2::{Digest, Sha256};
            use std::collections::BTreeSet;

            let k = 10usize.min(n_layers); // 10 layers or all if fewer
            let mut indices = BTreeSet::new();
            let mut counter: u32 = 0;
            while indices.len() < k {
                let mut hasher = Sha256::new();
                hasher.update(seed);
                hasher.update(token_index.to_le_bytes());
                hasher.update(counter.to_le_bytes());
                let hash: [u8; 32] = hasher.finalize().into();
                let idx = u32::from_le_bytes(hash[..4].try_into().unwrap()) as usize % n_layers;
                indices.insert(idx);
                counter += 1;
            }
            indices.into_iter().collect()
        }
    }
}

/// Build an `AuditChallenge` from a seed and tier.
///
/// Picks a random token index from `0..n_tokens` and layer indices from
/// `0..n_layers` based on the tier.
pub fn build_audit_challenge(
    seed: &[u8; 32],
    n_tokens: u32,
    n_layers: usize,
    tier: AuditTier,
) -> AuditChallenge {
    use sha2::{Digest, Sha256};

    // Pick token index from seed
    let mut hasher = Sha256::new();
    hasher.update(b"vi-audit-token-v1");
    hasher.update(seed);
    let hash: [u8; 32] = hasher.finalize().into();
    let token_index = u32::from_le_bytes(hash[..4].try_into().unwrap()) % n_tokens;

    let layer_indices = derive_audit_layers(seed, token_index, n_layers, tier);

    AuditChallenge {
        token_index,
        layer_indices,
        tier,
    }
}

/// Result of an audit verification.
#[derive(Debug)]
pub struct AuditVerifyReport {
    pub verdict: Verdict,
    pub tier: AuditTier,
    pub token_index: u32,
    pub layers_checked: usize,
    pub checks_run: usize,
    pub checks_passed: usize,
    pub failures: Vec<String>,
    pub duration: Duration,
}

impl std::fmt::Display for AuditVerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => write!(
                f,
                "AUDIT PASS ({:?}): token {} — {}/{} checks on {} layers ({:.1}ms)",
                self.tier,
                self.token_index,
                self.checks_passed,
                self.checks_run,
                self.layers_checked,
                self.duration.as_secs_f64() * 1000.0
            ),
            Verdict::Fail => {
                writeln!(
                    f,
                    "AUDIT FAIL ({:?}): token {} — {} failures",
                    self.tier, self.token_index, self.failures.len()
                )?;
                for fail in &self.failures {
                    writeln!(f, "  {}", fail)?;
                }
                Ok(())
            }
        }
    }
}

/// Verify an audit response against a challenge and streaming KV commitments.
///
/// Performs on the challenged layers:
/// 1. Freivalds checks (Level A) for each matrix type
/// 2. SiLU check
/// 3. Intra-layer chain (x_ffn == requantize(attn_out))
/// 4. Attention replay from provided KV prefix
/// 5. Streaming KV commitment verification (full audits only)
///
/// Cross-layer chain is checked between consecutive opened layers.
pub fn verify_audit(
    key: &VerifierKey,
    challenge: &AuditChallenge,
    response: &AuditResponse,
    kv_verifier: &streaming::StreamingKvVerifier,
    attn_tolerance: Option<AttentionToleranceConfig>,
) -> AuditVerifyReport {
    let start = Instant::now();
    let mut failures = Vec::new();
    let attn_tol = attn_tolerance.unwrap_or_default();

    // Basic structural checks
    if response.token_index != challenge.token_index {
        failures.push(format!(
            "response token_index {} != challenge token_index {}",
            response.token_index, challenge.token_index
        ));
    }

    if response.layer_indices != challenge.layer_indices {
        failures.push(format!(
            "response layer_indices {:?} != challenge {:?}",
            response.layer_indices, challenge.layer_indices
        ));
    }

    if response.partial_layers.len() != challenge.layer_indices.len() {
        failures.push(format!(
            "response has {} layers but challenge requests {}",
            response.partial_layers.len(),
            challenge.layer_indices.len()
        ));
        return make_audit_report(challenge, &failures, start.elapsed());
    }

    if response.kv_k_prefix.len() != challenge.layer_indices.len()
        || response.kv_v_prefix.len() != challenge.layer_indices.len()
    {
        failures.push("KV prefix count doesn't match layer count".into());
        return make_audit_report(challenge, &failures, start.elapsed());
    }

    let mut checks_run = 0usize;

    // Verify each opened layer
    for (resp_idx, &layer_idx) in challenge.layer_indices.iter().enumerate() {
        let lt = &response.partial_layers[resp_idx];

        // Freivalds checks (7 per layer)
        let checks: [(MatrixType, &[i8], &[i32]); 7] = [
            (MatrixType::Wq, &lt.x_attn, &lt.q),
            (MatrixType::Wk, &lt.x_attn, &lt.k),
            (MatrixType::Wv, &lt.x_attn, &lt.v),
            (MatrixType::Wo, &lt.a, &lt.attn_out),
            (MatrixType::Wg, &lt.x_ffn, &lt.g),
            (MatrixType::Wu, &lt.x_ffn, &lt.u),
            (MatrixType::Wd, &lt.h, &lt.ffn_out),
        ];

        for (mt, input, output) in &checks {
            let v = key.v_for(layer_idx, *mt);
            let r = key.r_for(*mt);
            checks_run += 1;
            if !freivalds::check(v, input, r, output) {
                failures.push(format!(
                    "layer {} {:?}: Freivalds check failed",
                    layer_idx, mt
                ));
            }
        }

        // SiLU check
        checks_run += 1;
        let expected_h = verilm_core::silu::compute_h_unit_scale(&lt.g, &lt.u);
        if lt.h != expected_h {
            failures.push(format!("layer {}: SiLU check failed", layer_idx));
        }

        // Intra-layer chain: x_ffn == requantize(attn_out)
        checks_run += 1;
        if lt.x_ffn != requantize(&lt.attn_out) {
            failures.push(format!("layer {}: chain x_ffn mismatch", layer_idx));
        }

        // Attention replay from KV prefix
        checks_run += 1;
        let kv_k = &response.kv_k_prefix[resp_idx];
        let kv_v = &response.kv_v_prefix[resp_idx];

        if !kv_k.is_empty() && !kv_v.is_empty() {
            // Self-consistency: own KV entry must match requantize(k/v)
            let pos = response.token_index as usize;
            if pos < kv_k.len() {
                let expected_k = requantize(&lt.k);
                if kv_k[pos] != expected_k {
                    failures.push(format!(
                        "layer {}: KV self-consistency failed: kv_k[{}] != requantize(k)",
                        layer_idx, pos
                    ));
                }
            }
            if pos < kv_v.len() {
                let expected_v = requantize(&lt.v);
                if kv_v[pos] != expected_v {
                    failures.push(format!(
                        "layer {}: KV self-consistency failed: kv_v[{}] != requantize(v)",
                        layer_idx, pos
                    ));
                }
            }

            // Replay attention
            let q_i8: Vec<i8> = lt.q.iter().map(|&v| v.clamp(-128, 127) as i8).collect();
            let replayed = replay_attention_reference(&q_i8, kv_k, kv_v, &key.config);
            if let Some(max_diff) = compare_attention_output(&lt.a, &replayed, &attn_tol) {
                failures.push(format!(
                    "layer {}: attention replay mismatch (max_diff={}, tol={})",
                    layer_idx, max_diff, attn_tol.max_abs_diff
                ));
            }
        }
    }

    // Cross-layer chain between consecutive opened layers
    for w in challenge.layer_indices.windows(2) {
        if w[1] == w[0] + 1 {
            let prev_idx = challenge.layer_indices.iter().position(|&l| l == w[0]).unwrap();
            let next_idx = challenge.layer_indices.iter().position(|&l| l == w[1]).unwrap();
            checks_run += 1;
            let expected_input = requantize(&response.partial_layers[prev_idx].ffn_out);
            if response.partial_layers[next_idx].x_attn != expected_input {
                failures.push(format!(
                    "layer {}->{}: cross-layer chain mismatch",
                    w[0], w[1]
                ));
            }
        }
    }

    // Streaming KV commitment verification via per-layer Merkle proofs.
    // Works for both routine (subset) and full (all layers) audits.
    if !kv_verifier.is_empty() && response.kv_layer_proofs.len() == challenge.layer_indices.len() {
        for (resp_idx, &layer_idx) in challenge.layer_indices.iter().enumerate() {
            checks_run += 1;
            let k_i8 = requantize(&response.partial_layers[resp_idx].k);
            let v_i8 = requantize(&response.partial_layers[resp_idx].v);
            let proof = &response.kv_layer_proofs[resp_idx];
            if let Err(msg) = kv_verifier.verify_layer_opening(
                challenge.token_index,
                layer_idx as u32,
                &k_i8,
                &v_i8,
                proof,
            ) {
                failures.push(msg);
            }
        }
    }

    let duration = start.elapsed();
    let checks_passed = checks_run.saturating_sub(failures.len());

    AuditVerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        tier: challenge.tier,
        token_index: challenge.token_index,
        layers_checked: challenge.layer_indices.len(),
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

fn make_audit_report(
    challenge: &AuditChallenge,
    failures: &[String],
    duration: Duration,
) -> AuditVerifyReport {
    AuditVerifyReport {
        verdict: Verdict::Fail,
        tier: challenge.tier,
        token_index: challenge.token_index,
        layers_checked: 0,
        checks_run: 0,
        checks_passed: 0,
        failures: failures.to_vec(),
        duration,
    }
}
