//! Verification library for the verified-inference protocol (V4).
//!
//! Provides key-only Freivalds verification against prover-supplied shell
//! openings. The verifier never recomputes matmuls — it checks the prover's
//! i32 accumulators with precomputed Freivalds keys and verifies bridge
//! consistency by deriving intermediate i8 values from the accumulators.

use std::time::{Duration, Instant};

use verilm_core::constants::MatrixType;
use verilm_core::types::{AuditChallenge, AuditTier, CommitmentVersion, DeploymentManifest, InputSpec, OutputSpec, V4AuditResponse, VerifierKey};
use verilm_core::{freivalds, merkle};

pub use verilm_core::requantize;
pub use verilm_core::types::ShellWeights;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
}

/// Tokenizer abstraction for canonical request→token reconstruction.
///
/// Implementors (typically Python-side with HuggingFace tokenizers) provide
/// a tokenization function that the verifier calls to reconstruct prompt
/// token IDs from raw prompt bytes and the committed `InputSpec`.
///
/// The verifier uses this to independently derive the expected token chain,
/// closing the gap between "caller supplies token IDs" and "verifier
/// reconstructs them from the raw request."
pub trait PromptTokenizer {
    /// Tokenize raw prompt bytes according to the given `InputSpec`.
    ///
    /// Returns the full list of prompt token IDs (including the first token
    /// that will be consumed as embedding input). The `InputSpec` provides
    /// the tokenizer_hash (identifies which vocabulary), bos_eos_policy,
    /// truncation_policy, and special_token_policy that govern preprocessing.
    ///
    /// Errors should be returned as `Err(description)`.
    fn tokenize(&self, prompt: &[u8], input_spec: &InputSpec) -> Result<Vec<u32>, String>;
}

/// Detokenizer abstraction for canonical output text verification.
///
/// Implementors (typically Python-side with HuggingFace tokenizers) decode
/// token IDs back to text under a committed detokenization policy. The
/// verifier uses this to check that the prover's claimed output text
/// matches what the token IDs actually decode to.
pub trait Detokenizer {
    /// Decode token IDs to text under the given detokenization policy.
    ///
    /// `policy` is the committed `detokenization_policy` from the OutputSpec
    /// (e.g. "default", "clean_spaces", "raw", or None).
    ///
    /// Returns the decoded text, or `Err(description)` on failure.
    fn decode(&self, token_ids: &[u32], policy: Option<&str>) -> Result<String, String>;
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

/// Verify a commitment's manifest hash from four individual specs.
///
/// M = H("vi-manifest-v4" || H_input || H_model || H_decode || H_output)
pub fn verify_manifest_specs(
    commitment: &verilm_core::types::BatchCommitment,
    input: &verilm_core::types::InputSpec,
    model: &verilm_core::types::ModelSpec,
    decode: &verilm_core::types::DecodeSpec,
    output: &verilm_core::types::OutputSpec,
) -> Vec<String> {
    let computed = merkle::hash_manifest_composed(
        merkle::hash_input_spec(input),
        merkle::hash_model_spec(model),
        merkle::hash_decode_spec(decode),
        merkle::hash_output_spec(output),
    );
    match commitment.manifest_hash {
        None => vec!["commitment missing manifest_hash".into()],
        Some(h) if h != computed => vec!["manifest_hash mismatch".into()],
        Some(_) => Vec::new(),
    }
}

/// Verify that externally-computed prompt token IDs match the committed token chain.
///
/// The caller (typically Python with a HuggingFace tokenizer) tokenizes the raw
/// prompt using the committed InputSpec (tokenizer, chat template, BOS/EOS policy)
/// and passes the resulting token IDs here. This function checks:
///
/// 1. `expected_prompt_token_ids.len()` matches `response.n_prompt_tokens`
/// 2. The prompt portion of `prefix_token_ids` matches `expected[1..]`
///    (first token is consumed as embedding input, not in the committed array)
///
/// Returns failure descriptions (empty = pass).
pub fn verify_input_tokenization(
    response: &V4AuditResponse,
    expected_prompt_token_ids: &[u32],
) -> Vec<String> {
    let mut failures = Vec::new();

    // Check prompt token count matches.
    if let Some(committed_npt) = response.n_prompt_tokens {
        if expected_prompt_token_ids.len() as u32 != committed_npt {
            failures.push(format!(
                "tokenization produced {} prompt tokens but commitment has n_prompt_tokens={}",
                expected_prompt_token_ids.len(), committed_npt
            ));
            return failures;
        }
    }

    // The committed token_ids array omits the first token (used as embedding input).
    // prefix_token_ids[0..n_prompt-1] are the remaining prompt tokens.
    // After that come generated tokens.
    if expected_prompt_token_ids.is_empty() {
        return failures;
    }

    let n_prompt_in_chain = expected_prompt_token_ids.len() - 1; // first consumed as embedding

    // Collect all chain token IDs: prefix_token_ids + [token_id]
    let all_chain: Vec<u32> = response.prefix_token_ids.iter()
        .copied()
        .chain(std::iter::once(response.token_id))
        .collect();

    if n_prompt_in_chain > all_chain.len() {
        failures.push(format!(
            "prompt has {} tokens in chain but response only has {} total chain entries",
            n_prompt_in_chain, all_chain.len()
        ));
        return failures;
    }

    // Check each prompt token (after the first) against the chain.
    for (i, &expected_tid) in expected_prompt_token_ids[1..].iter().enumerate() {
        if all_chain[i] != expected_tid {
            failures.push(format!(
                "prompt token mismatch at position {}: expected {} but chain has {}",
                i + 1, expected_tid, all_chain[i]
            ));
        }
    }

    failures
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
// ---------------------------------------------------------------------------
// Stratified audit verification
// ---------------------------------------------------------------------------

/// Derive deterministic layer indices for a routine audit.
///
/// Uses `SHA256(challenge_seed || token_index || counter)` to pick `k` unique
/// layer indices from `0..n_layers`. For a full audit, just returns
/// `0..n_layers`.
pub fn derive_audit_layers(
    challenge_seed: &[u8; 32],
    token_index: u32,
    n_layers: usize,
    tier: AuditTier,
) -> Vec<usize> {
    match tier {
        AuditTier::Full => (0..n_layers).collect(),
        AuditTier::Routine => {
            use sha2::{Digest, Sha256};

            // Contiguous prefix 0..=L_max. L_max is derived from the
            // challenge seed so the prover cannot predict the depth.
            // Minimum prefix length: min(10, n_layers).
            let min_prefix = 10usize.min(n_layers);
            let mut hasher = Sha256::new();
            hasher.update(b"vi-audit-prefix-v1");
            hasher.update(challenge_seed);
            hasher.update(token_index.to_le_bytes());
            let hash: [u8; 32] = hasher.finalize().into();
            let raw = u32::from_le_bytes(hash[..4].try_into().unwrap()) as usize;
            let l_max = (raw % n_layers).max(min_prefix.saturating_sub(1));
            (0..=l_max).collect()
        }
    }
}

/// Build an `AuditChallenge` from a verifier-generated challenge seed and tier.
///
/// Picks a random token index from `0..n_tokens` and layer indices from
/// `0..n_layers` based on the tier.
pub fn build_audit_challenge(
    challenge_seed: &[u8; 32],
    n_tokens: u32,
    n_layers: usize,
    tier: AuditTier,
) -> AuditChallenge {
    use sha2::{Digest, Sha256};

    // Pick token index from the verifier's challenge seed.
    let mut hasher = Sha256::new();
    hasher.update(b"vi-audit-token-v1");
    hasher.update(challenge_seed);
    let hash: [u8; 32] = hasher.finalize().into();
    let token_index = u32::from_le_bytes(hash[..4].try_into().unwrap()) % n_tokens;

    let layer_indices = derive_audit_layers(challenge_seed, token_index, n_layers, tier);

    AuditChallenge {
        token_index,
        layer_indices,
        tier,
    }
}
// ---------------------------------------------------------------------------
// V4 retained-state verification
// ---------------------------------------------------------------------------

/// Result from V4 retained-state verification.
#[derive(Debug)]
pub struct V4VerifyReport {
    pub verdict: Verdict,
    pub token_index: u32,
    pub checks_run: usize,
    pub checks_passed: usize,
    pub failures: Vec<String>,
    pub duration: Duration,
}

impl std::fmt::Display for V4VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => write!(
                f,
                "V4 PASS: token {} — {}/{} checks ({:.1}ms)",
                self.token_index,
                self.checks_passed,
                self.checks_run,
                self.duration.as_secs_f64() * 1000.0
            ),
            Verdict::Fail => {
                writeln!(
                    f,
                    "V4 FAIL: token {} — {} failures",
                    self.token_index,
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


/// Verify V4 audit response: structural checks + key-only Freivalds on shell openings.
///
/// This is the protocol verifier — requires only the precomputed VerifierKey,
/// no weights. Checks:
/// 1. Structural: version, seed, prompt, Merkle proofs, IO chain, prefix count
/// 2. Shell opening: Freivalds on each matmul (W_o, W_g, W_u, W_d, optionally QKV)
/// 3. Bridge consistency: verifier derives intermediate i8 values from i32 accumulators
/// 4. Retained-state binding: shell openings must be consistent with committed `a`
/// 5. Input tokenization: when a `PromptTokenizer` is provided, the verifier
///    reconstructs prompt token IDs from raw prompt bytes + `InputSpec` and
///    checks them against the committed chain. Alternatively, pass
///    pre-tokenized IDs via `expected_prompt_token_ids`.
/// 6. Output policy: min_tokens, ignore_eos, and eos_policy enforcement.
pub fn verify_v4(
    key: &VerifierKey,
    response: &V4AuditResponse,
    expected_prompt_token_ids: Option<&[u32]>,
) -> V4VerifyReport {
    let no_tok: Option<&dyn PromptTokenizer> = None;
    let no_detok: Option<&dyn Detokenizer> = None;
    verify_v4_full(key, response, expected_prompt_token_ids, no_tok, no_detok)
}

/// Full verification with optional canonical tokenizer and detokenizer.
///
/// When `tokenizer` is `Some`, the verifier reconstructs prompt token IDs from
/// `response.prompt` + the manifest's `InputSpec`, then checks them against
/// the committed chain. This is the canonical path — the verifier independently
/// derives the expected tokens rather than trusting caller-supplied IDs.
///
/// When `detokenizer` is `Some` and `response.output_text` is present, the
/// verifier decodes the committed generation token IDs and compares against
/// the claimed output text under the committed `detokenization_policy`.
///
/// `expected_prompt_token_ids` is a fallback: used only when no tokenizer is
/// provided or when the response lacks the fields needed for reconstruction.
pub fn verify_v4_full(
    key: &VerifierKey,
    response: &V4AuditResponse,
    expected_prompt_token_ids: Option<&[u32]>,
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> V4VerifyReport {
    let start = Instant::now();
    let (mut checks_run, mut failures) = verify_v4_structural(response);

    // Input tokenization verification: canonical reconstruction when possible.
    //
    // Priority:
    // 1. tokenizer + prompt bytes + manifest → reconstruct and verify
    // 2. caller-supplied expected_prompt_token_ids → verify directly
    // 3. neither → skip (no tokenization check)
    let reconstructed_tids: Option<Vec<u32>>;
    let tids_to_check: Option<&[u32]> = if let Some(tok) = tokenizer {
        // Attempt canonical reconstruction from raw prompt + InputSpec.
        match (&response.prompt, &response.manifest) {
            (Some(prompt_bytes), Some(manifest)) => {
                let input_spec = InputSpec::from(manifest);
                match tok.tokenize(prompt_bytes, &input_spec) {
                    Ok(tids) => {
                        reconstructed_tids = Some(tids);
                        reconstructed_tids.as_deref()
                    }
                    Err(e) => {
                        failures.push(format!("tokenizer reconstruction failed: {}", e));
                        None
                    }
                }
            }
            (None, _) => {
                // No prompt bytes → can't reconstruct. Fall back to caller-supplied.
                expected_prompt_token_ids
            }
            (_, None) => {
                // No manifest → can't get InputSpec. Fall back to caller-supplied.
                expected_prompt_token_ids
            }
        }
    } else {
        expected_prompt_token_ids
    };

    if let Some(expected_tids) = tids_to_check {
        checks_run += 1;
        let tok_failures = verify_input_tokenization(response, expected_tids);
        failures.extend(tok_failures);
    }

    // Protocol path: check shell openings with key-only Freivalds.
    match &response.shell_opening {
        Some(shell) => {
            // Fail-closed embedding binding.
            //
            // If key has embedding_merkle_root: REQUIRE initial_residual + valid proof.
            // If key has no embedding_merkle_root: REJECT any initial_residual (unbound).
            if let Some(ref emb_root) = key.embedding_merkle_root {
                checks_run += 1;
                match (&shell.initial_residual, &shell.embedding_proof) {
                    (Some(ir), Some(proof)) => {
                        let leaf = verilm_core::merkle::hash_embedding_row(ir);
                        if proof.leaf_index != response.token_id {
                            failures.push(format!(
                                "embedding proof leaf_index {} != token_id {}",
                                proof.leaf_index, response.token_id
                            ));
                        } else if !verilm_core::merkle::verify(emb_root, &leaf, proof) {
                            failures.push("embedding Merkle proof verification failed".into());
                        }
                    }
                    (None, _) => {
                        failures.push(
                            "key has embedding_merkle_root but shell missing initial_residual".into()
                        );
                    }
                    (Some(_), None) => {
                        failures.push(
                            "key has embedding_merkle_root but shell missing embedding_proof".into()
                        );
                    }
                }
            } else if shell.initial_residual.is_some() {
                checks_run += 1;
                failures.push(
                    "shell has initial_residual but key has no embedding_merkle_root to verify it".into()
                );
            }

            // Rich prefix: verify embedding binding for all prefix tokens.
            //
            // When the response carries prefix_embedding_rows + proofs, verify each
            // one against the key's embedding Merkle root. This closes the
            // "embedding cross-check for prefix tokens" gap.
            if let (Some(ref emb_root), Some(ref prefix_rows), Some(ref prefix_proofs)) = (
                &key.embedding_merkle_root,
                &response.prefix_embedding_rows,
                &response.prefix_embedding_proofs,
            ) {
                if prefix_rows.len() != response.prefix_token_ids.len()
                    || prefix_proofs.len() != response.prefix_token_ids.len()
                {
                    checks_run += 1;
                    failures.push(format!(
                        "prefix embedding count mismatch: {} rows, {} proofs, {} token_ids",
                        prefix_rows.len(), prefix_proofs.len(), response.prefix_token_ids.len()
                    ));
                } else {
                    for (j, ((row, proof), &tid)) in prefix_rows.iter()
                        .zip(prefix_proofs.iter())
                        .zip(response.prefix_token_ids.iter())
                        .enumerate()
                    {
                        checks_run += 1;
                        let leaf = verilm_core::merkle::hash_embedding_row(row);
                        if proof.leaf_index != tid {
                            failures.push(format!(
                                "prefix token {}: embedding proof leaf_index {} != token_id {}",
                                j, proof.leaf_index, tid
                            ));
                        } else if !verilm_core::merkle::verify(emb_root, &leaf, proof) {
                            failures.push(format!(
                                "prefix token {}: embedding Merkle proof verification failed",
                                j
                            ));
                        }
                    }
                }
            }

            // Deep prefix: verify prefix_retained[j] hashes to prefix_leaf_hashes[j],
            // then run shell Freivalds+bridge on each prefix token.
            if let (Some(ref prefix_ret), Some(ref prefix_shells)) = (
                &response.prefix_retained,
                &response.prefix_shell_openings,
            ) {
                if prefix_ret.len() != response.prefix_leaf_hashes.len()
                    || prefix_shells.len() != response.prefix_leaf_hashes.len()
                {
                    checks_run += 1;
                    failures.push(format!(
                        "deep prefix count mismatch: {} retained, {} shells, {} leaf_hashes",
                        prefix_ret.len(), prefix_shells.len(), response.prefix_leaf_hashes.len()
                    ));
                } else {
                    for (j, ((ret_j, shell_j), &expected_hash)) in prefix_ret.iter()
                        .zip(prefix_shells.iter())
                        .zip(response.prefix_leaf_hashes.iter())
                        .enumerate()
                    {
                        // 1. Hash consistency: retained state must match committed leaf.
                        checks_run += 1;
                        let fr_ref = shell_j.final_residual.as_deref();
                        let hash_j = verilm_core::merkle::hash_retained_with_residual(ret_j, fr_ref);
                        if hash_j != expected_hash {
                            failures.push(format!(
                                "prefix token {}: retained hash mismatch (deep audit)",
                                j
                            ));
                            continue;
                        }

                        // 2. Shell Freivalds + bridge verification (same as challenged token).
                        let (c, f, _) = verify_shell_opening(key, ret_j, shell_j);
                        checks_run += c;
                        for failure in f {
                            failures.push(format!("prefix token {}: {}", j, failure));
                        }
                    }
                }
            }

            // Prompt/generation boundary consistency: if we know n_prompt_tokens and
            // the challenged token is within the prompt range, verify the embedding
            // proof binds the correct token_id (already done above). Additionally,
            // if the challenged token is a generated token (token_index >= n_prompt-1),
            // sampling replay must agree — this is checked later in LM-head verification.
            if let Some(npt) = response.n_prompt_tokens {
                // The first prompt token (position 0 in all_token_ids) is the
                // embedding input and is not in the committed token_ids array.
                // Committed positions 0..npt-2 are prompt tokens (positions 1..npt-1 in all_token_ids).
                // Committed positions npt-1.. are generated tokens.
                let gen_start = npt.saturating_sub(1); // first generated token in committed array
                if response.token_index >= gen_start {
                    // This is a generated token — sampling replay will verify it.
                    // The embedding proof (if present) binds token_id to the correct row.
                } else {
                    // This is a prompt token — it should match the tokenizer's output.
                    // The verifier can't re-tokenize, but the binding chain is:
                    //   prompt_hash binds prompt text
                    //   n_prompt_tokens binds the boundary
                    //   IO chain binds each token_id
                    //   embedding proof binds token_id → embedding row
                    // No additional check needed here beyond what's already verified.
                }
            }

            let (c, f, shell_final_hidden) = verify_shell_opening(key, &response.retained, shell);
            checks_run += c;
            failures.extend(f);

            // Four-spec manifest binding: split the manifest, verify each spec hash
            // individually against the commitment, then verify the composed M.
            let decode_params = if let Some(ref manifest) = response.manifest {
                let (input_spec, model_spec, decode_spec, output_spec) = manifest.split();
                let h_in = merkle::hash_input_spec(&input_spec);
                let h_mod = merkle::hash_model_spec(&model_spec);
                let h_dec = merkle::hash_decode_spec(&decode_spec);
                let h_out = merkle::hash_output_spec(&output_spec);

                // Fail-closed: all four spec hashes MUST be present in the commitment.
                checks_run += 4;
                match response.commitment.input_spec_hash {
                    None => failures.push("commitment missing input_spec_hash".into()),
                    Some(committed) if h_in != committed =>
                        failures.push("input_spec_hash mismatch".into()),
                    _ => {}
                }
                match response.commitment.model_spec_hash {
                    None => failures.push("commitment missing model_spec_hash".into()),
                    Some(committed) if h_mod != committed =>
                        failures.push("model_spec_hash mismatch".into()),
                    _ => {}
                }
                match response.commitment.decode_spec_hash {
                    None => failures.push("commitment missing decode_spec_hash".into()),
                    Some(committed) if h_dec != committed =>
                        failures.push("decode_spec_hash mismatch".into()),
                    _ => {}
                }
                match response.commitment.output_spec_hash {
                    None => failures.push("commitment missing output_spec_hash".into()),
                    Some(committed) if h_out != committed =>
                        failures.push("output_spec_hash mismatch".into()),
                    _ => {}
                }

                // Verify composed manifest hash: M = H(H_input || H_model || H_decode || H_output).
                checks_run += 1;
                match response.commitment.manifest_hash {
                    None => failures.push("commitment missing manifest_hash".into()),
                    Some(committed_hash) => {
                        let computed = merkle::hash_manifest_composed(h_in, h_mod, h_dec, h_out);
                        if computed != committed_hash {
                            failures.push("manifest hash does not match commitment".into());
                        }
                    }
                }

                // Reject decode parameters the canonical sampler doesn't support.
                // These are bound in the spec hash, so a prover can't hide them.
                checks_run += 1;
                if decode_spec.repetition_penalty != 1.0 {
                    failures.push(format!(
                        "unsupported repetition_penalty={} (canonical sampler requires 1.0)",
                        decode_spec.repetition_penalty
                    ));
                }
                if decode_spec.frequency_penalty != 0.0 {
                    failures.push(format!(
                        "unsupported frequency_penalty={} (canonical sampler requires 0.0)",
                        decode_spec.frequency_penalty
                    ));
                }
                if decode_spec.presence_penalty != 0.0 {
                    failures.push(format!(
                        "unsupported presence_penalty={} (canonical sampler requires 0.0)",
                        decode_spec.presence_penalty
                    ));
                }
                if !decode_spec.logit_bias.is_empty() {
                    failures.push(format!(
                        "unsupported logit_bias ({} entries, canonical sampler requires empty)",
                        decode_spec.logit_bias.len()
                    ));
                }
                if !decode_spec.guided_decoding.is_empty() {
                    failures.push(format!(
                        "unsupported guided_decoding='{}' (canonical sampler requires empty)",
                        decode_spec.guided_decoding
                    ));
                }

                // Output spec checks.
                if !output_spec.stop_sequences.is_empty() {
                    failures.push(format!(
                        "unsupported stop_sequences ({} entries, canonical sampler requires empty)",
                        output_spec.stop_sequences.len()
                    ));
                }
                if output_spec.max_tokens > 0 {
                    if response.token_index >= output_spec.max_tokens {
                        failures.push(format!(
                            "token_index {} exceeds output_spec max_tokens {}",
                            response.token_index, output_spec.max_tokens
                        ));
                    }
                    if response.commitment.n_tokens > output_spec.max_tokens {
                        failures.push(format!(
                            "committed n_tokens {} exceeds output_spec max_tokens {}",
                            response.commitment.n_tokens, output_spec.max_tokens
                        ));
                    }
                }

                // Output policy enforcement: min_tokens and ignore_eos.
                //
                // These checks verify that the committed generation length
                // and challenged token are consistent with the output policy.
                // Requires eos_token_id to identify EOS tokens.
                if let Some(eos_id) = output_spec.eos_token_id {
                    // min_tokens: generation must produce at least this many tokens
                    // before the model is allowed to select EOS.
                    // n_prompt_tokens - 1 = number of prompt tokens in the committed array.
                    // Generated tokens start at index (n_prompt_tokens - 1).
                    if output_spec.min_tokens > 0 {
                        checks_run += 1;
                        let gen_start = response.n_prompt_tokens
                            .unwrap_or(1)
                            .saturating_sub(1);
                        // The number of generated tokens committed.
                        let n_generated = response.commitment.n_tokens.saturating_sub(gen_start);
                        if n_generated < output_spec.min_tokens {
                            failures.push(format!(
                                "committed only {} generated tokens but output_spec requires min_tokens={}",
                                n_generated, output_spec.min_tokens
                            ));
                        }
                        // If the challenged token is within the min_tokens window and is EOS, reject.
                        let gen_index = response.token_index.saturating_sub(gen_start);
                        if gen_index < output_spec.min_tokens && response.token_id == eos_id {
                            failures.push(format!(
                                "token at generation position {} is EOS ({}) but min_tokens={}",
                                gen_index, eos_id, output_spec.min_tokens
                            ));
                        }
                    }

                    // ignore_eos: if true, the model must never select EOS.
                    if output_spec.ignore_eos {
                        checks_run += 1;
                        if response.token_id == eos_id {
                            failures.push(format!(
                                "token_id {} is EOS but ignore_eos=true",
                                response.token_id
                            ));
                        }
                    }
                } else if output_spec.min_tokens > 0 || output_spec.ignore_eos {
                    // Fail-closed: can't enforce min_tokens or ignore_eos without eos_token_id.
                    failures.push(
                        "output_spec requires min_tokens or ignore_eos but eos_token_id is missing".into()
                    );
                }

                // eos_policy enforcement.
                //
                // "stop": generation terminates when EOS is produced. If the
                // challenged token is EOS, it must be the last committed token
                // (no tokens after EOS). Conversely, if there are tokens after
                // the last generated position, none of the intermediate tokens
                // should be EOS (the prover continued past an EOS).
                if output_spec.eos_policy == "stop" {
                    if let Some(eos_id) = output_spec.eos_token_id {
                        checks_run += 1;
                        let is_last_token = response.token_index == response.commitment.n_tokens.saturating_sub(1);
                        if response.token_id == eos_id && !is_last_token {
                            failures.push(format!(
                                "eos_policy='stop' but EOS token ({}) at index {} is not the last token (n_tokens={})",
                                eos_id, response.token_index, response.commitment.n_tokens
                            ));
                        }
                    }
                } else if output_spec.eos_policy != "sample" {
                    // Only "stop" and "sample" are recognized. Reject unknown policies.
                    failures.push(format!(
                        "unknown eos_policy='{}' (expected 'stop' or 'sample')",
                        output_spec.eos_policy
                    ));
                }

                Some(verilm_core::sampling::DecodeParams {
                    temperature: decode_spec.temperature,
                    top_k: decode_spec.top_k,
                    top_p: decode_spec.top_p,
                })
            } else {
                None
            };

            // LM head + sampling verification.
            //
            // Two sources for final_hidden:
            // 1. Captured final_residual (from live GPU inference) — exact.
            //    Apply final RMSNorm + quantize to get the true final_hidden.
            // 2. Shell-replayed final_hidden — approximate (diverges after many
            //    layers of approximate attention). Fallback for toy model only.
            //
            // Fail-closed: when the key has both lm_head and final_norm_weights,
            // the prover MUST supply final_residual. Missing it means the exact
            // tail path was not captured, and we reject rather than silently
            // downgrade to the approximate shell replay.
            let final_hidden = if let Some(ref fr) = shell.final_residual {
                if let Some(ref fnw) = key.final_norm_weights {
                    let res_f64: Vec<f64> = fr.iter().map(|&v| v as f64).collect();
                    let normed = verilm_core::rmsnorm::rmsnorm_f64_input(&res_f64, fnw, key.rmsnorm_eps);
                    Some(normed.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect())
                } else {
                    // No final_norm_weights in key — fall back to shell replay (toy model).
                    shell_final_hidden
                }
            } else if key.final_norm_weights.is_some() && key.lm_head.is_some() {
                // Fail-closed: key requires exact verification but no captured state.
                failures.push(
                    "key has lm_head + final_norm_weights but shell missing final_residual \
                     (exact tail verification required, cannot fall back to shell replay)".into()
                );
                None
            } else {
                // Toy model / no final_norm_weights — shell replay fallback is acceptable.
                shell_final_hidden
            };

            // LM-head Freivalds: verify the prover's claimed logits_i32.
            //
            // Fail-closed: if the key has LmHead Freivalds vectors, the prover
            // MUST supply logits_i32. Missing claim = reject.
            let r_lm = key.r_for(MatrixType::LmHead);
            let key_has_lm_freivalds = !r_lm.is_empty() && key.v_lm_head.is_some();

            if key_has_lm_freivalds {
                let v_lm = key.v_lm_head.as_ref().unwrap();

                match (&shell.logits_i32, &final_hidden) {
                    (Some(ref claimed_logits), Some(ref fh)) => {
                        checks_run += 1;
                        if !freivalds::check(v_lm, fh, r_lm, claimed_logits) {
                            failures.push("lm_head: Freivalds check failed on prover's logits_i32 claim".into());
                        }

                        // Token replay: use the prover's (now Freivalds-verified) logits.
                        if key.config.vocab_size > 0 {
                            checks_run += 1;
                            let logits: Vec<f32> = claimed_logits.iter().map(|&v| v as f32).collect();

                            let expected_token = if let Some(ref dp) = decode_params {
                                let token_seed = verilm_core::sampling::derive_token_seed(
                                    &response.revealed_seed, response.token_index,
                                );
                                verilm_core::sampling::sample(&logits, dp, &token_seed)
                            } else {
                                logits.iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .map(|(i, _)| i as u32)
                                    .unwrap_or(0)
                            };

                            if expected_token != response.token_id {
                                failures.push(format!(
                                    "lm_head: expected token_id={} but claimed token_id={}",
                                    expected_token, response.token_id
                                ));
                            }
                        }
                    }
                    (None, _) => {
                        failures.push(
                            "lm_head: key requires logits_i32 but shell opening did not provide it".into()
                        );
                    }
                    (Some(_), None) => {
                        failures.push(
                            "lm_head: logits_i32 present but no final_hidden to check against".into()
                        );
                    }
                }
            }
        }
        None => {
            failures.push("V4 audit response missing shell_opening".into());
        }
    }

    // Detokenization verification: check claimed output text matches decoded tokens.
    //
    // Fail-closed: when a detokenizer is provided, the response MUST carry
    // output_text. Missing output_text with a detokenizer = prover did not
    // include the claimed text for verification.
    if let Some(detok) = detokenizer {
        checks_run += 1;
        match &response.output_text {
            Some(ref claimed_text) => {
                let detok_policy = response.manifest.as_ref()
                    .map(|m| OutputSpec::from(m))
                    .and_then(|os| os.detokenization_policy);

                // Collect generation token IDs: prefix after prompt boundary + challenged token.
                let gen_start = response.n_prompt_tokens
                    .unwrap_or(1)
                    .saturating_sub(1) as usize;
                let mut gen_token_ids: Vec<u32> = response.prefix_token_ids
                    .get(gen_start..)
                    .unwrap_or(&[])
                    .to_vec();
                gen_token_ids.push(response.token_id);

                let is_last_token = response.token_index == response.commitment.n_tokens.saturating_sub(1);

                match detok.decode(&gen_token_ids, detok_policy.as_deref()) {
                    Ok(decoded) => {
                        if is_last_token {
                            // Full generation — exact match required.
                            if decoded != *claimed_text {
                                failures.push(format!(
                                    "detokenization mismatch (policy={:?}): decoded={:?} vs claimed={:?}",
                                    detok_policy, decoded, claimed_text
                                ));
                            }
                        } else {
                            // Partial generation — decoded must be a prefix of the claimed text.
                            if !claimed_text.starts_with(&decoded) {
                                failures.push(format!(
                                    "detokenization prefix mismatch (policy={:?}): decoded={:?} is not a prefix of claimed={:?}",
                                    detok_policy, decoded, claimed_text
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        failures.push(format!("detokenization failed: {}", e));
                    }
                }
            }
            None => {
                failures.push(
                    "detokenizer provided but response missing output_text \
                     (prover must include claimed output text for detokenization verification)".into()
                );
            }
        }
    }

    let duration = start.elapsed();
    let checks_passed = checks_run.saturating_sub(failures.len());

    V4VerifyReport {
        verdict: if failures.is_empty() { Verdict::Pass } else { Verdict::Fail },
        token_index: response.token_index,
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

/// Verify shell opening for one token using key-only Freivalds + bridge checks.
///
/// The verifier does NOT recompute matmuls — it checks the prover's i32
/// accumulators with precomputed Freivalds keys and verifies bridge
/// consistency by deriving intermediate i8 values from the accumulators.
///
/// When `shell.initial_residual` is present and the key has RMSNorm weights,
/// uses the full bridge (dequant → residual → RMSNorm → quantize). This
/// also enables QKV Freivalds at layer 0.
///
/// Returns `(checks_run, failures, final_hidden)`. `final_hidden` is the
/// i8 hidden state after the last layer, derived from the bridge chain.
/// Used by verify_v4 for lm_head logit verification.
fn verify_shell_opening(
    key: &VerifierKey,
    retained: &verilm_core::types::RetainedTokenState,
    shell: &verilm_core::types::ShellTokenOpening,
) -> (usize, Vec<String>, Option<Vec<i8>>) {
    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // Resolve which layers are present.
    let opened_layers: Vec<usize> = shell.layer_indices.clone()
        .unwrap_or_else(|| (0..retained.layers.len()).collect());

    if shell.layers.len() != opened_layers.len() {
        failures.push(format!(
            "shell_opening has {} layers but layer_indices specifies {}",
            shell.layers.len(), opened_layers.len()
        ));
        return (checks_run, failures, None);
    }

    // Build a lookup: shell_idx for a given layer_idx (None if not opened).
    let max_layer = opened_layers.iter().copied().max().unwrap_or(0);
    let mut shell_idx_for = vec![None; max_layer + 1];
    for (si, &li) in opened_layers.iter().enumerate() {
        if li <= max_layer {
            shell_idx_for[li] = Some(si);
        }
    }

    // Full bridge: only when initial_residual is authenticated.
    let use_full_bridge = shell.initial_residual.is_some()
        && !key.rmsnorm_attn_weights.is_empty()
        && key.embedding_merkle_root.is_some();

    let (mut residual, mut x_attn) = if use_full_bridge {
        let ir = shell.initial_residual.as_ref().unwrap();
        let res: Vec<f64> = ir.iter().map(|&v| v as f64).collect();
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
            &res, &key.rmsnorm_attn_weights[0], key.rmsnorm_eps,
        );
        let xa = verilm_core::rmsnorm::quantize_f64_to_i8(
            &normed, retained.layers[0].scale_x_attn as f64,
        );
        (Some(res), Some(xa))
    } else {
        (None, None)
    };

    // Iterate 0..=max_layer: bridge is sequential, Freivalds only on opened layers.
    for layer_idx in 0..=max_layer {
        if layer_idx >= retained.layers.len() { break; }
        let rs = &retained.layers[layer_idx];
        let a = &rs.a;

        // If this layer has a shell opening, run Freivalds checks.
        if let Some(si) = shell_idx_for[layer_idx] {
            let sl = &shell.layers[si];

            // QKV Freivalds (full bridge: all layers; toy: layers > 0)
            if let Some(ref xa) = x_attn {
                let qkv = [
                    (MatrixType::Wq, &sl.q, key.config.hidden_dim),
                    (MatrixType::Wk, &sl.k, key.config.kv_dim),
                    (MatrixType::Wv, &sl.v, key.config.kv_dim),
                ];
                for (mt, opened_acc, _expected_rows) in &qkv {
                    match opened_acc {
                        Some(z) => {
                            checks_run += 1;
                            if !freivalds::check(
                                key.v_for(layer_idx, *mt),
                                xa,
                                key.r_for(*mt),
                                z,
                            ) {
                                failures.push(format!(
                                    "layer {} {:?}: Freivalds failed on shell opening",
                                    layer_idx, mt
                                ));
                            }
                        }
                        None => {
                            failures.push(format!(
                                "layer {} {:?}: shell opening missing QKV (x_attn derivable)",
                                layer_idx, mt
                            ));
                        }
                    }
                }
            }

            // W_o @ a
            checks_run += 1;
            if !freivalds::check(
                key.v_for(layer_idx, MatrixType::Wo),
                a,
                key.r_for(MatrixType::Wo),
                &sl.attn_out,
            ) {
                failures.push(format!(
                    "layer {} Wo: Freivalds failed on shell opening", layer_idx
                ));
            }

            // Post-attention bridge: derive x_ffn
            let x_ffn = if let Some(ref mut res) = residual {
                verilm_core::rmsnorm::bridge_residual_rmsnorm(
                    &sl.attn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wo),
                    rs.scale_a,
                    res,
                    &key.rmsnorm_ffn_weights[layer_idx],
                    key.rmsnorm_eps,
                    rs.scale_x_ffn,
                )
            } else {
                verilm_core::bridge_requantize(
                    &sl.attn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wo),
                    rs.scale_a,
                    rs.scale_x_ffn,
                )
            };

            // W_g, W_u @ x_ffn
            for (mt, z) in [(MatrixType::Wg, &sl.g), (MatrixType::Wu, &sl.u)] {
                checks_run += 1;
                if !freivalds::check(
                    key.v_for(layer_idx, mt),
                    &x_ffn,
                    key.r_for(mt),
                    z,
                ) {
                    failures.push(format!(
                        "layer {} {:?}: Freivalds failed on shell opening",
                        layer_idx, mt
                    ));
                }
            }

            // Bridge: verifier derives h from prover's g, u using scales
            let h = verilm_core::silu::compute_h_scaled(
                &sl.g, &sl.u,
                key.weight_scale_for(layer_idx, MatrixType::Wg),
                key.weight_scale_for(layer_idx, MatrixType::Wu),
                rs.scale_x_ffn,
                rs.scale_h,
            );

            // W_d @ h
            checks_run += 1;
            if !freivalds::check(
                key.v_for(layer_idx, MatrixType::Wd),
                &h,
                key.r_for(MatrixType::Wd),
                &sl.ffn_out,
            ) {
                failures.push(format!(
                    "layer {} Wd: Freivalds failed on shell opening", layer_idx
                ));
            }

            // Post-FFN bridge: derive x_attn for next layer
            let next_scale_x_attn = retained.layers
                .get(layer_idx + 1)
                .map(|r| r.scale_x_attn)
                .unwrap_or(1.0);

            x_attn = if let Some(ref mut res) = residual {
                if layer_idx + 1 < key.rmsnorm_attn_weights.len() {
                    Some(verilm_core::rmsnorm::bridge_residual_rmsnorm(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        rs.scale_h,
                        res,
                        &key.rmsnorm_attn_weights[layer_idx + 1],
                        key.rmsnorm_eps,
                        next_scale_x_attn,
                    ))
                } else {
                    verilm_core::rmsnorm::dequant_add_residual(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        rs.scale_h,
                        res,
                    );
                    Some(verilm_core::bridge_requantize(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        rs.scale_h,
                        next_scale_x_attn,
                    ))
                }
            } else {
                Some(verilm_core::bridge_requantize(
                    &sl.ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    rs.scale_h,
                    next_scale_x_attn,
                ))
            };
        }
        // Layer not opened: no Freivalds, but bridge state must NOT advance
        // (the prover didn't provide intermediates for this layer).
        // For contiguous prefix semantics, all layers up to max are opened,
        // so this branch is only hit for non-prefix gaps (shouldn't happen
        // with proper challenge derivation). Bridge tracking stops.
    }

    // Derive final_hidden from the chain endpoint.
    // For full bridge with final_norm_weights: apply final RMSNorm to residual.
    // For simplified bridge (toy model): x_attn after last layer is requantize(ffn_out).
    let final_hidden = if let (Some(ref res), Some(ref fnw)) = (&residual, &key.final_norm_weights) {
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(res, fnw, key.rmsnorm_eps);
        // Quantize to i8 with unit scale (simple clamp) for lm_head matmul.
        Some(normed.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect())
    } else {
        // Simplified bridge: x_attn after last layer IS the final hidden state.
        x_attn.clone()
    };

    (checks_run, failures, final_hidden)
}

/// Verify V4 with verifier-side weight-backed replay (debug/oracle path).
///
/// The verifier independently reconstructs the computation shell from
/// committed retained `a` + public weights. This is NOT the protocol path —
/// it requires the verifier to hold full weights. Use for:
/// - Differential testing against the protocol verifier
/// - Stronger local checking when weights are available
/// - Dev-mode paranoid verification
pub fn verify_v4_with_weights(
    key: &VerifierKey,
    response: &V4AuditResponse,
    weights: &dyn ShellWeights,
) -> V4VerifyReport {
    let start = Instant::now();
    let (mut checks_run, mut failures) = verify_v4_structural(response);

    let cfg = &key.config;

    // Extract initial_residual only when authenticated (embedding root present).
    let initial_residual = if key.embedding_merkle_root.is_some() {
        response.shell_opening
            .as_ref()
            .and_then(|s| s.initial_residual.as_deref())
    } else {
        None
    };

    // NOTE: prefix replay is no longer possible — prefix_leaf_hashes are
    // compact 32-byte hashes, not full RetainedTokenState.  The structural
    // checks in verify_v4_structural still verify the prefix Merkle proofs
    // and IO chain using these hashes.

    // Replay the challenged token (with initial_residual if available).
    let (c, f) = replay_token_shell(
        key,
        cfg,
        &response.retained,
        weights,
        response.prefix_leaf_hashes.len(),
        initial_residual,
    );
    checks_run += c;
    failures.extend(f);

    let duration = start.elapsed();
    let checks_passed = checks_run.saturating_sub(failures.len());

    V4VerifyReport {
        verdict: if failures.is_empty() { Verdict::Pass } else { Verdict::Fail },
        token_index: response.token_index,
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

/// Structural checks shared by verify_v4 and verify_v4_with_replay.
fn verify_v4_structural(
    response: &V4AuditResponse,
) -> (usize, Vec<String>) {
    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // 1. Commitment version must be V4
    checks_run += 1;
    if response.commitment.version != CommitmentVersion::V4 {
        failures.push(format!(
            "expected V4 commitment, got {:?}",
            response.commitment.version
        ));
    }

    // 2. Seed commitment: hash(revealed_seed) == commitment.seed_commitment
    checks_run += 1;
    match response.commitment.seed_commitment {
        Some(expected) => {
            let computed = merkle::hash_seed(&response.revealed_seed);
            if computed != expected {
                failures.push(
                    "seed commitment mismatch: hash(revealed_seed) != commitment.seed_commitment"
                        .into(),
                );
            }
        }
        None => {
            failures.push("V4 commitment missing seed_commitment".into());
        }
    }

    // 3. Prompt hash binding (fail-closed): prompt MUST be present and hash MUST match.
    checks_run += 1;
    match (&response.prompt, response.commitment.prompt_hash) {
        (Some(prompt), Some(committed_hash)) => {
            let computed = merkle::hash_prompt(prompt);
            if computed != committed_hash {
                failures.push("prompt_hash mismatch: hash(prompt) != commitment.prompt_hash".into());
            }
        }
        (None, Some(_)) => {
            failures.push("commitment has prompt_hash but response missing prompt bytes".into());
        }
        (Some(_), None) => {
            failures.push("response has prompt but commitment missing prompt_hash".into());
        }
        (None, None) => {
            failures.push("V4 commitment missing prompt_hash".into());
        }
    }

    // 3b. Prompt token count binding (fail-closed): n_prompt_tokens MUST be present.
    checks_run += 1;
    match (response.commitment.n_prompt_tokens, response.n_prompt_tokens) {
        (Some(committed_npt), Some(response_npt)) => {
            if committed_npt != response_npt {
                failures.push(format!(
                    "n_prompt_tokens mismatch: commitment={} response={}",
                    committed_npt, response_npt
                ));
            }
            // Sanity: n_prompt_tokens must be <= total tokens + 1 (the +1 is the
            // first token consumed as embedding input, not in the committed array).
            if committed_npt > response.commitment.n_tokens + 1 {
                failures.push(format!(
                    "n_prompt_tokens {} exceeds n_tokens {} + 1",
                    committed_npt, response.commitment.n_tokens
                ));
            }
        }
        (None, _) => {
            failures.push("commitment missing n_prompt_tokens".into());
        }
        (_, None) => {
            failures.push("response missing n_prompt_tokens".into());
        }
    }

    // 4. Challenged token retained leaf Merkle proof.
    // When shell_opening has final_residual, include it in the leaf hash
    // (it was bound into the commitment at commit time).
    checks_run += 1;
    let final_residual_ref = response.shell_opening.as_ref()
        .and_then(|s| s.final_residual.as_deref());
    let leaf_hash = merkle::hash_retained_with_residual(&response.retained, final_residual_ref);
    if !merkle::verify(
        &response.commitment.merkle_root,
        &leaf_hash,
        &response.merkle_proof,
    ) {
        failures.push(format!(
            "token {}: retained leaf Merkle proof failed",
            response.token_index
        ));
    }

    // 5. Prefix retained leaf Merkle proofs
    for (j, (prefix_leaf_hash, proof)) in response
        .prefix_leaf_hashes
        .iter()
        .zip(response.prefix_merkle_proofs.iter())
        .enumerate()
    {
        checks_run += 1;
        if !merkle::verify(&response.commitment.merkle_root, prefix_leaf_hash, proof) {
            failures.push(format!(
                "prefix token {}: retained leaf Merkle proof failed",
                j
            ));
        }
    }

    // 6. IO chain verification (genesis bound to request via prompt_hash).
    checks_run += 1;
    let io_genesis = match response.commitment.prompt_hash {
        Some(ph) => merkle::io_genesis_v4(ph),
        None => [0u8; 32], // already failed at check 3
    };
    let mut prev_io = io_genesis;
    for (j, prefix_leaf_hash) in response.prefix_leaf_hashes.iter().enumerate() {
        prev_io = merkle::io_hash_v4(*prefix_leaf_hash, response.prefix_token_ids[j], prev_io);
    }

    if prev_io != response.prev_io_hash {
        failures.push(format!(
            "token {}: prev_io_hash doesn't match recomputed chain from prefix",
            response.token_index
        ));
    }

    checks_run += 1;
    let challenged_io = merkle::io_hash_v4(leaf_hash, response.token_id, prev_io);
    if !merkle::verify(
        &response.commitment.io_root,
        &challenged_io,
        &response.io_proof,
    ) {
        failures.push(format!(
            "token {}: IO chain proof verification failed",
            response.token_index
        ));
    }

    // 7. Prefix count == token_index
    checks_run += 1;
    if response.prefix_leaf_hashes.len() != response.token_index as usize {
        failures.push(format!(
            "token {}: expected {} prefix tokens but got {}",
            response.token_index,
            response.token_index,
            response.prefix_leaf_hashes.len()
        ));
    }

    (checks_run, failures)
}

/// Replay one token's computation shell from retained state + public weights.
///
/// When the key has RMSNorm weights and `initial_residual` is provided,
/// uses full bridge and enables QKV at layer 0. Otherwise falls back to
/// simplified bridge (layer 0 QKV skipped).
///
/// Returns (checks_run, failures).
fn replay_token_shell(
    key: &VerifierKey,
    cfg: &verilm_core::constants::ModelConfig,
    retained: &verilm_core::types::RetainedTokenState,
    weights: &dyn ShellWeights,
    token_pos: usize,
    initial_residual: Option<&[f32]>,
) -> (usize, Vec<String>) {
    use verilm_core::matmul::matmul_i32;

    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // Full bridge: only when initial_residual is provided AND authenticated.
    let use_full_bridge = initial_residual.is_some()
        && !key.rmsnorm_attn_weights.is_empty()
        && key.embedding_merkle_root.is_some();

    let (mut residual, mut x_attn) = if use_full_bridge {
        let ir = initial_residual.unwrap();
        let res: Vec<f64> = ir.iter().map(|&v| v as f64).collect();
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
            &res, &key.rmsnorm_attn_weights[0], key.rmsnorm_eps,
        );
        let xa = verilm_core::rmsnorm::quantize_f64_to_i8(
            &normed, retained.layers[0].scale_x_attn as f64,
        );
        (Some(res), Some(xa))
    } else {
        (None, None)
    };

    for (layer_idx, rs) in retained.layers.iter().enumerate() {
        let a = &rs.a;

        // QKV Freivalds (full bridge: all layers; toy: layers > 0)
        if let Some(ref xa) = x_attn {
            let qkv_mats = [
                (MatrixType::Wq, cfg.hidden_dim, cfg.hidden_dim),
                (MatrixType::Wk, cfg.kv_dim, cfg.hidden_dim),
                (MatrixType::Wv, cfg.kv_dim, cfg.hidden_dim),
            ];
            for (mt, rows, cols) in &qkv_mats {
                let z = matmul_i32(weights.weight(layer_idx, *mt), xa, *rows, *cols);
                checks_run += 1;
                if !freivalds::check(
                    key.v_for(layer_idx, *mt),
                    xa,
                    key.r_for(*mt),
                    &z,
                ) {
                    failures.push(format!(
                        "token {} layer {} {:?}: Freivalds weight-binding failed",
                        token_pos, layer_idx, mt
                    ));
                }
            }
        }

        // W_o: attn_out = W_o @ a (verifier-computed)
        let attn_out = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wo),
            a,
            cfg.hidden_dim,
            cfg.hidden_dim,
        );
        checks_run += 1;
        if !freivalds::check(
            key.v_for(layer_idx, MatrixType::Wo),
            a,
            key.r_for(MatrixType::Wo),
            &attn_out,
        ) {
            failures.push(format!(
                "token {} layer {} Wo: Freivalds weight-binding failed",
                token_pos, layer_idx
            ));
        }

        // Post-attention bridge: derive x_ffn
        let x_ffn = if let Some(ref mut res) = residual {
            verilm_core::rmsnorm::bridge_residual_rmsnorm(
                &attn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wo),
                rs.scale_a,
                res,
                &key.rmsnorm_ffn_weights[layer_idx],
                key.rmsnorm_eps,
                rs.scale_x_ffn,
            )
        } else {
            verilm_core::bridge_requantize(
                &attn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wo),
                rs.scale_a,
                rs.scale_x_ffn,
            )
        };

        // W_g, W_u: gate and up projections
        let g = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wg),
            &x_ffn,
            cfg.ffn_dim,
            cfg.hidden_dim,
        );
        let u = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wu),
            &x_ffn,
            cfg.ffn_dim,
            cfg.hidden_dim,
        );

        // Freivalds on W_g, W_u
        for (mt, z) in [(MatrixType::Wg, &g), (MatrixType::Wu, &u)] {
            checks_run += 1;
            if !freivalds::check(
                key.v_for(layer_idx, mt),
                &x_ffn,
                key.r_for(mt),
                z,
            ) {
                failures.push(format!(
                    "token {} layer {} {:?}: Freivalds weight-binding failed",
                    token_pos, layer_idx, mt
                ));
            }
        }

        // SiLU: h via scale-aware bridge
        let h = verilm_core::silu::compute_h_scaled(
            &g, &u,
            key.weight_scale_for(layer_idx, MatrixType::Wg),
            key.weight_scale_for(layer_idx, MatrixType::Wu),
            rs.scale_x_ffn,
            rs.scale_h,
        );

        // W_d: ffn_out = W_d @ h
        let ffn_out = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wd),
            &h,
            cfg.hidden_dim,
            cfg.ffn_dim,
        );
        checks_run += 1;
        if !freivalds::check(
            key.v_for(layer_idx, MatrixType::Wd),
            &h,
            key.r_for(MatrixType::Wd),
            &ffn_out,
        ) {
            failures.push(format!(
                "token {} layer {} Wd: Freivalds weight-binding failed",
                token_pos, layer_idx
            ));
        }

        // Post-FFN bridge: derive x_attn for next layer
        let next_scale_x_attn = retained.layers
            .get(layer_idx + 1)
            .map(|r| r.scale_x_attn)
            .unwrap_or(1.0);

        x_attn = if let Some(ref mut res) = residual {
            if layer_idx + 1 < key.rmsnorm_attn_weights.len() {
                Some(verilm_core::rmsnorm::bridge_residual_rmsnorm(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    rs.scale_h,
                    res,
                    &key.rmsnorm_attn_weights[layer_idx + 1],
                    key.rmsnorm_eps,
                    next_scale_x_attn,
                ))
            } else {
                verilm_core::rmsnorm::dequant_add_residual(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    rs.scale_h,
                    res,
                );
                Some(verilm_core::bridge_requantize(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    rs.scale_h,
                    next_scale_x_attn,
                ))
            }
        } else {
            Some(verilm_core::bridge_requantize(
                &ffn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wd),
                rs.scale_h,
                next_scale_x_attn,
            ))
        };
    }

    (checks_run, failures)
}
