//! Verification library for the verified-inference protocol (V4).
//!
//! Provides key-only Freivalds verification against prover-supplied shell
//! openings. The verifier never recomputes matmuls — it checks the prover's
//! i32 accumulators with precomputed Freivalds keys and verifies bridge
//! consistency by deriving intermediate i8 values from the accumulators.

pub mod canonical;
pub mod client;
pub mod corridor;

use std::time::{Duration, Instant};

use verilm_core::constants::MatrixType;
use verilm_core::types::{
    AuditChallenge, AuditTier, CommitmentVersion, DeploymentManifest, InputSpec, OutputSpec,
    V4AuditResponse, VerifierKey,
};
use verilm_core::{freivalds, merkle};

pub use verilm_core::requantize;
pub use verilm_core::types::ShellWeights;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
}

/// Audit coverage level, derived from the shell opening's layer indices.
///
/// Distinguishes full audits (all layers checked) from routine/partial audits
/// (a contiguous prefix of layers). Consumers MUST NOT treat a routine-audit
/// pass as equivalent to a full-audit pass — routine audits provide statistical
/// detection, not exhaustive coverage.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "level", rename_all = "snake_case")]
pub enum AuditCoverage {
    /// All layers were opened and checked.
    Full { layers_checked: usize },
    /// A contiguous prefix of layers was opened (routine audit).
    Routine {
        layers_checked: usize,
        layers_total: usize,
    },
    /// Coverage could not be determined (no shell opening present).
    Unknown,
}

impl std::fmt::Display for AuditCoverage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Full { layers_checked } => {
                write!(f, "full ({}/{} layers)", layers_checked, layers_checked)
            }
            Self::Routine {
                layers_checked,
                layers_total,
            } => write!(f, "routine ({}/{} layers)", layers_checked, layers_total),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Classification of verification failures.
///
/// Every failure the verifier emits belongs to exactly one category.
/// Consumers can use this to route failures (e.g. cryptographic failures
/// are non-negotiable rejects; configuration failures may be actionable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    /// Payload could not be deserialized or is structurally invalid
    /// (wrong commitment version, missing required fields).
    Structural,
    /// Cryptographic binding check failed: Merkle proof, hash chain,
    /// Freivalds check, or commitment hash mismatch.
    CryptographicBinding,
    /// Committed spec/manifest value does not match the verifier key
    /// or the committed values are internally inconsistent.
    SpecMismatch,
    /// A feature or value is not supported by the canonical verifier
    /// and is rejected fail-closed (unknown sampler, unsupported decode
    /// mode, unknown policy).
    Unsupported,
    /// Token selection, output policy, or generation-length constraint
    /// violated (wrong argmax, exceeded max_tokens, EOS policy breach).
    SemanticViolation,
    /// Operational or configuration issue (tokenizer error, detokenizer
    /// error, missing external dependency).
    Operational,
}

impl std::fmt::Display for FailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Structural => write!(f, "structural"),
            Self::CryptographicBinding => write!(f, "cryptographic_binding"),
            Self::SpecMismatch => write!(f, "spec_mismatch"),
            Self::Unsupported => write!(f, "unsupported"),
            Self::SemanticViolation => write!(f, "semantic_violation"),
            Self::Operational => write!(f, "operational"),
        }
    }
}

/// Stable failure code for machine consumption.
///
/// Each variant maps to exactly one [`FailureCategory`]. Consumers match on
/// codes rather than parsing message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    // -- Structural --
    WrongCommitmentVersion,
    MissingShellOpening,
    MissingSeedCommitment,
    MissingPromptHash,
    MissingPromptBytes,
    UncommittedPrompt,
    MissingNPromptTokens,
    MissingSpecHash,
    MissingManifestHash,
    MissingInitialResidual,
    MissingEmbeddingProof,
    MissingLogits,
    MissingFinalHidden,
    MissingFinalResidual,
    MissingOutputText,
    MissingEosTokenId,
    MissingQkv,
    ShellLayerCountMismatch,
    NonContiguousLayerIndices,
    PrefixCountMismatch,
    UnboundInitialResidual,

    // -- Cryptographic binding --
    FreivaldsFailed,
    LmHeadFreivaldsFailed,
    MerkleProofFailed,
    RetainedHashMismatch,
    IoChainMismatch,
    IoChainProofFailed,
    SeedMismatch,
    PromptHashMismatch,
    ManifestHashMismatch,
    SpecHashMismatch,
    EmbeddingProofFailed,
    EmbeddingLeafMismatch,
    AttentionReplayMismatch,

    // -- Spec mismatch --
    SpecFieldMismatch,

    // -- Unsupported --
    UnsupportedSamplerVersion,
    UnsupportedDecodeMode,
    UnsupportedDecodeFeature,
    UnknownEosPolicy,

    // -- Challenge binding --
    ChallengeTokenMismatch,
    ChallengeLayerMismatch,

    // -- Bridge trust boundary --
    BridgeXAttnMismatch,
    BridgeScaleMismatch,

    // -- Score witnessing --
    ScoreAnchorMismatch,
    WitnessedScoreStructuralError,

    // -- KV transcript --
    KvRootsCountMismatch,
    KvEntriesCountMismatch,
    KvProofInvalid,
    KvProofCountMismatch,

    // -- Semantic violation --
    TokenSelectionMismatch,
    ExceedsMaxTokens,
    MinTokensViolated,
    EosPolicyViolated,
    IgnoreEosViolated,
    DecodeModeTempInconsistent,
    PromptTokenMismatch,
    PromptTokenCountMismatch,
    NPromptTokensMismatch,
    NPromptTokensBound,
    PrefixTokenCountMismatch,
    DetokenizationMismatch,

    // -- Operational --
    TokenizerError,
    DetokenizerError,
}

impl FailureCode {
    /// The category this code belongs to. Stable — each code maps to exactly one category.
    pub fn category(self) -> FailureCategory {
        use FailureCode::*;
        match self {
            WrongCommitmentVersion
            | MissingShellOpening
            | MissingSeedCommitment
            | MissingPromptHash
            | MissingPromptBytes
            | UncommittedPrompt
            | MissingNPromptTokens
            | MissingSpecHash
            | MissingManifestHash
            | MissingInitialResidual
            | MissingEmbeddingProof
            | MissingLogits
            | MissingFinalHidden
            | MissingFinalResidual
            | MissingOutputText
            | MissingEosTokenId
            | MissingQkv
            | ShellLayerCountMismatch
            | NonContiguousLayerIndices
            | PrefixCountMismatch
            | UnboundInitialResidual => FailureCategory::Structural,

            FreivaldsFailed
            | LmHeadFreivaldsFailed
            | MerkleProofFailed
            | RetainedHashMismatch
            | IoChainMismatch
            | IoChainProofFailed
            | SeedMismatch
            | PromptHashMismatch
            | ManifestHashMismatch
            | SpecHashMismatch
            | EmbeddingProofFailed
            | EmbeddingLeafMismatch
            | AttentionReplayMismatch
            | ChallengeTokenMismatch
            | ChallengeLayerMismatch
            | BridgeXAttnMismatch
            | BridgeScaleMismatch
            | KvRootsCountMismatch
            | KvEntriesCountMismatch
            | KvProofInvalid
            | KvProofCountMismatch
            | ScoreAnchorMismatch
            | WitnessedScoreStructuralError => FailureCategory::CryptographicBinding,

            SpecFieldMismatch => FailureCategory::SpecMismatch,

            UnsupportedSamplerVersion
            | UnsupportedDecodeMode
            | UnsupportedDecodeFeature
            | UnknownEosPolicy => FailureCategory::Unsupported,

            TokenSelectionMismatch
            | ExceedsMaxTokens
            | MinTokensViolated
            | EosPolicyViolated
            | IgnoreEosViolated
            | DecodeModeTempInconsistent
            | PromptTokenMismatch
            | PromptTokenCountMismatch
            | NPromptTokensMismatch
            | NPromptTokensBound
            | PrefixTokenCountMismatch
            | DetokenizationMismatch => FailureCategory::SemanticViolation,

            TokenizerError | DetokenizerError => FailureCategory::Operational,
        }
    }
}

impl std::fmt::Display for FailureCode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Use serde's rename_all = "snake_case" convention
        write!(
            f,
            "{}",
            serde_json::to_string(self)
                .unwrap_or_else(|_| format!("{:?}", self))
                .trim_matches('"')
        )
    }
}

/// Optional structured context for a verification failure.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct FailureContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
}

impl FailureContext {
    fn is_empty(&self) -> bool {
        self.token_index.is_none()
            && self.layer.is_none()
            && self.matrix.is_none()
            && self.field.is_none()
            && self.spec.is_none()
            && self.expected.is_none()
            && self.actual.is_none()
    }
}

/// A single classified verification failure with a stable code.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VerificationFailure {
    pub code: FailureCode,
    pub category: FailureCategory,
    pub message: String,
    #[serde(skip_serializing_if = "FailureContext::is_empty")]
    pub context: FailureContext,
}

impl std::fmt::Display for VerificationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{}] {}", self.category, self.message)
    }
}

/// Create a [`VerificationFailure`] from a code and message.
pub(crate) fn vfail(code: FailureCode, msg: impl Into<String>) -> VerificationFailure {
    VerificationFailure {
        category: code.category(),
        code,
        message: msg.into(),
        context: FailureContext::default(),
    }
}

/// Create a [`VerificationFailure`] with structured context.
pub(crate) fn vfail_ctx(
    code: FailureCode,
    msg: impl Into<String>,
    ctx: FailureContext,
) -> VerificationFailure {
    VerificationFailure {
        category: code.category(),
        code,
        message: msg.into(),
        context: ctx,
    }
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
) -> Vec<VerificationFailure> {
    let computed = merkle::hash_manifest(expected_manifest);
    match commitment.manifest_hash {
        None => vec![vfail(
            FailureCode::MissingManifestHash,
            "commitment missing manifest_hash",
        )],
        Some(h) if h != computed => vec![vfail(
            FailureCode::ManifestHashMismatch,
            "manifest_hash mismatch",
        )],
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
) -> Vec<VerificationFailure> {
    let computed = merkle::hash_manifest_composed(
        merkle::hash_input_spec(input),
        merkle::hash_model_spec(model),
        merkle::hash_decode_spec(decode),
        merkle::hash_output_spec(output),
    );
    match commitment.manifest_hash {
        None => vec![vfail(
            FailureCode::MissingManifestHash,
            "commitment missing manifest_hash",
        )],
        Some(h) if h != computed => vec![vfail(
            FailureCode::ManifestHashMismatch,
            "manifest_hash mismatch",
        )],
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
) -> Vec<VerificationFailure> {
    let mut failures = Vec::new();

    // Check prompt token count matches.
    if let Some(committed_npt) = response.n_prompt_tokens {
        if expected_prompt_token_ids.len() as u32 != committed_npt {
            failures.push(vfail(
                FailureCode::PromptTokenCountMismatch,
                format!(
                    "tokenization produced {} prompt tokens but commitment has n_prompt_tokens={}",
                    expected_prompt_token_ids.len(),
                    committed_npt
                ),
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
    let all_chain: Vec<u32> = response
        .prefix_token_ids
        .iter()
        .copied()
        .chain(std::iter::once(response.token_id))
        .collect();

    if n_prompt_in_chain > all_chain.len() {
        failures.push(vfail(
            FailureCode::PromptTokenCountMismatch,
            format!(
                "prompt has {} tokens in chain but response only has {} total chain entries",
                n_prompt_in_chain,
                all_chain.len()
            ),
        ));
        return failures;
    }

    // Check each prompt token (after the first) against the chain.
    for (i, &expected_tid) in expected_prompt_token_ids[1..].iter().enumerate() {
        if all_chain[i] != expected_tid {
            failures.push(vfail_ctx(
                FailureCode::PromptTokenMismatch,
                format!(
                    "prompt token mismatch at position {}: expected {} but chain has {}",
                    i + 1,
                    expected_tid,
                    all_chain[i]
                ),
                FailureContext {
                    token_index: Some((i + 1) as u32),
                    ..Default::default()
                },
            ));
        }
    }

    failures
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
    /// Structured failures with stable codes, categories, messages, and context.
    pub failures: Vec<VerificationFailure>,
    /// Audit coverage level: full (all layers), routine (prefix), or unknown.
    /// Consumers MUST distinguish routine-audit passes from full-audit passes.
    pub coverage: AuditCoverage,
    pub duration: Duration,
    /// Checks that were skipped because the profile does not support them.
    /// Each entry describes what was skipped and why.
    pub skipped: Vec<String>,
}

impl V4VerifyReport {
    /// Human-readable failure messages (backward-compatible accessor).
    pub fn failure_messages(&self) -> Vec<&str> {
        self.failures.iter().map(|f| f.message.as_str()).collect()
    }
}

impl std::fmt::Display for V4VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => {
                write!(
                    f,
                    "V4 PASS: token {} — {}/{} checks, coverage: {} ({:.1}ms)",
                    self.token_index,
                    self.checks_passed,
                    self.checks_run,
                    self.coverage,
                    self.duration.as_secs_f64() * 1000.0
                )?;
                for s in &self.skipped {
                    write!(f, "\n  [skipped] {}", s)?;
                }
                Ok(())
            }
            Verdict::Fail => {
                writeln!(
                    f,
                    "V4 FAIL: token {} — {} failures, coverage: {}",
                    self.token_index,
                    self.failures.len(),
                    self.coverage,
                )?;
                for fail in &self.failures {
                    writeln!(f, "  {}", fail)?;
                }
                for s in &self.skipped {
                    writeln!(f, "  [skipped] {}", s)?;
                }
                Ok(())
            }
        }
    }
}

/// Verify V4 audit response using the canonical verifier.
///
/// Delegates to [`canonical::verify_response`] — binary-only mental model,
/// key-only Freivalds, full-bridge-only, fail-closed on missing fields.
///
/// `expected_prompt_token_ids` is accepted for API compatibility but ignored;
/// pass a `PromptTokenizer` via [`verify_v4_full`] for tokenization checks.
pub fn verify_v4(
    key: &VerifierKey,
    response: &V4AuditResponse,
    _expected_prompt_token_ids: Option<&[u32]>,
) -> V4VerifyReport {
    canonical::verify_response(key, response, None, None)
}

/// Full verification with optional canonical tokenizer and detokenizer.
///
/// Delegates to [`canonical::verify_response`]. The canonical verifier
/// reconstructs prompt token IDs from `response.prompt` + manifest `InputSpec`
/// when a tokenizer is provided. `expected_prompt_token_ids` is accepted for
/// API compatibility but ignored — the canonical path requires a tokenizer.
pub fn verify_v4_full(
    key: &VerifierKey,
    response: &V4AuditResponse,
    _expected_prompt_token_ids: Option<&[u32]>,
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> V4VerifyReport {
    canonical::verify_response(key, response, tokenizer, detokenizer)
}

/// Legacy verification path — kept for differential testing and rollback.
///
/// This is the original monolithic verifier. It will be removed once the
/// canonical path is fully validated in production.
pub fn verify_v4_legacy(
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
                        failures.push(vfail(
                            FailureCode::TokenizerError,
                            format!("tokenizer reconstruction failed: {}", e),
                        ));
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
                            failures.push(vfail(
                                FailureCode::EmbeddingLeafMismatch,
                                format!(
                                    "embedding proof leaf_index {} != token_id {}",
                                    proof.leaf_index, response.token_id
                                ),
                            ));
                        } else if !verilm_core::merkle::verify(emb_root, &leaf, proof) {
                            failures.push(vfail(
                                FailureCode::EmbeddingProofFailed,
                                "embedding Merkle proof verification failed",
                            ));
                        }
                    }
                    (None, _) => {
                        failures.push(vfail(
                            FailureCode::MissingInitialResidual,
                            "key has embedding_merkle_root but shell missing initial_residual",
                        ));
                    }
                    (Some(_), None) => {
                        failures.push(vfail(
                            FailureCode::MissingEmbeddingProof,
                            "key has embedding_merkle_root but shell missing embedding_proof",
                        ));
                    }
                }
            } else if shell.initial_residual.is_some() {
                checks_run += 1;
                failures.push(vfail(
                    FailureCode::UnboundInitialResidual,
                    "shell has initial_residual but key has no embedding_merkle_root to verify it",
                ));
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
                    failures.push(vfail(
                        FailureCode::PrefixCountMismatch,
                        format!(
                            "prefix embedding count mismatch: {} rows, {} proofs, {} token_ids",
                            prefix_rows.len(),
                            prefix_proofs.len(),
                            response.prefix_token_ids.len()
                        ),
                    ));
                } else {
                    for (j, ((row, proof), &tid)) in prefix_rows
                        .iter()
                        .zip(prefix_proofs.iter())
                        .zip(response.prefix_token_ids.iter())
                        .enumerate()
                    {
                        checks_run += 1;
                        let leaf = verilm_core::merkle::hash_embedding_row(row);
                        if proof.leaf_index != tid {
                            failures.push(vfail_ctx(
                                FailureCode::EmbeddingLeafMismatch,
                                format!(
                                    "prefix token {}: embedding proof leaf_index {} != token_id {}",
                                    j, proof.leaf_index, tid
                                ),
                                FailureContext {
                                    token_index: Some(j as u32),
                                    ..Default::default()
                                },
                            ));
                        } else if !verilm_core::merkle::verify(emb_root, &leaf, proof) {
                            failures.push(vfail_ctx(
                                FailureCode::EmbeddingProofFailed,
                                format!(
                                    "prefix token {}: embedding Merkle proof verification failed",
                                    j
                                ),
                                FailureContext {
                                    token_index: Some(j as u32),
                                    ..Default::default()
                                },
                            ));
                        }
                    }
                }
            }

            // Deep prefix: verify prefix_retained[j] hashes to prefix_leaf_hashes[j],
            // then run shell Freivalds+bridge on each prefix token.
            if let (Some(ref prefix_ret), Some(ref prefix_shells)) =
                (&response.prefix_retained, &response.prefix_shell_openings)
            {
                if prefix_ret.len() != response.prefix_leaf_hashes.len()
                    || prefix_shells.len() != response.prefix_leaf_hashes.len()
                {
                    checks_run += 1;
                    failures.push(vfail(
                        FailureCode::PrefixCountMismatch,
                        format!(
                            "deep prefix count mismatch: {} retained, {} shells, {} leaf_hashes",
                            prefix_ret.len(),
                            prefix_shells.len(),
                            response.prefix_leaf_hashes.len()
                        ),
                    ));
                } else {
                    for (j, ((ret_j, shell_j), &expected_hash)) in prefix_ret
                        .iter()
                        .zip(prefix_shells.iter())
                        .zip(response.prefix_leaf_hashes.iter())
                        .enumerate()
                    {
                        // 1. Hash consistency: retained state must match committed leaf.
                        checks_run += 1;
                        let fr_ref = shell_j.final_residual.as_deref();
                        let lp_ref = shell_j.lp_hidden_bf16.as_deref();
                        let hash_j =
                            verilm_core::merkle::hash_retained_with_lp_hidden(ret_j, fr_ref, lp_ref);
                        if hash_j != expected_hash {
                            failures.push(vfail_ctx(
                                FailureCode::RetainedHashMismatch,
                                format!("prefix token {}: retained hash mismatch (deep audit)", j),
                                FailureContext {
                                    token_index: Some(j as u32),
                                    ..Default::default()
                                },
                            ));
                            continue;
                        }

                        // 2. Shell Freivalds + bridge verification (same as challenged token).
                        let (c, f, _) = verify_shell_opening(key, ret_j, shell_j);
                        checks_run += c;
                        for mut failure in f {
                            failure.message = format!("prefix token {}: {}", j, failure.message);
                            failures.push(failure);
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
                    None => failures.push(vfail_ctx(
                        FailureCode::MissingSpecHash,
                        "commitment missing input_spec_hash",
                        FailureContext {
                            spec: Some("input".into()),
                            ..Default::default()
                        },
                    )),
                    Some(committed) if h_in != committed => failures.push(vfail_ctx(
                        FailureCode::SpecHashMismatch,
                        "input_spec_hash mismatch",
                        FailureContext {
                            spec: Some("input".into()),
                            ..Default::default()
                        },
                    )),
                    _ => {}
                }
                match response.commitment.model_spec_hash {
                    None => failures.push(vfail_ctx(
                        FailureCode::MissingSpecHash,
                        "commitment missing model_spec_hash",
                        FailureContext {
                            spec: Some("model".into()),
                            ..Default::default()
                        },
                    )),
                    Some(committed) if h_mod != committed => failures.push(vfail_ctx(
                        FailureCode::SpecHashMismatch,
                        "model_spec_hash mismatch",
                        FailureContext {
                            spec: Some("model".into()),
                            ..Default::default()
                        },
                    )),
                    _ => {}
                }

                // Cross-check: manifest rmsnorm_eps must agree with verifier key.
                if let Some(manifest_eps) = model_spec.rmsnorm_eps {
                    checks_run += 1;
                    if (manifest_eps - key.rmsnorm_eps).abs() > f64::EPSILON {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "rmsnorm_eps mismatch: manifest={} key={}",
                                manifest_eps, key.rmsnorm_eps
                            ),
                            FailureContext {
                                field: Some("rmsnorm_eps".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: manifest rope_config_hash must agree with verifier key.
                if let (Some(manifest_rope), Some(key_rope)) =
                    (model_spec.rope_config_hash, key.rope_config_hash)
                {
                    checks_run += 1;
                    if manifest_rope != key_rope {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            "rope_config_hash mismatch: manifest != key",
                            FailureContext {
                                field: Some("rope_config_hash".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: manifest weight_hash (R_W) must agree with verifier key.
                if let (Some(manifest_rw), Some(key_rw)) = (model_spec.weight_hash, key.weight_hash)
                {
                    checks_run += 1;
                    if manifest_rw != key_rw {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            "weight_hash mismatch: manifest != key",
                            FailureContext {
                                field: Some("weight_hash".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: n_layers
                if let Some(n) = model_spec.n_layers {
                    checks_run += 1;
                    if n as usize != key.config.n_layers {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "n_layers mismatch: manifest={} key={}",
                                n, key.config.n_layers
                            ),
                            FailureContext {
                                field: Some("n_layers".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: hidden_dim
                if let Some(d) = model_spec.hidden_dim {
                    checks_run += 1;
                    if d as usize != key.config.hidden_dim {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "hidden_dim mismatch: manifest={} key={}",
                                d, key.config.hidden_dim
                            ),
                            FailureContext {
                                field: Some("hidden_dim".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: vocab_size
                if let Some(v) = model_spec.vocab_size {
                    checks_run += 1;
                    if v as usize != key.config.vocab_size {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "vocab_size mismatch: manifest={} key={}",
                                v, key.config.vocab_size
                            ),
                            FailureContext {
                                field: Some("vocab_size".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: embedding_merkle_root
                if let (Some(manifest_emr), Some(key_emr)) =
                    (model_spec.embedding_merkle_root, key.embedding_merkle_root)
                {
                    checks_run += 1;
                    if manifest_emr != key_emr {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            "embedding_merkle_root mismatch: manifest != key",
                            FailureContext {
                                field: Some("embedding_merkle_root".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: quant_family
                if let (Some(ref manifest_qf), Some(ref key_qf)) =
                    (&model_spec.quant_family, &key.quant_family)
                {
                    checks_run += 1;
                    if manifest_qf != key_qf {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "quant_family mismatch: manifest='{}' key='{}'",
                                manifest_qf, key_qf
                            ),
                            FailureContext {
                                field: Some("quant_family".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: scale_derivation
                if let (Some(ref manifest_sd), Some(ref key_sd)) =
                    (&model_spec.scale_derivation, &key.scale_derivation)
                {
                    checks_run += 1;
                    if manifest_sd != key_sd {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "scale_derivation mismatch: manifest='{}' key='{}'",
                                manifest_sd, key_sd
                            ),
                            FailureContext {
                                field: Some("scale_derivation".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: quant_block_size
                if let (Some(manifest_bs), Some(key_bs)) =
                    (model_spec.quant_block_size, key.quant_block_size)
                {
                    checks_run += 1;
                    if manifest_bs != key_bs {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "quant_block_size mismatch: manifest={} key={}",
                                manifest_bs, key_bs
                            ),
                            FailureContext {
                                field: Some("quant_block_size".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: kv_dim
                if let Some(v) = model_spec.kv_dim {
                    checks_run += 1;
                    if v as usize != key.config.kv_dim {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!("kv_dim mismatch: manifest={} key={}", v, key.config.kv_dim),
                            FailureContext {
                                field: Some("kv_dim".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: ffn_dim
                if let Some(v) = model_spec.ffn_dim {
                    checks_run += 1;
                    if v as usize != key.config.ffn_dim {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "ffn_dim mismatch: manifest={} key={}",
                                v, key.config.ffn_dim
                            ),
                            FailureContext {
                                field: Some("ffn_dim".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: d_head
                if let Some(v) = model_spec.d_head {
                    checks_run += 1;
                    if v as usize != key.config.d_head {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!("d_head mismatch: manifest={} key={}", v, key.config.d_head),
                            FailureContext {
                                field: Some("d_head".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: n_q_heads
                if let Some(v) = model_spec.n_q_heads {
                    checks_run += 1;
                    if v as usize != key.config.n_q_heads {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "n_q_heads mismatch: manifest={} key={}",
                                v, key.config.n_q_heads
                            ),
                            FailureContext {
                                field: Some("n_q_heads".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: n_kv_heads
                if let Some(v) = model_spec.n_kv_heads {
                    checks_run += 1;
                    if v as usize != key.config.n_kv_heads {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "n_kv_heads mismatch: manifest={} key={}",
                                v, key.config.n_kv_heads
                            ),
                            FailureContext {
                                field: Some("n_kv_heads".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                // Cross-check: rope_theta
                if let Some(manifest_theta) = model_spec.rope_theta {
                    checks_run += 1;
                    if (manifest_theta - key.config.rope_theta).abs() > f64::EPSILON {
                        failures.push(vfail_ctx(
                            FailureCode::SpecFieldMismatch,
                            format!(
                                "rope_theta mismatch: manifest={} key={}",
                                manifest_theta, key.config.rope_theta
                            ),
                            FailureContext {
                                field: Some("rope_theta".into()),
                                ..Default::default()
                            },
                        ));
                    }
                }

                match response.commitment.decode_spec_hash {
                    None => failures.push(vfail_ctx(
                        FailureCode::MissingSpecHash,
                        "commitment missing decode_spec_hash",
                        FailureContext {
                            spec: Some("decode".into()),
                            ..Default::default()
                        },
                    )),
                    Some(committed) if h_dec != committed => failures.push(vfail_ctx(
                        FailureCode::SpecHashMismatch,
                        "decode_spec_hash mismatch",
                        FailureContext {
                            spec: Some("decode".into()),
                            ..Default::default()
                        },
                    )),
                    _ => {}
                }
                match response.commitment.output_spec_hash {
                    None => failures.push(vfail_ctx(
                        FailureCode::MissingSpecHash,
                        "commitment missing output_spec_hash",
                        FailureContext {
                            spec: Some("output".into()),
                            ..Default::default()
                        },
                    )),
                    Some(committed) if h_out != committed => failures.push(vfail_ctx(
                        FailureCode::SpecHashMismatch,
                        "output_spec_hash mismatch",
                        FailureContext {
                            spec: Some("output".into()),
                            ..Default::default()
                        },
                    )),
                    _ => {}
                }

                // Verify composed manifest hash: M = H(H_input || H_model || H_decode || H_output).
                checks_run += 1;
                match response.commitment.manifest_hash {
                    None => failures.push(vfail(
                        FailureCode::MissingManifestHash,
                        "commitment missing manifest_hash",
                    )),
                    Some(committed_hash) => {
                        let computed = merkle::hash_manifest_composed(h_in, h_mod, h_dec, h_out);
                        if computed != committed_hash {
                            failures.push(vfail(
                                FailureCode::ManifestHashMismatch,
                                "manifest hash does not match commitment",
                            ));
                        }
                    }
                }

                // Reject unknown sampler versions: fail-closed.
                checks_run += 1;
                match decode_spec.sampler_version.as_deref() {
                    Some("chacha20-vi-sample-v1") | None => {} // supported
                    Some(other) => {
                        failures.push(vfail(
                            FailureCode::UnsupportedSamplerVersion,
                            format!(
                                "unsupported sampler_version='{}' \
                                 (expected 'chacha20-vi-sample-v1' or absent)",
                                other
                            ),
                        ));
                    }
                }

                // Cross-check: decode_mode must be consistent with temperature.
                if let Some(ref dm) = decode_spec.decode_mode {
                    checks_run += 1;
                    match dm.as_str() {
                        "greedy" => {
                            if decode_spec.temperature != 0.0 {
                                failures.push(vfail(
                                    FailureCode::DecodeModeTempInconsistent,
                                    format!(
                                        "decode_mode='greedy' but temperature={} (must be 0.0)",
                                        decode_spec.temperature
                                    ),
                                ));
                            }
                        }
                        "sampled" => {
                            if decode_spec.temperature == 0.0 {
                                failures.push(vfail(
                                    FailureCode::DecodeModeTempInconsistent,
                                    "decode_mode='sampled' but temperature=0.0 (must be >0.0)",
                                ));
                            }
                        }
                        other => {
                            failures.push(vfail(
                                FailureCode::UnsupportedDecodeMode,
                                format!(
                                    "unsupported decode_mode='{}' (expected 'greedy' or 'sampled')",
                                    other
                                ),
                            ));
                        }
                    }
                }

                // Reject decode parameters the canonical sampler doesn't support.
                // These are bound in the spec hash, so a prover can't hide them.
                checks_run += 1;
                if decode_spec.repetition_penalty != 1.0 {
                    failures.push(vfail_ctx(
                        FailureCode::UnsupportedDecodeFeature,
                        format!(
                            "unsupported repetition_penalty={} (canonical sampler requires 1.0)",
                            decode_spec.repetition_penalty
                        ),
                        FailureContext {
                            field: Some("repetition_penalty".into()),
                            ..Default::default()
                        },
                    ));
                }
                if decode_spec.frequency_penalty != 0.0 {
                    failures.push(vfail_ctx(
                        FailureCode::UnsupportedDecodeFeature,
                        format!(
                            "unsupported frequency_penalty={} (canonical sampler requires 0.0)",
                            decode_spec.frequency_penalty
                        ),
                        FailureContext {
                            field: Some("frequency_penalty".into()),
                            ..Default::default()
                        },
                    ));
                }
                if decode_spec.presence_penalty != 0.0 {
                    failures.push(vfail_ctx(
                        FailureCode::UnsupportedDecodeFeature,
                        format!(
                            "unsupported presence_penalty={} (canonical sampler requires 0.0)",
                            decode_spec.presence_penalty
                        ),
                        FailureContext {
                            field: Some("presence_penalty".into()),
                            ..Default::default()
                        },
                    ));
                }
                if !decode_spec.logit_bias.is_empty() {
                    failures.push(vfail_ctx(
                        FailureCode::UnsupportedDecodeFeature,
                        format!(
                            "unsupported logit_bias ({} entries, canonical sampler requires empty)",
                            decode_spec.logit_bias.len()
                        ),
                        FailureContext {
                            field: Some("logit_bias".into()),
                            ..Default::default()
                        },
                    ));
                }
                if !decode_spec.bad_word_ids.is_empty() {
                    failures.push(vfail_ctx(
                        FailureCode::UnsupportedDecodeFeature,
                        format!("unsupported bad_word_ids ({} entries, canonical sampler requires empty)", decode_spec.bad_word_ids.len()),
                        FailureContext { field: Some("bad_word_ids".into()), ..Default::default() },
                    ));
                }
                if !decode_spec.guided_decoding.is_empty() {
                    failures.push(vfail_ctx(
                        FailureCode::UnsupportedDecodeFeature,
                        format!(
                            "unsupported guided_decoding='{}' (canonical sampler requires empty)",
                            decode_spec.guided_decoding
                        ),
                        FailureContext {
                            field: Some("guided_decoding".into()),
                            ..Default::default()
                        },
                    ));
                }

                // Output spec checks.
                if !output_spec.stop_sequences.is_empty() {
                    failures.push(vfail_ctx(
                        FailureCode::UnsupportedDecodeFeature,
                        format!("unsupported stop_sequences ({} entries, canonical sampler requires empty)", output_spec.stop_sequences.len()),
                        FailureContext { field: Some("stop_sequences".into()), ..Default::default() },
                    ));
                }
                if output_spec.max_tokens > 0 {
                    // The committed token_ids array omits the first embedding token,
                    // so positions 0..n_prompt-2 are prompt tokens and n_prompt-1..
                    // are generated tokens. Thus:
                    //   n_generated = n_tokens - (n_prompt - 1) = n_tokens - n_prompt + 1
                    let n_prompt = response.commitment.n_prompt_tokens.unwrap_or(0);
                    let n_generated = response
                        .commitment
                        .n_tokens
                        .saturating_sub(n_prompt.saturating_sub(1));
                    if response.token_index >= response.commitment.n_tokens {
                        failures.push(vfail(
                            FailureCode::ExceedsMaxTokens,
                            format!(
                                "token_index {} exceeds committed n_tokens {}",
                                response.token_index, response.commitment.n_tokens
                            ),
                        ));
                    }
                    if n_generated > output_spec.max_tokens {
                        failures.push(vfail(
                            FailureCode::ExceedsMaxTokens,
                            format!(
                                "generated {} tokens exceeds output_spec max_tokens {}",
                                n_generated, output_spec.max_tokens
                            ),
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
                        let gen_start = response.n_prompt_tokens.unwrap_or(1).saturating_sub(1);
                        // The number of generated tokens committed.
                        let n_generated = response.commitment.n_tokens.saturating_sub(gen_start);
                        if n_generated < output_spec.min_tokens {
                            failures.push(vfail(
                                FailureCode::MinTokensViolated,
                                format!(
                                    "committed only {} generated tokens but output_spec requires min_tokens={}",
                                    n_generated, output_spec.min_tokens
                                ),
                            ));
                        }
                        // If the challenged token is within the min_tokens window and is EOS, reject.
                        let gen_index = response.token_index.saturating_sub(gen_start);
                        if gen_index < output_spec.min_tokens && response.token_id == eos_id {
                            failures.push(vfail(
                                FailureCode::MinTokensViolated,
                                format!(
                                    "token at generation position {} is EOS ({}) but min_tokens={}",
                                    gen_index, eos_id, output_spec.min_tokens
                                ),
                            ));
                        }
                    }

                    // ignore_eos: if true, the model must never select EOS.
                    if output_spec.ignore_eos {
                        checks_run += 1;
                        if response.token_id == eos_id {
                            failures.push(vfail(
                                FailureCode::IgnoreEosViolated,
                                format!(
                                    "token_id {} is EOS but ignore_eos=true",
                                    response.token_id
                                ),
                            ));
                        }
                    }
                } else if output_spec.min_tokens > 0 || output_spec.ignore_eos {
                    // Fail-closed: can't enforce min_tokens or ignore_eos without eos_token_id.
                    failures.push(vfail(
                        FailureCode::MissingEosTokenId,
                        "output_spec requires min_tokens or ignore_eos but eos_token_id is missing",
                    ));
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
                        let is_last_token =
                            response.token_index == response.commitment.n_tokens.saturating_sub(1);
                        if response.token_id == eos_id && !is_last_token {
                            failures.push(vfail(
                                FailureCode::EosPolicyViolated,
                                format!(
                                    "eos_policy='stop' but EOS token ({}) at index {} is not the last token (n_tokens={})",
                                    eos_id, response.token_index, response.commitment.n_tokens
                                ),
                            ));
                        }
                    }
                } else if output_spec.eos_policy != "sample" {
                    // Only "stop" and "sample" are recognized. Reject unknown policies.
                    failures.push(vfail(
                        FailureCode::UnknownEosPolicy,
                        format!(
                            "unknown eos_policy='{}' (expected 'stop' or 'sample')",
                            output_spec.eos_policy
                        ),
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
                    let normed =
                        verilm_core::rmsnorm::rmsnorm_f64_input(&res_f64, fnw, key.rmsnorm_eps);
                    Some(
                        normed
                            .iter()
                            .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
                            .collect(),
                    )
                } else {
                    // No final_norm_weights in key — fall back to shell replay (toy model).
                    shell_final_hidden
                }
            } else if key.final_norm_weights.is_some() && key.lm_head.is_some() {
                // Fail-closed: key requires exact verification but no captured state.
                failures.push(vfail(
                    FailureCode::MissingFinalResidual,
                    "key has lm_head + final_norm_weights but shell missing final_residual \
                     (exact tail verification required, cannot fall back to shell replay)",
                ));
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
                            failures.push(vfail(
                                FailureCode::LmHeadFreivaldsFailed,
                                "lm_head: Freivalds check failed on prover's logits_i32 claim",
                            ));
                        }

                        // Token replay: use the prover's (now Freivalds-verified) logits.
                        // Only replay for generated tokens — prompt-side tokens were
                        // chosen by the tokenizer, not by argmax/sampling over logits.
                        let gen_start = response
                            .n_prompt_tokens
                            .map(|npt| npt.saturating_sub(1))
                            .unwrap_or(0);
                        let is_generated = response.token_index >= gen_start;

                        if key.config.vocab_size > 0 && is_generated {
                            checks_run += 1;
                            let logits: Vec<f32> =
                                claimed_logits.iter().map(|&v| v as f32).collect();

                            let expected_token = if let Some(ref dp) = decode_params {
                                let token_seed = verilm_core::sampling::derive_token_seed(
                                    &response.revealed_seed,
                                    response.token_index,
                                );
                                verilm_core::sampling::sample(&logits, dp, &token_seed)
                            } else {
                                logits
                                    .iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .map(|(i, _)| i as u32)
                                    .unwrap_or(0)
                            };

                            if expected_token != response.token_id {
                                failures.push(vfail(
                                    FailureCode::TokenSelectionMismatch,
                                    format!(
                                        "lm_head: expected token_id={} but claimed token_id={}",
                                        expected_token, response.token_id
                                    ),
                                ));
                            }
                        }
                    }
                    (None, _) => {
                        failures.push(vfail(
                            FailureCode::MissingLogits,
                            "lm_head: key requires logits_i32 but shell opening did not provide it",
                        ));
                    }
                    (Some(_), None) => {
                        failures.push(vfail(
                            FailureCode::MissingFinalHidden,
                            "lm_head: logits_i32 present but no final_hidden to check against",
                        ));
                    }
                }
            }
        }
        None => {
            failures.push(vfail(
                FailureCode::MissingShellOpening,
                "V4 audit response missing shell_opening",
            ));
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
                let detok_policy = response
                    .manifest
                    .as_ref()
                    .map(|m| OutputSpec::from(m))
                    .and_then(|os| os.detokenization_policy);

                // Collect generation token IDs: prefix after prompt boundary + challenged token.
                let gen_start = response.n_prompt_tokens.unwrap_or(1).saturating_sub(1) as usize;
                let mut gen_token_ids: Vec<u32> = response
                    .prefix_token_ids
                    .get(gen_start..)
                    .unwrap_or(&[])
                    .to_vec();
                gen_token_ids.push(response.token_id);

                let is_last_token =
                    response.token_index == response.commitment.n_tokens.saturating_sub(1);

                match detok.decode(&gen_token_ids, detok_policy.as_deref()) {
                    Ok(decoded) => {
                        if is_last_token {
                            // Full generation — exact match required.
                            if decoded != *claimed_text {
                                failures.push(vfail(
                                    FailureCode::DetokenizationMismatch,
                                    format!(
                                        "detokenization mismatch (policy={:?}): decoded={:?} vs claimed={:?}",
                                        detok_policy, decoded, claimed_text
                                    ),
                                ));
                            }
                        } else {
                            // Partial generation — decoded must be a prefix of the claimed text.
                            if !claimed_text.starts_with(&decoded) {
                                failures.push(vfail(
                                    FailureCode::DetokenizationMismatch,
                                    format!(
                                        "detokenization prefix mismatch (policy={:?}): decoded={:?} is not a prefix of claimed={:?}",
                                        detok_policy, decoded, claimed_text
                                    ),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        failures.push(vfail(
                            FailureCode::DetokenizerError,
                            format!("detokenization failed: {}", e),
                        ));
                    }
                }
            }
            None => {
                failures.push(vfail(
                    FailureCode::MissingOutputText,
                    "detokenizer provided but response missing output_text \
                     (prover must include claimed output text for detokenization verification)",
                ));
            }
        }
    }

    // Compute audit coverage from the shell opening's layer_indices.
    let coverage = match &response.shell_opening {
        Some(shell) => {
            let layers_checked = shell
                .layer_indices
                .as_ref()
                .map_or(shell.layers.len(), |v| v.len());
            if layers_checked >= key.config.n_layers {
                AuditCoverage::Full { layers_checked }
            } else {
                AuditCoverage::Routine {
                    layers_checked,
                    layers_total: key.config.n_layers,
                }
            }
        }
        None => AuditCoverage::Unknown,
    };

    let duration = start.elapsed();
    let checks_passed = checks_run.saturating_sub(failures.len());

    V4VerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        token_index: response.token_index,
        checks_run,
        checks_passed,
        failures,
        coverage,
        duration,
        skipped: Vec::new(), // legacy path does not support tier-aware skipping
    }
}

/// Verify shell opening for one token using key-only Freivalds + bridge checks.
///
/// The verifier does NOT recompute matmuls — it checks the prover's i32
/// accumulators with precomputed Freivalds keys and verifies bridge
/// consistency by deriving intermediate i8 values from the accumulators.
///
/// **Canonical path**: when `shell.initial_residual` is present and the key
/// has RMSNorm weights, uses the full bridge
/// (dequant → residual += → RMSNorm → quantize) via `bridge_residual_rmsnorm()`.
/// This enables QKV Freivalds at all layers including layer 0.
///
/// **Toy fallback**: when residual data is absent (toy-model tests), falls
/// back to `bridge_requantize()` with no residual tracking. QKV at layer 0
/// is skipped. Not valid for production W8A8 verification.
///
/// Returns `(checks_run, failures, final_hidden)`. `final_hidden` is the
/// i8 hidden state after the last layer, derived from the bridge chain.
/// Used by verify_v4 for lm_head logit verification.
fn verify_shell_opening(
    key: &VerifierKey,
    retained: &verilm_core::types::RetainedTokenState,
    shell: &verilm_core::types::ShellTokenOpening,
) -> (usize, Vec<VerificationFailure>, Option<Vec<i8>>) {
    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // Resolve which layers are present.
    let opened_layers: Vec<usize> = shell
        .layer_indices
        .clone()
        .unwrap_or_else(|| (0..retained.layers.len()).collect());

    if shell.layers.len() != opened_layers.len() {
        failures.push(vfail(
            FailureCode::ShellLayerCountMismatch,
            format!(
                "shell_opening has {} layers but layer_indices specifies {}",
                shell.layers.len(),
                opened_layers.len()
            ),
        ));
        return (checks_run, failures, None);
    }

    // layer_indices must be a contiguous prefix 0..=L_max for the bridge chain.
    // Non-contiguous indices would cause the residual chain to silently skip layers.
    if let Some(indices) = &shell.layer_indices {
        let is_contiguous = indices.iter().enumerate().all(|(i, &li)| li == i);
        if !is_contiguous {
            failures.push(vfail(
                FailureCode::NonContiguousLayerIndices,
                format!(
                    "layer_indices must be a contiguous prefix 0..N, got {:?}",
                    indices
                ),
            ));
            return (checks_run, failures, None);
        }
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
            &res,
            &key.rmsnorm_attn_weights[0],
            key.rmsnorm_eps,
        );
        let xa =
            verilm_core::rmsnorm::quantize_f64_to_i8(&normed, shell.layers[0].scale_x_attn as f64);
        (Some(res), Some(xa))
    } else {
        (None, None)
    };

    // Iterate 0..=max_layer: bridge is sequential, Freivalds only on opened layers.
    for layer_idx in 0..=max_layer {
        if layer_idx >= retained.layers.len() {
            break;
        }
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
                            if !freivalds::check(key.v_for(layer_idx, *mt), xa, key.r_for(*mt), z) {
                                failures.push(vfail_ctx(
                                    FailureCode::FreivaldsFailed,
                                    format!(
                                        "layer {} {:?}: Freivalds failed on shell opening",
                                        layer_idx, mt
                                    ),
                                    FailureContext {
                                        layer: Some(layer_idx),
                                        matrix: Some(format!("{:?}", mt)),
                                        ..Default::default()
                                    },
                                ));
                            }
                        }
                        None => {
                            failures.push(vfail_ctx(
                                FailureCode::MissingQkv,
                                format!(
                                    "layer {} {:?}: shell opening missing QKV (x_attn derivable)",
                                    layer_idx, mt
                                ),
                                FailureContext {
                                    layer: Some(layer_idx),
                                    matrix: Some(format!("{:?}", mt)),
                                    ..Default::default()
                                },
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
                failures.push(vfail_ctx(
                    FailureCode::FreivaldsFailed,
                    format!("layer {} Wo: Freivalds failed on shell opening", layer_idx),
                    FailureContext {
                        layer: Some(layer_idx),
                        matrix: Some("Wo".into()),
                        ..Default::default()
                    },
                ));
            }

            // Post-attention bridge: derive x_ffn
            let x_ffn = if let Some(ref mut res) = residual {
                // Canonical: dequant → residual += attn_out → RMSNorm_ffn → quantize
                verilm_core::rmsnorm::bridge_residual_rmsnorm(
                    &sl.attn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wo),
                    rs.scale_a,
                    res,
                    &key.rmsnorm_ffn_weights[layer_idx],
                    key.rmsnorm_eps,
                    sl.scale_x_ffn,
                )
            } else {
                // Toy-model fallback: no residual, no RMSNorm
                verilm_core::bridge_requantize(
                    &sl.attn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wo),
                    rs.scale_a,
                    sl.scale_x_ffn,
                )
            };

            // W_g, W_u @ x_ffn
            for (mt, z) in [(MatrixType::Wg, &sl.g), (MatrixType::Wu, &sl.u)] {
                checks_run += 1;
                if !freivalds::check(key.v_for(layer_idx, mt), &x_ffn, key.r_for(mt), z) {
                    failures.push(vfail_ctx(
                        FailureCode::FreivaldsFailed,
                        format!(
                            "layer {} {:?}: Freivalds failed on shell opening",
                            layer_idx, mt
                        ),
                        FailureContext {
                            layer: Some(layer_idx),
                            matrix: Some(format!("{:?}", mt)),
                            ..Default::default()
                        },
                    ));
                }
            }

            // Bridge: verifier derives h from prover's g, u using scales
            let h = verilm_core::silu::compute_h_scaled(
                &sl.g,
                &sl.u,
                key.weight_scale_for(layer_idx, MatrixType::Wg),
                key.weight_scale_for(layer_idx, MatrixType::Wu),
                sl.scale_x_ffn,
                sl.scale_h,
            );

            // W_d @ h
            checks_run += 1;
            if !freivalds::check(
                key.v_for(layer_idx, MatrixType::Wd),
                &h,
                key.r_for(MatrixType::Wd),
                &sl.ffn_out,
            ) {
                failures.push(vfail_ctx(
                    FailureCode::FreivaldsFailed,
                    format!("layer {} Wd: Freivalds failed on shell opening", layer_idx),
                    FailureContext {
                        layer: Some(layer_idx),
                        matrix: Some("Wd".into()),
                        ..Default::default()
                    },
                ));
            }

            // Post-FFN bridge: derive x_attn for next layer
            let next_scale_x_attn = shell
                .layers
                .get(si + 1)
                .map(|s| s.scale_x_attn)
                .unwrap_or(1.0);

            x_attn = if let Some(ref mut res) = residual {
                if layer_idx + 1 < key.rmsnorm_attn_weights.len() {
                    // Canonical: dequant → residual += ffn_out → RMSNorm_attn_{l+1} → quantize
                    Some(verilm_core::rmsnorm::bridge_residual_rmsnorm(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        sl.scale_h,
                        res,
                        &key.rmsnorm_attn_weights[layer_idx + 1],
                        key.rmsnorm_eps,
                        next_scale_x_attn,
                    ))
                } else {
                    // Last layer: update residual for final_hidden, no subsequent RMSNorm
                    verilm_core::rmsnorm::dequant_add_residual(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        sl.scale_h,
                        res,
                    );
                    Some(verilm_core::bridge_requantize(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        sl.scale_h,
                        next_scale_x_attn,
                    ))
                }
            } else {
                // Toy-model fallback: no residual, no RMSNorm
                Some(verilm_core::bridge_requantize(
                    &sl.ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    sl.scale_h,
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
    let final_hidden = if let (Some(ref res), Some(ref fnw)) = (&residual, &key.final_norm_weights)
    {
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(res, fnw, key.rmsnorm_eps);
        // Quantize to i8 with unit scale (simple clamp) for lm_head matmul.
        Some(
            normed
                .iter()
                .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
                .collect(),
        )
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
        response
            .shell_opening
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

    // Weight-backed verification always replays all layers.
    let coverage = AuditCoverage::Full {
        layers_checked: key.config.n_layers,
    };

    let duration = start.elapsed();
    let checks_passed = checks_run.saturating_sub(failures.len());

    V4VerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        token_index: response.token_index,
        checks_run,
        checks_passed,
        failures,
        coverage,
        duration,
        skipped: Vec::new(), // legacy path does not support tier-aware skipping
    }
}

/// Structural checks shared by verify_v4 and verify_v4_with_replay.
fn verify_v4_structural(response: &V4AuditResponse) -> (usize, Vec<VerificationFailure>) {
    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // 1. Commitment version must be V4
    checks_run += 1;
    if response.commitment.version != CommitmentVersion::V4 {
        failures.push(vfail(
            FailureCode::WrongCommitmentVersion,
            format!(
                "expected V4 commitment, got {:?}",
                response.commitment.version
            ),
        ));
    }

    // 2. Seed commitment: hash(revealed_seed) == commitment.seed_commitment
    checks_run += 1;
    match response.commitment.seed_commitment {
        Some(expected) => {
            let computed = merkle::hash_seed(&response.revealed_seed);
            if computed != expected {
                failures.push(vfail(
                    FailureCode::SeedMismatch,
                    "seed commitment mismatch: hash(revealed_seed) != commitment.seed_commitment",
                ));
            }
        }
        None => {
            failures.push(vfail(
                FailureCode::MissingSeedCommitment,
                "V4 commitment missing seed_commitment",
            ));
        }
    }

    // 3. Prompt hash binding (fail-closed): prompt MUST be present and hash MUST match.
    checks_run += 1;
    match (&response.prompt, response.commitment.prompt_hash) {
        (Some(prompt), Some(committed_hash)) => {
            let computed = merkle::hash_prompt(prompt);
            if computed != committed_hash {
                failures.push(vfail(
                    FailureCode::PromptHashMismatch,
                    "prompt_hash mismatch: hash(prompt) != commitment.prompt_hash",
                ));
            }
        }
        (None, Some(_)) => {
            failures.push(vfail(
                FailureCode::MissingPromptBytes,
                "commitment has prompt_hash but response missing prompt bytes",
            ));
        }
        (Some(_), None) => {
            failures.push(vfail(
                FailureCode::UncommittedPrompt,
                "response has prompt but commitment missing prompt_hash",
            ));
        }
        (None, None) => {
            failures.push(vfail(
                FailureCode::MissingPromptHash,
                "V4 commitment missing prompt_hash",
            ));
        }
    }

    // 3b. Prompt token count binding (fail-closed): n_prompt_tokens MUST be present.
    checks_run += 1;
    match (
        response.commitment.n_prompt_tokens,
        response.n_prompt_tokens,
    ) {
        (Some(committed_npt), Some(response_npt)) => {
            if committed_npt != response_npt {
                failures.push(vfail(
                    FailureCode::NPromptTokensMismatch,
                    format!(
                        "n_prompt_tokens mismatch: commitment={} response={}",
                        committed_npt, response_npt
                    ),
                ));
            }
            // Sanity: n_prompt_tokens must be <= total tokens + 1 (the +1 is the
            // first token consumed as embedding input, not in the committed array).
            if committed_npt > response.commitment.n_tokens + 1 {
                failures.push(vfail(
                    FailureCode::NPromptTokensBound,
                    format!(
                        "n_prompt_tokens {} exceeds n_tokens {} + 1",
                        committed_npt, response.commitment.n_tokens
                    ),
                ));
            }
        }
        (None, _) => {
            failures.push(vfail(
                FailureCode::MissingNPromptTokens,
                "commitment missing n_prompt_tokens",
            ));
        }
        (_, None) => {
            failures.push(vfail(
                FailureCode::MissingNPromptTokens,
                "response missing n_prompt_tokens",
            ));
        }
    }

    // 4. Challenged token retained leaf Merkle proof.
    // When shell_opening has final_residual, include it in the leaf hash
    // (it was bound into the commitment at commit time).
    checks_run += 1;
    let final_residual_ref = response
        .shell_opening
        .as_ref()
        .and_then(|s| s.final_residual.as_deref());
    let lp_hidden_ref = response
        .shell_opening
        .as_ref()
        .and_then(|s| s.lp_hidden_bf16.as_deref());
    let leaf_hash = merkle::hash_retained_with_lp_hidden(&response.retained, final_residual_ref, lp_hidden_ref);
    if !merkle::verify(
        &response.commitment.merkle_root,
        &leaf_hash,
        &response.merkle_proof,
    ) {
        failures.push(vfail_ctx(
            FailureCode::MerkleProofFailed,
            format!(
                "token {}: retained leaf Merkle proof failed",
                response.token_index
            ),
            FailureContext {
                token_index: Some(response.token_index),
                ..Default::default()
            },
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
            failures.push(vfail_ctx(
                FailureCode::MerkleProofFailed,
                format!("prefix token {}: retained leaf Merkle proof failed", j),
                FailureContext {
                    token_index: Some(j as u32),
                    ..Default::default()
                },
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
        failures.push(vfail_ctx(
            FailureCode::IoChainMismatch,
            format!(
                "token {}: prev_io_hash doesn't match recomputed chain from prefix",
                response.token_index
            ),
            FailureContext {
                token_index: Some(response.token_index),
                ..Default::default()
            },
        ));
    }

    checks_run += 1;
    let challenged_io = merkle::io_hash_v4(leaf_hash, response.token_id, prev_io);
    if !merkle::verify(
        &response.commitment.io_root,
        &challenged_io,
        &response.io_proof,
    ) {
        failures.push(vfail_ctx(
            FailureCode::IoChainProofFailed,
            format!(
                "token {}: IO chain proof verification failed",
                response.token_index
            ),
            FailureContext {
                token_index: Some(response.token_index),
                ..Default::default()
            },
        ));
    }

    // 7. Prefix count == token_index
    checks_run += 1;
    if response.prefix_leaf_hashes.len() != response.token_index as usize {
        failures.push(vfail_ctx(
            FailureCode::PrefixTokenCountMismatch,
            format!(
                "token {}: expected {} prefix tokens but got {}",
                response.token_index,
                response.token_index,
                response.prefix_leaf_hashes.len()
            ),
            FailureContext {
                token_index: Some(response.token_index),
                ..Default::default()
            },
        ));
    }

    (checks_run, failures)
}

/// Replay one token's computation shell from retained state + public weights.
///
/// **Canonical path**: when the key has RMSNorm weights and `initial_residual`
/// is provided, uses full bridge (`bridge_residual_rmsnorm`) and enables QKV
/// Freivalds at all layers including layer 0.
///
/// **Toy fallback**: when `initial_residual` is absent, falls back to
/// `bridge_requantize()` with no residual tracking. Layer 0 QKV is skipped.
/// Not valid for production W8A8 verification.
///
/// Returns (checks_run, failures).
fn replay_token_shell(
    key: &VerifierKey,
    cfg: &verilm_core::constants::ModelConfig,
    retained: &verilm_core::types::RetainedTokenState,
    weights: &dyn ShellWeights,
    token_pos: usize,
    initial_residual: Option<&[f32]>,
) -> (usize, Vec<VerificationFailure>) {
    use verilm_core::matmul::matmul_i32;

    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // Full bridge: only when initial_residual is provided AND authenticated.
    let use_full_bridge = initial_residual.is_some()
        && !key.rmsnorm_attn_weights.is_empty()
        && key.embedding_merkle_root.is_some();

    // NOTE: replay_token_shell uses weight-backed replay with unit scales
    // (toy model). Scales don't matter for this path since bridge_requantize
    // uses them symmetrically. We use 1.0 defaults.
    let (mut residual, mut x_attn) = if use_full_bridge {
        let ir = initial_residual.unwrap();
        let res: Vec<f64> = ir.iter().map(|&v| v as f64).collect();
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
            &res,
            &key.rmsnorm_attn_weights[0],
            key.rmsnorm_eps,
        );
        // Weight-backed replay doesn't have shell layer scales. Use 1.0.
        let xa = verilm_core::rmsnorm::quantize_f64_to_i8(&normed, 1.0_f64);
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
                if !freivalds::check(key.v_for(layer_idx, *mt), xa, key.r_for(*mt), &z) {
                    failures.push(vfail_ctx(
                        FailureCode::FreivaldsFailed,
                        format!(
                            "token {} layer {} {:?}: Freivalds weight-binding failed",
                            token_pos, layer_idx, mt
                        ),
                        FailureContext {
                            token_index: Some(token_pos as u32),
                            layer: Some(layer_idx),
                            matrix: Some(format!("{:?}", mt)),
                            ..Default::default()
                        },
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
            failures.push(vfail_ctx(
                FailureCode::FreivaldsFailed,
                format!(
                    "token {} layer {} Wo: Freivalds weight-binding failed",
                    token_pos, layer_idx
                ),
                FailureContext {
                    token_index: Some(token_pos as u32),
                    layer: Some(layer_idx),
                    matrix: Some("Wo".into()),
                    ..Default::default()
                },
            ));
        }

        // Post-attention bridge: derive x_ffn (unit scales for weight-backed replay)
        let x_ffn = if let Some(ref mut res) = residual {
            verilm_core::rmsnorm::bridge_residual_rmsnorm(
                &attn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wo),
                rs.scale_a,
                res,
                &key.rmsnorm_ffn_weights[layer_idx],
                key.rmsnorm_eps,
                1.0,
            )
        } else {
            verilm_core::bridge_requantize(
                &attn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wo),
                rs.scale_a,
                1.0,
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
            if !freivalds::check(key.v_for(layer_idx, mt), &x_ffn, key.r_for(mt), z) {
                failures.push(vfail_ctx(
                    FailureCode::FreivaldsFailed,
                    format!(
                        "token {} layer {} {:?}: Freivalds weight-binding failed",
                        token_pos, layer_idx, mt
                    ),
                    FailureContext {
                        token_index: Some(token_pos as u32),
                        layer: Some(layer_idx),
                        matrix: Some(format!("{:?}", mt)),
                        ..Default::default()
                    },
                ));
            }
        }

        // SiLU: h via scale-aware bridge (unit scales for weight-backed replay)
        let h = verilm_core::silu::compute_h_scaled(
            &g,
            &u,
            key.weight_scale_for(layer_idx, MatrixType::Wg),
            key.weight_scale_for(layer_idx, MatrixType::Wu),
            1.0,
            1.0,
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
            failures.push(vfail_ctx(
                FailureCode::FreivaldsFailed,
                format!(
                    "token {} layer {} Wd: Freivalds weight-binding failed",
                    token_pos, layer_idx
                ),
                FailureContext {
                    token_index: Some(token_pos as u32),
                    layer: Some(layer_idx),
                    matrix: Some("Wd".into()),
                    ..Default::default()
                },
            ));
        }

        // Post-FFN bridge: derive x_attn for next layer (unit scales for weight-backed replay)
        x_attn = if let Some(ref mut res) = residual {
            if layer_idx + 1 < key.rmsnorm_attn_weights.len() {
                Some(verilm_core::rmsnorm::bridge_residual_rmsnorm(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    1.0,
                    res,
                    &key.rmsnorm_attn_weights[layer_idx + 1],
                    key.rmsnorm_eps,
                    1.0,
                ))
            } else {
                verilm_core::rmsnorm::dequant_add_residual(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    1.0,
                    res,
                );
                Some(verilm_core::bridge_requantize(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    1.0,
                    1.0,
                ))
            }
        } else {
            Some(verilm_core::bridge_requantize(
                &ffn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wd),
                1.0,
                1.0,
            ))
        };
    }

    (checks_run, failures)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vfail_structural_codes() {
        let f = vfail(
            FailureCode::WrongCommitmentVersion,
            "expected V4 commitment version",
        );
        assert_eq!(f.category, FailureCategory::Structural);
        assert_eq!(f.code, FailureCode::WrongCommitmentVersion);

        let f = vfail(
            FailureCode::MissingShellOpening,
            "V4 audit response missing shell_opening",
        );
        assert_eq!(f.category, FailureCategory::Structural);

        let f = vfail(
            FailureCode::MissingLogits,
            "lm_head: key requires logits_i32 but shell opening did not provide it",
        );
        assert_eq!(f.category, FailureCategory::Structural);
    }

    #[test]
    fn vfail_cryptographic_binding_codes() {
        assert_eq!(
            FailureCode::FreivaldsFailed.category(),
            FailureCategory::CryptographicBinding
        );
        assert_eq!(
            FailureCode::MerkleProofFailed.category(),
            FailureCategory::CryptographicBinding
        );
        assert_eq!(
            FailureCode::IoChainMismatch.category(),
            FailureCategory::CryptographicBinding
        );
        assert_eq!(
            FailureCode::SeedMismatch.category(),
            FailureCategory::CryptographicBinding
        );
        assert_eq!(
            FailureCode::PromptHashMismatch.category(),
            FailureCategory::CryptographicBinding
        );
        assert_eq!(
            FailureCode::ManifestHashMismatch.category(),
            FailureCategory::CryptographicBinding
        );
        assert_eq!(
            FailureCode::SpecHashMismatch.category(),
            FailureCategory::CryptographicBinding
        );
        assert_eq!(
            FailureCode::RetainedHashMismatch.category(),
            FailureCategory::CryptographicBinding
        );
    }

    #[test]
    fn vfail_spec_mismatch_codes() {
        assert_eq!(
            FailureCode::SpecFieldMismatch.category(),
            FailureCategory::SpecMismatch
        );
    }

    #[test]
    fn vfail_unsupported_codes() {
        assert_eq!(
            FailureCode::UnsupportedSamplerVersion.category(),
            FailureCategory::Unsupported
        );
        assert_eq!(
            FailureCode::UnsupportedDecodeFeature.category(),
            FailureCategory::Unsupported
        );
        assert_eq!(
            FailureCode::UnknownEosPolicy.category(),
            FailureCategory::Unsupported
        );
    }

    #[test]
    fn vfail_semantic_violation_codes() {
        assert_eq!(
            FailureCode::TokenSelectionMismatch.category(),
            FailureCategory::SemanticViolation
        );
        assert_eq!(
            FailureCode::ExceedsMaxTokens.category(),
            FailureCategory::SemanticViolation
        );
        assert_eq!(
            FailureCode::MinTokensViolated.category(),
            FailureCategory::SemanticViolation
        );
        assert_eq!(
            FailureCode::EosPolicyViolated.category(),
            FailureCategory::SemanticViolation
        );
        assert_eq!(
            FailureCode::DecodeModeTempInconsistent.category(),
            FailureCategory::SemanticViolation
        );
    }

    #[test]
    fn vfail_operational_codes() {
        assert_eq!(
            FailureCode::TokenizerError.category(),
            FailureCategory::Operational
        );
        assert_eq!(
            FailureCode::DetokenizerError.category(),
            FailureCategory::Operational
        );
    }

    #[test]
    fn vfail_ctx_carries_context() {
        let f = vfail_ctx(
            FailureCode::FreivaldsFailed,
            "layer 0 Wo: Freivalds failed",
            FailureContext {
                layer: Some(0),
                matrix: Some("Wo".into()),
                ..Default::default()
            },
        );
        assert_eq!(f.code, FailureCode::FreivaldsFailed);
        assert_eq!(f.category, FailureCategory::CryptographicBinding);
        assert_eq!(f.context.layer, Some(0));
        assert_eq!(f.context.matrix.as_deref(), Some("Wo"));
        assert_eq!(f.message, "layer 0 Wo: Freivalds failed");
    }
}
