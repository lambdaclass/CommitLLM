//! Client-side verification wrapper.
//!
//! Performs protocol-level external checks (challenge binding, etc.) that live
//! outside the canonical verifier's trust boundary, then delegates proof
//! verification to [`canonical::verify_binary`].
//!
//! The canonical verifier answers: "is this audit proof mathematically correct?"
//! This wrapper answers: "is this audit proof a valid response to *my* challenge?"

use verilm_core::serialize;
use verilm_core::types::{AuditChallenge, V4AuditResponse, VerifierKey};

use crate::{
    canonical, Detokenizer, FailureCode, FailureContext, PromptTokenizer, V4VerifyReport, Verdict,
    VerificationFailure,
};

/// Verify a V4 audit binary with protocol-level challenge binding.
///
/// Runs external checks first (challenge match), then delegates to the
/// canonical verifier. Returns a single unified report.
///
/// Returns `Err` only for format-level failures (bad magic, decompression).
/// All protocol-level failures produce `Ok(report)`.
pub fn verify_challenged_binary(
    challenge: &AuditChallenge,
    key: &VerifierKey,
    audit_bytes: &[u8],
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> Result<V4VerifyReport, String> {
    let response = serialize::deserialize_v4_audit(audit_bytes)?;
    Ok(verify_challenged_response(
        challenge,
        key,
        &response,
        tokenizer,
        detokenizer,
    ))
}

/// Verify a deserialized V4 audit response with protocol-level challenge binding.
pub fn verify_challenged_response(
    challenge: &AuditChallenge,
    key: &VerifierKey,
    response: &V4AuditResponse,
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> V4VerifyReport {
    let mut report = canonical::verify_response(key, response, None, tokenizer, detokenizer);

    // External check: response.token_index matches the challenged token.
    report.checks_run += 1;
    if response.token_index != challenge.token_index {
        report.failures.push(VerificationFailure {
            category: FailureCode::ChallengeTokenMismatch.category(),
            code: FailureCode::ChallengeTokenMismatch,
            message: format!(
                "response token_index {} != challenged token_index {}",
                response.token_index, challenge.token_index,
            ),
            context: FailureContext::default(),
        });
    } else {
        report.checks_passed += 1;
    }

    // External check: opened layer_indices match the challenged layers.
    // When shell.layer_indices is None, the prover opened all layers (0..n).
    report.checks_run += 1;
    let opened: Vec<usize> = match response.shell_opening.as_ref() {
        Some(s) => match s.layer_indices.as_ref() {
            Some(v) => v.clone(),
            None => (0..s.layers.len()).collect(),
        },
        None => vec![],
    };
    if opened != challenge.layer_indices {
        report.failures.push(VerificationFailure {
            category: FailureCode::ChallengeLayerMismatch.category(),
            code: FailureCode::ChallengeLayerMismatch,
            message: format!(
                "opened layer_indices {:?} != challenged layer_indices {:?}",
                opened, challenge.layer_indices,
            ),
            context: FailureContext::default(),
        });
    } else {
        report.checks_passed += 1;
    }

    // Recompute verdict: any failure (canonical or external) means Fail.
    if !report.failures.is_empty() {
        report.verdict = Verdict::Fail;
    }

    report
}
