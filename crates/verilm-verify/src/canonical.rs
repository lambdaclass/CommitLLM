//! Minimal canonical verifier: binary-only, key-only, full-bridge-only.
//!
//! # Entry point
//!
//! [`verify_binary`] accepts a verifier key and raw audit bytes (VV4A magic +
//! zstd + bincode). Returns `Err` for deserialization failures, `Ok(report)`
//! for all verification outcomes.
//!
//! # Constraints
//!
//! - **Binary-only**: canonical VV4A wire format
//! - **Key-only**: precomputed Freivalds vectors, no weight replay
//! - **Full bridge only**: `bridge_residual_rmsnorm`, no toy requantize fallback
//! - **Final residual + final norm**: exact LM-head path, no shell replay
//! - **Fail-closed**: missing or unsupported features are rejected

use std::time::Instant;

use verilm_core::constants::MatrixType;
use verilm_core::types::{
    CommitmentVersion, InputSpec, OutputSpec, RetainedTokenState, ShellTokenOpening,
    V4AuditResponse, VerifierKey,
};
use verilm_core::{freivalds, merkle, serialize};

use crate::{
    vfail, vfail_ctx, AuditCoverage, Detokenizer, FailureCode, FailureContext,
    PromptTokenizer, V4VerifyReport, Verdict, VerificationFailure,
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Verify a V4 audit from canonical binary wire format.
///
/// Returns `Err(description)` only for format-level failures (bad magic,
/// decompression error, malformed bincode). All protocol-level failures
/// produce `Ok(report)` with `verdict == Fail`.
pub fn verify_binary(
    key: &VerifierKey,
    audit_bytes: &[u8],
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> Result<V4VerifyReport, String> {
    let response = serialize::deserialize_v4_audit(audit_bytes)?;
    Ok(verify_response(key, &response, tokenizer, detokenizer))
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

fn verify_response(
    key: &VerifierKey,
    r: &V4AuditResponse,
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> V4VerifyReport {
    let start = Instant::now();
    let mut c = 0usize;
    let mut f: Vec<VerificationFailure> = Vec::new();

    // Phase 1: Structural
    structural(r, &mut c, &mut f);

    // Phase 2: Shell opening (required)
    let shell = match &r.shell_opening {
        Some(s) => s,
        None => {
            f.push(vfail(
                FailureCode::MissingShellOpening,
                "canonical: shell_opening required",
            ));
            return build_report(r, key, c, f, start);
        }
    };

    // Phase 3: Embedding binding
    embedding_binding(key, r, shell, &mut c, &mut f);
    rich_prefix_embeddings(key, r, &mut c, &mut f);

    // Phase 4: Manifest + spec binding → decode params
    let (decode_params, output_spec) = spec_binding(key, r, &mut c, &mut f);

    // Phase 5: Output policy
    if let Some(ref os) = output_spec {
        output_policy(r, os, &mut c, &mut f);
    }

    // Phase 6: Full bridge (Freivalds + bridge chain)
    full_bridge(key, &r.retained, shell, &mut c, &mut f);

    // Phase 7: LM-head + token replay
    lm_head(key, r, shell, decode_params, &mut c, &mut f);

    // Phase 8: Deep prefix
    deep_prefix(key, r, &mut c, &mut f);

    // Phase 9: Tokenization
    tokenization_verify(r, tokenizer, &mut c, &mut f);

    // Phase 10: Detokenization
    detokenization_verify(r, detokenizer, &mut c, &mut f);

    build_report(r, key, c, f, start)
}

// ---------------------------------------------------------------------------
// Phase 1: Structural checks
// ---------------------------------------------------------------------------

fn structural(
    r: &V4AuditResponse,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    // 1. V4 commitment version
    *c += 1;
    if r.commitment.version != CommitmentVersion::V4 {
        f.push(vfail(
            FailureCode::WrongCommitmentVersion,
            format!("expected V4, got {:?}", r.commitment.version),
        ));
    }

    // 2. Seed commitment
    *c += 1;
    match r.commitment.seed_commitment {
        Some(expected) => {
            if merkle::hash_seed(&r.revealed_seed) != expected {
                f.push(vfail(
                    FailureCode::SeedMismatch,
                    "hash(revealed_seed) != commitment.seed_commitment",
                ));
            }
        }
        None => f.push(vfail(
            FailureCode::MissingSeedCommitment,
            "V4 commitment missing seed_commitment",
        )),
    }

    // 3. Prompt hash binding (fail-closed)
    *c += 1;
    match (&r.prompt, r.commitment.prompt_hash) {
        (Some(prompt), Some(committed)) => {
            if merkle::hash_prompt(prompt) != committed {
                f.push(vfail(FailureCode::PromptHashMismatch, "hash(prompt) != prompt_hash"));
            }
        }
        (None, Some(_)) => f.push(vfail(FailureCode::MissingPromptBytes, "commitment has prompt_hash but response missing prompt")),
        (Some(_), None) => f.push(vfail(FailureCode::UncommittedPrompt, "response has prompt but commitment missing prompt_hash")),
        (None, None) => f.push(vfail(FailureCode::MissingPromptHash, "V4 commitment missing prompt_hash")),
    }

    // 4. n_prompt_tokens binding
    *c += 1;
    match (r.commitment.n_prompt_tokens, r.n_prompt_tokens) {
        (Some(committed), Some(response)) => {
            if committed != response {
                f.push(vfail(
                    FailureCode::NPromptTokensMismatch,
                    format!("n_prompt_tokens: commitment={} response={}", committed, response),
                ));
            }
            if committed > r.commitment.n_tokens + 1 {
                f.push(vfail(
                    FailureCode::NPromptTokensBound,
                    format!("n_prompt_tokens {} exceeds n_tokens {} + 1", committed, r.commitment.n_tokens),
                ));
            }
        }
        (None, _) => f.push(vfail(FailureCode::MissingNPromptTokens, "commitment missing n_prompt_tokens")),
        (_, None) => f.push(vfail(FailureCode::MissingNPromptTokens, "response missing n_prompt_tokens")),
    }

    // 5. Retained leaf Merkle proof (includes final_residual when present)
    *c += 1;
    let fr_ref = r.shell_opening.as_ref().and_then(|s| s.final_residual.as_deref());
    let leaf_hash = merkle::hash_retained_with_residual(&r.retained, fr_ref);
    if !merkle::verify(&r.commitment.merkle_root, &leaf_hash, &r.merkle_proof) {
        f.push(vfail_ctx(
            FailureCode::MerkleProofFailed,
            format!("token {}: retained leaf Merkle proof failed", r.token_index),
            FailureContext { token_index: Some(r.token_index), ..Default::default() },
        ));
    }

    // 6. Prefix retained leaf Merkle proofs
    for (j, (hash, proof)) in r.prefix_leaf_hashes.iter().zip(r.prefix_merkle_proofs.iter()).enumerate() {
        *c += 1;
        if !merkle::verify(&r.commitment.merkle_root, hash, proof) {
            f.push(vfail_ctx(
                FailureCode::MerkleProofFailed,
                format!("prefix token {}: Merkle proof failed", j),
                FailureContext { token_index: Some(j as u32), ..Default::default() },
            ));
        }
    }

    // 7. IO chain
    *c += 1;
    let io_genesis = match r.commitment.prompt_hash {
        Some(ph) => merkle::io_genesis_v4(ph),
        None => [0u8; 32],
    };
    let mut prev_io = io_genesis;
    for (j, hash) in r.prefix_leaf_hashes.iter().enumerate() {
        prev_io = merkle::io_hash_v4(*hash, r.prefix_token_ids[j], prev_io);
    }
    if prev_io != r.prev_io_hash {
        f.push(vfail_ctx(
            FailureCode::IoChainMismatch,
            format!("token {}: prev_io_hash mismatch", r.token_index),
            FailureContext { token_index: Some(r.token_index), ..Default::default() },
        ));
    }

    // 8. IO chain proof
    *c += 1;
    let challenged_io = merkle::io_hash_v4(leaf_hash, r.token_id, prev_io);
    if !merkle::verify(&r.commitment.io_root, &challenged_io, &r.io_proof) {
        f.push(vfail_ctx(
            FailureCode::IoChainProofFailed,
            format!("token {}: IO chain proof failed", r.token_index),
            FailureContext { token_index: Some(r.token_index), ..Default::default() },
        ));
    }

    // 9. Prefix count == token_index
    *c += 1;
    if r.prefix_leaf_hashes.len() != r.token_index as usize {
        f.push(vfail_ctx(
            FailureCode::PrefixTokenCountMismatch,
            format!(
                "token {}: expected {} prefix tokens, got {}",
                r.token_index, r.token_index, r.prefix_leaf_hashes.len()
            ),
            FailureContext { token_index: Some(r.token_index), ..Default::default() },
        ));
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Embedding binding
// ---------------------------------------------------------------------------

fn embedding_binding(
    key: &VerifierKey,
    r: &V4AuditResponse,
    shell: &ShellTokenOpening,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    let emb_root = match &key.embedding_merkle_root {
        Some(root) => root,
        None => {
            // Canonical: embedding root required for authenticated initial_residual.
            if shell.initial_residual.is_some() {
                *c += 1;
                f.push(vfail(
                    FailureCode::UnboundInitialResidual,
                    "shell has initial_residual but key has no embedding_merkle_root",
                ));
            }
            return;
        }
    };

    *c += 1;
    match (&shell.initial_residual, &shell.embedding_proof) {
        (Some(ir), Some(proof)) => {
            let leaf = merkle::hash_embedding_row(ir);
            if proof.leaf_index != r.token_id {
                f.push(vfail(
                    FailureCode::EmbeddingLeafMismatch,
                    format!("embedding proof leaf_index {} != token_id {}", proof.leaf_index, r.token_id),
                ));
            } else if !merkle::verify(emb_root, &leaf, proof) {
                f.push(vfail(FailureCode::EmbeddingProofFailed, "embedding Merkle proof failed"));
            }
        }
        (None, _) => f.push(vfail(
            FailureCode::MissingInitialResidual,
            "canonical: initial_residual required (key has embedding_merkle_root)",
        )),
        (Some(_), None) => f.push(vfail(
            FailureCode::MissingEmbeddingProof,
            "canonical: embedding_proof required (key has embedding_merkle_root)",
        )),
    }
}

fn rich_prefix_embeddings(
    key: &VerifierKey,
    r: &V4AuditResponse,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    let (emb_root, rows, proofs) = match (
        &key.embedding_merkle_root,
        &r.prefix_embedding_rows,
        &r.prefix_embedding_proofs,
    ) {
        (Some(root), Some(rows), Some(proofs)) => (root, rows, proofs),
        _ => return,
    };

    if rows.len() != r.prefix_token_ids.len() || proofs.len() != r.prefix_token_ids.len() {
        *c += 1;
        f.push(vfail(
            FailureCode::PrefixCountMismatch,
            format!(
                "prefix embedding count: {} rows, {} proofs, {} token_ids",
                rows.len(), proofs.len(), r.prefix_token_ids.len()
            ),
        ));
        return;
    }

    for (j, ((row, proof), &tid)) in rows.iter().zip(proofs.iter()).zip(r.prefix_token_ids.iter()).enumerate() {
        *c += 1;
        let leaf = merkle::hash_embedding_row(row);
        if proof.leaf_index != tid {
            f.push(vfail_ctx(
                FailureCode::EmbeddingLeafMismatch,
                format!("prefix {}: embedding leaf_index {} != token_id {}", j, proof.leaf_index, tid),
                FailureContext { token_index: Some(j as u32), ..Default::default() },
            ));
        } else if !merkle::verify(emb_root, &leaf, proof) {
            f.push(vfail_ctx(
                FailureCode::EmbeddingProofFailed,
                format!("prefix {}: embedding Merkle proof failed", j),
                FailureContext { token_index: Some(j as u32), ..Default::default() },
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 4: Manifest + spec binding
// ---------------------------------------------------------------------------

fn spec_binding(
    key: &VerifierKey,
    r: &V4AuditResponse,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) -> (Option<verilm_core::sampling::DecodeParams>, Option<OutputSpec>) {
    let manifest = match &r.manifest {
        Some(m) => m,
        None => {
            f.push(vfail(
                FailureCode::MissingManifestHash,
                "canonical: manifest required for spec verification",
            ));
            return (None, None);
        }
    };

    let (input_spec, model_spec, decode_spec, output_spec) = manifest.split();
    let h_in = merkle::hash_input_spec(&input_spec);
    let h_mod = merkle::hash_model_spec(&model_spec);
    let h_dec = merkle::hash_decode_spec(&decode_spec);
    let h_out = merkle::hash_output_spec(&output_spec);

    // Four spec hashes (fail-closed)
    *c += 4;
    check_spec_hash(r.commitment.input_spec_hash, h_in, "input", f);
    check_spec_hash(r.commitment.model_spec_hash, h_mod, "model", f);
    check_spec_hash(r.commitment.decode_spec_hash, h_dec, "decode", f);
    check_spec_hash(r.commitment.output_spec_hash, h_out, "output", f);

    // Composed manifest hash
    *c += 1;
    match r.commitment.manifest_hash {
        None => f.push(vfail(FailureCode::MissingManifestHash, "commitment missing manifest_hash")),
        Some(committed) => {
            if merkle::hash_manifest_composed(h_in, h_mod, h_dec, h_out) != committed {
                f.push(vfail(FailureCode::ManifestHashMismatch, "manifest hash mismatch"));
            }
        }
    }

    // Model spec cross-checks against verifier key
    model_cross_checks(key, &model_spec, c, f);

    // Sampler version
    *c += 1;
    match decode_spec.sampler_version.as_deref() {
        Some("chacha20-vi-sample-v1") | None => {}
        Some(other) => f.push(vfail(
            FailureCode::UnsupportedSamplerVersion,
            format!("unsupported sampler_version='{}'", other),
        )),
    }

    // Decode mode consistency
    if let Some(ref dm) = decode_spec.decode_mode {
        *c += 1;
        match dm.as_str() {
            "greedy" if decode_spec.temperature != 0.0 => f.push(vfail(
                FailureCode::DecodeModeTempInconsistent,
                format!("decode_mode='greedy' but temperature={}", decode_spec.temperature),
            )),
            "sampled" if decode_spec.temperature == 0.0 => f.push(vfail(
                FailureCode::DecodeModeTempInconsistent,
                "decode_mode='sampled' but temperature=0.0",
            )),
            "greedy" | "sampled" => {}
            other => f.push(vfail(
                FailureCode::UnsupportedDecodeMode,
                format!("unsupported decode_mode='{}'", other),
            )),
        }
    }

    // Unsupported decode features (fail-closed)
    *c += 1;
    reject_unsupported_feature(decode_spec.repetition_penalty != 1.0, "repetition_penalty", &format!("{}", decode_spec.repetition_penalty), "1.0", f);
    reject_unsupported_feature(decode_spec.frequency_penalty != 0.0, "frequency_penalty", &format!("{}", decode_spec.frequency_penalty), "0.0", f);
    reject_unsupported_feature(decode_spec.presence_penalty != 0.0, "presence_penalty", &format!("{}", decode_spec.presence_penalty), "0.0", f);
    reject_unsupported_feature(!decode_spec.logit_bias.is_empty(), "logit_bias", &format!("{} entries", decode_spec.logit_bias.len()), "empty", f);
    reject_unsupported_feature(!decode_spec.bad_word_ids.is_empty(), "bad_word_ids", &format!("{} entries", decode_spec.bad_word_ids.len()), "empty", f);
    reject_unsupported_feature(!decode_spec.guided_decoding.is_empty(), "guided_decoding", &decode_spec.guided_decoding, "empty", f);
    reject_unsupported_feature(!output_spec.stop_sequences.is_empty(), "stop_sequences", &format!("{} entries", output_spec.stop_sequences.len()), "empty", f);

    let decode_params = verilm_core::sampling::DecodeParams {
        temperature: decode_spec.temperature,
        top_k: decode_spec.top_k,
        top_p: decode_spec.top_p,
    };

    (Some(decode_params), Some(output_spec))
}

fn check_spec_hash(
    committed: Option<[u8; 32]>,
    computed: [u8; 32],
    name: &str,
    f: &mut Vec<VerificationFailure>,
) {
    match committed {
        None => f.push(vfail_ctx(
            FailureCode::MissingSpecHash,
            format!("commitment missing {}_spec_hash", name),
            FailureContext { spec: Some(name.into()), ..Default::default() },
        )),
        Some(h) if h != computed => f.push(vfail_ctx(
            FailureCode::SpecHashMismatch,
            format!("{}_spec_hash mismatch", name),
            FailureContext { spec: Some(name.into()), ..Default::default() },
        )),
        _ => {}
    }
}

fn model_cross_checks(
    key: &VerifierKey,
    spec: &verilm_core::types::ModelSpec,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    // Each cross-check is opt-in: only fires when the manifest declares the field.
    macro_rules! cross_check_usize {
        ($field:ident, $key_val:expr) => {
            if let Some(v) = spec.$field {
                *c += 1;
                if v as usize != $key_val {
                    f.push(vfail_ctx(
                        FailureCode::SpecFieldMismatch,
                        format!(concat!(stringify!($field), " mismatch: manifest={} key={}"), v, $key_val),
                        FailureContext { field: Some(stringify!($field).into()), ..Default::default() },
                    ));
                }
            }
        };
    }

    macro_rules! cross_check_hash {
        ($field:ident, $key_val:expr) => {
            if let (Some(mval), Some(kval)) = (spec.$field, $key_val) {
                *c += 1;
                if mval != kval {
                    f.push(vfail_ctx(
                        FailureCode::SpecFieldMismatch,
                        format!(concat!(stringify!($field), " mismatch: manifest != key")),
                        FailureContext { field: Some(stringify!($field).into()), ..Default::default() },
                    ));
                }
            }
        };
    }

    macro_rules! cross_check_string {
        ($field:ident, $key_val:expr) => {
            if let (Some(ref mval), Some(ref kval)) = (&spec.$field, &$key_val) {
                *c += 1;
                if mval != kval {
                    f.push(vfail_ctx(
                        FailureCode::SpecFieldMismatch,
                        format!(concat!(stringify!($field), " mismatch: manifest='{}' key='{}'"), mval, kval),
                        FailureContext { field: Some(stringify!($field).into()), ..Default::default() },
                    ));
                }
            }
        };
    }

    // f64 fields
    if let Some(eps) = spec.rmsnorm_eps {
        *c += 1;
        if (eps - key.rmsnorm_eps).abs() > f64::EPSILON {
            f.push(vfail_ctx(
                FailureCode::SpecFieldMismatch,
                format!("rmsnorm_eps mismatch: manifest={} key={}", eps, key.rmsnorm_eps),
                FailureContext { field: Some("rmsnorm_eps".into()), ..Default::default() },
            ));
        }
    }
    if let Some(theta) = spec.rope_theta {
        *c += 1;
        if (theta - key.config.rope_theta).abs() > f64::EPSILON {
            f.push(vfail_ctx(
                FailureCode::SpecFieldMismatch,
                format!("rope_theta mismatch: manifest={} key={}", theta, key.config.rope_theta),
                FailureContext { field: Some("rope_theta".into()), ..Default::default() },
            ));
        }
    }

    // Hash fields
    cross_check_hash!(rope_config_hash, key.rope_config_hash);
    cross_check_hash!(weight_hash, key.weight_hash);
    cross_check_hash!(embedding_merkle_root, key.embedding_merkle_root);

    // usize fields
    cross_check_usize!(n_layers, key.config.n_layers);
    cross_check_usize!(hidden_dim, key.config.hidden_dim);
    cross_check_usize!(vocab_size, key.config.vocab_size);
    cross_check_usize!(kv_dim, key.config.kv_dim);
    cross_check_usize!(ffn_dim, key.config.ffn_dim);
    cross_check_usize!(d_head, key.config.d_head);
    cross_check_usize!(n_q_heads, key.config.n_q_heads);
    cross_check_usize!(n_kv_heads, key.config.n_kv_heads);

    // String fields
    cross_check_string!(quant_family, key.quant_family);
    cross_check_string!(scale_derivation, key.scale_derivation);

    // u32 field
    if let (Some(m_bs), Some(k_bs)) = (spec.quant_block_size, key.quant_block_size) {
        *c += 1;
        if m_bs != k_bs {
            f.push(vfail_ctx(
                FailureCode::SpecFieldMismatch,
                format!("quant_block_size mismatch: manifest={} key={}", m_bs, k_bs),
                FailureContext { field: Some("quant_block_size".into()), ..Default::default() },
            ));
        }
    }
}

fn reject_unsupported_feature(
    rejected: bool,
    field: &str,
    actual: &str,
    expected: &str,
    f: &mut Vec<VerificationFailure>,
) {
    if rejected {
        f.push(vfail_ctx(
            FailureCode::UnsupportedDecodeFeature,
            format!("unsupported {}={} (canonical requires {})", field, actual, expected),
            FailureContext { field: Some(field.into()), ..Default::default() },
        ));
    }
}

// ---------------------------------------------------------------------------
// Phase 5: Output policy
// ---------------------------------------------------------------------------

fn output_policy(
    r: &V4AuditResponse,
    os: &OutputSpec,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    // max_tokens
    if os.max_tokens > 0 {
        let n_prompt = r.commitment.n_prompt_tokens.unwrap_or(0);
        let n_generated = r.commitment.n_tokens.saturating_sub(n_prompt.saturating_sub(1));

        if r.token_index >= r.commitment.n_tokens {
            f.push(vfail(
                FailureCode::ExceedsMaxTokens,
                format!("token_index {} >= n_tokens {}", r.token_index, r.commitment.n_tokens),
            ));
        }
        if n_generated > os.max_tokens {
            f.push(vfail(
                FailureCode::ExceedsMaxTokens,
                format!("generated {} tokens > max_tokens {}", n_generated, os.max_tokens),
            ));
        }
    }

    // min_tokens + ignore_eos require eos_token_id
    if let Some(eos_id) = os.eos_token_id {
        if os.min_tokens > 0 {
            *c += 1;
            let gen_start = r.n_prompt_tokens.unwrap_or(1).saturating_sub(1);
            let n_generated = r.commitment.n_tokens.saturating_sub(gen_start);
            if n_generated < os.min_tokens {
                f.push(vfail(
                    FailureCode::MinTokensViolated,
                    format!("only {} generated tokens, min_tokens={}", n_generated, os.min_tokens),
                ));
            }
            let gen_index = r.token_index.saturating_sub(gen_start);
            if gen_index < os.min_tokens && r.token_id == eos_id {
                f.push(vfail(
                    FailureCode::MinTokensViolated,
                    format!("EOS at generation position {} < min_tokens={}", gen_index, os.min_tokens),
                ));
            }
        }

        if os.ignore_eos {
            *c += 1;
            if r.token_id == eos_id {
                f.push(vfail(
                    FailureCode::IgnoreEosViolated,
                    format!("token_id {} is EOS but ignore_eos=true", r.token_id),
                ));
            }
        }

        // eos_policy
        if os.eos_policy == "stop" {
            *c += 1;
            let is_last = r.token_index == r.commitment.n_tokens.saturating_sub(1);
            if r.token_id == eos_id && !is_last {
                f.push(vfail(
                    FailureCode::EosPolicyViolated,
                    format!(
                        "eos_policy='stop': EOS at index {} is not last (n_tokens={})",
                        r.token_index, r.commitment.n_tokens
                    ),
                ));
            }
        } else if os.eos_policy != "sample" {
            f.push(vfail(
                FailureCode::UnknownEosPolicy,
                format!("unknown eos_policy='{}'", os.eos_policy),
            ));
        }
    } else if os.min_tokens > 0 || os.ignore_eos {
        f.push(vfail(
            FailureCode::MissingEosTokenId,
            "output_spec needs eos_token_id for min_tokens/ignore_eos enforcement",
        ));
    }
}

// ---------------------------------------------------------------------------
// Phase 6: Full bridge (Freivalds + bridge chain)
// ---------------------------------------------------------------------------

fn full_bridge(
    key: &VerifierKey,
    retained: &RetainedTokenState,
    shell: &ShellTokenOpening,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    // Prerequisites for full bridge
    let ir = match &shell.initial_residual {
        Some(ir) => ir,
        None => return, // Already recorded in embedding_binding
    };
    if key.rmsnorm_attn_weights.is_empty() || key.rmsnorm_ffn_weights.is_empty() {
        return; // Can't run full bridge without RMSNorm weights
    }
    if retained.layers.is_empty() {
        return;
    }

    // Resolve opened layers
    let opened_layers: Vec<usize> = shell
        .layer_indices
        .clone()
        .unwrap_or_else(|| (0..retained.layers.len()).collect());

    if shell.layers.len() != opened_layers.len() {
        f.push(vfail(
            FailureCode::ShellLayerCountMismatch,
            format!("shell has {} layers, indices specify {}", shell.layers.len(), opened_layers.len()),
        ));
        return;
    }

    // Contiguous prefix required
    if let Some(ref indices) = shell.layer_indices {
        if !indices.iter().enumerate().all(|(i, &li)| li == i) {
            f.push(vfail(
                FailureCode::NonContiguousLayerIndices,
                format!("layer_indices must be contiguous prefix 0..N, got {:?}", indices),
            ));
            return;
        }
    }

    let max_layer = opened_layers.iter().copied().max().unwrap_or(0);
    let mut shell_idx_for = vec![None; max_layer + 1];
    for (si, &li) in opened_layers.iter().enumerate() {
        if li <= max_layer {
            shell_idx_for[li] = Some(si);
        }
    }

    // Initialize: RMSNorm(initial_residual) → x_attn_0
    let mut residual: Vec<f64> = ir.iter().map(|&v| v as f64).collect();
    let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
        &residual,
        &key.rmsnorm_attn_weights[0],
        key.rmsnorm_eps,
    );
    let mut x_attn = verilm_core::rmsnorm::quantize_f64_to_i8(
        &normed,
        retained.layers[0].scale_x_attn as f64,
    );

    for layer_idx in 0..=max_layer {
        if layer_idx >= retained.layers.len() {
            break;
        }
        let rs = &retained.layers[layer_idx];
        let si = match shell_idx_for[layer_idx] {
            Some(si) => si,
            None => continue,
        };
        let sl = &shell.layers[si];

        // QKV Freivalds — all required in canonical path
        for (mt, acc, _dim) in [
            (MatrixType::Wq, &sl.q, key.config.hidden_dim),
            (MatrixType::Wk, &sl.k, key.config.kv_dim),
            (MatrixType::Wv, &sl.v, key.config.kv_dim),
        ] {
            match acc {
                Some(z) => {
                    *c += 1;
                    if !freivalds::check(key.v_for(layer_idx, mt), &x_attn, key.r_for(mt), z) {
                        f.push(vfail_ctx(
                            FailureCode::FreivaldsFailed,
                            format!("layer {} {:?}: Freivalds failed", layer_idx, mt),
                            FailureContext {
                                layer: Some(layer_idx),
                                matrix: Some(format!("{:?}", mt)),
                                ..Default::default()
                            },
                        ));
                    }
                }
                None => f.push(vfail_ctx(
                    FailureCode::MissingQkv,
                    format!("layer {} {:?}: QKV required in canonical path", layer_idx, mt),
                    FailureContext {
                        layer: Some(layer_idx),
                        matrix: Some(format!("{:?}", mt)),
                        ..Default::default()
                    },
                )),
            }
        }

        // Wo @ a
        *c += 1;
        if !freivalds::check(
            key.v_for(layer_idx, MatrixType::Wo),
            &rs.a,
            key.r_for(MatrixType::Wo),
            &sl.attn_out,
        ) {
            f.push(vfail_ctx(
                FailureCode::FreivaldsFailed,
                format!("layer {} Wo: Freivalds failed", layer_idx),
                FailureContext {
                    layer: Some(layer_idx),
                    matrix: Some("Wo".into()),
                    ..Default::default()
                },
            ));
        }

        // Post-attention bridge: dequant → residual += attn_out → RMSNorm_ffn → quantize
        let x_ffn = verilm_core::rmsnorm::bridge_residual_rmsnorm(
            &sl.attn_out,
            key.weight_scale_for(layer_idx, MatrixType::Wo),
            rs.scale_a,
            &mut residual,
            &key.rmsnorm_ffn_weights[layer_idx],
            key.rmsnorm_eps,
            rs.scale_x_ffn,
        );

        // Wg, Wu @ x_ffn
        for (mt, z) in [(MatrixType::Wg, &sl.g), (MatrixType::Wu, &sl.u)] {
            *c += 1;
            if !freivalds::check(key.v_for(layer_idx, mt), &x_ffn, key.r_for(mt), z) {
                f.push(vfail_ctx(
                    FailureCode::FreivaldsFailed,
                    format!("layer {} {:?}: Freivalds failed", layer_idx, mt),
                    FailureContext {
                        layer: Some(layer_idx),
                        matrix: Some(format!("{:?}", mt)),
                        ..Default::default()
                    },
                ));
            }
        }

        // SiLU gate → h
        let h = verilm_core::silu::compute_h_scaled(
            &sl.g,
            &sl.u,
            key.weight_scale_for(layer_idx, MatrixType::Wg),
            key.weight_scale_for(layer_idx, MatrixType::Wu),
            rs.scale_x_ffn,
            rs.scale_h,
        );

        // Wd @ h
        *c += 1;
        if !freivalds::check(
            key.v_for(layer_idx, MatrixType::Wd),
            &h,
            key.r_for(MatrixType::Wd),
            &sl.ffn_out,
        ) {
            f.push(vfail_ctx(
                FailureCode::FreivaldsFailed,
                format!("layer {} Wd: Freivalds failed", layer_idx),
                FailureContext {
                    layer: Some(layer_idx),
                    matrix: Some("Wd".into()),
                    ..Default::default()
                },
            ));
        }

        // Post-FFN bridge → x_attn for next layer
        let next_scale = retained
            .layers
            .get(layer_idx + 1)
            .map(|r| r.scale_x_attn)
            .unwrap_or(1.0);

        x_attn = if layer_idx + 1 < key.rmsnorm_attn_weights.len() {
            verilm_core::rmsnorm::bridge_residual_rmsnorm(
                &sl.ffn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wd),
                rs.scale_h,
                &mut residual,
                &key.rmsnorm_attn_weights[layer_idx + 1],
                key.rmsnorm_eps,
                next_scale,
            )
        } else {
            // Last layer: update residual for final_hidden, no next-layer RMSNorm
            verilm_core::rmsnorm::dequant_add_residual(
                &sl.ffn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wd),
                rs.scale_h,
                &mut residual,
            );
            Vec::new() // not used after last layer
        };
    }
}

// ---------------------------------------------------------------------------
// Phase 7: LM-head + token replay
// ---------------------------------------------------------------------------

fn lm_head(
    key: &VerifierKey,
    r: &V4AuditResponse,
    shell: &ShellTokenOpening,
    decode_params: Option<verilm_core::sampling::DecodeParams>,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    // Canonical: derive final_hidden from captured final_residual + final_norm.
    // No shell-replay fallback.
    let final_hidden: Option<Vec<i8>> = match (&shell.final_residual, &key.final_norm_weights) {
        (Some(fr), Some(fnw)) => {
            let res_f64: Vec<f64> = fr.iter().map(|&v| v as f64).collect();
            let normed = verilm_core::rmsnorm::rmsnorm_f64_input(&res_f64, fnw, key.rmsnorm_eps);
            Some(
                normed
                    .iter()
                    .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
                    .collect(),
            )
        }
        (None, Some(_)) if key.lm_head.is_some() => {
            f.push(vfail(
                FailureCode::MissingFinalResidual,
                "canonical: final_residual required for LM-head (no shell replay fallback)",
            ));
            None
        }
        _ => None, // key lacks final_norm_weights — no LM-head verification possible
    };

    // LM-head Freivalds
    let r_lm = key.r_for(MatrixType::LmHead);
    let has_lm_freivalds = !r_lm.is_empty() && key.v_lm_head.is_some();

    if !has_lm_freivalds {
        return;
    }

    let v_lm = key.v_lm_head.as_ref().unwrap();

    match (&shell.logits_i32, &final_hidden) {
        (Some(logits), Some(fh)) => {
            *c += 1;
            if !freivalds::check(v_lm, fh, r_lm, logits) {
                f.push(vfail(
                    FailureCode::LmHeadFreivaldsFailed,
                    "lm_head: Freivalds check failed on logits_i32",
                ));
            }

            // Token replay (generated tokens only)
            let gen_start = r.n_prompt_tokens.map(|npt| npt.saturating_sub(1)).unwrap_or(0);
            if key.config.vocab_size > 0 && r.token_index >= gen_start {
                *c += 1;
                let logits_f32: Vec<f32> = logits.iter().map(|&v| v as f32).collect();
                let expected = if let Some(ref dp) = decode_params {
                    let seed =
                        verilm_core::sampling::derive_token_seed(&r.revealed_seed, r.token_index);
                    verilm_core::sampling::sample(&logits_f32, dp, &seed)
                } else {
                    logits_f32
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                };
                if expected != r.token_id {
                    f.push(vfail(
                        FailureCode::TokenSelectionMismatch,
                        format!("lm_head: expected token {} but got {}", expected, r.token_id),
                    ));
                }
            }
        }
        (None, _) => f.push(vfail(
            FailureCode::MissingLogits,
            "lm_head: key requires logits_i32 but shell missing it",
        )),
        (Some(_), None) => f.push(vfail(
            FailureCode::MissingFinalHidden,
            "lm_head: logits_i32 present but no final_hidden to verify against",
        )),
    }
}

// ---------------------------------------------------------------------------
// Phase 8: Deep prefix
// ---------------------------------------------------------------------------

fn deep_prefix(
    key: &VerifierKey,
    r: &V4AuditResponse,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    let (prefix_ret, prefix_shells) = match (&r.prefix_retained, &r.prefix_shell_openings) {
        (Some(ret), Some(shells)) => (ret, shells),
        _ => return,
    };

    if prefix_ret.len() != r.prefix_leaf_hashes.len()
        || prefix_shells.len() != r.prefix_leaf_hashes.len()
    {
        *c += 1;
        f.push(vfail(
            FailureCode::PrefixCountMismatch,
            format!(
                "deep prefix count: {} retained, {} shells, {} leaf_hashes",
                prefix_ret.len(),
                prefix_shells.len(),
                r.prefix_leaf_hashes.len()
            ),
        ));
        return;
    }

    for (j, ((ret_j, shell_j), &expected_hash)) in prefix_ret
        .iter()
        .zip(prefix_shells.iter())
        .zip(r.prefix_leaf_hashes.iter())
        .enumerate()
    {
        // Hash consistency
        *c += 1;
        let fr_ref = shell_j.final_residual.as_deref();
        let hash_j = merkle::hash_retained_with_residual(ret_j, fr_ref);
        if hash_j != expected_hash {
            f.push(vfail_ctx(
                FailureCode::RetainedHashMismatch,
                format!("prefix token {}: retained hash mismatch", j),
                FailureContext {
                    token_index: Some(j as u32),
                    ..Default::default()
                },
            ));
            continue;
        }

        // Shell Freivalds + bridge (reuses full_bridge)
        let before = f.len();
        full_bridge(key, ret_j, shell_j, c, f);
        // Tag bridge failures with the prefix token index.
        for failure in &mut f[before..] {
            failure.message = format!("prefix token {}: {}", j, failure.message);
            if failure.context.token_index.is_none() {
                failure.context.token_index = Some(j as u32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 9: Tokenization
// ---------------------------------------------------------------------------

fn tokenization_verify(
    r: &V4AuditResponse,
    tokenizer: Option<&dyn PromptTokenizer>,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    let tok = match tokenizer {
        Some(t) => t,
        None => return,
    };

    let (prompt, manifest) = match (&r.prompt, &r.manifest) {
        (Some(p), Some(m)) => (p, m),
        _ => return, // Can't reconstruct without prompt + manifest
    };

    let input_spec = InputSpec::from(manifest);
    match tok.tokenize(prompt, &input_spec) {
        Ok(tids) => {
            *c += 1;
            let tok_failures = crate::verify_input_tokenization(r, &tids);
            f.extend(tok_failures);
        }
        Err(e) => f.push(vfail(
            FailureCode::TokenizerError,
            format!("tokenizer reconstruction failed: {}", e),
        )),
    }
}

// ---------------------------------------------------------------------------
// Phase 10: Detokenization
// ---------------------------------------------------------------------------

fn detokenization_verify(
    r: &V4AuditResponse,
    detokenizer: Option<&dyn Detokenizer>,
    c: &mut usize,
    f: &mut Vec<VerificationFailure>,
) {
    let detok = match detokenizer {
        Some(d) => d,
        None => return,
    };

    *c += 1;
    let claimed = match &r.output_text {
        Some(t) => t,
        None => {
            f.push(vfail(
                FailureCode::MissingOutputText,
                "detokenizer provided but response missing output_text",
            ));
            return;
        }
    };

    let policy = r
        .manifest
        .as_ref()
        .map(|m| OutputSpec::from(m))
        .and_then(|os| os.detokenization_policy);

    // Collect generation token IDs
    let gen_start = r.n_prompt_tokens.unwrap_or(1).saturating_sub(1) as usize;
    let mut gen_tids: Vec<u32> = r
        .prefix_token_ids
        .get(gen_start..)
        .unwrap_or(&[])
        .to_vec();
    gen_tids.push(r.token_id);

    let is_last = r.token_index == r.commitment.n_tokens.saturating_sub(1);

    match detok.decode(&gen_tids, policy.as_deref()) {
        Ok(decoded) => {
            if is_last {
                if decoded != *claimed {
                    f.push(vfail(
                        FailureCode::DetokenizationMismatch,
                        format!("detokenization mismatch: decoded={:?} claimed={:?}", decoded, claimed),
                    ));
                }
            } else if !claimed.starts_with(&decoded) {
                f.push(vfail(
                    FailureCode::DetokenizationMismatch,
                    format!(
                        "detokenization prefix mismatch: decoded={:?} not prefix of claimed={:?}",
                        decoded, claimed
                    ),
                ));
            }
        }
        Err(e) => f.push(vfail(
            FailureCode::DetokenizerError,
            format!("detokenization failed: {}", e),
        )),
    }
}

// ---------------------------------------------------------------------------
// Report builder
// ---------------------------------------------------------------------------

fn build_report(
    r: &V4AuditResponse,
    key: &VerifierKey,
    checks_run: usize,
    failures: Vec<VerificationFailure>,
    start: Instant,
) -> V4VerifyReport {
    let coverage = match &r.shell_opening {
        Some(shell) => {
            let checked = shell
                .layer_indices
                .as_ref()
                .map_or(shell.layers.len(), |v| v.len());
            if checked >= key.config.n_layers {
                AuditCoverage::Full {
                    layers_checked: checked,
                }
            } else {
                AuditCoverage::Routine {
                    layers_checked: checked,
                    layers_total: key.config.n_layers,
                }
            }
        }
        None => AuditCoverage::Unknown,
    };

    V4VerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        token_index: r.token_index,
        checks_run,
        checks_passed: checks_run.saturating_sub(failures.len()),
        failures,
        coverage,
        duration: start.elapsed(),
    }
}
