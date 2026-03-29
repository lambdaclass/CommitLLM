//! Canonical production verifier.
//!
//! **Accepts**: VV4A binary audit only.
//! **Trusted mode**: key-only, full-bridge-only.
//! **Rejects**: missing shell, missing manifest, missing initial_residual,
//!   missing embedding proof, unsupported decode features.
//! **Pass means**: every enabled canonical check succeeded.
//!
//! # Structure
//!
//! ```text
//! verify_binary / verify_response
//!   └─ Ctx::new() → run(Ctx) → V4VerifyReport
//!        ├─ phase_structural  → StructuralState { shell, coverage }
//!        ├─ phase_embedding     (reads StructuralState)
//!        ├─ phase_specs       → SpecState { decode_params, output_spec, input_spec, detok_policy }
//!        ├─ phase_output_policy (reads SpecState.output_spec)
//!        ├─ phase_bridge      → BridgeState { final_hidden }
//!        │    └─ bridge_layers
//!        │         ├─ check_qkv         (Wq/Wk/Wv Freivalds)
//!        │         ├─ check_wo          (Wo Freivalds)
//!        │         ├─ bridge_attn_to_ffn (residual + RMSNorm → x_ffn)
//!        │         ├─ check_ffn         (Wg/Wu/Wd Freivalds + SiLU)
//!        │         └─ bridge_ffn_to_next (residual → next x_attn)
//!        ├─ phase_lm_head       (reads BridgeState + SpecState)
//!        ├─ phase_deep_prefix
//!        │    ├─ replay_deep_prefix_attention (prefix tokens j≥1)
//!        │    └─ replay_opened_token_layer    (opened token via prefix KV)
//!        ├─ phase_tokenization   (reads SpecState.input_spec)
//!        └─ phase_detokenization (reads SpecState.detok_policy)
//! ```

use std::time::Instant;

use verilm_core::constants::MatrixType;
use verilm_core::types::{
    CommitmentVersion, InputSpec, OutputSpec, RetainedTokenState, ShellTokenOpening,
    V4AuditResponse, VerifierKey,
};
use verilm_core::{freivalds, merkle, serialize};

use crate::{
    AuditCoverage, Detokenizer, FailureCode, FailureContext, PromptTokenizer, V4VerifyReport,
    Verdict, VerificationFailure,
};

// --- Public API

/// Verify a V4 audit from canonical binary wire format.
///
/// Returns `Err` only for format-level failures (bad magic, decompression,
/// malformed bincode). All protocol-level failures produce `Ok(report)`.
pub fn verify_binary(
    key: &VerifierKey,
    audit_bytes: &[u8],
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> Result<V4VerifyReport, String> {
    let response = serialize::deserialize_v4_audit(audit_bytes)?;
    Ok(verify_response(key, &response, tokenizer, detokenizer))
}

/// Verify a deserialized V4 audit response.
///
/// This is the trusted path — all public entrypoints delegate here.
pub fn verify_response(
    key: &VerifierKey,
    r: &V4AuditResponse,
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> V4VerifyReport {
    run(&Ctx::new(key, r, tokenizer, detokenizer))
}

// --- Orchestrator state

struct Ctx<'a> {
    key: &'a VerifierKey,
    r: &'a V4AuditResponse,
    tokenizer: Option<&'a dyn PromptTokenizer>,
    detokenizer: Option<&'a dyn Detokenizer>,
    start: Instant,
    // Precomputed
    gen_start: u32,
    is_last: bool,
    n_prompt: u32,
}

impl<'a> Ctx<'a> {
    fn new(
        key: &'a VerifierKey,
        r: &'a V4AuditResponse,
        tokenizer: Option<&'a dyn PromptTokenizer>,
        detokenizer: Option<&'a dyn Detokenizer>,
    ) -> Self {
        let n_prompt = r.commitment.n_prompt_tokens.or(r.n_prompt_tokens).unwrap_or(0);
        Self {
            key,
            r,
            tokenizer,
            detokenizer,
            start: Instant::now(),
            gen_start: n_prompt.saturating_sub(1),
            is_last: r.token_index == r.commitment.n_tokens.saturating_sub(1),
            n_prompt,
        }
    }
}

struct St {
    checks: usize,
    failures: Vec<VerificationFailure>,
}

impl St {
    fn new() -> Self { Self { checks: 0, failures: Vec::new() } }
    fn check(&mut self) { self.checks += 1; }
    fn fail(&mut self, code: FailureCode, msg: impl Into<String>) {
        self.failures.push(VerificationFailure {
            category: code.category(),
            code,
            message: msg.into(),
            context: FailureContext::default(),
        });
    }
    fn fail_ctx(&mut self, code: FailureCode, msg: impl Into<String>, ctx: FailureContext) {
        self.failures.push(VerificationFailure {
            category: code.category(),
            code,
            message: msg.into(),
            context: ctx,
        });
    }
}

struct StructuralState<'a> {
    shell: Option<&'a ShellTokenOpening>,
    coverage: AuditCoverage,
}

struct SpecState {
    decode_params: Option<verilm_core::sampling::DecodeParams>,
    output_spec: Option<OutputSpec>,
    input_spec: Option<InputSpec>,
    detokenization_policy: Option<String>,
}

// --- Orchestrator

fn run(ctx: &Ctx) -> V4VerifyReport {
    let mut st = St::new();

    let structural = phase_structural(ctx, &mut st);
    phase_embedding(ctx, &structural, &mut st);
    let specs = phase_specs(ctx, &mut st);

    if let Some(shell) = structural.shell {
        if let Some(ref os) = specs.output_spec {
            phase_output_policy(ctx, os, &mut st);
        }
        let bridge = phase_bridge(ctx, shell, &mut st);
        phase_lm_head(ctx, shell, &bridge, &specs, &mut st);
    }

    phase_deep_prefix(ctx, &mut st);
    phase_tokenization(ctx, &specs, &mut st);
    phase_detokenization(ctx, &specs, &mut st);

    finish(ctx, st, structural.coverage)
}

// --- Phase 1: Structural checks

fn phase_structural<'a>(ctx: &'a Ctx, st: &mut St) -> StructuralState<'a> {
    check_version(ctx, st);
    check_seed(ctx, st);
    check_prompt_hash(ctx, st);
    check_n_prompt_tokens(ctx, st);
    check_merkle_proofs(ctx, st);
    check_io_chain(ctx, st);
    check_prefix_count(ctx, st);

    let shell = match &ctx.r.shell_opening {
        Some(s) => {
            let checked = s
                .layer_indices
                .as_ref()
                .map_or(s.layers.len(), |v| v.len());
            let coverage = if checked >= ctx.key.config.n_layers {
                AuditCoverage::Full {
                    layers_checked: checked,
                }
            } else {
                AuditCoverage::Routine {
                    layers_checked: checked,
                    layers_total: ctx.key.config.n_layers,
                }
            };
            return StructuralState {
                shell: Some(s),
                coverage,
            };
        }
        None => {
            st.fail(
                FailureCode::MissingShellOpening,
                "canonical: shell_opening required",
            );
            None
        }
    };

    StructuralState {
        shell,
        coverage: AuditCoverage::Unknown,
    }
}
fn check_version(ctx: &Ctx, st: &mut St) {
    st.check();
    if ctx.r.commitment.version != CommitmentVersion::V4 {
        st.fail(
            FailureCode::WrongCommitmentVersion,
            format!("expected V4, got {:?}", ctx.r.commitment.version),
        );
    }
}
fn check_seed(ctx: &Ctx, st: &mut St) {
    st.check();
    match ctx.r.commitment.seed_commitment {
        Some(expected) => {
            if merkle::hash_seed(&ctx.r.revealed_seed) != expected {
                st.fail(
                    FailureCode::SeedMismatch,
                    "hash(revealed_seed) != commitment.seed_commitment",
                );
            }
        }
        None => st.fail(
            FailureCode::MissingSeedCommitment,
            "V4 commitment missing seed_commitment",
        ),
    }
}

fn check_prompt_hash(ctx: &Ctx, st: &mut St) {
    st.check();
    match (&ctx.r.prompt, ctx.r.commitment.prompt_hash) {
        (Some(prompt), Some(committed)) => {
            if merkle::hash_prompt(prompt) != committed {
                st.fail(FailureCode::PromptHashMismatch, "hash(prompt) != prompt_hash");
            }
        }
        (None, Some(_)) => st.fail(
            FailureCode::MissingPromptBytes,
            "commitment has prompt_hash but response missing prompt",
        ),
        (Some(_), None) => st.fail(
            FailureCode::UncommittedPrompt,
            "response has prompt but commitment missing prompt_hash",
        ),
        (None, None) => st.fail(
            FailureCode::MissingPromptHash,
            "V4 commitment missing prompt_hash",
        ),
    }
}
fn check_n_prompt_tokens(ctx: &Ctx, st: &mut St) {
    st.check();
    match (ctx.r.commitment.n_prompt_tokens, ctx.r.n_prompt_tokens) {
        (Some(committed), Some(response)) => {
            if committed != response {
                st.fail(
                    FailureCode::NPromptTokensMismatch,
                    format!(
                        "n_prompt_tokens: commitment={} response={}",
                        committed, response
                    ),
                );
            }
            if committed > ctx.r.commitment.n_tokens + 1 {
                st.fail(
                    FailureCode::NPromptTokensBound,
                    format!(
                        "n_prompt_tokens {} exceeds n_tokens {} + 1",
                        committed, ctx.r.commitment.n_tokens
                    ),
                );
            }
        }
        (None, _) => st.fail(
            FailureCode::MissingNPromptTokens,
            "commitment missing n_prompt_tokens",
        ),
        (_, None) => st.fail(
            FailureCode::MissingNPromptTokens,
            "response missing n_prompt_tokens",
        ),
    }
}
fn check_merkle_proofs(ctx: &Ctx, st: &mut St) {
    // Retained leaf (includes final_residual when present)
    st.check();
    let fr_ref = ctx
        .r
        .shell_opening
        .as_ref()
        .and_then(|s| s.final_residual.as_deref());
    let leaf = merkle::hash_retained_with_residual(&ctx.r.retained, fr_ref);
    if !merkle::verify(&ctx.r.commitment.merkle_root, &leaf, &ctx.r.merkle_proof) {
        st.fail_ctx(
            FailureCode::MerkleProofFailed,
            format!(
                "token {}: retained leaf Merkle proof failed",
                ctx.r.token_index
            ),
            FailureContext {
                token_index: Some(ctx.r.token_index),
                ..Default::default()
            },
        );
    }

    // Prefix retained leaves
    for (j, (hash, proof)) in ctx
        .r
        .prefix_leaf_hashes
        .iter()
        .zip(ctx.r.prefix_merkle_proofs.iter())
        .enumerate()
    {
        st.check();
        if !merkle::verify(&ctx.r.commitment.merkle_root, hash, proof) {
            st.fail_ctx(
                FailureCode::MerkleProofFailed,
                format!("prefix token {}: Merkle proof failed", j),
                FailureContext {
                    token_index: Some(j as u32),
                    ..Default::default()
                },
            );
        }
    }
}
fn check_io_chain(ctx: &Ctx, st: &mut St) {
    let r = ctx.r;

    // Replay IO chain from genesis
    st.check();
    let io_genesis = match r.commitment.prompt_hash {
        Some(ph) => merkle::io_genesis_v4(ph),
        None => [0u8; 32],
    };
    let mut prev_io = io_genesis;
    for (j, hash) in r.prefix_leaf_hashes.iter().enumerate() {
        prev_io = merkle::io_hash_v4(*hash, r.prefix_token_ids[j], prev_io);
    }
    if prev_io != r.prev_io_hash {
        st.fail_ctx(
            FailureCode::IoChainMismatch,
            format!("token {}: prev_io_hash mismatch", r.token_index),
            FailureContext {
                token_index: Some(r.token_index),
                ..Default::default()
            },
        );
    }

    // IO chain proof
    st.check();
    let fr_ref = r
        .shell_opening
        .as_ref()
        .and_then(|s| s.final_residual.as_deref());
    let leaf = merkle::hash_retained_with_residual(&r.retained, fr_ref);
    let challenged_io = merkle::io_hash_v4(leaf, r.token_id, prev_io);
    if !merkle::verify(&r.commitment.io_root, &challenged_io, &r.io_proof) {
        st.fail_ctx(
            FailureCode::IoChainProofFailed,
            format!("token {}: IO chain proof failed", r.token_index),
            FailureContext {
                token_index: Some(r.token_index),
                ..Default::default()
            },
        );
    }
}
fn check_prefix_count(ctx: &Ctx, st: &mut St) {
    st.check();
    if ctx.r.prefix_leaf_hashes.len() != ctx.r.token_index as usize {
        st.fail_ctx(
            FailureCode::PrefixTokenCountMismatch,
            format!(
                "token {}: expected {} prefix tokens, got {}",
                ctx.r.token_index,
                ctx.r.token_index,
                ctx.r.prefix_leaf_hashes.len()
            ),
            FailureContext {
                token_index: Some(ctx.r.token_index),
                ..Default::default()
            },
        );
    }
}

// --- Phase 2: Embedding binding

fn phase_embedding(ctx: &Ctx, structural: &StructuralState, st: &mut St) {
    if let Some(shell) = structural.shell {
        check_embedding_binding(ctx, shell, st);
    }
    check_rich_prefix_embeddings(ctx, st);
}

fn check_embedding_binding(ctx: &Ctx, shell: &ShellTokenOpening, st: &mut St) {
    let emb_root = match &ctx.key.embedding_merkle_root {
        Some(root) => root,
        None => {
            if shell.initial_residual.is_some() {
                st.check();
                st.fail(
                    FailureCode::UnboundInitialResidual,
                    "shell has initial_residual but key has no embedding_merkle_root",
                );
            }
            return;
        }
    };

    st.check();
    match (&shell.initial_residual, &shell.embedding_proof) {
        (Some(ir), Some(proof)) => {
            let leaf = merkle::hash_embedding_row(ir);
            if proof.leaf_index != ctx.r.token_id {
                st.fail(
                    FailureCode::EmbeddingLeafMismatch,
                    format!(
                        "embedding proof leaf_index {} != token_id {}",
                        proof.leaf_index, ctx.r.token_id
                    ),
                );
            } else if !merkle::verify(emb_root, &leaf, proof) {
                st.fail(
                    FailureCode::EmbeddingProofFailed,
                    "embedding Merkle proof failed",
                );
            }
        }
        (None, _) => st.fail(
            FailureCode::MissingInitialResidual,
            "canonical: initial_residual required (key has embedding_merkle_root)",
        ),
        (Some(_), None) => st.fail(
            FailureCode::MissingEmbeddingProof,
            "canonical: embedding_proof required (key has embedding_merkle_root)",
        ),
    }
}

fn check_rich_prefix_embeddings(ctx: &Ctx, st: &mut St) {
    let (emb_root, rows, proofs) = match (
        &ctx.key.embedding_merkle_root,
        &ctx.r.prefix_embedding_rows,
        &ctx.r.prefix_embedding_proofs,
    ) {
        (Some(root), Some(rows), Some(proofs)) => (root, rows, proofs),
        _ => return,
    };

    if rows.len() != ctx.r.prefix_token_ids.len() || proofs.len() != ctx.r.prefix_token_ids.len()
    {
        st.check();
        st.fail(
            FailureCode::PrefixCountMismatch,
            format!(
                "prefix embedding count: {} rows, {} proofs, {} token_ids",
                rows.len(),
                proofs.len(),
                ctx.r.prefix_token_ids.len()
            ),
        );
        return;
    }

    for (j, ((row, proof), &tid)) in rows
        .iter()
        .zip(proofs.iter())
        .zip(ctx.r.prefix_token_ids.iter())
        .enumerate()
    {
        st.check();
        let leaf = merkle::hash_embedding_row(row);
        if proof.leaf_index != tid {
            st.fail_ctx(
                FailureCode::EmbeddingLeafMismatch,
                format!(
                    "prefix {}: embedding leaf_index {} != token_id {}",
                    j, proof.leaf_index, tid
                ),
                FailureContext {
                    token_index: Some(j as u32),
                    ..Default::default()
                },
            );
        } else if !merkle::verify(emb_root, &leaf, proof) {
            st.fail_ctx(
                FailureCode::EmbeddingProofFailed,
                format!("prefix {}: embedding Merkle proof failed", j),
                FailureContext {
                    token_index: Some(j as u32),
                    ..Default::default()
                },
            );
        }
    }
}

// --- Phase 3: Manifest + spec binding

fn phase_specs(ctx: &Ctx, st: &mut St) -> SpecState {
    let manifest = match &ctx.r.manifest {
        Some(m) => m,
        None => {
            st.fail(
                FailureCode::MissingManifestHash,
                "canonical: manifest required for spec verification",
            );
            return SpecState {
                decode_params: None,
                output_spec: None,
                input_spec: None,
                detokenization_policy: None,
            };
        }
    };

    let (input_spec, model_spec, decode_spec, output_spec) = manifest.split();
    let hashes = [
        merkle::hash_input_spec(&input_spec),
        merkle::hash_model_spec(&model_spec),
        merkle::hash_decode_spec(&decode_spec),
        merkle::hash_output_spec(&output_spec),
    ];
    check_spec_hashes(ctx, &hashes, st);
    check_manifest_hash(ctx, &hashes, st);
    cross_check_model_vs_key(ctx, &model_spec, st);
    check_decode_features(&decode_spec, &output_spec, st);

    let detokenization_policy = output_spec.detokenization_policy.clone();

    SpecState {
        decode_params: Some(verilm_core::sampling::DecodeParams {
            temperature: decode_spec.temperature,
            top_k: decode_spec.top_k,
            top_p: decode_spec.top_p,
        }),
        output_spec: Some(output_spec),
        input_spec: Some(input_spec),
        detokenization_policy,
    }
}

fn check_spec_hashes(ctx: &Ctx, hashes: &[[u8; 32]; 4], st: &mut St) {
    let pairs: [(&str, Option<[u8; 32]>, [u8; 32]); 4] = [
        ("input",  ctx.r.commitment.input_spec_hash,  hashes[0]),
        ("model",  ctx.r.commitment.model_spec_hash,  hashes[1]),
        ("decode", ctx.r.commitment.decode_spec_hash, hashes[2]),
        ("output", ctx.r.commitment.output_spec_hash, hashes[3]),
    ];
    for (name, committed, computed) in &pairs {
        st.check();
        match committed {
            None => st.fail_ctx(
                FailureCode::MissingSpecHash,
                format!("commitment missing {}_spec_hash", name),
                FailureContext {
                    spec: Some((*name).into()),
                    ..Default::default()
                },
            ),
            Some(h) if h != computed => st.fail_ctx(
                FailureCode::SpecHashMismatch,
                format!("{}_spec_hash mismatch", name),
                FailureContext {
                    spec: Some((*name).into()),
                    ..Default::default()
                },
            ),
            _ => {}
        }
    }
}

fn check_manifest_hash(ctx: &Ctx, hashes: &[[u8; 32]; 4], st: &mut St) {
    st.check();
    match ctx.r.commitment.manifest_hash {
        None => st.fail(FailureCode::MissingManifestHash, "commitment missing manifest_hash"),
        Some(committed) => {
            if merkle::hash_manifest_composed(hashes[0], hashes[1], hashes[2], hashes[3]) != committed {
                st.fail(FailureCode::ManifestHashMismatch, "manifest hash mismatch");
            }
        }
    }
}

fn cross_check_model_vs_key(
    ctx: &Ctx,
    spec: &verilm_core::types::ModelSpec,
    st: &mut St,
) {
    xcheck_f64(st, "rmsnorm_eps", spec.rmsnorm_eps, ctx.key.rmsnorm_eps);
    xcheck_f64(st, "rope_theta", spec.rope_theta, ctx.key.config.rope_theta);
    xcheck_hash(st, "rope_config_hash", spec.rope_config_hash, ctx.key.rope_config_hash);
    xcheck_hash(st, "weight_hash", spec.weight_hash, ctx.key.weight_hash);
    xcheck_hash(st, "embedding_merkle_root", spec.embedding_merkle_root, ctx.key.embedding_merkle_root);
    xcheck_dim(st, "n_layers", spec.n_layers, ctx.key.config.n_layers);
    xcheck_dim(st, "hidden_dim", spec.hidden_dim, ctx.key.config.hidden_dim);
    xcheck_dim(st, "vocab_size", spec.vocab_size, ctx.key.config.vocab_size);
    xcheck_dim(st, "kv_dim", spec.kv_dim, ctx.key.config.kv_dim);
    xcheck_dim(st, "ffn_dim", spec.ffn_dim, ctx.key.config.ffn_dim);
    xcheck_dim(st, "d_head", spec.d_head, ctx.key.config.d_head);
    xcheck_dim(st, "n_q_heads", spec.n_q_heads, ctx.key.config.n_q_heads);
    xcheck_dim(st, "n_kv_heads", spec.n_kv_heads, ctx.key.config.n_kv_heads);
    xcheck_str(st, "quant_family", spec.quant_family.as_deref(), ctx.key.quant_family.as_deref());
    xcheck_str(st, "scale_derivation", spec.scale_derivation.as_deref(), ctx.key.scale_derivation.as_deref());
    if let (Some(m), Some(k)) = (spec.quant_block_size, ctx.key.quant_block_size) {
        st.check();
        if m != k {
            st.fail_ctx(
                FailureCode::SpecFieldMismatch,
                format!("quant_block_size mismatch: manifest={} key={}", m, k),
                FailureContext {
                    field: Some("quant_block_size".into()),
                    ..Default::default()
                },
            );
        }
    }
}

fn check_decode_features(
    decode: &verilm_core::types::DecodeSpec,
    output: &OutputSpec,
    st: &mut St,
) {
    // Sampler version
    st.check();
    match decode.sampler_version.as_deref() {
        Some("chacha20-vi-sample-v1") | None => {}
        Some(other) => st.fail(
            FailureCode::UnsupportedSamplerVersion,
            format!("unsupported sampler_version='{}'", other),
        ),
    }

    // Decode mode consistency
    if let Some(ref dm) = decode.decode_mode {
        st.check();
        match dm.as_str() {
            "greedy" if decode.temperature != 0.0 => st.fail(
                FailureCode::DecodeModeTempInconsistent,
                format!("decode_mode='greedy' but temperature={}", decode.temperature),
            ),
            "sampled" if decode.temperature == 0.0 => st.fail(
                FailureCode::DecodeModeTempInconsistent,
                "decode_mode='sampled' but temperature=0.0",
            ),
            "greedy" | "sampled" => {}
            other => st.fail(
                FailureCode::UnsupportedDecodeMode,
                format!("unsupported decode_mode='{}'", other),
            ),
        }
    }

    st.check();
    reject_feature(st, decode.repetition_penalty != 1.0, "repetition_penalty", &format!("{}", decode.repetition_penalty), "1.0");
    reject_feature(st, decode.frequency_penalty != 0.0, "frequency_penalty", &format!("{}", decode.frequency_penalty), "0.0");
    reject_feature(st, decode.presence_penalty != 0.0, "presence_penalty", &format!("{}", decode.presence_penalty), "0.0");
    reject_feature(st, !decode.logit_bias.is_empty(), "logit_bias", &format!("{} entries", decode.logit_bias.len()), "empty");
    reject_feature(st, !decode.bad_word_ids.is_empty(), "bad_word_ids", &format!("{} entries", decode.bad_word_ids.len()), "empty");
    reject_feature(st, !decode.guided_decoding.is_empty(), "guided_decoding", &decode.guided_decoding, "empty");
    reject_feature(st, !output.stop_sequences.is_empty(), "stop_sequences", &format!("{} entries", output.stop_sequences.len()), "empty");
}

// --- Phase 4: Output policy

fn phase_output_policy(ctx: &Ctx, os: &OutputSpec, st: &mut St) {
    let r = ctx.r;
    if os.max_tokens > 0 {
        let n_generated = r.commitment.n_tokens.saturating_sub(ctx.n_prompt.saturating_sub(1));
        if r.token_index >= r.commitment.n_tokens {
            st.fail(
                FailureCode::ExceedsMaxTokens,
                format!("token_index {} >= n_tokens {}", r.token_index, r.commitment.n_tokens),
            );
        }
        if n_generated > os.max_tokens {
            st.fail(
                FailureCode::ExceedsMaxTokens,
                format!("generated {} tokens > max_tokens {}", n_generated, os.max_tokens),
            );
        }
    }
    if let Some(eos_id) = os.eos_token_id {
        check_min_tokens(ctx, os, eos_id, st);
        check_ignore_eos(r, os, eos_id, st);
        check_eos_policy(ctx, os, eos_id, st);
    } else if os.min_tokens > 0 || os.ignore_eos {
        st.fail(
            FailureCode::MissingEosTokenId,
            "output_spec needs eos_token_id for min_tokens/ignore_eos enforcement",
        );
    }
}

fn check_min_tokens(ctx: &Ctx, os: &OutputSpec, eos_id: u32, st: &mut St) {
    if os.min_tokens == 0 {
        return;
    }
    st.check();
    let r = ctx.r;
    let n_generated = r.commitment.n_tokens.saturating_sub(ctx.gen_start);
    if n_generated < os.min_tokens {
        st.fail(
            FailureCode::MinTokensViolated,
            format!("only {} generated tokens, min_tokens={}", n_generated, os.min_tokens),
        );
    }
    let gen_index = r.token_index.saturating_sub(ctx.gen_start);
    if gen_index < os.min_tokens && r.token_id == eos_id {
        st.fail(
            FailureCode::MinTokensViolated,
            format!("EOS at generation position {} < min_tokens={}", gen_index, os.min_tokens),
        );
    }
}
fn check_ignore_eos(r: &V4AuditResponse, os: &OutputSpec, eos_id: u32, st: &mut St) {
    if !os.ignore_eos {
        return;
    }
    st.check();
    if r.token_id == eos_id {
        st.fail(
            FailureCode::IgnoreEosViolated,
            format!("token_id {} is EOS but ignore_eos=true", r.token_id),
        );
    }
}
fn check_eos_policy(ctx: &Ctx, os: &OutputSpec, eos_id: u32, st: &mut St) {
    if os.eos_policy == "stop" {
        st.check();
        let r = ctx.r;
        if r.token_id == eos_id && !ctx.is_last {
            st.fail(
                FailureCode::EosPolicyViolated,
                format!(
                    "eos_policy='stop': EOS at index {} is not last (n_tokens={})",
                    r.token_index, r.commitment.n_tokens
                ),
            );
        }
    } else if os.eos_policy != "sample" {
        st.fail(
            FailureCode::UnknownEosPolicy,
            format!("unknown eos_policy='{}'", os.eos_policy),
        );
    }
}

// --- Phase 5: Full bridge (Freivalds + residual chain)

struct BridgeState {
    final_hidden: Option<Vec<i8>>,
}

fn phase_bridge(ctx: &Ctx, shell: &ShellTokenOpening, st: &mut St) -> BridgeState {
    bridge_layers(ctx.key, &ctx.r.retained, shell, ctx.r.token_index, st);

    // Derive final_hidden from captured final_residual + final_norm.
    // No shell-replay fallback in canonical path.
    let final_hidden: Option<Vec<i8>> = match (&shell.final_residual, &ctx.key.final_norm_weights)
    {
        (Some(fr), Some(fnw)) => {
            let res_f64: Vec<f64> = fr.iter().map(|&v| v as f64).collect();
            let normed = verilm_core::rmsnorm::rmsnorm_f64_input(&res_f64, fnw, ctx.key.rmsnorm_eps);
            Some(normed.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect())
        }
        (None, Some(_)) if ctx.key.lm_head.is_some() => {
            st.fail(
                FailureCode::MissingFinalResidual,
                "canonical: final_residual required for LM-head (no shell replay fallback)",
            );
            None
        }
        _ => None,
    };

    BridgeState { final_hidden }
}

/// Validate bridge shape: layer_indices contiguity and count match.
///
/// Returns the opened layer index list, or `None` if validation failed.
fn validate_bridge_shape(
    shell: &ShellTokenOpening,
    retained: &RetainedTokenState,
    st: &mut St,
) -> Option<Vec<usize>> {
    let opened: Vec<usize> = shell
        .layer_indices
        .clone()
        .unwrap_or_else(|| (0..retained.layers.len()).collect());

    if shell.layers.len() != opened.len() {
        st.fail(
            FailureCode::ShellLayerCountMismatch,
            format!(
                "shell has {} layers, indices specify {}",
                shell.layers.len(),
                opened.len()
            ),
        );
        return None;
    }
    if let Some(ref indices) = shell.layer_indices {
        if !indices.iter().enumerate().all(|(i, &li)| li == i) {
            st.fail(
                FailureCode::NonContiguousLayerIndices,
                format!(
                    "layer_indices must be contiguous prefix 0..N, got {:?}",
                    indices
                ),
            );
            return None;
        }
    }

    Some(opened)
}

fn bridge_layers(
    key: &VerifierKey,
    retained: &RetainedTokenState,
    shell: &ShellTokenOpening,
    token_index: u32,
    st: &mut St,
) {
    let ir = match &shell.initial_residual {
        Some(ir) => ir,
        None => return, // already recorded in phase_embedding
    };
    if key.rmsnorm_attn_weights.is_empty()
        || key.rmsnorm_ffn_weights.is_empty()
        || retained.layers.is_empty()
    {
        return;
    }
    // validate_bridge_shape enforces contiguous 0..N, so shell.layers[i]
    // corresponds to retained.layers[i] — no index map needed.
    if validate_bridge_shape(shell, retained, st).is_none() {
        return;
    }

    let (mut residual, mut x_attn) =
        init_residual_chain(ir, key, shell.layers[0].scale_x_attn);

    let n_layers = shell.layers.len().min(retained.layers.len());
    for layer_idx in 0..n_layers {
        let rs = &retained.layers[layer_idx];
        let sl = &shell.layers[layer_idx];

        check_qkv(key, st, layer_idx, sl, &x_attn);
        // Token-0 attention replay: with seq_len=1, softmax is trivial,
        // so we can verify a = GQA_expand(requant(Wv·x)) at zero extra cost.
        if token_index == 0 {
            check_attention_token0(key, st, layer_idx, sl, rs);
        }
        check_wo(key, st, layer_idx, &rs.a, &sl.attn_out);
        let x_ffn = bridge_attn_to_ffn(key, layer_idx, rs, sl, &sl.attn_out, &mut residual);
        check_ffn(key, st, layer_idx, sl, &x_ffn);
        let next_scale = shell
            .layers
            .get(layer_idx + 1)
            .map(|s| s.scale_x_attn)
            .unwrap_or(1.0);
        x_attn = bridge_ffn_to_next(key, layer_idx, sl, &sl.ffn_out, &mut residual, next_scale);
    }
}

/// Token-0 attention replay: verify retained `a` matches replay from QKV.
///
/// For token_index == 0 the KV cache has exactly one entry (self), so
/// softmax([score]) = [1.0] and the attention output is simply the GQA
/// head-expanded requantized V projection. This costs no extra data —
/// Q, K, V accumulators are already in the shell opening.
///
/// Two paths:
/// - **Toy/reference** (`rope_aware_replay == false`): raw `requantize` i32→i8, no RoPE.
/// - **Production** (`rope_aware_replay == true`): dequantize using weight+activation
///   scales, apply RoPE (identity at position 0), replay in f64, requantize with `scale_a`.
fn check_attention_token0(
    key: &VerifierKey,
    st: &mut St,
    layer_idx: usize,
    sl: &verilm_core::types::ShellLayerOpening,
    rs: &verilm_core::types::RetainedLayerState,
) {
    let (q_acc, k_acc, v_acc) = match (&sl.q, &sl.k, &sl.v) {
        (Some(q), Some(k), Some(v)) => (q, k, v),
        _ => return, // QKV missing — already reported by check_qkv
    };

    st.check();

    let expected_a = if key.rope_aware_replay {
        let (q_roped, k_roped, v_deq) =
            dequant_rope_qkv(key, layer_idx, q_acc, k_acc, v_acc, sl.scale_x_attn, 0);
        verilm_core::attention::replay_attention_roped(
            &q_roped, &[k_roped], &[v_deq], rs.scale_a as f64, &key.config,
        )
    } else {
        let q_i8 = verilm_core::requantize(q_acc);
        let k_i8 = verilm_core::requantize(k_acc);
        let v_i8 = verilm_core::requantize(v_acc);
        verilm_core::attention::replay_attention_reference(
            &q_i8, &[k_i8], &[v_i8], &key.config,
        )
    };

    let tolerance = verilm_core::attention::AttentionToleranceConfig::default();
    if let Some(max_diff) = verilm_core::attention::compare_attention_output(
        &rs.a, &expected_a, &tolerance,
    ) {
        st.fail_ctx(
            FailureCode::AttentionReplayMismatch,
            format!(
                "layer {}: token-0 attention replay mismatch (max_diff={})",
                layer_idx, max_diff
            ),
            FailureContext {
                layer: Some(layer_idx),
                ..Default::default()
            },
        );
    }
}

/// Dequantize Q/K/V accumulators and apply RoPE to Q and K.
///
/// Returns `(q_roped_f64, k_roped_f64, v_deq_f64)`.
fn dequant_rope_qkv(
    key: &VerifierKey,
    layer_idx: usize,
    q_acc: &[i32],
    k_acc: &[i32],
    v_acc: &[i32],
    scale_x_attn: f32,
    position: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let scale_wq = key.weight_scale_for(layer_idx, MatrixType::Wq);
    let scale_wk = key.weight_scale_for(layer_idx, MatrixType::Wk);
    let scale_wv = key.weight_scale_for(layer_idx, MatrixType::Wv);
    let sx = Some(scale_x_attn);

    let q_f64 = verilm_core::rope::dequantize_acc(q_acc, Some(scale_wq), sx);
    let k_f64 = verilm_core::rope::dequantize_acc(k_acc, Some(scale_wk), sx);
    let v_f64 = verilm_core::rope::dequantize_acc(v_acc, Some(scale_wv), sx);

    let q_roped = verilm_core::rope::apply_rope_q(&q_f64, position, &key.config);
    let k_roped = verilm_core::rope::apply_rope_k(&k_f64, position, &key.config);

    (q_roped, k_roped, v_f64)
}

fn init_residual_chain(ir: &[f32], key: &VerifierKey, first_scale: f32) -> (Vec<f64>, Vec<i8>) {
    let residual: Vec<f64> = ir.iter().map(|&v| v as f64).collect();
    let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
        &residual,
        &key.rmsnorm_attn_weights[0],
        key.rmsnorm_eps,
    );
    let x_attn = verilm_core::rmsnorm::quantize_f64_to_i8(&normed, first_scale as f64);
    (residual, x_attn)
}

fn check_qkv(
    key: &VerifierKey,
    st: &mut St,
    layer_idx: usize,
    sl: &verilm_core::types::ShellLayerOpening,
    x_attn: &[i8],
) {
    for (mt, acc) in [
        (MatrixType::Wq, &sl.q),
        (MatrixType::Wk, &sl.k),
        (MatrixType::Wv, &sl.v),
    ] {
        match acc {
            Some(z) => verify_freivalds(key, st, layer_idx, mt, x_attn, z),
            None => st.fail_ctx(
                FailureCode::MissingQkv,
                format!("layer {} {:?}: QKV required in canonical path", layer_idx, mt),
                FailureContext {
                    layer: Some(layer_idx),
                    matrix: Some(format!("{:?}", mt)),
                    ..Default::default()
                },
            ),
        }
    }
}

fn check_wo(
    key: &VerifierKey,
    st: &mut St,
    layer_idx: usize,
    a: &[i8],
    attn_out: &[i32],
) {
    verify_freivalds(key, st, layer_idx, MatrixType::Wo, a, attn_out);
}

fn bridge_attn_to_ffn(
    key: &VerifierKey,
    layer_idx: usize,
    rs: &verilm_core::types::RetainedLayerState,
    sl: &verilm_core::types::ShellLayerOpening,
    attn_out: &[i32],
    residual: &mut Vec<f64>,
) -> Vec<i8> {
    verilm_core::rmsnorm::bridge_residual_rmsnorm(
        attn_out,
        key.weight_scale_for(layer_idx, MatrixType::Wo),
        rs.scale_a,
        residual,
        &key.rmsnorm_ffn_weights[layer_idx],
        key.rmsnorm_eps,
        sl.scale_x_ffn,
    )
}

fn check_ffn(
    key: &VerifierKey,
    st: &mut St,
    layer_idx: usize,
    sl: &verilm_core::types::ShellLayerOpening,
    x_ffn: &[i8],
) {
    verify_freivalds(key, st, layer_idx, MatrixType::Wg, x_ffn, &sl.g);
    verify_freivalds(key, st, layer_idx, MatrixType::Wu, x_ffn, &sl.u);
    let h = verilm_core::silu::compute_h_scaled(
        &sl.g,
        &sl.u,
        key.weight_scale_for(layer_idx, MatrixType::Wg),
        key.weight_scale_for(layer_idx, MatrixType::Wu),
        sl.scale_x_ffn,
        sl.scale_h,
    );
    verify_freivalds(key, st, layer_idx, MatrixType::Wd, &h, &sl.ffn_out);
}

fn bridge_ffn_to_next(
    key: &VerifierKey,
    layer_idx: usize,
    sl: &verilm_core::types::ShellLayerOpening,
    ffn_out: &[i32],
    residual: &mut Vec<f64>,
    next_scale: f32,
) -> Vec<i8> {
    if layer_idx + 1 < key.rmsnorm_attn_weights.len() {
        verilm_core::rmsnorm::bridge_residual_rmsnorm(
            ffn_out,
            key.weight_scale_for(layer_idx, MatrixType::Wd),
            sl.scale_h,
            residual,
            &key.rmsnorm_attn_weights[layer_idx + 1],
            key.rmsnorm_eps,
            next_scale,
        )
    } else {
        // Last layer: update residual, no next-layer RMSNorm
        verilm_core::rmsnorm::dequant_add_residual(
            ffn_out,
            key.weight_scale_for(layer_idx, MatrixType::Wd),
            sl.scale_h,
            residual,
        );
        Vec::new()
    }
}

// --- Phase 6: LM-head + token replay

fn phase_lm_head(ctx: &Ctx, shell: &ShellTokenOpening, bridge: &BridgeState, specs: &SpecState, st: &mut St) {
    // LM-head Freivalds
    let r_lm = ctx.key.r_for(MatrixType::LmHead);
    if r_lm.is_empty() || ctx.key.v_lm_head.is_none() {
        return;
    }
    match (&shell.logits_i32, &bridge.final_hidden) {
        (Some(logits), Some(fh)) => {
            st.check();
            if !freivalds::check(ctx.key.v_lm_head.as_ref().unwrap(), fh, r_lm, logits) {
                st.fail(
                    FailureCode::LmHeadFreivaldsFailed,
                    "lm_head: Freivalds check failed on logits_i32",
                );
            }

            // Token replay (generated tokens only)
            if ctx.key.config.vocab_size > 0 && ctx.r.token_index >= ctx.gen_start {
                st.check();
                let logits_f32: Vec<f32> = logits.iter().map(|&v| v as f32).collect();
                let expected = if let Some(ref dp) = specs.decode_params {
                    let seed = verilm_core::sampling::derive_token_seed(
                        &ctx.r.revealed_seed,
                        ctx.r.token_index,
                    );
                    verilm_core::sampling::sample(&logits_f32, dp, &seed)
                } else {
                    logits_f32
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                };
                if expected != ctx.r.token_id {
                    st.fail(
                        FailureCode::TokenSelectionMismatch,
                        format!("lm_head: expected token {} but got {}", expected, ctx.r.token_id),
                    );
                }
            }
        }
        (None, _) => st.fail(
            FailureCode::MissingLogits,
            "lm_head: key requires logits_i32 but shell missing it",
        ),
        (Some(_), None) => st.fail(
            FailureCode::MissingFinalHidden,
            "lm_head: logits_i32 present but no final_hidden to verify against",
        ),
    }
}

// --- Phase 7: Deep prefix

fn phase_deep_prefix(ctx: &Ctx, st: &mut St) {
    let (prefix_ret, prefix_shells) = match (&ctx.r.prefix_retained, &ctx.r.prefix_shell_openings)
    {
        (Some(ret), Some(shells)) => (ret, shells),
        _ => return,
    };

    if prefix_ret.len() != ctx.r.prefix_leaf_hashes.len()
        || prefix_shells.len() != ctx.r.prefix_leaf_hashes.len()
    {
        st.check();
        st.fail(
            FailureCode::PrefixCountMismatch,
            format!(
                "deep prefix count: {} retained, {} shells, {} leaf_hashes",
                prefix_ret.len(),
                prefix_shells.len(),
                ctx.r.prefix_leaf_hashes.len()
            ),
        );
        return;
    }

    for (j, ((ret_j, shell_j), &expected_hash)) in prefix_ret
        .iter()
        .zip(prefix_shells.iter())
        .zip(ctx.r.prefix_leaf_hashes.iter())
        .enumerate()
    {
        // Hash consistency
        st.check();
        let fr_ref = shell_j.final_residual.as_deref();
        let hash_j = merkle::hash_retained_with_residual(ret_j, fr_ref);
        if hash_j != expected_hash {
            st.fail_ctx(
                FailureCode::RetainedHashMismatch,
                format!("prefix token {}: retained hash mismatch", j),
                FailureContext {
                    token_index: Some(j as u32),
                    ..Default::default()
                },
            );
            continue;
        }

        // Shell Freivalds + bridge (reuses bridge_layers).
        // Note: bridge_layers already does token-0 attention replay for j==0
        // via check_attention_token0. The multi-token replay below handles j>0.
        let before = st.failures.len();
        bridge_layers(ctx.key, ret_j, shell_j, j as u32, st);
        for failure in &mut st.failures[before..] {
            failure.message = format!("prefix token {}: {}", j, failure.message);
            if failure.context.token_index.is_none() {
                failure.context.token_index = Some(j as u32);
            }
        }
    }

    // Deep-prefix attention replay: verify retained `a` for all prefix tokens
    // using the accumulated KV cache from prior tokens' shell openings.
    // Token 0 is already covered by check_attention_token0 inside bridge_layers.
    replay_deep_prefix_attention(ctx, prefix_ret, prefix_shells, st);
}

/// Deep-prefix attention replay: verify `a` for prefix tokens j >= 1 and
/// for the opened token, using the accumulated KV cache.
///
/// Two paths:
/// - **Toy** (`rope_aware_replay == false`): raw `requantize` i32→i8, no RoPE.
/// - **Production** (`rope_aware_replay == true`): dequantize → RoPE → f64 replay
///   → requantize with `scale_a`.
///
/// Token 0 is already checked by `check_attention_token0` inside `bridge_layers`.
fn replay_deep_prefix_attention(
    ctx: &Ctx,
    prefix_ret: &[RetainedTokenState],
    prefix_shells: &[ShellTokenOpening],
    st: &mut St,
) {
    if prefix_shells.is_empty() {
        return; // No prefix data to build KV cache from
    }

    let cfg = &ctx.key.config;
    let n_layers = cfg.n_layers.min(
        prefix_shells
            .iter()
            .map(|s| s.layers.len())
            .min()
            .unwrap_or(0),
    );

    if ctx.key.rope_aware_replay {
        replay_deep_prefix_roped(ctx, prefix_ret, prefix_shells, n_layers, st);
    } else {
        replay_deep_prefix_toy(ctx, prefix_ret, prefix_shells, n_layers, st);
    }
}

/// Toy/reference path: raw i8 requantize, no RoPE.
fn replay_deep_prefix_toy(
    ctx: &Ctx,
    prefix_ret: &[RetainedTokenState],
    prefix_shells: &[ShellTokenOpening],
    n_layers: usize,
    st: &mut St,
) {
    let cfg = &ctx.key.config;

    for layer_idx in 0..n_layers {
        let mut kv_k: Vec<Vec<i8>> = Vec::new();
        let mut kv_v: Vec<Vec<i8>> = Vec::new();

        for (j, (shell_j, ret_j)) in prefix_shells.iter().zip(prefix_ret.iter()).enumerate() {
            if layer_idx >= ret_j.layers.len() { break; }
            let sl = &shell_j.layers[layer_idx];
            let rs = &ret_j.layers[layer_idx];
            let (q_acc, k_acc, v_acc) = match (&sl.q, &sl.k, &sl.v) {
                (Some(q), Some(k), Some(v)) => (q, k, v),
                _ => break,
            };

            kv_k.push(verilm_core::requantize(k_acc));
            kv_v.push(verilm_core::requantize(v_acc));

            if j == 0 { continue; } // token 0 handled by check_attention_token0

            st.check();
            let q_i8 = verilm_core::requantize(q_acc);
            let expected_a = verilm_core::attention::replay_attention_reference(
                &q_i8, &kv_k, &kv_v, cfg,
            );
            check_attention_result(st, &rs.a, &expected_a, "prefix token", j, layer_idx);
        }

        // Opened token replay
        if let Some((q_acc, k_acc, v_acc, rs)) = opened_token_qkv(ctx, layer_idx) {
            kv_k.push(verilm_core::requantize(k_acc));
            kv_v.push(verilm_core::requantize(v_acc));
            st.check();
            let q_i8 = verilm_core::requantize(q_acc);
            let expected_a = verilm_core::attention::replay_attention_reference(
                &q_i8, &kv_k, &kv_v, cfg,
            );
            check_attention_result_opened(ctx, st, &rs.a, &expected_a, layer_idx);
        }
    }
}

/// Production path: dequantize + RoPE + f64 replay.
fn replay_deep_prefix_roped(
    ctx: &Ctx,
    prefix_ret: &[RetainedTokenState],
    prefix_shells: &[ShellTokenOpening],
    n_layers: usize,
    st: &mut St,
) {
    let cfg = &ctx.key.config;

    for layer_idx in 0..n_layers {
        let mut kv_k: Vec<Vec<f64>> = Vec::new();
        let mut kv_v: Vec<Vec<f64>> = Vec::new();

        for (j, (shell_j, ret_j)) in prefix_shells.iter().zip(prefix_ret.iter()).enumerate() {
            if layer_idx >= ret_j.layers.len() { break; }
            let sl = &shell_j.layers[layer_idx];
            let rs = &ret_j.layers[layer_idx];
            let (q_acc, k_acc, v_acc) = match (&sl.q, &sl.k, &sl.v) {
                (Some(q), Some(k), Some(v)) => (q, k, v),
                _ => break,
            };

            let (_, k_roped, v_deq) =
                dequant_rope_qkv(ctx.key, layer_idx, q_acc, k_acc, v_acc, sl.scale_x_attn, j);
            kv_k.push(k_roped);
            kv_v.push(v_deq);

            if j == 0 { continue; }

            st.check();
            let (q_roped, _, _) =
                dequant_rope_qkv(ctx.key, layer_idx, q_acc, k_acc, v_acc, sl.scale_x_attn, j);
            let expected_a = verilm_core::attention::replay_attention_roped(
                &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
            );
            check_attention_result(st, &rs.a, &expected_a, "prefix token", j, layer_idx);
        }

        // Opened token replay
        if let Some((q_acc, k_acc, v_acc, rs)) = opened_token_qkv(ctx, layer_idx) {
            let shell = ctx.r.shell_opening.as_ref().unwrap();
            let sl = &shell.layers[layer_idx];
            let pos = ctx.r.token_index as usize;

            let (q_roped, k_roped, v_deq) =
                dequant_rope_qkv(ctx.key, layer_idx, q_acc, k_acc, v_acc, sl.scale_x_attn, pos);
            kv_k.push(k_roped);
            kv_v.push(v_deq);

            st.check();
            let expected_a = verilm_core::attention::replay_attention_roped(
                &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
            );
            check_attention_result_opened(ctx, st, &rs.a, &expected_a, layer_idx);
        }
    }
}

/// Extract the opened token's QKV accumulators and retained state for a given layer.
fn opened_token_qkv<'a>(
    ctx: &'a Ctx,
    layer_idx: usize,
) -> Option<(&'a [i32], &'a [i32], &'a [i32], &'a verilm_core::types::RetainedLayerState)> {
    let shell = ctx.r.shell_opening.as_ref()?;
    if layer_idx >= shell.layers.len() || layer_idx >= ctx.r.retained.layers.len() {
        return None;
    }
    let sl = &shell.layers[layer_idx];
    let rs = &ctx.r.retained.layers[layer_idx];
    match (&sl.q, &sl.k, &sl.v) {
        (Some(q), Some(k), Some(v)) => Some((q, k, v, rs)),
        _ => None,
    }
}

/// Check an attention replay result and emit failure if mismatched (prefix tokens).
fn check_attention_result(
    st: &mut St,
    claimed: &[i8],
    expected: &[i8],
    label: &str,
    token_j: usize,
    layer_idx: usize,
) {
    let tolerance = verilm_core::attention::AttentionToleranceConfig::default();
    if let Some(max_diff) =
        verilm_core::attention::compare_attention_output(claimed, expected, &tolerance)
    {
        st.fail_ctx(
            FailureCode::AttentionReplayMismatch,
            format!(
                "{} {} layer {}: attention replay mismatch (max_diff={})",
                label, token_j, layer_idx, max_diff
            ),
            FailureContext {
                token_index: Some(token_j as u32),
                layer: Some(layer_idx),
                ..Default::default()
            },
        );
    }
}

/// Check an attention replay result for the opened token.
fn check_attention_result_opened(
    ctx: &Ctx,
    st: &mut St,
    claimed: &[i8],
    expected: &[i8],
    layer_idx: usize,
) {
    let tolerance = verilm_core::attention::AttentionToleranceConfig::default();
    if let Some(max_diff) =
        verilm_core::attention::compare_attention_output(claimed, expected, &tolerance)
    {
        st.fail_ctx(
            FailureCode::AttentionReplayMismatch,
            format!(
                "opened token layer {}: deep-prefix attention replay mismatch (max_diff={})",
                layer_idx, max_diff
            ),
            FailureContext {
                token_index: Some(ctx.r.token_index),
                layer: Some(layer_idx),
                ..Default::default()
            },
        );
    }
}

// --- Phase 8: Tokenization

fn phase_tokenization(ctx: &Ctx, specs: &SpecState, st: &mut St) {
    let tok = match ctx.tokenizer {
        Some(t) => t,
        None => return,
    };
    let prompt = match &ctx.r.prompt {
        Some(p) => p,
        None => return,
    };
    let input_spec = match &specs.input_spec {
        Some(is) => is,
        None => return,
    };

    match tok.tokenize(prompt, input_spec) {
        Ok(tids) => {
            st.check();
            let tok_failures = crate::verify_input_tokenization(ctx.r, &tids);
            st.failures.extend(tok_failures);
        }
        Err(e) => st.fail(
            FailureCode::TokenizerError,
            format!("tokenizer reconstruction failed: {}", e),
        ),
    }
}

// --- Phase 9: Detokenization

fn phase_detokenization(ctx: &Ctx, specs: &SpecState, st: &mut St) {
    let detok = match ctx.detokenizer {
        Some(d) => d,
        None => return,
    };

    st.check();
    let claimed = match &ctx.r.output_text {
        Some(t) => t,
        None => {
            st.fail(
                FailureCode::MissingOutputText,
                "detokenizer provided but response missing output_text",
            );
            return;
        }
    };

    let policy = specs.detokenization_policy.as_deref();

    let gen_start = ctx.gen_start as usize;
    let mut gen_tids: Vec<u32> = ctx
        .r
        .prefix_token_ids
        .get(gen_start..)
        .unwrap_or(&[])
        .to_vec();
    gen_tids.push(ctx.r.token_id);

    match detok.decode(&gen_tids, policy) {
        Ok(decoded) => {
            if ctx.is_last {
                if decoded != *claimed {
                    st.fail(
                        FailureCode::DetokenizationMismatch,
                        format!(
                            "detokenization mismatch: decoded={:?} claimed={:?}",
                            decoded, claimed
                        ),
                    );
                }
            } else if !claimed.starts_with(&decoded) {
                st.fail(
                    FailureCode::DetokenizationMismatch,
                    format!(
                        "detokenization prefix mismatch: decoded={:?} not prefix of claimed={:?}",
                        decoded, claimed
                    ),
                );
            }
        }
        Err(e) => st.fail(
            FailureCode::DetokenizerError,
            format!("detokenization failed: {}", e),
        ),
    }
}

// --- Report builder

fn finish(ctx: &Ctx, st: St, coverage: AuditCoverage) -> V4VerifyReport {
    V4VerifyReport {
        verdict: if st.failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        token_index: ctx.r.token_index,
        checks_run: st.checks,
        checks_passed: st.checks.saturating_sub(st.failures.len()),
        failures: st.failures,
        coverage,
        duration: ctx.start.elapsed(),
    }
}

// --- Small helpers

fn verify_freivalds(key: &VerifierKey, st: &mut St, layer: usize, mt: MatrixType, input: &[i8], accum: &[i32]) {
    st.check();
    if !freivalds::check(key.v_for(layer, mt), input, key.r_for(mt), accum) {
        st.fail_ctx(
            FailureCode::FreivaldsFailed,
            format!("layer {} {:?}: Freivalds failed", layer, mt),
            FailureContext {
                layer: Some(layer),
                matrix: Some(format!("{:?}", mt)),
                ..Default::default()
            },
        );
    }
}

fn xcheck_f64(st: &mut St, name: &str, manifest: Option<f64>, key: f64) {
    if let Some(v) = manifest {
        st.check();
        if (v - key).abs() > f64::EPSILON {
            st.fail_ctx(
                FailureCode::SpecFieldMismatch,
                format!("{} mismatch: manifest={} key={}", name, v, key),
                FailureContext { field: Some(name.into()), ..Default::default() },
            );
        }
    }
}
fn xcheck_hash(st: &mut St, name: &str, manifest: Option<[u8; 32]>, key: Option<[u8; 32]>) {
    if let (Some(m), Some(k)) = (manifest, key) {
        st.check();
        if m != k {
            st.fail_ctx(
                FailureCode::SpecFieldMismatch,
                format!("{} mismatch: manifest != key", name),
                FailureContext { field: Some(name.into()), ..Default::default() },
            );
        }
    }
}
fn xcheck_dim(st: &mut St, name: &str, manifest: Option<u32>, key: usize) {
    if let Some(v) = manifest {
        st.check();
        if v as usize != key {
            st.fail_ctx(
                FailureCode::SpecFieldMismatch,
                format!("{} mismatch: manifest={} key={}", name, v, key),
                FailureContext { field: Some(name.into()), ..Default::default() },
            );
        }
    }
}
fn xcheck_str(st: &mut St, name: &str, manifest: Option<&str>, key: Option<&str>) {
    if let (Some(m), Some(k)) = (manifest, key) {
        st.check();
        if m != k {
            st.fail_ctx(
                FailureCode::SpecFieldMismatch,
                format!("{} mismatch: manifest='{}' key='{}'", name, m, k),
                FailureContext { field: Some(name.into()), ..Default::default() },
            );
        }
    }
}
fn reject_feature(st: &mut St, rejected: bool, field: &str, actual: &str, expected: &str) {
    if rejected {
        st.fail_ctx(
            FailureCode::UnsupportedDecodeFeature,
            format!("unsupported {}={} (canonical requires {})", field, actual, expected),
            FailureContext { field: Some(field.into()), ..Default::default() },
        );
    }
}

// --- Unit tests for Ctx::new() precomputed fields

#[cfg(test)]
mod tests {
    use super::*;
    use verilm_core::constants::ModelConfig;
    use verilm_core::merkle::MerkleProof;
    use verilm_core::types::{BatchCommitment, RetainedTokenState};

    fn dummy_key() -> VerifierKey {
        VerifierKey {
            version: 1,
            config: ModelConfig::toy(),
            seed: [0u8; 32],
            source_dtype: "int8".into(),
            quantization_scales: vec![],
            r_vectors: vec![],
            v_vectors: vec![],
            wo_norms: vec![],
            max_v_norm: 0.0,
            lm_head: None,
            v_lm_head: None,
            weight_hash: None,
            rmsnorm_attn_weights: vec![],
            rmsnorm_ffn_weights: vec![],
            weight_scales: vec![],
            rmsnorm_eps: 1e-5,
            rope_config_hash: None,
            embedding_merkle_root: None,
            final_norm_weights: None,
            quant_family: None,
            scale_derivation: None,
            quant_block_size: None,
            rope_aware_replay: false,
        }
    }

    fn dummy_response(
        token_index: u32,
        n_tokens: u32,
        commit_n_prompt: Option<u32>,
        resp_n_prompt: Option<u32>,
    ) -> V4AuditResponse {
        V4AuditResponse {
            token_index,
            retained: RetainedTokenState { layers: vec![] },
            merkle_proof: MerkleProof { leaf_index: 0, siblings: vec![] },
            io_proof: MerkleProof { leaf_index: 0, siblings: vec![] },
            token_id: 0,
            prev_io_hash: [0u8; 32],
            prefix_leaf_hashes: vec![],
            prefix_merkle_proofs: vec![],
            prefix_token_ids: vec![],
            commitment: BatchCommitment {
                merkle_root: [0u8; 32],
                io_root: [0u8; 32],
                n_tokens,
                manifest_hash: None,
                input_spec_hash: None,
                model_spec_hash: None,
                decode_spec_hash: None,
                output_spec_hash: None,
                version: CommitmentVersion::V4,
                prompt_hash: None,
                seed_commitment: None,
                n_prompt_tokens: commit_n_prompt,
            },
            revealed_seed: [0u8; 32],
            shell_opening: None,
            manifest: None,
            prompt: None,
            n_prompt_tokens: resp_n_prompt,
            output_text: None,
            prefix_embedding_rows: None,
            prefix_embedding_proofs: None,
            prefix_retained: None,
            prefix_shell_openings: None,
        }
    }

    #[test]
    fn ctx_n_prompt_prefers_commitment() {
        let key = dummy_key();
        let r = dummy_response(0, 10, Some(5), Some(3));
        let ctx = Ctx::new(&key, &r, None, None);
        assert_eq!(ctx.n_prompt, 5);
    }

    #[test]
    fn ctx_n_prompt_falls_back_to_response() {
        let key = dummy_key();
        let r = dummy_response(0, 10, None, Some(3));
        let ctx = Ctx::new(&key, &r, None, None);
        assert_eq!(ctx.n_prompt, 3);
    }

    #[test]
    fn ctx_n_prompt_defaults_to_zero() {
        let key = dummy_key();
        let r = dummy_response(0, 10, None, None);
        let ctx = Ctx::new(&key, &r, None, None);
        assert_eq!(ctx.n_prompt, 0);
    }

    #[test]
    fn ctx_gen_start_from_n_prompt() {
        let key = dummy_key();
        let r = dummy_response(0, 10, Some(5), Some(5));
        let ctx = Ctx::new(&key, &r, None, None);
        assert_eq!(ctx.gen_start, 4); // 5 - 1
    }

    #[test]
    fn ctx_gen_start_saturates_at_zero() {
        let key = dummy_key();
        let r = dummy_response(0, 10, Some(0), Some(0));
        let ctx = Ctx::new(&key, &r, None, None);
        assert_eq!(ctx.gen_start, 0);
    }

    #[test]
    fn ctx_is_last_final_token() {
        let key = dummy_key();
        let r = dummy_response(9, 10, Some(1), Some(1));
        let ctx = Ctx::new(&key, &r, None, None);
        assert!(ctx.is_last);
    }

    #[test]
    fn ctx_is_last_not_final() {
        let key = dummy_key();
        let r = dummy_response(5, 10, Some(1), Some(1));
        let ctx = Ctx::new(&key, &r, None, None);
        assert!(!ctx.is_last);
    }

    #[test]
    fn ctx_is_last_single_token() {
        let key = dummy_key();
        let r = dummy_response(0, 1, Some(1), Some(1));
        let ctx = Ctx::new(&key, &r, None, None);
        assert!(ctx.is_last);
    }
}
