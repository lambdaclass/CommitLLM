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
//!        ├─ phase_kv_transcript    (verify KV Merkle proofs against kv_roots)
//!        ├─ phase_exact_attention  (f64 Q·K^T replay — exact-replay profiles)
//!        │    OR
//!        ├─ phase_witnessed_score_attention (GPU scores — witnessed-score profiles)
//!        ├─ phase_deep_prefix
//!        │    ├─ replay_deep_prefix_attention (prefix tokens j≥1)
//!        │    └─ replay_opened_token_layer    (opened token via prefix KV)
//!        ├─ phase_tokenization   (reads SpecState.input_spec)
//!        └─ phase_detokenization (reads SpecState.detok_policy)
//! ```

use std::time::Instant;

use verilm_core::constants::MatrixType;
use verilm_core::merkle::MerkleProof;
use verilm_core::types::{
    CommitmentVersion, DecodeArtifact, InputSpec, KvEntry, OutputSpec, RetainedLayerState,
    RetainedTokenState, ShellTokenOpening, V4AuditResponse, VerifierKey,
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
    decode_artifact: Option<&DecodeArtifact>,
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> Result<V4VerifyReport, String> {
    let response = serialize::deserialize_v4_audit(audit_bytes)?;
    Ok(verify_response(key, &response, decode_artifact, tokenizer, detokenizer))
}

/// Verify a deserialized V4 audit response.
///
/// This is the trusted path — all public entrypoints delegate here.
pub fn verify_response(
    key: &VerifierKey,
    r: &V4AuditResponse,
    decode_artifact: Option<&DecodeArtifact>,
    tokenizer: Option<&dyn PromptTokenizer>,
    detokenizer: Option<&dyn Detokenizer>,
) -> V4VerifyReport {
    run(&Ctx::new(key, r, decode_artifact, tokenizer, detokenizer))
}

// --- Orchestrator state

struct Ctx<'a> {
    key: &'a VerifierKey,
    r: &'a V4AuditResponse,
    decode_artifact: Option<&'a DecodeArtifact>,
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
        decode_artifact: Option<&'a DecodeArtifact>,
        tokenizer: Option<&'a dyn PromptTokenizer>,
        detokenizer: Option<&'a dyn Detokenizer>,
    ) -> Self {
        let n_prompt = r
            .commitment
            .n_prompt_tokens
            .or(r.n_prompt_tokens)
            .unwrap_or(0);
        Self {
            key,
            r,
            decode_artifact,
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
    skipped: Vec<String>,
    attention_status: Option<crate::AttentionStatus>,
    kv_provenance_run: Option<KvProvenanceRun>,
}

impl St {
    fn new() -> Self {
        Self {
            checks: 0,
            failures: Vec::new(),
            skipped: Vec::new(),
            attention_status: None,
            kv_provenance_run: None,
        }
    }
    fn check(&mut self) {
        self.checks += 1;
    }
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

    let kv_transcript_ok = phase_kv_transcript(ctx, &mut st);

    // Route attention verification based on profile's attention_mode.
    // ExactReplay: f64 Q·K^T replay (Llama — verifier independently recomputes).
    // WitnessedScores: GPU-captured scores + anchoring (Qwen — f64 diverges at later positions).
    match ctx.key.attention_mode() {
        verilm_core::types::AttentionVerificationMode::WitnessedScores => {
            phase_witnessed_score_attention(ctx, kv_transcript_ok, &mut st);
        }
        verilm_core::types::AttentionVerificationMode::ExactReplay => {
            phase_exact_attention(ctx, kv_transcript_ok, &mut st);
        }
        verilm_core::types::AttentionVerificationMode::AuditedInputsOnly => {
            phase_audited_inputs_only(ctx, &mut st);
        }
        verilm_core::types::AttentionVerificationMode::DeterministicKernel => {
            // Future: deterministic kernel path — not yet implemented.
            st.skipped.push("attention: deterministic-kernel mode — not yet implemented".into());
        }
    }
    phase_deep_prefix(ctx, kv_transcript_ok, &mut st);
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
            let checked = s.layer_indices.as_ref().map_or(s.layers.len(), |v| v.len());
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
                st.fail(
                    FailureCode::PromptHashMismatch,
                    "hash(prompt) != prompt_hash",
                );
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
    let lp_ref = ctx
        .r
        .shell_opening
        .as_ref()
        .and_then(|s| s.lp_hidden_bf16.as_deref());
    let cl_ref = ctx
        .r
        .shell_opening
        .as_ref()
        .and_then(|s| s.captured_logits_f32.as_deref());
    let leaf = merkle::hash_retained_with_captured_logits(&ctx.r.retained, fr_ref, lp_ref, cl_ref);
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
    let lp_ref = r
        .shell_opening
        .as_ref()
        .and_then(|s| s.lp_hidden_bf16.as_deref());
    let cl_ref = r
        .shell_opening
        .as_ref()
        .and_then(|s| s.captured_logits_f32.as_deref());
    let leaf = merkle::hash_retained_with_captured_logits(&r.retained, fr_ref, lp_ref, cl_ref);
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

    if rows.len() != ctx.r.prefix_token_ids.len() || proofs.len() != ctx.r.prefix_token_ids.len() {
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
        ("input", ctx.r.commitment.input_spec_hash, hashes[0]),
        ("model", ctx.r.commitment.model_spec_hash, hashes[1]),
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
        None => st.fail(
            FailureCode::MissingManifestHash,
            "commitment missing manifest_hash",
        ),
        Some(committed) => {
            if merkle::hash_manifest_composed(hashes[0], hashes[1], hashes[2], hashes[3])
                != committed
            {
                st.fail(FailureCode::ManifestHashMismatch, "manifest hash mismatch");
            }
        }
    }
}

fn cross_check_model_vs_key(ctx: &Ctx, spec: &verilm_core::types::ModelSpec, st: &mut St) {
    xcheck_f64(st, "rmsnorm_eps", spec.rmsnorm_eps, ctx.key.rmsnorm_eps);
    xcheck_f64(st, "rope_theta", spec.rope_theta, ctx.key.config.rope_theta);
    xcheck_hash(
        st,
        "rope_config_hash",
        spec.rope_config_hash,
        ctx.key.rope_config_hash,
    );
    xcheck_hash(st, "weight_hash", spec.weight_hash, ctx.key.weight_hash);
    xcheck_hash(
        st,
        "embedding_merkle_root",
        spec.embedding_merkle_root,
        ctx.key.embedding_merkle_root,
    );
    xcheck_dim(st, "n_layers", spec.n_layers, ctx.key.config.n_layers);
    xcheck_dim(st, "hidden_dim", spec.hidden_dim, ctx.key.config.hidden_dim);
    xcheck_dim(st, "vocab_size", spec.vocab_size, ctx.key.config.vocab_size);
    xcheck_dim(st, "kv_dim", spec.kv_dim, ctx.key.config.kv_dim);
    xcheck_dim(st, "ffn_dim", spec.ffn_dim, ctx.key.config.ffn_dim);
    xcheck_dim(st, "d_head", spec.d_head, ctx.key.config.d_head);
    xcheck_dim(st, "n_q_heads", spec.n_q_heads, ctx.key.config.n_q_heads);
    xcheck_dim(st, "n_kv_heads", spec.n_kv_heads, ctx.key.config.n_kv_heads);
    xcheck_str(
        st,
        "quant_family",
        spec.quant_family.as_deref(),
        ctx.key.quant_family.as_deref(),
    );
    xcheck_str(
        st,
        "scale_derivation",
        spec.scale_derivation.as_deref(),
        ctx.key.scale_derivation.as_deref(),
    );
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
    xcheck_str(
        st,
        "attn_backend",
        spec.attn_backend.as_deref(),
        ctx.key.attn_backend.as_deref(),
    );
    xcheck_str(
        st,
        "attn_dtype",
        spec.attn_dtype.as_deref(),
        ctx.key.attn_dtype.as_deref(),
    );

    // Fail closed: W8A8 requires a known-good attention backend.
    // - eager is incompatible (compressed_tensors decompression breaks).
    // - missing/unknown backend is rejected (fail closed).
    if let Some(qf) = spec.quant_family.as_deref() {
        if qf == "W8A8" {
            st.check();
            match spec.attn_backend.as_deref() {
                Some("sdpa") | Some("flash_attention_2") => {
                    // Known-good backends for W8A8.
                }
                Some("eager") => {
                    st.fail_ctx(
                        FailureCode::SpecFieldMismatch,
                        "attn_backend=\"eager\" is incompatible with quant_family=\"W8A8\" \
                         (produces incorrect outputs with compressed_tensors). \
                         Use \"sdpa\" instead."
                            .to_string(),
                        FailureContext {
                            field: Some("attn_backend".into()),
                            ..Default::default()
                        },
                    );
                }
                Some(other) => {
                    st.fail_ctx(
                        FailureCode::SpecFieldMismatch,
                        format!(
                            "attn_backend=\"{}\" is not a known-good backend for W8A8; \
                             only \"sdpa\" and \"flash_attention_2\" are accepted",
                            other
                        ),
                        FailureContext {
                            field: Some("attn_backend".into()),
                            ..Default::default()
                        },
                    );
                }
                None => {
                    st.fail_ctx(
                        FailureCode::SpecFieldMismatch,
                        "W8A8 requires attn_backend to be specified (fail closed); \
                         set attn_backend=\"sdpa\" in the manifest and verifier key"
                            .to_string(),
                        FailureContext {
                            field: Some("attn_backend".into()),
                            ..Default::default()
                        },
                    );
                }
            }
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
                format!(
                    "decode_mode='greedy' but temperature={}",
                    decode.temperature
                ),
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
    reject_feature(
        st,
        decode.repetition_penalty != 1.0,
        "repetition_penalty",
        &format!("{}", decode.repetition_penalty),
        "1.0",
    );
    reject_feature(
        st,
        decode.frequency_penalty != 0.0,
        "frequency_penalty",
        &format!("{}", decode.frequency_penalty),
        "0.0",
    );
    reject_feature(
        st,
        decode.presence_penalty != 0.0,
        "presence_penalty",
        &format!("{}", decode.presence_penalty),
        "0.0",
    );
    reject_feature(
        st,
        !decode.logit_bias.is_empty(),
        "logit_bias",
        &format!("{} entries", decode.logit_bias.len()),
        "empty",
    );
    reject_feature(
        st,
        !decode.bad_word_ids.is_empty(),
        "bad_word_ids",
        &format!("{} entries", decode.bad_word_ids.len()),
        "empty",
    );
    reject_feature(
        st,
        !decode.guided_decoding.is_empty(),
        "guided_decoding",
        &decode.guided_decoding,
        "empty",
    );
    reject_feature(
        st,
        !output.stop_sequences.is_empty(),
        "stop_sequences",
        &format!("{} entries", output.stop_sequences.len()),
        "empty",
    );
}

// --- Phase 4: Output policy

fn phase_output_policy(ctx: &Ctx, os: &OutputSpec, st: &mut St) {
    let r = ctx.r;
    if os.max_tokens > 0 {
        let n_generated = r
            .commitment
            .n_tokens
            .saturating_sub(ctx.n_prompt.saturating_sub(1));
        if r.token_index >= r.commitment.n_tokens {
            st.fail(
                FailureCode::ExceedsMaxTokens,
                format!(
                    "token_index {} >= n_tokens {}",
                    r.token_index, r.commitment.n_tokens
                ),
            );
        }
        if n_generated > os.max_tokens {
            st.fail(
                FailureCode::ExceedsMaxTokens,
                format!(
                    "generated {} tokens > max_tokens {}",
                    n_generated, os.max_tokens
                ),
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
            format!(
                "only {} generated tokens, min_tokens={}",
                n_generated, os.min_tokens
            ),
        );
    }
    let gen_index = r.token_index.saturating_sub(ctx.gen_start);
    if gen_index < os.min_tokens && r.token_id == eos_id {
        st.fail(
            FailureCode::MinTokensViolated,
            format!(
                "EOS at generation position {} < min_tokens={}",
                gen_index, os.min_tokens
            ),
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
    if bridge_replay_key_shapes_compatible(ctx, shell, st) {
        bridge_layers(ctx.key, &ctx.r.retained, shell, st);
    }

    // Derive final_hidden from captured final_residual + final_norm.
    // No shell-replay fallback in canonical path.
    let final_hidden: Option<Vec<i8>> = match (&shell.final_residual, &ctx.key.final_norm_weights) {
        (Some(fr), Some(fnw)) => {
            st.check();
            if fnw.len() != fr.len() {
                st.fail_ctx(
                    FailureCode::SpecFieldMismatch,
                    format!(
                        "final_norm_weights len {} != final_residual len {}",
                        fnw.len(),
                        fr.len()
                    ),
                    FailureContext {
                        field: Some("final_norm_weights".into()),
                        ..Default::default()
                    },
                );
                return BridgeState { final_hidden: None };
            }
            let res_f64: Vec<f64> = fr.iter().map(|&v| v as f64).collect();
            let normed =
                verilm_core::rmsnorm::rmsnorm_f64_input(&res_f64, fnw, ctx.key.rmsnorm_eps);
            Some(
                normed
                    .iter()
                    .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
                    .collect(),
            )
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

fn bridge_replay_key_shapes_compatible(ctx: &Ctx, shell: &ShellTokenOpening, st: &mut St) -> bool {
    let ir = match &shell.initial_residual {
        Some(ir) => ir,
        None => return false,
    };
    if ctx.key.rmsnorm_attn_weights.is_empty()
        || ctx.key.rmsnorm_ffn_weights.is_empty()
        || ctx.r.retained.layers.is_empty()
    {
        return false;
    }

    let hidden_dim = ir.len();
    let n_layers = shell.layers.len().min(ctx.r.retained.layers.len());
    let mut ok = true;

    st.check();
    if hidden_dim != ctx.key.config.hidden_dim {
        st.fail_ctx(
            FailureCode::SpecFieldMismatch,
            format!(
                "initial_residual len {} != key hidden_dim {}",
                hidden_dim,
                ctx.key.config.hidden_dim
            ),
            FailureContext {
                field: Some("hidden_dim".into()),
                ..Default::default()
            },
        );
        ok = false;
    }

    st.check();
    if ctx.key.rmsnorm_attn_weights.len() < n_layers {
        st.fail_ctx(
            FailureCode::SpecFieldMismatch,
            format!(
                "key has {} attention RMSNorm vectors but bridge replay needs {}",
                ctx.key.rmsnorm_attn_weights.len(),
                n_layers
            ),
            FailureContext {
                field: Some("rmsnorm_attn_weights".into()),
                ..Default::default()
            },
        );
        ok = false;
    }

    st.check();
    if ctx.key.rmsnorm_ffn_weights.len() < n_layers {
        st.fail_ctx(
            FailureCode::SpecFieldMismatch,
            format!(
                "key has {} FFN RMSNorm vectors but bridge replay needs {}",
                ctx.key.rmsnorm_ffn_weights.len(),
                n_layers
            ),
            FailureContext {
                field: Some("rmsnorm_ffn_weights".into()),
                ..Default::default()
            },
        );
        ok = false;
    }

    for layer_idx in 0..n_layers {
        st.check();
        if let Some(weights) = ctx.key.rmsnorm_attn_weights.get(layer_idx) {
            if weights.len() != hidden_dim {
                st.fail_ctx(
                    FailureCode::SpecFieldMismatch,
                    format!(
                        "layer {} attention RMSNorm len {} != residual len {}",
                        layer_idx,
                        weights.len(),
                        hidden_dim
                    ),
                    FailureContext {
                        layer: Some(layer_idx),
                        field: Some("rmsnorm_attn_weights".into()),
                        ..Default::default()
                    },
                );
                ok = false;
            }
        }

        st.check();
        if let Some(weights) = ctx.key.rmsnorm_ffn_weights.get(layer_idx) {
            if weights.len() != hidden_dim {
                st.fail_ctx(
                    FailureCode::SpecFieldMismatch,
                    format!(
                        "layer {} FFN RMSNorm len {} != residual len {}",
                        layer_idx,
                        weights.len(),
                        hidden_dim
                    ),
                    FailureContext {
                        layer: Some(layer_idx),
                        field: Some("rmsnorm_ffn_weights".into()),
                        ..Default::default()
                    },
                );
                ok = false;
            }
        }
    }

    ok
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

    let (mut residual, derived_x_attn) = init_residual_chain(ir, key, shell.layers[0].scale_x_attn);

    // Check-and-gate layer 0: compare committed x_attn against canonical derivation.
    let mut x_attn = check_and_gate_x_attn(
        st,
        key,
        0,
        &derived_x_attn,
        &retained.layers[0],
        shell.layers[0].scale_x_attn,
    );

    let qkv_freivalds = key
        .verification_profile
        .as_ref()
        .map_or(true, |p| p.supports_qkv_freivalds);
    if !qkv_freivalds {
        let profile_name = key
            .verification_profile
            .as_ref()
            .map_or("unknown", |p| &p.name);
        st.skipped.push(format!(
            "Wq/Wk/Wv Freivalds: unsupported for profile '{}' \
             (bridge replay cannot match GPU quantized GEMM)",
            profile_name
        ));
    }

    let n_layers = shell.layers.len().min(retained.layers.len());
    for layer_idx in 0..n_layers {
        let rs = &retained.layers[layer_idx];
        let sl = &shell.layers[layer_idx];

        // QKV Freivalds uses the gated x_attn (committed when available).
        // Skip when the profile says bridge replay can't match GPU GEMM.
        if qkv_freivalds {
            check_qkv(key, st, layer_idx, sl, &x_attn);
        }
        // Downstream exact checks continue from committed `rs.a`.
        check_wo(key, st, layer_idx, &rs.a, &sl.attn_out);
        let x_ffn = bridge_attn_to_ffn(key, layer_idx, rs, sl, &sl.attn_out, &mut residual);
        check_ffn(key, st, layer_idx, sl, &x_ffn);
        let next_scale = shell
            .layers
            .get(layer_idx + 1)
            .map(|s| s.scale_x_attn)
            .unwrap_or(1.0);
        let derived_next =
            bridge_ffn_to_next(key, layer_idx, sl, &sl.ffn_out, &mut residual, next_scale);
        // Check-and-gate at layer boundary: compare committed vs derived.
        if layer_idx + 1 < retained.layers.len() {
            x_attn = check_and_gate_x_attn(
                st,
                key,
                layer_idx + 1,
                &derived_next,
                &retained.layers[layer_idx + 1],
                next_scale,
            );
        } else {
            x_attn = derived_next;
        }
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
    // Use per-channel weight scales (W8A8) when available, else per-tensor.
    let q_f64 = dequant_acc(key, layer_idx, MatrixType::Wq, q_acc, scale_x_attn);
    let k_f64 = dequant_acc(key, layer_idx, MatrixType::Wk, k_acc, scale_x_attn);
    let v_f64 = dequant_acc(key, layer_idx, MatrixType::Wv, v_acc, scale_x_attn);

    // Add projection biases (model-dependent, e.g. Qwen2)
    let q_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wq, q_f64);
    let k_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wk, k_f64);
    let v_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wv, v_f64);

    let q_roped = verilm_core::rope::apply_rope_q(&q_f64, position, &key.config);
    let k_roped = verilm_core::rope::apply_rope_k(&k_f64, position, &key.config);

    (q_roped, k_roped, v_f64)
}

/// Dequantize Q accumulators and apply RoPE. Used when K/V comes from
/// committed KV transcript and only Q needs reconstruction.
fn dequant_rope_q(
    key: &VerifierKey,
    layer_idx: usize,
    q_acc: &[i32],
    scale_x_attn: f32,
    position: usize,
) -> Vec<f64> {
    let q_f64 = dequant_acc(key, layer_idx, MatrixType::Wq, q_acc, scale_x_attn);
    let q_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wq, q_f64);
    verilm_core::rope::apply_rope_q(&q_f64, position, &key.config)
}

/// Add QKV projection bias if the model has one for this matrix type.
fn add_qkv_bias(key: &VerifierKey, layer_idx: usize, mt: MatrixType, mut v: Vec<f64>) -> Vec<f64> {
    if let Some(bias) = key.qkv_bias_for(layer_idx, mt) {
        for (x, &b) in v.iter_mut().zip(bias) {
            *x += b as f64;
        }
    }
    v
}

/// Dequantize i32 accumulators using per-channel or per-tensor weight scales.
fn dequant_acc(
    key: &VerifierKey,
    layer_idx: usize,
    mt: MatrixType,
    acc: &[i32],
    scale_x: f32,
) -> Vec<f64> {
    if let Some(pc_scales) = key.per_channel_scales_for(layer_idx, mt) {
        verilm_core::rope::dequantize_acc_per_channel(acc, pc_scales, scale_x)
    } else {
        let scale_w = key.weight_scale_for(layer_idx, mt);
        verilm_core::rope::dequantize_acc(acc, Some(scale_w), Some(scale_x))
    }
}

/// Dequantize Q/K accumulators and apply RoPE in f32.
///
/// Full path matching witness capture.py:
/// 1. `cutlass_epilogue_bf16()` — exact CUTLASS epilogue with FMA for bias path
/// 2. RoPE in f32 (no bf16 truncation — matches witness score computation)
///
/// Used for score anchoring on models that run in bfloat16 (e.g. Qwen, Llama).
fn dequant_rope_f32(
    key: &VerifierKey,
    layer_idx: usize,
    mt: MatrixType,
    acc: &[i32],
    scale_x: f32,
    n_heads: usize,
    position: usize,
) -> Vec<f32> {
    // Get per-channel weight scales.
    let pc_scales = key
        .per_channel_scales_for(layer_idx, mt)
        .expect("bf16 anchoring requires per-channel weight scales");
    let bias = key.qkv_bias_for(layer_idx, mt);

    // Deterministic epilogue matching CUTLASS exactly (FMA for bias path).
    let k_f32 =
        verilm_core::attention::cutlass_epilogue_bf16(acc, pc_scales, scale_x, bias);

    // RoPE in f32 — no bf16 truncation. Matches witness capture.py which promotes
    // to f32, applies RoPE, and computes scores all in f32.
    verilm_core::attention::apply_rope_f32(&k_f32, n_heads, position, &key.config)
}

/// Per-step tolerance for bridge x_attn check-and-gate.
///
/// Delegates to `VerifierKey::bridge_tolerance()` which reads from the
/// verification profile when present, or falls back to toy=0, production=1.
fn bridge_x_attn_tolerance(key: &VerifierKey) -> u8 {
    key.bridge_tolerance()
}

/// Attention replay tolerance derived from the verifier key profile.
fn attention_tolerance(key: &VerifierKey) -> verilm_core::attention::AttentionToleranceConfig {
    verilm_core::attention::AttentionToleranceConfig {
        max_abs_diff: key.attention_tolerance(),
    }
}

/// Check-and-gate: compare committed x_attn against canonical derivation.
///
/// If committed value is within tolerance, returns it for downstream use.
/// If outside tolerance, records failure but still returns committed value
/// (downstream checks continue from committed to give a clean error trail).
/// If no committed value, returns derived value (backward compat).
fn check_and_gate_x_attn(
    st: &mut St,
    key: &VerifierKey,
    layer_idx: usize,
    derived_x_attn: &[i8],
    committed: &RetainedLayerState,
    shell_scale: f32,
) -> Vec<i8> {
    match (&committed.x_attn_i8, committed.scale_x_attn) {
        (Some(committed_xa), Some(committed_scale)) => {
            // Scale consistency: committed scale must match shell-provided scale.
            if (committed_scale - shell_scale).abs() > f32::EPSILON {
                st.fail_ctx(
                    FailureCode::BridgeScaleMismatch,
                    format!(
                        "layer {}: committed scale_x_attn ({}) != shell scale ({})",
                        layer_idx, committed_scale, shell_scale
                    ),
                    FailureContext {
                        layer: Some(layer_idx),
                        ..Default::default()
                    },
                );
            }
            // L-inf comparison within tolerance.
            let tolerance = bridge_x_attn_tolerance(key);
            if committed_xa.len() != derived_x_attn.len() {
                st.fail_ctx(
                    FailureCode::BridgeXAttnMismatch,
                    format!(
                        "layer {}: committed x_attn len {} != derived len {}",
                        layer_idx,
                        committed_xa.len(),
                        derived_x_attn.len()
                    ),
                    FailureContext {
                        layer: Some(layer_idx),
                        ..Default::default()
                    },
                );
            } else {
                let max_diff: i16 = committed_xa
                    .iter()
                    .zip(derived_x_attn.iter())
                    .map(|(&a, &b)| (a as i16 - b as i16).abs())
                    .max()
                    .unwrap_or(0);
                if max_diff > tolerance as i16 {
                    st.fail_ctx(
                        FailureCode::BridgeXAttnMismatch,
                        format!(
                            "layer {}: committed x_attn vs derived L-inf={} > tolerance={}",
                            layer_idx, max_diff, tolerance
                        ),
                        FailureContext {
                            layer: Some(layer_idx),
                            ..Default::default()
                        },
                    );
                }
            }
            // GATE: always continue from committed value.
            committed_xa.clone()
        }
        _ => {
            // No committed x_attn — fall back to derived (backward compat).
            derived_x_attn.to_vec()
        }
    }
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
                format!(
                    "layer {} {:?}: QKV required in canonical path",
                    layer_idx, mt
                ),
                FailureContext {
                    layer: Some(layer_idx),
                    matrix: Some(format!("{:?}", mt)),
                    ..Default::default()
                },
            ),
        }
    }
}

fn check_wo(key: &VerifierKey, st: &mut St, layer_idx: usize, a: &[i8], attn_out: &[i32]) {
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
    bridge_dequant_rmsnorm(
        key,
        layer_idx,
        MatrixType::Wo,
        attn_out,
        rs.scale_a,
        residual,
        &key.rmsnorm_ffn_weights[layer_idx],
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
    let h = match (
        key.per_channel_scales_for(layer_idx, MatrixType::Wg),
        key.per_channel_scales_for(layer_idx, MatrixType::Wu),
    ) {
        (Some(pc_g), Some(pc_u)) => verilm_core::silu::compute_h_per_channel(
            &sl.g,
            &sl.u,
            pc_g,
            pc_u,
            sl.scale_x_ffn,
            sl.scale_h,
        ),
        _ => verilm_core::silu::compute_h_scaled(
            &sl.g,
            &sl.u,
            key.weight_scale_for(layer_idx, MatrixType::Wg),
            key.weight_scale_for(layer_idx, MatrixType::Wu),
            sl.scale_x_ffn,
            sl.scale_h,
        ),
    };
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
        bridge_dequant_rmsnorm(
            key,
            layer_idx,
            MatrixType::Wd,
            ffn_out,
            sl.scale_h,
            residual,
            &key.rmsnorm_attn_weights[layer_idx + 1],
            next_scale,
        )
    } else {
        // Last layer: update residual, no next-layer RMSNorm
        dequant_add_residual_dispatch(
            key,
            layer_idx,
            MatrixType::Wd,
            ffn_out,
            sl.scale_h,
            residual,
        );
        Vec::new()
    }
}

/// Bridge helper: dequant + residual += + RMSNorm + quantize.
/// Dispatches to per-channel or per-tensor dequant based on VerifierKey.
fn bridge_dequant_rmsnorm(
    key: &VerifierKey,
    layer_idx: usize,
    mt: MatrixType,
    acc: &[i32],
    scale_x: f32,
    residual: &mut Vec<f64>,
    rmsnorm_weights: &[f32],
    scale_next: f32,
) -> Vec<i8> {
    dequant_add_residual_dispatch(key, layer_idx, mt, acc, scale_x, residual);
    let normed =
        verilm_core::rmsnorm::rmsnorm_f64_input(residual, rmsnorm_weights, key.rmsnorm_eps);
    verilm_core::rmsnorm::quantize_f64_to_i8(&normed, scale_next as f64)
}

/// Dispatch dequant + residual add to per-channel or per-tensor path.
fn dequant_add_residual_dispatch(
    key: &VerifierKey,
    layer_idx: usize,
    mt: MatrixType,
    acc: &[i32],
    scale_x: f32,
    residual: &mut Vec<f64>,
) {
    if let Some(pc_scales) = key.per_channel_scales_for(layer_idx, mt) {
        verilm_core::rmsnorm::dequant_add_residual_per_channel(acc, pc_scales, scale_x, residual);
    } else {
        let scale_w = key.weight_scale_for(layer_idx, mt);
        verilm_core::rmsnorm::dequant_add_residual(acc, scale_w, scale_x, residual);
    }
}

// --- Phase 6: LM-head + token replay

fn phase_lm_head(
    ctx: &Ctx,
    shell: &ShellTokenOpening,
    bridge: &BridgeState,
    specs: &SpecState,
    st: &mut St,
) {
    use verilm_core::types::DecodeAcceptanceMode;

    let mode = ctx
        .key
        .verification_profile
        .as_ref()
        .map(|p| &p.decode_acceptance)
        .unwrap_or(&DecodeAcceptanceMode::ExactTokenIdentity);

    // CapturedLogits path: exact sampling from GPU logits + Freivalds binding.
    if matches!(mode, DecodeAcceptanceMode::CapturedLogits) {
        phase_lm_head_captured_logits(ctx, shell, specs, st);
        return;
    }

    // LP hidden bf16 path: verify token identity via bf16 lm_head matmul
    // from the captured decode boundary. No Freivalds — direct matmul.
    if matches!(mode, DecodeAcceptanceMode::LpHiddenBf16) {
        phase_lm_head_lp_hidden(ctx, shell, specs, st);
        return;
    }

    // Legacy i8→i32 path: Freivalds + optional token identity.
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

            // Token-identity check (generated tokens only).
            if ctx.key.config.vocab_size > 0 && ctx.r.token_index >= ctx.gen_start {
                match mode {
                    DecodeAcceptanceMode::Unsupported => {
                        let profile_name = ctx
                            .key
                            .verification_profile
                            .as_ref()
                            .map_or("unknown", |p| &p.name);
                        st.skipped.push(format!(
                            "lm_head token identity: unsupported for profile '{}' \
                             (quantized logits diverge from GPU bf16 path)",
                            profile_name
                        ));
                    }
                    DecodeAcceptanceMode::ExactTokenIdentity => {
                        st.check();
                        let logits_f32: Vec<f32> = logits.iter().map(|&v| v as f32).collect();
                        let expected = if let Some(ref dp) = specs.decode_params {
                            // Prover uses generation-local index (0-based from first
                            // generated token). The trace skips BOS, so gen_start =
                            // n_prompt - 1 is the first generated position in the trace.
                            let gen_index = ctx.r.token_index.saturating_sub(ctx.gen_start);
                            let seed = verilm_core::sampling::derive_token_seed(
                                &ctx.r.revealed_seed,
                                gen_index,
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
                                format!(
                                    "lm_head: expected token {} but got {}",
                                    expected, ctx.r.token_id
                                ),
                            );
                        }
                    }
                    DecodeAcceptanceMode::LpHiddenBf16 => unreachable!(),
                    DecodeAcceptanceMode::CapturedLogits => unreachable!(),
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

/// CapturedLogits decode verification.
///
/// The prover captures the actual f32 logits from the GPU's LogitsProcessor
/// and commits them in the Merkle leaf. The verifier:
///   1. Samples from the captured logits (exact — same f32 values the GPU used).
///   2. Freivalds-binds the captured logits to `lp_hidden × lm_head_bf16` using
///      a secret ±1 random projection (two dot products, not a full matmul).
///
/// This solves both correctness (exact sampling) and cost (no matmul replay).
fn phase_lm_head_captured_logits(
    ctx: &Ctx,
    shell: &ShellTokenOpening,
    specs: &SpecState,
    st: &mut St,
) {
    // 1. Require captured logits.
    let captured_logits = match &shell.captured_logits_f32 {
        Some(cl) => cl,
        None => {
            st.fail(
                FailureCode::MissingLogits,
                "lm_head (captured logits): profile requires captured_logits_f32 but shell missing it",
            );
            return;
        }
    };

    // 2. Require LP hidden (needed for Freivalds binding).
    let lp_hidden = match &shell.lp_hidden_bf16 {
        Some(lp) => lp,
        None => {
            st.fail(
                FailureCode::MissingFinalHidden,
                "lm_head (captured logits): profile requires lp_hidden_bf16 for Freivalds binding",
            );
            return;
        }
    };

    // 3. Validate dimensions.
    let vocab_size = ctx.key.config.vocab_size;
    let hidden_dim = ctx.key.config.hidden_dim;

    if captured_logits.len() != vocab_size {
        st.fail(
            FailureCode::MissingLogits,
            format!(
                "lm_head (captured logits): expected vocab_size={}, got {}",
                vocab_size,
                captured_logits.len()
            ),
        );
        return;
    }

    if lp_hidden.len() != hidden_dim {
        st.fail(
            FailureCode::MissingFinalHidden,
            format!(
                "lm_head (captured logits): expected hidden_dim={}, got {}",
                hidden_dim,
                lp_hidden.len()
            ),
        );
        return;
    }

    // 4. Token identity check from captured logits (exact — these are the real GPU logits).
    if vocab_size > 0 && ctx.r.token_index >= ctx.gen_start {
        st.check();
        let expected = if let Some(ref dp) = specs.decode_params {
            let gen_index = ctx.r.token_index.saturating_sub(ctx.gen_start);
            let seed = verilm_core::sampling::derive_token_seed(
                &ctx.r.revealed_seed,
                gen_index,
            );
            verilm_core::sampling::sample(captured_logits, dp, &seed)
        } else {
            captured_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        if expected != ctx.r.token_id {
            st.fail(
                FailureCode::TokenSelectionMismatch,
                format!(
                    "lm_head (captured logits): expected token {} but got {}",
                    expected, ctx.r.token_id
                ),
            );
        }
    }

    // 5. Freivalds binding: captured_logits ≈ lp_hidden × lm_head_bf16.
    let freivalds_seed = match ctx.key.captured_logits_freivalds_seed {
        Some(seed) => seed,
        None => {
            st.skipped.push(
                "lm_head (captured logits): no Freivalds seed in key, skipping binding check"
                    .to_string(),
            );
            return;
        }
    };
    let v_precomputed = match &ctx.key.v_lm_head_f64 {
        Some(v) => v,
        None => {
            st.skipped.push(
                "lm_head (captured logits): no v_lm_head_f64 in key, skipping binding check"
                    .to_string(),
            );
            return;
        }
    };

    st.check();
    let r = verilm_core::freivalds::derive_pm1_vector(&freivalds_seed, vocab_size);
    // Tolerance: GPU tensor-core accumulation order differs from f64 reference.
    // Empirically, the gap for Llama 8B / Qwen 7B is in the range of
    // ~1e3–1e4 (vocab_size × hidden_dim products accumulated differently).
    // We use a generous tolerance for now; tightened after empirical measurement.
    let tolerance = 1e6_f64;
    let (pass, lhs, rhs) = verilm_core::freivalds::check_captured_logits(
        &r,
        v_precomputed,
        captured_logits,
        lp_hidden,
        tolerance,
    );
    if !pass {
        st.fail(
            FailureCode::LmHeadFreivaldsFailed,
            format!(
                "lm_head (captured logits): Freivalds binding failed. lhs={:.6}, rhs={:.6}, diff={:.6}",
                lhs, rhs, (lhs - rhs).abs()
            ),
        );
    }
}

/// LP hidden bf16 decode verification.
///
/// Reconstructs logits from the captured LP hidden state via bf16 lm_head matmul,
/// then verifies token identity (argmax for greedy, canonical sample for sampled).
///
/// The LP hidden is the exact tensor that LogitsProcessor received on the GPU.
/// The bf16 matmul on CPU reproduces GPU logits exactly (validated 192/192 greedy,
/// 32/32 sampled on Qwen W8A8).
fn phase_lm_head_lp_hidden(
    ctx: &Ctx,
    shell: &ShellTokenOpening,
    specs: &SpecState,
    st: &mut St,
) {
    let lp_hidden = match &shell.lp_hidden_bf16 {
        Some(lp) => lp,
        None => {
            st.fail(
                FailureCode::MissingFinalHidden,
                "lm_head (LP hidden): profile requires lp_hidden_bf16 but shell missing it",
            );
            return;
        }
    };

    let artifact = match ctx.decode_artifact {
        Some(a) => a,
        None => {
            st.fail(
                FailureCode::MissingLogits,
                "lm_head (LP hidden): decode artifact not provided for LpHiddenBf16 mode",
            );
            return;
        }
    };

    // Validate artifact hash matches key commitment.
    st.check();
    if let Some(expected_hash) = ctx.key.lm_head_bf16_hash {
        let actual_hash = artifact.content_hash();
        if actual_hash != expected_hash {
            st.fail(
                FailureCode::DecodeArtifactHashMismatch,
                "lm_head (LP hidden): decode artifact hash does not match key's lm_head_bf16_hash",
            );
            return;
        }
    }

    let lm_head_bf16 = &artifact.lm_head_bf16;
    let vocab_size = artifact.vocab_size;
    let hidden_dim = artifact.hidden_dim;

    if lp_hidden.len() != hidden_dim {
        st.fail(
            FailureCode::MissingFinalHidden,
            format!(
                "lm_head (LP hidden): expected hidden_dim={}, got {}",
                hidden_dim,
                lp_hidden.len()
            ),
        );
        return;
    }

    if lm_head_bf16.len() != vocab_size * hidden_dim {
        st.fail(
            FailureCode::MissingLogits,
            format!(
                "lm_head (LP hidden): lm_head_bf16 size mismatch: expected {}, got {}",
                vocab_size * hidden_dim,
                lm_head_bf16.len()
            ),
        );
        return;
    }

    // Compute logits via bf16 matmul: lm_head_bf16 @ lp_hidden_bf16.
    // Each row of lm_head is one vocabulary entry.
    // Accumulate in f32: GPU tensor cores accumulate in f32 within tiles,
    // not bf16. A bf16 accumulator diverges and flips argmax at ~3/386 positions.
    st.check();
    let logits_f32: Vec<f32> = (0..vocab_size)
        .map(|v| {
            let row_start = v * hidden_dim;
            let mut acc = 0.0_f32;
            for j in 0..hidden_dim {
                let w = half::bf16::from_bits(lm_head_bf16[row_start + j]);
                let h = half::bf16::from_bits(lp_hidden[j]);
                acc += w.to_f32() * h.to_f32();
            }
            acc
        })
        .collect();

    // Token identity check (generated tokens only).
    if ctx.key.config.vocab_size > 0 && ctx.r.token_index >= ctx.gen_start {
        st.check();
        let expected = if let Some(ref dp) = specs.decode_params {
            // Prover uses generation-local index (0-based from first
            // generated token). The trace skips BOS, so gen_start =
            // n_prompt - 1 is the first generated position in the trace.
            let gen_index = ctx.r.token_index.saturating_sub(ctx.gen_start);
            let seed = verilm_core::sampling::derive_token_seed(
                &ctx.r.revealed_seed,
                gen_index,
            );
            verilm_core::sampling::sample(&logits_f32, dp, &seed)
        } else {
            // Greedy: argmax.
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
                format!(
                    "lm_head (LP hidden bf16): expected token {} but got {}",
                    expected, ctx.r.token_id
                ),
            );
        }
    }
}

// --- Score-anchor helpers
//
// Score anchoring compares GPU-captured pre-softmax scores against the
// verifier's canonical `Q·K^T/√d` recomputation from opened Q accumulators
// and committed K entries. It is the strongest cheap attention-input audit
// we run: it pins scores, exercises Q/K dequantization, RoPE, and GQA head
// mapping, and rejects tampered witnessed scores.
//
// Shared between two callers:
//   - `phase_witnessed_score_attention` (witnessed-score verification mode —
//     anchor is step 2 of a larger pipeline that also replays softmax @ V).
//   - `phase_audited_inputs_only` (audit-only mode — anchor is the attention
//     input audit; no full attention output verification is performed).

/// Aggregate result of score-anchor audit across all challenged layers.
#[derive(Debug, Clone)]
pub(crate) struct ScoreAnchorResult {
    /// Number of layers successfully anchored (passed structural checks).
    pub n_layers_checked: usize,
    /// Maximum anchor gap across all checked layers, or `None` if zero
    /// layers were checked.
    pub max_gap: Option<f64>,
}

/// Anchor witnessed scores against canonical `Q·K^T/√d` for one layer.
///
/// Chooses f32 (CUTLASS-matching) or f64 anchoring pipeline based on whether
/// the verifier key has per-channel weight scales for this layer's `Wq`.
/// Shape and KV coverage must be validated by the caller.
fn anchor_scores_for_layer(
    key: &VerifierKey,
    layer_idx: usize,
    token_index: usize,
    q_acc: &[i32],
    scale_x_attn: f32,
    layer_kv: &[verilm_core::types::KvEntry],
    ws: &verilm_core::types::WitnessedScores,
) -> verilm_core::attention::ScoreAnchorStats {
    let cfg = &key.config;
    let absolute_pos = token_index + 1;
    let has_per_channel = key.per_channel_scales_for(layer_idx, MatrixType::Wq).is_some();

    if has_per_channel {
        let q_roped_f32 = dequant_rope_f32(
            key,
            layer_idx,
            MatrixType::Wq,
            q_acc,
            scale_x_attn,
            cfg.n_q_heads,
            token_index,
        );
        let kv_k_f32: Vec<Vec<f32>> = layer_kv[..=token_index]
            .iter()
            .map(|e| e.k_roped.iter().map(|&x| x as f32).collect())
            .collect();
        let canonical_f32 =
            verilm_core::attention::compute_canonical_scores_gpu_like(&q_roped_f32, &kv_k_f32, cfg);
        let canonical_f64: Vec<f64> = canonical_f32.iter().map(|&x| x as f64).collect();
        verilm_core::attention::anchor_witnessed_scores(
            &ws.scores,
            &canonical_f64,
            ws.n_q_heads,
            ws.seq_len,
            layer_idx,
        )
    } else {
        let q_roped = dequant_rope_q(key, layer_idx, q_acc, scale_x_attn, absolute_pos);
        let kv_k: Vec<Vec<f64>> = layer_kv[..=token_index]
            .iter()
            .map(|e| e.k_roped.clone())
            .collect();
        let canonical_scores =
            verilm_core::attention::compute_canonical_scores(&q_roped, &kv_k, cfg);
        verilm_core::attention::anchor_witnessed_scores(
            &ws.scores,
            &canonical_scores,
            ws.n_q_heads,
            ws.seq_len,
            layer_idx,
        )
    }
}

/// Run score anchoring across all challenged layers.
///
/// Returns `None` when the top-level prerequisites are absent (no shell
/// opening, no KV entries, no witnessed scores, no profile threshold).
/// Callers decide whether that is a hard fail (witnessed-score profiles
/// require scores) or a soft skip (audit-only profiles report `score_anchor
/// = None`).
///
/// Per-layer structural issues (incomplete KV coverage, shape mismatch) are
/// hard-failed via `st` — those represent bad provider data in either mode.
///
/// Per-layer threshold breaches are hard-failed via `st` with
/// `ScoreAnchorMismatch` **only** in witnessed-score verification mode.
/// In `AuditedInputsOnly` mode, `max_gap` is evidence, not a verification
/// gate — we populate `score_anchor.max_gap` in the audit status and let
/// the reader reason about it; the report must not fail on threshold.
fn run_score_anchor_audit(ctx: &Ctx, st: &mut St) -> Option<ScoreAnchorResult> {
    let kv_entries = ctx.r.kv_entries.as_ref()?;
    let shell = ctx.r.shell_opening.as_ref()?;
    let witnessed_scores = ctx.r.witnessed_scores.as_ref()?;
    let threshold = ctx.key.score_anchor_threshold()?;
    let is_audit_only = matches!(
        ctx.key.attention_mode(),
        verilm_core::types::AttentionVerificationMode::AuditedInputsOnly,
    );

    let cfg = &ctx.key.config;
    let token_index = ctx.r.token_index as usize;
    let n_layers = cfg
        .n_layers
        .min(shell.layers.len())
        .min(kv_entries.len())
        .min(witnessed_scores.len());

    let mut n_layers_checked = 0usize;
    let mut overall_max: Option<f64> = None;

    for layer_idx in 0..n_layers {
        let sl = &shell.layers[layer_idx];
        let ws = &witnessed_scores[layer_idx];
        let q_acc = match &sl.q {
            Some(q) => q,
            None => continue,
        };

        let layer_kv = &kv_entries[layer_idx];
        if layer_kv.len() < token_index + 1 {
            st.check();
            st.fail_ctx(
                FailureCode::AttentionKvCoverageIncomplete,
                format!(
                    "layer {}: KV entries cover {} positions, need {}",
                    layer_idx,
                    layer_kv.len(),
                    token_index + 1,
                ),
                FailureContext {
                    layer: Some(layer_idx),
                    token_index: Some(ctx.r.token_index),
                    ..Default::default()
                },
            );
            continue;
        }

        let seq_len = token_index + 1;
        st.check();
        if ws.n_q_heads != cfg.n_q_heads || ws.seq_len != seq_len {
            st.fail_ctx(
                FailureCode::WitnessedScoreStructuralError,
                format!(
                    "layer {}: witnessed scores shape ({} heads, {} seq) \
                     doesn't match expected ({} heads, {} seq)",
                    layer_idx, ws.n_q_heads, ws.seq_len, cfg.n_q_heads, seq_len,
                ),
                FailureContext {
                    layer: Some(layer_idx),
                    token_index: Some(ctx.r.token_index),
                    ..Default::default()
                },
            );
            continue;
        }

        let stats = anchor_scores_for_layer(
            ctx.key,
            layer_idx,
            token_index,
            q_acc,
            sl.scale_x_attn,
            layer_kv,
            ws,
        );

        n_layers_checked += 1;
        overall_max = Some(match overall_max {
            Some(m) => m.max(stats.max_gap),
            None => stats.max_gap,
        });

        if !is_audit_only {
            st.check();
            if stats.max_gap > threshold as f64 {
                st.fail_ctx(
                    FailureCode::ScoreAnchorMismatch,
                    format!(
                        "layer {}: witnessed score anchor gap {:.4} exceeds threshold {:.1}",
                        layer_idx, stats.max_gap, threshold,
                    ),
                    FailureContext {
                        layer: Some(layer_idx),
                        token_index: Some(ctx.r.token_index),
                        ..Default::default()
                    },
                );
            }
        }
    }

    Some(ScoreAnchorResult {
        n_layers_checked,
        max_gap: overall_max,
    })
}

// --- Phase 7a-ws: Witnessed-score attention verification
//
// Strong-tier attention path for profiles where f64 Q·K^T replay diverges
// from GPU bf16 (e.g. Qwen W8A8). Instead of replaying Q·K^T/√d, uses
// GPU-captured pre-softmax scores anchored against canonical reconstruction.
//
// Pipeline:
//   1. Verify witnessed scores are present and structurally valid
//   2. Anchor: compare witnessed scores against canonical Q·K^T/√d (f32 or f64)
//      — delegated to `run_score_anchor_audit`
//   3. Replay: softmax(witnessed_scores) @ committed_V → requantize
//   4. Compare resulting `a` against committed `a` with ±1 LSB tolerance

fn phase_witnessed_score_attention(ctx: &Ctx, kv_transcript_ok: bool, st: &mut St) {
    if !kv_transcript_ok {
        return;
    }
    let kv_entries = match &ctx.r.kv_entries {
        Some(e) => e,
        None => return,
    };
    let shell = match &ctx.r.shell_opening {
        Some(s) => s,
        None => return,
    };
    let witnessed_scores = match &ctx.r.witnessed_scores {
        Some(ws) => ws,
        None => return, // silently skip — no scores captured
    };
    if ctx.key.score_anchor_threshold().is_none() {
        return; // should not happen (caller checked), but defensive
    }

    // Step 1+2: structural checks + anchor comparison are delegated to the
    // shared helper. Any threshold breach / KV-coverage / shape mismatch is
    // hard-failed via `st` inside the helper.
    run_score_anchor_audit(ctx, st);

    // Step 3+4: softmax(witnessed_scores) @ committed_V → requantize, compare
    // against committed `a`. This part is specific to the witnessed-score
    // verification path (not part of audit-only mode).
    let cfg = &ctx.key.config;
    let token_index = ctx.r.token_index as usize;
    let n_layers = cfg
        .n_layers
        .min(shell.layers.len())
        .min(ctx.r.retained.layers.len())
        .min(kv_entries.len())
        .min(witnessed_scores.len());

    for layer_idx in 0..n_layers {
        let sl = &shell.layers[layer_idx];
        let rs = &ctx.r.retained.layers[layer_idx];
        let ws = &witnessed_scores[layer_idx];

        if sl.q.is_none() {
            continue;
        }
        let layer_kv = &kv_entries[layer_idx];
        if layer_kv.len() < token_index + 1 {
            continue; // already reported by run_score_anchor_audit
        }
        let seq_len = token_index + 1;
        if ws.n_q_heads != cfg.n_q_heads || ws.seq_len != seq_len {
            continue; // already reported by run_score_anchor_audit
        }

        // Build V cache from committed KV transcript.
        let kv_v: Vec<Vec<f64>> = layer_kv[..=token_index]
            .iter()
            .map(|e| e.v_deq.clone())
            .collect();

        let (expected_a, _) =
            verilm_core::attention::replay_attention_witnessed_scores(
                &ws.scores,
                ws.n_q_heads,
                ws.seq_len,
                &kv_v,
                rs.scale_a as f64,
                cfg,
            );

        // Diagnostic tolerance for witnessed-score replay.
        //
        // A narrow early benchmark fit within ±3, but the broader 39-prompt
        // Qwen sweep breached it (global max_diff=9). This path must not be
        // used as a production strong-tier attention claim until replaced by
        // a kernel-aligned witness or deterministic attention kernel.
        const WITNESSED_SCORE_TOLERANCE: u16 = 3;
        st.check();
        let max_diff = expected_a
            .iter()
            .zip(rs.a.iter())
            .map(|(&e, &c)| (e as i16 - c as i16).unsigned_abs())
            .max()
            .unwrap_or(0);
        if max_diff > WITNESSED_SCORE_TOLERANCE {
            st.fail_ctx(
                FailureCode::AttentionExactMismatch,
                format!(
                    "layer {}: witnessed-score attention mismatch (max_diff={}, tolerance={})",
                    layer_idx, max_diff, WITNESSED_SCORE_TOLERANCE
                ),
                FailureContext {
                    layer: Some(layer_idx),
                    token_index: Some(ctx.r.token_index),
                    ..Default::default()
                },
            );
        }
    }
}

// --- Phase 7a: Exact attention verification
//
// Verifies the committed `a` vector for the challenged token by replaying
// canonical f64 attention: Q·K^T/√d → softmax → weights·V → requantize.
//
// Requires committed KV transcript (Merkle-verified by phase_kv_transcript)
// and shell Q accumulators. This is the high-assurance attention tier:
// exact match, no tolerance.

fn phase_exact_attention(ctx: &Ctx, kv_transcript_ok: bool, st: &mut St) {
    // Guard: only runs when KV transcript is committed and verified.
    if !kv_transcript_ok {
        return;
    }
    let kv_entries = match &ctx.r.kv_entries {
        Some(e) => e,
        None => return,
    };
    let shell = match &ctx.r.shell_opening {
        Some(s) => s,
        None => return,
    };

    let cfg = &ctx.key.config;
    let token_index = ctx.r.token_index as usize;
    let n_layers = cfg
        .n_layers
        .min(shell.layers.len())
        .min(ctx.r.retained.layers.len())
        .min(kv_entries.len());

    if ctx.key.rope_aware_replay {
        exact_attention_roped(ctx, kv_entries, shell, n_layers, token_index, st);
    } else {
        exact_attention_toy(ctx, kv_entries, shell, n_layers, token_index, st);
    }
}

/// Exact attention replay for production models (dequant → RoPE → f64 replay).
fn exact_attention_roped(
    ctx: &Ctx,
    kv_entries: &[Vec<KvEntry>],
    shell: &ShellTokenOpening,
    n_layers: usize,
    token_index: usize,
    st: &mut St,
) {
    let cfg = &ctx.key.config;
    // Position convention: opened token at `token_index + 1` matches prover's
    // `compute_kv_transcript` which uses 0-indexed positions for prefix and
    // the deep-prefix replay which uses `pos + 1` for the opened token.
    let absolute_pos = token_index + 1;

    for layer_idx in 0..n_layers {
        let sl = &shell.layers[layer_idx];
        let rs = &ctx.r.retained.layers[layer_idx];

        // Need Q accumulators for the challenged token.
        let q_acc = match &sl.q {
            Some(q) => q,
            None => continue, // Q not available (layer 0 without bridge, or missing data)
        };

        // Check KV coverage: need entries for positions 0..=token_index.
        let layer_kv = &kv_entries[layer_idx];
        if layer_kv.len() < token_index + 1 {
            st.check();
            st.fail_ctx(
                FailureCode::AttentionKvCoverageIncomplete,
                format!(
                    "layer {}: KV entries cover {} positions, need {}",
                    layer_idx,
                    layer_kv.len(),
                    token_index + 1,
                ),
                FailureContext {
                    layer: Some(layer_idx),
                    token_index: Some(ctx.r.token_index),
                    ..Default::default()
                },
            );
            continue;
        }

        // Build K/V cache from committed KV entries (already f64, RoPE applied to K).
        let kv_k: Vec<Vec<f64>> = layer_kv[..=token_index]
            .iter()
            .map(|e| e.k_roped.clone())
            .collect();
        let kv_v: Vec<Vec<f64>> = layer_kv[..=token_index]
            .iter()
            .map(|e| e.v_deq.clone())
            .collect();

        // Reconstruct Q: dequantize i32 accumulators → add bias → apply RoPE.
        let q_roped = dequant_rope_q(ctx.key, layer_idx, q_acc, sl.scale_x_attn, absolute_pos);

        // Replay canonical attention: Q·K^T/√d → softmax → weights·V → requantize.
        let expected_a = verilm_core::attention::replay_attention_roped(
            &q_roped,
            &kv_k,
            &kv_v,
            rs.scale_a as f64,
            cfg,
        );

        // Use the small profile attention tolerance. Large FlashAttention-vs-f64
        // divergence at non-zero token positions is a real unsolved gap, not a
        // reason to widen acceptance.
        let tol = attention_tolerance(ctx.key);
        st.check();
        let max_diff = expected_a
            .iter()
            .zip(rs.a.iter())
            .map(|(&e, &c)| (e as i16 - c as i16).unsigned_abs())
            .max()
            .unwrap_or(0);
        if max_diff > tol.max_abs_diff as u16 {
            st.fail_ctx(
                FailureCode::AttentionExactMismatch,
                format!(
                    "layer {}: attention mismatch (max_diff={}, tolerance={})",
                    layer_idx, max_diff, tol.max_abs_diff
                ),
                FailureContext {
                    layer: Some(layer_idx),
                    token_index: Some(ctx.r.token_index),
                    ..Default::default()
                },
            );
        }
    }
}

/// Exact attention replay for toy models (raw i32→i8, no RoPE).
fn exact_attention_toy(
    ctx: &Ctx,
    kv_entries: &[Vec<KvEntry>],
    shell: &ShellTokenOpening,
    n_layers: usize,
    token_index: usize,
    st: &mut St,
) {
    let cfg = &ctx.key.config;

    for layer_idx in 0..n_layers {
        let sl = &shell.layers[layer_idx];
        let rs = &ctx.r.retained.layers[layer_idx];

        let q_acc = match &sl.q {
            Some(q) => q,
            None => continue,
        };

        let layer_kv = &kv_entries[layer_idx];
        if layer_kv.len() < token_index + 1 {
            st.check();
            st.fail_ctx(
                FailureCode::AttentionKvCoverageIncomplete,
                format!(
                    "layer {}: KV entries cover {} positions, need {}",
                    layer_idx,
                    layer_kv.len(),
                    token_index + 1,
                ),
                FailureContext {
                    layer: Some(layer_idx),
                    token_index: Some(ctx.r.token_index),
                    ..Default::default()
                },
            );
            continue;
        }

        // Toy path: KV entries store f64-cast i8 values. Q is requantized directly.
        let kv_k: Vec<Vec<f64>> = layer_kv[..=token_index]
            .iter()
            .map(|e| e.k_roped.clone())
            .collect();
        let kv_v: Vec<Vec<f64>> = layer_kv[..=token_index]
            .iter()
            .map(|e| e.v_deq.clone())
            .collect();

        let q_i8 = verilm_core::requantize(q_acc);
        let q_f64: Vec<f64> = q_i8.iter().map(|&x| x as f64).collect();

        let expected_a = verilm_core::attention::replay_attention_roped(
            &q_f64,
            &kv_k,
            &kv_v,
            rs.scale_a as f64,
            cfg,
        );

        st.check();
        if expected_a != rs.a {
            let max_diff = expected_a
                .iter()
                .zip(rs.a.iter())
                .map(|(&e, &c)| (e as i16 - c as i16).unsigned_abs())
                .max()
                .unwrap_or(0);
            st.fail_ctx(
                FailureCode::AttentionExactMismatch,
                format!(
                    "layer {}: exact attention mismatch (max_diff={})",
                    layer_idx, max_diff
                ),
                FailureContext {
                    layer: Some(layer_idx),
                    token_index: Some(ctx.r.token_index),
                    ..Default::default()
                },
            );
        }
    }
}

// --- Wiring audit helpers (B3)
//
// Structural checks that the prover's attention wiring matches the committed
// architecture. These are cheap invariant checks, not replay paths:
//   - GQA: q_head → kv_head grouping shape (n_q_heads divisible by n_kv_heads,
//     kv_dim = n_kv_heads * d_head, per-entry K/V row lengths match).
//   - RoPE / position: rope_config_hash is bound in the verifier key, and KV
//     transcript covers exactly positions 0..=token_index (no gaps/extras).
//   - Causal mask: witnessed scores cover exactly seq_len = token_index + 1
//     (no future positions reachable for the challenged token).

/// Aggregate result of the wiring audit. Each sub-field is `None` when no
/// evidence to audit was present in the response, `Some(true)` when evidence
/// was present and consistent, `Some(false)` when a violation was detected
/// (and hard-failed via `st` in the same pass).
#[derive(Debug, Clone)]
pub(crate) struct WiringAuditRun {
    pub mask_ok: Option<bool>,
    pub rope_ok: Option<bool>,
    pub gqa_ok: Option<bool>,
}

/// Pure: the model config's GQA invariants are internally consistent.
fn check_gqa_config(cfg: &verilm_core::constants::ModelConfig) -> bool {
    cfg.n_q_heads > 0
        && cfg.n_kv_heads > 0
        && cfg.n_q_heads % cfg.n_kv_heads == 0
        && cfg.kv_dim == cfg.n_kv_heads * cfg.d_head
}

/// Pure: an opened KV entry's K/V row lengths match `n_kv_heads * d_head`.
fn check_kv_entry_gqa_shape(
    cfg: &verilm_core::constants::ModelConfig,
    entry: &verilm_core::types::KvEntry,
) -> bool {
    let expected = cfg.n_kv_heads * cfg.d_head;
    entry.k_roped.len() == expected && entry.v_deq.len() == expected
}

/// Pure: witnessed scores cover exactly positions 0..=token_index (seq_len
/// equals token_index + 1 — no future positions, no gaps).
fn check_causal_mask_ws(
    ws: &verilm_core::types::WitnessedScores,
    token_index: usize,
) -> bool {
    ws.seq_len == token_index + 1
}

/// Run wiring audit across GQA, RoPE/position, and causal mask. Returns
/// `Some(false)` for any sub-audit whose evidence was present and violated,
/// with a matching hard-fail recorded in `st`. Sub-audits with no basis in
/// the response remain `None`.
fn run_wiring_audit(ctx: &Ctx, st: &mut St) -> WiringAuditRun {
    let cfg = &ctx.key.config;
    let token_index = ctx.r.token_index as usize;

    // --- GQA ---
    // Config invariants are checked unconditionally (cheap; catches profile
    // or key-build bugs). Per-entry K/V row lengths are checked when KV
    // entries are present.
    let gqa_config_ok = check_gqa_config(cfg);
    if !gqa_config_ok {
        st.check();
        st.fail(
            FailureCode::AttentionWiringMismatch,
            format!(
                "GQA config inconsistent: n_q_heads={}, n_kv_heads={}, kv_dim={}, d_head={}",
                cfg.n_q_heads, cfg.n_kv_heads, cfg.kv_dim, cfg.d_head,
            ),
        );
    }

    let mut gqa_ok: Option<bool> = None;
    if let Some(kv_entries) = ctx.r.kv_entries.as_ref() {
        let mut shape_ok = true;
        for (layer_idx, layer_kv) in kv_entries.iter().enumerate() {
            for (pos, entry) in layer_kv.iter().enumerate() {
                if !check_kv_entry_gqa_shape(cfg, entry) {
                    st.check();
                    st.fail_ctx(
                        FailureCode::AttentionWiringMismatch,
                        format!(
                            "KV entry shape mismatch at layer {} pos {}: k_len={} v_len={} expected={}",
                            layer_idx,
                            pos,
                            entry.k_roped.len(),
                            entry.v_deq.len(),
                            cfg.n_kv_heads * cfg.d_head,
                        ),
                        FailureContext {
                            layer: Some(layer_idx),
                            token_index: Some(pos as u32),
                            ..Default::default()
                        },
                    );
                    shape_ok = false;
                }
            }
        }
        gqa_ok = Some(gqa_config_ok && shape_ok);
    } else if !gqa_config_ok {
        gqa_ok = Some(false);
    }

    // --- RoPE / position ---
    // Structural checks: rope_config_hash must be bound (it is cross-checked
    // against the manifest earlier in phase_specs), and KV transcript — if
    // present — must cover exactly positions 0..=token_index per layer.
    let mut rope_ok: Option<bool> = None;
    if let Some(kv_entries) = ctx.r.kv_entries.as_ref() {
        let hash_bound = ctx.key.rope_config_hash.is_some();
        if !hash_bound {
            st.check();
            st.fail(
                FailureCode::AttentionWiringMismatch,
                "rope_config_hash not bound in verifier key — position/RoPE semantics not committed",
            );
        }
        let mut coverage_ok = true;
        for (layer_idx, layer_kv) in kv_entries.iter().enumerate() {
            if layer_kv.len() != token_index + 1 {
                st.check();
                st.fail_ctx(
                    FailureCode::AttentionWiringMismatch,
                    format!(
                        "RoPE position coverage mismatch at layer {}: {} entries for token_index {} (expected {})",
                        layer_idx,
                        layer_kv.len(),
                        token_index,
                        token_index + 1,
                    ),
                    FailureContext {
                        layer: Some(layer_idx),
                        ..Default::default()
                    },
                );
                coverage_ok = false;
            }
        }
        rope_ok = Some(hash_bound && coverage_ok);
    }

    // --- Causal mask ---
    // Witnessed scores, if present, must cover exactly seq_len = token_index + 1.
    let mask_ok = ctx.r.witnessed_scores.as_ref().map(|ws_vec| {
        let mut ok = true;
        for (layer_idx, ws) in ws_vec.iter().enumerate() {
            if !check_causal_mask_ws(ws, token_index) {
                st.check();
                st.fail_ctx(
                    FailureCode::AttentionWiringMismatch,
                    format!(
                        "causal-mask violation at layer {}: witnessed seq_len={} but token_index={} (expected seq_len={})",
                        layer_idx,
                        ws.seq_len,
                        token_index,
                        token_index + 1,
                    ),
                    FailureContext {
                        layer: Some(layer_idx),
                        token_index: Some(ctx.r.token_index),
                        ..Default::default()
                    },
                );
                ok = false;
            }
        }
        ok
    });

    WiringAuditRun {
        mask_ok,
        rope_ok,
        gqa_ok,
    }
}

// --- Phase 7a: Audit-only attention reporting
//
// Arbitrary-position attention outputs are NOT verified in the kept product
// path. This phase emits the honest audit-only status. Individual audits
// (score anchoring, KV provenance, mask/RoPE/GQA wiring, local replay smoke)
// are wired below; local replay remains `None` until B4.

fn phase_audited_inputs_only(ctx: &Ctx, st: &mut St) {
    // Score anchoring is the one real audit wired in B1. It returns `None`
    // when prerequisites are absent (no witnessed scores, no shell opening,
    // no KV entries, no threshold) — that's a soft skip in audit-only mode.
    // Threshold breaches / structural errors are hard-failed inside the
    // helper via `st`.
    let score_anchor = run_score_anchor_audit(ctx, st).map(|r| crate::ScoreAnchorAudit {
        checked: r.n_layers_checked > 0,
        max_gap: r.max_gap.map(|g| g as f32),
    });

    // KV provenance is run once in `phase_kv_transcript` (before attention
    // dispatch). We reuse its result here. Absent → `None`; present + good →
    // `checked: true`; present + bad was already hard-failed via `st`.
    let kv_provenance = st
        .kv_provenance_run
        .as_ref()
        .filter(|r| r.entries_present)
        .map(|r| crate::KvProvenanceAudit {
            checked: r.all_verified,
        });

    // Wiring audits (B3): GQA shapes, RoPE/position coverage, causal mask.
    // Each sub-audit is `None` when no evidence was present, `Some(true)` when
    // present + consistent, `Some(false)` when present + violated (hard-failed
    // via `st` in the same pass).
    let wiring_run = run_wiring_audit(ctx, st);
    let wiring = if wiring_run.mask_ok.is_some()
        || wiring_run.rope_ok.is_some()
        || wiring_run.gqa_ok.is_some()
    {
        Some(crate::WiringAudit {
            mask_ok: wiring_run.mask_ok,
            rope_ok: wiring_run.rope_ok,
            gqa_ok: wiring_run.gqa_ok,
        })
    } else {
        None
    };

    st.attention_status = Some(crate::AttentionStatus::AuditedInputsNotVerified {
        score_anchor,
        kv_provenance,
        wiring,
        local_replay: None,
    });
    st.skipped.push(
        "attention: audit-only mode — arbitrary-position attention outputs \
         are not verified. Score anchoring and the causal-mask wiring audit \
         run only when the audited token is the last generated token \
         (witness retains Q for the final decode step only); KV provenance, \
         GQA, and RoPE-config audits run on the full opened range. Local \
         replay smoke (token-0 only) is not wired here."
            .into(),
    );
}

// --- Phase 7b: Deep prefix

fn phase_deep_prefix(ctx: &Ctx, kv_transcript_ok: bool, st: &mut St) {
    let (prefix_ret, prefix_shells) = match (&ctx.r.prefix_retained, &ctx.r.prefix_shell_openings) {
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
        let lp_ref = shell_j.lp_hidden_bf16.as_deref();
        let cl_ref = shell_j.captured_logits_f32.as_deref();
        let hash_j = merkle::hash_retained_with_captured_logits(ret_j, fr_ref, lp_ref, cl_ref);
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

        // Shell Freivalds + bridge.
        let before = st.failures.len();
        bridge_layers(ctx.key, ret_j, shell_j, st);
        for failure in &mut st.failures[before..] {
            failure.message = format!("prefix token {}: {}", j, failure.message);
            if failure.context.token_index.is_none() {
                failure.context.token_index = Some(j as u32);
            }
        }
    }

    // Deep-prefix attention replay: verify retained `a` for all prefix tokens
    // whose full attention context is available inside the committed chain.
    replay_deep_prefix_attention(ctx, prefix_ret, prefix_shells, kv_transcript_ok, st);
}

/// Deep-prefix attention replay: verify `a` for prefix tokens j >= 1 and
/// for the opened token, using the accumulated KV cache.
///
/// Two paths:
/// - **Toy** (`rope_aware_replay == false`): raw `requantize` i32→i8, no RoPE.
/// - **Production** (`rope_aware_replay == true`): dequantize → RoPE → f64 replay
///   → requantize with `scale_a`.
///
fn replay_deep_prefix_attention(
    ctx: &Ctx,
    prefix_ret: &[RetainedTokenState],
    prefix_shells: &[ShellTokenOpening],
    kv_transcript_ok: bool,
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

    // Use committed KV entries (already verified by phase_kv_transcript)
    // instead of reconstructing K/V from shell accumulators.
    let committed_kv = if kv_transcript_ok {
        ctx.r.kv_entries.as_deref()
    } else {
        None
    };

    if ctx.key.rope_aware_replay {
        replay_deep_prefix_roped(ctx, prefix_ret, prefix_shells, n_layers, committed_kv, st);
    } else {
        replay_deep_prefix_toy(ctx, prefix_ret, prefix_shells, n_layers, committed_kv, st);
    }
}

/// Toy/reference path: raw i8 requantize, no RoPE.
///
/// When `committed_kv` is `Some`, K/V cache entries come from the committed
/// KV transcript (already verified by `phase_kv_transcript`). Otherwise,
/// K/V is reconstructed from shell QKV accumulators (legacy path).
fn replay_deep_prefix_toy(
    ctx: &Ctx,
    prefix_ret: &[RetainedTokenState],
    prefix_shells: &[ShellTokenOpening],
    n_layers: usize,
    committed_kv: Option<&[Vec<KvEntry>]>,
    st: &mut St,
) {
    let key = &ctx.key;
    let cfg = &key.config;

    for layer_idx in 0..n_layers {
        let mut kv_k: Vec<Vec<i8>> = Vec::new();
        let mut kv_v: Vec<Vec<i8>> = Vec::new();

        for (j, (shell_j, ret_j)) in prefix_shells.iter().zip(prefix_ret.iter()).enumerate() {
            if layer_idx >= ret_j.layers.len() {
                break;
            }
            let sl = &shell_j.layers[layer_idx];
            let rs = &ret_j.layers[layer_idx];

            // Populate KV cache: committed entries if available, else reconstruct.
            if let Some(entry) = committed_kv
                .and_then(|kv| kv.get(layer_idx))
                .and_then(|layer| layer.get(j))
            {
                kv_k.push(entry.k_roped.iter().map(|&x| x as i8).collect());
                kv_v.push(entry.v_deq.iter().map(|&x| x as i8).collect());
            } else {
                let (_, k_acc, v_acc) = match (&sl.q, &sl.k, &sl.v) {
                    (Some(q), Some(k), Some(v)) => (q, k, v),
                    _ => break,
                };
                kv_k.push(verilm_core::requantize(k_acc));
                kv_v.push(verilm_core::requantize(v_acc));
            }

            if j == 0 {
                continue;
            }

            // Q always comes from shell accumulators.
            let q_acc = match &sl.q {
                Some(q) => q,
                None => break,
            };

            st.check();
            let q_i8 = verilm_core::requantize(q_acc);
            let expected_a =
                verilm_core::attention::replay_attention_reference(&q_i8, &kv_k, &kv_v, cfg);
            check_attention_result(
                ctx.key,
                st,
                &rs.a,
                &expected_a,
                "prefix token",
                j,
                layer_idx,
            );
        }

        // Opened token replay
        if let Some((q_acc, k_acc, v_acc, rs)) = opened_token_qkv(ctx, layer_idx) {
            let pos = ctx.r.token_index as usize;
            if let Some(entry) = committed_kv
                .and_then(|kv| kv.get(layer_idx))
                .and_then(|layer| layer.get(pos))
            {
                kv_k.push(entry.k_roped.iter().map(|&x| x as i8).collect());
                kv_v.push(entry.v_deq.iter().map(|&x| x as i8).collect());
            } else {
                kv_k.push(verilm_core::requantize(k_acc));
                kv_v.push(verilm_core::requantize(v_acc));
            }
            st.check();
            let q_i8 = verilm_core::requantize(q_acc);
            let expected_a =
                verilm_core::attention::replay_attention_reference(&q_i8, &kv_k, &kv_v, cfg);
            check_attention_result_opened(ctx, ctx.key, st, &rs.a, &expected_a, layer_idx);
        }
    }
}

/// Production path: dequantize + RoPE + f64 replay.
///
/// When `committed_kv` is `Some`, K/V cache entries come from the committed
/// KV transcript (already verified by `phase_kv_transcript`). Otherwise,
/// K/V is reconstructed via dequant + RoPE from shell QKV accumulators.
fn replay_deep_prefix_roped(
    ctx: &Ctx,
    prefix_ret: &[RetainedTokenState],
    prefix_shells: &[ShellTokenOpening],
    n_layers: usize,
    committed_kv: Option<&[Vec<KvEntry>]>,
    st: &mut St,
) {
    let cfg = &ctx.key.config;

    for layer_idx in 0..n_layers {
        let mut kv_k: Vec<Vec<f64>> = Vec::new();
        let mut kv_v: Vec<Vec<f64>> = Vec::new();

        for (j, (shell_j, ret_j)) in prefix_shells.iter().zip(prefix_ret.iter()).enumerate() {
            if layer_idx >= ret_j.layers.len() {
                break;
            }
            let sl = &shell_j.layers[layer_idx];
            let rs = &ret_j.layers[layer_idx];

            // Populate KV cache: committed entries if available, else reconstruct.
            if let Some(entry) = committed_kv
                .and_then(|kv| kv.get(layer_idx))
                .and_then(|layer| layer.get(j))
            {
                kv_k.push(entry.k_roped.clone());
                kv_v.push(entry.v_deq.clone());
            } else {
                let (q_acc, k_acc, v_acc) = match (&sl.q, &sl.k, &sl.v) {
                    (Some(q), Some(k), Some(v)) => (q, k, v),
                    _ => break,
                };
                let (_, k_roped, v_deq) = dequant_rope_qkv(
                    ctx.key,
                    layer_idx,
                    q_acc,
                    k_acc,
                    v_acc,
                    sl.scale_x_attn,
                    j + 1,
                );
                kv_k.push(k_roped);
                kv_v.push(v_deq);
            }

            if j == 0 {
                continue;
            }

            // Q always comes from shell accumulators.
            let q_acc = match &sl.q {
                Some(q) => q,
                None => break,
            };

            st.check();
            let q_roped = dequant_rope_q(ctx.key, layer_idx, q_acc, sl.scale_x_attn, j + 1);
            let expected_a = verilm_core::attention::replay_attention_roped(
                &q_roped,
                &kv_k,
                &kv_v,
                rs.scale_a as f64,
                cfg,
            );
            check_attention_result(
                ctx.key,
                st,
                &rs.a,
                &expected_a,
                "prefix token",
                j,
                layer_idx,
            );
        }

        // Opened token replay
        if let Some((q_acc, k_acc, v_acc, rs)) = opened_token_qkv(ctx, layer_idx) {
            let shell = ctx.r.shell_opening.as_ref().unwrap();
            let sl = &shell.layers[layer_idx];
            let pos = ctx.r.token_index as usize;
            let absolute_pos = pos + 1;

            if let Some(entry) = committed_kv
                .and_then(|kv| kv.get(layer_idx))
                .and_then(|layer| layer.get(pos))
            {
                kv_k.push(entry.k_roped.clone());
                kv_v.push(entry.v_deq.clone());
            } else {
                let (_, k_roped, v_deq) = dequant_rope_qkv(
                    ctx.key,
                    layer_idx,
                    q_acc,
                    k_acc,
                    v_acc,
                    sl.scale_x_attn,
                    absolute_pos,
                );
                kv_k.push(k_roped);
                kv_v.push(v_deq);
            }

            st.check();
            let q_roped = dequant_rope_q(ctx.key, layer_idx, q_acc, sl.scale_x_attn, absolute_pos);

            // Witnessed-score replay: when witnessed scores are available and the
            // profile has a score anchor threshold, verify the anchor and use
            // witnessed scores for a tighter softmax·V replay.
            let expected_a = if let (Some(ws_all), Some(threshold)) = (
                ctx.r.witnessed_scores.as_ref(),
                ctx.key.score_anchor_threshold(),
            ) {
                if let Some(ws) = ws_all.get(layer_idx) {
                    // Structural check: shape must match.
                    if ws.n_q_heads != cfg.n_q_heads || ws.seq_len != kv_v.len() {
                        st.fail_ctx(
                            FailureCode::WitnessedScoreStructuralError,
                            format!(
                                "layer {}: witnessed scores shape ({} heads, {} seq) \
                                 doesn't match expected ({} heads, {} seq)",
                                layer_idx, ws.n_q_heads, ws.seq_len,
                                cfg.n_q_heads, kv_v.len(),
                            ),
                            FailureContext {
                                layer: Some(layer_idx),
                                ..Default::default()
                            },
                        );
                        // Fall back to standard replay.
                        verilm_core::attention::replay_attention_roped(
                            &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
                        )
                    } else {
                        // Anchor check: compare witnessed scores against canonical Q·K^T/√d.
                        //
                        // For models with per-channel scales (W8A8 bf16), use exact CUTLASS
                        // epilogue + f32 RoPE (no bf16 truncation). Matches witness capture.py
                        // which promotes to f32, applies RoPE, and computes Q·K^T all in f32.
                        //
                        // For models without per-channel scales (toy/INT8), fall back to f64.
                        let has_per_channel =
                            ctx.key.per_channel_scales_for(layer_idx, MatrixType::Wq).is_some();

                        let anchor = if has_per_channel {
                            // Reconstruct Q: epilogue → RoPE in f32 (no bf16 truncation).
                            // Position = pos (= token_index), matching witness gen_pos = seq_len-1.
                            // NOT absolute_pos (pos+1) — that's an off-by-one vs witness.
                            let q_roped_f32 = dequant_rope_f32(
                                ctx.key, layer_idx, MatrixType::Wq,
                                q_acc, sl.scale_x_attn, cfg.n_q_heads, pos,
                            );

                            // Build K cache from shell accumulators.
                            // Position `j` maps to RoPE position `j` (matching witness arange).
                            let mut kv_k_f32: Vec<Vec<f32>> = Vec::with_capacity(kv_k.len());
                            for (j, shell_j) in prefix_shells.iter().enumerate() {
                                if j >= kv_k.len() {
                                    break;
                                }
                                let prefix_sl = &shell_j.layers[layer_idx];
                                if let Some(k_acc_j) = &prefix_sl.k {
                                    let k_roped = dequant_rope_f32(
                                        ctx.key, layer_idx, MatrixType::Wk,
                                        k_acc_j, prefix_sl.scale_x_attn,
                                        cfg.n_kv_heads, j,
                                    );
                                    kv_k_f32.push(k_roped);
                                } else {
                                    // No accumulator — use committed K_roped cast to f32.
                                    kv_k_f32.push(
                                        kv_k[j].iter().map(|&x| x as f32).collect()
                                    );
                                }
                            }
                            // Opened token's K — position = pos (matching witness).
                            if kv_k_f32.len() < kv_k.len() {
                                if let Some(k_acc_open) = &sl.k {
                                    let k_roped = dequant_rope_f32(
                                        ctx.key, layer_idx, MatrixType::Wk,
                                        k_acc_open, sl.scale_x_attn,
                                        cfg.n_kv_heads, pos,
                                    );
                                    kv_k_f32.push(k_roped);
                                } else {
                                    kv_k_f32.push(
                                        kv_k.last().unwrap().iter()
                                            .map(|&x| x as f32).collect()
                                    );
                                }
                            }

                            let canonical_f32 =
                                verilm_core::attention::compute_canonical_scores_gpu_like(
                                    &q_roped_f32, &kv_k_f32, cfg,
                                );
                            let canonical_f64: Vec<f64> =
                                canonical_f32.iter().map(|&x| x as f64).collect();
                            verilm_core::attention::anchor_witnessed_scores(
                                &ws.scores, &canonical_f64,
                                ws.n_q_heads, ws.seq_len, layer_idx,
                            )
                        } else {
                            // f64 anchoring (toy/INT8 models, or Llama where gap is <0.1).
                            let canonical_scores =
                                verilm_core::attention::compute_canonical_scores(
                                    &q_roped, &kv_k, cfg,
                                );
                            verilm_core::attention::anchor_witnessed_scores(
                                &ws.scores, &canonical_scores,
                                ws.n_q_heads, ws.seq_len, layer_idx,
                            )
                        };

                        if anchor.max_gap > threshold as f64 {
                            st.fail_ctx(
                                FailureCode::ScoreAnchorMismatch,
                                format!(
                                    "layer {}: witnessed score anchor gap {:.4} exceeds \
                                     threshold {:.1}",
                                    layer_idx, anchor.max_gap, threshold,
                                ),
                                FailureContext {
                                    layer: Some(layer_idx),
                                    ..Default::default()
                                },
                            );
                        }
                        // Use witnessed scores for replay (tighter corridor).
                        let (a_i8, _) =
                            verilm_core::attention::replay_attention_witnessed_scores(
                                &ws.scores,
                                ws.n_q_heads,
                                ws.seq_len,
                                &kv_v,
                                rs.scale_a as f64,
                                cfg,
                            );
                        a_i8
                    }
                } else {
                    // Layer index out of bounds in witnessed_scores — fallback.
                    verilm_core::attention::replay_attention_roped(
                        &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
                    )
                }
            } else {
                // No witnessed scores or no anchor threshold — standard replay.
                verilm_core::attention::replay_attention_roped(
                    &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
                )
            };
            check_attention_result_opened(ctx, ctx.key, st, &rs.a, &expected_a, layer_idx);
        }
    }
}

/// Extract the opened token's QKV accumulators and retained state for a given layer.
fn opened_token_qkv<'a>(
    ctx: &'a Ctx,
    layer_idx: usize,
) -> Option<(
    &'a [i32],
    &'a [i32],
    &'a [i32],
    &'a verilm_core::types::RetainedLayerState,
)> {
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
///
/// This function is intentionally compare-only: callers should keep using the
/// committed/opened `claimed` value for downstream verification rather than the
/// replayed `expected` value.
fn check_attention_result(
    key: &VerifierKey,
    st: &mut St,
    claimed: &[i8],
    expected: &[i8],
    label: &str,
    token_j: usize,
    layer_idx: usize,
) {
    let tolerance = attention_tolerance(key);
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
///
/// Like [`check_attention_result`], this is a non-state-replacing tolerance
/// check: the verifier continues from the committed/opened token state if the
/// comparison passes.
fn check_attention_result_opened(
    ctx: &Ctx,
    key: &VerifierKey,
    st: &mut St,
    claimed: &[i8],
    expected: &[i8],
    layer_idx: usize,
) {
    let tolerance = attention_tolerance(key);
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

// --- Phase 7b: KV transcript verification

/// Verify committed KV transcript Merkle proofs against `kv_roots`.
///
/// If the response includes `kv_entries` and `kv_proofs`, each entry is
/// re-hashed with `hash_kv_entry` and the Merkle proof is verified against
/// the corresponding `kv_roots[layer]`. This establishes that the KV values
/// opened during audit are exactly the values committed at generation time.
///
/// Skipped silently when `kv_roots` is empty (legacy / no KV commitment).
// --- KV provenance audit helpers
//
// KV provenance checks that each opened (layer, position) KV entry hashes to
// the committed `kv_roots[layer]` via its Merkle proof. Shared between:
//   - `phase_kv_transcript` (product path — hard-fails on any mismatch, used
//     as a gate for downstream attention replay phases).
//   - `phase_audited_inputs_only` (audit-only mode — reports whether
//     provenance was audited; hard-fail behavior is inherited from the
//     single shared run).

/// Aggregate result of the KV provenance audit.
#[derive(Debug, Clone)]
pub(crate) struct KvProvenanceRun {
    /// True when opened KV entries/proofs were present in the response. When
    /// false, no per-entry verification happened (legacy response or verifier
    /// did not request KV opening).
    pub entries_present: bool,
    /// True when every opened entry's Merkle proof verified against its
    /// committed `kv_roots[layer]` root.
    pub all_verified: bool,
}

/// Pure per-entry KV provenance check.
///
/// Recomputes the leaf hash from (layer_idx, pos, k_roped, v_deq) and verifies
/// the Merkle proof against the committed root. Any tampering of k, v, or
/// position produces a different leaf and fails verification.
fn verify_kv_entry_provenance(
    kv_root: &[u8; 32],
    layer_idx: usize,
    pos: usize,
    entry: &verilm_core::types::KvEntry,
    proof: &MerkleProof,
) -> bool {
    let leaf_hash = merkle::hash_kv_entry(layer_idx, pos, &entry.k_roped, &entry.v_deq);
    merkle::verify(kv_root, &leaf_hash, proof)
}

/// Run KV provenance audit across all opened layer groups.
///
/// Structural errors (kv_roots count mismatch, entries/proofs pair-up,
/// out-of-range layer group) and per-entry Merkle failures are hard-failed
/// via `st`. Returns a summary for callers that need to report audit status.
fn run_kv_provenance_audit(ctx: &Ctx, st: &mut St) -> KvProvenanceRun {
    let kv_roots = &ctx.r.commitment.kv_roots;
    if kv_roots.is_empty() {
        // Legacy response without KV commitment — nothing to verify.
        return KvProvenanceRun {
            entries_present: false,
            all_verified: true,
        };
    }

    let n_layers = ctx.key.config.n_layers;

    st.check();
    if kv_roots.len() != n_layers {
        st.fail_ctx(
            FailureCode::KvRootsCountMismatch,
            format!(
                "kv_roots has {} entries, expected {} (n_layers)",
                kv_roots.len(),
                n_layers
            ),
            FailureContext {
                expected: Some(n_layers.to_string()),
                actual: Some(kv_roots.len().to_string()),
                ..Default::default()
            },
        );
        return KvProvenanceRun {
            entries_present: ctx.r.kv_entries.is_some(),
            all_verified: false,
        };
    }

    let (entries, proofs) = match (&ctx.r.kv_entries, &ctx.r.kv_proofs) {
        (Some(e), Some(p)) => (e, p),
        (None, None) => {
            return KvProvenanceRun {
                entries_present: false,
                all_verified: true,
            };
        }
        _ => {
            st.check();
            st.fail(
                FailureCode::KvProofCountMismatch,
                "kv_entries and kv_proofs must both be present or both absent",
            );
            return KvProvenanceRun {
                entries_present: true,
                all_verified: false,
            };
        }
    };

    st.check();
    if entries.len() != proofs.len() {
        st.fail(
            FailureCode::KvProofCountMismatch,
            format!(
                "kv_entries has {} layer groups, kv_proofs has {}",
                entries.len(),
                proofs.len()
            ),
        );
        return KvProvenanceRun {
            entries_present: true,
            all_verified: false,
        };
    }

    let mut ok = true;
    for (group_idx, (layer_entries, layer_proofs)) in entries.iter().zip(proofs.iter()).enumerate()
    {
        st.check();
        if layer_entries.len() != layer_proofs.len() {
            st.fail_ctx(
                FailureCode::KvProofCountMismatch,
                format!(
                    "layer group {}: {} entries but {} proofs",
                    group_idx,
                    layer_entries.len(),
                    layer_proofs.len()
                ),
                FailureContext {
                    layer: Some(group_idx),
                    ..Default::default()
                },
            );
            ok = false;
            continue;
        }

        for (pos, (entry, proof)) in layer_entries.iter().zip(layer_proofs.iter()).enumerate() {
            // Prover opens layers in order, so group_idx == layer_idx until
            // selective layer opening is implemented.
            let layer_idx = group_idx;
            if layer_idx >= kv_roots.len() {
                st.check();
                st.fail_ctx(
                    FailureCode::KvEntriesCountMismatch,
                    format!(
                        "layer group {} exceeds kv_roots count {}",
                        group_idx,
                        kv_roots.len()
                    ),
                    FailureContext {
                        layer: Some(group_idx),
                        ..Default::default()
                    },
                );
                ok = false;
                break;
            }

            st.check();
            if !verify_kv_entry_provenance(&kv_roots[layer_idx], layer_idx, pos, entry, proof) {
                if std::env::var("VERILM_DEBUG_KV").map_or(false, |v| v == "1") {
                    let leaf_hash =
                        merkle::hash_kv_entry(layer_idx, pos, &entry.k_roped, &entry.v_deq);
                    let k_preview: Vec<String> = entry
                        .k_roped
                        .iter()
                        .take(4)
                        .map(|v| format!("{:.17e} ({:016x})", v, v.to_bits()))
                        .collect();
                    let v_preview: Vec<String> = entry
                        .v_deq
                        .iter()
                        .take(4)
                        .map(|v| format!("{:.17e} ({:016x})", v, v.to_bits()))
                        .collect();
                    let leaf_hex: String =
                        leaf_hash.iter().map(|b| format!("{:02x}", b)).collect();
                    let root_hex: String = kv_roots[layer_idx]
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect();
                    eprintln!(
                        "[KV DEBUG] layer={} pos={} k_len={} v_len={} leaf={} root={} proof_idx={} siblings={}",
                        layer_idx,
                        pos,
                        entry.k_roped.len(),
                        entry.v_deq.len(),
                        leaf_hex,
                        root_hex,
                        proof.leaf_index,
                        proof.siblings.len()
                    );
                    eprintln!("[KV DEBUG]   k_roped[:4] = {:?}", k_preview);
                    eprintln!("[KV DEBUG]   v_deq[:4]   = {:?}", v_preview);
                }
                st.fail_ctx(
                    FailureCode::KvProofInvalid,
                    format!(
                        "KV Merkle proof failed: layer {} position {}",
                        layer_idx, pos
                    ),
                    FailureContext {
                        layer: Some(layer_idx),
                        token_index: Some(pos as u32),
                        ..Default::default()
                    },
                );
                ok = false;
            }
        }
    }

    KvProvenanceRun {
        entries_present: true,
        all_verified: ok,
    }
}

fn phase_kv_transcript(ctx: &Ctx, st: &mut St) -> bool {
    let run = run_kv_provenance_audit(ctx, st);
    let ok = run.all_verified;
    st.kv_provenance_run = Some(run);
    ok
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
        skipped: st.skipped,
        attention_status: st.attention_status,
    }
}

// --- Small helpers

fn verify_freivalds(
    key: &VerifierKey,
    st: &mut St,
    layer: usize,
    mt: MatrixType,
    input: &[i8],
    accum: &[i32],
) {
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
                FailureContext {
                    field: Some(name.into()),
                    ..Default::default()
                },
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
                FailureContext {
                    field: Some(name.into()),
                    ..Default::default()
                },
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
                FailureContext {
                    field: Some(name.into()),
                    ..Default::default()
                },
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
                FailureContext {
                    field: Some(name.into()),
                    ..Default::default()
                },
            );
        }
    }
}
fn reject_feature(st: &mut St, rejected: bool, field: &str, actual: &str, expected: &str) {
    if rejected {
        st.fail_ctx(
            FailureCode::UnsupportedDecodeFeature,
            format!(
                "unsupported {}={} (canonical requires {})",
                field, actual, expected
            ),
            FailureContext {
                field: Some(field.into()),
                ..Default::default()
            },
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
            v_lm_head_f64: None,
            captured_logits_freivalds_seed: None,
            lm_head_bf16_hash: None,
            weight_hash: None,
            rmsnorm_attn_weights: vec![],
            rmsnorm_ffn_weights: vec![],
            weight_scales: vec![],
            per_channel_weight_scales: vec![],
            rmsnorm_eps: 1e-5,
            rope_config_hash: None,
            embedding_merkle_root: None,
            final_norm_weights: None,
            quant_family: None,
            scale_derivation: None,
            quant_block_size: None,
            attn_backend: None,
            attn_dtype: None,
            o_proj_alpha: vec![],
            rope_aware_replay: false,
            qkv_biases: vec![],
            verification_profile: None,
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
            merkle_proof: MerkleProof {
                leaf_index: 0,
                siblings: vec![],
            },
            io_proof: MerkleProof {
                leaf_index: 0,
                siblings: vec![],
            },
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
                kv_roots: vec![],
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
            kv_entries: None,
            kv_proofs: None,
            witnessed_scores: None,
        }
    }

    #[test]
    fn ctx_n_prompt_prefers_commitment() {
        let key = dummy_key();
        let r = dummy_response(0, 10, Some(5), Some(3));
        let ctx = Ctx::new(&key, &r, None, None, None);
        assert_eq!(ctx.n_prompt, 5);
    }

    #[test]
    fn ctx_n_prompt_falls_back_to_response() {
        let key = dummy_key();
        let r = dummy_response(0, 10, None, Some(3));
        let ctx = Ctx::new(&key, &r, None, None, None);
        assert_eq!(ctx.n_prompt, 3);
    }

    #[test]
    fn ctx_n_prompt_defaults_to_zero() {
        let key = dummy_key();
        let r = dummy_response(0, 10, None, None);
        let ctx = Ctx::new(&key, &r, None, None, None);
        assert_eq!(ctx.n_prompt, 0);
    }

    #[test]
    fn ctx_gen_start_from_n_prompt() {
        let key = dummy_key();
        let r = dummy_response(0, 10, Some(5), Some(5));
        let ctx = Ctx::new(&key, &r, None, None, None);
        assert_eq!(ctx.gen_start, 4); // 5 - 1
    }

    #[test]
    fn ctx_gen_start_saturates_at_zero() {
        let key = dummy_key();
        let r = dummy_response(0, 10, Some(0), Some(0));
        let ctx = Ctx::new(&key, &r, None, None, None);
        assert_eq!(ctx.gen_start, 0);
    }

    #[test]
    fn ctx_is_last_final_token() {
        let key = dummy_key();
        let r = dummy_response(9, 10, Some(1), Some(1));
        let ctx = Ctx::new(&key, &r, None, None, None);
        assert!(ctx.is_last);
    }

    #[test]
    fn ctx_is_last_not_final() {
        let key = dummy_key();
        let r = dummy_response(5, 10, Some(1), Some(1));
        let ctx = Ctx::new(&key, &r, None, None, None);
        assert!(!ctx.is_last);
    }

    #[test]
    fn ctx_is_last_single_token() {
        let key = dummy_key();
        let r = dummy_response(0, 1, Some(1), Some(1));
        let ctx = Ctx::new(&key, &r, None, None, None);
        assert!(ctx.is_last);
    }

    // ----- Score-anchor helper tests -----

    fn score_anchor_test_setup() -> (
        VerifierKey,
        Vec<i32>,
        f32,
        usize,
        Vec<verilm_core::types::KvEntry>,
        Vec<f64>,
    ) {
        let mut key = dummy_key();
        let cfg = key.config.clone();
        let n_mt = MatrixType::PER_LAYER.len();
        key.weight_scales = (0..cfg.n_layers).map(|_| vec![1.0f32; n_mt]).collect();

        let q_acc: Vec<i32> = (0..cfg.hidden_dim as i32)
            .map(|i| (i * 7 % 20) - 10)
            .collect();
        let scale_x_attn = 0.01f32;
        let token_index = 0usize;
        let layer_kv = vec![verilm_core::types::KvEntry {
            k_roped: (0..cfg.kv_dim).map(|i| 0.05 + 0.01 * (i as f64)).collect(),
            v_deq: vec![0.1; cfg.kv_dim],
        }];

        let q_roped = dequant_rope_q(&key, 0, &q_acc, scale_x_attn, token_index + 1);
        let kv_k: Vec<Vec<f64>> = layer_kv.iter().map(|e| e.k_roped.clone()).collect();
        let canonical_f64 =
            verilm_core::attention::compute_canonical_scores(&q_roped, &kv_k, &cfg);

        (key, q_acc, scale_x_attn, token_index, layer_kv, canonical_f64)
    }

    #[test]
    fn score_anchor_honest_scores_returns_small_gap() {
        let (key, q_acc, scale_x_attn, token_index, layer_kv, canonical) =
            score_anchor_test_setup();
        let witnessed: Vec<f32> = canonical.iter().map(|&v| v as f32).collect();
        let ws = verilm_core::types::WitnessedScores {
            scores: witnessed,
            n_q_heads: key.config.n_q_heads,
            seq_len: 1,
        };

        let stats =
            anchor_scores_for_layer(&key, 0, token_index, &q_acc, scale_x_attn, &layer_kv, &ws);

        assert!(
            stats.max_gap < 1e-5,
            "honest anchor gap too large: {}",
            stats.max_gap
        );
    }

    #[test]
    fn score_anchor_tampered_scores_returns_large_gap() {
        let (key, q_acc, scale_x_attn, token_index, layer_kv, canonical) =
            score_anchor_test_setup();
        let tampered: Vec<f32> = canonical.iter().map(|&v| (v + 1.0) as f32).collect();
        let ws = verilm_core::types::WitnessedScores {
            scores: tampered,
            n_q_heads: key.config.n_q_heads,
            seq_len: 1,
        };

        let stats =
            anchor_scores_for_layer(&key, 0, token_index, &q_acc, scale_x_attn, &layer_kv, &ws);

        assert!(
            stats.max_gap > 0.9,
            "tampered anchor gap too small: {}",
            stats.max_gap
        );
    }

    // --- B2: KV provenance audit helper tests ---

    fn kv_provenance_test_entry() -> (
        verilm_core::types::KvEntry,
        usize,
        usize,
        [u8; 32],
        MerkleProof,
    ) {
        let layer_idx = 0usize;
        let pos = 0usize;
        let entry = verilm_core::types::KvEntry {
            k_roped: vec![0.1, 0.2, 0.3, 0.4],
            v_deq: vec![0.5, 0.6, 0.7, 0.8],
        };
        // Single-leaf tree: root == leaf hash, empty sibling path.
        let leaf = merkle::hash_kv_entry(layer_idx, pos, &entry.k_roped, &entry.v_deq);
        let proof = MerkleProof {
            leaf_index: 0,
            siblings: vec![],
        };
        (entry, layer_idx, pos, leaf, proof)
    }

    #[test]
    fn kv_provenance_honest_entry_verifies() {
        let (entry, layer_idx, pos, root, proof) = kv_provenance_test_entry();
        assert!(verify_kv_entry_provenance(
            &root, layer_idx, pos, &entry, &proof,
        ));
    }

    #[test]
    fn kv_provenance_tampered_k_fails() {
        let (mut entry, layer_idx, pos, root, proof) = kv_provenance_test_entry();
        entry.k_roped[0] += 1e-6;
        assert!(!verify_kv_entry_provenance(
            &root, layer_idx, pos, &entry, &proof,
        ));
    }

    #[test]
    fn kv_provenance_tampered_v_fails() {
        let (mut entry, layer_idx, pos, root, proof) = kv_provenance_test_entry();
        entry.v_deq[2] = 0.0;
        assert!(!verify_kv_entry_provenance(
            &root, layer_idx, pos, &entry, &proof,
        ));
    }

    #[test]
    fn kv_provenance_wrong_position_fails() {
        let (entry, layer_idx, _pos, root, proof) = kv_provenance_test_entry();
        // Position is mixed into the leaf hash via domain separation; using
        // a different `pos` recomputes a different leaf that shouldn't match.
        assert!(!verify_kv_entry_provenance(
            &root, layer_idx, 1, &entry, &proof,
        ));
    }

    // --- B3: wiring audit helper tests ---

    fn wiring_test_cfg() -> ModelConfig {
        let mut cfg = ModelConfig::toy();
        cfg.n_q_heads = 8;
        cfg.n_kv_heads = 2;
        cfg.d_head = 4;
        cfg.kv_dim = cfg.n_kv_heads * cfg.d_head; // = 8
        cfg
    }

    #[test]
    fn wiring_gqa_config_honest_passes() {
        let cfg = wiring_test_cfg();
        assert!(check_gqa_config(&cfg));
    }

    #[test]
    fn wiring_gqa_config_not_divisible_fails() {
        let mut cfg = wiring_test_cfg();
        cfg.n_q_heads = 7; // 7 % 2 != 0
        assert!(!check_gqa_config(&cfg));
    }

    #[test]
    fn wiring_gqa_config_kv_dim_mismatch_fails() {
        let mut cfg = wiring_test_cfg();
        cfg.kv_dim = 7; // != n_kv_heads * d_head
        assert!(!check_gqa_config(&cfg));
    }

    #[test]
    fn wiring_gqa_shape_honest_entry_passes() {
        let cfg = wiring_test_cfg();
        let entry = verilm_core::types::KvEntry {
            k_roped: vec![0.0; cfg.kv_dim],
            v_deq: vec![0.0; cfg.kv_dim],
        };
        assert!(check_kv_entry_gqa_shape(&cfg, &entry));
    }

    #[test]
    fn wiring_gqa_shape_wrong_k_len_fails() {
        let cfg = wiring_test_cfg();
        let entry = verilm_core::types::KvEntry {
            k_roped: vec![0.0; cfg.kv_dim - 1],
            v_deq: vec![0.0; cfg.kv_dim],
        };
        assert!(!check_kv_entry_gqa_shape(&cfg, &entry));
    }

    #[test]
    fn wiring_gqa_shape_wrong_v_len_fails() {
        let cfg = wiring_test_cfg();
        let entry = verilm_core::types::KvEntry {
            k_roped: vec![0.0; cfg.kv_dim],
            v_deq: vec![0.0; cfg.kv_dim + 1],
        };
        assert!(!check_kv_entry_gqa_shape(&cfg, &entry));
    }

    #[test]
    fn wiring_mask_exact_seq_passes() {
        let ws = verilm_core::types::WitnessedScores {
            scores: vec![0.0; 8 * 5],
            n_q_heads: 8,
            seq_len: 5,
        };
        let token_index = 4usize;
        assert!(check_causal_mask_ws(&ws, token_index));
    }

    #[test]
    fn wiring_mask_future_position_fails() {
        let ws = verilm_core::types::WitnessedScores {
            scores: vec![0.0; 8 * 6],
            n_q_heads: 8,
            seq_len: 6, // reaches beyond token_index
        };
        let token_index = 4usize;
        assert!(!check_causal_mask_ws(&ws, token_index));
    }

    #[test]
    fn wiring_mask_short_seq_fails() {
        let ws = verilm_core::types::WitnessedScores {
            scores: vec![0.0; 8 * 3],
            n_q_heads: 8,
            seq_len: 3, // too short for token_index
        };
        let token_index = 4usize;
        assert!(!check_causal_mask_ws(&ws, token_index));
    }
}
