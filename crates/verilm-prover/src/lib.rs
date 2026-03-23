//! Prover engine: commitment generation and proof opening for the VeriLM protocol.

use verilm_core::constants::MatrixType;
use verilm_core::merkle;
use verilm_core::requantize;
use verilm_core::types::{
    AuditChallenge, AuditResponse, BatchCommitment, BatchProof, CommitmentVersion,
    DeploymentManifest, LayerTrace, RetainedLayerState, RetainedTokenState,
    TokenTrace, VerificationPolicy, VerifierKey,
};

/// Model geometry for trace construction from raw captures.
pub struct ModelGeometry {
    pub n_layers: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub intermediate_size: usize,
}

/// A single capture from the cutlass_scaled_mm hook.
pub struct CaptureEntry {
    pub x_i8: Vec<i8>,
    pub acc_i32: Vec<i32>,
    pub scale_a: f32,
}

/// Per-token, per-layer requantized K/V (incremental, not prefix snapshot).
///
/// `token_kvs[token_idx][layer_idx]` = `(k_i8, v_i8)` for that single token.
/// Used for KV chain computation and audit prefix reconstruction without
/// the O(N²) cost of cloning the full prefix into every `LayerTrace`.
pub type TokenKvs = Vec<Vec<(Vec<i8>, Vec<i8>)>>;

/// Build `Vec<Vec<LayerTrace>>` and per-token KV data from raw captures.
///
/// Replaces the Python `build_layer_traces` function. Takes flat capture
/// tuples (in call order) and reorganizes them into per-token, per-layer
/// traces with GQA-aware QKV split, gate/up split, and incremental KV storage.
///
/// The returned `TokenKvs` stores only each token's own requantized K_t, V_t.
/// KV cache fields in `LayerTrace` are left empty — KV provenance is committed
/// via the separate KV chain tree, not embedded in trace objects.
///
/// Captures arrive in order: for each forward pass, for each layer,
/// 4 projections: qkv_proj, o_proj, gate_up_proj, down_proj.
pub fn build_traces_from_captures(
    captures: &[CaptureEntry],
    geom: &ModelGeometry,
    fwd_batch_sizes: &[usize],
    level_c: bool,
    residuals: Option<&[Vec<f32>]>,
) -> (Vec<Vec<LayerTrace>>, Option<TokenKvs>) {
    const PROJS_PER_LAYER: usize = 4;
    let calls_per_fwd = geom.n_layers * PROJS_PER_LAYER;
    let total_tokens: usize = fwd_batch_sizes.iter().sum();

    let mut all_tokens: Vec<Vec<LayerTrace>> = Vec::with_capacity(total_tokens);
    let mut all_token_kvs: Option<Vec<Vec<(Vec<i8>, Vec<i8>)>>> = if level_c {
        Some(Vec::with_capacity(total_tokens))
    } else {
        None
    };

    for (fwd_idx, &batch_size) in fwd_batch_sizes.iter().enumerate() {
        let base = fwd_idx * calls_per_fwd;

        for b in 0..batch_size {
            let mut token_layers: Vec<LayerTrace> = Vec::with_capacity(geom.n_layers);
            let mut token_kv: Vec<(Vec<i8>, Vec<i8>)> = if level_c {
                Vec::with_capacity(geom.n_layers)
            } else {
                Vec::new()
            };

            for l in 0..geom.n_layers {
                let cap_base = base + l * PROJS_PER_LAYER;
                let qkv = &captures[cap_base];
                let o = &captures[cap_base + 1];
                let gu = &captures[cap_base + 2];
                let d = &captures[cap_base + 3];

                // Compute per-row dimensions from data length / batch_size.
                let qkv_in = qkv.x_i8.len() / batch_size;
                let qkv_out = qkv.acc_i32.len() / batch_size;
                let o_in = o.x_i8.len() / batch_size;
                let o_out = o.acc_i32.len() / batch_size;
                let gu_in = gu.x_i8.len() / batch_size;
                let gu_out = gu.acc_i32.len() / batch_size;
                let d_in = d.x_i8.len() / batch_size;
                let d_out = d.acc_i32.len() / batch_size;

                // Extract row b from each capture.
                let x_attn = qkv.x_i8[b * qkv_in..(b + 1) * qkv_in].to_vec();
                let qkv_row = &qkv.acc_i32[b * qkv_out..(b + 1) * qkv_out];
                let a = o.x_i8[b * o_in..(b + 1) * o_in].to_vec();
                let attn_out = o.acc_i32[b * o_out..(b + 1) * o_out].to_vec();
                let x_ffn = gu.x_i8[b * gu_in..(b + 1) * gu_in].to_vec();
                let gu_row = &gu.acc_i32[b * gu_out..(b + 1) * gu_out];
                let h = d.x_i8[b * d_in..(b + 1) * d_in].to_vec();
                let ffn_out = d.acc_i32[b * d_out..(b + 1) * d_out].to_vec();

                // GQA-aware QKV split.
                let q = qkv_row[..geom.q_dim].to_vec();
                let k = qkv_row[geom.q_dim..geom.q_dim + geom.kv_dim].to_vec();
                let v = qkv_row[geom.q_dim + geom.kv_dim..].to_vec();

                // Gate/up equal split.
                let half = gu_row.len() / 2;
                let g = gu_row[..half].to_vec();
                let u = gu_row[half..].to_vec();

                // Incremental KV: requantize once, store per-token (not prefix).
                if level_c {
                    let k_i8 = requantize(&k);
                    let v_i8 = requantize(&v);
                    token_kv.push((k_i8, v_i8));
                }

                // Residual: indexed by fwd_pass * n_layers + layer.
                let residual = residuals.and_then(|res| {
                    let res_idx = fwd_idx * geom.n_layers + l;
                    res.get(res_idx).and_then(|r| {
                        let dim = r.len() / batch_size;
                        let start = b * dim;
                        let end = start + dim;
                        if end <= r.len() { Some(r[start..end].to_vec()) } else { None }
                    })
                });

                token_layers.push(LayerTrace {
                    x_attn,
                    q,
                    k,
                    v,
                    a,
                    attn_out,
                    x_ffn,
                    g,
                    u,
                    h,
                    ffn_out,
                    kv_cache_k: Vec::new(),
                    kv_cache_v: Vec::new(),
                    scale_x_attn: Some(qkv.scale_a),
                    scale_a: Some(o.scale_a),
                    scale_x_ffn: Some(gu.scale_a),
                    scale_h: Some(d.scale_a),
                    residual,
                });
            }

            all_tokens.push(token_layers);
            if let Some(ref mut kvs) = all_token_kvs {
                kvs.push(token_kv);
            }
        }
    }

    (all_tokens, all_token_kvs)
}

/// Prover-side state after Phase 1 (commit) but before Phase 2 (open).
///
/// Holds the Merkle trees and raw layer data. Only the commitment is
/// published; the full traces stay with the prover until challenged.
pub struct BatchState {
    pub trace_tree: merkle::MerkleTree,
    pub io_tree: merkle::MerkleTree,
    pub all_layers: Vec<Vec<LayerTrace>>,
    /// Manifest hash from the commitment, if any.
    pub manifest_hash: Option<[u8; 32]>,
    /// Per-token emitted token IDs (V2/V3 IO hash). `None` for legacy commits.
    pub token_ids: Option<Vec<u32>>,
    /// Protocol version for the IO hash format.
    pub version: CommitmentVersion,
    /// Prompt hash for request identity binding.
    pub prompt_hash: Option<[u8; 32]>,
    /// Seed commitment for sampling binding.
    pub seed_commitment: Option<[u8; 32]>,
    /// Revealed sampling seed (stored for open phase).
    pub revealed_seed: Option<[u8; 32]>,
    /// Per-token IO hashes (V3 only). Used to provide `prev_io_hash` during open.
    pub io_hashes: Option<Vec<[u8; 32]>>,
    /// Per-token KV chain hashes (V3 only). Used to provide `prev_kv_hash` during open.
    pub kv_chain_hashes: Option<Vec<[u8; 32]>>,
    /// KV chain Merkle tree root (V3 only).
    pub kv_chain_root: Option<[u8; 32]>,
    /// Per-token KV Merkle roots, computed during generation.
    /// Each root is the Merkle root over per-layer KV hashes for that token.
    /// Published before the challenge so the verifier can build a StreamingKvVerifier.
    pub kv_merkle_roots: Option<Vec<[u8; 32]>>,
    /// Pre-computed per-token, per-layer requantized K/V (incremental).
    /// `token_kvs[token_idx][layer_idx]` = `(k_i8, v_i8)`.
    /// Used for KV prefix reconstruction during audit. `None` for Level A/B.
    pub token_kvs: Option<TokenKvs>,
}

/// Full binding parameters for V3 commit.
pub struct FullBindingParams<'a> {
    pub token_ids: &'a [u32],
    pub prompt: &'a [u8],
    pub sampling_seed: [u8; 32],
    /// Optional deployment manifest. When provided, its hash is bound into
    /// the commitment alongside the other V3 fields.
    pub manifest: Option<&'a DeploymentManifest>,
}

/// Phase 1 — Commit (legacy V1 IO hash, no token-ID binding).
///
/// Builds the trace tree and IO tree over all N token traces.
/// Returns a small commitment (two roots + count) that the prover
/// publishes, plus opaque state for selective opening later.
/// No per-token Merkle proofs are generated at this stage.
///
/// Use `commit_with_token_ids` for V2 IO hash that binds emitted tokens.
pub fn commit(all_layers: Vec<Vec<LayerTrace>>) -> (BatchCommitment, BatchState) {
    commit_inner(all_layers, None)
}

/// Alias for `commit` — makes the legacy (no token-ID) path explicit in tests.
pub fn commit_legacy(all_layers: Vec<Vec<LayerTrace>>) -> (BatchCommitment, BatchState) {
    commit(all_layers)
}

/// Phase 1 — Commit with token-ID binding (V2 IO hash).
///
/// Same as `commit`, but each IO leaf includes the emitted token ID:
///   `H("vi-io-v2" || first_input || last_output || token_id)`
///
/// `token_ids.len()` must equal `all_layers.len()`.
pub fn commit_with_token_ids(
    all_layers: Vec<Vec<LayerTrace>>,
    token_ids: &[u32],
) -> (BatchCommitment, BatchState) {
    assert_eq!(
        all_layers.len(),
        token_ids.len(),
        "token_ids length must match number of tokens"
    );
    commit_inner(all_layers, Some(token_ids))
}

fn commit_inner(
    all_layers: Vec<Vec<LayerTrace>>,
    token_ids: Option<&[u32]>,
) -> (BatchCommitment, BatchState) {
    let trace_leaves: Vec<[u8; 32]> = all_layers
        .iter()
        .map(|layers| {
            let fh = layers.last().map(|lt| requantize(&lt.ffn_out));
            merkle::hash_trace_direct(layers, fh.as_deref())
        })
        .collect();

    let io_leaves: Vec<[u8; 32]> = all_layers
        .iter()
        .enumerate()
        .map(|(i, layers)| {
            let tid = token_ids.map(|ids| ids[i]);
            merkle::io_hash(&layers[0].x_attn, &layers.last().unwrap().ffn_out, &merkle::io_binding(tid, None))
        })
        .collect();

    let trace_tree = merkle::build_tree(&trace_leaves);
    let io_tree = merkle::build_tree(&io_leaves);

    let version = if token_ids.is_some() {
        CommitmentVersion::V2
    } else {
        CommitmentVersion::V1
    };

    let commitment = BatchCommitment {
        merkle_root: trace_tree.root,
        io_root: io_tree.root,
        n_tokens: all_layers.len() as u32,
        manifest_hash: None,
        version,
        prompt_hash: None,
        seed_commitment: None,
        kv_chain_root: None,
    };

    let state = BatchState {
        trace_tree,
        io_tree,
        all_layers,
        manifest_hash: None,
        token_ids: token_ids.map(|ids| ids.to_vec()),
        version,
        prompt_hash: None,
        seed_commitment: None,
        revealed_seed: None,
        io_hashes: None,
        kv_chain_hashes: None,
        kv_chain_root: None,
        kv_merkle_roots: None,
        token_kvs: None,
    };

    (commitment, state)
}

/// Phase 1 — Commit with deployment manifest binding.
///
/// Same as `commit()` but also computes `hash_manifest(&manifest)` and sets
/// `commitment.manifest_hash = Some(hash)`. This binds sampling parameters,
/// tokenizer identity, and EOS policy to the batch commitment.
pub fn commit_with_manifest(
    all_layers: Vec<Vec<LayerTrace>>,
    manifest: &DeploymentManifest,
) -> (BatchCommitment, BatchState) {
    let (mut commitment, mut state) = commit(all_layers);
    let mh = Some(merkle::hash_manifest(manifest));
    commitment.manifest_hash = mh;
    state.manifest_hash = mh;
    (commitment, state)
}

/// Phase 1 — Commit with full binding (V3): token IDs, transcript chaining,
/// prompt hash, and sampling seed commitment.
///
/// IO leaves use chained hashing: each token's IO hash depends on the previous.
/// This prevents insertion, deletion, reordering, and retroactive edits.
///
/// `token_kvs`: pre-computed per-token, per-layer requantized K/V from
/// `build_traces_from_captures`. When `Some`, KV chain and Merkle roots
/// are computed from these without re-requantizing. When `None`, falls
/// back to `requantize(&lt.k)` per layer.
pub fn commit_with_full_binding(
    all_layers: Vec<Vec<LayerTrace>>,
    params: &FullBindingParams,
    token_kvs: Option<TokenKvs>,
) -> (BatchCommitment, BatchState) {
    assert_eq!(
        all_layers.len(),
        params.token_ids.len(),
        "token_ids length must match number of tokens"
    );

    let trace_leaves: Vec<[u8; 32]> = all_layers
        .iter()
        .map(|layers| {
            let fh = layers.last().map(|lt| requantize(&lt.ffn_out));
            merkle::hash_trace_direct(layers, fh.as_deref())
        })
        .collect();

    // Build chained IO leaves: each depends on the previous
    let mut io_leaves: Vec<[u8; 32]> = Vec::with_capacity(all_layers.len());
    let mut prev_io = [0u8; 32]; // genesis: zero hash for first token
    for (i, layers) in all_layers.iter().enumerate() {
        let binding = merkle::IoHashBinding::Chained {
            token_id: params.token_ids[i],
            prev_io_hash: prev_io,
        };
        let io = merkle::io_hash(
            &layers[0].x_attn,
            &layers.last().unwrap().ffn_out,
            &binding,
        );
        io_leaves.push(io);
        prev_io = io;
    }

    // Build KV provenance chain using pre-computed KVs when available.
    // Borrows slices from token_kvs — no cloning.
    let mut kv_chain_leaves: Vec<[u8; 32]> = Vec::with_capacity(all_layers.len());
    let mut kv_merkle_roots: Vec<[u8; 32]> = Vec::with_capacity(all_layers.len());
    let mut prev_kv = [0u8; 32]; // genesis: zero hash for first token
    for (i, layers) in all_layers.iter().enumerate() {
        if let Some(ref kvs) = token_kvs {
            // Fast path: borrow pre-computed requantized K/V (no clone, no re-requantize).
            let k_refs: Vec<&[i8]> = kvs[i].iter().map(|(k, _)| k.as_slice()).collect();
            let v_refs: Vec<&[i8]> = kvs[i].iter().map(|(_, v)| v.as_slice()).collect();
            let kv = merkle::kv_chain_hash(&prev_kv, &k_refs, &v_refs, i as u32);
            kv_chain_leaves.push(kv);
            prev_kv = kv;
            // Root-only: no full tree storage at commit time.
            kv_merkle_roots.push(verilm_core::streaming::kv_layer_root(i as u32, &k_refs, &v_refs));
        } else {
            // Fallback: requantize from i32 projections.
            let k: Vec<Vec<i8>> = layers.iter().map(|lt| requantize(&lt.k)).collect();
            let v: Vec<Vec<i8>> = layers.iter().map(|lt| requantize(&lt.v)).collect();
            let kv = merkle::kv_chain_hash(&prev_kv, &k, &v, i as u32);
            kv_chain_leaves.push(kv);
            prev_kv = kv;
            kv_merkle_roots.push(verilm_core::streaming::kv_layer_root(i as u32, &k, &v));
        };
    }

    let trace_tree = merkle::build_tree(&trace_leaves);
    let io_tree = merkle::build_tree(&io_leaves);
    let kv_chain_tree = merkle::build_tree(&kv_chain_leaves);

    let manifest_hash = params.manifest.map(|m| merkle::hash_manifest(m));

    let commitment = BatchCommitment {
        merkle_root: trace_tree.root,
        io_root: io_tree.root,
        n_tokens: all_layers.len() as u32,
        manifest_hash,
        version: CommitmentVersion::V3,
        prompt_hash: Some(merkle::hash_prompt(params.prompt)),
        seed_commitment: Some(merkle::hash_seed(&params.sampling_seed)),
        kv_chain_root: Some(kv_chain_tree.root),
    };

    let state = BatchState {
        trace_tree,
        io_tree,
        all_layers,
        manifest_hash,
        token_ids: Some(params.token_ids.to_vec()),
        version: CommitmentVersion::V3,
        prompt_hash: Some(merkle::hash_prompt(params.prompt)),
        seed_commitment: Some(merkle::hash_seed(&params.sampling_seed)),
        revealed_seed: Some(params.sampling_seed),
        io_hashes: Some(io_leaves),
        kv_chain_hashes: Some(kv_chain_leaves),
        kv_chain_root: Some(kv_chain_tree.root),
        kv_merkle_roots: Some(kv_merkle_roots),
        token_kvs,
    };

    (commitment, state)
}

// ===========================================================================
// V4: Minimal retained-state commitment (paper-minimal online path)
// ===========================================================================

/// Prover-side state after V4 (minimal retained-state) commit.
///
/// Stores only the non-replayable attention boundary and protocol bindings.
/// No full LayerTrace, no i32 accumulators, no KV chain at commit time.
pub struct MinimalBatchState {
    pub retained_tree: merkle::MerkleTree,
    pub io_tree: merkle::MerkleTree,
    pub all_retained: Vec<RetainedTokenState>,
    pub manifest_hash: Option<[u8; 32]>,
    pub token_ids: Vec<u32>,
    pub prompt_hash: [u8; 32],
    pub seed_commitment: [u8; 32],
    pub revealed_seed: [u8; 32],
    pub io_hashes: Vec<[u8; 32]>,
}

/// Build retained state from minimal captures (no i32 accumulators).
///
/// Each `MinimalCaptureEntry` contains only the o_proj input (`a`)
/// and the four per-layer quantization scales. No `_int_mm` needed.
pub struct MinimalCaptureEntry {
    /// Post-attention INT8 output (W_o input). Extracted from o_proj's x_i8.
    pub a_i8: Vec<i8>,
    /// Activation scale for QKV projection input.
    pub scale_x_attn: f32,
    /// Activation scale for W_o projection input.
    pub scale_a: f32,
    /// Activation scale for gate_up projection input.
    pub scale_x_ffn: f32,
    /// Activation scale for down projection input.
    pub scale_h: f32,
}

/// Build `Vec<RetainedTokenState>` from minimal captures.
///
/// `captures` has one entry per layer per forward pass, ordered by
/// forward pass → layer. Each entry's `a_i8` contains rows for all
/// tokens in that forward pass's batch (contiguous, row-major).
pub fn build_retained_from_captures(
    captures: &[MinimalCaptureEntry],
    n_layers: usize,
    fwd_batch_sizes: &[usize],
) -> Vec<RetainedTokenState> {
    let mut all_retained = Vec::new();
    let mut cap_idx = 0;

    for &batch_size in fwd_batch_sizes {
        for b in 0..batch_size {
            let mut layers = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                let entry = &captures[cap_idx + l];
                let a_dim = entry.a_i8.len() / batch_size;
                let a = entry.a_i8[b * a_dim..(b + 1) * a_dim].to_vec();

                layers.push(RetainedLayerState {
                    a,
                    scale_a: entry.scale_a,
                    scale_x_attn: entry.scale_x_attn,
                    scale_x_ffn: entry.scale_x_ffn,
                    scale_h: entry.scale_h,
                });
            }

            all_retained.push(RetainedTokenState { layers });
        }
        cap_idx += n_layers;
    }

    all_retained
}

/// V4 commit: retained-state commitment.
///
/// Trace tree: Merkle over `hash_retained_state_direct(token)` leaves.
/// IO tree: chain `H("vi-io-v4" || leaf_hash || token_id || prev_io)`.
/// Prefix binding: auditor opens prior retained leaves from the Merkle tree.
pub fn commit_minimal(
    all_retained: Vec<RetainedTokenState>,
    params: &FullBindingParams,
) -> (BatchCommitment, MinimalBatchState) {
    assert_eq!(
        all_retained.len(),
        params.token_ids.len(),
        "retained state count must match token_ids"
    );

    // Trace tree: hash retained state per token.
    let trace_leaves: Vec<[u8; 32]> = all_retained
        .iter()
        .map(|rs| merkle::hash_retained_state_direct(rs))
        .collect();

    // IO tree: chain the leaf hash (not ad hoc features) for splice resistance.
    let mut io_leaves = Vec::with_capacity(all_retained.len());
    let mut prev_io = [0u8; 32];
    for (i, leaf_hash) in trace_leaves.iter().enumerate() {
        let io = merkle::io_hash_v4(*leaf_hash, params.token_ids[i], prev_io);
        io_leaves.push(io);
        prev_io = io;
    }

    let trace_tree = merkle::build_tree(&trace_leaves);
    let io_tree = merkle::build_tree(&io_leaves);
    let manifest_hash = params.manifest.map(|m| merkle::hash_manifest(m));

    let commitment = BatchCommitment {
        merkle_root: trace_tree.root,
        io_root: io_tree.root,
        n_tokens: all_retained.len() as u32,
        manifest_hash,
        version: CommitmentVersion::V4,
        prompt_hash: Some(merkle::hash_prompt(params.prompt)),
        seed_commitment: Some(merkle::hash_seed(&params.sampling_seed)),
        kv_chain_root: None, // prefix binding via retained Merkle tree, not separate KV root
    };

    let state = MinimalBatchState {
        retained_tree: trace_tree,
        io_tree,
        all_retained,
        manifest_hash,
        token_ids: params.token_ids.to_vec(),
        prompt_hash: merkle::hash_prompt(params.prompt),
        seed_commitment: merkle::hash_seed(&params.sampling_seed),
        revealed_seed: params.sampling_seed,
        io_hashes: io_leaves,
    };

    (commitment, state)
}

/// Phase 2 — Open.
///
/// Given the prover's internal state and a set of challenge indices,
/// generates Merkle proofs *only* for the challenged tokens and
/// assembles the BatchProof. This is O(k log N) instead of O(N log N).
pub fn open(state: &BatchState, challenge_indices: &[u32]) -> BatchProof {
    let commitment = BatchCommitment {
        merkle_root: state.trace_tree.root,
        io_root: state.io_tree.root,
        n_tokens: state.all_layers.len() as u32,
        manifest_hash: state.manifest_hash,
        version: state.version,
        prompt_hash: state.prompt_hash,
        seed_commitment: state.seed_commitment,
        kv_chain_root: state.kv_chain_root,
    };

    let traces: Vec<TokenTrace> = challenge_indices
        .iter()
        .map(|&idx| {
            let i = idx as usize;
            let merkle_proof = merkle::prove(&state.trace_tree, i);
            let io_proof = merkle::prove(&state.io_tree, i);
            let token_id = state.token_ids.as_ref().map(|ids| ids[i]);
            // For V3, provide prev_io_hash so the verifier can reconstruct
            // the chain even for non-consecutive challenge sets.
            let prev_io_hash = state.io_hashes.as_ref().map(|hashes| {
                if i == 0 {
                    [0u8; 32] // genesis zero hash
                } else {
                    hashes[i - 1]
                }
            });
            // For KV provenance chaining, provide prev_kv_hash.
            let prev_kv_hash = state.kv_chain_hashes.as_ref().map(|hashes| {
                if i == 0 {
                    [0u8; 32] // genesis zero hash
                } else {
                    hashes[i - 1]
                }
            });
            // Derive final_hidden from last layer's ffn_out (requantize i32 → i8).
            let final_hidden = state.all_layers[i]
                .last()
                .map(|lt| requantize(&lt.ffn_out));

            TokenTrace {
                token_index: idx,
                layers: state.all_layers[i].clone(),
                merkle_root: state.trace_tree.root,
                merkle_proof,
                io_proof,
                margin_cert: None,
                final_hidden,
                token_id,
                prev_io_hash,
                prev_kv_hash,
            }
        })
        .collect();

    BatchProof {
        commitment,
        traces,
        revealed_seed: state.revealed_seed,
    }
}

/// Convenience wrapper: commit + open all tokens (for tests that need every trace).
pub fn build_batch(
    all_layers: Vec<Vec<LayerTrace>>,
) -> (BatchCommitment, BatchState, Vec<TokenTrace>) {
    let n = all_layers.len() as u32;
    let (commitment, state) = commit(all_layers);
    let all_indices: Vec<u32> = (0..n).collect();
    let proof = open(&state, &all_indices);
    (commitment, state, proof.traces)
}

/// Build an `AuditResponse` for a stratified audit challenge.
///
/// Standalone version that recomputes KV and trace Merkle trees from raw traces.
/// Prefer `build_audit_response_from_state` when a `BatchState` is available.
pub fn build_audit_response(
    all_traces: &[Vec<LayerTrace>],
    challenge: &AuditChallenge,
) -> AuditResponse {
    use verilm_core::streaming;

    let token_idx = challenge.token_index as usize;
    assert!(token_idx < all_traces.len(), "token_index out of range");
    let token_layers = &all_traces[token_idx];

    let partial_layers: Vec<LayerTrace> = challenge.layer_indices.iter()
        .map(|&l| token_layers[l].clone()).collect();

    // Reconstruct KV prefix from all previous tokens' traces.
    let (kv_k_prefix, kv_v_prefix) = reconstruct_kv_prefix(
        all_traces, token_idx, &challenge.layer_indices, None,
    );

    // Build KV layer tree for per-layer Merkle proofs.
    let k_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.k)).collect();
    let v_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.v)).collect();
    let kv_tree = streaming::build_kv_layer_tree(challenge.token_index, &k_per_layer, &v_per_layer);

    let kv_layer_proofs: Vec<verilm_core::merkle::MerkleProof> = challenge
        .layer_indices.iter().map(|&l| verilm_core::merkle::prove(&kv_tree, l)).collect();

    // Build Merkle commitment for the full trace (all layers)
    let n_tokens = all_traces.len();
    let mut leaves = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let fh = all_traces[t].last().map(|lt| requantize(&lt.ffn_out));
        leaves.push(merkle::hash_trace_direct(&all_traces[t], fh.as_deref()));
    }
    let trace_tree = verilm_core::merkle::build_tree(&leaves);
    let merkle_proof = verilm_core::merkle::prove(&trace_tree, token_idx);

    let final_hidden = all_traces[token_idx]
        .last()
        .map(|lt| requantize(&lt.ffn_out));

    AuditResponse {
        token_index: challenge.token_index,
        partial_layers,
        layer_indices: challenge.layer_indices.clone(),
        kv_k_prefix,
        kv_v_prefix,
        kv_layer_proofs,
        merkle_root: trace_tree.root,
        merkle_proof,
        final_hidden,
        token_id: None,
    }
}

/// Build an `AuditResponse` using pre-computed trees from `BatchState`.
///
/// Uses `state.trace_tree` for the Merkle proof and recomputes only the
/// per-token KV layer tree (needed for per-layer opening proofs).
pub fn build_audit_response_from_state(
    state: &BatchState,
    challenge: &AuditChallenge,
) -> AuditResponse {
    use verilm_core::streaming;

    let token_idx = challenge.token_index as usize;
    assert!(token_idx < state.all_layers.len(), "token_index out of range");
    let token_layers = &state.all_layers[token_idx];

    let partial_layers: Vec<LayerTrace> = challenge.layer_indices.iter()
        .map(|&l| token_layers[l].clone()).collect();

    // Reconstruct KV prefix: use pre-computed token_kvs if available.
    let (kv_k_prefix, kv_v_prefix) = reconstruct_kv_prefix(
        &state.all_layers, token_idx, &challenge.layer_indices, state.token_kvs.as_ref(),
    );

    // KV layer tree: borrow pre-computed token_kvs or requantize.
    // Audit needs full tree (for Merkle proofs), not just root.
    let kv_tree = if let Some(ref kvs) = state.token_kvs {
        let k_refs: Vec<&[i8]> = kvs[token_idx].iter().map(|(k, _)| k.as_slice()).collect();
        let v_refs: Vec<&[i8]> = kvs[token_idx].iter().map(|(_, v)| v.as_slice()).collect();
        streaming::build_kv_layer_tree(challenge.token_index, &k_refs, &v_refs)
    } else {
        let k: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.k)).collect();
        let v: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.v)).collect();
        streaming::build_kv_layer_tree(challenge.token_index, &k, &v)
    };

    let kv_layer_proofs: Vec<verilm_core::merkle::MerkleProof> = challenge
        .layer_indices.iter().map(|&l| verilm_core::merkle::prove(&kv_tree, l)).collect();

    // Use pre-computed trace tree from BatchState.
    let merkle_proof = verilm_core::merkle::prove(&state.trace_tree, token_idx);

    // Derive final_hidden from last layer's ffn_out (same as normal open path).
    let final_hidden = token_layers
        .last()
        .map(|lt| requantize(&lt.ffn_out));

    let token_id = state.token_ids.as_ref().map(|ids| ids[token_idx]);

    AuditResponse {
        token_index: challenge.token_index,
        partial_layers,
        layer_indices: challenge.layer_indices.clone(),
        kv_k_prefix,
        kv_v_prefix,
        kv_layer_proofs,
        merkle_root: state.trace_tree.root,
        merkle_proof,
        final_hidden,
        token_id,
    }
}

/// Reconstruct KV prefix for audit: collect K/V from tokens 0..=token_idx for each challenged layer.
///
/// Uses `token_kvs` (pre-computed requantized K/V) when available. Falls back to
/// `requantize(&lt.k)` from `all_layers` otherwise.
fn reconstruct_kv_prefix(
    all_layers: &[Vec<LayerTrace>],
    token_idx: usize,
    layer_indices: &[usize],
    token_kvs: Option<&TokenKvs>,
) -> (Vec<Vec<Vec<i8>>>, Vec<Vec<Vec<i8>>>) {
    let mut kv_k_prefix = Vec::with_capacity(layer_indices.len());
    let mut kv_v_prefix = Vec::with_capacity(layer_indices.len());

    for &l in layer_indices {
        let mut k_for_layer = Vec::with_capacity(token_idx + 1);
        let mut v_for_layer = Vec::with_capacity(token_idx + 1);

        for t in 0..=token_idx {
            if let Some(kvs) = token_kvs {
                let (ref k, ref v) = kvs[t][l];
                k_for_layer.push(k.clone());
                v_for_layer.push(v.clone());
            } else {
                k_for_layer.push(requantize(&all_layers[t][l].k));
                v_for_layer.push(requantize(&all_layers[t][l].v));
            }
        }

        kv_k_prefix.push(k_for_layer);
        kv_v_prefix.push(v_for_layer);
    }

    (kv_k_prefix, kv_v_prefix)
}

/// Build a `StreamingKvVerifier` from full trace data.
///
/// Computes per-token KV Merkle roots from the requantized K/V across all layers.
pub fn build_streaming_kv_verifier(
    all_traces: &[Vec<LayerTrace>],
) -> verilm_core::streaming::StreamingKvVerifier {
    use verilm_core::streaming;

    let mut verifier = streaming::StreamingKvVerifier::new();
    for (t, token_layers) in all_traces.iter().enumerate() {
        let k_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.k)).collect();
        let v_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.v)).collect();
        verifier.ingest(streaming::kv_layer_root(t as u32, &k_per_layer, &v_per_layer));
    }
    verifier
}

/// Verify a batch proof: check each opened trace, IO commitment, and cross-token chains.
pub fn verify_batch(
    key: &VerifierKey,
    proof: &BatchProof,
    expected_challenges: &[u32],
) -> (bool, Vec<String>) {
    verify_batch_with_policy(key, proof, expected_challenges, &VerificationPolicy::default())
}

/// Verify a batch proof with explicit verification policy.
pub fn verify_batch_with_policy(
    key: &VerifierKey,
    proof: &BatchProof,
    expected_challenges: &[u32],
    policy: &VerificationPolicy,
) -> (bool, Vec<String>) {
    let mut failures = Vec::new();

    if proof.traces.len() != expected_challenges.len() {
        failures.push(format!(
            "expected {} challenged traces, got {}",
            expected_challenges.len(),
            proof.traces.len()
        ));
        return (false, failures);
    }

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
            _ => {}
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
            _ => {}
        }
    }

    // Version consistency checks
    if proof.commitment.version == CommitmentVersion::V2
        || proof.commitment.version == CommitmentVersion::V3
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

    // V3: seed commitment verification
    if proof.commitment.version == CommitmentVersion::V3 {
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
        if proof.commitment.prompt_hash.is_none() {
            failures.push("V3 commitment is missing prompt_hash".into());
        }

        // V3: all traces must have prev_io_hash
        for trace in &proof.traces {
            if trace.prev_io_hash.is_none() {
                failures.push(format!(
                    "token {}: V3 trace is missing prev_io_hash for chain verification",
                    trace.token_index
                ));
            }
        }
    }

    // Index opened traces by token_index for cross-token lookups
    let mut opened_by_index: std::collections::BTreeMap<u32, &TokenTrace> =
        std::collections::BTreeMap::new();

    // For V3, build IO hash map using prev_io_hash from each trace.
    let mut io_hash_map: std::collections::BTreeMap<u32, [u8; 32]> = std::collections::BTreeMap::new();
    if proof.commitment.version == CommitmentVersion::V3 {
        let mut sorted: Vec<&TokenTrace> = proof.traces.iter().collect();
        sorted.sort_by_key(|t| t.token_index);

        for trace in &sorted {
            let first_input = &trace.layers[0].x_attn;
            let last_output = &trace.layers.last().unwrap().ffn_out;

            if let (Some(tid), Some(claimed_prev)) = (trace.token_id, trace.prev_io_hash) {
                // Cross-check: if previous token is also opened, claimed prev must match
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

    for (i, trace) in proof.traces.iter().enumerate() {
        // Check token index matches challenge
        if trace.token_index != expected_challenges[i] {
            failures.push(format!(
                "trace {} has token_index={}, expected {}",
                i, trace.token_index, expected_challenges[i]
            ));
            continue;
        }

        // Check merkle proof leaf_index matches token_index
        if trace.merkle_proof.leaf_index != trace.token_index {
            failures.push(format!(
                "token {}: merkle_proof.leaf_index={} doesn't match token_index",
                trace.token_index, trace.merkle_proof.leaf_index
            ));
        }

        // Check merkle_root matches commitment
        if trace.merkle_root != proof.commitment.merkle_root {
            failures.push(format!(
                "token {}: merkle_root doesn't match commitment",
                trace.token_index
            ));
        }

        // Verify IO proof
        let first_input = &trace.layers[0].x_attn;
        let last_output = &trace.layers.last().unwrap().ffn_out;

        let io = if proof.commitment.version == CommitmentVersion::V3 {
            if let Some(&precomputed) = io_hash_map.get(&trace.token_index) {
                precomputed
            } else {
                merkle::io_hash(first_input, last_output, &merkle::io_binding(trace.token_id, None))
            }
        } else {
            merkle::io_hash(first_input, last_output, &merkle::io_binding(trace.token_id, None))
        };

        if !merkle::verify(&proof.commitment.io_root, &io, &trace.io_proof) {
            failures.push(format!(
                "token {}: IO proof verification failed",
                trace.token_index
            ));
        }

        // Run full single-token verification
        let (passed, token_failures) = verify_trace(key, trace);
        if !passed {
            for f in token_failures {
                failures.push(format!("token {}: {}", trace.token_index, f));
            }
        }

        opened_by_index.insert(trace.token_index, trace);
    }

    // Cross-token chain: for consecutive opened tokens, verify input/output chain.
    // Token t+1's first-layer input must equal requantize(token t's last-layer output).
    for (&idx, &trace) in &opened_by_index {
        if idx == 0 { continue; }
        if let Some(&prev_trace) = opened_by_index.get(&(idx - 1)) {
            let expected_input = requantize(&prev_trace.layers.last().unwrap().ffn_out);
            if trace.layers[0].x_attn != expected_input {
                failures.push(format!(
                    "token {}->{}: cross-token chain failed (input != requant(prev output))",
                    idx - 1, idx
                ));
            }
        }
    }

    // KV provenance chain: verify prev_kv_hash cross-checks for consecutive opened tokens.
    if proof.commitment.kv_chain_root.is_some() {
        // Recompute KV chain hash for each opened token and cross-check consecutives
        let mut kv_hashes: std::collections::BTreeMap<u32, [u8; 32]> = std::collections::BTreeMap::new();
        for (&pos, &trace) in &opened_by_index {
            if let Some(claimed_prev) = trace.prev_kv_hash {
                let k_per_layer: Vec<Vec<i8>> = trace.layers.iter()
                    .map(|lt| requantize(&lt.k))
                    .collect();
                let v_per_layer: Vec<Vec<i8>> = trace.layers.iter()
                    .map(|lt| requantize(&lt.v))
                    .collect();
                let kv = merkle::kv_chain_hash(&claimed_prev, &k_per_layer, &v_per_layer, pos);
                kv_hashes.insert(pos, kv);

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
    }

    let passed = failures.is_empty();
    (passed, failures)
}

/// Verify a token trace against a verifier key.
/// Returns (passed, list of failure descriptions).
pub fn verify_trace(key: &VerifierKey, trace: &TokenTrace) -> (bool, Vec<String>) {
    let mut failures = Vec::new();

    // Step 1: Merkle commitment binding (direct hash, no bincode)
    let leaf_hash = merkle::hash_trace_direct(&trace.layers, trace.final_hidden.as_deref());
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

            if !verilm_core::freivalds::check(v, input, r, output) {
                failures.push(format!(
                    "layer {} {:?}: Freivalds check failed",
                    layer_idx, mt
                ));
            }
        }

        // Step 3: SiLU verification — h == SiLU(requant(g)) * requant(u) via LUT
        let expected_h = verilm_core::silu::compute_h_unit_scale(&lt.g, &lt.u);
        if lt.h != expected_h {
            failures.push(format!(
                "layer {}: SiLU check failed (h mismatch)",
                layer_idx
            ));
        }

        // Step 5a: Intra-layer chain — x_ffn == requantize(attn_out)
        let expected_x_ffn = requantize(&lt.attn_out);
        if lt.x_ffn != expected_x_ffn {
            failures.push(format!(
                "layer {}: chain check failed (x_ffn != requantize(attn_out))",
                layer_idx
            ));
        }

        // Step 5b: Cross-layer chain — x_attn[i+1] == requantize(ffn_out[i])
        if layer_idx + 1 < trace.layers.len() {
            let expected_next_input = requantize(&lt.ffn_out);
            if trace.layers[layer_idx + 1].x_attn != expected_next_input {
                failures.push(format!(
                    "layer {}->{}: chain check failed (next x_attn != requantize(ffn_out))",
                    layer_idx,
                    layer_idx + 1
                ));
            }
        }
    }

    let passed = failures.is_empty();
    (passed, failures)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_minimal_captures(n_layers: usize, n_tokens: usize, hidden: usize) -> Vec<MinimalCaptureEntry> {
        let mut captures = Vec::new();
        for t in 0..n_tokens {
            for l in 0..n_layers {
                captures.push(MinimalCaptureEntry {
                    a_i8: vec![(t * n_layers + l) as i8; hidden],
                    scale_x_attn: 0.1 * (l + 1) as f32,
                    scale_a: 0.2 * (l + 1) as f32,
                    scale_x_ffn: 0.3 * (l + 1) as f32,
                    scale_h: 0.4 * (l + 1) as f32,
                });
            }
        }
        captures
    }

    #[test]
    fn test_build_retained_from_captures() {
        let n_layers = 2;
        let captures = make_minimal_captures(n_layers, 3, 8);
        let fwd_batch_sizes = vec![1, 1, 1];

        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);
        assert_eq!(retained.len(), 3);
        assert_eq!(retained[0].layers.len(), n_layers);
        assert_eq!(retained[0].layers[0].a.len(), 8);
        assert_eq!(retained[0].layers[0].scale_a, 0.2);
        assert_eq!(retained[0].layers[1].scale_a, 0.4);
    }

    #[test]
    fn test_build_retained_batched_prefill() {
        let n_layers = 2;
        let hidden = 4;
        // Batched prefill: 3 tokens in one forward pass.
        let mut captures = Vec::new();
        for l in 0..n_layers {
            captures.push(MinimalCaptureEntry {
                a_i8: vec![(l + 1) as i8; 3 * hidden],
                scale_x_attn: 0.1,
                scale_a: 0.2,
                scale_x_ffn: 0.3,
                scale_h: 0.4,
            });
        }
        let fwd_batch_sizes = vec![3];
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);
        assert_eq!(retained.len(), 3);
        for t in 0..3 {
            assert_eq!(retained[t].layers.len(), n_layers);
            assert_eq!(retained[t].layers[0].a.len(), hidden);
        }
    }

    #[test]
    fn test_commit_minimal_roundtrip() {
        let n_layers = 2;
        let captures = make_minimal_captures(n_layers, 3, 8);
        let fwd_batch_sizes = vec![1, 1, 1];
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

        let params = FullBindingParams {
            token_ids: &[100, 200, 300],
            prompt: b"test prompt",
            sampling_seed: [42u8; 32],
            manifest: None,
        };

        let (commitment, state) = commit_minimal(retained, &params);

        assert_eq!(commitment.version, CommitmentVersion::V4);
        assert_eq!(commitment.n_tokens, 3);
        assert!(commitment.prompt_hash.is_some());
        assert!(commitment.seed_commitment.is_some());
        assert!(commitment.kv_chain_root.is_none());
        assert_eq!(state.token_ids, vec![100, 200, 300]);
        assert_eq!(state.all_retained.len(), 3);

        // IO chain: each hash depends on previous.
        assert_ne!(state.io_hashes[0], state.io_hashes[1]);
        assert_ne!(state.io_hashes[1], state.io_hashes[2]);
    }

    #[test]
    fn test_commit_minimal_io_uses_leaf_hash() {
        let n_layers = 1;
        let captures = make_minimal_captures(n_layers, 2, 4);
        let fwd_batch_sizes = vec![1, 1];
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

        let params = FullBindingParams {
            token_ids: &[10, 20],
            prompt: b"p",
            sampling_seed: [0u8; 32],
            manifest: None,
        };

        let (_, state) = commit_minimal(retained.clone(), &params);

        // Verify IO chain uses leaf hashes, not ad hoc features.
        let leaf0 = merkle::hash_retained_state_direct(&retained[0]);
        let leaf1 = merkle::hash_retained_state_direct(&retained[1]);

        let expected_io0 = merkle::io_hash_v4(leaf0, 10, [0u8; 32]);
        let expected_io1 = merkle::io_hash_v4(leaf1, 20, expected_io0);

        assert_eq!(state.io_hashes[0], expected_io0);
        assert_eq!(state.io_hashes[1], expected_io1);
    }

    #[test]
    fn test_commit_minimal_deterministic() {
        let n_layers = 2;
        let captures = make_minimal_captures(n_layers, 2, 8);
        let fwd_batch_sizes = vec![1, 1];

        let params = FullBindingParams {
            token_ids: &[1, 2],
            prompt: b"hello",
            sampling_seed: [7u8; 32],
            manifest: None,
        };

        let retained1 = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);
        let retained2 = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

        let (c1, _) = commit_minimal(retained1, &params);
        let (c2, _) = commit_minimal(retained2, &params);

        assert_eq!(c1.merkle_root, c2.merkle_root);
        assert_eq!(c1.io_root, c2.io_root);
    }
}
