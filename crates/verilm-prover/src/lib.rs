//! Prover engine: commitment generation and proof opening for the VeriLM protocol.

use rayon::prelude::*;
use verilm_core::bridge_requantize;
use verilm_core::constants::MatrixType;
use verilm_core::matmul::matmul_i32;
use verilm_core::merkle;
use verilm_core::requantize;
use verilm_core::rmsnorm::{bridge_residual_rmsnorm, dequant_add_residual, rmsnorm_f64_input, quantize_f64_to_i8};
use verilm_core::types::{
    AuditChallenge, AuditResponse, BatchCommitment, BatchProof, BridgeParams, CommitmentVersion,
    DeploymentManifest, LayerTrace, RetainedLayerState, RetainedTokenState,
    ShellLayerOpening, ShellTokenOpening, ShellWeights,
    TokenTrace, V4AuditResponse, VerificationPolicy, VerifierKey,
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
    /// Deployment manifest, stored for inclusion in V4 audit responses.
    pub manifest: Option<DeploymentManifest>,
    /// Per-token captured final residual (pre-final-norm) from GPU inference.
    /// Used at audit time for exact LM-head verification.
    pub final_residuals: Option<Vec<Vec<f32>>>,
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
    final_residuals: Option<Vec<Vec<f32>>>,
) -> (BatchCommitment, MinimalBatchState) {
    assert_eq!(
        all_retained.len(),
        params.token_ids.len(),
        "retained state count must match token_ids"
    );

    // Trace tree: hash retained state per token (parallel).
    // When final_residuals is present, bind each into the leaf hash.
    let trace_leaves: Vec<[u8; 32]> = all_retained
        .par_iter()
        .enumerate()
        .map(|(i, rs)| {
            let fr = final_residuals.as_ref().and_then(|frs| frs.get(i)).map(|v| v.as_slice());
            merkle::hash_retained_with_residual(rs, fr)
        })
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
        manifest: params.manifest.cloned(),
        final_residuals,
    };

    (commitment, state)
}

// ── Packed commit path ──────────────────────────────────────────────
//
// Eliminates per-token Vec<RetainedLayerState> materialization at commit
// time. All capture data stays in contiguous packed buffers; individual
// tokens are hashed directly from buffer slices (parallel), and only
// the challenged token is reconstructed at audit time.

/// Precomputed index mapping global token index → (fwd_pass, batch_position).
#[derive(Debug, Clone)]
pub struct TokenIndex {
    /// cumulative[f] = sum of fwd_batch_sizes[0..f].
    /// cumulative[0] = 0, cumulative[n_fwd] = n_tokens.
    pub cumulative: Vec<usize>,
}

impl TokenIndex {
    pub fn new(fwd_batch_sizes: &[usize]) -> Self {
        let mut cumulative = Vec::with_capacity(fwd_batch_sizes.len() + 1);
        cumulative.push(0);
        for &bs in fwd_batch_sizes {
            cumulative.push(cumulative.last().unwrap() + bs);
        }
        TokenIndex { cumulative }
    }

    /// Returns (fwd_pass_index, position_within_batch).
    pub fn locate(&self, token_global: usize) -> (usize, usize) {
        // Binary search for the fwd pass containing this token.
        let fwd = match self.cumulative.binary_search(&token_global) {
            Ok(f) => f,             // token is first in fwd pass f
            Err(f) => f - 1,        // token is in the middle of fwd pass f-1
        };
        let pos = token_global - self.cumulative[fwd];
        (fwd, pos)
    }

    pub fn n_tokens(&self) -> usize {
        *self.cumulative.last().unwrap_or(&0)
    }

    pub fn n_fwd(&self) -> usize {
        self.cumulative.len().saturating_sub(1)
    }
}

/// Packed batch state: stores capture data in contiguous buffers.
///
/// Drop-in replacement for `MinimalBatchState` that avoids per-token
/// Vec allocations. Individual tokens are reconstructed on demand
/// (audit time only).
#[derive(Debug, Clone)]
pub struct PackedBatchState {
    pub retained_tree: merkle::MerkleTree,
    pub io_tree: merkle::MerkleTree,
    // ── packed capture data ──
    /// Contiguous i8 bytes, layout: fwd-major × layer-major × batch-row-major.
    /// For fwd f, layer l, batch pos b: offset = cum[f]*n_layers*hidden_dim + l*batch_sz*hidden_dim + b*hidden_dim.
    pub packed_a: Vec<u8>,
    /// 4 scales per (fwd, layer): [scale_x_attn, scale_a, scale_x_ffn, scale_h].
    pub packed_scales: Vec<f32>,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub fwd_batch_sizes: Vec<usize>,
    pub token_index: TokenIndex,
    // ── binding data (same as MinimalBatchState) ──
    pub manifest_hash: Option<[u8; 32]>,
    pub token_ids: Vec<u32>,
    pub prompt_hash: [u8; 32],
    pub seed_commitment: [u8; 32],
    pub revealed_seed: [u8; 32],
    pub io_hashes: Vec<[u8; 32]>,
    pub manifest: Option<DeploymentManifest>,
    /// Contiguous f32 bytes for per-token final residuals (pre-final-norm).
    /// Layout: token-major, each entry is `final_res_dim` f32 values.
    pub packed_final_res: Option<Vec<u8>>,
    pub final_res_dim: usize,
}

impl PackedBatchState {
    /// Hash one token directly from packed buffers (no intermediate alloc).
    ///
    /// Produces the same hash as `merkle::hash_retained_with_residual()`
    /// on the equivalent `RetainedTokenState`.
    pub fn hash_token(&self, token_global: usize) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let (fwd, pos) = self.token_index.locate(token_global);
        let batch_sz = self.fwd_batch_sizes[fwd];
        let hd = self.hidden_dim;

        // Base byte offset for this fwd pass in packed_a.
        let fwd_byte_offset = self.token_index.cumulative[fwd] * self.n_layers * hd;

        let mut hasher = Sha256::new();
        hasher.update(b"vi-retained-v1");

        for l in 0..self.n_layers {
            // a slice for (fwd, layer, batch_pos)
            let layer_base = fwd_byte_offset + l * batch_sz * hd;
            let a_start = layer_base + pos * hd;
            hasher.update(&self.packed_a[a_start..a_start + hd]);

            // Scales: same order as hash_retained_state_direct.
            let s_base = (fwd * self.n_layers + l) * 4;
            hasher.update(self.packed_scales[s_base + 1].to_le_bytes()); // scale_a
            hasher.update(self.packed_scales[s_base + 0].to_le_bytes()); // scale_x_attn
            hasher.update(self.packed_scales[s_base + 2].to_le_bytes()); // scale_x_ffn
            hasher.update(self.packed_scales[s_base + 3].to_le_bytes()); // scale_h
        }

        let base: [u8; 32] = hasher.finalize().into();

        // Bind final residual if present.
        match &self.packed_final_res {
            Some(fr_buf) => {
                let fr_bytes = self.final_res_dim * 4; // f32
                let fr_start = token_global * fr_bytes;
                let mut h2 = Sha256::new();
                h2.update(b"vi-retained-fr-v1");
                h2.update(base);
                h2.update(&fr_buf[fr_start..fr_start + fr_bytes]);
                h2.finalize().into()
            }
            None => base,
        }
    }

    /// Reconstruct one token's `RetainedTokenState` from packed buffers.
    ///
    /// Used at audit time for the challenged token (passed to
    /// `compute_shell_opening`).
    pub fn extract_token(&self, token_global: usize) -> RetainedTokenState {
        let (fwd, pos) = self.token_index.locate(token_global);
        let batch_sz = self.fwd_batch_sizes[fwd];
        let hd = self.hidden_dim;
        let fwd_byte_offset = self.token_index.cumulative[fwd] * self.n_layers * hd;

        let mut layers = Vec::with_capacity(self.n_layers);
        for l in 0..self.n_layers {
            let layer_base = fwd_byte_offset + l * batch_sz * hd;
            let a_start = layer_base + pos * hd;
            let a_bytes = &self.packed_a[a_start..a_start + hd];
            // Reinterpret u8 → i8 (identical memory layout).
            let a: Vec<i8> = a_bytes.iter().map(|&b| b as i8).collect();

            let s_base = (fwd * self.n_layers + l) * 4;
            layers.push(RetainedLayerState {
                a,
                scale_x_attn: self.packed_scales[s_base],
                scale_a: self.packed_scales[s_base + 1],
                scale_x_ffn: self.packed_scales[s_base + 2],
                scale_h: self.packed_scales[s_base + 3],
            });
        }
        RetainedTokenState { layers }
    }

    /// Extract final residual for one token as `Vec<f32>`.
    pub fn extract_final_residual(&self, token_global: usize) -> Option<Vec<f32>> {
        self.packed_final_res.as_ref().map(|fr_buf| {
            let fr_bytes = self.final_res_dim * 4;
            let fr_start = token_global * fr_bytes;
            let raw = &fr_buf[fr_start..fr_start + fr_bytes];
            // Reinterpret &[u8] → &[f32] (native-endian).
            let mut out = vec![0f32; self.final_res_dim];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    out.as_mut_ptr() as *mut u8,
                    fr_bytes,
                );
            }
            out
        })
    }

    pub fn n_tokens(&self) -> usize {
        self.token_index.n_tokens()
    }
}

/// V4 commit from packed buffers: hash directly from contiguous capture data.
///
/// Produces identical Merkle roots and IO chain as `commit_minimal()` on
/// the equivalent `Vec<RetainedTokenState>`, but without materializing
/// per-token Vecs.
pub fn commit_minimal_packed(
    packed_a: Vec<u8>,
    packed_scales: Vec<f32>,
    n_layers: usize,
    hidden_dim: usize,
    fwd_batch_sizes: Vec<usize>,
    params: &FullBindingParams,
    packed_final_res: Option<Vec<u8>>,
    final_res_dim: usize,
) -> (BatchCommitment, PackedBatchState) {
    let idx = TokenIndex::new(&fwd_batch_sizes);
    let n_tokens = idx.n_tokens();

    assert_eq!(n_tokens, params.token_ids.len(), "token count must match token_ids");

    // Validate buffer sizes.
    let expected_a_bytes: usize = fwd_batch_sizes.iter().sum::<usize>() * n_layers * hidden_dim;
    assert_eq!(
        packed_a.len(), expected_a_bytes,
        "packed_a length ({}) != expected ({} tokens × {} layers × {} dim)",
        packed_a.len(), n_tokens, n_layers, hidden_dim
    );
    let expected_scales = fwd_batch_sizes.len() * n_layers * 4;
    assert_eq!(
        packed_scales.len(), expected_scales,
        "packed_scales length ({}) != expected ({} fwd × {} layers × 4)",
        packed_scales.len(), fwd_batch_sizes.len(), n_layers
    );
    if let Some(ref fr) = packed_final_res {
        let expected_fr = n_tokens * final_res_dim * 4;
        assert_eq!(
            fr.len(), expected_fr,
            "packed_final_res length ({}) != expected ({} tokens × {} dim × 4)",
            fr.len(), n_tokens, final_res_dim
        );
    }

    // Build temporary state for hashing (borrows only, no ownership transfer).
    let dummy_tree = merkle::MerkleTree { root: [0u8; 32], nodes: vec![], n_leaves: 0, padded_size: 0 };
    let state = PackedBatchState {
        retained_tree: dummy_tree.clone(),
        io_tree: dummy_tree,
        packed_a,
        packed_scales,
        n_layers,
        hidden_dim,
        fwd_batch_sizes: fwd_batch_sizes.clone(),
        token_index: idx,
        manifest_hash: None,
        token_ids: Vec::new(),
        prompt_hash: [0u8; 32],
        seed_commitment: [0u8; 32],
        revealed_seed: [0u8; 32],
        io_hashes: Vec::new(),
        manifest: None,
        packed_final_res,
        final_res_dim,
    };

    // Trace tree: hash each token in parallel.
    let trace_leaves: Vec<[u8; 32]> = (0..n_tokens)
        .into_par_iter()
        .map(|t| state.hash_token(t))
        .collect();

    // IO tree: sequential chain.
    let mut io_leaves = Vec::with_capacity(n_tokens);
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
        n_tokens: n_tokens as u32,
        manifest_hash,
        version: CommitmentVersion::V4,
        prompt_hash: Some(merkle::hash_prompt(params.prompt)),
        seed_commitment: Some(merkle::hash_seed(&params.sampling_seed)),
        kv_chain_root: None,
    };

    // Reconstitute final state with real trees.
    let final_state = PackedBatchState {
        retained_tree: trace_tree,
        io_tree,
        packed_a: state.packed_a,
        packed_scales: state.packed_scales,
        n_layers,
        hidden_dim,
        fwd_batch_sizes,
        token_index: state.token_index,
        manifest_hash,
        token_ids: params.token_ids.to_vec(),
        prompt_hash: merkle::hash_prompt(params.prompt),
        seed_commitment: merkle::hash_seed(&params.sampling_seed),
        revealed_seed: params.sampling_seed,
        io_hashes: io_leaves,
        manifest: params.manifest.cloned(),
        packed_final_res: state.packed_final_res,
        final_res_dim,
    };

    (commitment, final_state)
}

/// V4 audit from packed state: structural proofs + shell opening.
pub fn open_v4_packed(
    state: &PackedBatchState,
    token_index: u32,
    weights: &dyn ShellWeights,
    cfg: &verilm_core::constants::ModelConfig,
    weight_scales: &[Vec<f32>],
    bridge: Option<&BridgeParams>,
    layer_filter: Option<&[usize]>,
) -> V4AuditResponse {
    let i = token_index as usize;
    assert!(i < state.n_tokens(), "token_index out of range");

    let commitment = BatchCommitment {
        merkle_root: state.retained_tree.root,
        io_root: state.io_tree.root,
        n_tokens: state.n_tokens() as u32,
        manifest_hash: state.manifest_hash,
        version: CommitmentVersion::V4,
        prompt_hash: Some(state.prompt_hash),
        seed_commitment: Some(state.seed_commitment),
        kv_chain_root: None,
    };

    // Prefix leaf hashes: hash from packed buffers (no materialization).
    let mut prefix_leaf_hashes = Vec::with_capacity(i);
    let mut prefix_merkle_proofs = Vec::with_capacity(i);
    let mut prefix_token_ids = Vec::with_capacity(i);
    for j in 0..i {
        prefix_leaf_hashes.push(state.hash_token(j));
        prefix_merkle_proofs.push(merkle::prove(&state.retained_tree, j));
        prefix_token_ids.push(state.token_ids[j]);
    }

    let prev_io_hash = if i == 0 {
        [0u8; 32]
    } else {
        state.io_hashes[i - 1]
    };

    // Reconstruct ONLY the challenged token for shell computation.
    let retained_token = state.extract_token(i);

    let mut shell = compute_shell_opening(
        &retained_token, weights, cfg, weight_scales, bridge, layer_filter,
    );
    shell.final_residual = state.extract_final_residual(i);

    V4AuditResponse {
        token_index,
        retained: retained_token,
        merkle_proof: merkle::prove(&state.retained_tree, i),
        io_proof: merkle::prove(&state.io_tree, i),
        token_id: state.token_ids[i],
        prev_io_hash,
        prefix_leaf_hashes,
        prefix_merkle_proofs,
        prefix_token_ids,
        commitment,
        revealed_seed: state.revealed_seed,
        shell_opening: Some(shell),
        manifest: state.manifest.clone(),
    }
}

/// Compute shell opening for a single token from retained state + public weights.
///
/// The prover reconstructs all matmul intermediates that the verifier
/// will check with key-only Freivalds. Bridge values (requantize, SiLU)
/// are not sent — the verifier derives them from the i32 accumulators
/// and the committed scales.
///
/// `weight_scales` is per-layer, per-MatrixType quantization scales.
/// Empty for native INT8 / toy model (bridges fall back to clamp).
///
/// When `bridge` is `Some`, uses the paper-correct full bridge:
///   dequant → residual += → RMSNorm → quantize
/// When `None`, falls back to simplified bridge.
pub fn compute_shell_opening(
    retained: &RetainedTokenState,
    weights: &dyn ShellWeights,
    cfg: &verilm_core::constants::ModelConfig,
    weight_scales: &[Vec<f32>],
    bridge: Option<&BridgeParams>,
    layer_filter: Option<&[usize]>,
) -> ShellTokenOpening {
    // Determine how many layers to compute. For a contiguous prefix filter
    // 0..=L_max we iterate through L_max+1 layers (bridge is sequential).
    let max_layer = layer_filter
        .map(|f| f.iter().copied().max().map_or(0, |m| m + 1))
        .unwrap_or(retained.layers.len());
    let max_layer = max_layer.min(retained.layers.len());

    let mut layers = Vec::new();

    let ws = |layer: usize, mt: MatrixType| -> f32 {
        if weight_scales.is_empty() || layer >= weight_scales.len() {
            return 0.0;
        }
        let idx = MatrixType::ALL.iter().position(|&m| m == mt).unwrap();
        weight_scales[layer][idx]
    };

    // Full bridge: init residual from embedding and derive x_attn_0
    let (mut residual, mut x_attn) = if let Some(b) = bridge {
        let res: Vec<f64> = b.initial_residual.iter().map(|&v| v as f64).collect();
        let normed = rmsnorm_f64_input(&res, &b.rmsnorm_attn_weights[0], b.rmsnorm_eps);
        let xa = quantize_f64_to_i8(&normed, retained.layers[0].scale_x_attn as f64);
        (Some(res), Some(xa))
    } else {
        (None, None)
    };

    for (layer_idx, rs) in retained.layers.iter().enumerate().take(max_layer) {
        let a = &rs.a;

        // QKV: when x_attn is known (full bridge: all layers; toy: layers > 0)
        let (q, k, v) = if let Some(ref xa) = x_attn {
            (
                Some(matmul_i32(weights.weight(layer_idx, MatrixType::Wq), xa, cfg.hidden_dim, cfg.hidden_dim)),
                Some(matmul_i32(weights.weight(layer_idx, MatrixType::Wk), xa, cfg.kv_dim, cfg.hidden_dim)),
                Some(matmul_i32(weights.weight(layer_idx, MatrixType::Wv), xa, cfg.kv_dim, cfg.hidden_dim)),
            )
        } else {
            (None, None, None)
        };

        // W_o @ a
        let attn_out = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wo), a, cfg.hidden_dim, cfg.hidden_dim,
        );

        // Post-attention bridge: derive x_ffn
        let x_ffn = if let (Some(ref mut res), Some(b)) = (&mut residual, bridge) {
            bridge_residual_rmsnorm(
                &attn_out, ws(layer_idx, MatrixType::Wo), rs.scale_a,
                res, &b.rmsnorm_ffn_weights[layer_idx], b.rmsnorm_eps, rs.scale_x_ffn,
            )
        } else {
            bridge_requantize(&attn_out, ws(layer_idx, MatrixType::Wo), rs.scale_a, rs.scale_x_ffn)
        };

        // W_g, W_u @ x_ffn
        let g = matmul_i32(weights.weight(layer_idx, MatrixType::Wg), &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(weights.weight(layer_idx, MatrixType::Wu), &x_ffn, cfg.ffn_dim, cfg.hidden_dim);

        // SiLU bridge (unchanged — no residual involved)
        let h = verilm_core::silu::compute_h_scaled(
            &g, &u,
            ws(layer_idx, MatrixType::Wg),
            ws(layer_idx, MatrixType::Wu),
            rs.scale_x_ffn,
            rs.scale_h,
        );

        // W_d @ h
        let ffn_out = matmul_i32(weights.weight(layer_idx, MatrixType::Wd), &h, cfg.hidden_dim, cfg.ffn_dim);

        // Post-FFN bridge: derive x_attn for next layer
        let next_scale_x_attn = retained.layers
            .get(layer_idx + 1)
            .map(|r| r.scale_x_attn)
            .unwrap_or(1.0);

        x_attn = if let (Some(ref mut res), Some(b)) = (&mut residual, bridge) {
            if layer_idx + 1 < b.rmsnorm_attn_weights.len() {
                Some(bridge_residual_rmsnorm(
                    &ffn_out, ws(layer_idx, MatrixType::Wd), rs.scale_h,
                    res, &b.rmsnorm_attn_weights[layer_idx + 1], b.rmsnorm_eps,
                    next_scale_x_attn,
                ))
            } else {
                // Last layer: update residual, simplified bridge for unused x_attn
                dequant_add_residual(&ffn_out, ws(layer_idx, MatrixType::Wd), rs.scale_h, res);
                Some(bridge_requantize(
                    &ffn_out, ws(layer_idx, MatrixType::Wd), rs.scale_h, next_scale_x_attn,
                ))
            }
        } else {
            Some(bridge_requantize(
                &ffn_out, ws(layer_idx, MatrixType::Wd), rs.scale_h, next_scale_x_attn,
            ))
        };

        // Only include layers in the filter
        if layer_filter.map_or(true, |f| f.contains(&layer_idx)) {
            layers.push(ShellLayerOpening { attn_out, g, u, ffn_out, q, k, v });
        }
    }

    ShellTokenOpening {
        layers,
        layer_indices: layer_filter.map(|f| f.to_vec()),
        initial_residual: bridge.map(|b| b.initial_residual.to_vec()),
        embedding_proof: bridge.and_then(|b| b.embedding_proof.clone()),
        final_residual: None, // Set by open_v4 from captured GPU state
    }
}

/// V4 open: open a challenged token with prover-computed shell intermediates.
///
/// This is the protocol path: the prover reconstructs shell intermediates
/// from retained `a` + weights. The verifier checks with key-only Freivalds.
///
/// `weight_scales` is per-layer, per-MatrixType quantization scales.
/// Empty for native INT8 / toy model.
///
/// When `bridge` is `Some`, uses full dequant→residual→RMSNorm→quantize chain.
/// When `None`, falls back to simplified bridge.
pub fn open_v4(
    state: &MinimalBatchState,
    token_index: u32,
    weights: &dyn ShellWeights,
    cfg: &verilm_core::constants::ModelConfig,
    weight_scales: &[Vec<f32>],
    bridge: Option<&BridgeParams>,
    layer_filter: Option<&[usize]>,
) -> V4AuditResponse {
    let mut response = open_v4_structural(state, token_index);
    let mut shell = compute_shell_opening(
        &state.all_retained[token_index as usize], weights, cfg, weight_scales, bridge,
        layer_filter,
    );
    // Attach captured final residual from GPU inference (if available).
    shell.final_residual = state.final_residuals
        .as_ref()
        .and_then(|frs| frs.get(token_index as usize))
        .cloned();
    response.shell_opening = Some(shell);
    response
}

/// V4 open: structural only (Merkle proofs + IO chain, no shell intermediates).
///
/// Use `open_v4` for the full protocol path with shell openings.
pub fn open_v4_structural(state: &MinimalBatchState, token_index: u32) -> V4AuditResponse {
    let i = token_index as usize;
    assert!(i < state.all_retained.len(), "token_index out of range");

    let commitment = BatchCommitment {
        merkle_root: state.retained_tree.root,
        io_root: state.io_tree.root,
        n_tokens: state.all_retained.len() as u32,
        manifest_hash: state.manifest_hash,
        version: CommitmentVersion::V4,
        prompt_hash: Some(state.prompt_hash),
        seed_commitment: Some(state.seed_commitment),
        kv_chain_root: None,
    };

    let mut prefix_leaf_hashes = Vec::with_capacity(i);
    let mut prefix_merkle_proofs = Vec::with_capacity(i);
    let mut prefix_token_ids = Vec::with_capacity(i);
    for j in 0..i {
        let fr = state.final_residuals.as_ref().and_then(|frs| frs.get(j)).map(|v| v.as_slice());
        prefix_leaf_hashes.push(merkle::hash_retained_with_residual(&state.all_retained[j], fr));
        prefix_merkle_proofs.push(merkle::prove(&state.retained_tree, j));
        prefix_token_ids.push(state.token_ids[j]);
    }

    let prev_io_hash = if i == 0 {
        [0u8; 32]
    } else {
        state.io_hashes[i - 1]
    };

    V4AuditResponse {
        token_index,
        retained: state.all_retained[i].clone(),
        merkle_proof: merkle::prove(&state.retained_tree, i),
        io_proof: merkle::prove(&state.io_tree, i),
        token_id: state.token_ids[i],
        prev_io_hash,
        prefix_leaf_hashes,
        prefix_merkle_proofs,
        prefix_token_ids,
        commitment,
        revealed_seed: state.revealed_seed,
        shell_opening: None,
        manifest: state.manifest.clone(),
    }
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

        let (commitment, state) = commit_minimal(retained, &params, None);

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

        let (_, state) = commit_minimal(retained.clone(), &params, None);

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

        let (c1, _) = commit_minimal(retained1, &params, None);
        let (c2, _) = commit_minimal(retained2, &params, None);

        assert_eq!(c1.merkle_root, c2.merkle_root);
        assert_eq!(c1.io_root, c2.io_root);
    }

    #[test]
    fn test_open_v4_returns_prefix_and_proofs() {
        let n_layers = 2;
        let captures = make_minimal_captures(n_layers, 4, 8);
        let fwd_batch_sizes = vec![1, 1, 1, 1];
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

        let params = FullBindingParams {
            token_ids: &[10, 20, 30, 40],
            prompt: b"test",
            sampling_seed: [5u8; 32],
            manifest: None,
        };

        let (commitment, state) = commit_minimal(retained, &params, None);

        // Open token 2 — should include prefix tokens 0, 1.
        let response = open_v4_structural(&state, 2);
        assert_eq!(response.token_index, 2);
        assert_eq!(response.token_id, 30);
        assert_eq!(response.prefix_leaf_hashes.len(), 2);
        assert_eq!(response.prefix_merkle_proofs.len(), 2);
        assert_eq!(response.prefix_token_ids, vec![10, 20]);
        assert_eq!(response.commitment.merkle_root, commitment.merkle_root);

        // Verify challenged token's Merkle proof.
        let leaf_hash = merkle::hash_retained_state_direct(&response.retained);
        assert!(merkle::verify(
            &commitment.merkle_root,
            &leaf_hash,
            &response.merkle_proof,
        ));

        // Verify each prefix token's Merkle proof.
        for (j, prefix_leaf_hash) in response.prefix_leaf_hashes.iter().enumerate() {
            assert!(merkle::verify(
                &commitment.merkle_root,
                prefix_leaf_hash,
                &response.prefix_merkle_proofs[j],
            ));
        }

        // Verify IO chain.
        let leaf_hash_0 = response.prefix_leaf_hashes[0];
        let leaf_hash_1 = response.prefix_leaf_hashes[1];
        let io_0 = merkle::io_hash_v4(leaf_hash_0, 10, [0u8; 32]);
        let io_1 = merkle::io_hash_v4(leaf_hash_1, 20, io_0);
        assert_eq!(response.prev_io_hash, io_1);
    }

    #[test]
    fn test_open_v4_token_zero_has_empty_prefix() {
        let n_layers = 1;
        let captures = make_minimal_captures(n_layers, 2, 4);
        let fwd_batch_sizes = vec![1, 1];
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

        let params = FullBindingParams {
            token_ids: &[1, 2],
            prompt: b"p",
            sampling_seed: [0u8; 32],
            manifest: None,
        };

        let (_, state) = commit_minimal(retained, &params, None);
        let response = open_v4_structural(&state, 0);

        assert_eq!(response.prefix_leaf_hashes.len(), 0);
        assert_eq!(response.prefix_merkle_proofs.len(), 0);
        assert_eq!(response.prev_io_hash, [0u8; 32]);
    }

    // ── Packed commit equivalence tests ────────────────────────────────

    /// Build packed buffers from the same test data that make_minimal_captures uses.
    /// Layout matches what Python would produce: fwd-major × layer-major.
    fn make_packed_data(
        n_layers: usize, n_tokens: usize, hidden: usize, fwd_batch_sizes: &[usize],
    ) -> (Vec<u8>, Vec<f32>) {
        let mut packed_a = Vec::new();
        let mut packed_scales = Vec::new();
        let mut token_counter = 0;

        for &batch_sz in fwd_batch_sizes {
            for l in 0..n_layers {
                // Each layer in this fwd pass: batch_sz rows of `hidden` bytes.
                for b in 0..batch_sz {
                    let t = token_counter + b;
                    let val = ((t * n_layers + l) & 0xFF) as u8;
                    packed_a.extend(std::iter::repeat(val).take(hidden));
                }
                // 4 scales per (fwd, layer)
                packed_scales.push(0.1 * (l + 1) as f32);  // scale_x_attn
                packed_scales.push(0.2 * (l + 1) as f32);  // scale_a
                packed_scales.push(0.3 * (l + 1) as f32);  // scale_x_ffn
                packed_scales.push(0.4 * (l + 1) as f32);  // scale_h
            }
            token_counter += batch_sz;
        }
        assert_eq!(token_counter, n_tokens);
        (packed_a, packed_scales)
    }

    #[test]
    fn test_packed_commit_roots_match_unpacked() {
        let n_layers = 2;
        let hidden = 8;
        let n_tokens = 3;
        let fwd_batch_sizes = vec![1, 1, 1];

        // Unpacked path.
        let captures = make_minimal_captures(n_layers, n_tokens, hidden);
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);
        let params = FullBindingParams {
            token_ids: &[100, 200, 300],
            prompt: b"test prompt",
            sampling_seed: [42u8; 32],
            manifest: None,
        };
        let (commit_old, state_old) = commit_minimal(retained, &params, None);

        // Packed path.
        let (packed_a, packed_scales) = make_packed_data(n_layers, n_tokens, hidden, &fwd_batch_sizes);
        let (commit_new, state_new) = commit_minimal_packed(
            packed_a, packed_scales, n_layers, hidden, fwd_batch_sizes, &params, None, 0,
        );

        assert_eq!(commit_old.merkle_root, commit_new.merkle_root, "trace Merkle root mismatch");
        assert_eq!(commit_old.io_root, commit_new.io_root, "IO root mismatch");
        assert_eq!(commit_old.n_tokens, commit_new.n_tokens);
        assert_eq!(commit_old.prompt_hash, commit_new.prompt_hash);
        assert_eq!(commit_old.seed_commitment, commit_new.seed_commitment);
        assert_eq!(state_old.io_hashes, state_new.io_hashes, "IO chain hashes mismatch");
    }

    #[test]
    fn test_packed_commit_roots_match_batched_prefill() {
        // Batched prefill: 3 tokens in first fwd pass, then 2 decode steps.
        let n_layers = 2;
        let hidden = 4;
        let fwd_batch_sizes = vec![3, 1, 1];
        let n_tokens = 5;

        // Build captures in fwd-major order for unpacked path.
        let mut captures = Vec::new();
        let mut token_counter = 0;
        for &batch_sz in &fwd_batch_sizes {
            for l in 0..n_layers {
                let mut a_i8 = Vec::new();
                for b in 0..batch_sz {
                    let t = token_counter + b;
                    a_i8.extend(std::iter::repeat((t * n_layers + l) as i8).take(hidden));
                }
                captures.push(MinimalCaptureEntry {
                    a_i8,
                    scale_x_attn: 0.1 * (l + 1) as f32,
                    scale_a: 0.2 * (l + 1) as f32,
                    scale_x_ffn: 0.3 * (l + 1) as f32,
                    scale_h: 0.4 * (l + 1) as f32,
                });
            }
            token_counter += batch_sz;
        }
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

        let params = FullBindingParams {
            token_ids: &[10, 20, 30, 40, 50],
            prompt: b"batched",
            sampling_seed: [7u8; 32],
            manifest: None,
        };
        let (commit_old, state_old) = commit_minimal(retained, &params, None);

        // Packed path.
        let (packed_a, packed_scales) = make_packed_data(n_layers, n_tokens, hidden, &fwd_batch_sizes);
        let (commit_new, state_new) = commit_minimal_packed(
            packed_a, packed_scales, n_layers, hidden, fwd_batch_sizes, &params, None, 0,
        );

        assert_eq!(commit_old.merkle_root, commit_new.merkle_root, "batched: trace root mismatch");
        assert_eq!(commit_old.io_root, commit_new.io_root, "batched: IO root mismatch");
        assert_eq!(state_old.io_hashes, state_new.io_hashes, "batched: IO chain mismatch");
    }

    #[test]
    fn test_packed_extract_token_matches_retained() {
        let n_layers = 2;
        let hidden = 8;
        let fwd_batch_sizes = vec![1, 1, 1];
        let n_tokens = 3;

        let captures = make_minimal_captures(n_layers, n_tokens, hidden);
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

        let params = FullBindingParams {
            token_ids: &[1, 2, 3],
            prompt: b"p",
            sampling_seed: [0u8; 32],
            manifest: None,
        };
        let (packed_a, packed_scales) = make_packed_data(n_layers, n_tokens, hidden, &fwd_batch_sizes);
        let (_, packed_state) = commit_minimal_packed(
            packed_a, packed_scales, n_layers, hidden, fwd_batch_sizes, &params, None, 0,
        );

        for t in 0..n_tokens {
            let extracted = packed_state.extract_token(t);
            assert_eq!(
                extracted, retained[t],
                "token {} extraction mismatch", t
            );
        }
    }

    #[test]
    fn test_packed_prefix_hashes_match_unpacked() {
        let n_layers = 2;
        let hidden = 8;
        let fwd_batch_sizes = vec![1, 1, 1];
        let n_tokens = 3;

        let captures = make_minimal_captures(n_layers, n_tokens, hidden);
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

        let params = FullBindingParams {
            token_ids: &[10, 20, 30],
            prompt: b"p",
            sampling_seed: [0u8; 32],
            manifest: None,
        };

        let (_, state_old) = commit_minimal(retained.clone(), &params, None);
        let (packed_a, packed_scales) = make_packed_data(n_layers, n_tokens, hidden, &fwd_batch_sizes);
        let (_, packed_state) = commit_minimal_packed(
            packed_a, packed_scales, n_layers, hidden, fwd_batch_sizes, &params, None, 0,
        );

        // Open token 2 from both paths and compare prefix leaf hashes.
        let old_response = open_v4_structural(&state_old, 2);

        let mut packed_prefix = Vec::new();
        for j in 0..2 {
            packed_prefix.push(packed_state.hash_token(j));
        }

        assert_eq!(
            old_response.prefix_leaf_hashes, packed_prefix,
            "prefix leaf hashes mismatch"
        );
    }

    #[test]
    fn test_packed_with_final_residuals_matches() {
        let n_layers = 1;
        let hidden = 4;
        let fwd_batch_sizes = vec![1, 1];
        let n_tokens = 2;
        let fr_dim = 4;

        // Unpacked path with final residuals.
        let captures = make_minimal_captures(n_layers, n_tokens, hidden);
        let retained = build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);
        let final_residuals = Some(vec![
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![5.0f32, 6.0, 7.0, 8.0],
        ]);

        let params = FullBindingParams {
            token_ids: &[10, 20],
            prompt: b"fr",
            sampling_seed: [0u8; 32],
            manifest: None,
        };
        let (commit_old, _) = commit_minimal(retained, &params, final_residuals);

        // Packed path: pack final residuals as contiguous f32 bytes.
        let (packed_a, packed_scales) = make_packed_data(n_layers, n_tokens, hidden, &fwd_batch_sizes);
        let mut packed_fr = Vec::new();
        for v in &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
            packed_fr.extend_from_slice(&v.to_ne_bytes());
        }

        let (commit_new, packed_state) = commit_minimal_packed(
            packed_a, packed_scales, n_layers, hidden, fwd_batch_sizes, &params,
            Some(packed_fr), fr_dim,
        );

        assert_eq!(commit_old.merkle_root, commit_new.merkle_root, "roots with final_residuals mismatch");
        assert_eq!(commit_old.io_root, commit_new.io_root);

        // Verify extract_final_residual roundtrips correctly.
        let fr0 = packed_state.extract_final_residual(0).unwrap();
        assert_eq!(fr0, vec![1.0f32, 2.0, 3.0, 4.0]);
        let fr1 = packed_state.extract_final_residual(1).unwrap();
        assert_eq!(fr1, vec![5.0f32, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_token_index_locate() {
        let idx = TokenIndex::new(&[3, 1, 2]);
        assert_eq!(idx.n_tokens(), 6);
        assert_eq!(idx.locate(0), (0, 0));
        assert_eq!(idx.locate(1), (0, 1));
        assert_eq!(idx.locate(2), (0, 2));
        assert_eq!(idx.locate(3), (1, 0));
        assert_eq!(idx.locate(4), (2, 0));
        assert_eq!(idx.locate(5), (2, 1));
    }

    #[test]
    fn test_packed_scale_ordering_matches_hash() {
        // Directly verify that hash_token on packed data produces the same
        // hash as hash_retained_state_direct on the equivalent RetainedTokenState.
        let n_layers = 3;
        let hidden = 16;

        // Build a single-token retained state with distinctive scale values.
        let mut layers = Vec::new();
        for l in 0..n_layers {
            layers.push(RetainedLayerState {
                a: vec![(l * 7 + 3) as i8; hidden],
                scale_x_attn: 0.11 * (l + 1) as f32,
                scale_a: 0.22 * (l + 1) as f32,
                scale_x_ffn: 0.33 * (l + 1) as f32,
                scale_h: 0.44 * (l + 1) as f32,
            });
        }
        let state = RetainedTokenState { layers };
        let expected = merkle::hash_retained_state_direct(&state);

        // Build packed representation: 1 fwd pass, 1 token.
        let mut packed_a = Vec::new();
        let mut packed_scales = Vec::new();
        for l in 0..n_layers {
            packed_a.extend(std::iter::repeat(((l * 7 + 3) & 0xFF) as u8).take(hidden));
            packed_scales.push(0.11 * (l + 1) as f32);  // scale_x_attn
            packed_scales.push(0.22 * (l + 1) as f32);  // scale_a
            packed_scales.push(0.33 * (l + 1) as f32);  // scale_x_ffn
            packed_scales.push(0.44 * (l + 1) as f32);  // scale_h
        }

        let params = FullBindingParams {
            token_ids: &[1],
            prompt: b"s",
            sampling_seed: [0u8; 32],
            manifest: None,
        };
        let (_, packed_state) = commit_minimal_packed(
            packed_a, packed_scales, n_layers, hidden, vec![1], &params, None, 0,
        );
        let got = packed_state.hash_token(0);

        assert_eq!(got, expected, "packed hash_token scale ordering mismatch vs hash_retained_state_direct");
    }
}
