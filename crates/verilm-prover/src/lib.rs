//! Prover engine: commitment generation and proof opening for the VeriLM protocol.

use rayon::prelude::*;
use verilm_core::bridge_requantize;
use verilm_core::constants::MatrixType;
use verilm_core::matmul::matmul_i32;
use verilm_core::merkle;
use verilm_core::rmsnorm::{bridge_residual_rmsnorm, dequant_add_residual, rmsnorm_f64_input, quantize_f64_to_i8};
use verilm_core::types::{
    BatchCommitment, BridgeParams, CommitmentVersion, TailParams,
    DeploymentManifest, RetainedLayerState, RetainedTokenState,
    ShellLayerOpening, ShellTokenOpening, ShellWeights,
    V4AuditResponse,
};

/// Binding parameters for commit (token IDs, prompt, seed, manifest).
pub struct FullBindingParams<'a> {
    pub token_ids: &'a [u32],
    pub prompt: &'a [u8],
    pub sampling_seed: [u8; 32],
    /// Optional deployment manifest. When provided, its hash is bound into
    /// the commitment alongside the other V3 fields.
    pub manifest: Option<&'a DeploymentManifest>,
    /// Number of prompt tokens (full count including the first token that is
    /// consumed as embedding input and not included in `token_ids`).
    /// `None` for legacy callers that don't track the prompt/generation boundary.
    pub n_prompt_tokens: Option<u32>,
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
    pub input_spec_hash: Option<[u8; 32]>,
    pub model_spec_hash: Option<[u8; 32]>,
    pub decode_spec_hash: Option<[u8; 32]>,
    pub output_spec_hash: Option<[u8; 32]>,
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
    /// Raw prompt bytes, stored for inclusion in audit responses.
    pub prompt: Vec<u8>,
    /// Number of prompt tokens (full count including first).
    pub n_prompt_tokens: Option<u32>,
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
    let n_tokens = all_retained.len();
    assert_eq!(
        n_tokens,
        params.token_ids.len(),
        "retained state count must match token_ids"
    );

    let timers = std::env::var("VERILM_COMMIT_TIMERS").map_or(false, |v| v == "1");
    let t0 = if timers { Some(std::time::Instant::now()) } else { None };

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

    let t_leaf = if timers { Some(std::time::Instant::now()) } else { None };

    // IO tree: chain the leaf hash for splice resistance.
    // Genesis is bound to the request via prompt_hash.
    let prompt_hash = merkle::hash_prompt(params.prompt);
    let mut io_leaves = Vec::with_capacity(n_tokens);
    let mut prev_io = merkle::io_genesis_v4(prompt_hash);
    for (i, leaf_hash) in trace_leaves.iter().enumerate() {
        let io = merkle::io_hash_v4(*leaf_hash, params.token_ids[i], prev_io);
        io_leaves.push(io);
        prev_io = io;
    }

    let t_io_chain = if timers { Some(std::time::Instant::now()) } else { None };

    let trace_tree = merkle::build_tree(&trace_leaves);
    let io_tree = merkle::build_tree(&io_leaves);

    // Compute four-spec hashes and composed manifest hash.
    let (manifest_hash, input_spec_hash, model_spec_hash, decode_spec_hash, output_spec_hash) =
        match params.manifest {
            Some(m) => {
                let (input, model, decode, output) = m.split();
                let h_in = merkle::hash_input_spec(&input);
                let h_mod = merkle::hash_model_spec(&model);
                let h_dec = merkle::hash_decode_spec(&decode);
                let h_out = merkle::hash_output_spec(&output);
                let m_hash = merkle::hash_manifest_composed(h_in, h_mod, h_dec, h_out);
                (Some(m_hash), Some(h_in), Some(h_mod), Some(h_dec), Some(h_out))
            }
            None => (None, None, None, None, None),
        };

    let t_trees = if timers { Some(std::time::Instant::now()) } else { None };

    let commitment = BatchCommitment {
        merkle_root: trace_tree.root,
        io_root: io_tree.root,
        n_tokens: n_tokens as u32,
        manifest_hash,
        input_spec_hash,
        model_spec_hash,
        decode_spec_hash,
        output_spec_hash,
        version: CommitmentVersion::V4,
        prompt_hash: Some(merkle::hash_prompt(params.prompt)),
        seed_commitment: Some(merkle::hash_seed(&params.sampling_seed)),
        kv_chain_root: None,
        n_prompt_tokens: params.n_prompt_tokens,
    };

    let state = MinimalBatchState {
        retained_tree: trace_tree,
        io_tree,
        all_retained,
        manifest_hash,
        input_spec_hash,
        model_spec_hash,
        decode_spec_hash,
        output_spec_hash,
        token_ids: params.token_ids.to_vec(),
        prompt_hash: merkle::hash_prompt(params.prompt),
        seed_commitment: merkle::hash_seed(&params.sampling_seed),
        revealed_seed: params.sampling_seed,
        io_hashes: io_leaves,
        manifest: params.manifest.cloned(),
        final_residuals,
        prompt: params.prompt.to_vec(),
        n_prompt_tokens: params.n_prompt_tokens,
    };

    if let (Some(t0), Some(t_leaf), Some(t_io_chain), Some(t_trees)) =
        (t0, t_leaf, t_io_chain, t_trees)
    {
        let now = std::time::Instant::now();
        eprintln!(
            "verilm rust commit_minimal timers: leaf_hash={:.1}ms io_chain={:.1}ms trees={:.1}ms assemble={:.1}ms total={:.1}ms ({} tokens)",
            t_leaf.duration_since(t0).as_secs_f64() * 1000.0,
            t_io_chain.duration_since(t_leaf).as_secs_f64() * 1000.0,
            t_trees.duration_since(t_io_chain).as_secs_f64() * 1000.0,
            now.duration_since(t_trees).as_secs_f64() * 1000.0,
            now.duration_since(t0).as_secs_f64() * 1000.0,
            n_tokens,
        );
    }

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
    pub input_spec_hash: Option<[u8; 32]>,
    pub model_spec_hash: Option<[u8; 32]>,
    pub decode_spec_hash: Option<[u8; 32]>,
    pub output_spec_hash: Option<[u8; 32]>,
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
    /// Raw prompt bytes.
    pub prompt: Vec<u8>,
    /// Number of prompt tokens (full count including first).
    pub n_prompt_tokens: Option<u32>,
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

    let timers = std::env::var("VERILM_COMMIT_TIMERS").map_or(false, |v| v == "1");

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
        input_spec_hash: None,
        model_spec_hash: None,
        decode_spec_hash: None,
        output_spec_hash: None,
        token_ids: Vec::new(),
        prompt_hash: [0u8; 32],
        seed_commitment: [0u8; 32],
        revealed_seed: [0u8; 32],
        io_hashes: Vec::new(),
        manifest: None,
        packed_final_res,
        final_res_dim,
        prompt: Vec::new(),
        n_prompt_tokens: None,
    };

    let t0 = if timers { Some(std::time::Instant::now()) } else { None };

    // Trace tree: hash each token in parallel.
    let trace_leaves: Vec<[u8; 32]> = (0..n_tokens)
        .into_par_iter()
        .map(|t| state.hash_token(t))
        .collect();

    let t_leaf = if timers { Some(std::time::Instant::now()) } else { None };

    // IO tree: sequential chain, genesis bound to request.
    let prompt_hash = merkle::hash_prompt(params.prompt);
    let mut io_leaves = Vec::with_capacity(n_tokens);
    let mut prev_io = merkle::io_genesis_v4(prompt_hash);
    for (i, leaf_hash) in trace_leaves.iter().enumerate() {
        let io = merkle::io_hash_v4(*leaf_hash, params.token_ids[i], prev_io);
        io_leaves.push(io);
        prev_io = io;
    }

    let t_io_chain = if timers { Some(std::time::Instant::now()) } else { None };

    let trace_tree = merkle::build_tree(&trace_leaves);
    let io_tree = merkle::build_tree(&io_leaves);

    let (manifest_hash, input_spec_hash, model_spec_hash, decode_spec_hash, output_spec_hash) =
        match params.manifest {
            Some(m) => {
                let (input, model, decode, output) = m.split();
                let h_in = merkle::hash_input_spec(&input);
                let h_mod = merkle::hash_model_spec(&model);
                let h_dec = merkle::hash_decode_spec(&decode);
                let h_out = merkle::hash_output_spec(&output);
                let m_hash = merkle::hash_manifest_composed(h_in, h_mod, h_dec, h_out);
                (Some(m_hash), Some(h_in), Some(h_mod), Some(h_dec), Some(h_out))
            }
            None => (None, None, None, None, None),
        };

    let t_trees = if timers { Some(std::time::Instant::now()) } else { None };

    let commitment = BatchCommitment {
        merkle_root: trace_tree.root,
        io_root: io_tree.root,
        n_tokens: n_tokens as u32,
        manifest_hash,
        input_spec_hash,
        model_spec_hash,
        decode_spec_hash,
        output_spec_hash,
        version: CommitmentVersion::V4,
        prompt_hash: Some(merkle::hash_prompt(params.prompt)),
        seed_commitment: Some(merkle::hash_seed(&params.sampling_seed)),
        kv_chain_root: None,
        n_prompt_tokens: params.n_prompt_tokens,
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
        input_spec_hash,
        model_spec_hash,
        decode_spec_hash,
        output_spec_hash,
        token_ids: params.token_ids.to_vec(),
        prompt_hash: merkle::hash_prompt(params.prompt),
        seed_commitment: merkle::hash_seed(&params.sampling_seed),
        revealed_seed: params.sampling_seed,
        io_hashes: io_leaves,
        manifest: params.manifest.cloned(),
        packed_final_res: state.packed_final_res,
        final_res_dim,
        prompt: params.prompt.to_vec(),
        n_prompt_tokens: params.n_prompt_tokens,
    };

    if let (Some(t0), Some(t_leaf), Some(t_io_chain), Some(t_trees)) =
        (t0, t_leaf, t_io_chain, t_trees)
    {
        let now = std::time::Instant::now();
        eprintln!(
            "verilm rust commit_packed timers: leaf_hash={:.1}ms io_chain={:.1}ms trees={:.1}ms assemble={:.1}ms total={:.1}ms ({} tokens)",
            t_leaf.duration_since(t0).as_secs_f64() * 1000.0,
            t_io_chain.duration_since(t_leaf).as_secs_f64() * 1000.0,
            t_trees.duration_since(t_io_chain).as_secs_f64() * 1000.0,
            now.duration_since(t_trees).as_secs_f64() * 1000.0,
            now.duration_since(t0).as_secs_f64() * 1000.0,
            n_tokens,
        );
    }

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
    tail: Option<&TailParams>,
    layer_filter: Option<&[usize]>,
) -> V4AuditResponse {
    let i = token_index as usize;
    assert!(i < state.n_tokens(), "token_index out of range");

    let commitment = BatchCommitment {
        merkle_root: state.retained_tree.root,
        io_root: state.io_tree.root,
        n_tokens: state.n_tokens() as u32,
        manifest_hash: state.manifest_hash,
        input_spec_hash: state.input_spec_hash,
        model_spec_hash: state.model_spec_hash,
        decode_spec_hash: state.decode_spec_hash,
        output_spec_hash: state.output_spec_hash,
        version: CommitmentVersion::V4,
        prompt_hash: Some(state.prompt_hash),
        seed_commitment: Some(state.seed_commitment),
        kv_chain_root: None,
        n_prompt_tokens: state.n_prompt_tokens,
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
        merkle::io_genesis_v4(state.prompt_hash)
    } else {
        state.io_hashes[i - 1]
    };

    // Reconstruct ONLY the challenged token for shell computation.
    let retained_token = state.extract_token(i);

    let mut shell = compute_shell_opening(
        &retained_token, weights, cfg, weight_scales, bridge, layer_filter,
    );
    shell.final_residual = state.extract_final_residual(i);

    // Compute LM-head logits claim for Freivalds verification.
    if let (Some(ref fr), Some(tp)) = (&shell.final_residual, tail) {
        let res_f64: Vec<f64> = fr.iter().map(|&v| v as f64).collect();
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
            &res_f64, tp.final_norm_weights, tp.rmsnorm_eps,
        );
        let fh: Vec<i8> = normed.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect();
        let logits = matmul_i32(tp.lm_head, &fh, cfg.vocab_size, cfg.hidden_dim);
        shell.logits_i32 = Some(logits);
    }

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
        prompt: Some(state.prompt.clone()),
        n_prompt_tokens: state.n_prompt_tokens,
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
        let idx = MatrixType::PER_LAYER.iter().position(|&m| m == mt).unwrap();
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
        logits_i32: None,     // Set by open_v4 when tail params available
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
///
/// When `tail` is `Some`, computes `logits_i32 = lm_head @ quantize(final_norm(final_residual))`
/// for Freivalds verification by the verifier.
pub fn open_v4(
    state: &MinimalBatchState,
    token_index: u32,
    weights: &dyn ShellWeights,
    cfg: &verilm_core::constants::ModelConfig,
    weight_scales: &[Vec<f32>],
    bridge: Option<&BridgeParams>,
    tail: Option<&TailParams>,
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

    // Compute LM-head logits claim for Freivalds verification.
    if let (Some(ref fr), Some(tp)) = (&shell.final_residual, tail) {
        let res_f64: Vec<f64> = fr.iter().map(|&v| v as f64).collect();
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
            &res_f64, tp.final_norm_weights, tp.rmsnorm_eps,
        );
        let fh: Vec<i8> = normed.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect();
        let logits = matmul_i32(tp.lm_head, &fh, cfg.vocab_size, cfg.hidden_dim);
        shell.logits_i32 = Some(logits);
    }

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
        input_spec_hash: state.input_spec_hash,
        model_spec_hash: state.model_spec_hash,
        decode_spec_hash: state.decode_spec_hash,
        output_spec_hash: state.output_spec_hash,
        version: CommitmentVersion::V4,
        prompt_hash: Some(state.prompt_hash),
        seed_commitment: Some(state.seed_commitment),
        kv_chain_root: None,
        n_prompt_tokens: state.n_prompt_tokens,
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
        merkle::io_genesis_v4(state.prompt_hash)
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
        prompt: Some(state.prompt.clone()),
        n_prompt_tokens: state.n_prompt_tokens,
    }
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
            n_prompt_tokens: Some(1),
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
            n_prompt_tokens: Some(1),
        };

        let (_, state) = commit_minimal(retained.clone(), &params, None);

        // Verify IO chain uses leaf hashes, not ad hoc features.
        let leaf0 = merkle::hash_retained_state_direct(&retained[0]);
        let leaf1 = merkle::hash_retained_state_direct(&retained[1]);

        let genesis = merkle::io_genesis_v4(merkle::hash_prompt(b"p"));
        let expected_io0 = merkle::io_hash_v4(leaf0, 10, genesis);
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
            n_prompt_tokens: Some(1),
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
            n_prompt_tokens: Some(1),
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
        let genesis = merkle::io_genesis_v4(merkle::hash_prompt(b"test"));
        let io_0 = merkle::io_hash_v4(leaf_hash_0, 10, genesis);
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
            n_prompt_tokens: Some(1),
        };

        let (_, state) = commit_minimal(retained, &params, None);
        let response = open_v4_structural(&state, 0);

        assert_eq!(response.prefix_leaf_hashes.len(), 0);
        assert_eq!(response.prefix_merkle_proofs.len(), 0);
        let genesis = merkle::io_genesis_v4(merkle::hash_prompt(b"p"));
        assert_eq!(response.prev_io_hash, genesis);
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
            n_prompt_tokens: Some(1),
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
            n_prompt_tokens: Some(1),
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
            n_prompt_tokens: Some(1),
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
            n_prompt_tokens: Some(1),
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
            n_prompt_tokens: Some(1),
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
            n_prompt_tokens: Some(1),
        };
        let (_, packed_state) = commit_minimal_packed(
            packed_a, packed_scales, n_layers, hidden, vec![1], &params, None, 0,
        );
        let got = packed_state.hash_token(0);

        assert_eq!(got, expected, "packed hash_token scale ordering mismatch vs hash_retained_state_direct");
    }
}
