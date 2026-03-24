//! SHA-256 Merkle tree for token trace commitments.
//!
//! The tree commits to per-token hashes. The prover sends the root
//! with the response, then opens individual leaves on challenge.

use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

/// Build a Merkle tree from leaf hashes. Returns all nodes (bottom-up)
/// and the root hash.
pub fn build_tree(leaves: &[[u8; 32]]) -> MerkleTree {
    assert!(!leaves.is_empty(), "need at least one leaf");

    // Pad to next power of 2
    let n = leaves.len().next_power_of_two();
    let mut nodes = Vec::with_capacity(2 * n);

    // Level 0: leaves (padded with zero hashes)
    for leaf in leaves {
        nodes.push(*leaf);
    }
    let zero = [0u8; 32];
    for _ in leaves.len()..n {
        nodes.push(zero);
    }

    // Build parent levels
    let mut level_start = 0;
    let mut level_size = n;
    while level_size > 1 {
        for i in (0..level_size).step_by(2) {
            let left = nodes[level_start + i];
            let right = nodes[level_start + i + 1];
            let parent = hash_pair(&left, &right);
            nodes.push(parent);
        }
        level_start += level_size;
        level_size /= 2;
    }

    let root = *nodes.last().unwrap();
    MerkleTree {
        nodes,
        n_leaves: leaves.len(),
        padded_size: n,
        root,
    }
}

/// Compute the Merkle root from leaf hashes without storing intermediate nodes.
///
/// Same hash as `build_tree(...).root` but uses O(log N) stack space
/// instead of O(N) heap for the full node array. Use when only the root
/// is needed (e.g. commit-time KV Merkle roots).
pub fn compute_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(!leaves.is_empty(), "need at least one leaf");

    let n = leaves.len().next_power_of_two();
    let zero = [0u8; 32];

    // Build level 0 (padded)
    let mut level: Vec<[u8; 32]> = Vec::with_capacity(n);
    level.extend_from_slice(leaves);
    level.resize(n, zero);

    // Iteratively reduce
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(hash_pair(&pair[0], &pair[1]));
        }
        level = next;
    }

    level[0]
}

/// Generate a proof for the leaf at `index`.
pub fn prove(tree: &MerkleTree, index: usize) -> MerkleProof {
    assert!(index < tree.n_leaves, "leaf index out of range");

    let mut siblings = Vec::new();
    let mut idx = index;
    let mut level_start = 0;
    let mut level_size = tree.padded_size;

    while level_size > 1 {
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        siblings.push(tree.nodes[level_start + sibling_idx]);
        level_start += level_size;
        level_size /= 2;
        idx /= 2;
    }

    MerkleProof {
        leaf_index: index as u32,
        siblings,
    }
}

/// Verify a Merkle proof against a known root.
pub fn verify(root: &[u8; 32], leaf: &[u8; 32], proof: &MerkleProof) -> bool {
    let mut current = *leaf;
    let mut idx = proof.leaf_index as usize;

    for sibling in &proof.siblings {
        current = if idx % 2 == 0 {
            hash_pair(&current, sibling)
        } else {
            hash_pair(sibling, &current)
        };
        idx /= 2;
    }

    current == *root
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Hash arbitrary data to produce a leaf hash.
pub fn hash_leaf(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Compute the Merkle leaf hash for a token trace (legacy, bincode-based).
///
/// If `final_hidden` is `Some`, the hash covers both the serialized layers
/// and the final hidden state, binding the logit computation to the commitment.
/// If `None`, falls back to hashing only the serialized layers (backward compat).
pub fn trace_leaf_hash(serialized_layers: &[u8], final_hidden: Option<&[i8]>) -> [u8; 32] {
    match final_hidden {
        None => hash_leaf(serialized_layers),
        Some(fh) => {
            let mut hasher = Sha256::new();
            hasher.update(serialized_layers);
            // Domain separator to prevent collisions with the layers-only hash.
            hasher.update(b"final_hidden");
            for &b in fh {
                hasher.update([b as u8]);
            }
            hasher.finalize().into()
        }
    }
}

/// Compute the Merkle leaf hash for a token trace by hashing fields directly.
///
/// Hashes each `LayerTrace` field in canonical order without intermediate
/// serialization. KV cache fields (`kv_cache_k`, `kv_cache_v`) are **excluded**:
/// KV is committed separately via the KV chain tree and per-token KV Merkle roots.
///
/// This replaces `bincode::serialize(layers) → trace_leaf_hash(bytes)`:
/// - No intermediate allocation (fields are fed directly to SHA-256)
/// - O(computation_data) instead of O(computation_data + KV_prefix²)
/// - Architecturally correct: trace tree commits computation, KV tree commits provenance
pub fn hash_trace_direct(layers: &[crate::types::LayerTrace], final_hidden: Option<&[i8]>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-trace-v2");

    for lt in layers {
        hash_i8_into(&mut hasher, &lt.x_attn);
        hash_i32_into(&mut hasher, &lt.q);
        hash_i32_into(&mut hasher, &lt.k);
        hash_i32_into(&mut hasher, &lt.v);
        hash_i8_into(&mut hasher, &lt.a);
        hash_i32_into(&mut hasher, &lt.attn_out);
        hash_i8_into(&mut hasher, &lt.x_ffn);
        hash_i32_into(&mut hasher, &lt.g);
        hash_i32_into(&mut hasher, &lt.u);
        hash_i8_into(&mut hasher, &lt.h);
        hash_i32_into(&mut hasher, &lt.ffn_out);
        // kv_cache_k, kv_cache_v: excluded — committed in KV chain tree
        hash_opt_f32_into(&mut hasher, lt.scale_x_attn);
        hash_opt_f32_into(&mut hasher, lt.scale_a);
        hash_opt_f32_into(&mut hasher, lt.scale_x_ffn);
        hash_opt_f32_into(&mut hasher, lt.scale_h);
        match &lt.residual {
            Some(res) => {
                hasher.update([1u8]);
                hash_f32_into(&mut hasher, res);
            }
            None => hasher.update([0u8]),
        }
    }

    if let Some(fh) = final_hidden {
        hasher.update(b"final_hidden");
        hash_i8_into(&mut hasher, fh);
    }

    hasher.finalize().into()
}

/// Compute the Merkle leaf hash for a token's minimal retained state.
///
/// Hashes only the non-replayable attention boundary (`a`, `scale_a`)
/// and the dynamic quantization scales needed for audit replay.
/// Used for V4 (retained-state) commitments.
///
/// Domain separator `"vi-retained-v1"` prevents collisions with
/// the full-trace hash (`"vi-trace-v2"`).
pub fn hash_retained_state_direct(state: &crate::types::RetainedTokenState) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-retained-v1");

    for ls in &state.layers {
        hash_i8_into(&mut hasher, &ls.a);
        hasher.update(ls.scale_a.to_le_bytes());
        // Transitional replay scales (dropped in V5 once audit-time
        // scale derivation is proven to match runtime quantization).
        hasher.update(ls.scale_x_attn.to_le_bytes());
        hasher.update(ls.scale_x_ffn.to_le_bytes());
        hasher.update(ls.scale_h.to_le_bytes());
    }

    hasher.finalize().into()
}

/// Compute the committed leaf hash for a token with optional final residual binding.
///
/// When `final_residual` is `Some`, the leaf hash binds the captured pre-final-norm
/// residual into the Merkle tree. This prevents a malicious prover from swapping
/// the boundary state after seeing the audit challenge.
///
/// When `None`, returns `hash_retained_state_direct(state)` (backward compatible).
pub fn hash_retained_with_residual(
    state: &crate::types::RetainedTokenState,
    final_residual: Option<&[f32]>,
) -> [u8; 32] {
    let base = hash_retained_state_direct(state);
    match final_residual {
        Some(fr) => {
            let mut hasher = Sha256::new();
            hasher.update(b"vi-retained-fr-v1");
            hasher.update(base);
            hash_f32_into(&mut hasher, fr);
            hasher.finalize().into()
        }
        None => base,
    }
}

/// Compute the V4 IO chain hash for a token.
///
/// `io_t = H("vi-io-v4" || leaf_hash_t || token_id_t || prev_io_hash)`
///
/// Chains the retained leaf hash (the exact committed object) into the
/// transcript chain. This ties order/splice resistance to the retained
/// Merkle tree rather than ad hoc token features.
pub fn io_hash_v4(
    leaf_hash: [u8; 32],
    token_id: u32,
    prev_io_hash: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-io-v4");
    hasher.update(leaf_hash);
    hasher.update(token_id.to_le_bytes());
    hasher.update(prev_io_hash);
    hasher.finalize().into()
}

// --- Direct hash helpers: feed raw bytes into SHA-256 without allocation ---

fn hash_i8_into(hasher: &mut Sha256, data: &[i8]) {
    // SAFETY: i8 and u8 have identical size/alignment.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    hasher.update(bytes);
}

fn hash_i32_into(hasher: &mut Sha256, data: &[i32]) {
    // SAFETY: i32 is 4 bytes; reinterpret as native-endian byte slice.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    hasher.update(bytes);
}

fn hash_f32_into(hasher: &mut Sha256, data: &[f32]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    hasher.update(bytes);
}

fn hash_opt_f32_into(hasher: &mut Sha256, val: Option<f32>) {
    match val {
        Some(v) => {
            hasher.update([1u8]);
            hasher.update(v.to_le_bytes());
        }
        None => hasher.update([0u8]),
    }
}

#[derive(Debug, Clone)]
pub struct MerkleTree {
    pub nodes: Vec<[u8; 32]>,
    pub n_leaves: usize,
    pub padded_size: usize,
    pub root: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_index: u32,
    pub siblings: Vec<[u8; 32]>,
}

/// Hash an embedding table row (f32 values) to produce a Merkle leaf hash.
///
/// Domain separator `"vi-embedding-v1"` prevents collisions with other hash domains.
pub fn hash_embedding_row(row: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-embedding-v1");
    hash_f32_into(&mut hasher, row);
    hasher.finalize().into()
}

/// Compute a deployment manifest hash: SHA-256 over all manifest fields in canonical order.
///
/// The hash covers (with domain separator `b"vi-manifest-v1"`):
///   1. `tokenizer_hash` (32 bytes)
///   2. `temperature` as little-endian f32 (4 bytes)
///   3. `top_k` as little-endian u32 (4 bytes)
///   4. `top_p` as little-endian f32 (4 bytes)
///   5. `eos_policy` as UTF-8 bytes
pub fn hash_manifest(manifest: &crate::types::DeploymentManifest) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-manifest-v3");
    hasher.update(manifest.tokenizer_hash);
    hasher.update(manifest.temperature.to_le_bytes());
    hasher.update(manifest.top_k.to_le_bytes());
    hasher.update(manifest.top_p.to_le_bytes());
    hasher.update(manifest.eos_policy.as_bytes());
    // R_W: weight root binds the commitment to a specific quantized checkpoint.
    if let Some(wh) = manifest.weight_hash {
        hasher.update(b"\x01"); // presence tag
        hasher.update(wh);
    } else {
        hasher.update(b"\x00");
    }
    // Quantization scheme hash.
    if let Some(qh) = manifest.quant_hash {
        hasher.update(b"\x01");
        hasher.update(qh);
    } else {
        hasher.update(b"\x00");
    }
    // System prompt hash.
    if let Some(sph) = manifest.system_prompt_hash {
        hasher.update(b"\x01");
        hasher.update(sph);
    } else {
        hasher.update(b"\x00");
    }
    // Logit-modifying parameters.
    hasher.update(manifest.repetition_penalty.to_le_bytes());
    hasher.update(manifest.frequency_penalty.to_le_bytes());
    hasher.update(manifest.presence_penalty.to_le_bytes());
    // Logit bias: length-prefixed sorted pairs.
    hasher.update((manifest.logit_bias.len() as u32).to_le_bytes());
    for &(token_id, bias) in &manifest.logit_bias {
        hasher.update(token_id.to_le_bytes());
        hasher.update(bias.to_le_bytes());
    }
    // Guided decoding constraint.
    hasher.update((manifest.guided_decoding.len() as u32).to_le_bytes());
    hasher.update(manifest.guided_decoding.as_bytes());
    // Output-level parameters.
    hasher.update((manifest.stop_sequences.len() as u32).to_le_bytes());
    for s in &manifest.stop_sequences {
        hasher.update((s.len() as u32).to_le_bytes());
        hasher.update(s.as_bytes());
    }
    hasher.update(manifest.max_tokens.to_le_bytes());
    hasher.finalize().into()
}

/// Compute a weight-chain hash: SHA-256 over all INT8 weights in canonical order.
///
/// The hash covers:
///   1. `source_dtype` string (e.g. "I8", "BF16", "F16")
///   2. For each layer (0..n_layers), for each matrix type (Wq, Wk, Wv, Wo, Wg, Wu, Wd):
///      a. The quantization scale as little-endian f32 bytes (0.0 for native INT8)
///      b. The INT8 weight bytes (cast to u8)
///
/// This binds the verifier key to a specific set of quantized weights,
/// closing the gap between "we verified INT8 computation" and "we verified
/// that INT8 computation corresponds to the public checkpoint."
///
/// Both the production keygen (`verilm-keygen`) and the toy keygen (`verilm-test-vectors`)
/// must compute this hash identically for the same weights.
pub fn hash_weights(
    source_dtype: &str,
    n_layers: usize,
    quantization_scales: &[Vec<f32>],
    weights_iter: impl Fn(usize, usize) -> Vec<i8>,
    n_matrix_types: usize,
) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();

    // Domain separator
    hasher.update(b"vi-weight-chain-v1");

    // Source dtype
    hasher.update(source_dtype.as_bytes());

    for layer in 0..n_layers {
        for mt_idx in 0..n_matrix_types {
            // Quantization scale (0.0 for native INT8, nonzero for BF16/FP16)
            let scale = if !quantization_scales.is_empty() {
                quantization_scales[layer][mt_idx]
            } else {
                0.0f32
            };
            hasher.update(scale.to_le_bytes());

            // INT8 weight bytes
            let weights = weights_iter(layer, mt_idx);
            // Safety: i8 and u8 have identical layout
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len())
            };
            hasher.update(bytes);
        }
    }

    hasher.finalize().into()
}

/// Binding context for computing IO leaf hashes.
///
/// Determines the hash version and what fields are included.
pub enum IoHashBinding {
    /// V1 legacy: `H(first_input || requant(last_output))`.
    Legacy,
    /// V2: `H("vi-io-v2" || first_input || requant(last_output) || token_id)`.
    TokenId(u32),
    /// V3: `H("vi-io-v3" || first_input || requant(last_output) || token_id || prev_io_hash)`.
    /// Transcript-chained: each token's IO hash depends on the previous token's IO hash.
    Chained { token_id: u32, prev_io_hash: [u8; 32] },
}

/// Compute the IO hash for a token.
///
/// The domain separator in each version ensures hashes never collide
/// across versions, so old proofs cannot ambiguously verify under a
/// newer scheme.
pub fn io_hash(first_input: &[i8], last_output_i32: &[i32], binding: &IoHashBinding) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Common helper: hash the tensor data
    let hash_tensors = |h: &mut Sha256| {
        for &b in first_input {
            h.update([b as u8]);
        }
        for &v in last_output_i32 {
            let clamped = v.clamp(-128, 127) as i8;
            h.update([clamped as u8]);
        }
    };

    match binding {
        IoHashBinding::Legacy => {
            hash_tensors(&mut hasher);
        }
        IoHashBinding::TokenId(tid) => {
            hasher.update(b"vi-io-v2");
            hash_tensors(&mut hasher);
            hasher.update(tid.to_le_bytes());
        }
        IoHashBinding::Chained { token_id, prev_io_hash } => {
            hasher.update(b"vi-io-v3");
            hash_tensors(&mut hasher);
            hasher.update(token_id.to_le_bytes());
            hasher.update(prev_io_hash);
        }
    }
    hasher.finalize().into()
}

/// Convenience: build `IoHashBinding` from optional fields.
/// - `None` token_id → Legacy
/// - `Some` token_id, `None` prev → TokenId
/// - `Some` token_id, `Some` prev → Chained
pub fn io_binding(token_id: Option<u32>, prev_io_hash: Option<[u8; 32]>) -> IoHashBinding {
    match (token_id, prev_io_hash) {
        (None, _) => IoHashBinding::Legacy,
        (Some(tid), None) => IoHashBinding::TokenId(tid),
        (Some(tid), Some(prev)) => IoHashBinding::Chained { token_id: tid, prev_io_hash: prev },
    }
}

/// Compute the KV provenance chain hash for a token.
///
/// Chain: `H("vi-kv-v1" || prev_kv_hash || requantize(k_t) || requantize(v_t) || token_index_le32)`
///
/// This binds each token's K/V projections into a running chain, so that
/// under partial opening the verifier can detect fabricated KV cache entries
/// for consecutive opened tokens.
///
/// For token 0, `prev_kv_hash` is the genesis zero hash `[0u8; 32]`.
/// The hash covers all layers' K and V (concatenated in layer order).
///
/// Accepts any type that derefs to `[i8]` (`Vec<i8>`, `&[i8]`, etc.).
pub fn kv_chain_hash<K: AsRef<[i8]>, V: AsRef<[i8]>>(
    prev_kv_hash: &[u8; 32],
    k_i8_per_layer: &[K],
    v_i8_per_layer: &[V],
    token_index: u32,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-kv-v1");
    hasher.update(prev_kv_hash);
    for k in k_i8_per_layer {
        let k = k.as_ref();
        // Safety: i8 and u8 have identical layout
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len())
        };
        hasher.update(bytes);
    }
    for v in v_i8_per_layer {
        let v = v.as_ref();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len())
        };
        hasher.update(bytes);
    }
    hasher.update(token_index.to_le_bytes());
    hasher.finalize().into()
}

/// SHA-256 of a sampling seed (for seed commitment).
pub fn hash_seed(seed: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-seed-v1");
    hasher.update(seed);
    hasher.finalize().into()
}

/// SHA-256 of canonicalized prompt bytes (for prompt binding).
pub fn hash_prompt(prompt_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-prompt-v1");
    hasher.update(prompt_bytes);
    hasher.finalize().into()
}

/// Derive `k` unique challenge indices from a commitment root and verifier challenge seed.
///
/// Uses `SHA256(root || challenge_seed || counter)` to generate each index mod
/// `n_tokens`. This is interactive challenge expansion, not Fiat-Shamir: the
/// verifier samples `challenge_seed` after seeing the commitment root, so the
/// prover cannot know the challenge set at commitment time.
pub fn derive_challenges(
    root: &[u8; 32],
    challenge_seed: &[u8; 32],
    n_tokens: u32,
    k: u32,
) -> Vec<u32> {
    use std::collections::BTreeSet;
    let k = k.min(n_tokens); // can't challenge more tokens than exist
    let mut indices = BTreeSet::new();
    let mut counter: u32 = 0;
    while (indices.len() as u32) < k {
        let mut hasher = Sha256::new();
        hasher.update(root);
        hasher.update(challenge_seed);
        hasher.update(counter.to_le_bytes());
        let hash: [u8; 32] = hasher.finalize().into();
        let idx = u32::from_le_bytes(hash[..4].try_into().unwrap()) % n_tokens;
        indices.insert(idx);
        counter += 1;
    }
    indices.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_leaf() {
        let leaf = hash_leaf(b"hello");
        let tree = build_tree(&[leaf]);
        assert_eq!(tree.root, leaf);
        let proof = prove(&tree, 0);
        assert!(verify(&tree.root, &leaf, &proof));
    }

    #[test]
    fn test_two_leaves() {
        let a = hash_leaf(b"a");
        let b = hash_leaf(b"b");
        let tree = build_tree(&[a, b]);

        let proof_a = prove(&tree, 0);
        assert!(verify(&tree.root, &a, &proof_a));

        let proof_b = prove(&tree, 1);
        assert!(verify(&tree.root, &b, &proof_b));

        // Wrong leaf should fail
        assert!(!verify(&tree.root, &b, &proof_a));
    }

    #[test]
    fn test_five_leaves() {
        let leaves: Vec<[u8; 32]> = (0..5u8).map(|i| hash_leaf(&[i])).collect();
        let tree = build_tree(&leaves);

        for (i, leaf) in leaves.iter().enumerate() {
            let proof = prove(&tree, i);
            assert!(verify(&tree.root, leaf, &proof));
        }
    }

    #[test]
    fn test_tampered_proof_fails() {
        let leaves: Vec<[u8; 32]> = (0..4u8).map(|i| hash_leaf(&[i])).collect();
        let tree = build_tree(&leaves);
        let mut proof = prove(&tree, 0);
        // Tamper with a sibling
        proof.siblings[0][0] ^= 0xff;
        assert!(!verify(&tree.root, &leaves[0], &proof));
    }

    #[test]
    fn test_compute_root_matches_build_tree() {
        for n in [1, 2, 3, 5, 8, 13, 28, 32] {
            let leaves: Vec<[u8; 32]> = (0..n as u8).map(|i| hash_leaf(&[i])).collect();
            let tree_root = build_tree(&leaves).root;
            let fast_root = compute_root(&leaves);
            assert_eq!(tree_root, fast_root, "mismatch for {n} leaves");
        }
    }

    #[test]
    fn test_retained_state_hash_deterministic() {
        use crate::types::{RetainedLayerState, RetainedTokenState};

        let state = RetainedTokenState {
            layers: vec![RetainedLayerState {
                a: vec![1, 2, 3, 4],
                scale_a: 0.5,
                scale_x_attn: 0.1,
                scale_x_ffn: 0.2,
                scale_h: 0.3,
            }],
        };
        let h1 = hash_retained_state_direct(&state);
        let h2 = hash_retained_state_direct(&state);
        assert_eq!(h1, h2, "same state must produce same hash");

        // Different state must produce different hash.
        let state2 = RetainedTokenState {
            layers: vec![RetainedLayerState {
                a: vec![1, 2, 3, 5], // changed
                scale_a: 0.5,
                scale_x_attn: 0.1,
                scale_x_ffn: 0.2,
                scale_h: 0.3,
            }],
        };
        assert_ne!(h1, hash_retained_state_direct(&state2));
    }

    #[test]
    fn test_io_hash_v4_chains_leaf_hash() {
        use crate::types::{RetainedLayerState, RetainedTokenState};

        let state = RetainedTokenState {
            layers: vec![RetainedLayerState {
                a: vec![10, 20],
                scale_a: 1.0,
                scale_x_attn: 0.5,
                scale_x_ffn: 0.5,
                scale_h: 0.5,
            }],
        };
        let leaf_hash = hash_retained_state_direct(&state);
        let prev = [0u8; 32];

        let io = io_hash_v4(leaf_hash, 42, prev);

        // Must be deterministic.
        assert_eq!(io, io_hash_v4(leaf_hash, 42, prev));

        // Different token ID → different hash.
        assert_ne!(io, io_hash_v4(leaf_hash, 43, prev));

        // Different prev → different hash (chain property).
        let prev2 = [1u8; 32];
        assert_ne!(io, io_hash_v4(leaf_hash, 42, prev2));

        // Different leaf → different hash.
        let state2 = RetainedTokenState {
            layers: vec![RetainedLayerState {
                a: vec![11, 20],
                scale_a: 1.0,
                scale_x_attn: 0.5,
                scale_x_ffn: 0.5,
                scale_h: 0.5,
            }],
        };
        let leaf_hash2 = hash_retained_state_direct(&state2);
        assert_ne!(io, io_hash_v4(leaf_hash2, 42, prev));
    }
}
