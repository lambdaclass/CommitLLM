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

/// Compute the Merkle leaf hash for a token's minimal retained state.
///
/// Hashes only the irreducible attention boundary (`a`, `scale_a`).
/// Bridge replay scales have moved to [`ShellLayerOpening`] and are
/// verified implicitly by Freivalds at audit time.
/// Used for V4 (retained-state) commitments.
///
/// Domain separator `"vi-retained-v3"` ensures hash uniqueness.
///
/// v3 adds committed bridge trust boundary fields (`x_attn_i8`, `scale_x_attn`)
/// to the hash when present. A presence marker byte distinguishes committed
/// vs uncommitted layers so that `None` fields produce a distinct hash.
pub fn hash_retained_state_direct(state: &crate::types::RetainedTokenState) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-retained-v3");

    for ls in &state.layers {
        hash_i8_into(&mut hasher, &ls.a);
        hasher.update(ls.scale_a.to_le_bytes());
        if let (Some(ref xa), Some(sx)) = (&ls.x_attn_i8, ls.scale_x_attn) {
            hasher.update(b"\x01");
            hash_i8_into(&mut hasher, xa);
            hasher.update(sx.to_le_bytes());
        } else {
            hasher.update(b"\x00");
        }
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

fn hash_f32_into(hasher: &mut Sha256, data: &[f32]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    hasher.update(bytes);
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

// ---------------------------------------------------------------------------
// Four-spec hashing (paper §3)
// ---------------------------------------------------------------------------

/// Hash an `InputSpec`: tokenizer, system prompt, chat template, preprocessing policies.
pub fn hash_input_spec(spec: &crate::types::InputSpec) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-input-v1");
    hasher.update(spec.tokenizer_hash);
    hash_optional_32(&mut hasher, spec.system_prompt_hash.as_ref());
    hash_optional_32(&mut hasher, spec.chat_template_hash.as_ref());
    hash_optional_string(&mut hasher, spec.bos_eos_policy.as_deref());
    hash_optional_string(&mut hasher, spec.truncation_policy.as_deref());
    hash_optional_string(&mut hasher, spec.special_token_policy.as_deref());
    hash_optional_string(&mut hasher, spec.padding_policy.as_deref());
    hasher.finalize().into()
}

/// Hash a `ModelSpec`: weight identity, quantization, architecture knobs.
pub fn hash_model_spec(spec: &crate::types::ModelSpec) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-model-v1");
    hash_optional_32(&mut hasher, spec.weight_hash.as_ref());
    hash_optional_32(&mut hasher, spec.quant_hash.as_ref());
    hash_optional_32(&mut hasher, spec.rope_config_hash.as_ref());
    match spec.rmsnorm_eps {
        Some(eps) => {
            hasher.update(b"\x01");
            hasher.update(eps.to_le_bytes());
        }
        None => hasher.update(b"\x00"),
    }
    hash_optional_32(&mut hasher, spec.adapter_hash.as_ref());
    hash_optional_u32(&mut hasher, spec.n_layers);
    hash_optional_u32(&mut hasher, spec.hidden_dim);
    hash_optional_u32(&mut hasher, spec.vocab_size);
    hash_optional_32(&mut hasher, spec.embedding_merkle_root.as_ref());
    hash_optional_string(&mut hasher, spec.quant_family.as_deref());
    hash_optional_string(&mut hasher, spec.scale_derivation.as_deref());
    hash_optional_u32(&mut hasher, spec.quant_block_size);
    hash_optional_u32(&mut hasher, spec.kv_dim);
    hash_optional_u32(&mut hasher, spec.ffn_dim);
    hash_optional_u32(&mut hasher, spec.d_head);
    hash_optional_u32(&mut hasher, spec.n_q_heads);
    hash_optional_u32(&mut hasher, spec.n_kv_heads);
    hash_optional_f64(&mut hasher, spec.rope_theta);
    hasher.finalize().into()
}

/// Hash a `DecodeSpec`: sampling algorithm and parameters.
pub fn hash_decode_spec(spec: &crate::types::DecodeSpec) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-decode-v1");
    hasher.update(spec.temperature.to_le_bytes());
    hasher.update(spec.top_k.to_le_bytes());
    hasher.update(spec.top_p.to_le_bytes());
    hasher.update(spec.repetition_penalty.to_le_bytes());
    hasher.update(spec.frequency_penalty.to_le_bytes());
    hasher.update(spec.presence_penalty.to_le_bytes());
    hasher.update((spec.logit_bias.len() as u32).to_le_bytes());
    for &(token_id, bias) in &spec.logit_bias {
        hasher.update(token_id.to_le_bytes());
        hasher.update(bias.to_le_bytes());
    }
    hasher.update((spec.bad_word_ids.len() as u32).to_le_bytes());
    for &token_id in &spec.bad_word_ids {
        hasher.update(token_id.to_le_bytes());
    }
    hasher.update((spec.guided_decoding.len() as u32).to_le_bytes());
    hasher.update(spec.guided_decoding.as_bytes());
    match &spec.sampler_version {
        Some(sv) => {
            hasher.update(b"\x01");
            hasher.update((sv.len() as u32).to_le_bytes());
            hasher.update(sv.as_bytes());
        }
        None => hasher.update(b"\x00"),
    }
    hash_optional_string(&mut hasher, spec.decode_mode.as_deref());
    hasher.finalize().into()
}

/// Hash an `OutputSpec`: termination and post-processing rules.
pub fn hash_output_spec(spec: &crate::types::OutputSpec) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-output-v1");
    hasher.update(spec.eos_policy.as_bytes());
    hasher.update((spec.stop_sequences.len() as u32).to_le_bytes());
    for s in &spec.stop_sequences {
        hasher.update((s.len() as u32).to_le_bytes());
        hasher.update(s.as_bytes());
    }
    hasher.update(spec.max_tokens.to_le_bytes());
    hasher.update(spec.min_tokens.to_le_bytes());
    hasher.update([spec.ignore_eos as u8]);
    hash_optional_string(&mut hasher, spec.detokenization_policy.as_deref());
    match spec.eos_token_id {
        Some(id) => { hasher.update([1u8]); hasher.update(id.to_le_bytes()); }
        None => { hasher.update([0u8]); }
    }
    hasher.finalize().into()
}

/// Compose four spec hashes into the manifest commitment:
///
///   M = H("vi-manifest-v4" || H_input || H_model || H_decode || H_output)
pub fn hash_manifest_composed(
    h_input: [u8; 32],
    h_model: [u8; 32],
    h_decode: [u8; 32],
    h_output: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-manifest-v4");
    hasher.update(h_input);
    hasher.update(h_model);
    hasher.update(h_decode);
    hasher.update(h_output);
    hasher.finalize().into()
}

/// Compute the manifest hash from a `DeploymentManifest` via the four-spec split.
///
/// This is the canonical path: splits the flat manifest into four specs,
/// hashes each independently, and composes them into M.
pub fn hash_manifest(manifest: &crate::types::DeploymentManifest) -> [u8; 32] {
    let (input, model, decode, output) = manifest.split();
    hash_manifest_composed(
        hash_input_spec(&input),
        hash_model_spec(&model),
        hash_decode_spec(&decode),
        hash_output_spec(&output),
    )
}

fn hash_optional_32(hasher: &mut Sha256, opt: Option<&[u8; 32]>) {
    match opt {
        Some(h) => {
            hasher.update(b"\x01");
            hasher.update(h);
        }
        None => hasher.update(b"\x00"),
    }
}

fn hash_optional_u32(hasher: &mut Sha256, opt: Option<u32>) {
    match opt {
        Some(v) => {
            hasher.update(b"\x01");
            hasher.update(v.to_le_bytes());
        }
        None => hasher.update(b"\x00"),
    }
}

fn hash_optional_f64(hasher: &mut Sha256, opt: Option<f64>) {
    match opt {
        Some(v) => {
            hasher.update(b"\x01");
            hasher.update(v.to_le_bytes());
        }
        None => hasher.update(b"\x00"),
    }
}

fn hash_optional_string(hasher: &mut Sha256, opt: Option<&str>) {
    match opt {
        Some(s) => {
            hasher.update(b"\x01");
            hasher.update((s.len() as u32).to_le_bytes());
            hasher.update(s.as_bytes());
        }
        None => hasher.update(b"\x00"),
    }
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

/// Compute the IO chain genesis from the committed prompt hash.
///
/// Replaces the previous `[0u8; 32]` genesis, binding the IO chain start
/// to the specific request. Prevents cross-request IO chain splicing.
pub fn io_genesis_v4(prompt_hash: [u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-io-genesis-v4");
    hasher.update(prompt_hash);
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
                x_attn_i8: None,
                scale_x_attn: None,
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
                x_attn_i8: None,
                scale_x_attn: None,
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
                x_attn_i8: None,
                scale_x_attn: None,
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
                x_attn_i8: None,
                scale_x_attn: None,
            }],
        };
        let leaf_hash2 = hash_retained_state_direct(&state2);
        assert_ne!(io, io_hash_v4(leaf_hash2, 42, prev));
    }

    #[test]
    fn test_four_spec_hash_deterministic() {
        use crate::types::{InputSpec, ModelSpec, DecodeSpec, OutputSpec};

        let input = InputSpec {
            tokenizer_hash: [1u8; 32],
            system_prompt_hash: Some([2u8; 32]),
            chat_template_hash: None,
            bos_eos_policy: None,
            truncation_policy: None,
            special_token_policy: None,
            padding_policy: None,
        };
        let model = ModelSpec {
            weight_hash: Some([3u8; 32]),
            quant_hash: None,
            rope_config_hash: None,
            rmsnorm_eps: Some(1e-5),
            adapter_hash: None,
            n_layers: None,
            hidden_dim: None,
            vocab_size: None,
            embedding_merkle_root: None,
            quant_family: None,
            scale_derivation: None,
            quant_block_size: None,
            kv_dim: None,
            ffn_dim: None,
            d_head: None,
            n_q_heads: None,
            n_kv_heads: None,
            rope_theta: None,
        };
        let decode = DecodeSpec {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: vec![],
            bad_word_ids: vec![],
            guided_decoding: String::new(),
            sampler_version: None,
            decode_mode: None,
        };
        let output = OutputSpec {
            eos_policy: "stop".into(),
            stop_sequences: vec![],
            max_tokens: 100,
            min_tokens: 0,
            ignore_eos: false,
            detokenization_policy: None,
            eos_token_id: None,
        };

        let h1 = hash_manifest_composed(
            hash_input_spec(&input),
            hash_model_spec(&model),
            hash_decode_spec(&decode),
            hash_output_spec(&output),
        );
        let h2 = hash_manifest_composed(
            hash_input_spec(&input),
            hash_model_spec(&model),
            hash_decode_spec(&decode),
            hash_output_spec(&output),
        );
        assert_eq!(h1, h2, "same specs must produce same manifest hash");
    }

    #[test]
    fn test_four_spec_hash_differs_per_spec() {
        use crate::types::{InputSpec, ModelSpec, DecodeSpec, OutputSpec};

        let input = InputSpec {
            tokenizer_hash: [1u8; 32],
            system_prompt_hash: None,
            chat_template_hash: None,
            bos_eos_policy: None,
            truncation_policy: None,
            special_token_policy: None,
            padding_policy: None,
        };
        let model = ModelSpec {
            weight_hash: None,
            quant_hash: None,
            rope_config_hash: None,
            rmsnorm_eps: None,
            adapter_hash: None,
            n_layers: None,
            hidden_dim: None,
            vocab_size: None,
            embedding_merkle_root: None,
            quant_family: None,
            scale_derivation: None,
            quant_block_size: None,
            kv_dim: None,
            ffn_dim: None,
            d_head: None,
            n_q_heads: None,
            n_kv_heads: None,
            rope_theta: None,
        };
        let decode = DecodeSpec {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: vec![],
            bad_word_ids: vec![],
            guided_decoding: String::new(),
            sampler_version: None,
            decode_mode: None,
        };
        let output = OutputSpec {
            eos_policy: "stop".into(),
            stop_sequences: vec![],
            max_tokens: 0,
            min_tokens: 0,
            ignore_eos: false,
            detokenization_policy: None,
            eos_token_id: None,
        };

        let base = hash_manifest_composed(
            hash_input_spec(&input),
            hash_model_spec(&model),
            hash_decode_spec(&decode),
            hash_output_spec(&output),
        );

        // Change only input spec.
        let input2 = InputSpec {
            tokenizer_hash: [2u8; 32],
            ..input.clone()
        };
        let changed_input = hash_manifest_composed(
            hash_input_spec(&input2),
            hash_model_spec(&model),
            hash_decode_spec(&decode),
            hash_output_spec(&output),
        );
        assert_ne!(base, changed_input, "different input spec must change M");

        // Change only decode spec.
        let decode2 = DecodeSpec {
            temperature: 0.5,
            ..decode.clone()
        };
        let changed_decode = hash_manifest_composed(
            hash_input_spec(&input),
            hash_model_spec(&model),
            hash_decode_spec(&decode2),
            hash_output_spec(&output),
        );
        assert_ne!(base, changed_decode, "different decode spec must change M");
    }

    #[test]
    fn test_manifest_split_round_trips_hash() {
        use crate::types::DeploymentManifest;

        let manifest = DeploymentManifest {
            tokenizer_hash: [42u8; 32],
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            eos_policy: "stop".into(),
            weight_hash: Some([99u8; 32]),
            quant_hash: None,
            system_prompt_hash: Some([77u8; 32]),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: vec![(5, 1.0)],
            bad_word_ids: vec![],
            guided_decoding: String::new(),
            stop_sequences: vec!["<|end|>".into()],
            max_tokens: 256,
            chat_template_hash: Some([55u8; 32]),
            rope_config_hash: None,
            rmsnorm_eps: Some(1e-5),
            sampler_version: Some("chacha20-vi-sample-v1".into()),
            bos_eos_policy: None,
            truncation_policy: None,
            special_token_policy: None,
            adapter_hash: None,
            n_layers: None,
            hidden_dim: None,
            vocab_size: None,
            embedding_merkle_root: None,
            quant_family: None,
            scale_derivation: None,
            quant_block_size: None,
            kv_dim: None,
            ffn_dim: None,
            d_head: None,
            n_q_heads: None,
            n_kv_heads: None,
            rope_theta: None,
            min_tokens: 0,
            ignore_eos: false,
            detokenization_policy: None,
            eos_token_id: None,
            padding_policy: None,
            decode_mode: None,
        };

        // hash_manifest splits internally, so this tests the round-trip.
        let h1 = hash_manifest(&manifest);

        // Manually split and compose.
        let (input, model, decode, output) = manifest.split();
        let h2 = hash_manifest_composed(
            hash_input_spec(&input),
            hash_model_spec(&model),
            hash_decode_spec(&decode),
            hash_output_spec(&output),
        );

        assert_eq!(h1, h2, "hash_manifest must equal manual split+compose");
    }

    #[test]
    fn test_quant_family_changes_model_spec_hash() {
        use crate::types::ModelSpec;

        let base = ModelSpec {
            weight_hash: None, quant_hash: None, rope_config_hash: None,
            rmsnorm_eps: None, adapter_hash: None,
            n_layers: None, hidden_dim: None, vocab_size: None,
            embedding_merkle_root: None,
            quant_family: None, scale_derivation: None, quant_block_size: None,
            kv_dim: None, ffn_dim: None, d_head: None,
            n_q_heads: None, n_kv_heads: None, rope_theta: None,
        };
        let with_family = ModelSpec {
            quant_family: Some("W8A8".into()),
            ..base.clone()
        };
        let h_base = hash_model_spec(&base);
        let h_with = hash_model_spec(&with_family);
        assert_ne!(h_base, h_with, "quant_family must affect model spec hash");

        // Different family values must differ.
        let other_family = ModelSpec {
            quant_family: Some("GPTQ".into()),
            ..base.clone()
        };
        assert_ne!(h_with, hash_model_spec(&other_family),
            "different quant_family values must produce different hashes");
    }

    #[test]
    fn test_scale_derivation_changes_model_spec_hash() {
        use crate::types::ModelSpec;

        let base = ModelSpec {
            weight_hash: None, quant_hash: None, rope_config_hash: None,
            rmsnorm_eps: None, adapter_hash: None,
            n_layers: None, hidden_dim: None, vocab_size: None,
            embedding_merkle_root: None,
            quant_family: None, scale_derivation: None, quant_block_size: None,
            kv_dim: None, ffn_dim: None, d_head: None,
            n_q_heads: None, n_kv_heads: None, rope_theta: None,
        };
        let with_sd = ModelSpec {
            scale_derivation: Some("absmax".into()),
            ..base.clone()
        };
        assert_ne!(hash_model_spec(&base), hash_model_spec(&with_sd),
            "scale_derivation must affect model spec hash");

        let other_sd = ModelSpec {
            scale_derivation: Some("zeropoint".into()),
            ..base.clone()
        };
        assert_ne!(hash_model_spec(&with_sd), hash_model_spec(&other_sd),
            "different scale_derivation values must produce different hashes");
    }

    #[test]
    fn test_quant_block_size_changes_model_spec_hash() {
        use crate::types::ModelSpec;

        let base = ModelSpec {
            weight_hash: None, quant_hash: None, rope_config_hash: None,
            rmsnorm_eps: None, adapter_hash: None,
            n_layers: None, hidden_dim: None, vocab_size: None,
            embedding_merkle_root: None,
            quant_family: None, scale_derivation: None, quant_block_size: None,
            kv_dim: None, ffn_dim: None, d_head: None,
            n_q_heads: None, n_kv_heads: None, rope_theta: None,
        };
        let with_bs = ModelSpec {
            quant_block_size: Some(32),
            ..base.clone()
        };
        assert_ne!(hash_model_spec(&base), hash_model_spec(&with_bs),
            "quant_block_size must affect model spec hash");

        let other_bs = ModelSpec {
            quant_block_size: Some(128),
            ..base.clone()
        };
        assert_ne!(hash_model_spec(&with_bs), hash_model_spec(&other_bs),
            "different quant_block_size values must produce different hashes");
    }
}
