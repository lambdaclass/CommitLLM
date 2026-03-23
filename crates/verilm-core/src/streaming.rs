//! Streaming KV commitment verifier.
//!
//! During inference the prover streams a 32-byte KV root per token.
//! The root is a Merkle tree over per-layer KV hashes:
//!
//! ```text
//! leaf_l = SHA256("vi-kv-layer-v1" || token_index_le || layer_index_le || K_{t,l} || V_{t,l})
//! root_t = MerkleRoot(leaf_0, leaf_1, ..., leaf_{L-1})
//! ```
//!
//! At audit time the prover opens individual layers with Merkle proofs;
//! the verifier recomputes the layer hash and checks the proof against the
//! stored root. This allows routine audits (10/80 layers) to verify KV
//! commitments without needing all layers' data.

use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::merkle::{self, MerkleProof, MerkleTree};

/// Compute the KV hash for a single layer at a single token position.
pub fn compute_kv_layer_hash(
    token_index: u32,
    layer_index: u32,
    k: &[i8],
    v: &[i8],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-kv-layer-v1");
    hasher.update(token_index.to_le_bytes());
    hasher.update(layer_index.to_le_bytes());
    // Safety: i8 and u8 have identical layout
    let k_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len()) };
    let v_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
    hasher.update(k_bytes);
    hasher.update(v_bytes);
    hasher.finalize().into()
}

/// Build a Merkle tree over per-layer KV hashes for a single token.
///
/// Returns the full tree (needed for proof generation at audit time).
/// For commit-time use where only the root is needed, prefer [`kv_layer_root`].
pub fn build_kv_layer_tree<K: AsRef<[i8]>, V: AsRef<[i8]>>(
    token_index: u32,
    k_per_layer: &[K],
    v_per_layer: &[V],
) -> MerkleTree {
    let leaves = kv_layer_leaves(token_index, k_per_layer, v_per_layer);
    merkle::build_tree(&leaves)
}

/// Compute only the Merkle root over per-layer KV hashes for a single token.
///
/// Same hash as `build_kv_layer_tree(...).root` but avoids storing the
/// full tree. Use at commit time when proofs are not yet needed.
pub fn kv_layer_root<K: AsRef<[i8]>, V: AsRef<[i8]>>(
    token_index: u32,
    k_per_layer: &[K],
    v_per_layer: &[V],
) -> [u8; 32] {
    let leaves = kv_layer_leaves(token_index, k_per_layer, v_per_layer);
    merkle::compute_root(&leaves)
}

/// Shared: compute per-layer KV leaf hashes.
fn kv_layer_leaves<K: AsRef<[i8]>, V: AsRef<[i8]>>(
    token_index: u32,
    k_per_layer: &[K],
    v_per_layer: &[V],
) -> Vec<[u8; 32]> {
    assert_eq!(
        k_per_layer.len(),
        v_per_layer.len(),
        "k_per_layer and v_per_layer must have the same number of layers"
    );

    k_per_layer
        .iter()
        .zip(v_per_layer.iter())
        .enumerate()
        .map(|(l, (k, v))| compute_kv_layer_hash(token_index, l as u32, k.as_ref(), v.as_ref()))
        .collect()
}

/// Compute the streaming KV commitment for a single token (flat hash, legacy).
///
/// This is the original all-layers-at-once hash. Kept for backward
/// compatibility with existing tests. New code should prefer
/// [`build_kv_layer_tree`] which enables per-layer opening.
pub fn compute_kv_commitment<K: AsRef<[i8]>, V: AsRef<[i8]>>(
    token_index: u32,
    k_per_layer: &[K],
    v_per_layer: &[V],
) -> [u8; 32] {
    assert_eq!(
        k_per_layer.len(),
        v_per_layer.len(),
        "k_per_layer and v_per_layer must have the same number of layers"
    );

    let mut hasher = Sha256::new();
    hasher.update(b"vi-kv-stream-v1");
    hasher.update(token_index.to_le_bytes());

    for (k, v) in k_per_layer.iter().zip(v_per_layer.iter()) {
        let k = k.as_ref();
        let v = v.as_ref();
        let k_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len()) };
        let v_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
        hasher.update(k_bytes);
        hasher.update(v_bytes);
    }

    hasher.finalize().into()
}

/// Streaming KV verifier state.
///
/// Accumulates per-token KV roots during inference. At audit time,
/// verifies that opened layers match their committed hashes via Merkle proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingKvVerifier {
    /// Per-token KV roots (Merkle root over layer hashes), indexed by token position.
    commitments: Vec<[u8; 32]>,
}

impl StreamingKvVerifier {
    /// Create a new empty verifier.
    pub fn new() -> Self {
        Self {
            commitments: Vec::new(),
        }
    }

    /// Ingest a KV commitment (Merkle root) for the next token.
    /// Commitments must be ingested in order (token 0, 1, 2, ...).
    pub fn ingest(&mut self, commitment: [u8; 32]) {
        self.commitments.push(commitment);
    }

    /// Number of tokens ingested so far.
    pub fn len(&self) -> usize {
        self.commitments.len()
    }

    /// Returns `true` if no commitments have been ingested.
    pub fn is_empty(&self) -> bool {
        self.commitments.is_empty()
    }

    /// Get the commitment (root) for a specific token index.
    pub fn get(&self, token_index: u32) -> Option<&[u8; 32]> {
        self.commitments.get(token_index as usize)
    }

    /// Verify a single opened layer's KV data against the stored Merkle root.
    ///
    /// The verifier recomputes `leaf_hash = compute_kv_layer_hash(...)` and
    /// checks the Merkle proof against the stored root for that token.
    pub fn verify_layer_opening(
        &self,
        token_index: u32,
        layer_index: u32,
        k: &[i8],
        v: &[i8],
        proof: &MerkleProof,
    ) -> Result<(), String> {
        let root = match self.get(token_index) {
            Some(r) => r,
            None => {
                return Err(format!(
                    "token {}: no commitment stored (only {} tokens ingested)",
                    token_index,
                    self.len()
                ));
            }
        };

        let leaf_hash = compute_kv_layer_hash(token_index, layer_index, k, v);
        if !merkle::verify(root, &leaf_hash, proof) {
            return Err(format!(
                "token {} layer {}: KV Merkle proof verification failed",
                token_index, layer_index
            ));
        }

        Ok(())
    }

    /// Verify that opened KV data for a set of token positions matches
    /// the streaming commitments (legacy flat-hash verification).
    ///
    /// `opened_kv` maps `token_index -> (k_per_layer, v_per_layer)`.
    /// Returns failure descriptions (empty = pass).
    pub fn verify_opened_kv(
        &self,
        opened_kv: &BTreeMap<u32, (&[Vec<i8>], &[Vec<i8>])>,
    ) -> Vec<String> {
        let mut failures = Vec::new();

        for (&token_index, &(k_per_layer, v_per_layer)) in opened_kv {
            let stored = match self.get(token_index) {
                Some(c) => c,
                None => {
                    failures.push(format!(
                        "token {}: no commitment stored (only {} tokens ingested)",
                        token_index,
                        self.len()
                    ));
                    continue;
                }
            };

            let recomputed = compute_kv_commitment(token_index, k_per_layer, v_per_layer);
            if recomputed != *stored {
                failures.push(format!(
                    "token {}: commitment mismatch (recomputed {:02x?} != stored {:02x?})",
                    token_index,
                    &recomputed[..4],
                    &stored[..4],
                ));
            }
        }

        failures
    }
}

impl Default for StreamingKvVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_kv(layers: usize, dim: usize, seed: u8) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
        let k: Vec<Vec<i8>> = (0..layers)
            .map(|l| (0..dim).map(|i| ((seed as usize + l * dim + i) % 256) as i8).collect())
            .collect();
        let v: Vec<Vec<i8>> = (0..layers)
            .map(|l| (0..dim).map(|i| ((seed as usize + 128 + l * dim + i) % 256) as i8).collect())
            .collect();
        (k, v)
    }

    // --- Legacy flat-hash tests ---

    #[test]
    fn commitment_is_deterministic() {
        let (k, v) = dummy_kv(2, 4, 42);
        let c1 = compute_kv_commitment(0, &k, &v);
        let c2 = compute_kv_commitment(0, &k, &v);
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_token_index_different_commitment() {
        let (k, v) = dummy_kv(2, 4, 42);
        let c0 = compute_kv_commitment(0, &k, &v);
        let c1 = compute_kv_commitment(1, &k, &v);
        assert_ne!(c0, c1);
    }

    #[test]
    fn ingest_and_verify_pass() {
        let (k0, v0) = dummy_kv(2, 4, 0);
        let (k1, v1) = dummy_kv(2, 4, 1);

        let mut verifier = StreamingKvVerifier::new();
        verifier.ingest(compute_kv_commitment(0, &k0, &v0));
        verifier.ingest(compute_kv_commitment(1, &k1, &v1));
        assert_eq!(verifier.len(), 2);
        assert!(!verifier.is_empty());

        let mut opened = BTreeMap::new();
        opened.insert(0u32, (k0.as_slice(), v0.as_slice()));
        opened.insert(1u32, (k1.as_slice(), v1.as_slice()));

        let failures = verifier.verify_opened_kv(&opened);
        assert!(failures.is_empty(), "expected pass, got: {:?}", failures);
    }

    #[test]
    fn tampered_kv_detected() {
        let (k0, v0) = dummy_kv(2, 4, 0);

        let mut verifier = StreamingKvVerifier::new();
        verifier.ingest(compute_kv_commitment(0, &k0, &v0));

        let (k_bad, v_bad) = dummy_kv(2, 4, 99);
        let mut opened = BTreeMap::new();
        opened.insert(0u32, (k_bad.as_slice(), v_bad.as_slice()));

        let failures = verifier.verify_opened_kv(&opened);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("commitment mismatch"));
    }

    #[test]
    fn out_of_range_token_detected() {
        let verifier = StreamingKvVerifier::new();
        let (k, v) = dummy_kv(1, 2, 0);
        let mut opened = BTreeMap::new();
        opened.insert(5u32, (k.as_slice(), v.as_slice()));

        let failures = verifier.verify_opened_kv(&opened);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("no commitment stored"));
    }

    #[test]
    fn empty_verifier() {
        let v = StreamingKvVerifier::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert!(v.get(0).is_none());
    }

    // --- Per-layer Merkle tree tests ---

    #[test]
    fn layer_hash_deterministic() {
        let k = vec![1i8, 2, 3, 4];
        let v = vec![5i8, 6, 7, 8];
        let h1 = compute_kv_layer_hash(0, 0, &k, &v);
        let h2 = compute_kv_layer_hash(0, 0, &k, &v);
        assert_eq!(h1, h2);
    }

    #[test]
    fn layer_hash_differs_by_layer_index() {
        let k = vec![1i8, 2, 3, 4];
        let v = vec![5i8, 6, 7, 8];
        let h0 = compute_kv_layer_hash(0, 0, &k, &v);
        let h1 = compute_kv_layer_hash(0, 1, &k, &v);
        assert_ne!(h0, h1);
    }

    #[test]
    fn layer_hash_differs_by_token_index() {
        let k = vec![1i8, 2, 3, 4];
        let v = vec![5i8, 6, 7, 8];
        let h0 = compute_kv_layer_hash(0, 0, &k, &v);
        let h1 = compute_kv_layer_hash(1, 0, &k, &v);
        assert_ne!(h0, h1);
    }

    #[test]
    fn build_tree_and_verify_layer_opening() {
        let (k, v) = dummy_kv(4, 8, 42);
        let tree = build_kv_layer_tree(0, &k, &v);

        let mut verifier = StreamingKvVerifier::new();
        verifier.ingest(tree.root);

        // Verify each layer individually
        for l in 0..4u32 {
            let proof = merkle::prove(&tree, l as usize);
            let result = verifier.verify_layer_opening(0, l, &k[l as usize], &v[l as usize], &proof);
            assert!(result.is_ok(), "layer {} failed: {:?}", l, result);
        }
    }

    #[test]
    fn tampered_layer_opening_detected() {
        let (k, v) = dummy_kv(4, 8, 42);
        let tree = build_kv_layer_tree(0, &k, &v);

        let mut verifier = StreamingKvVerifier::new();
        verifier.ingest(tree.root);

        // Tampered K data for layer 1
        let mut k_bad = k[1].clone();
        k_bad[0] ^= 0x7F;

        let proof = merkle::prove(&tree, 1);
        let result = verifier.verify_layer_opening(0, 1, &k_bad, &v[1], &proof);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Merkle proof verification failed"));
    }

    #[test]
    fn wrong_layer_proof_detected() {
        let (k, v) = dummy_kv(4, 8, 42);
        let tree = build_kv_layer_tree(0, &k, &v);

        let mut verifier = StreamingKvVerifier::new();
        verifier.ingest(tree.root);

        // Use proof for layer 0 but claim layer 1
        let proof = merkle::prove(&tree, 0);
        let result = verifier.verify_layer_opening(0, 1, &k[1], &v[1], &proof);
        assert!(result.is_err());
    }

    #[test]
    fn layer_opening_no_commitment() {
        let verifier = StreamingKvVerifier::new();
        let k = vec![1i8, 2, 3, 4];
        let v = vec![5i8, 6, 7, 8];
        let proof = MerkleProof {
            leaf_index: 0,
            siblings: Vec::new(),
        };
        let result = verifier.verify_layer_opening(0, 0, &k, &v, &proof);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no commitment stored"));
    }
}
