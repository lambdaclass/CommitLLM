//! Binary serialization for verifier keys and traces.
//!
//! Uses bincode for simplicity. The format is versioned so we can
//! change it later without breaking existing files.
//!
//! # Serialization layers
//!
//! There are two independent axes:
//!
//! **Schema** — full (`LayerTrace`) vs compact (`CompactLayerTrace`).
//! Compact omits derivable fields (`x_ffn`, `h`); the verifier
//! reconstructs them before checking the Merkle leaf hash.
//!
//! **Transport** — raw bytes vs zstd-compressed bytes.
//! Compression is a separate layer applied on top of any serialized
//! payload. It is not part of the canonical format.

use crate::types::{AuditResponse, BatchProof, CompactAuditResponse, CompactBatchProof, TokenTrace, V4AuditResponse, VerifierKey};

const KEY_MAGIC: &[u8; 4] = b"VKEY";
const TRACE_MAGIC: &[u8; 4] = b"VTRC";
const BATCH_MAGIC: &[u8; 4] = b"VBAT";
const COMPACT_BATCH_MAGIC: &[u8; 4] = b"VCBT";
const COMPACT_AUDIT_MAGIC: &[u8; 4] = b"VCAR";
const V4_AUDIT_MAGIC: &[u8; 4] = b"VV4A";

// ── Verifier key ────────────────────────────────────────────

pub fn serialize_key(key: &VerifierKey) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(KEY_MAGIC);
    let encoded = bincode::serialize(key).expect("key serialization failed");
    buf.extend_from_slice(&encoded);
    buf
}

pub fn deserialize_key(data: &[u8]) -> Result<VerifierKey, String> {
    if data.len() < 4 || &data[..4] != KEY_MAGIC {
        return Err("invalid key file magic".into());
    }
    bincode::deserialize(&data[4..]).map_err(|e| format!("key deserialization failed: {e}"))
}

// ── Single token trace ──────────────────────────────────────

pub fn serialize_trace(trace: &TokenTrace) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(TRACE_MAGIC);
    let encoded = bincode::serialize(trace).expect("trace serialization failed");
    buf.extend_from_slice(&encoded);
    buf
}

pub fn deserialize_trace(data: &[u8]) -> Result<TokenTrace, String> {
    if data.len() < 4 || &data[..4] != TRACE_MAGIC {
        return Err("invalid trace file magic".into());
    }
    bincode::deserialize(&data[4..]).map_err(|e| format!("trace deserialization failed: {e}"))
}

// ── Batch proof (full schema) ───────────────────────────────

pub fn serialize_batch(proof: &BatchProof) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(BATCH_MAGIC);
    let encoded = bincode::serialize(proof).expect("batch serialization failed");
    buf.extend_from_slice(&encoded);
    buf
}

pub fn deserialize_batch(data: &[u8]) -> Result<BatchProof, String> {
    if data.len() < 4 || &data[..4] != BATCH_MAGIC {
        return Err("invalid batch file magic".into());
    }
    bincode::deserialize(&data[4..]).map_err(|e| format!("batch deserialization failed: {e}"))
}

// ── Batch proof (compact schema) ────────────────────────────
//
// Omits `x_ffn` and `h` from each layer. The verifier reconstructs
// them via `CompactBatchProof::to_full()` before Merkle verification.

pub fn serialize_compact_batch(proof: &BatchProof) -> Vec<u8> {
    let compact = CompactBatchProof::from_full(proof);
    let mut buf = Vec::new();
    buf.extend_from_slice(COMPACT_BATCH_MAGIC);
    let encoded = bincode::serialize(&compact).expect("compact batch serialization failed");
    buf.extend_from_slice(&encoded);
    buf
}

pub fn deserialize_compact_batch(data: &[u8]) -> Result<BatchProof, String> {
    if data.len() < 4 || &data[..4] != COMPACT_BATCH_MAGIC {
        return Err("invalid compact batch file magic".into());
    }
    let compact: CompactBatchProof = bincode::deserialize(&data[4..])
        .map_err(|e| format!("compact batch deserialization failed: {e}"))?;
    compact.to_full()
}

// ── Audit response (compact schema) ──────────────────────────
//
// Omits `x_ffn` and `h` from each layer, same as compact batch.

pub fn serialize_compact_audit(resp: &AuditResponse) -> Vec<u8> {
    let compact = CompactAuditResponse::from_full(resp);
    let mut buf = Vec::new();
    buf.extend_from_slice(COMPACT_AUDIT_MAGIC);
    let encoded = bincode::serialize(&compact).expect("compact audit serialization failed");
    buf.extend_from_slice(&encoded);
    buf
}

pub fn deserialize_compact_audit(data: &[u8]) -> Result<AuditResponse, String> {
    if data.len() < 4 || &data[..4] != COMPACT_AUDIT_MAGIC {
        return Err("invalid compact audit file magic".into());
    }
    let compact: CompactAuditResponse = bincode::deserialize(&data[4..])
        .map_err(|e| format!("compact audit deserialization failed: {e}"))?;
    compact.to_full()
}

// ── V4 audit response (bincode + zstd) ──────────────────────
//
// Binary wire format for V4AuditResponse. Replaces JSON for
// production use — JSON remains available for debugging.

pub fn serialize_v4_audit(resp: &V4AuditResponse) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(V4_AUDIT_MAGIC);
    let encoded = bincode::serialize(resp).expect("V4 audit serialization failed");
    let compressed = zstd::encode_all(encoded.as_slice(), 3).expect("zstd compression failed");
    buf.extend_from_slice(&compressed);
    buf
}

pub fn deserialize_v4_audit(data: &[u8]) -> Result<V4AuditResponse, String> {
    if data.len() < 4 || &data[..4] != V4_AUDIT_MAGIC {
        return Err("invalid V4 audit magic".into());
    }
    let decompressed =
        zstd::decode_all(&data[4..]).map_err(|e| format!("zstd decompression failed: {e}"))?;
    bincode::deserialize(&decompressed)
        .map_err(|e| format!("V4 audit deserialization failed: {e}"))
}

// ── Transport compression (independent of schema) ───────────
//
// These work on arbitrary byte slices. Apply on top of any
// serialize_* output. Not part of the canonical format.

/// Compress bytes with zstd (level 3).
pub fn compress(data: &[u8]) -> Vec<u8> {
    zstd::encode_all(data, 3).expect("zstd compression failed")
}

/// Decompress zstd-compressed bytes.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    zstd::decode_all(data).map_err(|e| format!("zstd decompression failed: {e}"))
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::ModelConfig;
    use crate::field::Fp;
    use crate::merkle::MerkleProof;
    use crate::types::LayerTrace;

    #[test]
    fn test_key_roundtrip() {
        let key = VerifierKey {
            version: 1,
            config: ModelConfig::toy(),
            seed: [42u8; 32],
            source_dtype: "I8".into(),
            quantization_scales: Vec::new(),
            r_vectors: vec![vec![Fp(1), Fp(2)]; 7],
            v_vectors: vec![vec![vec![Fp(10), Fp(20)]; 7]; 2],
            wo_norms: Vec::new(),
            max_v_norm: 0.0,
            lm_head: None,
            weight_hash: None,
            rmsnorm_attn_weights: Vec::new(),
            rmsnorm_ffn_weights: Vec::new(),
            weight_scales: Vec::new(),
            rmsnorm_eps: 1e-5,
            embedding_merkle_root: None,
            final_norm_weights: None,
        };
        let data = serialize_key(&key);
        let key2 = deserialize_key(&data).unwrap();
        assert_eq!(key.version, key2.version);
        assert_eq!(key.config.name, key2.config.name);
        assert_eq!(key.r_vectors, key2.r_vectors);
    }

    #[test]
    fn test_trace_roundtrip() {
        let trace = TokenTrace {
            token_index: 42,
            layers: vec![LayerTrace {
                x_attn: vec![1i8, 2, 3],
                q: vec![4i32, 5, 6],
                k: vec![7i32],
                v: vec![8i32],
                a: vec![9i8, 10, 11],
                attn_out: vec![12i32, 13, 14],
                x_ffn: vec![15i8, 16, 17],
                g: vec![18i32, 19],
                u: vec![20i32, 21],
                h: vec![22i8, 23],
                ffn_out: vec![24i32, 25, 26],
                kv_cache_k: Vec::new(),
                kv_cache_v: Vec::new(),
                scale_x_attn: None,
                scale_a: None,
                scale_x_ffn: None,
                scale_h: None,
                residual: None,
            }],
            merkle_root: [0xbb; 32],
            merkle_proof: MerkleProof {
                leaf_index: 42,
                siblings: vec![[0xaa; 32]],
            },
            io_proof: MerkleProof {
                leaf_index: 42,
                siblings: vec![[0xcc; 32]],
            },
            margin_cert: None,
            final_hidden: None,
            token_id: None,
            prev_io_hash: None,
            prev_kv_hash: None,
        };
        let data = serialize_trace(&trace);
        let trace2 = deserialize_trace(&data).unwrap();
        assert_eq!(trace.token_index, trace2.token_index);
        assert_eq!(trace.layers[0].q, trace2.layers[0].q);
    }

    #[test]
    fn test_bad_magic() {
        assert!(deserialize_key(b"BAAD1234").is_err());
        assert!(deserialize_trace(b"BAAD1234").is_err());
    }

    #[test]
    fn test_compact_batch_roundtrip() {
        use crate::types::{BatchCommitment, CompactLayerTrace};

        let g = vec![3i32, 5];
        let u = vec![4i32, -2];
        let h = crate::silu::compute_h_unit_scale(&g, &u);

        let lt = LayerTrace {
            x_attn: vec![1i8, 2, 3],
            q: vec![4i32, 5, 6],
            k: vec![7i32],
            v: vec![8i32],
            a: vec![9i8, 10, 11],
            attn_out: vec![12i32, 13, 14],
            x_ffn: vec![12i8, 13, 14], // = requantize(attn_out)
            g,
            u,
            h, // SiLU(3)*4=11, SiLU(5)*(-2)=-10
            ffn_out: vec![24i32, 25, 26],
            kv_cache_k: Vec::new(),
            kv_cache_v: Vec::new(),
            scale_x_attn: None,
            scale_a: None,
            scale_x_ffn: None,
            scale_h: None,
            residual: None,
        };

        // Verify reconstruction matches
        let compact = CompactLayerTrace::from_full(&lt);
        let reconstructed = compact.to_full().expect("to_full should succeed for valid trace");
        assert_eq!(lt.x_ffn, reconstructed.x_ffn, "x_ffn reconstruction mismatch");
        assert_eq!(lt.h, reconstructed.h, "h reconstruction mismatch");

        let proof = BatchProof {
            commitment: BatchCommitment {
                merkle_root: [0xaa; 32],
                io_root: [0xbb; 32],
                n_tokens: 1,
                manifest_hash: None,
                version: Default::default(),
                prompt_hash: None,
                seed_commitment: None,
                kv_chain_root: None,
            },
            revealed_seed: None,
            traces: vec![TokenTrace {
                token_index: 0,
                layers: vec![lt],
                merkle_root: [0xaa; 32],
                merkle_proof: MerkleProof {
                    leaf_index: 0,
                    siblings: vec![[0xcc; 32]],
                },
                io_proof: MerkleProof {
                    leaf_index: 0,
                    siblings: vec![[0xdd; 32]],
                },
                margin_cert: None,
                final_hidden: None,
                token_id: None,
                prev_io_hash: None,
                prev_kv_hash: None,
            }],
        };

        // Compact roundtrip (no compression)
        let compact_data = serialize_compact_batch(&proof);
        let restored = deserialize_compact_batch(&compact_data).unwrap();
        assert_eq!(proof.traces[0].layers[0].q, restored.traces[0].layers[0].q);
        assert_eq!(proof.traces[0].layers[0].x_ffn, restored.traces[0].layers[0].x_ffn);
        assert_eq!(proof.traces[0].layers[0].h, restored.traces[0].layers[0].h);

        // Compact + compression roundtrip (composed layers)
        let compressed = compress(&compact_data);
        let decompressed = decompress(&compressed).unwrap();
        let restored2 = deserialize_compact_batch(&decompressed).unwrap();
        assert_eq!(proof.traces[0].layers[0].q, restored2.traces[0].layers[0].q);
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let data = b"hello world, this is some test data for compression";
        let compressed = compress(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_compact_audit_roundtrip() {
        use crate::types::AuditResponse;

        let g = vec![3i32, 5];
        let u = vec![4i32, -2];
        let h = crate::silu::compute_h_unit_scale(&g, &u);

        let lt = LayerTrace {
            x_attn: vec![1i8, 2, 3],
            q: vec![4i32, 5, 6],
            k: vec![7i32],
            v: vec![8i32],
            a: vec![9i8, 10, 11],
            attn_out: vec![12i32, 13, 14],
            x_ffn: vec![12i8, 13, 14],
            g,
            u,
            h,
            ffn_out: vec![24i32, 25, 26],
            kv_cache_k: vec![vec![1i8, 2]],
            kv_cache_v: vec![vec![3i8, 4]],
            scale_x_attn: None,
            scale_a: None,
            scale_x_ffn: None,
            scale_h: None,
            residual: None,
        };

        let resp = AuditResponse {
            token_index: 2,
            partial_layers: vec![lt],
            layer_indices: vec![3],
            kv_k_prefix: vec![vec![vec![1i8, 2]]],
            kv_v_prefix: vec![vec![vec![3i8, 4]]],
            kv_layer_proofs: vec![MerkleProof {
                leaf_index: 3,
                siblings: vec![[0xaa; 32]],
            }],
            merkle_root: [0xbb; 32],
            merkle_proof: MerkleProof {
                leaf_index: 2,
                siblings: vec![[0xcc; 32]],
            },
            final_hidden: None,
            token_id: Some(42),
        };

        // Raw roundtrip
        let data = serialize_compact_audit(&resp);
        let restored = deserialize_compact_audit(&data).unwrap();
        assert_eq!(restored.token_index, 2);
        assert_eq!(restored.partial_layers[0].q, vec![4i32, 5, 6]);
        assert_eq!(restored.partial_layers[0].x_ffn, vec![12i8, 13, 14]);
        assert_eq!(restored.partial_layers[0].h, resp.partial_layers[0].h);
        assert_eq!(restored.layer_indices, vec![3]);
        assert_eq!(restored.token_id, Some(42));

        // Compressed roundtrip
        let compressed = compress(&data);
        let decompressed = decompress(&compressed).unwrap();
        let restored2 = deserialize_compact_audit(&decompressed).unwrap();
        assert_eq!(restored2.token_index, 2);
        assert_eq!(restored2.partial_layers[0].q, vec![4i32, 5, 6]);
    }

    #[test]
    fn test_compact_audit_bad_magic() {
        assert!(deserialize_compact_audit(b"BAAD1234").is_err());
    }

    #[test]
    fn test_v4_audit_roundtrip() {
        use crate::types::{
            BatchCommitment, CommitmentVersion, RetainedLayerState, RetainedTokenState,
            ShellLayerOpening, ShellTokenOpening, V4AuditResponse,
        };

        let retained = RetainedTokenState {
            layers: vec![RetainedLayerState {
                a: vec![1i8, 2, 3],
                scale_a: 0.5,
                scale_x_attn: 0.25,
                scale_x_ffn: 0.125,
                scale_h: 0.0625,
            }],
        };
        let shell = ShellTokenOpening {
            layers: vec![ShellLayerOpening {
                attn_out: vec![10i32, 20],
                g: vec![30i32],
                u: vec![40i32],
                ffn_out: vec![50i32, 60],
                q: None,
                k: None,
                v: None,
            }],
            layer_indices: None,
            initial_residual: None,
            embedding_proof: None,
        };
        let resp = V4AuditResponse {
            token_index: 7,
            retained: retained.clone(),
            merkle_proof: MerkleProof {
                leaf_index: 7,
                siblings: vec![[0xaa; 32]],
            },
            io_proof: MerkleProof {
                leaf_index: 7,
                siblings: vec![[0xbb; 32]],
            },
            token_id: 42,
            prev_io_hash: [0xcc; 32],
            prefix_leaf_hashes: vec![],
            prefix_merkle_proofs: vec![],
            prefix_token_ids: vec![],
            commitment: BatchCommitment {
                merkle_root: [0xdd; 32],
                io_root: [0xee; 32],
                n_tokens: 8,
                manifest_hash: None,
                version: CommitmentVersion::V4,
                prompt_hash: None,
                seed_commitment: None,
                kv_chain_root: None,
            },
            revealed_seed: [0xff; 32],
            shell_opening: Some(shell),
        };

        let binary = serialize_v4_audit(&resp);
        assert_eq!(&binary[..4], V4_AUDIT_MAGIC);

        let restored = deserialize_v4_audit(&binary).unwrap();
        assert_eq!(restored.token_index, 7);
        assert_eq!(restored.token_id, 42);
        assert_eq!(restored.retained, retained);
        assert_eq!(restored.commitment.n_tokens, 8);
        let shell_r = restored.shell_opening.unwrap();
        assert_eq!(shell_r.layers[0].attn_out, vec![10i32, 20]);

        // Verify binary is reasonably compact (magic + compressed bincode).
        // Raw bincode of this small struct should be ~200 bytes; zstd adds overhead
        // for tiny payloads but stays under 500.
        assert!(
            binary.len() < 500,
            "binary too large: {} bytes",
            binary.len()
        );
    }

    #[test]
    fn test_v4_audit_bad_magic() {
        assert!(deserialize_v4_audit(b"BAAD1234").is_err());
    }
}
