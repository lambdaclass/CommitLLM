//! Binary serialization for verifier keys and V4 audit responses.
//!
//! This is the **normative** receipt/audit representation. The binary
//! format (magic + bincode + zstd) is the canonical wire format for
//! production use. JSON serialization via serde derives exists for
//! debugging and development only — it is not a production path.
//!
//! Uses bincode for simplicity. The format is versioned so we can
//! change it later without breaking existing files.
//!
//! **Transport** — raw bytes vs zstd-compressed bytes.
//! Compression is a separate layer applied on top of any serialized
//! payload. It is not part of the canonical format.

use crate::types::{DecodeArtifact, V4AuditResponse, VerifierKey};

const KEY_MAGIC: &[u8; 4] = b"VKEY";
const V4_AUDIT_MAGIC: &[u8; 4] = b"VV4A";
const DECODE_ARTIFACT_MAGIC: &[u8; 4] = b"VDEC";

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
    bincode::deserialize(&decompressed).map_err(|e| format!("V4 audit deserialization failed: {e}"))
}

// ── Decode artifact ──────────────────────────────────────────

pub fn serialize_decode_artifact(artifact: &DecodeArtifact) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(DECODE_ARTIFACT_MAGIC);
    let encoded = bincode::serialize(artifact).expect("decode artifact serialization failed");
    buf.extend_from_slice(&encoded);
    buf
}

pub fn deserialize_decode_artifact(data: &[u8]) -> Result<DecodeArtifact, String> {
    if data.len() < 4 || &data[..4] != DECODE_ARTIFACT_MAGIC {
        return Err("invalid decode artifact magic".into());
    }
    bincode::deserialize(&data[4..])
        .map_err(|e| format!("decode artifact deserialization failed: {e}"))
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

    #[test]
    fn test_key_roundtrip() {
        let key = VerifierKey {
            version: 1,
            config: ModelConfig::toy(),
            seed: [42u8; 32],
            source_dtype: "I8".into(),
            quantization_scales: Vec::new(),
            r_vectors: vec![vec![Fp(1), Fp(2)]; 8],
            v_vectors: vec![vec![vec![Fp(10), Fp(20)]; 7]; 2],
            wo_norms: Vec::new(),
            max_v_norm: 0.0,
            lm_head: None,
            v_lm_head: None,
            lm_head_bf16_hash: None,
            weight_hash: None,
            rmsnorm_attn_weights: Vec::new(),
            rmsnorm_ffn_weights: Vec::new(),
            weight_scales: Vec::new(),
            per_channel_weight_scales: Vec::new(),
            rmsnorm_eps: 1e-5,
            rope_config_hash: None,
            embedding_merkle_root: None,
            final_norm_weights: None,
            quant_family: None,
            scale_derivation: None,
            quant_block_size: None,
            attn_backend: None,
            attn_dtype: None,
            rope_aware_replay: false,
            qkv_biases: Vec::new(),
            verification_profile: None,
            v_lm_head_f64: None,
            captured_logits_freivalds_seed: None,
        };
        let data = serialize_key(&key);
        let key2 = deserialize_key(&data).unwrap();
        assert_eq!(key.version, key2.version);
        assert_eq!(key.config.name, key2.config.name);
        assert_eq!(key.r_vectors, key2.r_vectors);
    }

    #[test]
    fn test_bad_magic() {
        assert!(deserialize_key(b"BAAD1234").is_err());
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let data = b"hello world, this is some test data for compression";
        let compressed = compress(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, data);
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
                x_attn_i8: None,
                scale_x_attn: None,
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
                scale_x_attn: 0.25,
                scale_x_ffn: 0.125,
                scale_h: 0.0625,
            }],
            layer_indices: None,
            initial_residual: None,
            embedding_proof: None,
            final_residual: None,
            logits_i32: None,
            lp_hidden_bf16: None,
            captured_logits_f32: None,
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
                input_spec_hash: None,
                model_spec_hash: None,
                decode_spec_hash: None,
                output_spec_hash: None,
                version: CommitmentVersion::V4,
                prompt_hash: None,
                seed_commitment: None,
                n_prompt_tokens: Some(1),
                kv_roots: vec![],
            },
            revealed_seed: [0xff; 32],
            shell_opening: Some(shell),
            manifest: None,
            prompt: None,
            n_prompt_tokens: Some(1),
            output_text: None,
            prefix_embedding_rows: None,
            prefix_embedding_proofs: None,
            prefix_retained: None,
            prefix_shell_openings: None,
            kv_entries: None,
            kv_proofs: None,
            witnessed_scores: None,
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
