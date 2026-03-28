//! One-shot fixture generator for cross-version binary format tests.
//!
//! Run: cargo test -p verilm-verify --test gen_fixtures -- --ignored
//!
//! This generates the frozen binary fixtures under tests/fixtures/.
//! The fixtures are checked into git and must not change unless the
//! binary format intentionally changes (which requires a version bump).

use verilm_core::constants::ModelConfig;
use verilm_core::types::{RetainedLayerState, RetainedTokenState, ShellWeights};
use verilm_prover::{commit_minimal, open_v4, FullBindingParams};
use verilm_test_vectors::{forward_pass, generate_key, generate_model, LayerWeights};

struct ToyWeights<'a>(&'a [LayerWeights]);

impl ShellWeights for ToyWeights<'_> {
    fn weight(&self, layer: usize, mt: verilm_core::constants::MatrixType) -> &[i8] {
        let lw = &self.0[layer];
        match mt {
            verilm_core::constants::MatrixType::Wq => &lw.wq,
            verilm_core::constants::MatrixType::Wk => &lw.wk,
            verilm_core::constants::MatrixType::Wv => &lw.wv,
            verilm_core::constants::MatrixType::Wo => &lw.wo,
            verilm_core::constants::MatrixType::Wg => &lw.wg,
            verilm_core::constants::MatrixType::Wu => &lw.wu,
            verilm_core::constants::MatrixType::Wd => &lw.wd,
            verilm_core::constants::MatrixType::LmHead => panic!("no LmHead"),
        }
    }
}

fn retained_from_traces(traces: &[verilm_core::types::LayerTrace]) -> RetainedTokenState {
    RetainedTokenState {
        layers: traces
            .iter()
            .map(|lt| RetainedLayerState {
                a: lt.a.clone(),
                scale_a: lt.scale_a.unwrap_or(1.0),
                scale_x_attn: lt.scale_x_attn.unwrap_or(1.0),
                scale_x_ffn: lt.scale_x_ffn.unwrap_or(1.0),
                scale_h: lt.scale_h.unwrap_or(1.0),
            })
            .collect(),
    }
}

#[test]
#[ignore] // Run manually: cargo test -p verilm-verify --test gen_fixtures -- --ignored
fn generate_frozen_fixtures() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::create_dir_all(&dir).unwrap();

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);

    // --- Canonical V4 audit fixture ---
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();
    let traces = forward_pass(&cfg, &model, &input);
    let retained = retained_from_traces(&traces);

    let params = FullBindingParams {
        token_ids: &[42],
        prompt: b"frozen fixture prompt",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(vec![retained], &params, None);
    let response = open_v4(
        &state, 0, &ToyWeights(&model), &cfg, &[], None, None, None, None, false,
    );

    let audit_binary = verilm_core::serialize::serialize_v4_audit(&response);
    std::fs::write(dir.join("v4_audit_canonical.bin"), &audit_binary).unwrap();
    eprintln!("wrote v4_audit_canonical.bin ({} bytes)", audit_binary.len());

    // --- Canonical verifier key fixture ---
    let key_binary = verilm_core::serialize::serialize_key(&key);
    std::fs::write(dir.join("v4_key_canonical.bin"), &key_binary).unwrap();
    eprintln!("wrote v4_key_canonical.bin ({} bytes)", key_binary.len());

    // --- Rejection fixtures ---

    // Unknown magic "VV5A"
    let mut unknown_magic = audit_binary.clone();
    unknown_magic[2] = b'5';
    std::fs::write(dir.join("reject_unknown_magic.bin"), &unknown_magic).unwrap();

    // Truncated: magic + 8 bytes
    let truncated: Vec<u8> = audit_binary[..12.min(audit_binary.len())].to_vec();
    std::fs::write(dir.join("reject_truncated.bin"), &truncated).unwrap();

    // Cross-format: key bytes with audit magic (valid VKEY content, VV4A header)
    let mut cross_format = b"VV4A".to_vec();
    cross_format.extend_from_slice(&key_binary[4..]); // key body after VKEY magic
    std::fs::write(dir.join("reject_cross_format.bin"), &cross_format).unwrap();

    // Corrupted: valid zstd that decompresses to garbage bincode
    let garbage = vec![0xABu8; 64];
    let compressed = zstd::encode_all(garbage.as_slice(), 3).unwrap();
    let mut corrupted = b"VV4A".to_vec();
    corrupted.extend_from_slice(&compressed);
    std::fs::write(dir.join("reject_corrupted_bincode.bin"), &corrupted).unwrap();

    eprintln!("all fixtures generated in {:?}", dir);
}
