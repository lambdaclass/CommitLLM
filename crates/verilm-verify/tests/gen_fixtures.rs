//! One-shot fixture generator for cross-version binary format tests.
//!
//! Run: cargo test -p verilm-verify --test gen_fixtures -- --ignored
//!
//! This generates the frozen binary fixtures under tests/fixtures/.
//! The fixtures are checked into git and must not change unless the
//! binary format intentionally changes (which requires a version bump).

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{
    BridgeParams, DeploymentManifest, RetainedLayerState, RetainedTokenState, ShellWeights,
};
use verilm_prover::{commit_minimal, open_v4, CapturedLayerScales, FullBindingParams};
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
                x_attn_i8: None,
                scale_x_attn: None,
            })
            .collect(),
    }
}

fn unit_scales(n_layers: usize) -> Vec<CapturedLayerScales> {
    vec![
        CapturedLayerScales {
            scale_x_attn: 1.0,
            scale_x_ffn: 1.0,
            scale_h: 1.0
        };
        n_layers
    ]
}

// --- Full-bridge helpers (same as canonical.rs tests) ---

fn setup_full_bridge() -> (
    ModelConfig,
    Vec<LayerWeights>,
    verilm_core::types::VerifierKey,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<f32>,
) {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let mut key = generate_key(&cfg, &model, [1u8; 32]);

    let n_mt = MatrixType::PER_LAYER.len();
    let weight_scales: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| {
            (0..n_mt)
                .map(|m| 0.01 + 0.001 * (l * n_mt + m) as f32)
                .collect()
        })
        .collect();
    let rmsnorm_attn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| {
            (0..cfg.hidden_dim)
                .map(|i| 0.5 + 0.01 * ((l * cfg.hidden_dim + i) % 100) as f32)
                .collect()
        })
        .collect();
    let rmsnorm_ffn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| {
            (0..cfg.hidden_dim)
                .map(|i| 0.6 + 0.01 * ((l * cfg.hidden_dim + i + 37) % 100) as f32)
                .collect()
        })
        .collect();
    let initial_residual: Vec<f32> = (0..cfg.hidden_dim)
        .map(|i| 0.1 * (i as f32 - cfg.hidden_dim as f32 / 2.0))
        .collect();

    key.weight_scales = weight_scales.clone();
    key.rmsnorm_attn_weights = rmsnorm_attn.clone();
    key.rmsnorm_ffn_weights = rmsnorm_ffn.clone();
    key.rmsnorm_eps = 1e-5;

    (
        cfg,
        model,
        key,
        weight_scales,
        rmsnorm_attn,
        rmsnorm_ffn,
        initial_residual,
    )
}

fn bridge_scales(cfg: &ModelConfig) -> Vec<(f32, f32, f32, f32)> {
    (0..cfg.n_layers)
        .map(|l| {
            (
                0.3 + 0.05 * l as f32,
                0.5 + 0.1 * l as f32,
                0.4 + 0.07 * l as f32,
                0.6 + 0.03 * l as f32,
            )
        })
        .collect()
}

fn full_bridge_forward(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_residual: &[f32],
    rmsnorm_attn: &[Vec<f32>],
    rmsnorm_ffn: &[Vec<f32>],
    weight_scales: &[Vec<f32>],
    scales: &[(f32, f32, f32, f32)],
    eps: f64,
) -> (RetainedTokenState, Vec<CapturedLayerScales>) {
    use verilm_core::matmul::matmul_i32;
    use verilm_core::rmsnorm::{
        bridge_residual_rmsnorm, dequant_add_residual, quantize_f64_to_i8, rmsnorm_f64_input,
    };

    let mut residual: Vec<f64> = initial_residual.iter().map(|&v| v as f64).collect();
    let mut layers = Vec::new();
    let mut captured_scales = Vec::new();
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    for (l, lw) in model.iter().enumerate() {
        let (scale_x_attn, scale_a, scale_x_ffn, scale_h) = scales[l];
        let ws = |mt: MatrixType| -> f32 {
            let idx = MatrixType::PER_LAYER.iter().position(|&m| m == mt).unwrap();
            weight_scales[l][idx]
        };

        let normed = rmsnorm_f64_input(&residual, &rmsnorm_attn[l], eps);
        let x_attn = quantize_f64_to_i8(&normed, scale_x_attn as f64);
        let v_acc = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);
        let v_i8 = verilm_core::requantize(&v_acc);
        let mut a = vec![0i8; cfg.hidden_dim];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            let src = kv_head * cfg.d_head;
            let dst = qh * cfg.d_head;
            a[dst..dst + cfg.d_head].copy_from_slice(&v_i8[src..src + cfg.d_head]);
        }

        let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
        let x_ffn = bridge_residual_rmsnorm(
            &attn_out,
            ws(MatrixType::Wo),
            scale_a,
            &mut residual,
            &rmsnorm_ffn[l],
            eps,
            scale_x_ffn,
        );

        let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let h = verilm_core::silu::compute_h_scaled(
            &g,
            &u,
            ws(MatrixType::Wg),
            ws(MatrixType::Wu),
            scale_x_ffn,
            scale_h,
        );
        let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);

        if l + 1 < rmsnorm_attn.len() {
            let next_scale = scales.get(l + 1).map(|s| s.0).unwrap_or(1.0);
            bridge_residual_rmsnorm(
                &ffn_out,
                ws(MatrixType::Wd),
                scale_h,
                &mut residual,
                &rmsnorm_attn[l + 1],
                eps,
                next_scale,
            );
        } else {
            dequant_add_residual(&ffn_out, ws(MatrixType::Wd), scale_h, &mut residual);
        }

        layers.push(RetainedLayerState {
            a,
            scale_a,
            x_attn_i8: None,
            scale_x_attn: None,
        });
        captured_scales.push(CapturedLayerScales {
            scale_x_attn,
            scale_x_ffn,
            scale_h,
        });
    }

    (RetainedTokenState { layers }, captured_scales)
}

fn setup_embedding_tree(
    initial_residual: &[f32],
    token_id: u32,
    n_vocab: usize,
) -> (verilm_core::merkle::MerkleTree, [u8; 32]) {
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if i == token_id as usize {
            leaves.push(verilm_core::merkle::hash_embedding_row(initial_residual));
        } else {
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    let root = tree.root;
    (tree, root)
}

fn make_manifest() -> DeploymentManifest {
    DeploymentManifest {
        tokenizer_hash: [0u8; 32],
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        eos_policy: "stop".into(),
        weight_hash: None,
        quant_hash: None,
        system_prompt_hash: None,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        logit_bias: vec![],
        bad_word_ids: vec![],
        guided_decoding: String::new(),
        stop_sequences: vec![],
        max_tokens: 0,
        chat_template_hash: None,
        rope_config_hash: None,
        rmsnorm_eps: None,
        sampler_version: None,
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
        attn_backend: None,
        attn_dtype: None,
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
    }
}

#[test]
#[ignore] // Run manually: cargo test -p verilm-verify --test gen_fixtures -- --ignored
fn generate_frozen_fixtures() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::create_dir_all(&dir).unwrap();

    // ==================================================================
    // Legacy fixtures (toy model, no bridge, no manifest)
    // ==================================================================

    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let key = generate_key(&cfg, &model, [1u8; 32]);

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
    let (_commitment, state) = commit_minimal(
        vec![retained],
        &params,
        None,
        vec![unit_scales(cfg.n_layers)],
        None,
        None,
    );
    let response = open_v4(
        &state,
        0,
        &ToyWeights(&model),
        &cfg,
        &[],
        &[],
        None,
        None,
        None,
        None,
        false,
        false,
    );

    let audit_binary = verilm_core::serialize::serialize_v4_audit(&response);
    std::fs::write(dir.join("v4_audit_canonical.bin"), &audit_binary).unwrap();
    eprintln!(
        "wrote v4_audit_canonical.bin ({} bytes)",
        audit_binary.len()
    );

    let key_binary = verilm_core::serialize::serialize_key(&key);
    std::fs::write(dir.join("v4_key_canonical.bin"), &key_binary).unwrap();
    eprintln!("wrote v4_key_canonical.bin ({} bytes)", key_binary.len());

    // ==================================================================
    // Canonical fixtures (full bridge + manifest + embedding proof)
    // ==================================================================

    let (cfg_fb, model_fb, mut key_fb, ws, rmsnorm_attn, rmsnorm_ffn, initial_residual) =
        setup_full_bridge();
    let scales = bridge_scales(&cfg_fb);
    let token_id = 42u32;

    let (tree, root) = setup_embedding_tree(&initial_residual, token_id, 128);
    key_fb.embedding_merkle_root = Some(root);

    let (retained_fb, captured_scales_fb) = full_bridge_forward(
        &cfg_fb,
        &model_fb,
        &initial_residual,
        &rmsnorm_attn,
        &rmsnorm_ffn,
        &ws,
        &scales,
        1e-5,
    );

    let proof = verilm_core::merkle::prove(&tree, token_id as usize);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &rmsnorm_attn,
        rmsnorm_ffn_weights: &rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &initial_residual,
        embedding_proof: Some(proof),
    };

    let manifest = make_manifest();
    let params_fb = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"canonical frozen fixture",
        sampling_seed: [7u8; 32],
        manifest: Some(&manifest),
        n_prompt_tokens: Some(1),
    };
    let (_commitment_fb, state_fb) = commit_minimal(
        vec![retained_fb],
        &params_fb,
        None,
        vec![captured_scales_fb],
        None,
        None,
    );
    let response_fb = open_v4(
        &state_fb,
        0,
        &ToyWeights(&model_fb),
        &cfg_fb,
        &ws,
        &[],
        Some(&bridge),
        None,
        None,
        None,
        false,
        false,
    );

    let audit_fb_binary = verilm_core::serialize::serialize_v4_audit(&response_fb);
    std::fs::write(dir.join("v4_audit_fullbridge.bin"), &audit_fb_binary).unwrap();
    eprintln!(
        "wrote v4_audit_fullbridge.bin ({} bytes)",
        audit_fb_binary.len()
    );

    let key_fb_binary = verilm_core::serialize::serialize_key(&key_fb);
    std::fs::write(dir.join("v4_key_fullbridge.bin"), &key_fb_binary).unwrap();
    eprintln!(
        "wrote v4_key_fullbridge.bin ({} bytes)",
        key_fb_binary.len()
    );

    // Verify the canonical fixture passes canonical verification before freezing
    let report = verilm_verify::canonical::verify_response(&key_fb, &response_fb, None, None);
    assert_eq!(
        report.verdict,
        verilm_verify::Verdict::Pass,
        "canonical fixture must pass canonical verification: {:?}",
        report.failures
    );
    eprintln!(
        "canonical fixture verified: {} checks, verdict={:?}",
        report.checks_run, report.verdict
    );

    // ==================================================================
    // Rejection fixtures
    // ==================================================================

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
