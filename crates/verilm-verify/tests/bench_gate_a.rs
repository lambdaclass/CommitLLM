//! Benchmark Gate A: baseline numbers at the fail-closed full-bridge checkpoint.
//!
//! Measures:
//! - Baseline forward pass (toy model, simplified bridge)
//! - Minimal online path (capture → commit, V4 retained-state)
//! - Audit open time (open_v4, simplified vs full bridge)
//! - Verifier time (verify_v4, simplified vs full bridge)
//! - Audit payload size (serialized V4AuditResponse)
//! - Retained bytes per token (RetainedTokenState)

use std::time::{Duration, Instant};

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{BridgeParams, RetainedLayerState, RetainedTokenState, ShellWeights};
use verilm_prover::{commit_minimal, open_v4, CapturedLayerScales, FullBindingParams};
use verilm_test_vectors::{
    forward_pass_autoregressive, generate_key, generate_model, LayerWeights,
};
use verilm_verify::{verify_v4_legacy, verify_v4_with_weights, Verdict};

/// Thin wrapper: routes through the legacy verifier while canonical is validated.
fn verify_v4(
    key: &verilm_core::types::VerifierKey,
    response: &verilm_core::types::V4AuditResponse,
    expected_prompt_token_ids: Option<&[u32]>,
) -> verilm_verify::V4VerifyReport {
    verify_v4_legacy(key, response, expected_prompt_token_ids, None, None)
}

struct ToyWeights<'a>(&'a [LayerWeights]);

impl ShellWeights for ToyWeights<'_> {
    fn weight(&self, layer: usize, mt: MatrixType) -> &[i8] {
        let lw = &self.0[layer];
        match mt {
            MatrixType::Wq => &lw.wq,
            MatrixType::Wk => &lw.wk,
            MatrixType::Wv => &lw.wv,
            MatrixType::Wo => &lw.wo,
            MatrixType::Wg => &lw.wg,
            MatrixType::Wu => &lw.wu,
            MatrixType::Wd => &lw.wd,
            MatrixType::LmHead => panic!("ToyWeights: LmHead is not a per-layer weight"),
        }
    }
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times[times.len() / 2]
}

// ───────────────────────────── full-bridge helpers ─────────────────────────

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
    let model = generate_model(&cfg, 42);
    let mut key = generate_key(&cfg, &model, [7u8; 32]);

    let rmsnorm_attn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| {
            (0..cfg.hidden_dim)
                .map(|i| 0.8 + 0.01 * (l * cfg.hidden_dim + i) as f32)
                .collect()
        })
        .collect();
    let rmsnorm_ffn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| {
            (0..cfg.hidden_dim)
                .map(|i| 0.9 + 0.005 * (l * cfg.hidden_dim + i) as f32)
                .collect()
        })
        .collect();

    let initial_residual: Vec<f32> = (0..cfg.hidden_dim)
        .map(|i| 0.1 * (i as f32) - 0.5 * cfg.hidden_dim as f32 * 0.1)
        .collect();

    let weight_scales: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| {
            MatrixType::PER_LAYER
                .iter()
                .enumerate()
                .map(|(j, _)| 0.01 + 0.002 * (l * 7 + j) as f32)
                .collect()
        })
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
            a[qh * cfg.d_head..(qh + 1) * cfg.d_head]
                .copy_from_slice(&v_i8[kv_head * cfg.d_head..(kv_head + 1) * cfg.d_head]);
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
    ir: &[f32],
    token_id: u32,
    n_vocab: usize,
) -> (verilm_core::merkle::MerkleTree, [u8; 32]) {
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if i == token_id as usize {
            leaves.push(verilm_core::merkle::hash_embedding_row(ir));
        } else {
            let row: Vec<f32> = (0..ir.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    let root = tree.root;
    (tree, root)
}

// ───────────────────────────── GATE A BENCHMARK ─────────────────────────

const ITERS: usize = 20;
const N_TOKENS: usize = 10;

#[test]
fn bench_gate_a() {
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let input: Vec<i8> = (0..cfg.hidden_dim).map(|i| (i % 256) as i8).collect();
    let key = generate_key(&cfg, &model, [7u8; 32]);

    // ── 1. Baseline: forward pass (10 tokens) ──
    let mut fwd_times = Vec::with_capacity(ITERS);
    let mut all_layers = Vec::new();
    for _ in 0..ITERS {
        let start = Instant::now();
        all_layers = forward_pass_autoregressive(&cfg, &model, &input, N_TOKENS);
        fwd_times.push(start.elapsed());
    }
    let fwd_med = median(&mut fwd_times);

    // ── 2. Minimal online path: retained-state extraction + Merkle commit ──
    // Simulate retained state extraction from full traces (the online cost).
    let token_ids: Vec<u32> = (0..N_TOKENS as u32).collect();
    let mut retained_all: Vec<RetainedTokenState> = Vec::with_capacity(N_TOKENS);
    let mut scales_all: Vec<Vec<CapturedLayerScales>> = Vec::with_capacity(N_TOKENS);
    for t in 0..N_TOKENS {
        let layers: Vec<RetainedLayerState> = all_layers[t]
            .iter()
            .map(|lt| RetainedLayerState {
                a: lt.a.clone(),
                scale_a: lt.scale_a.unwrap_or(1.0),
                x_attn_i8: None,
                scale_x_attn: None,
            })
            .collect();
        let token_scales: Vec<CapturedLayerScales> = all_layers[t]
            .iter()
            .map(|lt| CapturedLayerScales {
                scale_x_attn: lt.scale_x_attn.unwrap_or(1.0),
                scale_x_ffn: lt.scale_x_ffn.unwrap_or(1.0),
                scale_h: lt.scale_h.unwrap_or(1.0),
            })
            .collect();
        retained_all.push(RetainedTokenState { layers });
        scales_all.push(token_scales);
    }

    let mut commit_times = Vec::with_capacity(ITERS);
    let mut state = None;
    for _ in 0..ITERS {
        let start = Instant::now();
        let params = FullBindingParams {
            token_ids: &token_ids,
            prompt: b"bench gate a",
            sampling_seed: [1u8; 32],
            manifest: None,
            n_prompt_tokens: Some(1),
        };
        let (_c, s) = commit_minimal(
            retained_all.clone(),
            &params,
            None,
            scales_all.clone(),
            None,
            None,
            None,
            None,
        );
        commit_times.push(start.elapsed());
        state = Some(s);
    }
    let commit_med = median(&mut commit_times);
    let state = state.unwrap();

    // ── 3. Audit open time (simplified bridge, no full bridge) ──
    let mut open_simple_times = Vec::with_capacity(ITERS);
    let mut response_simple = None;
    for _ in 0..ITERS {
        let start = Instant::now();
        let r = open_v4(
            &state,
            5,
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
        open_simple_times.push(start.elapsed());
        response_simple = Some(r);
    }
    let open_simple_med = median(&mut open_simple_times);
    let response_simple = response_simple.unwrap();

    // ── 4. Verifier time (simplified bridge) ──
    let mut verify_simple_times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let report = verify_v4(&key, &response_simple, None);
        verify_simple_times.push(start.elapsed());
        assert_eq!(
            report.verdict,
            Verdict::Pass,
            "simplified: {:?}",
            report.failures
        );
    }
    let verify_simple_med = median(&mut verify_simple_times);

    // ── 5. Full-bridge open + verify ──
    let (fb_cfg, fb_model, mut fb_key, fb_ws, fb_rn_attn, fb_rn_ffn, fb_ir) = setup_full_bridge();
    let fb_scales = bridge_scales(&fb_cfg);
    let fb_token_id = 42u32;

    let (emb_tree, emb_root) = setup_embedding_tree(&fb_ir, fb_token_id, 128);
    fb_key.embedding_merkle_root = Some(emb_root);

    let (fb_retained, fb_captured_scales) = full_bridge_forward(
        &fb_cfg,
        &fb_model,
        &fb_ir,
        &fb_rn_attn,
        &fb_rn_ffn,
        &fb_ws,
        &fb_scales,
        1e-5,
    );
    let fb_proof = verilm_core::merkle::prove(&emb_tree, fb_token_id as usize);

    let fb_params = FullBindingParams {
        token_ids: &[fb_token_id],
        prompt: b"bench full bridge",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_fb_commitment, fb_state) = commit_minimal(
        vec![fb_retained],
        &fb_params,
        None,
        vec![fb_captured_scales],
        None,
        None,
        None,
        None,
    );

    let fb_bridge = BridgeParams {
        rmsnorm_attn_weights: &fb_rn_attn,
        rmsnorm_ffn_weights: &fb_rn_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &fb_ir,
        embedding_proof: Some(fb_proof),
    };

    let mut open_full_times = Vec::with_capacity(ITERS);
    let mut response_full = None;
    for _ in 0..ITERS {
        let start = Instant::now();
        let r = open_v4(
            &fb_state,
            0,
            &ToyWeights(&fb_model),
            &fb_cfg,
            &fb_ws,
            &[],
            Some(&fb_bridge),
            None,
            None,
            None,
            false,
            false,
        );
        open_full_times.push(start.elapsed());
        response_full = Some(r);
    }
    let open_full_med = median(&mut open_full_times);
    let response_full = response_full.unwrap();

    let mut verify_full_times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let report = verify_v4(&fb_key, &response_full, None);
        verify_full_times.push(start.elapsed());
        assert_eq!(
            report.verdict,
            Verdict::Pass,
            "full bridge: {:?}",
            report.failures
        );
    }
    let verify_full_med = median(&mut verify_full_times);

    // ── 6. Verifier with weights (debug/oracle path) ──
    let mut verify_weights_times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let report = verify_v4_with_weights(&fb_key, &response_full, &ToyWeights(&fb_model));
        verify_weights_times.push(start.elapsed());
        assert_eq!(
            report.verdict,
            Verdict::Pass,
            "oracle: {:?}",
            report.failures
        );
    }
    let verify_weights_med = median(&mut verify_weights_times);

    // ── 7. Payload sizes ──
    let payload_simple = serde_json::to_vec(&response_simple).unwrap().len();
    let payload_full = serde_json::to_vec(&response_full).unwrap().len();

    // Retained bytes per token
    let retained_json = serde_json::to_vec(&retained_all[0]).unwrap().len();
    let retained_bincode = bincode::serialize(&retained_all[0]).unwrap().len();

    // ── Report ──
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!(
        "║          BENCHMARK GATE A  —  toy model ({} layers, {}D)              ║",
        cfg.n_layers, cfg.hidden_dim
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  {:44} {:>10} {:>8}  ║", "", "median", "unit");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        format!("1. Baseline forward ({} tokens)", N_TOKENS),
        fwd_med.as_micros(),
        "us"
    );
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        format!("2. Minimal commit ({} tokens)", N_TOKENS),
        commit_med.as_micros(),
        "us"
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "3a. Audit open (simplified bridge)",
        open_simple_med.as_micros(),
        "us"
    );
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "3b. Audit open (full bridge)",
        open_full_med.as_micros(),
        "us"
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "4a. Verify (simplified, key-only)",
        verify_simple_med.as_micros(),
        "us"
    );
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "4b. Verify (full bridge, key-only)",
        verify_full_med.as_micros(),
        "us"
    );
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "4c. Verify (full bridge, with weights)",
        verify_weights_med.as_micros(),
        "us"
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "5a. Audit payload (simplified, JSON)", payload_simple, "bytes"
    );
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "5b. Audit payload (full bridge, JSON)", payload_full, "bytes"
    );
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "6a. Retained/token (JSON)", retained_json, "bytes"
    );
    println!(
        "║  {:44} {:>10} {:>8}  ║",
        "6b. Retained/token (bincode)", retained_bincode, "bytes"
    );
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Sanity: all times should be under generous bounds
    assert!(
        fwd_med < Duration::from_millis(200),
        "forward pass too slow: {:?}",
        fwd_med
    );
    assert!(
        commit_med < Duration::from_millis(100),
        "commit too slow: {:?}",
        commit_med
    );
    assert!(
        open_simple_med < Duration::from_millis(50),
        "simplified open too slow: {:?}",
        open_simple_med
    );
    assert!(
        open_full_med < Duration::from_millis(50),
        "full bridge open too slow: {:?}",
        open_full_med
    );
    assert!(
        verify_simple_med < Duration::from_millis(50),
        "simplified verify too slow: {:?}",
        verify_simple_med
    );
    assert!(
        verify_full_med < Duration::from_millis(50),
        "full bridge verify too slow: {:?}",
        verify_full_med
    );
}
