//! Corridor amplification attack tests.
//!
//! These tests investigate whether an adversary can exploit the attention
//! corridor tolerance (±τ per layer) to produce meaningfully different
//! outputs while staying within bounds at every individual layer.
//!
//! The core question: if each layer's attention output `a` can differ by
//! up to ±τ from the honest value, does the error compound across L layers
//! enough to flip the output token?
//!
//! If yes, the protocol has a real hole: an adversary can serve a different
//! computation while passing all per-layer corridor checks.

use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::rmsnorm::{
    bridge_residual_rmsnorm, dequant_add_residual, quantize_f64_to_i8, rmsnorm_f64_input,
};
use verilm_test_vectors::{
    forward_pass, generate_model, generate_model_with_head, matmul_i32, LayerWeights,
};

/// Requantize i32 accumulators to i8 (same as verilm_core::requantize).
fn requantize(acc: &[i32]) -> Vec<i8> {
    verilm_core::requantize(acc)
}

/// Run forward pass but with `a` perturbed by `deltas[layer][i]` at each layer.
/// Recomputes everything downstream of the perturbation honestly.
/// Returns (per-layer traces, final hidden state).
fn forward_pass_with_perturbed_attention(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    input: &[i8],
    deltas: &[Vec<i8>], // deltas[layer][element] — perturbation to add to a
) -> (Vec<Vec<i8>>, Vec<i8>) {
    let mut x = input.to_vec();
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let mut all_a = Vec::new();

    for (layer_idx, lw) in model.iter().enumerate() {
        let x_attn = x.clone();

        // Honest attention projections
        let _q = matmul_i32(&lw.wq, &x_attn, cfg.hidden_dim, cfg.hidden_dim);
        let _k = matmul_i32(&lw.wk, &x_attn, cfg.kv_dim, cfg.hidden_dim);
        let v = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);

        // Honest attention output (single-token: softmax([score]) = [1.0])
        let v_i8 = requantize(&v);
        let mut a = vec![0i8; cfg.hidden_dim];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            let src_start = kv_head * cfg.d_head;
            let dst_start = qh * cfg.d_head;
            a[dst_start..dst_start + cfg.d_head]
                .copy_from_slice(&v_i8[src_start..src_start + cfg.d_head]);
        }

        // APPLY PERTURBATION — clamp to i8 range
        for i in 0..cfg.hidden_dim {
            let perturbed = a[i] as i16 + deltas[layer_idx][i] as i16;
            a[i] = perturbed.clamp(-128, 127) as i8;
        }

        all_a.push(a.clone());

        // Recompute everything downstream from perturbed a
        let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
        let x_ffn = requantize(&attn_out);

        let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let h = verilm_core::silu::compute_h_unit_scale(&g, &u);
        let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);

        x = requantize(&ffn_out);
    }

    (all_a, x)
}

/// Compute logits from final hidden state using lm_head.
fn compute_logits_i32(lm_head: &[i8], hidden: &[i8], vocab_size: usize, hidden_dim: usize) -> Vec<i32> {
    matmul_i32(lm_head, hidden, vocab_size, hidden_dim)
}

/// L-infinity distance between two i8 vectors.
fn linf_i8(a: &[i8], b: &[i8]) -> i16 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).abs())
        .max()
        .unwrap_or(0)
}

/// L-infinity distance between two i32 vectors.
fn linf_i32(a: &[i32], b: &[i32]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i64 - y as i64).abs())
        .max()
        .unwrap_or(0)
}

/// Mean absolute difference between two i8 vectors.
fn mean_abs_diff_i8(a: &[i8], b: &[i8]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum();
    sum / a.len() as f64
}

// ───────────────────────────────────────────────────────────────────
// Test 1: Measure amplification factor across layers
// ───────────────────────────────────────────────────────────────────

#[test]
fn corridor_amplification_uniform_max_perturbation() {
    // Perturb every element of `a` by +τ at every layer.
    // Measure how much the final hidden state diverges.
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 42);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let honest_traces = forward_pass(&cfg, &model, &input);
    let honest_final = requantize(&honest_traces.last().unwrap().ffn_out);

    // Test at production tolerance τ=10
    for tau in [1i8, 5, 10, 20] {
        let deltas: Vec<Vec<i8>> = (0..cfg.n_layers)
            .map(|_| vec![tau; cfg.hidden_dim])
            .collect();

        let (_, perturbed_final) =
            forward_pass_with_perturbed_attention(&cfg, &model, &input, &deltas);

        let linf = linf_i8(&honest_final, &perturbed_final);
        let mean_diff = mean_abs_diff_i8(&honest_final, &perturbed_final);

        eprintln!(
            "τ={:>2}: final hidden L∞={:>3}, mean_abs_diff={:.1}, changed_elements={}/{}",
            tau,
            linf,
            mean_diff,
            honest_final
                .iter()
                .zip(perturbed_final.iter())
                .filter(|(a, b)| a != b)
                .count(),
            cfg.hidden_dim,
        );
    }

    // The test passes — it's a measurement. But flag if τ=10 causes no change
    // (would mean the model is insensitive to attention perturbation, which
    // would be suspicious).
    let deltas_10: Vec<Vec<i8>> = (0..cfg.n_layers)
        .map(|_| vec![10i8; cfg.hidden_dim])
        .collect();
    let (_, perturbed_10) =
        forward_pass_with_perturbed_attention(&cfg, &model, &input, &deltas_10);
    let linf_10 = linf_i8(&honest_final, &perturbed_10);

    eprintln!("\n=== Amplification factor at τ=10: L∞ divergence in final hidden = {} ===", linf_10);
    // This is informational — we want to know the number.
    // A high amplification factor means the corridor tolerance may be exploitable.
}

// ───────────────────────────────────────────────────────────────────
// Test 2: Can corridor perturbation flip the argmax token?
// ───────────────────────────────────────────────────────────────────

#[test]
fn corridor_amplification_can_flip_token() {
    // The critical security question: can an adversary stay within ±τ
    // at every layer but produce a different output token?
    let cfg = ModelConfig::toy();
    let toy = generate_model_with_head(&cfg, 42);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    // Honest forward pass
    let honest_traces = forward_pass(&cfg, &toy.layers, &input);
    let honest_final = requantize(&honest_traces.last().unwrap().ffn_out);
    let honest_logits = compute_logits_i32(&toy.lm_head, &honest_final, cfg.vocab_size, cfg.hidden_dim);
    let honest_token = honest_logits
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .unwrap()
        .0;

    let mut flipped = false;
    let mut best_adversarial_gap: i64 = 0;

    // Try many perturbation strategies at τ=10 (production tolerance)
    let tau = 10i8;

    // Strategy 1: all +τ
    {
        let deltas: Vec<Vec<i8>> = (0..cfg.n_layers)
            .map(|_| vec![tau; cfg.hidden_dim])
            .collect();
        let (_, pf) = forward_pass_with_perturbed_attention(&cfg, &toy.layers, &input, &deltas);
        let pl = compute_logits_i32(&toy.lm_head, &pf, cfg.vocab_size, cfg.hidden_dim);
        let pt = pl.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;
        let gap = linf_i32(&honest_logits, &pl);
        if pt != honest_token {
            flipped = true;
        }
        best_adversarial_gap = best_adversarial_gap.max(gap);
        eprintln!("All +τ: honest_token={}, perturbed_token={}, logit_L∞={}", honest_token, pt, gap);
    }

    // Strategy 2: all -τ
    {
        let deltas: Vec<Vec<i8>> = (0..cfg.n_layers)
            .map(|_| vec![-tau; cfg.hidden_dim])
            .collect();
        let (_, pf) = forward_pass_with_perturbed_attention(&cfg, &toy.layers, &input, &deltas);
        let pl = compute_logits_i32(&toy.lm_head, &pf, cfg.vocab_size, cfg.hidden_dim);
        let pt = pl.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;
        let gap = linf_i32(&honest_logits, &pl);
        if pt != honest_token {
            flipped = true;
        }
        best_adversarial_gap = best_adversarial_gap.max(gap);
        eprintln!("All -τ: honest_token={}, perturbed_token={}, logit_L∞={}", honest_token, pt, gap);
    }

    // Strategy 3: random ±τ (many seeds)
    for seed in 0u64..100 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let deltas: Vec<Vec<i8>> = (0..cfg.n_layers)
            .map(|_| {
                (0..cfg.hidden_dim)
                    .map(|_| if rng.gen_bool(0.5) { tau } else { -tau })
                    .collect()
            })
            .collect();
        let (_, pf) = forward_pass_with_perturbed_attention(&cfg, &toy.layers, &input, &deltas);
        let pl = compute_logits_i32(&toy.lm_head, &pf, cfg.vocab_size, cfg.hidden_dim);
        let pt = pl.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;
        let gap = linf_i32(&honest_logits, &pl);
        if pt != honest_token {
            flipped = true;
            eprintln!("FLIP at seed {}: honest={}, perturbed={}, logit_L∞={}", seed, honest_token, pt, gap);
        }
        best_adversarial_gap = best_adversarial_gap.max(gap);
    }

    eprintln!(
        "\n=== Token flip: {}, best logit L∞ divergence: {} ===",
        if flipped { "YES — CORRIDOR IS EXPLOITABLE" } else { "NO" },
        best_adversarial_gap,
    );

    // This is a measurement test. If flipped=true, it's a critical finding
    // that should be documented in the security analysis.
}

// ───────────────────────────────────────────────────────────────────
// Test 3: Amplification vs number of layers
// ───────────────────────────────────────────────────────────────────

#[test]
fn corridor_amplification_vs_depth() {
    // Use models of increasing depth to measure how amplification scales.
    // Production models have 28-80 layers; toy has 2.
    let tau = 10i8;

    for n_layers in [2, 4, 8, 16, 28] {
        let cfg = ModelConfig {
            name: format!("toy-{}L", n_layers),
            hidden_dim: 16,
            kv_dim: 4,
            ffn_dim: 32,
            d_head: 2,
            n_layers,
            n_q_heads: 8,
            n_kv_heads: 2,
            vocab_size: 64,
            rope_theta: 10000.0,
            rope_scaling: None,
        };
        let model = generate_model(&cfg, 42);
        let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

        let honest_traces = forward_pass(&cfg, &model, &input);
        let honest_final = requantize(&honest_traces.last().unwrap().ffn_out);

        let deltas: Vec<Vec<i8>> = (0..n_layers)
            .map(|_| vec![tau; cfg.hidden_dim])
            .collect();
        let (_, perturbed_final) =
            forward_pass_with_perturbed_attention(&cfg, &model, &input, &deltas);

        let linf = linf_i8(&honest_final, &perturbed_final);
        let mean_diff = mean_abs_diff_i8(&honest_final, &perturbed_final);
        let changed = honest_final
            .iter()
            .zip(perturbed_final.iter())
            .filter(|(a, b)| a != b)
            .count();

        eprintln!(
            "L={:>2}: final hidden L∞={:>3}, mean_abs={:.1}, changed={}/{}",
            n_layers, linf, mean_diff, changed, cfg.hidden_dim,
        );
    }
}

// ───────────────────────────────────────────────────────────────────
// Test 4: Selective layer cheating — perturb only K out of L layers
// ───────────────────────────────────────────────────────────────────

#[test]
fn selective_layer_cheating_detection_probability() {
    // If the adversary cheats on only K layers out of L, and the auditor
    // samples S layers for deep audit, what's the probability of detection?
    //
    // This test measures the output divergence from cheating on subsets of layers.
    let cfg = ModelConfig {
        name: "toy-28L".into(),
        hidden_dim: 16,
        kv_dim: 4,
        ffn_dim: 32,
        d_head: 2,
        n_layers: 28,
        n_q_heads: 8,
        n_kv_heads: 2,
        vocab_size: 64,
        rope_theta: 10000.0,
        rope_scaling: None,
    };
    let toy = generate_model_with_head(&cfg, 42);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let honest_traces = forward_pass(&cfg, &toy.layers, &input);
    let honest_final = requantize(&honest_traces.last().unwrap().ffn_out);
    let honest_logits = compute_logits_i32(&toy.lm_head, &honest_final, cfg.vocab_size, cfg.hidden_dim);
    let honest_token = honest_logits.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;

    let tau = 10i8;

    eprintln!("Honest token: {}", honest_token);
    eprintln!("{:>12} {:>10} {:>10} {:>12}", "cheated_layers", "hidden_L∞", "logit_L∞", "token_flipped");

    for n_cheated in [1, 2, 4, 7, 14, 28] {
        let mut flips = 0u32;
        let mut max_logit_gap: i64 = 0;
        let mut max_hidden_gap: i16 = 0;
        let n_trials = 50;

        for seed in 0u64..n_trials {
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);

            // Choose which layers to cheat on
            let mut layer_indices: Vec<usize> = (0..28).collect();
            layer_indices.shuffle(&mut rng);
            let cheated: std::collections::HashSet<usize> =
                layer_indices[..n_cheated].iter().copied().collect();

            let deltas: Vec<Vec<i8>> = (0..28)
                .map(|l| {
                    if cheated.contains(&l) {
                        vec![tau; cfg.hidden_dim]
                    } else {
                        vec![0i8; cfg.hidden_dim]
                    }
                })
                .collect();

            let (_, pf) = forward_pass_with_perturbed_attention(&cfg, &toy.layers, &input, &deltas);
            let pl = compute_logits_i32(&toy.lm_head, &pf, cfg.vocab_size, cfg.hidden_dim);
            let pt = pl.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;

            if pt != honest_token {
                flips += 1;
            }
            max_logit_gap = max_logit_gap.max(linf_i32(&honest_logits, &pl));
            max_hidden_gap = max_hidden_gap.max(linf_i8(&honest_final, &pf));
        }

        eprintln!(
            "{:>12} {:>10} {:>10} {:>8}/{} ({:.0}%)",
            n_cheated,
            max_hidden_gap,
            max_logit_gap,
            flips,
            n_trials,
            flips as f64 / n_trials as f64 * 100.0,
        );
    }
}

// ───────────────────────────────────────────────────────────────────
// Test 5: Adversarial gradient-free search for token-flipping perturbation
// ───────────────────────────────────────────────────────────────────

#[test]
fn adversarial_search_for_token_flip() {
    // Greedy search: for each layer and element, try +τ and -τ,
    // keep whichever maximizes the gap between honest token logit
    // and the best alternative logit. This is a simple gradient-free
    // optimization that an adversary could run.
    let cfg = ModelConfig::toy();
    let toy = generate_model_with_head(&cfg, 42);
    let input: Vec<i8> = (0..cfg.hidden_dim as i8).collect();

    let honest_traces = forward_pass(&cfg, &toy.layers, &input);
    let honest_final = requantize(&honest_traces.last().unwrap().ffn_out);
    let honest_logits = compute_logits_i32(&toy.lm_head, &honest_final, cfg.vocab_size, cfg.hidden_dim);
    let honest_token = honest_logits.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;
    let honest_top_logit = honest_logits[honest_token];

    let tau = 10i8;

    // Start with zero perturbation, greedily optimize
    let mut best_deltas: Vec<Vec<i8>> = (0..cfg.n_layers)
        .map(|_| vec![0i8; cfg.hidden_dim])
        .collect();

    for layer in 0..cfg.n_layers {
        for elem in 0..cfg.hidden_dim {
            let mut best_score = i64::MIN;
            let mut best_val = 0i8;

            for candidate in [-tau, 0, tau] {
                let mut trial = best_deltas.clone();
                trial[layer][elem] = candidate;

                let (_, pf) = forward_pass_with_perturbed_attention(&cfg, &toy.layers, &input, &trial);
                let pl = compute_logits_i32(&toy.lm_head, &pf, cfg.vocab_size, cfg.hidden_dim);

                // Score: maximize gap between best non-honest logit and honest logit
                // (adversary wants to suppress the honest token and boost an alternative)
                let best_other = pl
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != honest_token)
                    .max_by_key(|(_, v)| *v)
                    .unwrap()
                    .1;
                let score = *best_other as i64 - pl[honest_token] as i64;

                if score > best_score {
                    best_score = score;
                    best_val = candidate;
                }
            }
            best_deltas[layer][elem] = best_val;
        }
    }

    let (_, adv_final) = forward_pass_with_perturbed_attention(&cfg, &toy.layers, &input, &best_deltas);
    let adv_logits = compute_logits_i32(&toy.lm_head, &adv_final, cfg.vocab_size, cfg.hidden_dim);
    let adv_token = adv_logits.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;

    let margin_honest = honest_top_logit as i64
        - *honest_logits
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != honest_token)
            .max_by_key(|(_, v)| *v)
            .unwrap()
            .1 as i64;

    let margin_adversarial = adv_logits[adv_token] as i64
        - *adv_logits
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != adv_token)
            .max_by_key(|(_, v)| *v)
            .unwrap()
            .1 as i64;

    eprintln!("Honest: token={}, margin={}", honest_token, margin_honest);
    eprintln!("Adversarial: token={}, margin={}", adv_token, margin_adversarial);
    eprintln!("Hidden L∞: {}", linf_i8(&honest_final, &adv_final));
    eprintln!("Logit L∞: {}", linf_i32(&honest_logits, &adv_logits));
    eprintln!(
        "Token flipped: {}",
        if adv_token != honest_token { "YES" } else { "NO" }
    );

    let n_perturbed: usize = best_deltas
        .iter()
        .flat_map(|d| d.iter())
        .filter(|&&v| v != 0)
        .count();
    let total = cfg.n_layers * cfg.hidden_dim;
    eprintln!("Elements perturbed: {}/{} ({:.0}%)", n_perturbed, total, n_perturbed as f64 / total as f64 * 100.0);
}

// ───────────────────────────────────────────────────────────────────
// Test 6: Scaling with hidden dimension
// ───────────────────────────────────────────────────────────────────

#[test]
fn corridor_amplification_vs_hidden_dim() {
    // Production models have hidden_dim=4096. Does the effect survive
    // when there are more dimensions to average over?
    let tau = 10i8;

    for hidden_dim in [16, 64, 256, 512] {
        // Keep d_head=2 (toy), scale n_q_heads to fill hidden_dim
        let n_q_heads = hidden_dim / 2;
        let n_kv_heads = (n_q_heads / 4).max(1);
        let kv_dim = n_kv_heads * 2;
        let ffn_dim = hidden_dim * 2;

        let cfg = ModelConfig {
            name: format!("toy-d{}", hidden_dim),
            hidden_dim,
            kv_dim,
            ffn_dim,
            d_head: 2,
            n_layers: 2,
            n_q_heads,
            n_kv_heads,
            vocab_size: 64,
            rope_theta: 10000.0,
            rope_scaling: None,
        };
        let toy = generate_model_with_head(&cfg, 42);
        let input: Vec<i8> = (0..hidden_dim).map(|i| (i % 256) as i8).collect();

        let honest_traces = forward_pass(&cfg, &toy.layers, &input);
        let honest_final = requantize(&honest_traces.last().unwrap().ffn_out);
        let honest_logits =
            compute_logits_i32(&toy.lm_head, &honest_final, cfg.vocab_size, hidden_dim);
        let honest_token = honest_logits.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;

        let mut flips = 0u32;
        let n_trials = 50u64;
        let mut max_logit_gap: i64 = 0;

        for seed in 0..n_trials {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let deltas: Vec<Vec<i8>> = (0..cfg.n_layers)
                .map(|_| {
                    (0..hidden_dim)
                        .map(|_| if rng.gen_bool(0.5) { tau } else { -tau })
                        .collect()
                })
                .collect();
            let (_, pf) =
                forward_pass_with_perturbed_attention(&cfg, &toy.layers, &input, &deltas);
            let pl = compute_logits_i32(&toy.lm_head, &pf, cfg.vocab_size, hidden_dim);
            let pt = pl.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;
            if pt != honest_token {
                flips += 1;
            }
            max_logit_gap = max_logit_gap.max(linf_i32(&honest_logits, &pl));
        }

        eprintln!(
            "d={:>4}: flips={:>2}/{}, max_logit_L∞={:>10}, honest_token={}",
            hidden_dim, flips, n_trials, max_logit_gap, honest_token,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// FULL BRIDGE TESTS — with residual connections and RMSNorm
// ═══════════════════════════════════════════════════════════════════
//
// The tests above use the simplified toy model (no residuals, no RMSNorm).
// Production models maintain a residual stream in f64:
//   residual += dequant(W_o @ a)
//   x_next = quantize(RMSNorm(residual))
//
// RMSNorm normalizes the vector magnitude, which may dampen amplification.
// Residual connections mean perturbations are added to (not replacing)
// the accumulated signal.

/// Synthetic weight scales for bridge tests.
fn bridge_weight_scales(n_layers: usize) -> Vec<Vec<f32>> {
    let n_mt = MatrixType::PER_LAYER.len();
    (0..n_layers)
        .map(|l| {
            (0..n_mt)
                .map(|m| 0.01 + 0.001 * (l * n_mt + m) as f32)
                .collect()
        })
        .collect()
}

/// Synthetic RMSNorm weights (positive, like real models).
fn bridge_rmsnorm_weights(n_layers: usize, hidden_dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let attn: Vec<Vec<f32>> = (0..n_layers)
        .map(|l| {
            (0..hidden_dim)
                .map(|i| 0.5 + 0.01 * ((l * hidden_dim + i) % 100) as f32)
                .collect()
        })
        .collect();
    let ffn: Vec<Vec<f32>> = (0..n_layers)
        .map(|l| {
            (0..hidden_dim)
                .map(|i| 0.6 + 0.01 * ((l * hidden_dim + i + 37) % 100) as f32)
                .collect()
        })
        .collect();
    (attn, ffn)
}

/// Per-layer activation scales for bridge tests.
fn bridge_activation_scales(n_layers: usize) -> Vec<(f32, f32, f32, f32)> {
    (0..n_layers)
        .map(|l| {
            (
                0.3 + 0.05 * l as f32, // scale_x_attn
                0.5 + 0.1 * l as f32,  // scale_a
                0.4 + 0.07 * l as f32, // scale_x_ffn
                0.6 + 0.03 * l as f32, // scale_h
            )
        })
        .collect()
}

/// Full-bridge forward pass with perturbed attention outputs.
/// Like `full_bridge_forward` in v4_e2e.rs, but applies delta to `a` at each layer.
///
/// Returns the final f64 residual stream (pre-logit).
fn full_bridge_forward_perturbed(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_residual: &[f32],
    rmsnorm_attn: &[Vec<f32>],
    rmsnorm_ffn: &[Vec<f32>],
    weight_scales: &[Vec<f32>],
    scales: &[(f32, f32, f32, f32)],
    eps: f64,
    deltas: &[Vec<i8>], // perturbation per layer
) -> Vec<f64> {
    let mut residual: Vec<f64> = initial_residual.iter().map(|&v| v as f64).collect();
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    for (l, lw) in model.iter().enumerate() {
        let (scale_x_attn, scale_a, scale_x_ffn, scale_h) = scales[l];

        let ws = |mt: MatrixType| -> f32 {
            let idx = MatrixType::PER_LAYER.iter().position(|&m| m == mt).unwrap();
            weight_scales[l][idx]
        };

        // x_attn = quantize(RMSNorm_attn(residual), scale_x_attn)
        let normed = rmsnorm_f64_input(&residual, &rmsnorm_attn[l], eps);
        let x_attn = quantize_f64_to_i8(&normed, scale_x_attn as f64);

        // Single-token attention: a = expand(requantize(V))
        let v_acc = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);
        let v_i8 = verilm_core::requantize(&v_acc);
        let mut a = vec![0i8; cfg.hidden_dim];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            let src = kv_head * cfg.d_head;
            let dst = qh * cfg.d_head;
            a[dst..dst + cfg.d_head].copy_from_slice(&v_i8[src..src + cfg.d_head]);
        }

        // APPLY PERTURBATION
        for i in 0..cfg.hidden_dim {
            let perturbed = a[i] as i16 + deltas[l][i] as i16;
            a[i] = perturbed.clamp(-128, 127) as i8;
        }

        // Bridge: attn_out → dequant → residual += → RMSNorm → quantize
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

        // FFN
        let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let h = verilm_core::silu::compute_h_scaled(
            &g, &u,
            ws(MatrixType::Wg), ws(MatrixType::Wu),
            scale_x_ffn, scale_h,
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
    }

    residual
}

/// L-infinity distance between two f64 vectors.
fn linf_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Mean absolute difference between two f64 vectors.
fn mean_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    let sum: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum();
    sum / a.len() as f64
}

// ───────────────────────────────────────────────────────────────────
// Test 7: Full bridge amplification measurement
// ───────────────────────────────────────────────────────────────────

#[test]
fn full_bridge_corridor_amplification() {
    // Same as the toy model tests, but with residual stream + RMSNorm.
    // Does RMSNorm dampen the amplification?
    let cfg = ModelConfig::toy();
    let model = generate_model(&cfg, 12345);
    let initial_residual: Vec<f32> = (0..cfg.hidden_dim)
        .map(|i| 0.1 * (i as f32 - cfg.hidden_dim as f32 / 2.0))
        .collect();
    let (rmsnorm_attn, rmsnorm_ffn) = bridge_rmsnorm_weights(cfg.n_layers, cfg.hidden_dim);
    let weight_scales = bridge_weight_scales(cfg.n_layers);
    let scales = bridge_activation_scales(cfg.n_layers);
    let eps = 1e-5;

    // Honest forward pass
    let zero_deltas: Vec<Vec<i8>> = (0..cfg.n_layers)
        .map(|_| vec![0i8; cfg.hidden_dim])
        .collect();
    let honest_residual = full_bridge_forward_perturbed(
        &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn,
        &weight_scales, &scales, eps, &zero_deltas,
    );

    eprintln!("=== Full Bridge (residual + RMSNorm) Corridor Amplification ===\n");

    for tau in [1i8, 5, 10, 20] {
        // Uniform +τ
        let deltas_pos: Vec<Vec<i8>> = (0..cfg.n_layers)
            .map(|_| vec![tau; cfg.hidden_dim])
            .collect();
        let res_pos = full_bridge_forward_perturbed(
            &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn,
            &weight_scales, &scales, eps, &deltas_pos,
        );
        let linf_pos = linf_f64(&honest_residual, &res_pos);
        let mean_pos = mean_abs_diff_f64(&honest_residual, &res_pos);

        // Random ±τ (seed 0)
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let deltas_rand: Vec<Vec<i8>> = (0..cfg.n_layers)
            .map(|_| {
                (0..cfg.hidden_dim)
                    .map(|_| if rng.gen_bool(0.5) { tau } else { -tau })
                    .collect()
            })
            .collect();
        let res_rand = full_bridge_forward_perturbed(
            &cfg, &model, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn,
            &weight_scales, &scales, eps, &deltas_rand,
        );
        let linf_rand = linf_f64(&honest_residual, &res_rand);
        let mean_rand = mean_abs_diff_f64(&honest_residual, &res_rand);

        eprintln!(
            "τ={:>2}: uniform(+τ) residual L∞={:.4}, mean={:.4} | random(±τ) residual L∞={:.4}, mean={:.4}",
            tau, linf_pos, mean_pos, linf_rand, mean_rand,
        );
    }
}

// ───────────────────────────────────────────────────────────────────
// Test 8: Full bridge token flip with lm_head
// ───────────────────────────────────────────────────────────────────

#[test]
fn full_bridge_corridor_can_flip_token() {
    let tau = 10i8;

    for hidden_dim in [16, 64, 256] {
        let n_q_heads = hidden_dim / 2;
        let n_kv_heads = (n_q_heads / 4).max(1);
        let kv_dim = n_kv_heads * 2;
        let ffn_dim = hidden_dim * 2;
        let n_layers = 2;

        let cfg = ModelConfig {
            name: format!("bridge-d{}", hidden_dim),
            hidden_dim,
            kv_dim,
            ffn_dim,
            d_head: 2,
            n_layers,
            n_q_heads,
            n_kv_heads,
            vocab_size: 64,
            rope_theta: 10000.0,
            rope_scaling: None,
        };
        let toy = generate_model_with_head(&cfg, 42);
        let initial_residual: Vec<f32> = (0..hidden_dim)
            .map(|i| 0.1 * (i as f32 - hidden_dim as f32 / 2.0))
            .collect();
        let (rmsnorm_attn, rmsnorm_ffn) = bridge_rmsnorm_weights(n_layers, hidden_dim);
        let weight_scales = bridge_weight_scales(n_layers);
        let scales = bridge_activation_scales(n_layers);
        let eps = 1e-5;

        // Honest pass
        let zero_deltas: Vec<Vec<i8>> = (0..n_layers)
            .map(|_| vec![0i8; hidden_dim])
            .collect();
        let honest_residual = full_bridge_forward_perturbed(
            &cfg, &toy.layers, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn,
            &weight_scales, &scales, eps, &zero_deltas,
        );
        // Quantize final residual to compute logits
        // In production, there's a final RMSNorm here; we skip it for simplicity
        let honest_final: Vec<i8> = honest_residual
            .iter()
            .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
            .collect();
        let honest_logits = compute_logits_i32(&toy.lm_head, &honest_final, cfg.vocab_size, hidden_dim);
        let honest_token = honest_logits.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;

        let mut flips = 0u32;
        let n_trials = 50u64;
        let mut max_logit_gap: i64 = 0;

        for seed in 0..n_trials {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let deltas: Vec<Vec<i8>> = (0..n_layers)
                .map(|_| {
                    (0..hidden_dim)
                        .map(|_| if rng.gen_bool(0.5) { tau } else { -tau })
                        .collect()
                })
                .collect();
            let perturbed_residual = full_bridge_forward_perturbed(
                &cfg, &toy.layers, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn,
                &weight_scales, &scales, eps, &deltas,
            );
            let perturbed_final: Vec<i8> = perturbed_residual
                .iter()
                .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
                .collect();
            let pl = compute_logits_i32(&toy.lm_head, &perturbed_final, cfg.vocab_size, hidden_dim);
            let pt = pl.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;
            if pt != honest_token {
                flips += 1;
            }
            max_logit_gap = max_logit_gap.max(linf_i32(&honest_logits, &pl));
        }

        eprintln!(
            "BRIDGE d={:>4}: flips={:>2}/{}, max_logit_L∞={:>10}, honest_token={}",
            hidden_dim, flips, n_trials, max_logit_gap, honest_token,
        );
    }
}

// ───────────────────────────────────────────────────────────────────
// Test 9: Full bridge depth scaling (production-like 28 layers)
// ───────────────────────────────────────────────────────────────────

#[test]
fn full_bridge_corridor_vs_depth() {
    let tau = 10i8;
    let hidden_dim = 64;
    let n_q_heads = hidden_dim / 2;
    let n_kv_heads = (n_q_heads / 4).max(1);
    let kv_dim = n_kv_heads * 2;
    let ffn_dim = hidden_dim * 2;

    eprintln!("=== Full Bridge Depth Scaling (d={}, τ={}) ===\n", hidden_dim, tau);
    eprintln!("{:>8} {:>12} {:>12} {:>12}", "layers", "flips/50", "max_logit_L∞", "residual_L∞");

    for n_layers in [2, 4, 8, 16, 28] {
        let cfg = ModelConfig {
            name: format!("bridge-{}L", n_layers),
            hidden_dim,
            kv_dim,
            ffn_dim,
            d_head: 2,
            n_layers,
            n_q_heads,
            n_kv_heads,
            vocab_size: 64,
            rope_theta: 10000.0,
            rope_scaling: None,
        };
        let toy = generate_model_with_head(&cfg, 42);
        let initial_residual: Vec<f32> = (0..hidden_dim)
            .map(|i| 0.1 * (i as f32 - hidden_dim as f32 / 2.0))
            .collect();
        let (rmsnorm_attn, rmsnorm_ffn) = bridge_rmsnorm_weights(n_layers, hidden_dim);
        let weight_scales = bridge_weight_scales(n_layers);
        let scales = bridge_activation_scales(n_layers);
        let eps = 1e-5;

        let zero_deltas: Vec<Vec<i8>> = (0..n_layers)
            .map(|_| vec![0i8; hidden_dim])
            .collect();
        let honest_residual = full_bridge_forward_perturbed(
            &cfg, &toy.layers, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn,
            &weight_scales, &scales, eps, &zero_deltas,
        );
        let honest_final: Vec<i8> = honest_residual
            .iter()
            .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
            .collect();
        let honest_logits = compute_logits_i32(&toy.lm_head, &honest_final, cfg.vocab_size, hidden_dim);
        let honest_token = honest_logits.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;

        let mut flips = 0u32;
        let n_trials = 50u64;
        let mut max_logit_gap: i64 = 0;
        let mut max_residual_gap: f64 = 0.0;

        for seed in 0..n_trials {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let deltas: Vec<Vec<i8>> = (0..n_layers)
                .map(|_| {
                    (0..hidden_dim)
                        .map(|_| if rng.gen_bool(0.5) { tau } else { -tau })
                        .collect()
                })
                .collect();
            let perturbed_residual = full_bridge_forward_perturbed(
                &cfg, &toy.layers, &initial_residual, &rmsnorm_attn, &rmsnorm_ffn,
                &weight_scales, &scales, eps, &deltas,
            );
            max_residual_gap = max_residual_gap.max(linf_f64(&honest_residual, &perturbed_residual));

            let perturbed_final: Vec<i8> = perturbed_residual
                .iter()
                .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
                .collect();
            let pl = compute_logits_i32(&toy.lm_head, &perturbed_final, cfg.vocab_size, hidden_dim);
            let pt = pl.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;
            if pt != honest_token {
                flips += 1;
            }
            max_logit_gap = max_logit_gap.max(linf_i32(&honest_logits, &pl));
        }

        eprintln!(
            "{:>8} {:>8}/{} {:>12} {:>12.2}",
            n_layers, flips, n_trials, max_logit_gap, max_residual_gap,
        );
    }
}
