//! Toy model for end-to-end testing of the verification pipeline.
//!
//! Generates random INT8 weights, computes a fake forward pass,
//! generates a verifier key, produces a trace, and verifies it.
//! This validates the entire math pipeline before touching real models.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::field::Fp;
use verilm_core::freivalds;
use verilm_core::merkle;
pub use verilm_core::requantize;
use verilm_core::types::{LayerTrace, TokenTrace, VerifierKey};

// Re-export prover types and functions so existing tests keep working.
pub use verilm_prover::{
    BatchState, FullBindingParams,
    commit, commit_legacy, commit_with_token_ids, commit_with_manifest,
    commit_with_full_binding, open, build_batch,
    build_audit_response, build_audit_response_from_state,
    build_streaming_kv_verifier,
    verify_batch, verify_batch_with_policy, verify_trace,
};

/// Random INT8 weight matrix (row-major).
pub fn random_weights(rng: &mut impl Rng, rows: usize, cols: usize) -> Vec<i8> {
    (0..rows * cols).map(|_| rng.gen::<i8>()).collect()
}

/// INT8 matrix-vector multiply (row-major W, returns i32 accumulators).
///
/// In real inference: input (INT8) -> matmul (i32 accumulator) -> requant (INT8).
/// The Freivalds check verifies the matmul step using the FULL i32 result,
/// not the requantized INT8. The prover must send the i32 accumulators
/// (or their reduction mod p) for each matrix multiplication.
pub fn matmul_i32(w: &[i8], x: &[i8], rows: usize, cols: usize) -> Vec<i32> {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| w[r * cols + c] as i32 * x[c] as i32)
                .sum()
        })
        .collect()
}

/// All 7 weight matrices for one layer.
#[derive(Clone)]
pub struct LayerWeights {
    pub wq: Vec<i8>,
    pub wk: Vec<i8>,
    pub wv: Vec<i8>,
    pub wo: Vec<i8>,
    pub wg: Vec<i8>,
    pub wu: Vec<i8>,
    pub wd: Vec<i8>,
}

/// Model with per-layer weights plus an unembedding head (lm_head).
pub struct ToyModel {
    pub layers: Vec<LayerWeights>,
    /// Unembedding matrix: shape (vocab_size, hidden_dim).
    pub lm_head: Vec<i8>,
}

/// Generate a complete toy model: weights for all layers + lm_head.
pub fn generate_model(cfg: &ModelConfig, seed: u64) -> Vec<LayerWeights> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..cfg.n_layers)
        .map(|_| LayerWeights {
            wq: random_weights(&mut rng, cfg.hidden_dim, cfg.hidden_dim),
            wk: random_weights(&mut rng, cfg.kv_dim, cfg.hidden_dim),
            wv: random_weights(&mut rng, cfg.kv_dim, cfg.hidden_dim),
            wo: random_weights(&mut rng, cfg.hidden_dim, cfg.hidden_dim),
            wg: random_weights(&mut rng, cfg.ffn_dim, cfg.hidden_dim),
            wu: random_weights(&mut rng, cfg.ffn_dim, cfg.hidden_dim),
            wd: random_weights(&mut rng, cfg.hidden_dim, cfg.ffn_dim),
        })
        .collect()
}

/// Generate a toy model with lm_head for Level B testing.
pub fn generate_model_with_head(cfg: &ModelConfig, seed: u64) -> ToyModel {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let layers = (0..cfg.n_layers)
        .map(|_| LayerWeights {
            wq: random_weights(&mut rng, cfg.hidden_dim, cfg.hidden_dim),
            wk: random_weights(&mut rng, cfg.kv_dim, cfg.hidden_dim),
            wv: random_weights(&mut rng, cfg.kv_dim, cfg.hidden_dim),
            wo: random_weights(&mut rng, cfg.hidden_dim, cfg.hidden_dim),
            wg: random_weights(&mut rng, cfg.ffn_dim, cfg.hidden_dim),
            wu: random_weights(&mut rng, cfg.ffn_dim, cfg.hidden_dim),
            wd: random_weights(&mut rng, cfg.hidden_dim, cfg.ffn_dim),
        })
        .collect();
    let lm_head = random_weights(&mut rng, cfg.vocab_size, cfg.hidden_dim);
    ToyModel { layers, lm_head }
}

/// Compute logit vector from the last hidden state via lm_head matmul.
///
/// logits = lm_head @ last_hidden_state (i32 accumulators, then cast to f32).
pub fn compute_logits(lm_head: &[i8], last_hidden: &[i8], vocab_size: usize, hidden_dim: usize) -> Vec<f32> {
    matmul_i32(lm_head, last_hidden, vocab_size, hidden_dim)
        .iter()
        .map(|&v| v as f32)
        .collect()
}

/// Run a single-token forward pass through all layers with real GQA attention.
///
/// For a single token (no KV cache), softmax of one score is always 1.0,
/// so the attention output per query head is the V vector of its KV head.
/// With GQA, multiple query heads share one KV head.
pub fn forward_pass(cfg: &ModelConfig, model: &[LayerWeights], input: &[i8]) -> Vec<LayerTrace> {
    let mut x = input.to_vec();
    let mut layers = Vec::new();
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    for lw in model {
        let x_attn = x.clone();

        // Attention projections (i32 accumulators)
        let q = matmul_i32(&lw.wq, &x_attn, cfg.hidden_dim, cfg.hidden_dim);
        let k = matmul_i32(&lw.wk, &x_attn, cfg.kv_dim, cfg.hidden_dim);
        let v = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);

        // Real single-token GQA attention.
        // With one token, softmax([score]) = [1.0], so attn output = V.
        // Each query head's output is the V from its corresponding KV head.
        let v_i8 = requantize(&v);
        let mut a = vec![0i8; cfg.hidden_dim];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            let src_start = kv_head * cfg.d_head;
            let dst_start = qh * cfg.d_head;
            a[dst_start..dst_start + cfg.d_head]
                .copy_from_slice(&v_i8[src_start..src_start + cfg.d_head]);
        }
        let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);

        // Residual (simplified: requantize attn_out as FFN input)
        let x_ffn = requantize(&attn_out);

        // FFN (i32 accumulators)
        let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);

        // SiLU(g) * u -> h (LUT-based, unit scale)
        let h = verilm_core::silu::compute_h_unit_scale(&g, &u);

        let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);

        // Next layer input (simplified residual)
        x = requantize(&ffn_out);

        layers.push(LayerTrace {
            x_attn,
            q,
            k,
            v,
            a,
            attn_out,
            x_ffn,
            g,
            u,
            h,
            ffn_out,
            kv_cache_k: Vec::new(),
            kv_cache_v: Vec::new(),
            scale_x_attn: None,
            scale_a: None,
            scale_x_ffn: None,
            scale_h: None,
            residual: None,
        });
    }

    layers
}

/// Run a multi-token forward pass with KV cache and real GQA attention.
///
/// Each token sees K/V from all previous tokens via the cache.
/// Attention uses f64 softmax(QK^T/sqrt(d)) V, requantized to i8 for W_o.
/// Returns traces[token_index][layer_index].
pub fn forward_pass_multi(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    inputs: &[Vec<i8>],
) -> Vec<Vec<LayerTrace>> {
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let d_head = cfg.d_head;
    let inv_sqrt_d = 1.0 / (d_head as f64).sqrt();

    // KV cache per layer: (k_vecs, v_vecs) where k_vecs[token] = Vec<i8> of kv_dim
    let mut kv_cache: Vec<(Vec<Vec<i8>>, Vec<Vec<i8>>)> =
        (0..cfg.n_layers).map(|_| (Vec::new(), Vec::new())).collect();

    let mut all_traces = Vec::new();

    for (token_idx, input) in inputs.iter().enumerate() {
        let mut x = input.clone();
        let mut token_layers = Vec::new();

        for (layer_idx, lw) in model.iter().enumerate() {
            let x_attn = x.clone();

            let q = matmul_i32(&lw.wq, &x_attn, cfg.hidden_dim, cfg.hidden_dim);
            let k = matmul_i32(&lw.wk, &x_attn, cfg.kv_dim, cfg.hidden_dim);
            let v = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);

            let q_i8 = requantize(&q);
            let k_i8 = requantize(&k);
            let v_i8 = requantize(&v);

            // Add current token's K, V to cache
            kv_cache[layer_idx].0.push(k_i8);
            kv_cache[layer_idx].1.push(v_i8);

            // GQA attention with KV cache
            let seq_len = token_idx + 1;
            let mut a = vec![0i8; cfg.hidden_dim];

            for qh in 0..cfg.n_q_heads {
                let kv_head = qh / heads_per_kv;

                // Extract Q head (from requantized q)
                let q_head: Vec<f64> = (0..d_head)
                    .map(|i| q_i8[qh * d_head + i] as f64)
                    .collect();

                // Compute attention scores: score[t] = q · k_t / sqrt(d)
                let scores: Vec<f64> = (0..seq_len)
                    .map(|t| {
                        let k_t = &kv_cache[layer_idx].0[t];
                        let dot: f64 = (0..d_head)
                            .map(|i| q_head[i] * k_t[kv_head * d_head + i] as f64)
                            .sum();
                        dot * inv_sqrt_d
                    })
                    .collect();

                // Softmax
                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> =
                    scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();
                let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

                // Weighted sum of V cache
                let mut head_out = vec![0.0f64; d_head];
                for t in 0..seq_len {
                    let v_t = &kv_cache[layer_idx].1[t];
                    for i in 0..d_head {
                        head_out[i] += weights[t] * v_t[kv_head * d_head + i] as f64;
                    }
                }

                // Requantize head output to i8
                for i in 0..d_head {
                    a[qh * d_head + i] = head_out[i].round().clamp(-128.0, 127.0) as i8;
                }
            }

            let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
            let x_ffn = requantize(&attn_out);

            let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
            let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);

            // SiLU(g) * u -> h (LUT-based, unit scale)
            let h = verilm_core::silu::compute_h_unit_scale(&g, &u);

            let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);
            x = requantize(&ffn_out);

            token_layers.push(LayerTrace {
                x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
                kv_cache_k: Vec::new(),
                kv_cache_v: Vec::new(),
                scale_x_attn: None,
                scale_a: None,
                scale_x_ffn: None,
                scale_h: None,
                residual: None,
            });
        }

        all_traces.push(token_layers);
    }

    all_traces
}

/// Autoregressive multi-token forward pass: token 0 uses the given input,
/// subsequent tokens use requantize(previous token's last-layer output).
/// This mirrors real autoregressive decode where each token chains from the previous.
pub fn forward_pass_autoregressive(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_input: &[i8],
    n_tokens: usize,
) -> Vec<Vec<LayerTrace>> {
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let d_head = cfg.d_head;
    let inv_sqrt_d = 1.0 / (d_head as f64).sqrt();

    let mut kv_cache: Vec<(Vec<Vec<i8>>, Vec<Vec<i8>>)> =
        (0..cfg.n_layers).map(|_| (Vec::new(), Vec::new())).collect();

    let mut all_traces = Vec::new();

    for token_idx in 0..n_tokens {
        let mut x = if token_idx == 0 {
            initial_input.to_vec()
        } else {
            // Chain from previous token's output
            let prev = all_traces.last().unwrap();
            let prev_layers: &Vec<LayerTrace> = prev;
            requantize(&prev_layers.last().unwrap().ffn_out)
        };
        let mut token_layers = Vec::new();

        for (layer_idx, lw) in model.iter().enumerate() {
            let x_attn = x.clone();

            let q = matmul_i32(&lw.wq, &x_attn, cfg.hidden_dim, cfg.hidden_dim);
            let k = matmul_i32(&lw.wk, &x_attn, cfg.kv_dim, cfg.hidden_dim);
            let v = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);

            let q_i8 = requantize(&q);
            let k_i8 = requantize(&k);
            let v_i8 = requantize(&v);

            kv_cache[layer_idx].0.push(k_i8);
            kv_cache[layer_idx].1.push(v_i8);

            let seq_len = token_idx + 1;
            let mut a = vec![0i8; cfg.hidden_dim];

            for qh in 0..cfg.n_q_heads {
                let kv_head = qh / heads_per_kv;

                let q_head: Vec<f64> = (0..d_head)
                    .map(|i| q_i8[qh * d_head + i] as f64)
                    .collect();

                let scores: Vec<f64> = (0..seq_len)
                    .map(|t| {
                        let k_t = &kv_cache[layer_idx].0[t];
                        let dot: f64 = (0..d_head)
                            .map(|i| q_head[i] * k_t[kv_head * d_head + i] as f64)
                            .sum();
                        dot * inv_sqrt_d
                    })
                    .collect();

                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> =
                    scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();
                let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

                let mut head_out = vec![0.0f64; d_head];
                for t in 0..seq_len {
                    let v_t = &kv_cache[layer_idx].1[t];
                    for i in 0..d_head {
                        head_out[i] += weights[t] * v_t[kv_head * d_head + i] as f64;
                    }
                }

                for i in 0..d_head {
                    a[qh * d_head + i] = head_out[i].round().clamp(-128.0, 127.0) as i8;
                }
            }

            let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
            let x_ffn = requantize(&attn_out);

            let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
            let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);

            // SiLU(g) * u -> h (LUT-based, unit scale)
            let h = verilm_core::silu::compute_h_unit_scale(&g, &u);

            let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);
            x = requantize(&ffn_out);

            token_layers.push(LayerTrace {
                x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
                kv_cache_k: Vec::new(),
                kv_cache_v: Vec::new(),
                scale_x_attn: None,
                scale_a: None,
                scale_x_ffn: None,
                scale_h: None,
                residual: None,
            });
        }

        all_traces.push(token_layers);
    }

    all_traces
}

/// Single-token forward pass with KV cache emitted for Level C verification.
///
/// Same as `forward_pass` but populates `kv_cache_k` and `kv_cache_v`
/// in each LayerTrace (single entry: the current token's own K/V).
pub fn forward_pass_level_c(cfg: &ModelConfig, model: &[LayerWeights], input: &[i8]) -> Vec<LayerTrace> {
    let mut x = input.to_vec();
    let mut layers = Vec::new();
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    for lw in model {
        let x_attn = x.clone();

        let q = matmul_i32(&lw.wq, &x_attn, cfg.hidden_dim, cfg.hidden_dim);
        let k = matmul_i32(&lw.wk, &x_attn, cfg.kv_dim, cfg.hidden_dim);
        let v = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);

        let k_i8 = requantize(&k);
        let v_i8 = requantize(&v);

        let mut a = vec![0i8; cfg.hidden_dim];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            let src_start = kv_head * cfg.d_head;
            let dst_start = qh * cfg.d_head;
            a[dst_start..dst_start + cfg.d_head]
                .copy_from_slice(&v_i8[src_start..src_start + cfg.d_head]);
        }
        let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
        let x_ffn = requantize(&attn_out);

        let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let h = verilm_core::silu::compute_h_unit_scale(&g, &u);
        let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);
        x = requantize(&ffn_out);

        layers.push(LayerTrace {
            x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
            kv_cache_k: vec![k_i8],
            kv_cache_v: vec![v_i8],
            scale_x_attn: None,
            scale_a: None,
            scale_x_ffn: None,
            scale_h: None,
            residual: None,
        });
    }

    layers
}

/// Autoregressive multi-token forward pass with KV cache emitted for Level C.
///
/// Same as `forward_pass_autoregressive` but populates `kv_cache_k` and
/// `kv_cache_v` in each LayerTrace with the full KV cache snapshot at that point.
pub fn forward_pass_autoregressive_level_c(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    initial_input: &[i8],
    n_tokens: usize,
) -> Vec<Vec<LayerTrace>> {
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;
    let d_head = cfg.d_head;
    let inv_sqrt_d = 1.0 / (d_head as f64).sqrt();

    let mut kv_cache: Vec<(Vec<Vec<i8>>, Vec<Vec<i8>>)> =
        (0..cfg.n_layers).map(|_| (Vec::new(), Vec::new())).collect();

    let mut all_traces = Vec::new();

    for token_idx in 0..n_tokens {
        let mut x = if token_idx == 0 {
            initial_input.to_vec()
        } else {
            let prev_layers: &Vec<LayerTrace> = all_traces.last().unwrap();
            requantize(&prev_layers.last().unwrap().ffn_out)
        };
        let mut token_layers = Vec::new();

        for (layer_idx, lw) in model.iter().enumerate() {
            let x_attn = x.clone();

            let q = matmul_i32(&lw.wq, &x_attn, cfg.hidden_dim, cfg.hidden_dim);
            let k = matmul_i32(&lw.wk, &x_attn, cfg.kv_dim, cfg.hidden_dim);
            let v = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);

            let q_i8 = requantize(&q);
            let k_i8 = requantize(&k);
            let v_i8 = requantize(&v);

            kv_cache[layer_idx].0.push(k_i8);
            kv_cache[layer_idx].1.push(v_i8);

            let seq_len = token_idx + 1;
            let mut a = vec![0i8; cfg.hidden_dim];

            for qh in 0..cfg.n_q_heads {
                let kv_head = qh / heads_per_kv;

                let q_head: Vec<f64> = (0..d_head)
                    .map(|i| q_i8[qh * d_head + i] as f64)
                    .collect();

                let scores: Vec<f64> = (0..seq_len)
                    .map(|t| {
                        let k_t = &kv_cache[layer_idx].0[t];
                        let dot: f64 = (0..d_head)
                            .map(|i| q_head[i] * k_t[kv_head * d_head + i] as f64)
                            .sum();
                        dot * inv_sqrt_d
                    })
                    .collect();

                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> =
                    scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();
                let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

                let mut head_out = vec![0.0f64; d_head];
                for t in 0..seq_len {
                    let v_t = &kv_cache[layer_idx].1[t];
                    for i in 0..d_head {
                        head_out[i] += weights[t] * v_t[kv_head * d_head + i] as f64;
                    }
                }

                for i in 0..d_head {
                    a[qh * d_head + i] = head_out[i].round().clamp(-128.0, 127.0) as i8;
                }
            }

            let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
            let x_ffn = requantize(&attn_out);

            let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
            let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
            let h = verilm_core::silu::compute_h_unit_scale(&g, &u);
            let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);
            x = requantize(&ffn_out);

            // Snapshot the full KV cache at this point for Level C
            token_layers.push(LayerTrace {
                x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
                kv_cache_k: kv_cache[layer_idx].0.clone(),
                kv_cache_v: kv_cache[layer_idx].1.clone(),
                scale_x_attn: None,
                scale_a: None,
                scale_x_ffn: None,
                scale_h: None,
                residual: None,
            });
        }

        all_traces.push(token_layers);
    }

    all_traces
}

/// Generate a verifier-secret key from model weights.
///
/// This is a VERIFIER-SIDE operation. The returned key contains secret
/// random vectors r_j that must never be shared with the prover.
pub fn generate_key(cfg: &ModelConfig, model: &[LayerWeights], seed: [u8; 32]) -> VerifierKey {
    let mut rng = ChaCha20Rng::from_seed(seed);

    // Generate per-matrix-type r vectors
    let r_vectors: Vec<Vec<Fp>> = MatrixType::ALL
        .iter()
        .map(|mt| {
            let dim = mt.output_dim(cfg);
            (0..dim).map(|_| Fp::new(rng.gen::<u32>())).collect()
        })
        .collect();

    // Precompute v_j^(i) = r_j^T W_j^(i) for each layer and matrix type
    let v_vectors: Vec<Vec<Vec<Fp>>> = model
        .iter()
        .map(|lw| {
            MatrixType::ALL
                .iter()
                .enumerate()
                .map(|(j, mt)| {
                    let r = &r_vectors[j];
                    let rows = mt.output_dim(cfg);
                    let cols = mt.input_dim(cfg);
                    let w = match mt {
                        MatrixType::Wq => &lw.wq,
                        MatrixType::Wk => &lw.wk,
                        MatrixType::Wv => &lw.wv,
                        MatrixType::Wo => &lw.wo,
                        MatrixType::Wg => &lw.wg,
                        MatrixType::Wu => &lw.wu,
                        MatrixType::Wd => &lw.wd,
                    };
                    freivalds::precompute_v(r, w, rows, cols)
                })
                .collect()
        })
        .collect();

    // Compute weight-chain hash over all INT8 weights
    let weight_hash = verilm_core::merkle::hash_weights(
        "I8",
        cfg.n_layers,
        &[],  // no quantization scales for native INT8
        |layer, mt_idx| {
            let lw = &model[layer];
            match MatrixType::ALL[mt_idx] {
                MatrixType::Wq => lw.wq.clone(),
                MatrixType::Wk => lw.wk.clone(),
                MatrixType::Wv => lw.wv.clone(),
                MatrixType::Wo => lw.wo.clone(),
                MatrixType::Wg => lw.wg.clone(),
                MatrixType::Wu => lw.wu.clone(),
                MatrixType::Wd => lw.wd.clone(),
            }
        },
        MatrixType::ALL.len(),
    );

    VerifierKey {
        version: 1,
        config: cfg.clone(),
        seed,
        source_dtype: "I8".into(),
        quantization_scales: Vec::new(),
        r_vectors,
        v_vectors,
        wo_norms: Vec::new(),
        max_v_norm: 0.0,
        lm_head: None,
        r_lm_head: None,
        v_lm_head: None,
        weight_hash: Some(weight_hash),
        rmsnorm_attn_weights: Vec::new(),
        rmsnorm_ffn_weights: Vec::new(),
        weight_scales: Vec::new(),
        rmsnorm_eps: 1e-5,
        embedding_merkle_root: None,
        final_norm_weights: None,
    }
}

/// Generate a verifier key with W_o/V norms and lm_head for Level B verification.
pub fn generate_key_level_b(cfg: &ModelConfig, model: &[LayerWeights], seed: [u8; 32]) -> VerifierKey {
    generate_key_level_b_with_head(cfg, model, seed, None)
}

/// Generate a Level B verifier key, optionally including lm_head for logit binding.
pub fn generate_key_level_b_with_head(
    cfg: &ModelConfig,
    model: &[LayerWeights],
    seed: [u8; 32],
    lm_head: Option<Vec<i8>>,
) -> VerifierKey {
    let mut key = generate_key(cfg, model, seed);
    // Generate LM-head Freivalds vectors when lm_head is provided.
    if let Some(ref lm) = lm_head {
        use rand::{SeedableRng, Rng as _};
        // Derive a separate seed for lm_head r vector (avoid reusing shell seed).
        let mut lm_seed = seed;
        lm_seed[0] ^= 0xff;
        let mut lm_rng = ChaCha20Rng::from_seed(lm_seed);
        let vocab_size = cfg.vocab_size;
        let hidden_dim = cfg.hidden_dim;
        let r: Vec<verilm_core::field::Fp> = (0..vocab_size)
            .map(|_| verilm_core::field::Fp::new(lm_rng.gen::<u32>()))
            .collect();
        let v = verilm_core::freivalds::precompute_v(&r, lm, vocab_size, hidden_dim);
        key.r_lm_head = Some(r);
        key.v_lm_head = Some(v);
    }
    key.lm_head = lm_head;
    key.wo_norms = model
        .iter()
        .map(|lw| {
            verilm_core::margin::compute_matrix_linf_norm(
                &lw.wo,
                cfg.hidden_dim,
                cfg.hidden_dim,
                1.0, // unit scale for toy model
            )
        })
        .collect();
    // max_v_norm: max L-inf norm of W_v across all layers (unit scale).
    key.max_v_norm = model
        .iter()
        .map(|lw| {
            verilm_core::margin::compute_matrix_linf_norm(
                &lw.wv,
                cfg.kv_dim,
                cfg.hidden_dim,
                1.0, // unit scale for toy model
            )
        })
        .fold(0.0f32, f32::max);
    key
}

/// Build a complete token trace with Merkle commitment.
///
/// If `final_hidden` is provided, it is included in the Merkle leaf hash
/// (Level B binding). Pass `None` for Level A traces.
pub fn build_trace(layers: Vec<LayerTrace>, token_index: u32, n_tokens: usize) -> TokenTrace {
    build_trace_with_hidden(layers, token_index, n_tokens, None)
}

/// Build a token trace with optional final_hidden binding.
pub fn build_trace_with_hidden(
    layers: Vec<LayerTrace>,
    token_index: u32,
    n_tokens: usize,
    final_hidden: Option<Vec<i8>>,
) -> TokenTrace {
    // Hash each token's layer data to create Merkle leaves.
    // For testing, we create dummy leaves for other tokens.
    let mut leaves = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        if t == token_index as usize {
            leaves.push(merkle::hash_trace_direct(&layers, final_hidden.as_deref()));
        } else {
            // Dummy leaf for other tokens
            leaves.push(merkle::hash_leaf(&(t as u32).to_le_bytes()));
        }
    }

    let tree = merkle::build_tree(&leaves);
    let proof = merkle::prove(&tree, token_index as usize);

    TokenTrace {
        token_index,
        layers,
        merkle_root: tree.root,
        merkle_proof: proof,
        io_proof: merkle::MerkleProof {
            leaf_index: token_index,
            siblings: Vec::new(),
        },
        margin_cert: None,
        final_hidden,
        token_id: None,
        prev_io_hash: None,
        prev_kv_hash: None,
    }
}
