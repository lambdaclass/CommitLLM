//! Toy model for end-to-end testing of the verification pipeline.
//!
//! Generates random INT8 weights, computes a fake forward pass,
//! generates a verifier key, produces a trace, and verifies it.
//! This validates the entire math pipeline before touching real models.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

use vi_core::constants::{MatrixType, ModelConfig};
use vi_core::field::Fp;
use vi_core::freivalds;
use vi_core::merkle;
use vi_core::types::{BatchCommitment, BatchProof, CommitmentVersion, DeploymentManifest, LayerTrace, TokenTrace, VerifierKey};

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

pub use vi_core::requantize;

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
        let h = vi_core::silu::compute_h_unit_scale(&g, &u);

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
            let h = vi_core::silu::compute_h_unit_scale(&g, &u);

            let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);
            x = requantize(&ffn_out);

            token_layers.push(LayerTrace {
                x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
                kv_cache_k: Vec::new(),
                kv_cache_v: Vec::new(),
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
            let h = vi_core::silu::compute_h_unit_scale(&g, &u);

            let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);
            x = requantize(&ffn_out);

            token_layers.push(LayerTrace {
                x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
                kv_cache_k: Vec::new(),
                kv_cache_v: Vec::new(),
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
        let h = vi_core::silu::compute_h_unit_scale(&g, &u);
        let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);
        x = requantize(&ffn_out);

        layers.push(LayerTrace {
            x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
            kv_cache_k: vec![k_i8],
            kv_cache_v: vec![v_i8],
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
            let h = vi_core::silu::compute_h_unit_scale(&g, &u);
            let ffn_out = matmul_i32(&lw.wd, &h, cfg.hidden_dim, cfg.ffn_dim);
            x = requantize(&ffn_out);

            // Snapshot the full KV cache at this point for Level C
            token_layers.push(LayerTrace {
                x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out,
                kv_cache_k: kv_cache[layer_idx].0.clone(),
                kv_cache_v: kv_cache[layer_idx].1.clone(),
            });
        }

        all_traces.push(token_layers);
    }

    all_traces
}

/// Prover-side state after Phase 1 (commit) but before Phase 2 (open).
///
/// Holds the Merkle trees and raw layer data. Only the commitment is
/// published; the full traces stay with the prover until challenged.
pub struct BatchState {
    pub trace_tree: merkle::MerkleTree,
    pub io_tree: merkle::MerkleTree,
    pub all_layers: Vec<Vec<LayerTrace>>,
    /// Manifest hash from the commitment, if any.
    pub manifest_hash: Option<[u8; 32]>,
    /// Per-token emitted token IDs (V2/V3 IO hash). `None` for legacy commits.
    pub token_ids: Option<Vec<u32>>,
    /// Protocol version for the IO hash format.
    pub version: CommitmentVersion,
    /// Prompt hash for request identity binding.
    pub prompt_hash: Option<[u8; 32]>,
    /// Seed commitment for sampling binding.
    pub seed_commitment: Option<[u8; 32]>,
    /// Revealed sampling seed (stored for open phase).
    pub revealed_seed: Option<[u8; 32]>,
    /// Per-token IO hashes (V3 only). Used to provide `prev_io_hash` during open.
    pub io_hashes: Option<Vec<[u8; 32]>>,
    /// Per-token KV chain hashes (V3 only). Used to provide `prev_kv_hash` during open.
    pub kv_chain_hashes: Option<Vec<[u8; 32]>>,
    /// KV chain Merkle tree root (V3 only).
    pub kv_chain_root: Option<[u8; 32]>,
    /// Per-token KV Merkle roots, computed during generation.
    /// Each root is the Merkle root over per-layer KV hashes for that token.
    /// Published before the challenge so the verifier can build a StreamingKvVerifier.
    pub kv_merkle_roots: Option<Vec<[u8; 32]>>,
}

/// Phase 1 — Commit (legacy V1 IO hash, no token-ID binding).
///
/// Builds the trace tree and IO tree over all N token traces.
/// Returns a small commitment (two roots + count) that the prover
/// publishes, plus opaque state for selective opening later.
/// No per-token Merkle proofs are generated at this stage.
///
/// Use `commit_with_token_ids` for V2 IO hash that binds emitted tokens.
pub fn commit(all_layers: Vec<Vec<LayerTrace>>) -> (BatchCommitment, BatchState) {
    commit_inner(all_layers, None)
}

/// Alias for `commit` — makes the legacy (no token-ID) path explicit in tests.
pub fn commit_legacy(all_layers: Vec<Vec<LayerTrace>>) -> (BatchCommitment, BatchState) {
    commit(all_layers)
}

/// Phase 1 — Commit with token-ID binding (V2 IO hash).
///
/// Same as `commit`, but each IO leaf includes the emitted token ID:
///   `H("vi-io-v2" || first_input || last_output || token_id)`
///
/// `token_ids.len()` must equal `all_layers.len()`.
pub fn commit_with_token_ids(
    all_layers: Vec<Vec<LayerTrace>>,
    token_ids: &[u32],
) -> (BatchCommitment, BatchState) {
    assert_eq!(
        all_layers.len(),
        token_ids.len(),
        "token_ids length must match number of tokens"
    );
    commit_inner(all_layers, Some(token_ids))
}

fn commit_inner(
    all_layers: Vec<Vec<LayerTrace>>,
    token_ids: Option<&[u32]>,
) -> (BatchCommitment, BatchState) {
    let trace_leaves: Vec<[u8; 32]> = all_layers
        .iter()
        .map(|layers| {
            let data = bincode::serialize(layers).expect("serialize layers");
            let fh = layers.last().map(|lt| requantize(&lt.ffn_out));
            merkle::trace_leaf_hash(&data, fh.as_deref())
        })
        .collect();

    let io_leaves: Vec<[u8; 32]> = all_layers
        .iter()
        .enumerate()
        .map(|(i, layers)| {
            let tid = token_ids.map(|ids| ids[i]);
            merkle::io_hash(&layers[0].x_attn, &layers.last().unwrap().ffn_out, &merkle::io_binding(tid, None))
        })
        .collect();

    let trace_tree = merkle::build_tree(&trace_leaves);
    let io_tree = merkle::build_tree(&io_leaves);

    let version = if token_ids.is_some() {
        CommitmentVersion::V2
    } else {
        CommitmentVersion::V1
    };

    let commitment = BatchCommitment {
        merkle_root: trace_tree.root,
        io_root: io_tree.root,
        n_tokens: all_layers.len() as u32,
        manifest_hash: None,
        version,
        prompt_hash: None,
        seed_commitment: None,
        kv_chain_root: None,
    };

    let state = BatchState {
        trace_tree,
        io_tree,
        all_layers,
        manifest_hash: None,
        token_ids: token_ids.map(|ids| ids.to_vec()),
        version,
        prompt_hash: None,
        seed_commitment: None,
        revealed_seed: None,
        io_hashes: None,
        kv_chain_hashes: None,
        kv_chain_root: None,
        kv_merkle_roots: None,
    };

    (commitment, state)
}

/// Phase 1 — Commit with deployment manifest binding.
///
/// Same as `commit()` but also computes `hash_manifest(&manifest)` and sets
/// `commitment.manifest_hash = Some(hash)`. This binds sampling parameters,
/// tokenizer identity, and EOS policy to the batch commitment.
pub fn commit_with_manifest(
    all_layers: Vec<Vec<LayerTrace>>,
    manifest: &DeploymentManifest,
) -> (BatchCommitment, BatchState) {
    let (mut commitment, mut state) = commit(all_layers);
    let mh = Some(merkle::hash_manifest(manifest));
    commitment.manifest_hash = mh;
    state.manifest_hash = mh;
    (commitment, state)
}

/// Full binding parameters for V3 commit.
pub struct FullBindingParams<'a> {
    pub token_ids: &'a [u32],
    pub prompt: &'a [u8],
    pub sampling_seed: [u8; 32],
    /// Optional deployment manifest. When provided, its hash is bound into
    /// the commitment alongside the other V3 fields.
    pub manifest: Option<&'a DeploymentManifest>,
}

/// Phase 1 — Commit with full binding (V3): token IDs, transcript chaining,
/// prompt hash, and sampling seed commitment.
///
/// IO leaves use chained hashing: each token's IO hash depends on the previous.
/// This prevents insertion, deletion, reordering, and retroactive edits.
pub fn commit_with_full_binding(
    all_layers: Vec<Vec<LayerTrace>>,
    params: &FullBindingParams,
) -> (BatchCommitment, BatchState) {
    assert_eq!(
        all_layers.len(),
        params.token_ids.len(),
        "token_ids length must match number of tokens"
    );

    let trace_leaves: Vec<[u8; 32]> = all_layers
        .iter()
        .map(|layers| {
            let data = bincode::serialize(layers).expect("serialize layers");
            let fh = layers.last().map(|lt| requantize(&lt.ffn_out));
            merkle::trace_leaf_hash(&data, fh.as_deref())
        })
        .collect();

    // Build chained IO leaves: each depends on the previous
    let mut io_leaves: Vec<[u8; 32]> = Vec::with_capacity(all_layers.len());
    let mut prev_io = [0u8; 32]; // genesis: zero hash for first token
    for (i, layers) in all_layers.iter().enumerate() {
        let binding = merkle::IoHashBinding::Chained {
            token_id: params.token_ids[i],
            prev_io_hash: prev_io,
        };
        let io = merkle::io_hash(
            &layers[0].x_attn,
            &layers.last().unwrap().ffn_out,
            &binding,
        );
        io_leaves.push(io);
        prev_io = io;
    }

    // Build KV provenance chain: each token's KV hash depends on the previous
    let mut kv_chain_leaves: Vec<[u8; 32]> = Vec::with_capacity(all_layers.len());
    let mut kv_merkle_roots: Vec<[u8; 32]> = Vec::with_capacity(all_layers.len());
    let mut prev_kv = [0u8; 32]; // genesis: zero hash for first token
    for (i, layers) in all_layers.iter().enumerate() {
        let k_per_layer: Vec<Vec<i8>> = layers.iter().map(|lt| requantize(&lt.k)).collect();
        let v_per_layer: Vec<Vec<i8>> = layers.iter().map(|lt| requantize(&lt.v)).collect();
        let kv = merkle::kv_chain_hash(&prev_kv, &k_per_layer, &v_per_layer, i as u32);
        kv_chain_leaves.push(kv);
        prev_kv = kv;

        // Compute per-token KV Merkle root (streaming commitment)
        let kv_tree = vi_core::streaming::build_kv_layer_tree(i as u32, &k_per_layer, &v_per_layer);
        kv_merkle_roots.push(kv_tree.root);
    }

    let trace_tree = merkle::build_tree(&trace_leaves);
    let io_tree = merkle::build_tree(&io_leaves);
    let kv_chain_tree = merkle::build_tree(&kv_chain_leaves);

    let manifest_hash = params.manifest.map(|m| merkle::hash_manifest(m));

    let commitment = BatchCommitment {
        merkle_root: trace_tree.root,
        io_root: io_tree.root,
        n_tokens: all_layers.len() as u32,
        manifest_hash,
        version: CommitmentVersion::V3,
        prompt_hash: Some(merkle::hash_prompt(params.prompt)),
        seed_commitment: Some(merkle::hash_seed(&params.sampling_seed)),
        kv_chain_root: Some(kv_chain_tree.root),
    };

    let state = BatchState {
        trace_tree,
        io_tree,
        all_layers,
        manifest_hash,
        token_ids: Some(params.token_ids.to_vec()),
        version: CommitmentVersion::V3,
        prompt_hash: Some(merkle::hash_prompt(params.prompt)),
        seed_commitment: Some(merkle::hash_seed(&params.sampling_seed)),
        revealed_seed: Some(params.sampling_seed),
        io_hashes: Some(io_leaves),
        kv_chain_hashes: Some(kv_chain_leaves),
        kv_chain_root: Some(kv_chain_tree.root),
        kv_merkle_roots: Some(kv_merkle_roots),
    };

    (commitment, state)
}

/// Phase 2 — Open.
///
/// Given the prover's internal state and a set of challenge indices,
/// generates Merkle proofs *only* for the challenged tokens and
/// assembles the BatchProof. This is O(k log N) instead of O(N log N).
pub fn open(state: &BatchState, challenge_indices: &[u32]) -> BatchProof {
    let commitment = BatchCommitment {
        merkle_root: state.trace_tree.root,
        io_root: state.io_tree.root,
        n_tokens: state.all_layers.len() as u32,
        manifest_hash: state.manifest_hash,
        version: state.version,
        prompt_hash: state.prompt_hash,
        seed_commitment: state.seed_commitment,
        kv_chain_root: state.kv_chain_root,
    };

    let traces: Vec<TokenTrace> = challenge_indices
        .iter()
        .map(|&idx| {
            let i = idx as usize;
            let merkle_proof = merkle::prove(&state.trace_tree, i);
            let io_proof = merkle::prove(&state.io_tree, i);
            let token_id = state.token_ids.as_ref().map(|ids| ids[i]);
            // For V3, provide prev_io_hash so the verifier can reconstruct
            // the chain even for non-consecutive challenge sets.
            let prev_io_hash = state.io_hashes.as_ref().map(|hashes| {
                if i == 0 {
                    [0u8; 32] // genesis zero hash
                } else {
                    hashes[i - 1]
                }
            });
            // For KV provenance chaining, provide prev_kv_hash.
            let prev_kv_hash = state.kv_chain_hashes.as_ref().map(|hashes| {
                if i == 0 {
                    [0u8; 32] // genesis zero hash
                } else {
                    hashes[i - 1]
                }
            });
            // Derive final_hidden from last layer's ffn_out (requantize i32 → i8).
            let final_hidden = state.all_layers[i]
                .last()
                .map(|lt| requantize(&lt.ffn_out));

            TokenTrace {
                token_index: idx,
                layers: state.all_layers[i].clone(),
                merkle_root: state.trace_tree.root,
                merkle_proof,
                io_proof,
                margin_cert: None,
                final_hidden,
                token_id,
                prev_io_hash,
                prev_kv_hash,
            }
        })
        .collect();

    BatchProof {
        commitment,
        traces,
        revealed_seed: state.revealed_seed,
    }
}

/// Convenience wrapper: commit + open all tokens (for tests that need every trace).
pub fn build_batch(
    all_layers: Vec<Vec<LayerTrace>>,
) -> (BatchCommitment, BatchState, Vec<TokenTrace>) {
    let n = all_layers.len() as u32;
    let (commitment, state) = commit(all_layers);
    let all_indices: Vec<u32> = (0..n).collect();
    let proof = open(&state, &all_indices);
    (commitment, state, proof.traces)
}

/// Build an `AuditResponse` for a stratified audit challenge.
///
/// Standalone version that recomputes KV and trace Merkle trees from raw traces.
/// Prefer `build_audit_response_from_state` when a `BatchState` is available.
pub fn build_audit_response(
    all_traces: &[Vec<LayerTrace>],
    challenge: &vi_core::types::AuditChallenge,
) -> vi_core::types::AuditResponse {
    use vi_core::streaming;

    let token_idx = challenge.token_index as usize;
    assert!(token_idx < all_traces.len(), "token_index out of range");
    let token_layers = &all_traces[token_idx];

    let (partial_layers, kv_k_prefix, kv_v_prefix) = extract_audit_layers(token_layers, &challenge.layer_indices);

    // Build KV layer tree for per-layer Merkle proofs.
    let k_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.k)).collect();
    let v_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.v)).collect();
    let kv_tree = streaming::build_kv_layer_tree(challenge.token_index, &k_per_layer, &v_per_layer);

    let kv_layer_proofs: Vec<vi_core::merkle::MerkleProof> = challenge
        .layer_indices.iter().map(|&l| vi_core::merkle::prove(&kv_tree, l)).collect();

    // Build Merkle commitment for the full trace (all layers)
    let n_tokens = all_traces.len();
    let mut leaves = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let data = bincode::serialize(&all_traces[t]).expect("serialize layers");
        leaves.push(vi_core::merkle::trace_leaf_hash(&data, None));
    }
    let trace_tree = vi_core::merkle::build_tree(&leaves);
    let merkle_proof = vi_core::merkle::prove(&trace_tree, token_idx);

    vi_core::types::AuditResponse {
        token_index: challenge.token_index,
        partial_layers,
        layer_indices: challenge.layer_indices.clone(),
        kv_k_prefix,
        kv_v_prefix,
        kv_layer_proofs,
        merkle_root: trace_tree.root,
        merkle_proof,
        final_hidden: None,
        token_id: None,
    }
}

/// Build an `AuditResponse` using pre-computed trees from `BatchState`.
///
/// Uses `state.trace_tree` for the Merkle proof and recomputes only the
/// per-token KV layer tree (needed for per-layer opening proofs).
pub fn build_audit_response_from_state(
    state: &BatchState,
    challenge: &vi_core::types::AuditChallenge,
) -> vi_core::types::AuditResponse {
    use vi_core::streaming;

    let token_idx = challenge.token_index as usize;
    assert!(token_idx < state.all_layers.len(), "token_index out of range");
    let token_layers = &state.all_layers[token_idx];

    let (partial_layers, kv_k_prefix, kv_v_prefix) = extract_audit_layers(token_layers, &challenge.layer_indices);

    // KV layer tree: must rebuild for per-layer proofs (we only stored the roots).
    let k_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.k)).collect();
    let v_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.v)).collect();
    let kv_tree = streaming::build_kv_layer_tree(challenge.token_index, &k_per_layer, &v_per_layer);

    let kv_layer_proofs: Vec<vi_core::merkle::MerkleProof> = challenge
        .layer_indices.iter().map(|&l| vi_core::merkle::prove(&kv_tree, l)).collect();

    // Use pre-computed trace tree from BatchState.
    let merkle_proof = vi_core::merkle::prove(&state.trace_tree, token_idx);

    vi_core::types::AuditResponse {
        token_index: challenge.token_index,
        partial_layers,
        layer_indices: challenge.layer_indices.clone(),
        kv_k_prefix,
        kv_v_prefix,
        kv_layer_proofs,
        merkle_root: state.trace_tree.root,
        merkle_proof,
        final_hidden: None,
        token_id: None,
    }
}

/// Extract partial layers and KV prefix data for a set of layer indices.
fn extract_audit_layers(
    token_layers: &[LayerTrace],
    layer_indices: &[usize],
) -> (Vec<LayerTrace>, Vec<Vec<Vec<i8>>>, Vec<Vec<Vec<i8>>>) {
    let partial_layers: Vec<LayerTrace> = layer_indices.iter().map(|&l| token_layers[l].clone()).collect();
    let kv_k_prefix: Vec<Vec<Vec<i8>>> = layer_indices.iter().map(|&l| token_layers[l].kv_cache_k.clone()).collect();
    let kv_v_prefix: Vec<Vec<Vec<i8>>> = layer_indices.iter().map(|&l| token_layers[l].kv_cache_v.clone()).collect();
    (partial_layers, kv_k_prefix, kv_v_prefix)
}

/// Build a `StreamingKvVerifier` from full trace data.
///
/// Computes per-token KV Merkle roots from the requantized K/V across all layers.
pub fn build_streaming_kv_verifier(
    all_traces: &[Vec<LayerTrace>],
) -> vi_core::streaming::StreamingKvVerifier {
    use vi_core::streaming;

    let mut verifier = streaming::StreamingKvVerifier::new();
    for (t, token_layers) in all_traces.iter().enumerate() {
        let k_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.k)).collect();
        let v_per_layer: Vec<Vec<i8>> = token_layers.iter().map(|lt| requantize(&lt.v)).collect();
        let tree = streaming::build_kv_layer_tree(t as u32, &k_per_layer, &v_per_layer);
        verifier.ingest(tree.root);
    }
    verifier
}

/// Verify a batch proof: check each opened trace, IO commitment, and cross-token chains.
pub fn verify_batch(
    key: &VerifierKey,
    proof: &BatchProof,
    expected_challenges: &[u32],
) -> (bool, Vec<String>) {
    use vi_core::types::VerificationPolicy;
    verify_batch_with_policy(key, proof, expected_challenges, &VerificationPolicy::default())
}

/// Verify a batch proof with explicit verification policy.
pub fn verify_batch_with_policy(
    key: &VerifierKey,
    proof: &BatchProof,
    expected_challenges: &[u32],
    policy: &vi_core::types::VerificationPolicy,
) -> (bool, Vec<String>) {
    let mut failures = Vec::new();

    if proof.traces.len() != expected_challenges.len() {
        failures.push(format!(
            "expected {} challenged traces, got {}",
            expected_challenges.len(),
            proof.traces.len()
        ));
        return (false, failures);
    }

    // Policy: minimum version enforcement
    if let Some(min_ver) = policy.min_version {
        if (proof.commitment.version as u32) < (min_ver as u32) {
            failures.push(format!(
                "commitment version {:?} below minimum required {:?}",
                proof.commitment.version, min_ver
            ));
        }
    }

    // Policy: expected prompt hash verification
    if let Some(expected_prompt) = policy.expected_prompt_hash {
        match proof.commitment.prompt_hash {
            Some(actual) if actual != expected_prompt => {
                failures.push("prompt_hash mismatch: commitment prompt_hash != expected".into());
            }
            None => {
                failures.push("policy requires prompt_hash but commitment has None".into());
            }
            _ => {}
        }
    }

    // Policy: expected manifest hash verification
    if let Some(expected_manifest) = policy.expected_manifest_hash {
        match proof.commitment.manifest_hash {
            Some(actual) if actual != expected_manifest => {
                failures.push("manifest_hash mismatch: commitment manifest_hash != expected".into());
            }
            None => {
                failures.push("policy requires manifest_hash but commitment has None".into());
            }
            _ => {}
        }
    }

    // Version consistency checks
    if proof.commitment.version == CommitmentVersion::V2
        || proof.commitment.version == CommitmentVersion::V3
    {
        for trace in &proof.traces {
            if trace.token_id.is_none() {
                failures.push(format!(
                    "token {}: {:?} commitment requires token_id but trace has None",
                    trace.token_index, proof.commitment.version
                ));
            }
        }
    }

    // V3: seed commitment verification
    if proof.commitment.version == CommitmentVersion::V3 {
        match (proof.commitment.seed_commitment, proof.revealed_seed) {
            (Some(expected), Some(revealed)) => {
                let computed = merkle::hash_seed(&revealed);
                if computed != expected {
                    failures.push("seed commitment mismatch: hash(revealed_seed) != commitment.seed_commitment".into());
                }
            }
            (Some(_), None) => {
                failures.push("V3 commitment has seed_commitment but proof is missing revealed_seed".into());
            }
            (None, _) => {
                failures.push("V3 commitment is missing seed_commitment".into());
            }
        }
        if proof.commitment.prompt_hash.is_none() {
            failures.push("V3 commitment is missing prompt_hash".into());
        }

        // V3: all traces must have prev_io_hash
        for trace in &proof.traces {
            if trace.prev_io_hash.is_none() {
                failures.push(format!(
                    "token {}: V3 trace is missing prev_io_hash for chain verification",
                    trace.token_index
                ));
            }
        }
    }

    // Index opened traces by token_index for cross-token lookups
    let mut opened_by_index: std::collections::BTreeMap<u32, &TokenTrace> =
        std::collections::BTreeMap::new();

    // For V3, build IO hash map using prev_io_hash from each trace.
    let mut io_hash_map: std::collections::BTreeMap<u32, [u8; 32]> = std::collections::BTreeMap::new();
    if proof.commitment.version == CommitmentVersion::V3 {
        let mut sorted: Vec<&TokenTrace> = proof.traces.iter().collect();
        sorted.sort_by_key(|t| t.token_index);

        for trace in &sorted {
            let first_input = &trace.layers[0].x_attn;
            let last_output = &trace.layers.last().unwrap().ffn_out;

            if let (Some(tid), Some(claimed_prev)) = (trace.token_id, trace.prev_io_hash) {
                // Cross-check: if previous token is also opened, claimed prev must match
                if trace.token_index > 0 {
                    if let Some(&computed_prev) = io_hash_map.get(&(trace.token_index - 1)) {
                        if claimed_prev != computed_prev {
                            failures.push(format!(
                                "token {}: prev_io_hash mismatch — claimed != recomputed from token {}",
                                trace.token_index, trace.token_index - 1
                            ));
                        }
                    }
                }

                let binding = merkle::IoHashBinding::Chained {
                    token_id: tid,
                    prev_io_hash: claimed_prev,
                };
                let io = merkle::io_hash(first_input, last_output, &binding);
                io_hash_map.insert(trace.token_index, io);
            }
        }
    }

    for (i, trace) in proof.traces.iter().enumerate() {
        // Check token index matches challenge
        if trace.token_index != expected_challenges[i] {
            failures.push(format!(
                "trace {} has token_index={}, expected {}",
                i, trace.token_index, expected_challenges[i]
            ));
            continue;
        }

        // Check merkle proof leaf_index matches token_index
        if trace.merkle_proof.leaf_index != trace.token_index {
            failures.push(format!(
                "token {}: merkle_proof.leaf_index={} doesn't match token_index",
                trace.token_index, trace.merkle_proof.leaf_index
            ));
        }

        // Check merkle_root matches commitment
        if trace.merkle_root != proof.commitment.merkle_root {
            failures.push(format!(
                "token {}: merkle_root doesn't match commitment",
                trace.token_index
            ));
        }

        // Verify IO proof
        let first_input = &trace.layers[0].x_attn;
        let last_output = &trace.layers.last().unwrap().ffn_out;

        let io = if proof.commitment.version == CommitmentVersion::V3 {
            if let Some(&precomputed) = io_hash_map.get(&trace.token_index) {
                precomputed
            } else {
                merkle::io_hash(first_input, last_output, &merkle::io_binding(trace.token_id, None))
            }
        } else {
            merkle::io_hash(first_input, last_output, &merkle::io_binding(trace.token_id, None))
        };

        if !merkle::verify(&proof.commitment.io_root, &io, &trace.io_proof) {
            failures.push(format!(
                "token {}: IO proof verification failed",
                trace.token_index
            ));
        }

        // Run full single-token verification
        let (passed, token_failures) = verify_trace(key, trace);
        if !passed {
            for f in token_failures {
                failures.push(format!("token {}: {}", trace.token_index, f));
            }
        }

        opened_by_index.insert(trace.token_index, trace);
    }

    // Cross-token chain: for consecutive opened tokens, verify input/output chain.
    // Token t+1's first-layer input must equal requantize(token t's last-layer output).
    for (&idx, &trace) in &opened_by_index {
        if idx == 0 { continue; }
        if let Some(&prev_trace) = opened_by_index.get(&(idx - 1)) {
            let expected_input = requantize(&prev_trace.layers.last().unwrap().ffn_out);
            if trace.layers[0].x_attn != expected_input {
                failures.push(format!(
                    "token {}->{}: cross-token chain failed (input != requant(prev output))",
                    idx - 1, idx
                ));
            }
        }
    }

    // KV provenance chain: verify prev_kv_hash cross-checks for consecutive opened tokens.
    if proof.commitment.kv_chain_root.is_some() {
        // Recompute KV chain hash for each opened token and cross-check consecutives
        let mut kv_hashes: std::collections::BTreeMap<u32, [u8; 32]> = std::collections::BTreeMap::new();
        for (&pos, &trace) in &opened_by_index {
            if let Some(claimed_prev) = trace.prev_kv_hash {
                let k_per_layer: Vec<Vec<i8>> = trace.layers.iter()
                    .map(|lt| requantize(&lt.k))
                    .collect();
                let v_per_layer: Vec<Vec<i8>> = trace.layers.iter()
                    .map(|lt| requantize(&lt.v))
                    .collect();
                let kv = merkle::kv_chain_hash(&claimed_prev, &k_per_layer, &v_per_layer, pos);
                kv_hashes.insert(pos, kv);

                if pos > 0 {
                    if let Some(&computed_prev) = kv_hashes.get(&(pos - 1)) {
                        if claimed_prev != computed_prev {
                            failures.push(format!(
                                "token {}: prev_kv_hash mismatch — claimed != recomputed from token {}",
                                pos, pos - 1
                            ));
                        }
                    }
                }
            }
        }
    }

    let passed = failures.is_empty();
    (passed, failures)
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
    let weight_hash = vi_core::merkle::hash_weights(
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
        weight_hash: Some(weight_hash),
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
    key.lm_head = lm_head;
    key.wo_norms = model
        .iter()
        .map(|lw| {
            vi_core::margin::compute_matrix_linf_norm(
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
            vi_core::margin::compute_matrix_linf_norm(
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
            let data = bincode::serialize(&layers).expect("serialize layers");
            leaves.push(merkle::trace_leaf_hash(&data, final_hidden.as_deref()));
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

/// Verify a token trace against a verifier key.
/// Returns (passed, list of failure descriptions).
pub fn verify_trace(key: &VerifierKey, trace: &TokenTrace) -> (bool, Vec<String>) {
    let mut failures = Vec::new();

    // Step 1: Merkle commitment binding (includes final_hidden if present)
    let leaf_data = bincode::serialize(&trace.layers).expect("serialize layers");
    let leaf_hash = merkle::trace_leaf_hash(&leaf_data, trace.final_hidden.as_deref());
    if !merkle::verify(&trace.merkle_root, &leaf_hash, &trace.merkle_proof) {
        failures.push("Merkle proof verification failed".into());
    }

    for (layer_idx, lt) in trace.layers.iter().enumerate() {
        // Step 2: Freivalds checks for each matrix type
        let checks: [(MatrixType, &[i8], &[i32]); 7] = [
            (MatrixType::Wq, &lt.x_attn, &lt.q),
            (MatrixType::Wk, &lt.x_attn, &lt.k),
            (MatrixType::Wv, &lt.x_attn, &lt.v),
            (MatrixType::Wo, &lt.a, &lt.attn_out),
            (MatrixType::Wg, &lt.x_ffn, &lt.g),
            (MatrixType::Wu, &lt.x_ffn, &lt.u),
            (MatrixType::Wd, &lt.h, &lt.ffn_out),
        ];

        for (mt, input, output) in &checks {
            let v = key.v_for(layer_idx, *mt);
            let r = key.r_for(*mt);

            if !freivalds::check(v, input, r, output) {
                failures.push(format!(
                    "layer {} {:?}: Freivalds check failed",
                    layer_idx, mt
                ));
            }
        }

        // Step 3: SiLU verification — h == SiLU(requant(g)) * requant(u) via LUT
        let expected_h = vi_core::silu::compute_h_unit_scale(&lt.g, &lt.u);
        if lt.h != expected_h {
            failures.push(format!(
                "layer {}: SiLU check failed (h mismatch)",
                layer_idx
            ));
        }

        // Step 5a: Intra-layer chain — x_ffn == requantize(attn_out)
        let expected_x_ffn = requantize(&lt.attn_out);
        if lt.x_ffn != expected_x_ffn {
            failures.push(format!(
                "layer {}: chain check failed (x_ffn != requantize(attn_out))",
                layer_idx
            ));
        }

        // Step 5b: Cross-layer chain — x_attn[i+1] == requantize(ffn_out[i])
        if layer_idx + 1 < trace.layers.len() {
            let expected_next_input = requantize(&lt.ffn_out);
            if trace.layers[layer_idx + 1].x_attn != expected_next_input {
                failures.push(format!(
                    "layer {}->{}: chain check failed (next x_attn != requantize(ffn_out))",
                    layer_idx,
                    layer_idx + 1
                ));
            }
        }
    }

    let passed = failures.is_empty();
    (passed, failures)
}
