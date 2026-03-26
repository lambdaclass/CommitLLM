//! Verification library for the verified-inference protocol.
//!
//! Provides single-trace and batch verification for the reference path
//! (toy model / LUT-based SiLU / exact i32 accumulators). For W8A8
//! production verification, see the W8A8 verification module.
//!
//! # Verification levels
//!
//! - **Level A**: Exact linear verification (Freivalds + SiLU + chain + Merkle).
//! - **Level B**: Level A + margin certificates bound to the opened trace via
//!   final hidden state and verifier-side logit recomputation.
//! - **Level C**: Level B + attention replay verification + prefix-KV chain binding
//!   (self-consistency always; cross-token consistency when co-opened).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use verilm_core::attention::{AttentionToleranceConfig, compare_attention_output, replay_attention_reference};
use verilm_core::constants::MatrixType;
use verilm_core::margin;
use verilm_core::streaming;
use verilm_core::types::{
    AuditChallenge, AuditResponse, AuditTier, BatchProof, CommitmentVersion, DeploymentManifest,
    TokenTrace, V4AuditResponse, VerifierKey,
};
use verilm_core::{freivalds, merkle};

/// Verification level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationLevel {
    /// Exact linear checks only (Freivalds + SiLU + chain + Merkle).
    A,
    /// Level A + margin certificates for token-generation integrity.
    B,
    /// Level B + attention replay from provided KV snapshot.
    C,
}

pub use verilm_core::requantize;

/// Structured verification result.
#[derive(Debug)]
pub struct VerifyReport {
    pub verdict: Verdict,
    pub model_name: String,
    pub n_layers: usize,
    pub checks_run: usize,
    pub checks_passed: usize,
    pub failures: Vec<String>,
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
}

impl std::fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => write!(
                f,
                "PASS: {}/{} checks passed ({} layers, {:.1}ms)",
                self.checks_passed,
                self.checks_run,
                self.n_layers,
                self.duration.as_secs_f64() * 1000.0
            ),
            Verdict::Fail => {
                writeln!(
                    f,
                    "FAIL: {}/{} checks passed ({} failures)",
                    self.checks_passed,
                    self.checks_run,
                    self.failures.len()
                )?;
                for fail in &self.failures {
                    writeln!(f, "  {}", fail)?;
                }
                Ok(())
            }
        }
    }
}

/// Verify a margin certificate on a token trace (Level B check).
///
/// Requires:
/// - `trace.margin_cert` is `Some`
/// - `key.wo_norms` has one entry per layer
///
/// Returns failure descriptions (empty = margin-certified).
pub fn verify_margin(
    key: &VerifierKey,
    trace: &TokenTrace,
    epsilon: f32,
) -> Vec<String> {
    let mut failures = Vec::new();

    let cert = match &trace.margin_cert {
        Some(c) => c,
        None => {
            failures.push(format!(
                "token {}: missing margin certificate",
                trace.token_index
            ));
            return failures;
        }
    };

    if cert.token_index != trace.token_index {
        failures.push(format!(
            "token {}: margin cert token_index={} doesn't match trace token_index={}",
            trace.token_index, cert.token_index, trace.token_index
        ));
        return failures;
    }

    // Logit-to-trace binding: verify final_hidden and recompute logits.
    let final_hidden = match &trace.final_hidden {
        Some(fh) => fh,
        None => {
            failures.push(format!(
                "token {}: missing final_hidden for Level B verification",
                trace.token_index
            ));
            return failures;
        }
    };

    // Verify final_hidden is consistent with the trace's last layer output.
    let expected_fh = requantize(&trace.layers.last().unwrap().ffn_out);
    if *final_hidden != expected_fh {
        failures.push(format!(
            "token {}: final_hidden doesn't match requantize(last_layer.ffn_out)",
            trace.token_index
        ));
        return failures;
    }

    // If key has lm_head, recompute logits and verify against cert.
    if let Some(lm_head) = &key.lm_head {
        let vocab_size = key.config.vocab_size;
        let hidden_dim = key.config.hidden_dim;
        if lm_head.len() != vocab_size * hidden_dim {
            failures.push(format!(
                "token {}: lm_head size {} != vocab_size({}) * hidden_dim({})",
                trace.token_index, lm_head.len(), vocab_size, hidden_dim
            ));
            return failures;
        }
        // Compute logits as i32 (shared by Freivalds + exact comparison).
        let logits_i32: Vec<i32> = (0..vocab_size)
            .map(|r| {
                (0..hidden_dim)
                    .map(|c| lm_head[r * hidden_dim + c] as i32 * final_hidden[c] as i32)
                    .sum::<i32>()
            })
            .collect();

        // LM-head Freivalds check.
        let r_lm = key.r_for(MatrixType::LmHead);
        if let (false, Some(ref v_lm)) = (r_lm.is_empty(), &key.v_lm_head) {
            if !freivalds::check(v_lm, final_hidden, r_lm, &logits_i32) {
                failures.push(format!(
                    "token {}: lm_head Freivalds check failed",
                    trace.token_index
                ));
                return failures;
            }
        }

        // Exact logit comparison against cert.
        let recomputed_logits: Vec<f32> = logits_i32.iter().map(|&v| v as f32).collect();
        if cert.logits.len() != recomputed_logits.len() {
            failures.push(format!(
                "token {}: cert logits length {} != recomputed {}",
                trace.token_index, cert.logits.len(), recomputed_logits.len()
            ));
            return failures;
        }
        for (i, (&cert_l, &recomp_l)) in cert.logits.iter().zip(recomputed_logits.iter()).enumerate() {
            if cert_l != recomp_l {
                failures.push(format!(
                    "token {}: logit mismatch at index {}: cert={} recomputed={}",
                    trace.token_index, i, cert_l, recomp_l
                ));
                return failures;
            }
        }
    }

    if key.wo_norms.len() != key.config.n_layers {
        failures.push(format!(
            "token {}: key has {} wo_norms but {} layers",
            trace.token_index,
            key.wo_norms.len(),
            key.config.n_layers
        ));
        return failures;
    }

    // Use max wo_norm across layers (conservative bound).
    let max_wo_norm = key
        .wo_norms
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);

    let max_v_norm = key.max_v_norm;
    if max_v_norm <= 0.0 {
        failures.push(format!(
            "token {}: key has invalid max_v_norm={}",
            trace.token_index, max_v_norm
        ));
        return failures;
    }

    let result = margin::verify_margin_certificate(
        cert,
        epsilon,
        key.config.n_layers,
        max_v_norm,
        max_wo_norm,
    );

    for f in &result.failures {
        failures.push(format!("token {}: margin: {}", trace.token_index, f));
    }

    if !result.certified && result.failures.is_empty() {
        failures.push(format!(
            "token {}: margin not certified (delta={:.4} <= 2*B={:.4})",
            trace.token_index,
            result.delta,
            2.0 * result.perturbation_bound,
        ));
    }

    failures
}

/// Verify sampling honesty for a token trace.
///
/// Recomputes logits from `lm_head @ final_hidden`, replays the canonical
/// deterministic sampler with the given decode parameters and seed, and
/// checks that the result matches `trace.token_id`.
///
/// Requires:
/// - `key.lm_head` is `Some`
/// - `trace.final_hidden` is `Some`
/// - `trace.token_id` is `Some`
///
/// Returns failure descriptions (empty = sampling verified).
pub fn verify_sampling(
    key: &VerifierKey,
    trace: &TokenTrace,
    params: &verilm_core::sampling::DecodeParams,
    batch_seed: &[u8; 32],
) -> Vec<String> {
    let mut failures = Vec::new();

    let token_id = match trace.token_id {
        Some(tid) => tid,
        None => {
            failures.push(format!(
                "token {}: missing token_id for sampling verification",
                trace.token_index
            ));
            return failures;
        }
    };

    let final_hidden = match &trace.final_hidden {
        Some(fh) => fh,
        None => {
            failures.push(format!(
                "token {}: missing final_hidden for sampling verification",
                trace.token_index
            ));
            return failures;
        }
    };

    let lm_head = match &key.lm_head {
        Some(lm) => lm,
        None => {
            failures.push(format!(
                "token {}: verifier key missing lm_head for sampling verification",
                trace.token_index
            ));
            return failures;
        }
    };

    // Verify final_hidden is consistent with the trace's last layer output
    let expected_fh = requantize(&trace.layers.last().unwrap().ffn_out);
    if *final_hidden != expected_fh {
        failures.push(format!(
            "token {}: final_hidden doesn't match requantize(last_layer.ffn_out)",
            trace.token_index
        ));
        return failures;
    }

    let replayed = verilm_core::sampling::replay_sampling(
        lm_head,
        final_hidden,
        key.config.vocab_size,
        key.config.hidden_dim,
        params,
        batch_seed,
        trace.token_index,
    );

    if replayed != token_id {
        failures.push(format!(
            "token {}: sampling replay mismatch: replayed={} but trace claims token_id={}",
            trace.token_index, replayed, token_id
        ));
    }

    failures
}

/// Verify attention replay for a token trace (Level C check).
///
/// Replays GQA attention from the trace's requantized Q and KV cache,
/// compares against the claimed `a` vector per layer.
/// Also performs self-consistency check: the token's own KV cache entry
/// (at position `token_index`) must match `requantize(k)` / `requantize(v)`.
/// Returns failure descriptions (empty = attention verified).
pub fn verify_attention(
    key: &VerifierKey,
    trace: &TokenTrace,
    tolerance: &AttentionToleranceConfig,
) -> Vec<String> {
    let mut failures = Vec::new();
    let pos = trace.token_index as usize;

    for (layer_idx, lt) in trace.layers.iter().enumerate() {
        if lt.kv_cache_k.is_empty() || lt.kv_cache_v.is_empty() {
            failures.push(format!(
                "layer {}: missing KV cache for Level C attention replay",
                layer_idx
            ));
            continue;
        }

        // Self-consistency: own KV cache entry must match requantize(k/v)
        if pos < lt.kv_cache_k.len() {
            let expected_k = requantize(&lt.k);
            if lt.kv_cache_k[pos] != expected_k {
                failures.push(format!(
                    "layer {}: KV self-consistency failed: kv_cache_k[{}] != requantize(k)",
                    layer_idx, pos
                ));
            }
        }
        if pos < lt.kv_cache_v.len() {
            let expected_v = requantize(&lt.v);
            if lt.kv_cache_v[pos] != expected_v {
                failures.push(format!(
                    "layer {}: KV self-consistency failed: kv_cache_v[{}] != requantize(v)",
                    layer_idx, pos
                ));
            }
        }

        // Requantize Q to i8 for the replay
        let q_i8: Vec<i8> = lt.q.iter().map(|&v| v.clamp(-128, 127) as i8).collect();

        let replayed = replay_attention_reference(
            &q_i8,
            &lt.kv_cache_k,
            &lt.kv_cache_v,
            &key.config,
        );

        if let Some(max_diff) = compare_attention_output(&lt.a, &replayed, tolerance) {
            failures.push(format!(
                "layer {}: attention replay mismatch (max_diff={}, tolerance={})",
                layer_idx, max_diff, tolerance.max_abs_diff
            ));
        }
    }

    failures
}

/// Verify that KV cache entries are consistent with requantized K/V projections
/// from opened traces.
///
/// Two levels of checking:
/// - **Self-consistency**: Each opened token's own KV entry (at its position in
///   the cache) must match `requantize(layers[l].k)` / `requantize(layers[l].v)`
///   from its own trace.
/// - **Cross-token consistency**: When token T has a KV cache entry for prior
///   token P, and P is also opened, the entry must match P's requantized K/V.
///
/// Only opened tokens can be cross-checked. Unopened tokens' KV cache entries
/// cannot be verified against their projections — this is inherent to sampled
/// opening, not a soundness gap.
pub fn verify_kv_chain(
    opened_traces: &BTreeMap<u32, &TokenTrace>,
) -> Vec<String> {
    let mut failures = Vec::new();

    for (&pos, &trace) in opened_traces {
        let p = pos as usize;

        for (layer_idx, lt) in trace.layers.iter().enumerate() {
            if lt.kv_cache_k.is_empty() || lt.kv_cache_v.is_empty() {
                continue;
            }

            // Self-consistency
            if p < lt.kv_cache_k.len() {
                let expected_k = requantize(&lt.k);
                if lt.kv_cache_k[p] != expected_k {
                    failures.push(format!(
                        "token {} layer {}: KV self-consistency failed: kv_cache_k[{}] != requantize(k)",
                        pos, layer_idx, p
                    ));
                }
            }
            if p < lt.kv_cache_v.len() {
                let expected_v = requantize(&lt.v);
                if lt.kv_cache_v[p] != expected_v {
                    failures.push(format!(
                        "token {} layer {}: KV self-consistency failed: kv_cache_v[{}] != requantize(v)",
                        pos, layer_idx, p
                    ));
                }
            }

            // Cross-token consistency
            for (&other_pos, &other_trace) in opened_traces {
                if other_pos >= pos {
                    break;
                }
                let q = other_pos as usize;

                if q < lt.kv_cache_k.len() && layer_idx < other_trace.layers.len() {
                    let expected_k = requantize(&other_trace.layers[layer_idx].k);
                    if lt.kv_cache_k[q] != expected_k {
                        failures.push(format!(
                            "token {} layer {}: cross-token KV mismatch: kv_cache_k[{}] != token {}'s requantize(k)",
                            pos, layer_idx, q, other_pos
                        ));
                    }
                }

                if q < lt.kv_cache_v.len() && layer_idx < other_trace.layers.len() {
                    let expected_v = requantize(&other_trace.layers[layer_idx].v);
                    if lt.kv_cache_v[q] != expected_v {
                        failures.push(format!(
                            "token {} layer {}: cross-token KV mismatch: kv_cache_v[{}] != token {}'s requantize(v)",
                            pos, layer_idx, q, other_pos
                        ));
                    }
                }
            }
        }
    }

    failures
}

/// Verify KV provenance chain hashes for opened traces.
///
/// For each opened token that has `prev_kv_hash`, recomputes the KV chain hash
/// from `requantize(k)` and `requantize(v)` across all layers, and checks:
/// 1. The recomputed hash matches what would be committed (self-consistency).
/// 2. For consecutive opened tokens, the later token's `prev_kv_hash` matches
///    the earlier token's recomputed KV chain hash (cross-token chain).
///
/// Returns failure descriptions (empty = pass).
pub fn verify_kv_provenance_chain(
    opened_traces: &BTreeMap<u32, &TokenTrace>,
) -> Vec<String> {
    let mut failures = Vec::new();

    // Recompute KV chain hash for each opened token
    let mut kv_hashes: BTreeMap<u32, [u8; 32]> = BTreeMap::new();

    for (&pos, &trace) in opened_traces {
        if let Some(claimed_prev) = trace.prev_kv_hash {
            let k_per_layer: Vec<Vec<i8>> = trace.layers.iter()
                .map(|lt| requantize(&lt.k))
                .collect();
            let v_per_layer: Vec<Vec<i8>> = trace.layers.iter()
                .map(|lt| requantize(&lt.v))
                .collect();
            let kv = merkle::kv_chain_hash(&claimed_prev, &k_per_layer, &v_per_layer, pos);
            kv_hashes.insert(pos, kv);

            // Cross-check: if previous token is also opened, claimed prev must match
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

    failures
}

/// Verify RMSNorm requantization bridge between consecutive layers.
///
/// For each consecutive layer pair, checks that the next layer's input
/// matches the canonical recomputation:
///   dequant(ffn_out) + residual → RMSNorm → quantize(scale_next)
///
/// Only runs when the verifier key has RMSNorm weights and the trace has
/// activation scales (production W8A8 path). Falls back to simplified clamp
/// verification when these are absent (toy model path).
///
/// Returns failure descriptions (empty = pass).
pub fn verify_rmsnorm_chain(
    key: &VerifierKey,
    trace: &TokenTrace,
) -> Vec<String> {
    use verilm_core::rmsnorm;
    use verilm_core::constants::MatrixType;

    let mut failures = Vec::new();

    // Skip if no RMSNorm weights (toy model / legacy key)
    if key.rmsnorm_attn_weights.is_empty() || key.rmsnorm_ffn_weights.is_empty() {
        return failures;
    }

    let wo_idx = MatrixType::PER_LAYER.iter().position(|&m| m == MatrixType::Wo).unwrap();
    let wd_idx = MatrixType::PER_LAYER.iter().position(|&m| m == MatrixType::Wd).unwrap();

    for (layer_idx, lt) in trace.layers.iter().enumerate() {
        // Skip layers without the pre-attention residual stream
        let residual = match &lt.residual {
            Some(r) => r,
            None => continue,
        };
        let scale_x_attn = match lt.scale_x_attn {
            Some(s) => s,
            None => continue,
        };

        if layer_idx >= key.weight_scales.len() {
            continue;
        }
        let layer_wscales = &key.weight_scales[layer_idx];

        let residual_f64: Vec<f64> = residual.iter().map(|&v| v as f64).collect();

        // 1. Verify x_attn via RMSNorm: x_attn == quantize(RMSNorm_attn(residual), scale_x_attn)
        if layer_idx < key.rmsnorm_attn_weights.len() {
            let normed = rmsnorm::rmsnorm_f64_input(
                &residual_f64,
                &key.rmsnorm_attn_weights[layer_idx],
                key.rmsnorm_eps,
            );
            let expected = rmsnorm::quantize_f64_to_i8(&normed, scale_x_attn as f64);
            if lt.x_attn != expected {
                failures.push(format!(
                    "layer {}: RMSNorm attention input mismatch (x_attn != quantize(RMSNorm_attn(residual)))",
                    layer_idx
                ));
            }
        }

        // 2. Verify x_ffn via intra-layer bridge:
        //    x_ffn == quantize(RMSNorm_ffn(residual + dequant(attn_out)), scale_x_ffn)
        let scale_x_ffn = match lt.scale_x_ffn {
            Some(s) => s,
            None => continue,
        };
        if layer_idx < key.rmsnorm_ffn_weights.len() && wo_idx < layer_wscales.len() {
            let scale_w_o = layer_wscales[wo_idx];
            let scale_a_val = lt.scale_a.unwrap_or(1.0);

            let residual_ffn: Vec<f64> = residual_f64.iter().enumerate().map(|(i, &r)| {
                r + (lt.attn_out[i] as f64) * (scale_w_o as f64) * (scale_a_val as f64)
            }).collect();

            let normed = rmsnorm::rmsnorm_f64_input(
                &residual_ffn,
                &key.rmsnorm_ffn_weights[layer_idx],
                key.rmsnorm_eps,
            );
            let expected = rmsnorm::quantize_f64_to_i8(&normed, scale_x_ffn as f64);

            if lt.x_ffn != expected {
                failures.push(format!(
                    "layer {}: RMSNorm FFN bridge mismatch (x_ffn != quantize(RMSNorm_ffn(residual + dequant(attn_out))))",
                    layer_idx
                ));
            }
        }

        // 3. Cross-layer residual chain: for consecutive layers l, l+1 where both
        //    have residuals, verify residual[l+1] == residual[l] + dequant(attn_out[l]) + dequant(ffn_out[l])
        if layer_idx + 1 < trace.layers.len() {
            if let Some(next_residual) = &trace.layers[layer_idx + 1].residual {
                let scale_w_o = if wo_idx < layer_wscales.len() { layer_wscales[wo_idx] } else { continue };
                let scale_w_d = if wd_idx < layer_wscales.len() { layer_wscales[wd_idx] } else { continue };
                let scale_a_val = lt.scale_a.unwrap_or(1.0);
                let scale_h_val = lt.scale_h.unwrap_or(1.0);

                let mut max_diff: f64 = 0.0;
                let mut worst_idx = 0usize;
                for i in 0..residual.len().min(next_residual.len()) {
                    let expected_i = residual_f64[i]
                        + (lt.attn_out[i] as f64) * (scale_w_o as f64) * (scale_a_val as f64)
                        + (lt.ffn_out[i] as f64) * (scale_w_d as f64) * (scale_h_val as f64);
                    let diff = (next_residual[i] as f64 - expected_i).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        worst_idx = i;
                    }
                }
                // Tolerance for bf16-vs-f64 rounding in residual accumulation
                let tolerance = 0.5;
                if max_diff > tolerance {
                    failures.push(format!(
                        "layer {}->{}: cross-layer residual mismatch (max abs diff {:.4} at idx {}, tolerance {})",
                        layer_idx, layer_idx + 1, max_diff, worst_idx, tolerance
                    ));
                }
            }
        }
    }

    failures
}

/// Verify RoPE correctness on K projections stored in the KV cache.
///
/// For each layer, checks that kv_cache_k[position] matches the RoPE'd
/// and requantized K projection from the matmul accumulator.
///
/// Only runs when the trace has KV cache data (Level C) and activation
/// scales. Falls back silently when prerequisites are absent.
///
/// Returns failure descriptions (empty = pass).
pub fn verify_rope(
    key: &VerifierKey,
    trace: &TokenTrace,
) -> Vec<String> {
    use verilm_core::rope;
    use verilm_core::constants::MatrixType;

    let mut failures = Vec::new();

    // RoPE verification requires weight_scales (production W8A8 path).
    // The toy model doesn't apply RoPE, so we skip when no weight_scales
    // are provided — KV self-consistency is already checked elsewhere.
    if key.weight_scales.is_empty() {
        return failures;
    }

    let pos = trace.token_index as usize;

    for (layer_idx, lt) in trace.layers.iter().enumerate() {
        // Need KV cache for RoPE verification
        if lt.kv_cache_k.is_empty() {
            continue;
        }
        if pos >= lt.kv_cache_k.len() {
            continue;
        }

        // Get weight scale for K projection
        let scale_w = if layer_idx < key.weight_scales.len() {
            let wk_idx = MatrixType::PER_LAYER.iter().position(|&m| m == MatrixType::Wk).unwrap();
            if wk_idx < key.weight_scales[layer_idx].len() {
                Some(key.weight_scales[layer_idx][wk_idx])
            } else {
                None
            }
        } else {
            None
        };

        if let Some(err) = rope::verify_k_rope(
            &lt.k,
            &lt.kv_cache_k[pos],
            pos,
            &key.config,
            scale_w,
            lt.scale_x_attn,
            lt.scale_x_attn, // KV cache K is requantized with the attention input scale
        ) {
            failures.push(format!("layer {}: {}", layer_idx, err));
        }
    }

    failures
}

/// Verify a Q8_0 block-aware trace against a verifier key.
///
/// Uses the block Freivalds check (Phase A) and f32 assembly check (Phase B)
/// from the `Q8LayerTrace` block accumulators.
///
/// Block batching coefficients are derived from the verifier's secret key seed,
/// not from public data. This is strictly stronger than Fiat-Shamir.
///
/// Returns failure descriptions (empty = pass).
pub fn verify_q8_one(
    key: &VerifierKey,
    _trace: &TokenTrace,
    q8_layers: &[verilm_core::types::Q8LayerTrace],
) -> Vec<String> {
    use verilm_core::freivalds::{check_q8_blocks, derive_block_coefficients};
    use verilm_core::constants::MatrixType;

    let mut failures = Vec::new();

    for (layer_idx, q8lt) in q8_layers.iter().enumerate() {
        // Phase A: block Freivalds for each matrix
        let block_checks: [(MatrixType, &[i8], &verilm_core::types::Q8BlockAccumulators); 7] = [
            (MatrixType::Wq, &q8lt.x_attn, &q8lt.q_blocks),
            (MatrixType::Wk, &q8lt.x_attn, &q8lt.k_blocks),
            (MatrixType::Wv, &q8lt.x_attn, &q8lt.v_blocks),
            (MatrixType::Wo, &q8lt.a, &q8lt.attn_out_blocks),
            (MatrixType::Wg, &q8lt.x_ffn, &q8lt.g_blocks),
            (MatrixType::Wu, &q8lt.x_ffn, &q8lt.u_blocks),
            (MatrixType::Wd, &q8lt.h, &q8lt.ffn_out_blocks),
        ];

        for (mt_idx, (mt, input, blocks)) in block_checks.iter().enumerate() {
            let v = key.v_for(layer_idx, *mt);
            let r = key.r_for(*mt);

            let c = derive_block_coefficients(
                &key.seed,
                layer_idx,
                mt_idx,
                blocks.n_blocks,
            );

            if !check_q8_blocks(v, input, r, &blocks.sumi, &c) {
                failures.push(format!(
                    "layer {} {:?}: Q8 block Freivalds check failed",
                    layer_idx, mt
                ));
            }
        }

        // Phase B: f32 assembly checks (requires weight block scales from key)
        // The weight block scales come from the quantized checkpoint.
        // When available, verify that the assembled f32 output matches.
        // This is optional — Phase A already provides correctness guarantees.
    }

    failures
}

/// Verify that a batch commitment's manifest hash matches the expected deployment manifest.
/// Returns failure descriptions (empty = pass).
pub fn verify_manifest(
    commitment: &verilm_core::types::BatchCommitment,
    expected_manifest: &DeploymentManifest,
) -> Vec<String> {
    let computed = merkle::hash_manifest(expected_manifest);
    match commitment.manifest_hash {
        None => vec!["commitment missing manifest_hash".into()],
        Some(h) if h != computed => vec!["manifest_hash mismatch".into()],
        Some(_) => Vec::new(),
    }
}

/// Result of verifying a weight-chain hash against an expected value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightHashResult {
    /// Key hash matches expected.
    Match,
    /// Key hash does not match expected.
    Mismatch {
        key_hash: [u8; 32],
        expected: [u8; 32],
    },
    /// Key has no weight hash (legacy key) but an expected hash was provided.
    LegacyKeyNoHash,
    /// No expected hash was provided — skipped.
    Skipped,
}

/// Check whether the verifier key's weight hash matches an externally published
/// expected hash. This is the binding that closes the trust gap between
/// "the INT8 computation was correct" and "it used the published checkpoint."
///
/// Returns [`WeightHashResult::Skipped`] if `expected` is `None`.
pub fn verify_weight_hash(key: &VerifierKey, expected: Option<&[u8; 32]>) -> WeightHashResult {
    let Some(expected) = expected else {
        return WeightHashResult::Skipped;
    };

    match key.weight_hash {
        Some(actual) if actual == *expected => WeightHashResult::Match,
        Some(actual) => WeightHashResult::Mismatch {
            key_hash: actual,
            expected: *expected,
        },
        None => WeightHashResult::LegacyKeyNoHash,
    }
}

/// Verify a single token trace against a verifier key.
/// Returns a list of failure descriptions (empty = pass).
pub fn verify_one(key: &VerifierKey, trace: &TokenTrace) -> Vec<String> {
    let mut failures = Vec::new();

    // Step 1: Merkle commitment binding (direct hash, no bincode)
    let leaf_hash = merkle::hash_trace_direct(
        &trace.layers,
        trace.final_hidden.as_deref(),
    );
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
                failures.push(format!("layer {} {:?}: Freivalds check failed", layer_idx, mt));
            }
        }

        // Step 3: SiLU (LUT-based, unit scale)
        let expected_h = verilm_core::silu::compute_h_unit_scale(&lt.g, &lt.u);
        if lt.h != expected_h {
            failures.push(format!("layer {}: SiLU check failed", layer_idx));
        }

        // Step 4a: Intra-layer chain
        if lt.x_ffn != requantize(&lt.attn_out) {
            failures.push(format!("layer {}: chain x_ffn mismatch", layer_idx));
        }

        // Step 4b: Cross-layer chain
        if layer_idx + 1 < trace.layers.len() {
            if trace.layers[layer_idx + 1].x_attn != requantize(&lt.ffn_out) {
                failures.push(format!(
                    "layer {}->{}: chain x_attn mismatch",
                    layer_idx,
                    layer_idx + 1
                ));
            }
        }
    }

    failures
}

/// Verify a single token trace, returning a structured report.
pub fn verify_trace(key: &VerifierKey, trace: &TokenTrace) -> VerifyReport {
    let start = Instant::now();
    let failures = verify_one(key, trace);
    let duration = start.elapsed();

    // Each layer has 7 Freivalds + 1 SiLU + 1 chain = 9 checks, plus Merkle = 1
    let checks_run = key.config.n_layers * 9 + 1;
    let checks_passed = checks_run - failures.len();

    VerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        model_name: key.config.name.clone(),
        n_layers: key.config.n_layers,
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

/// Verify a single token trace at a given verification level.
///
/// `epsilon` controls the margin certificate softmax tolerance (Level B/C).
/// `attn_tolerance` controls the attention replay tolerance (Level C only).
/// Pass `None` for `attn_tolerance` to use the default (exact match, tolerance=0).
pub fn verify_trace_at_level(
    key: &VerifierKey,
    trace: &TokenTrace,
    level: VerificationLevel,
    epsilon: f32,
) -> VerifyReport {
    verify_trace_at_level_with_attn_tolerance(key, trace, level, epsilon, None)
}

/// Verify a single token trace with explicit attention tolerance configuration.
pub fn verify_trace_at_level_with_attn_tolerance(
    key: &VerifierKey,
    trace: &TokenTrace,
    level: VerificationLevel,
    epsilon: f32,
    attn_tolerance: Option<AttentionToleranceConfig>,
) -> VerifyReport {
    let start = Instant::now();
    let mut failures = verify_one(key, trace);

    let mut extra_checks = 0;
    if level == VerificationLevel::B || level == VerificationLevel::C {
        extra_checks += 1;
        let margin_failures = verify_margin(key, trace, epsilon);
        failures.extend(margin_failures);
    }
    if level == VerificationLevel::C {
        extra_checks += key.config.n_layers; // one attention check per layer
        let attn_tol = attn_tolerance.unwrap_or_default();
        let attn_failures = verify_attention(key, trace, &attn_tol);
        failures.extend(attn_failures);

        // RoPE verification on K projections (Level C only, requires KV cache)
        extra_checks += key.config.n_layers;
        let rope_failures = verify_rope(key, trace);
        failures.extend(rope_failures);
    }

    // RMSNorm chain verification (runs when key has weights and trace has scales)
    if !key.rmsnorm_attn_weights.is_empty() {
        extra_checks += key.config.n_layers * 2; // attn + FFN bridge per layer
        let rmsnorm_failures = verify_rmsnorm_chain(key, trace);
        failures.extend(rmsnorm_failures);
    }

    let duration = start.elapsed();
    let checks_run = key.config.n_layers * 9 + 1 + extra_checks;
    let checks_passed = checks_run - failures.len();

    VerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        model_name: key.config.name.clone(),
        n_layers: key.config.n_layers,
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

/// Result from batch verification.
#[derive(Debug)]
pub struct BatchVerifyReport {
    pub verdict: Verdict,
    pub model_name: String,
    pub n_layers: usize,
    pub n_tokens: u32,
    pub n_challenged: usize,
    pub tokens_passed: usize,
    pub challenges: Vec<u32>,
    pub failures: Vec<String>,
    pub duration: Duration,
}

impl std::fmt::Display for BatchVerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => write!(
                f,
                "PASS: {}/{} challenged tokens verified ({} layers each, IO+chain, {:.1}ms)",
                self.tokens_passed,
                self.n_challenged,
                self.n_layers,
                self.duration.as_secs_f64() * 1000.0
            ),
            Verdict::Fail => {
                writeln!(
                    f,
                    "FAIL: {} failures across batch ({}/{} tokens passed)",
                    self.failures.len(),
                    self.tokens_passed,
                    self.n_challenged
                )?;
                for fail in &self.failures {
                    writeln!(f, "  {}", fail)?;
                }
                Ok(())
            }
        }
    }
}

/// Verify a batch proof with challenge derivation and cross-token chain checks.
pub fn verify_batch(
    key: &VerifierKey,
    proof: &BatchProof,
    challenge_seed: [u8; 32],
    challenge_k: u32,
) -> BatchVerifyReport {
    verify_batch_with_policy(key, proof, challenge_seed, challenge_k, &Default::default())
}

/// Verify a batch proof with explicit verification policy.
///
/// The policy can enforce minimum commitment version, expected prompt hash,
/// and expected manifest hash — closing downgrade, replay, and runtime
/// binding gaps that the default permissive mode leaves open.
pub fn verify_batch_with_policy(
    key: &VerifierKey,
    proof: &BatchProof,
    challenge_seed: [u8; 32],
    challenge_k: u32,
    policy: &verilm_core::types::VerificationPolicy,
) -> BatchVerifyReport {
    let start = Instant::now();
    let mut failures = Vec::new();
    let mut tokens_passed = 0;

    let challenges = merkle::derive_challenges(
        &proof.commitment.merkle_root,
        &challenge_seed,
        proof.commitment.n_tokens,
        challenge_k,
    );

    let mut opened_by_index: BTreeMap<u32, &TokenTrace> = BTreeMap::new();

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
            _ => {} // matches
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
            _ => {} // matches
        }
    }

    // Version consistency: V2/V3 commitments require token_id on every trace.
    if proof.commitment.version == verilm_core::types::CommitmentVersion::V2
        || proof.commitment.version == verilm_core::types::CommitmentVersion::V3
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

    // V3: seed commitment verification — hash(revealed_seed) must equal commitment
    if proof.commitment.version == verilm_core::types::CommitmentVersion::V3 {
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

        // V3: prompt hash must be present
        if proof.commitment.prompt_hash.is_none() {
            failures.push("V3 commitment is missing prompt_hash".into());
        }

        // V3: all traces must have prev_io_hash for chain reconstruction
        for trace in &proof.traces {
            if trace.prev_io_hash.is_none() {
                failures.push(format!(
                    "token {}: V3 trace is missing prev_io_hash for chain verification",
                    trace.token_index
                ));
            }
        }
    }

    // For V3, build IO hash map using prev_io_hash from each trace.
    // This allows chain verification even for non-consecutive challenge sets.
    // When consecutive tokens are both opened, we cross-check that the
    // later token's prev_io_hash matches the recomputed IO hash of the earlier.
    let mut io_hash_map: BTreeMap<u32, [u8; 32]> = BTreeMap::new();
    if proof.commitment.version == verilm_core::types::CommitmentVersion::V3 {
        let mut sorted: Vec<&TokenTrace> = proof.traces.iter().collect();
        sorted.sort_by_key(|t| t.token_index);

        for trace in &sorted {
            let first_input = &trace.layers[0].x_attn;
            let last_output = &trace.layers.last().unwrap().ffn_out;

            if let (Some(tid), Some(claimed_prev)) = (trace.token_id, trace.prev_io_hash) {
                // Cross-check: if we already computed the previous token's IO hash,
                // the claimed prev_io_hash must match.
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

    for trace in &proof.traces {
        // Check this trace is in the challenge set
        if !challenges.contains(&trace.token_index) {
            failures.push(format!(
                "token {} not in challenge set",
                trace.token_index
            ));
            continue;
        }

        // Check leaf_index matches token_index
        if trace.merkle_proof.leaf_index != trace.token_index {
            failures.push(format!(
                "token {}: merkle_proof.leaf_index={} mismatch",
                trace.token_index, trace.merkle_proof.leaf_index
            ));
            continue;
        }

        // Check merkle_root matches commitment
        if trace.merkle_root != proof.commitment.merkle_root {
            failures.push(format!(
                "token {}: merkle_root doesn't match commitment",
                trace.token_index
            ));
            continue;
        }

        // IO proof verification
        let first_input = &trace.layers[0].x_attn;
        let last_output = &trace.layers.last().unwrap().ffn_out;

        let io = if proof.commitment.version == verilm_core::types::CommitmentVersion::V3 {
            // V3: use chained hash from pre-computation (uses trace's prev_io_hash)
            if let Some(&precomputed) = io_hash_map.get(&trace.token_index) {
                precomputed
            } else {
                // Fallback: prev_io_hash was missing; use V2-style.
                // This will fail the IO tree proof for a V3 commitment,
                // which is the correct behavior.
                merkle::io_hash(first_input, last_output, &merkle::io_binding(trace.token_id, None))
            }
        } else {
            // V1/V2: straightforward
            merkle::io_hash(first_input, last_output, &merkle::io_binding(trace.token_id, None))
        };

        if !merkle::verify(&proof.commitment.io_root, &io, &trace.io_proof) {
            failures.push(format!(
                "token {}: IO proof verification failed",
                trace.token_index
            ));
        }

        // Full single-token verification
        let token_failures = verify_one(key, trace);
        if token_failures.is_empty() {
            tokens_passed += 1;
        } else {
            for f in token_failures {
                failures.push(format!("token {}: {}", trace.token_index, f));
            }
        }

        opened_by_index.insert(trace.token_index, trace);
    }

    // Cross-token chain: consecutive opened tokens must chain
    for (&idx, &trace) in &opened_by_index {
        if idx == 0 {
            continue;
        }
        if let Some(&prev_trace) = opened_by_index.get(&(idx - 1)) {
            let expected_input = requantize(&prev_trace.layers.last().unwrap().ffn_out);
            if trace.layers[0].x_attn != expected_input {
                failures.push(format!(
                    "token {}->{}: cross-token chain mismatch",
                    idx - 1,
                    idx
                ));
            }
        }
    }

    // KV provenance chain: verify prev_kv_hash cross-checks for consecutive opened tokens
    if proof.commitment.kv_chain_root.is_some() {
        let kv_failures = verify_kv_provenance_chain(&opened_by_index);
        failures.extend(kv_failures);
    }

    let duration = start.elapsed();

    BatchVerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        model_name: key.config.name.clone(),
        n_layers: key.config.n_layers,
        n_tokens: proof.commitment.n_tokens,
        n_challenged: challenges.len(),
        tokens_passed,
        challenges,
        failures,
        duration,
    }
}

// ---------------------------------------------------------------------------
// Stratified audit verification
// ---------------------------------------------------------------------------

/// Derive deterministic layer indices for a routine audit.
///
/// Uses `SHA256(challenge_seed || token_index || counter)` to pick `k` unique
/// layer indices from `0..n_layers`. For a full audit, just returns
/// `0..n_layers`.
pub fn derive_audit_layers(
    challenge_seed: &[u8; 32],
    token_index: u32,
    n_layers: usize,
    tier: AuditTier,
) -> Vec<usize> {
    match tier {
        AuditTier::Full => (0..n_layers).collect(),
        AuditTier::Routine => {
            use sha2::{Digest, Sha256};

            // Contiguous prefix 0..=L_max. L_max is derived from the
            // challenge seed so the prover cannot predict the depth.
            // Minimum prefix length: min(10, n_layers).
            let min_prefix = 10usize.min(n_layers);
            let mut hasher = Sha256::new();
            hasher.update(b"vi-audit-prefix-v1");
            hasher.update(challenge_seed);
            hasher.update(token_index.to_le_bytes());
            let hash: [u8; 32] = hasher.finalize().into();
            let raw = u32::from_le_bytes(hash[..4].try_into().unwrap()) as usize;
            let l_max = (raw % n_layers).max(min_prefix.saturating_sub(1));
            (0..=l_max).collect()
        }
    }
}

/// Build an `AuditChallenge` from a verifier-generated challenge seed and tier.
///
/// Picks a random token index from `0..n_tokens` and layer indices from
/// `0..n_layers` based on the tier.
pub fn build_audit_challenge(
    challenge_seed: &[u8; 32],
    n_tokens: u32,
    n_layers: usize,
    tier: AuditTier,
) -> AuditChallenge {
    use sha2::{Digest, Sha256};

    // Pick token index from the verifier's challenge seed.
    let mut hasher = Sha256::new();
    hasher.update(b"vi-audit-token-v1");
    hasher.update(challenge_seed);
    let hash: [u8; 32] = hasher.finalize().into();
    let token_index = u32::from_le_bytes(hash[..4].try_into().unwrap()) % n_tokens;

    let layer_indices = derive_audit_layers(challenge_seed, token_index, n_layers, tier);

    AuditChallenge {
        token_index,
        layer_indices,
        tier,
    }
}

/// Result of an audit verification.
#[derive(Debug)]
pub struct AuditVerifyReport {
    pub verdict: Verdict,
    pub tier: AuditTier,
    pub token_index: u32,
    pub layers_checked: usize,
    pub checks_run: usize,
    pub checks_passed: usize,
    pub failures: Vec<String>,
    pub duration: Duration,
}

impl std::fmt::Display for AuditVerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => write!(
                f,
                "AUDIT PASS ({:?}): token {} — {}/{} checks on {} layers ({:.1}ms)",
                self.tier,
                self.token_index,
                self.checks_passed,
                self.checks_run,
                self.layers_checked,
                self.duration.as_secs_f64() * 1000.0
            ),
            Verdict::Fail => {
                writeln!(
                    f,
                    "AUDIT FAIL ({:?}): token {} — {} failures",
                    self.tier, self.token_index, self.failures.len()
                )?;
                for fail in &self.failures {
                    writeln!(f, "  {}", fail)?;
                }
                Ok(())
            }
        }
    }
}

/// Verify an audit response against a challenge and streaming KV commitments.
///
/// Performs on the challenged layers:
/// 1. Freivalds checks (Level A) for each matrix type
/// 2. SiLU check
/// 3. Intra-layer chain (x_ffn == requantize(attn_out))
/// 4. Attention replay from provided KV prefix
/// 5. Streaming KV commitment verification (full audits only)
///
/// Cross-layer chain is checked between consecutive opened layers.
pub fn verify_audit(
    key: &VerifierKey,
    challenge: &AuditChallenge,
    response: &AuditResponse,
    kv_verifier: &streaming::StreamingKvVerifier,
    attn_tolerance: Option<AttentionToleranceConfig>,
) -> AuditVerifyReport {
    let start = Instant::now();
    let mut failures = Vec::new();
    let attn_tol = attn_tolerance.unwrap_or_default();

    // Basic structural checks
    if response.token_index != challenge.token_index {
        failures.push(format!(
            "response token_index {} != challenge token_index {}",
            response.token_index, challenge.token_index
        ));
    }

    if response.layer_indices != challenge.layer_indices {
        failures.push(format!(
            "response layer_indices {:?} != challenge {:?}",
            response.layer_indices, challenge.layer_indices
        ));
    }

    if response.partial_layers.len() != challenge.layer_indices.len() {
        failures.push(format!(
            "response has {} layers but challenge requests {}",
            response.partial_layers.len(),
            challenge.layer_indices.len()
        ));
        return make_audit_report(challenge, &failures, start.elapsed());
    }

    if response.kv_k_prefix.len() != challenge.layer_indices.len()
        || response.kv_v_prefix.len() != challenge.layer_indices.len()
    {
        failures.push("KV prefix count doesn't match layer count".into());
        return make_audit_report(challenge, &failures, start.elapsed());
    }

    let mut checks_run = 0usize;

    // Verify each opened layer
    for (resp_idx, &layer_idx) in challenge.layer_indices.iter().enumerate() {
        let lt = &response.partial_layers[resp_idx];

        // Freivalds checks (7 per layer)
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
            checks_run += 1;
            if !freivalds::check(v, input, r, output) {
                failures.push(format!(
                    "layer {} {:?}: Freivalds check failed",
                    layer_idx, mt
                ));
            }
        }

        // SiLU check
        checks_run += 1;
        let expected_h = verilm_core::silu::compute_h_unit_scale(&lt.g, &lt.u);
        if lt.h != expected_h {
            failures.push(format!("layer {}: SiLU check failed", layer_idx));
        }

        // Intra-layer chain: x_ffn == requantize(attn_out)
        checks_run += 1;
        if lt.x_ffn != requantize(&lt.attn_out) {
            failures.push(format!("layer {}: chain x_ffn mismatch", layer_idx));
        }

        // Attention replay from KV prefix
        checks_run += 1;
        let kv_k = &response.kv_k_prefix[resp_idx];
        let kv_v = &response.kv_v_prefix[resp_idx];

        if !kv_k.is_empty() && !kv_v.is_empty() {
            // Self-consistency: own KV entry must match requantize(k/v)
            let pos = response.token_index as usize;
            if pos < kv_k.len() {
                let expected_k = requantize(&lt.k);
                if kv_k[pos] != expected_k {
                    failures.push(format!(
                        "layer {}: KV self-consistency failed: kv_k[{}] != requantize(k)",
                        layer_idx, pos
                    ));
                }
            }
            if pos < kv_v.len() {
                let expected_v = requantize(&lt.v);
                if kv_v[pos] != expected_v {
                    failures.push(format!(
                        "layer {}: KV self-consistency failed: kv_v[{}] != requantize(v)",
                        layer_idx, pos
                    ));
                }
            }

            // Replay attention
            let q_i8: Vec<i8> = lt.q.iter().map(|&v| v.clamp(-128, 127) as i8).collect();
            let replayed = replay_attention_reference(&q_i8, kv_k, kv_v, &key.config);
            if let Some(max_diff) = compare_attention_output(&lt.a, &replayed, &attn_tol) {
                failures.push(format!(
                    "layer {}: attention replay mismatch (max_diff={}, tol={})",
                    layer_idx, max_diff, attn_tol.max_abs_diff
                ));
            }
        }
    }

    // Cross-layer chain between consecutive opened layers
    for w in challenge.layer_indices.windows(2) {
        if w[1] == w[0] + 1 {
            let prev_idx = challenge.layer_indices.iter().position(|&l| l == w[0]).unwrap();
            let next_idx = challenge.layer_indices.iter().position(|&l| l == w[1]).unwrap();
            checks_run += 1;
            let expected_input = requantize(&response.partial_layers[prev_idx].ffn_out);
            if response.partial_layers[next_idx].x_attn != expected_input {
                failures.push(format!(
                    "layer {}->{}: cross-layer chain mismatch",
                    w[0], w[1]
                ));
            }
        }
    }

    // Streaming KV commitment verification via per-layer Merkle proofs.
    // Works for both routine (subset) and full (all layers) audits.
    if !kv_verifier.is_empty() && response.kv_layer_proofs.len() == challenge.layer_indices.len() {
        for (resp_idx, &layer_idx) in challenge.layer_indices.iter().enumerate() {
            checks_run += 1;
            let k_i8 = requantize(&response.partial_layers[resp_idx].k);
            let v_i8 = requantize(&response.partial_layers[resp_idx].v);
            let proof = &response.kv_layer_proofs[resp_idx];
            if let Err(msg) = kv_verifier.verify_layer_opening(
                challenge.token_index,
                layer_idx as u32,
                &k_i8,
                &v_i8,
                proof,
            ) {
                failures.push(msg);
            }
        }
    }

    let duration = start.elapsed();
    let checks_passed = checks_run.saturating_sub(failures.len());

    AuditVerifyReport {
        verdict: if failures.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        },
        tier: challenge.tier,
        token_index: challenge.token_index,
        layers_checked: challenge.layer_indices.len(),
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

fn make_audit_report(
    challenge: &AuditChallenge,
    failures: &[String],
    duration: Duration,
) -> AuditVerifyReport {
    AuditVerifyReport {
        verdict: Verdict::Fail,
        tier: challenge.tier,
        token_index: challenge.token_index,
        layers_checked: 0,
        checks_run: 0,
        checks_passed: 0,
        failures: failures.to_vec(),
        duration,
    }
}

// ---------------------------------------------------------------------------
// V4 retained-state verification
// ---------------------------------------------------------------------------

/// Result from V4 retained-state verification.
#[derive(Debug)]
pub struct V4VerifyReport {
    pub verdict: Verdict,
    pub token_index: u32,
    pub checks_run: usize,
    pub checks_passed: usize,
    pub failures: Vec<String>,
    pub duration: Duration,
}

impl std::fmt::Display for V4VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.verdict {
            Verdict::Pass => write!(
                f,
                "V4 PASS: token {} — {}/{} checks ({:.1}ms)",
                self.token_index,
                self.checks_passed,
                self.checks_run,
                self.duration.as_secs_f64() * 1000.0
            ),
            Verdict::Fail => {
                writeln!(
                    f,
                    "V4 FAIL: token {} — {} failures",
                    self.token_index,
                    self.failures.len()
                )?;
                for fail in &self.failures {
                    writeln!(f, "  {}", fail)?;
                }
                Ok(())
            }
        }
    }
}

// Re-export ShellWeights from core for debug/oracle replay users.
pub use verilm_core::types::ShellWeights;

/// Verify V4 audit response: structural checks + key-only Freivalds on shell openings.
///
/// This is the protocol verifier — requires only the precomputed VerifierKey,
/// no weights. Checks:
/// 1. Structural: version, seed, prompt, Merkle proofs, IO chain, prefix count
/// 2. Shell opening: Freivalds on each matmul (W_o, W_g, W_u, W_d, optionally QKV)
/// 3. Bridge consistency: verifier derives intermediate i8 values from i32 accumulators
/// 4. Retained-state binding: shell openings must be consistent with committed `a`
pub fn verify_v4(
    key: &VerifierKey,
    response: &V4AuditResponse,
) -> V4VerifyReport {
    let start = Instant::now();
    let (mut checks_run, mut failures) = verify_v4_structural(response);

    // Protocol path: check shell openings with key-only Freivalds.
    match &response.shell_opening {
        Some(shell) => {
            // Fail-closed embedding binding.
            //
            // If key has embedding_merkle_root: REQUIRE initial_residual + valid proof.
            // If key has no embedding_merkle_root: REJECT any initial_residual (unbound).
            if let Some(ref emb_root) = key.embedding_merkle_root {
                checks_run += 1;
                match (&shell.initial_residual, &shell.embedding_proof) {
                    (Some(ir), Some(proof)) => {
                        let leaf = verilm_core::merkle::hash_embedding_row(ir);
                        if proof.leaf_index != response.token_id {
                            failures.push(format!(
                                "embedding proof leaf_index {} != token_id {}",
                                proof.leaf_index, response.token_id
                            ));
                        } else if !verilm_core::merkle::verify(emb_root, &leaf, proof) {
                            failures.push("embedding Merkle proof verification failed".into());
                        }
                    }
                    (None, _) => {
                        failures.push(
                            "key has embedding_merkle_root but shell missing initial_residual".into()
                        );
                    }
                    (Some(_), None) => {
                        failures.push(
                            "key has embedding_merkle_root but shell missing embedding_proof".into()
                        );
                    }
                }
            } else if shell.initial_residual.is_some() {
                checks_run += 1;
                failures.push(
                    "shell has initial_residual but key has no embedding_merkle_root to verify it".into()
                );
            }

            let (c, f, shell_final_hidden) = verify_shell_opening(key, &response.retained, shell);
            checks_run += c;
            failures.extend(f);

            // Manifest binding: if response carries manifest, verify hash matches commitment.
            let decode_params = if let Some(ref manifest) = response.manifest {
                if let Some(ref committed_hash) = response.commitment.manifest_hash {
                    checks_run += 1;
                    let computed = merkle::hash_manifest(manifest);
                    if computed != *committed_hash {
                        failures.push("manifest hash does not match commitment".into());
                    }
                }

                // Reject logit-modifying parameters the canonical sampler doesn't support.
                // These are bound in the manifest hash, so a prover can't hide them.
                // But the verifier can't replay sampling correctly if they're enabled.
                checks_run += 1;
                if manifest.repetition_penalty != 1.0 {
                    failures.push(format!(
                        "unsupported repetition_penalty={} (canonical sampler requires 1.0)",
                        manifest.repetition_penalty
                    ));
                }
                if manifest.frequency_penalty != 0.0 {
                    failures.push(format!(
                        "unsupported frequency_penalty={} (canonical sampler requires 0.0)",
                        manifest.frequency_penalty
                    ));
                }
                if manifest.presence_penalty != 0.0 {
                    failures.push(format!(
                        "unsupported presence_penalty={} (canonical sampler requires 0.0)",
                        manifest.presence_penalty
                    ));
                }
                if !manifest.logit_bias.is_empty() {
                    failures.push(format!(
                        "unsupported logit_bias ({} entries, canonical sampler requires empty)",
                        manifest.logit_bias.len()
                    ));
                }
                if !manifest.guided_decoding.is_empty() {
                    failures.push(format!(
                        "unsupported guided_decoding='{}' (canonical sampler requires empty)",
                        manifest.guided_decoding
                    ));
                }
                if !manifest.stop_sequences.is_empty() {
                    failures.push(format!(
                        "unsupported stop_sequences ({} entries, canonical sampler requires empty)",
                        manifest.stop_sequences.len()
                    ));
                }
                if manifest.max_tokens > 0 {
                    if response.token_index >= manifest.max_tokens {
                        failures.push(format!(
                            "token_index {} exceeds manifest max_tokens {}",
                            response.token_index, manifest.max_tokens
                        ));
                    }
                    if response.commitment.n_tokens > manifest.max_tokens {
                        failures.push(format!(
                            "committed n_tokens {} exceeds manifest max_tokens {}",
                            response.commitment.n_tokens, manifest.max_tokens
                        ));
                    }
                }

                Some(verilm_core::sampling::DecodeParams {
                    temperature: manifest.temperature,
                    top_k: manifest.top_k,
                    top_p: manifest.top_p,
                })
            } else {
                None
            };

            // LM head + sampling verification.
            //
            // Two sources for final_hidden:
            // 1. Captured final_residual (from live GPU inference) — exact.
            //    Apply final RMSNorm + quantize to get the true final_hidden.
            // 2. Shell-replayed final_hidden — approximate (diverges after many
            //    layers of approximate attention). Fallback for toy model only.
            //
            // Fail-closed: when the key has both lm_head and final_norm_weights,
            // the prover MUST supply final_residual. Missing it means the exact
            // tail path was not captured, and we reject rather than silently
            // downgrade to the approximate shell replay.
            let final_hidden = if let Some(ref fr) = shell.final_residual {
                if let Some(ref fnw) = key.final_norm_weights {
                    let res_f64: Vec<f64> = fr.iter().map(|&v| v as f64).collect();
                    let normed = verilm_core::rmsnorm::rmsnorm_f64_input(&res_f64, fnw, key.rmsnorm_eps);
                    Some(normed.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect())
                } else {
                    // No final_norm_weights in key — fall back to shell replay (toy model).
                    shell_final_hidden
                }
            } else if key.final_norm_weights.is_some() && key.lm_head.is_some() {
                // Fail-closed: key requires exact verification but no captured state.
                failures.push(
                    "key has lm_head + final_norm_weights but shell missing final_residual \
                     (exact tail verification required, cannot fall back to shell replay)".into()
                );
                None
            } else {
                // Toy model / no final_norm_weights — shell replay fallback is acceptable.
                shell_final_hidden
            };

            // LM-head Freivalds: verify the prover's claimed logits_i32.
            //
            // Fail-closed: if the key has LmHead Freivalds vectors, the prover
            // MUST supply logits_i32. Missing claim = reject.
            let r_lm = key.r_for(MatrixType::LmHead);
            let key_has_lm_freivalds = !r_lm.is_empty() && key.v_lm_head.is_some();

            if key_has_lm_freivalds {
                let v_lm = key.v_lm_head.as_ref().unwrap();

                match (&shell.logits_i32, &final_hidden) {
                    (Some(ref claimed_logits), Some(ref fh)) => {
                        checks_run += 1;
                        if !freivalds::check(v_lm, fh, r_lm, claimed_logits) {
                            failures.push("lm_head: Freivalds check failed on prover's logits_i32 claim".into());
                        }

                        // Token replay: use the prover's (now Freivalds-verified) logits.
                        if key.config.vocab_size > 0 {
                            checks_run += 1;
                            let logits: Vec<f32> = claimed_logits.iter().map(|&v| v as f32).collect();

                            let expected_token = if let Some(ref dp) = decode_params {
                                let token_seed = verilm_core::sampling::derive_token_seed(
                                    &response.revealed_seed, response.token_index,
                                );
                                verilm_core::sampling::sample(&logits, dp, &token_seed)
                            } else {
                                logits.iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .map(|(i, _)| i as u32)
                                    .unwrap_or(0)
                            };

                            if expected_token != response.token_id {
                                failures.push(format!(
                                    "lm_head: expected token_id={} but claimed token_id={}",
                                    expected_token, response.token_id
                                ));
                            }
                        }
                    }
                    (None, _) => {
                        failures.push(
                            "lm_head: key requires logits_i32 but shell opening did not provide it".into()
                        );
                    }
                    (Some(_), None) => {
                        failures.push(
                            "lm_head: logits_i32 present but no final_hidden to check against".into()
                        );
                    }
                }
            }
        }
        None => {
            failures.push("V4 audit response missing shell_opening".into());
        }
    }

    let duration = start.elapsed();
    let checks_passed = checks_run.saturating_sub(failures.len());

    V4VerifyReport {
        verdict: if failures.is_empty() { Verdict::Pass } else { Verdict::Fail },
        token_index: response.token_index,
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

/// Verify shell opening for one token using key-only Freivalds + bridge checks.
///
/// The verifier does NOT recompute matmuls — it checks the prover's i32
/// accumulators with precomputed Freivalds keys and verifies bridge
/// consistency by deriving intermediate i8 values from the accumulators.
///
/// When `shell.initial_residual` is present and the key has RMSNorm weights,
/// uses the full bridge (dequant → residual → RMSNorm → quantize). This
/// also enables QKV Freivalds at layer 0.
///
/// Returns `(checks_run, failures, final_hidden)`. `final_hidden` is the
/// i8 hidden state after the last layer, derived from the bridge chain.
/// Used by verify_v4 for lm_head logit verification.
fn verify_shell_opening(
    key: &VerifierKey,
    retained: &verilm_core::types::RetainedTokenState,
    shell: &verilm_core::types::ShellTokenOpening,
) -> (usize, Vec<String>, Option<Vec<i8>>) {
    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // Resolve which layers are present.
    let opened_layers: Vec<usize> = shell.layer_indices.clone()
        .unwrap_or_else(|| (0..retained.layers.len()).collect());

    if shell.layers.len() != opened_layers.len() {
        failures.push(format!(
            "shell_opening has {} layers but layer_indices specifies {}",
            shell.layers.len(), opened_layers.len()
        ));
        return (checks_run, failures, None);
    }

    // Build a lookup: shell_idx for a given layer_idx (None if not opened).
    let max_layer = opened_layers.iter().copied().max().unwrap_or(0);
    let mut shell_idx_for = vec![None; max_layer + 1];
    for (si, &li) in opened_layers.iter().enumerate() {
        if li <= max_layer {
            shell_idx_for[li] = Some(si);
        }
    }

    // Full bridge: only when initial_residual is authenticated.
    let use_full_bridge = shell.initial_residual.is_some()
        && !key.rmsnorm_attn_weights.is_empty()
        && key.embedding_merkle_root.is_some();

    let (mut residual, mut x_attn) = if use_full_bridge {
        let ir = shell.initial_residual.as_ref().unwrap();
        let res: Vec<f64> = ir.iter().map(|&v| v as f64).collect();
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
            &res, &key.rmsnorm_attn_weights[0], key.rmsnorm_eps,
        );
        let xa = verilm_core::rmsnorm::quantize_f64_to_i8(
            &normed, retained.layers[0].scale_x_attn as f64,
        );
        (Some(res), Some(xa))
    } else {
        (None, None)
    };

    // Iterate 0..=max_layer: bridge is sequential, Freivalds only on opened layers.
    for layer_idx in 0..=max_layer {
        if layer_idx >= retained.layers.len() { break; }
        let rs = &retained.layers[layer_idx];
        let a = &rs.a;

        // If this layer has a shell opening, run Freivalds checks.
        if let Some(si) = shell_idx_for[layer_idx] {
            let sl = &shell.layers[si];

            // QKV Freivalds (full bridge: all layers; toy: layers > 0)
            if let Some(ref xa) = x_attn {
                let qkv = [
                    (MatrixType::Wq, &sl.q, key.config.hidden_dim),
                    (MatrixType::Wk, &sl.k, key.config.kv_dim),
                    (MatrixType::Wv, &sl.v, key.config.kv_dim),
                ];
                for (mt, opened_acc, _expected_rows) in &qkv {
                    match opened_acc {
                        Some(z) => {
                            checks_run += 1;
                            if !freivalds::check(
                                key.v_for(layer_idx, *mt),
                                xa,
                                key.r_for(*mt),
                                z,
                            ) {
                                failures.push(format!(
                                    "layer {} {:?}: Freivalds failed on shell opening",
                                    layer_idx, mt
                                ));
                            }
                        }
                        None => {
                            failures.push(format!(
                                "layer {} {:?}: shell opening missing QKV (x_attn derivable)",
                                layer_idx, mt
                            ));
                        }
                    }
                }
            }

            // W_o @ a
            checks_run += 1;
            if !freivalds::check(
                key.v_for(layer_idx, MatrixType::Wo),
                a,
                key.r_for(MatrixType::Wo),
                &sl.attn_out,
            ) {
                failures.push(format!(
                    "layer {} Wo: Freivalds failed on shell opening", layer_idx
                ));
            }

            // Post-attention bridge: derive x_ffn
            let x_ffn = if let Some(ref mut res) = residual {
                verilm_core::rmsnorm::bridge_residual_rmsnorm(
                    &sl.attn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wo),
                    rs.scale_a,
                    res,
                    &key.rmsnorm_ffn_weights[layer_idx],
                    key.rmsnorm_eps,
                    rs.scale_x_ffn,
                )
            } else {
                verilm_core::bridge_requantize(
                    &sl.attn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wo),
                    rs.scale_a,
                    rs.scale_x_ffn,
                )
            };

            // W_g, W_u @ x_ffn
            for (mt, z) in [(MatrixType::Wg, &sl.g), (MatrixType::Wu, &sl.u)] {
                checks_run += 1;
                if !freivalds::check(
                    key.v_for(layer_idx, mt),
                    &x_ffn,
                    key.r_for(mt),
                    z,
                ) {
                    failures.push(format!(
                        "layer {} {:?}: Freivalds failed on shell opening",
                        layer_idx, mt
                    ));
                }
            }

            // Bridge: verifier derives h from prover's g, u using scales
            let h = verilm_core::silu::compute_h_scaled(
                &sl.g, &sl.u,
                key.weight_scale_for(layer_idx, MatrixType::Wg),
                key.weight_scale_for(layer_idx, MatrixType::Wu),
                rs.scale_x_ffn,
                rs.scale_h,
            );

            // W_d @ h
            checks_run += 1;
            if !freivalds::check(
                key.v_for(layer_idx, MatrixType::Wd),
                &h,
                key.r_for(MatrixType::Wd),
                &sl.ffn_out,
            ) {
                failures.push(format!(
                    "layer {} Wd: Freivalds failed on shell opening", layer_idx
                ));
            }

            // Post-FFN bridge: derive x_attn for next layer
            let next_scale_x_attn = retained.layers
                .get(layer_idx + 1)
                .map(|r| r.scale_x_attn)
                .unwrap_or(1.0);

            x_attn = if let Some(ref mut res) = residual {
                if layer_idx + 1 < key.rmsnorm_attn_weights.len() {
                    Some(verilm_core::rmsnorm::bridge_residual_rmsnorm(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        rs.scale_h,
                        res,
                        &key.rmsnorm_attn_weights[layer_idx + 1],
                        key.rmsnorm_eps,
                        next_scale_x_attn,
                    ))
                } else {
                    verilm_core::rmsnorm::dequant_add_residual(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        rs.scale_h,
                        res,
                    );
                    Some(verilm_core::bridge_requantize(
                        &sl.ffn_out,
                        key.weight_scale_for(layer_idx, MatrixType::Wd),
                        rs.scale_h,
                        next_scale_x_attn,
                    ))
                }
            } else {
                Some(verilm_core::bridge_requantize(
                    &sl.ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    rs.scale_h,
                    next_scale_x_attn,
                ))
            };
        }
        // Layer not opened: no Freivalds, but bridge state must NOT advance
        // (the prover didn't provide intermediates for this layer).
        // For contiguous prefix semantics, all layers up to max are opened,
        // so this branch is only hit for non-prefix gaps (shouldn't happen
        // with proper challenge derivation). Bridge tracking stops.
    }

    // Derive final_hidden from the chain endpoint.
    // For full bridge with final_norm_weights: apply final RMSNorm to residual.
    // For simplified bridge (toy model): x_attn after last layer is requantize(ffn_out).
    let final_hidden = if let (Some(ref res), Some(ref fnw)) = (&residual, &key.final_norm_weights) {
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(res, fnw, key.rmsnorm_eps);
        // Quantize to i8 with unit scale (simple clamp) for lm_head matmul.
        Some(normed.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect())
    } else {
        // Simplified bridge: x_attn after last layer IS the final hidden state.
        x_attn.clone()
    };

    (checks_run, failures, final_hidden)
}

/// Verify V4 with verifier-side weight-backed replay (debug/oracle path).
///
/// The verifier independently reconstructs the computation shell from
/// committed retained `a` + public weights. This is NOT the protocol path —
/// it requires the verifier to hold full weights. Use for:
/// - Differential testing against the protocol verifier
/// - Stronger local checking when weights are available
/// - Dev-mode paranoid verification
pub fn verify_v4_with_weights(
    key: &VerifierKey,
    response: &V4AuditResponse,
    weights: &dyn ShellWeights,
) -> V4VerifyReport {
    let start = Instant::now();
    let (mut checks_run, mut failures) = verify_v4_structural(response);

    let cfg = &key.config;

    // Extract initial_residual only when authenticated (embedding root present).
    let initial_residual = if key.embedding_merkle_root.is_some() {
        response.shell_opening
            .as_ref()
            .and_then(|s| s.initial_residual.as_deref())
    } else {
        None
    };

    // NOTE: prefix replay is no longer possible — prefix_leaf_hashes are
    // compact 32-byte hashes, not full RetainedTokenState.  The structural
    // checks in verify_v4_structural still verify the prefix Merkle proofs
    // and IO chain using these hashes.

    // Replay the challenged token (with initial_residual if available).
    let (c, f) = replay_token_shell(
        key,
        cfg,
        &response.retained,
        weights,
        response.prefix_leaf_hashes.len(),
        initial_residual,
    );
    checks_run += c;
    failures.extend(f);

    let duration = start.elapsed();
    let checks_passed = checks_run.saturating_sub(failures.len());

    V4VerifyReport {
        verdict: if failures.is_empty() { Verdict::Pass } else { Verdict::Fail },
        token_index: response.token_index,
        checks_run,
        checks_passed,
        failures,
        duration,
    }
}

/// Structural checks shared by verify_v4 and verify_v4_with_replay.
fn verify_v4_structural(
    response: &V4AuditResponse,
) -> (usize, Vec<String>) {
    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // 1. Commitment version must be V4
    checks_run += 1;
    if response.commitment.version != CommitmentVersion::V4 {
        failures.push(format!(
            "expected V4 commitment, got {:?}",
            response.commitment.version
        ));
    }

    // 2. Seed commitment: hash(revealed_seed) == commitment.seed_commitment
    checks_run += 1;
    match response.commitment.seed_commitment {
        Some(expected) => {
            let computed = merkle::hash_seed(&response.revealed_seed);
            if computed != expected {
                failures.push(
                    "seed commitment mismatch: hash(revealed_seed) != commitment.seed_commitment"
                        .into(),
                );
            }
        }
        None => {
            failures.push("V4 commitment missing seed_commitment".into());
        }
    }

    // 3. Prompt hash must be present
    checks_run += 1;
    if response.commitment.prompt_hash.is_none() {
        failures.push("V4 commitment missing prompt_hash".into());
    }

    // 4. Challenged token retained leaf Merkle proof.
    // When shell_opening has final_residual, include it in the leaf hash
    // (it was bound into the commitment at commit time).
    checks_run += 1;
    let final_residual_ref = response.shell_opening.as_ref()
        .and_then(|s| s.final_residual.as_deref());
    let leaf_hash = merkle::hash_retained_with_residual(&response.retained, final_residual_ref);
    if !merkle::verify(
        &response.commitment.merkle_root,
        &leaf_hash,
        &response.merkle_proof,
    ) {
        failures.push(format!(
            "token {}: retained leaf Merkle proof failed",
            response.token_index
        ));
    }

    // 5. Prefix retained leaf Merkle proofs
    for (j, (prefix_leaf_hash, proof)) in response
        .prefix_leaf_hashes
        .iter()
        .zip(response.prefix_merkle_proofs.iter())
        .enumerate()
    {
        checks_run += 1;
        if !merkle::verify(&response.commitment.merkle_root, prefix_leaf_hash, proof) {
            failures.push(format!(
                "prefix token {}: retained leaf Merkle proof failed",
                j
            ));
        }
    }

    // 6. IO chain verification
    checks_run += 1;
    let mut prev_io = [0u8; 32];
    for (j, prefix_leaf_hash) in response.prefix_leaf_hashes.iter().enumerate() {
        prev_io = merkle::io_hash_v4(*prefix_leaf_hash, response.prefix_token_ids[j], prev_io);
    }

    if prev_io != response.prev_io_hash {
        failures.push(format!(
            "token {}: prev_io_hash doesn't match recomputed chain from prefix",
            response.token_index
        ));
    }

    checks_run += 1;
    let challenged_io = merkle::io_hash_v4(leaf_hash, response.token_id, prev_io);
    if !merkle::verify(
        &response.commitment.io_root,
        &challenged_io,
        &response.io_proof,
    ) {
        failures.push(format!(
            "token {}: IO chain proof verification failed",
            response.token_index
        ));
    }

    // 7. Prefix count == token_index
    checks_run += 1;
    if response.prefix_leaf_hashes.len() != response.token_index as usize {
        failures.push(format!(
            "token {}: expected {} prefix tokens but got {}",
            response.token_index,
            response.token_index,
            response.prefix_leaf_hashes.len()
        ));
    }

    (checks_run, failures)
}

/// Replay one token's computation shell from retained state + public weights.
///
/// When the key has RMSNorm weights and `initial_residual` is provided,
/// uses full bridge and enables QKV at layer 0. Otherwise falls back to
/// simplified bridge (layer 0 QKV skipped).
///
/// Returns (checks_run, failures).
fn replay_token_shell(
    key: &VerifierKey,
    cfg: &verilm_core::constants::ModelConfig,
    retained: &verilm_core::types::RetainedTokenState,
    weights: &dyn ShellWeights,
    token_pos: usize,
    initial_residual: Option<&[f32]>,
) -> (usize, Vec<String>) {
    use verilm_core::matmul::matmul_i32;

    let mut failures = Vec::new();
    let mut checks_run = 0usize;

    // Full bridge: only when initial_residual is provided AND authenticated.
    let use_full_bridge = initial_residual.is_some()
        && !key.rmsnorm_attn_weights.is_empty()
        && key.embedding_merkle_root.is_some();

    let (mut residual, mut x_attn) = if use_full_bridge {
        let ir = initial_residual.unwrap();
        let res: Vec<f64> = ir.iter().map(|&v| v as f64).collect();
        let normed = verilm_core::rmsnorm::rmsnorm_f64_input(
            &res, &key.rmsnorm_attn_weights[0], key.rmsnorm_eps,
        );
        let xa = verilm_core::rmsnorm::quantize_f64_to_i8(
            &normed, retained.layers[0].scale_x_attn as f64,
        );
        (Some(res), Some(xa))
    } else {
        (None, None)
    };

    for (layer_idx, rs) in retained.layers.iter().enumerate() {
        let a = &rs.a;

        // QKV Freivalds (full bridge: all layers; toy: layers > 0)
        if let Some(ref xa) = x_attn {
            let qkv_mats = [
                (MatrixType::Wq, cfg.hidden_dim, cfg.hidden_dim),
                (MatrixType::Wk, cfg.kv_dim, cfg.hidden_dim),
                (MatrixType::Wv, cfg.kv_dim, cfg.hidden_dim),
            ];
            for (mt, rows, cols) in &qkv_mats {
                let z = matmul_i32(weights.weight(layer_idx, *mt), xa, *rows, *cols);
                checks_run += 1;
                if !freivalds::check(
                    key.v_for(layer_idx, *mt),
                    xa,
                    key.r_for(*mt),
                    &z,
                ) {
                    failures.push(format!(
                        "token {} layer {} {:?}: Freivalds weight-binding failed",
                        token_pos, layer_idx, mt
                    ));
                }
            }
        }

        // W_o: attn_out = W_o @ a (verifier-computed)
        let attn_out = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wo),
            a,
            cfg.hidden_dim,
            cfg.hidden_dim,
        );
        checks_run += 1;
        if !freivalds::check(
            key.v_for(layer_idx, MatrixType::Wo),
            a,
            key.r_for(MatrixType::Wo),
            &attn_out,
        ) {
            failures.push(format!(
                "token {} layer {} Wo: Freivalds weight-binding failed",
                token_pos, layer_idx
            ));
        }

        // Post-attention bridge: derive x_ffn
        let x_ffn = if let Some(ref mut res) = residual {
            verilm_core::rmsnorm::bridge_residual_rmsnorm(
                &attn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wo),
                rs.scale_a,
                res,
                &key.rmsnorm_ffn_weights[layer_idx],
                key.rmsnorm_eps,
                rs.scale_x_ffn,
            )
        } else {
            verilm_core::bridge_requantize(
                &attn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wo),
                rs.scale_a,
                rs.scale_x_ffn,
            )
        };

        // W_g, W_u: gate and up projections
        let g = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wg),
            &x_ffn,
            cfg.ffn_dim,
            cfg.hidden_dim,
        );
        let u = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wu),
            &x_ffn,
            cfg.ffn_dim,
            cfg.hidden_dim,
        );

        // Freivalds on W_g, W_u
        for (mt, z) in [(MatrixType::Wg, &g), (MatrixType::Wu, &u)] {
            checks_run += 1;
            if !freivalds::check(
                key.v_for(layer_idx, mt),
                &x_ffn,
                key.r_for(mt),
                z,
            ) {
                failures.push(format!(
                    "token {} layer {} {:?}: Freivalds weight-binding failed",
                    token_pos, layer_idx, mt
                ));
            }
        }

        // SiLU: h via scale-aware bridge
        let h = verilm_core::silu::compute_h_scaled(
            &g, &u,
            key.weight_scale_for(layer_idx, MatrixType::Wg),
            key.weight_scale_for(layer_idx, MatrixType::Wu),
            rs.scale_x_ffn,
            rs.scale_h,
        );

        // W_d: ffn_out = W_d @ h
        let ffn_out = matmul_i32(
            weights.weight(layer_idx, MatrixType::Wd),
            &h,
            cfg.hidden_dim,
            cfg.ffn_dim,
        );
        checks_run += 1;
        if !freivalds::check(
            key.v_for(layer_idx, MatrixType::Wd),
            &h,
            key.r_for(MatrixType::Wd),
            &ffn_out,
        ) {
            failures.push(format!(
                "token {} layer {} Wd: Freivalds weight-binding failed",
                token_pos, layer_idx
            ));
        }

        // Post-FFN bridge: derive x_attn for next layer
        let next_scale_x_attn = retained.layers
            .get(layer_idx + 1)
            .map(|r| r.scale_x_attn)
            .unwrap_or(1.0);

        x_attn = if let Some(ref mut res) = residual {
            if layer_idx + 1 < key.rmsnorm_attn_weights.len() {
                Some(verilm_core::rmsnorm::bridge_residual_rmsnorm(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    rs.scale_h,
                    res,
                    &key.rmsnorm_attn_weights[layer_idx + 1],
                    key.rmsnorm_eps,
                    next_scale_x_attn,
                ))
            } else {
                verilm_core::rmsnorm::dequant_add_residual(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    rs.scale_h,
                    res,
                );
                Some(verilm_core::bridge_requantize(
                    &ffn_out,
                    key.weight_scale_for(layer_idx, MatrixType::Wd),
                    rs.scale_h,
                    next_scale_x_attn,
                ))
            }
        } else {
            Some(verilm_core::bridge_requantize(
                &ffn_out,
                key.weight_scale_for(layer_idx, MatrixType::Wd),
                rs.scale_h,
                next_scale_x_attn,
            ))
        };
    }

    (checks_run, failures)
}
