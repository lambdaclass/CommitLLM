//! Canonical deterministic sampler for the verified-inference protocol.
//!
//! Defines the exact sampling semantics that both prover and verifier must
//! agree on. Given logits, decode parameters, and a per-token seed, this
//! module produces a deterministic token ID.
//!
//! # Sampling pipeline
//!
//! 1. **Temperature scaling**: `logits[i] /= temperature` (temperature=0 → greedy argmax)
//! 2. **Top-k filtering**: Keep only the `top_k` highest logits, mask rest to −∞
//! 3. **Top-p (nucleus) filtering**: Sort by probability, keep cumulative ≤ `top_p`
//! 4. **Softmax**: Convert filtered logits to probabilities (f64 precision)
//! 5. **Seeded selection**: ChaCha20Rng seeded per-token, sample from distribution
//!
//! # Per-token seed derivation
//!
//! Each token gets a deterministic seed derived from the batch seed:
//! `SHA256("vi-sample-v1" || batch_seed || token_index_le32)`
//!
//! This ensures each token's sampling is independently reproducible.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

/// Decode parameters extracted from a deployment manifest.
#[derive(Debug, Clone)]
pub struct DecodeParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
}

/// Derive a per-token PRNG seed from the batch seed and token index.
///
/// `SHA256("vi-sample-v1" || batch_seed || token_index_le32)`
pub fn derive_token_seed(batch_seed: &[u8; 32], token_index: u32) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"vi-sample-v1");
    hasher.update(batch_seed);
    hasher.update(token_index.to_le_bytes());
    hasher.finalize().into()
}

/// Sample a token ID from logits using the canonical protocol sampler.
///
/// Returns the vocabulary index of the selected token.
///
/// # Greedy mode
/// When `temperature == 0.0`, returns `argmax(logits)` deterministically
/// without using the PRNG. Ties are broken by lowest index.
///
/// # Panics
/// Panics if `logits` is empty.
pub fn sample(logits: &[f32], params: &DecodeParams, token_seed: &[u8; 32]) -> u32 {
    assert!(!logits.is_empty(), "logits must not be empty");

    // Greedy: argmax (temperature = 0)
    if params.temperature == 0.0 {
        return argmax(logits);
    }

    // 1. Temperature scaling
    let scaled: Vec<f64> = logits.iter()
        .map(|&l| (l as f64) / (params.temperature as f64))
        .collect();

    // 2. Top-k filtering
    let after_top_k = apply_top_k(&scaled, params.top_k);

    // 3. Softmax
    let probs = softmax(&after_top_k);

    // 4. Top-p filtering
    let after_top_p = apply_top_p(&probs, params.top_p);

    // 5. Re-normalize after top-p
    let final_probs = renormalize(&after_top_p);

    // 6. Seeded random selection
    let mut rng = ChaCha20Rng::from_seed(*token_seed);
    categorical_sample(&final_probs, &mut rng)
}

/// Recompute logits from lm_head and final_hidden, then sample.
///
/// This is the full verifier-side replay: given the model's unembedding
/// matrix and the trace's final hidden state, recompute logits and
/// deterministically sample to verify the claimed token_id.
pub fn replay_sampling(
    lm_head: &[i8],
    final_hidden: &[i8],
    vocab_size: usize,
    hidden_dim: usize,
    params: &DecodeParams,
    batch_seed: &[u8; 32],
    token_index: u32,
) -> u32 {
    let logits = recompute_logits(lm_head, final_hidden, vocab_size, hidden_dim);
    let token_seed = derive_token_seed(batch_seed, token_index);
    sample(&logits, params, &token_seed)
}

/// Recompute logits: `logits[r] = sum_c(lm_head[r*hidden_dim + c] * final_hidden[c])`.
///
/// Matches the logit computation in `vi-verify::verify_margin`.
pub fn recompute_logits(
    lm_head: &[i8],
    final_hidden: &[i8],
    vocab_size: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(lm_head.len(), vocab_size * hidden_dim);
    assert_eq!(final_hidden.len(), hidden_dim);
    (0..vocab_size)
        .map(|r| {
            (0..hidden_dim)
                .map(|c| lm_head[r * hidden_dim + c] as i32 * final_hidden[c] as i32)
                .sum::<i32>() as f32
        })
        .collect()
}

// -- internal helpers --

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Apply top-k: keep only the k largest values, set rest to −∞.
/// If top_k == 0 or top_k >= logits.len(), no filtering.
fn apply_top_k(logits: &[f64], top_k: u32) -> Vec<f64> {
    let k = top_k as usize;
    if k == 0 || k >= logits.len() {
        return logits.to_vec();
    }

    // Find the k-th largest value
    let mut sorted: Vec<f64> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[k - 1];

    // Count how many values are >= threshold (handle ties)
    let count_above = logits.iter().filter(|&&v| v > threshold).count();
    let mut remaining_at_threshold = k - count_above;

    logits.iter().map(|&v| {
        if v > threshold {
            v
        } else if v == threshold && remaining_at_threshold > 0 {
            remaining_at_threshold -= 1;
            v
        } else {
            f64::NEG_INFINITY
        }
    }).collect()
}

/// Softmax with f64 precision. −∞ values become 0.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        // All masked — uniform over everything (shouldn't happen in practice)
        let n = logits.len() as f64;
        return vec![1.0 / n; logits.len()];
    }
    let exps: Vec<f64> = logits.iter().map(|&l| {
        if l == f64::NEG_INFINITY { 0.0 } else { (l - max).exp() }
    }).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Apply top-p (nucleus) filtering: zero out tokens beyond cumulative
/// probability threshold, scanning in descending probability order.
/// If top_p >= 1.0, no filtering.
fn apply_top_p(probs: &[f64], top_p: f32) -> Vec<f64> {
    if top_p >= 1.0 {
        return probs.to_vec();
    }
    let p = top_p as f64;

    // Sort indices by descending probability
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cumulative = 0.0;
    let mut keep = vec![false; probs.len()];

    for &idx in &indices {
        keep[idx] = true;
        cumulative += probs[idx];
        if cumulative >= p {
            break;
        }
    }

    probs.iter().enumerate().map(|(i, &prob)| {
        if keep[i] { prob } else { 0.0 }
    }).collect()
}

/// Re-normalize probabilities (after top-p zeroed some out).
fn renormalize(probs: &[f64]) -> Vec<f64> {
    let sum: f64 = probs.iter().sum();
    if sum == 0.0 || sum == 1.0 {
        return probs.to_vec();
    }
    probs.iter().map(|&p| p / sum).collect()
}

/// Sample from a categorical distribution using the given RNG.
fn categorical_sample(probs: &[f64], rng: &mut ChaCha20Rng) -> u32 {
    let u: f64 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i as u32;
        }
    }
    // Floating-point edge case: return last non-zero index
    for (i, &p) in probs.iter().enumerate().rev() {
        if p > 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_returns_argmax() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
        let seed = [0u8; 32];
        assert_eq!(sample(&logits, &params, &seed), 1);
    }

    #[test]
    fn test_greedy_tie_breaks_by_lowest_index() {
        let logits = vec![5.0, 3.0, 5.0, 2.0];
        let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
        let seed = [0u8; 32];
        assert_eq!(sample(&logits, &params, &seed), 0);
    }

    #[test]
    fn test_deterministic_same_seed() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let params = DecodeParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
        let seed = [42u8; 32];
        let a = sample(&logits, &params, &seed);
        let b = sample(&logits, &params, &seed);
        assert_eq!(a, b, "same seed must produce same result");
    }

    #[test]
    fn test_different_seeds_can_differ() {
        let logits = vec![1.0; 100]; // uniform — different seeds should eventually differ
        let params = DecodeParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
        let mut results = std::collections::HashSet::new();
        for i in 0..20u32 {
            let seed = derive_token_seed(&[i as u8; 32], 0);
            results.insert(sample(&logits, &params, &seed));
        }
        assert!(results.len() > 1, "different seeds should produce different results on uniform");
    }

    #[test]
    fn test_top_k_filters() {
        // With top_k=1, should always pick the max
        let logits = vec![1.0, 10.0, 5.0, 3.0];
        let params = DecodeParams { temperature: 1.0, top_k: 1, top_p: 1.0 };
        let seed = [99u8; 32];
        assert_eq!(sample(&logits, &params, &seed), 1);
    }

    #[test]
    fn test_top_p_filters() {
        // One token has 99% of the probability mass; top_p=0.5 should select it
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let params = DecodeParams { temperature: 1.0, top_k: 0, top_p: 0.5 };
        let seed = [77u8; 32];
        assert_eq!(sample(&logits, &params, &seed), 0);
    }

    #[test]
    fn test_derive_token_seed_deterministic() {
        let batch = [1u8; 32];
        let s1 = derive_token_seed(&batch, 5);
        let s2 = derive_token_seed(&batch, 5);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_derive_token_seed_varies_by_index() {
        let batch = [1u8; 32];
        let s0 = derive_token_seed(&batch, 0);
        let s1 = derive_token_seed(&batch, 1);
        assert_ne!(s0, s1);
    }

    #[test]
    fn test_derive_token_seed_varies_by_batch() {
        let s1 = derive_token_seed(&[1u8; 32], 0);
        let s2 = derive_token_seed(&[2u8; 32], 0);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_recompute_logits_matches() {
        // 2x3 lm_head, 3-dim hidden
        let lm_head: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let hidden: Vec<i8> = vec![1, 1, 1];
        let logits = recompute_logits(&lm_head, &hidden, 2, 3);
        assert_eq!(logits, vec![6.0, 15.0]); // 1+2+3=6, 4+5+6=15
    }

    #[test]
    fn test_softmax_basic() {
        let probs = softmax(&[0.0, 0.0]);
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_masked() {
        let probs = softmax(&[1.0, f64::NEG_INFINITY, 1.0]);
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert_eq!(probs[1], 0.0);
        assert!((probs[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_temperature_concentrates() {
        // Low temperature should concentrate on the max
        let logits = vec![1.0, 2.0, 10.0, 3.0];
        let params = DecodeParams { temperature: 0.01, top_k: 0, top_p: 1.0 };
        // With very low temp, should almost always pick index 2
        for i in 0..10u32 {
            let seed = derive_token_seed(&[i as u8; 32], 0);
            assert_eq!(sample(&logits, &params, &seed), 2,
                "low temperature should concentrate on max");
        }
    }

    #[test]
    fn test_replay_sampling_e2e() {
        let lm_head: Vec<i8> = vec![
            1, 0, 0,  // token 0: logit = 1
            0, 0, 0,  // token 1: logit = 0
            0, 0, 10, // token 2: logit = 10
            0, 1, 0,  // token 3: logit = 1
        ];
        let hidden: Vec<i8> = vec![1, 1, 1];
        let params = DecodeParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
        let batch_seed = [42u8; 32];
        let result = replay_sampling(&lm_head, &hidden, 4, 3, &params, &batch_seed, 0);
        assert_eq!(result, 2, "greedy should pick token with highest logit");
    }
}
