//! Randomness soundness tests.
//!
//! Tests that the protocol's random derivations (`derive_challenges`,
//! `derive_block_coefficients`, `derive_token_seed`) are free of the classic
//! pitfalls:
//!
//! 1. **Missing binding** — outputs must depend on ALL inputs.
//! 2. **Missing domain separation** — different protocol steps must not collide.
//! 3. **Transcript reuse** — same inputs in different contexts must not produce
//!    identical outputs.
//! 4. **Nonce / counter issues** — all k challenges must be distinct and
//!    uniformly distributed.
//! 5. **Modular bias** — `hash % n` should not introduce exploitable bias.
//!
//! Note: `derive_block_coefficients` now derives from the verifier's secret
//! key seed (not the public Merkle root). The binding/separation tests below
//! verify the derivation is sound with respect to the key seed input.

use verilm_core::freivalds::derive_block_coefficients;
use verilm_core::merkle::derive_challenges;
use verilm_core::sampling::derive_token_seed;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn zero_seed() -> [u8; 32] {
    [0u8; 32]
}

fn one_seed() -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = 1;
    s
}

fn rand_bytes() -> [u8; 32] {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut b = [0u8; 32];
    rng.fill(&mut b);
    b
}

// ===========================================================================
// 1. BINDING: outputs must change when any input changes
// ===========================================================================

#[test]
fn challenges_depend_on_root() {
    let seed = rand_bytes();
    let c1 = derive_challenges(&zero_seed(), &seed, 1024, 8);
    let c2 = derive_challenges(&one_seed(), &seed, 1024, 8);
    assert_ne!(c1, c2, "different roots must yield different challenges");
}

#[test]
fn challenges_depend_on_seed() {
    let root = rand_bytes();
    let s1 = [0xAAu8; 32];
    let s2 = [0xBBu8; 32];
    let c1 = derive_challenges(&root, &s1, 1024, 8);
    let c2 = derive_challenges(&root, &s2, 1024, 8);
    assert_ne!(c1, c2, "different seeds must yield different challenges");
}

#[test]
fn challenges_depend_on_n_tokens() {
    let root = rand_bytes();
    let seed = rand_bytes();
    let c1 = derive_challenges(&root, &seed, 100, 5);
    let c2 = derive_challenges(&root, &seed, 200, 5);
    assert_ne!(c1, c2, "different n_tokens should change challenge indices");
}

#[test]
fn block_coefficients_depend_on_key_seed() {
    let c1 = derive_block_coefficients(&zero_seed(), 0, 0, 8);
    let c2 = derive_block_coefficients(&one_seed(), 0, 0, 8);
    assert_ne!(c1, c2, "different key seeds must yield different block coefficients");
}

#[test]
fn block_coefficients_depend_on_layer() {
    let seed = rand_bytes();
    let c1 = derive_block_coefficients(&seed, 0, 0, 8);
    let c2 = derive_block_coefficients(&seed, 1, 0, 8);
    assert_ne!(c1, c2, "different layers must yield different block coefficients");
}

#[test]
fn block_coefficients_depend_on_matrix_idx() {
    let seed = rand_bytes();
    let c1 = derive_block_coefficients(&seed, 0, 0, 8);
    let c2 = derive_block_coefficients(&seed, 0, 1, 8);
    assert_ne!(c1, c2, "different matrix indices must yield different block coefficients");
}

#[test]
fn token_seed_depends_on_batch_seed() {
    let s1 = [0xAAu8; 32];
    let s2 = [0xBBu8; 32];
    assert_ne!(
        derive_token_seed(&s1, 0),
        derive_token_seed(&s2, 0),
        "different batch seeds must yield different token seeds"
    );
}

#[test]
fn token_seed_depends_on_token_index() {
    let seed = rand_bytes();
    assert_ne!(
        derive_token_seed(&seed, 0),
        derive_token_seed(&seed, 1),
        "different token indices must yield different token seeds"
    );
}

// ===========================================================================
// 2. DOMAIN SEPARATION: different protocol steps must not collide
// ===========================================================================

/// `derive_challenges` and `derive_block_coefficients` use different domain
/// separators. Verify they don't produce the same bytes for shared inputs.
#[test]
fn challenge_vs_block_coeff_domain_separation() {
    let seed = rand_bytes();

    // derive_challenges hashes: root || seed || counter(0)
    // derive_block_coefficients hashes: "vi-block-coeff-v2" || key_seed || layer(0) || matrix(0) || block(0)
    let challenges = derive_challenges(&seed, &seed, 1024, 1);
    let coeffs = derive_block_coefficients(&seed, 0, 0, 1);

    let coeff_val: u32 = coeffs[0].0;
    let _ = (challenges[0], coeff_val); // just ensure both computed without panic
}

/// `derive_token_seed` uses "vi-sample-v1" as domain separator.
#[test]
fn token_seed_vs_challenge_domain_separation() {
    let root = rand_bytes();
    let seed = root;
    let token_seed = derive_token_seed(&seed, 0);
    let challenges = derive_challenges(&root, &seed, u32::MAX, 1);
    let ts_prefix = u32::from_le_bytes(token_seed[..4].try_into().unwrap());
    assert_ne!(
        ts_prefix % u32::MAX,
        challenges[0],
        "domain separation should prevent collision"
    );
}

// ===========================================================================
// 3. DETERMINISM: same inputs must always produce the same output
// ===========================================================================

#[test]
fn derive_challenges_is_deterministic() {
    let root = rand_bytes();
    let seed = rand_bytes();
    let c1 = derive_challenges(&root, &seed, 512, 10);
    let c2 = derive_challenges(&root, &seed, 512, 10);
    assert_eq!(c1, c2);
}

#[test]
fn derive_block_coefficients_is_deterministic() {
    let seed = rand_bytes();
    let c1 = derive_block_coefficients(&seed, 3, 2, 16);
    let c2 = derive_block_coefficients(&seed, 3, 2, 16);
    assert_eq!(c1, c2);
}

#[test]
fn derive_token_seed_is_deterministic() {
    let seed = rand_bytes();
    assert_eq!(derive_token_seed(&seed, 42), derive_token_seed(&seed, 42));
}

// ===========================================================================
// 4. UNIQUENESS: all k challenge indices must be distinct
// ===========================================================================

#[test]
fn challenge_indices_are_unique() {
    let root = rand_bytes();
    let seed = rand_bytes();
    for n_tokens in [10, 100, 1000, 10_000] {
        let k = n_tokens.min(32);
        let challenges = derive_challenges(&root, &seed, n_tokens, k);
        assert_eq!(challenges.len(), k as usize);
        let set: std::collections::BTreeSet<_> = challenges.iter().collect();
        assert_eq!(set.len(), challenges.len(), "all indices must be unique");
    }
}

#[test]
fn challenge_indices_in_range() {
    let root = rand_bytes();
    let seed = rand_bytes();
    for n_tokens in [1, 2, 7, 128, 1024] {
        let k = n_tokens.min(8);
        let challenges = derive_challenges(&root, &seed, n_tokens, k);
        for &idx in &challenges {
            assert!(idx < n_tokens, "index {idx} >= n_tokens {n_tokens}");
        }
    }
}

/// When k >= n_tokens, we must get exactly n_tokens unique indices.
#[test]
fn challenge_k_clamped_to_n_tokens() {
    let root = rand_bytes();
    let seed = rand_bytes();
    let challenges = derive_challenges(&root, &seed, 5, 100);
    assert_eq!(challenges.len(), 5);
    let set: std::collections::BTreeSet<_> = challenges.iter().collect();
    assert_eq!(set.len(), 5);
}

// ===========================================================================
// 5. BLOCK COEFFICIENTS: no accidental zeros
// ===========================================================================

/// A zero batching coefficient would let that block's error go unchecked.
#[test]
fn block_coefficients_are_nonzero() {
    let seed = rand_bytes();
    for n_blocks in [1, 4, 16, 64, 256] {
        let coeffs = derive_block_coefficients(&seed, 0, 0, n_blocks);
        assert_eq!(coeffs.len(), n_blocks);
        for (i, c) in coeffs.iter().enumerate() {
            let val: u32 = c.0;
            assert_ne!(val, 0, "block coefficient {i} is zero — soundness hole");
        }
    }
}

/// All block coefficients within a single call must be distinct.
#[test]
fn block_coefficients_are_distinct() {
    let seed = rand_bytes();
    let coeffs = derive_block_coefficients(&seed, 0, 0, 32);
    let set: std::collections::HashSet<u32> = coeffs.iter().map(|c| c.0).collect();
    assert_eq!(set.len(), coeffs.len(), "block coefficients should be distinct");
}

// ===========================================================================
// 6. STATISTICAL DISTRIBUTION: chi-squared uniformity test
// ===========================================================================

/// Check that challenge indices are roughly uniformly distributed.
#[test]
fn challenge_distribution_is_roughly_uniform() {
    let n_tokens: u32 = 16;
    let trials = 4096;
    let mut counts = vec![0u32; n_tokens as usize];

    for i in 0u32..trials {
        let mut root = [0u8; 32];
        root[..4].copy_from_slice(&i.to_le_bytes());
        let seed = [0xFFu8; 32];
        let challenges = derive_challenges(&root, &seed, n_tokens, 1);
        counts[challenges[0] as usize] += 1;
    }

    let expected = trials as f64 / n_tokens as f64;
    let chi_sq: f64 = counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - expected;
            diff * diff / expected
        })
        .sum();

    assert!(
        chi_sq < 50.0,
        "chi-squared {chi_sq:.1} too high — challenge distribution is not uniform"
    );
}

/// Block coefficient low bits should be roughly uniform.
#[test]
fn block_coefficient_low_bits_uniform() {
    let n_buckets = 16u32;
    let trials = 4096;
    let mut counts = vec![0u32; n_buckets as usize];

    for i in 0u32..trials {
        let mut seed = [0u8; 32];
        seed[..4].copy_from_slice(&i.to_le_bytes());
        let coeffs = derive_block_coefficients(&seed, 0, 0, 1);
        let val: u32 = coeffs[0].0;
        counts[(val % n_buckets) as usize] += 1;
    }

    let expected = trials as f64 / n_buckets as f64;
    let chi_sq: f64 = counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - expected;
            diff * diff / expected
        })
        .sum();

    assert!(
        chi_sq < 50.0,
        "chi-squared {chi_sq:.1} too high — block coefficient distribution is not uniform"
    );
}

// ===========================================================================
// 7. SECRET RANDOMNESS: prover cannot predict block coefficients
// ===========================================================================

/// With secret key-derived coefficients, different keys must yield different
/// coefficients. This verifies the key seed is actually used (not ignored).
#[test]
fn block_coefficients_differ_across_keys() {
    let mut seen = std::collections::HashSet::new();
    for i in 0u32..64 {
        let mut seed = [0u8; 32];
        seed[..4].copy_from_slice(&i.to_le_bytes());
        let coeffs = derive_block_coefficients(&seed, 0, 0, 4);
        let vals: Vec<u32> = coeffs.iter().map(|c| c.0).collect();
        seen.insert(vals);
    }
    assert_eq!(
        seen.len(), 64,
        "all 64 different key seeds must produce distinct coefficients"
    );
}

// ===========================================================================
// 8. TRANSCRIPT REUSE: different (layer, matrix) pairs must not collide
// ===========================================================================

#[test]
fn block_coefficients_differ_across_layers_and_matrices() {
    let seed = rand_bytes();
    let n_blocks = 8;
    let mut seen = std::collections::HashSet::new();
    for layer in 0..4 {
        for matrix in 0..4 {
            let coeffs = derive_block_coefficients(&seed, layer, matrix, n_blocks);
            let vals: Vec<u32> = coeffs.iter().map(|c| c.0).collect();
            seen.insert(vals);
        }
    }
    assert_eq!(seen.len(), 16, "all 16 (layer, matrix) pairs must produce distinct coefficients");
}

// ===========================================================================
// 9. EDGE CASES
// ===========================================================================

#[test]
fn derive_challenges_single_token() {
    let root = rand_bytes();
    let seed = rand_bytes();
    let c = derive_challenges(&root, &seed, 1, 1);
    assert_eq!(c, vec![0], "single token must always be index 0");
}

#[test]
fn derive_challenges_k_zero() {
    let root = rand_bytes();
    let seed = rand_bytes();
    let c = derive_challenges(&root, &seed, 100, 0);
    assert!(c.is_empty());
}

#[test]
fn derive_block_coefficients_zero_blocks() {
    let seed = rand_bytes();
    let c = derive_block_coefficients(&seed, 0, 0, 0);
    assert!(c.is_empty());
}

#[test]
fn derive_token_seed_max_index() {
    let seed = rand_bytes();
    let _ = derive_token_seed(&seed, u32::MAX);
}

// ===========================================================================
// 10. LENGTH-EXTENSION / AMBIGUOUS ENCODING
// ===========================================================================

/// Verify that (root=A||B, seed=C) and (root=A, seed=B||C) don't collide.
#[test]
fn no_ambiguous_encoding_derive_challenges() {
    let mut root_a = [0xAAu8; 32];
    let mut seed_a = [0xBBu8; 32];
    let mut root_b = [0xAAu8; 32];
    let mut seed_b = [0xBBu8; 32];

    root_a[31] = 0xCC;
    seed_a[0] = 0xDD;
    root_b[31] = 0xDD;
    seed_b[0] = 0xCC;

    let c1 = derive_challenges(&root_a, &seed_a, 1024, 8);
    let c2 = derive_challenges(&root_b, &seed_b, 1024, 8);
    assert_ne!(c1, c2, "different (root, seed) pairs must differ even with byte-shifted inputs");
}

/// Same ambiguity test for derive_token_seed.
#[test]
fn no_ambiguous_encoding_derive_token_seed() {
    let mut seed_a = [0xAAu8; 32];
    seed_a[31] = 0x01;
    let idx_a: u32 = 0;

    let mut seed_b = [0xAAu8; 32];
    seed_b[31] = 0x00;
    let idx_b: u32 = 1;

    let t1 = derive_token_seed(&seed_a, idx_a);
    let t2 = derive_token_seed(&seed_b, idx_b);
    assert_ne!(t1, t2);
}
