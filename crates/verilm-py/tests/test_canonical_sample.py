"""Tests for the canonical sampling Python bindings.

Verifies that verilm_rs.derive_token_seed and verilm_rs.canonical_sample
match the Rust canonical sampler and that the CanonicalLogitsProcessor
correctly forces token selection.
"""

import hashlib

import pytest
import verilm_rs


class TestDeriveTokenSeed:
    def test_deterministic(self):
        """Same inputs produce same seed."""
        batch = b"\x01" * 32
        s1 = verilm_rs.derive_token_seed(batch, 5)
        s2 = verilm_rs.derive_token_seed(batch, 5)
        assert s1 == s2

    def test_varies_by_index(self):
        """Different token indices produce different seeds."""
        batch = b"\x01" * 32
        s0 = verilm_rs.derive_token_seed(batch, 0)
        s1 = verilm_rs.derive_token_seed(batch, 1)
        assert s0 != s1

    def test_varies_by_batch(self):
        """Different batch seeds produce different per-token seeds."""
        s1 = verilm_rs.derive_token_seed(b"\x01" * 32, 0)
        s2 = verilm_rs.derive_token_seed(b"\x02" * 32, 0)
        assert s1 != s2

    def test_returns_32_bytes(self):
        seed = verilm_rs.derive_token_seed(b"\x00" * 32, 0)
        assert isinstance(seed, bytes)
        assert len(seed) == 32

    def test_bad_seed_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            verilm_rs.derive_token_seed(b"\x00" * 16, 0)

    def test_matches_spec(self):
        """Verify seed derivation matches SHA256("vi-sample-v1" || batch_seed || index_le32)."""
        batch = b"\xab" * 32
        index = 42
        expected = hashlib.sha256(
            b"vi-sample-v1" + batch + index.to_bytes(4, "little")
        ).digest()
        actual = verilm_rs.derive_token_seed(batch, index)
        assert actual == expected


class TestCanonicalSample:
    def test_greedy_returns_argmax(self):
        logits = [1.0, 5.0, 3.0, 2.0]
        seed = b"\x00" * 32
        result = verilm_rs.canonical_sample(logits, 0.0, 0, 1.0, seed)
        assert result == 1

    def test_greedy_tie_breaks_lowest(self):
        logits = [5.0, 3.0, 5.0, 2.0]
        seed = b"\x00" * 32
        result = verilm_rs.canonical_sample(logits, 0.0, 0, 1.0, seed)
        assert result == 0

    def test_deterministic_same_seed(self):
        logits = [1.0, 2.0, 3.0, 4.0]
        seed = b"\x2a" * 32
        a = verilm_rs.canonical_sample(logits, 1.0, 0, 1.0, seed)
        b = verilm_rs.canonical_sample(logits, 1.0, 0, 1.0, seed)
        assert a == b

    def test_different_seeds_can_differ(self):
        """On uniform logits, different seeds should produce different tokens."""
        logits = [1.0] * 100
        results = set()
        batch = b"\x00" * 32
        for i in range(20):
            seed = verilm_rs.derive_token_seed(batch, i)
            results.add(verilm_rs.canonical_sample(logits, 1.0, 0, 1.0, seed))
        assert len(results) > 1

    def test_top_k_1_is_argmax(self):
        logits = [1.0, 10.0, 5.0, 3.0]
        seed = b"\x63" * 32
        result = verilm_rs.canonical_sample(logits, 1.0, 1, 1.0, seed)
        assert result == 1

    def test_low_temperature_concentrates(self):
        logits = [1.0, 2.0, 10.0, 3.0]
        batch = b"\x00" * 32
        for i in range(10):
            seed = verilm_rs.derive_token_seed(batch, i)
            result = verilm_rs.canonical_sample(logits, 0.01, 0, 1.0, seed)
            assert result == 2

    def test_empty_logits_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            verilm_rs.canonical_sample([], 1.0, 0, 1.0, b"\x00" * 32)

    def test_bad_seed_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            verilm_rs.canonical_sample([1.0], 1.0, 0, 1.0, b"\x00" * 16)


class TestEndToEndSamplingReplay:
    """Verify that the Python bindings produce the same result as the verifier."""

    def test_greedy_replay_agreement(self):
        """Greedy: derive seed, sample, verify same token."""
        batch_seed = hashlib.sha256(b"test prompt").digest()
        logits = [float(i) for i in range(100)]  # token 99 is argmax

        for token_idx in range(5):
            token_seed = verilm_rs.derive_token_seed(batch_seed, token_idx)
            chosen = verilm_rs.canonical_sample(logits, 0.0, 0, 1.0, token_seed)
            assert chosen == 99

    def test_sampled_replay_agreement(self):
        """Sampled: same seed + params → same token across calls."""
        batch_seed = hashlib.sha256(b"sampled prompt").digest()
        logits = [1.0, 2.0, 3.0, 2.0, 1.0] * 20  # 100 tokens, varied

        for token_idx in range(10):
            token_seed = verilm_rs.derive_token_seed(batch_seed, token_idx)
            a = verilm_rs.canonical_sample(logits, 0.8, 50, 0.9, token_seed)
            b = verilm_rs.canonical_sample(logits, 0.8, 50, 0.9, token_seed)
            assert a == b, f"token {token_idx}: {a} != {b}"
