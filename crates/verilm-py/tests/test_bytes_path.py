"""Tests for the bytes fast path in the Python → Rust FFI boundary.

Exercises verilm_rs.commit() with raw bytes (the production path) and
verifies that malformed byte lengths are rejected with clear errors.
"""

import hashlib
import struct

import pytest
import verilm_rs


def _make_i8_bytes(n: int) -> bytes:
    """n random-ish INT8 values as raw bytes."""
    return bytes(i % 256 for i in range(n))


def _make_i32_bytes(n: int) -> bytes:
    """n INT32 values as native-endian bytes."""
    return b"".join(struct.pack("=i", i % 1000) for i in range(n))


def _make_f32_bytes(n: int) -> bytes:
    """n FLOAT32 values as native-endian bytes."""
    return b"".join(struct.pack("=f", float(i) * 0.01) for i in range(n))


def _make_layer_trace_bytes(hidden: int = 64, intermediate: int = 128):
    """Build one LayerTrace dict with all fields as raw bytes."""
    return {
        "x_attn": _make_i8_bytes(hidden),
        "q": _make_i32_bytes(hidden),
        "k": _make_i32_bytes(hidden),
        "v": _make_i32_bytes(hidden),
        "a": _make_i8_bytes(hidden),
        "attn_out": _make_i32_bytes(hidden),
        "x_ffn": _make_i8_bytes(hidden),
        "g": _make_i32_bytes(intermediate),
        "u": _make_i32_bytes(intermediate),
        "h": _make_i8_bytes(intermediate),
        "ffn_out": _make_i32_bytes(hidden),
        "kv_cache_k": [],
        "kv_cache_v": [],
        "scale_x_attn": 0.05,
        "scale_a": 0.05,
        "scale_x_ffn": 0.05,
        "scale_h": 0.05,
    }


def _make_layer_trace_lists(hidden: int = 64, intermediate: int = 128):
    """Build one LayerTrace dict with all fields as Python lists (fallback path)."""
    return {
        "x_attn": [i % 128 for i in range(hidden)],
        "q": [i % 1000 for i in range(hidden)],
        "k": [i % 1000 for i in range(hidden)],
        "v": [i % 1000 for i in range(hidden)],
        "a": [i % 128 for i in range(hidden)],
        "attn_out": [i % 1000 for i in range(hidden)],
        "x_ffn": [i % 128 for i in range(hidden)],
        "g": [i % 1000 for i in range(intermediate)],
        "u": [i % 1000 for i in range(intermediate)],
        "h": [i % 128 for i in range(intermediate)],
        "ffn_out": [i % 1000 for i in range(hidden)],
        "kv_cache_k": [],
        "kv_cache_v": [],
        "scale_x_attn": 0.05,
        "scale_a": 0.05,
        "scale_x_ffn": 0.05,
        "scale_h": 0.05,
    }


def _manifest():
    return {
        "tokenizer_hash": "aa" * 32,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "eos_policy": "stop",
        "weight_hash": "bb" * 32,
        "quant_hash": "cc" * 32,
        "system_prompt_hash": "dd" * 32,
    }


def _commit_params(traces):
    prompt = b"test"
    seed = hashlib.sha256(prompt).digest()
    n_tokens = len(traces)
    return {
        "traces": traces,
        "token_ids": list(range(1, n_tokens + 1)),
        "prompt": prompt,
        "sampling_seed": seed,
        "manifest": _manifest(),
    }


# ── Happy path: bytes ──


class TestBytesHappyPath:
    def test_commit_with_bytes(self):
        """Commit succeeds when all tensor fields are raw bytes."""
        n_layers = 2
        n_tokens = 3
        traces = [[_make_layer_trace_bytes() for _ in range(n_layers)] for _ in range(n_tokens)]
        state = verilm_rs.commit(**_commit_params(traces))

        assert state.n_tokens() == n_tokens
        assert state.n_layers() == n_layers
        assert len(state.merkle_root_hex()) == 64
        assert len(state.io_root_hex()) == 64

    def test_commit_with_lists(self):
        """Commit succeeds with the fallback list-of-ints path."""
        n_layers = 2
        n_tokens = 3
        traces = [[_make_layer_trace_lists() for _ in range(n_layers)] for _ in range(n_tokens)]
        state = verilm_rs.commit(**_commit_params(traces))

        assert state.n_tokens() == n_tokens

    def test_bytes_and_lists_produce_same_commitment(self):
        """Bytes path and list path produce identical commitments for the same data."""
        hidden = 64
        # Build matching data: same values, different representations.
        values_i8 = [i % 128 for i in range(hidden)]
        values_i32 = [i % 1000 for i in range(hidden)]

        bytes_trace = {
            "x_attn": bytes(v & 0xFF for v in values_i8),
            "q": b"".join(struct.pack("=i", v) for v in values_i32),
            "k": b"".join(struct.pack("=i", v) for v in values_i32),
            "v": b"".join(struct.pack("=i", v) for v in values_i32),
            "a": bytes(v & 0xFF for v in values_i8),
            "attn_out": b"".join(struct.pack("=i", v) for v in values_i32),
            "x_ffn": bytes(v & 0xFF for v in values_i8),
            "g": b"".join(struct.pack("=i", v) for v in values_i32),
            "u": b"".join(struct.pack("=i", v) for v in values_i32),
            "h": bytes(v & 0xFF for v in values_i8),
            "ffn_out": b"".join(struct.pack("=i", v) for v in values_i32),
            "kv_cache_k": [],
            "kv_cache_v": [],
        }
        list_trace = {
            "x_attn": values_i8,
            "q": values_i32,
            "k": values_i32,
            "v": values_i32,
            "a": values_i8,
            "attn_out": values_i32,
            "x_ffn": values_i8,
            "g": values_i32,
            "u": values_i32,
            "h": values_i8,
            "ffn_out": values_i32,
            "kv_cache_k": [],
            "kv_cache_v": [],
        }

        traces_bytes = [[bytes_trace]]
        traces_lists = [[list_trace]]

        state_b = verilm_rs.commit(**_commit_params(traces_bytes))
        state_l = verilm_rs.commit(**_commit_params(traces_lists))

        assert state_b.merkle_root_hex() == state_l.merkle_root_hex()
        assert state_b.io_root_hex() == state_l.io_root_hex()

    def test_residual_bytes(self):
        """Commit succeeds when residual field is passed as f32 bytes."""
        hidden = 64
        trace = _make_layer_trace_bytes(hidden=hidden)
        trace["residual"] = _make_f32_bytes(hidden)

        traces = [[trace]]
        state = verilm_rs.commit(**_commit_params(traces))
        assert state.n_tokens() == 1

    def test_kv_cache_bytes(self):
        """Commit succeeds when kv_cache fields are lists of bytes."""
        hidden = 64
        trace = _make_layer_trace_bytes(hidden=hidden)
        trace["kv_cache_k"] = [_make_i8_bytes(hidden) for _ in range(3)]
        trace["kv_cache_v"] = [_make_i8_bytes(hidden) for _ in range(3)]

        traces = [[trace]]
        state = verilm_rs.commit(**_commit_params(traces))
        assert state.n_tokens() == 1

    def test_audit_after_bytes_commit(self):
        """Audit proof can be generated from a bytes-path commitment."""
        n_layers = 2
        traces = [[_make_layer_trace_bytes() for _ in range(n_layers)] for _ in range(3)]
        state = verilm_rs.commit(**_commit_params(traces))

        proof = state.audit_stratified(0, [0, 1], "routine")
        assert len(proof) > 0
        assert isinstance(proof, bytes)


# ── Failure tests: malformed byte lengths ──


class TestBytesBadLengths:
    def test_bad_i32_byte_length(self):
        """i32 field with byte length not a multiple of 4 is rejected."""
        trace = _make_layer_trace_bytes()
        trace["q"] = b"\x00\x01\x02"  # 3 bytes, not a multiple of 4

        with pytest.raises(ValueError, match="i32 bytes length 3 not a multiple of 4"):
            verilm_rs.commit(**_commit_params([[trace]]))

    def test_bad_f32_byte_length(self):
        """f32 residual field with byte length not a multiple of 4 is rejected."""
        trace = _make_layer_trace_bytes()
        trace["residual"] = b"\x00\x01\x02\x03\x05"  # 5 bytes

        with pytest.raises(ValueError, match="f32 bytes length 5 not a multiple of 4"):
            verilm_rs.commit(**_commit_params([[trace]]))

    def test_bad_i32_various_lengths(self):
        """Various non-multiple-of-4 lengths for i32 fields are all rejected."""
        for bad_len in [1, 2, 3, 5, 6, 7]:
            trace = _make_layer_trace_bytes()
            trace["attn_out"] = bytes(bad_len)
            with pytest.raises(ValueError, match="i32 bytes length"):
                verilm_rs.commit(**_commit_params([[trace]]))

    def test_bad_f32_various_lengths(self):
        """Various non-multiple-of-4 lengths for f32 residual are all rejected."""
        for bad_len in [1, 2, 3, 5, 6, 7]:
            trace = _make_layer_trace_bytes()
            trace["residual"] = bytes(bad_len)
            with pytest.raises(ValueError, match="f32 bytes length"):
                verilm_rs.commit(**_commit_params([[trace]]))

    def test_empty_bytes_accepted(self):
        """Empty bytes for i32 fields produce zero-length vectors (valid edge case)."""
        trace = _make_layer_trace_bytes()
        trace["q"] = b""
        # This should not raise — empty is a valid multiple of 4.
        # Whether the commitment engine accepts zero-length vectors is a
        # separate concern; we're testing the FFI boundary.
        try:
            verilm_rs.commit(**_commit_params([[trace]]))
        except Exception as e:
            # If the commitment engine rejects it, that's fine —
            # but it should NOT be a byte-length error.
            assert "not a multiple of 4" not in str(e)
