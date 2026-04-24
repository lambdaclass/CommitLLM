"""Tests for verilm_rs.deserialize_v4_audit — exposes the publicly-committed
fields of a V4 audit binary as a Python dict so downstream callers can
inspect `output_text`, commitment hashes, and prompt bytes without
re-implementing the binary wire format.

Full-path tests that exercise a real V4 audit binary end-to-end live in
the Rust-side prover/verifier integration tests. These tests focus on
the Python surface: symbol presence, error shape on malformed input.
"""

import pytest
import verilm_rs


class TestDeserializeV4Audit:
    def test_symbol_is_exported(self):
        """The function must be registered on the module."""
        assert hasattr(verilm_rs, "deserialize_v4_audit")
        assert callable(verilm_rs.deserialize_v4_audit)

    def test_rejects_non_audit_bytes(self):
        """Garbage bytes should raise a ValueError, not crash or
        return a half-constructed dict."""
        with pytest.raises(ValueError, match="deserialize"):
            verilm_rs.deserialize_v4_audit(b"not-a-real-audit-binary")

    def test_rejects_empty_bytes(self):
        with pytest.raises(ValueError, match="deserialize"):
            verilm_rs.deserialize_v4_audit(b"")

    def test_rejects_truncated_magic(self):
        """A few bytes that don't match the binary header must fail
        cleanly."""
        with pytest.raises(ValueError, match="deserialize"):
            verilm_rs.deserialize_v4_audit(b"\x00\x01\x02\x03")
