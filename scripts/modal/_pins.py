"""Canonical dependency pins for Modal scripts.

Two stacks:
  VERIFICATION — pinned tuple for reproducible E2E / benchmark / measurement runs.
  RESEARCH     — loose constraints for exploratory scripts testing new versions.

Usage in Modal scripts:

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from _pins import VERIFICATION, VERIFICATION_EXTRA

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*VERIFICATION, "httpx")  # add per-script extras
        ...
    )

To override a single package (e.g. test a new vllm):

    VERILM_VLLM_SPEC=vllm==0.8.4 modal run scripts/modal/test_e2e_v4.py
"""

import os

# ── Verification stack (pinned, reproducible) ──────────────────────
# Known-good tuple from Modal build 2026-04-11.
# Override individual packages via VERILM_*_SPEC env vars.

VLLM_SPEC = os.environ.get("VERILM_VLLM_SPEC", "vllm==0.8.3")
TORCH_SPEC = os.environ.get("VERILM_TORCH_SPEC", "torch==2.6.0")
TRANSFORMERS_SPEC = os.environ.get("VERILM_TRANSFORMERS_SPEC", "transformers==4.57.6")
COMPRESSED_TENSORS_SPEC = os.environ.get("VERILM_COMPRESSED_TENSORS_SPEC", "compressed-tensors==0.9.2")
NUMPY_SPEC = os.environ.get("VERILM_NUMPY_SPEC", "numpy==2.1.3")
SAFETENSORS_SPEC = os.environ.get("VERILM_SAFETENSORS_SPEC", "safetensors==0.7.0")

# Core verification stack: vllm + its critical deps, all pinned.
VERIFICATION = [
    VLLM_SPEC,
    TORCH_SPEC,
    TRANSFORMERS_SPEC,
    COMPRESSED_TENSORS_SPEC,
    NUMPY_SPEC,
    SAFETENSORS_SPEC,
    "fastapi",
    "maturin",
]

# Extra packages some scripts need (not pinned — they don't affect numerics).
VERIFICATION_EXTRA = {
    "httpx": "httpx",
    "uvicorn": "uvicorn",
    "zstandard": "zstandard",
    "ninja": "ninja",
}

# ── Keygen-only stack (no vllm/vllm GPU deps) ─────────────────────
KEYGEN = [
    TORCH_SPEC,
    NUMPY_SPEC,
    SAFETENSORS_SPEC,
    "huggingface_hub==0.36.2",
    "maturin",
]

# ── Research stack (loose, for testing new versions) ───────────────
RESEARCH = [
    "vllm>=0.8",
    "torch",
    "transformers",
    "numpy",
    "fastapi",
    "maturin",
]
