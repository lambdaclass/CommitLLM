"""
Core capture logic for verified inference.

Wraps cutlass_scaled_mm to capture inputs and identifies each capture's
position in the transformer (layer index, projection name).

What this captures and why:
  - INT8 matmul inputs (x_i8): verified against public weights via Freivalds
  - INT32 accumulator (acc_i32 = W_i8 @ x_i8): needed for Freivalds and R_T
  - Per-token activation scale: dynamic, not in public weights
  - Layer/projection identity: for trace reconstruction
  - The W_o input IS the post-attention output (attn_out_i8) — the only
    non-derivable intermediate in the protocol

NOT captured (verifier has these from public weights):
  - Weight matrices (W_i8), weight scales, biases

Layer/proj identity is derived from call counting (zero overhead) rather
than PyTorch model hooks. Both Llama and Qwen W8A8 models use fused
projections with a fixed 4-call pattern per layer:
  qkv_proj → o_proj → gate_up_proj → down_proj

The CUTLASS kernel runs unchanged — model output is bit-identical.
"""

import os
import logging
import time
from typing import Optional, List, Tuple

import torch

logger = logging.getLogger("verilm")

# ── Types ──

# Each capture: (layer, proj, input_i8, acc_i32, activation_scale)
CaptureTuple = Tuple[Optional[int], Optional[str], torch.Tensor, torch.Tensor, torch.Tensor]


# ── Call-counting projection identification ──
#
# W8A8 models in vLLM use fused projections: each transformer layer
# produces exactly 4 cutlass_scaled_mm calls in this fixed order.
# We derive layer/proj from the call index — zero Python hook overhead.

PROJ_SEQUENCE = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
PROJS_PER_LAYER = len(PROJ_SEQUENCE)

# Set by configure_layer_count() / configure_from_model() after model loading.
_n_layers = 0           # Number of transformer layers
_calls_per_fwd = 0      # n_layers * PROJS_PER_LAYER
_configured = False

# Model geometry — set by configure_from_model() for GQA-aware QKV split.
_num_heads = 0
_num_kv_heads = 0
_head_dim = 0
_hidden_size = 0
_intermediate_size = 0
# Derived split widths for fused projections.
_q_dim = 0              # num_heads * head_dim
_kv_dim = 0             # num_kv_heads * head_dim
_gate_up_half = 0       # intermediate_size (each half of gate_up)


def configure_layer_count(n_layers: int):
    """Set the number of transformer layers for call-counting identification.

    Must be called once after model loading. The wrapper uses this to derive
    layer index and projection name from the matmul call sequence.
    """
    global _n_layers, _calls_per_fwd, _configured
    _n_layers = n_layers
    _calls_per_fwd = n_layers * PROJS_PER_LAYER
    _configured = True
    logger.info(
        "verilm: configured for %d layers, %d calls/fwd (pid=%d)",
        n_layers, _calls_per_fwd, os.getpid(),
    )


def configure_from_model(model):
    """Auto-detect layer count and model geometry from a vLLM model object.

    Extracts num_heads, num_kv_heads, head_dim, hidden_size, and
    intermediate_size from the model config for GQA-aware QKV splitting.
    """
    global _num_heads, _num_kv_heads, _head_dim, _hidden_size, _intermediate_size
    global _q_dim, _kv_dim, _gate_up_half

    max_layer = -1
    for name, _ in model.named_modules():
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    max_layer = max(max_layer, int(parts[i + 1]))
                except ValueError:
                    pass
    if max_layer >= 0:
        configure_layer_count(max_layer + 1)
    else:
        logger.warning("verilm: could not detect layer count from model")
        return

    # Extract model geometry from config.
    cfg = getattr(model, "config", None)
    if cfg is None:
        logger.warning("verilm: model has no config, split dimensions unavailable")
        return

    _num_heads = getattr(cfg, "num_attention_heads", 0)
    _num_kv_heads = getattr(cfg, "num_key_value_heads", _num_heads)
    _head_dim = getattr(cfg, "head_dim", 0)
    if _head_dim == 0 and _num_heads > 0:
        _hidden_size = getattr(cfg, "hidden_size", 0)
        _head_dim = _hidden_size // _num_heads if _num_heads else 0
    else:
        _hidden_size = getattr(cfg, "hidden_size", 0)
    _intermediate_size = getattr(cfg, "intermediate_size", 0)

    # Derived split widths.
    _q_dim = _num_heads * _head_dim
    _kv_dim = _num_kv_heads * _head_dim
    _gate_up_half = _intermediate_size

    logger.info(
        "verilm: model geometry — heads=%d, kv_heads=%d, head_dim=%d, "
        "hidden=%d, intermediate=%d, q_dim=%d, kv_dim=%d (pid=%d)",
        _num_heads, _num_kv_heads, _head_dim,
        _hidden_size, _intermediate_size, _q_dim, _kv_dim, os.getpid(),
    )


# ── Capture buffer ──


class CaptureBuffer:
    """Lock-free buffer for captured matmul inputs.

    No lock needed: vLLM runs inference single-threaded per worker process.
    The buffer is a plain list for minimal append overhead.
    """

    def __init__(self, max_entries: int = 100_000):
        self._entries: List[CaptureTuple] = []
        self._max_entries = max_entries
        self.total_captured = 0
        self.enabled = True

        # Request tracking
        self._current_request_id: Optional[str] = None
        self._current_prompt_ids: List[int] = []
        self._request_start: float = 0.0
        self._request_entries_start: int = 0

    def append(self, entry: CaptureTuple):
        self._entries.append(entry)
        self.total_captured += 1
        # Trim when we hit 2x max to avoid checking every call
        if len(self._entries) > self._max_entries * 2:
            self._entries = self._entries[-self._max_entries:]

    def drain(self) -> List[CaptureTuple]:
        """Remove and return all entries."""
        entries = self._entries
        self._entries = []
        return entries

    def begin_request(
        self, request_id: str, prompt_token_ids: Optional[List[int]] = None
    ):
        """Mark the start of a generation request."""
        self._current_request_id = request_id
        self._current_prompt_ids = prompt_token_ids or []
        self._request_start = time.monotonic()
        self._request_entries_start = self.total_captured

    def end_request(
        self, generated_token_ids: Optional[List[int]] = None
    ) -> Optional[dict]:
        """Mark end of request. Returns dict with request captures."""
        if self._current_request_id is None:
            return None

        req_count = self.total_captured - self._request_entries_start
        entries = self._entries[-req_count:] if req_count > 0 else []

        result = {
            "request_id": self._current_request_id,
            "prompt_token_ids": self._current_prompt_ids,
            "generated_token_ids": generated_token_ids or [],
            "entries": entries,
            "started_at": self._request_start,
            "ended_at": time.monotonic(),
        }
        self._current_request_id = None
        self._current_prompt_ids = []
        return result

    def stats(self) -> dict:
        return {
            "buffered": len(self._entries),
            "total_captured": self.total_captured,
            "enabled": self.enabled,
            "in_request": self._current_request_id is not None,
        }


# Global capture buffer
_capture_buffer = CaptureBuffer()


def get_capture_buffer() -> CaptureBuffer:
    """Get the global capture buffer."""
    return _capture_buffer


# ── Kernel wrapper ──

# Stores the REAL cutlass_scaled_mm function (not our wrapper).
_real_kernel = [None]
_patched = False
_call_counter = 0
_log_interval = 5000

# Reusable padded buffers for decode-path _int_mm (M < 16).
# Keyed by (N, device) to avoid per-call torch.nn.functional.pad allocation.
# Each buffer is (32, N) int8 with rows [M:] pre-zeroed.
_pad_buffers: dict = {}


def _wrapped_cutlass_scaled_mm(
    a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None
):
    """
    Wrapper around cutlass_scaled_mm that captures inputs and INT32 accumulator.

    Layer/proj identity is derived from call counting (zero overhead).
    The acc_i32 is computed via torch._int_mm alongside the real kernel.
    """
    global _call_counter

    real = _real_kernel[0]
    if real is None or real is _wrapped_cutlass_scaled_mm:
        raise RuntimeError(
            "verilm: _real_kernel not set or points to wrapper — "
            "this is a patching bug"
        )

    # Always call the original kernel — output is unchanged
    output = real(a, b, scale_a, scale_b, out_dtype, bias)

    buf = _capture_buffer
    if not buf.enabled:
        return output

    # Compute INT32 accumulator (what the verifier needs for Freivalds).
    # torch._int_mm requires M >= 16; pad small batches (decode tokens).
    M = a.shape[0]
    if M <= 16:
        N = a.shape[1]
        pad_key = (N, a.device)
        pad_buf = _pad_buffers.get(pad_key)
        if pad_buf is None:
            pad_buf = torch.zeros(32, N, dtype=torch.int8, device=a.device)
            _pad_buffers[pad_key] = pad_buf
        pad_buf[:M, :] = a
        acc_i32 = torch._int_mm(pad_buf, b)[:M, :]
    else:
        acc_i32 = torch._int_mm(a, b)

    # Derive layer/proj from call count (zero overhead vs PyTorch hooks)
    if _configured:
        idx = _call_counter % _calls_per_fwd
        layer = idx // PROJS_PER_LAYER
        proj = PROJ_SEQUENCE[idx % PROJS_PER_LAYER]
    else:
        layer = None
        proj = None

    _call_counter += 1

    # Move to CPU here so downstream code doesn't need per-field .cpu() calls.
    # non_blocking=True allows the DMA to overlap with subsequent GPU kernels;
    # server.py calls torch.cuda.synchronize() after generate() to ensure
    # all transfers have completed before draining.
    buf.append((layer, proj, a.to("cpu", non_blocking=True),
                acc_i32.to("cpu", non_blocking=True), scale_a))

    if buf.total_captured % _log_interval == 0:
        logger.info(
            "verilm: %d captures (pid=%d)", buf.total_captured, os.getpid()
        )

    return output


# ── Model access helpers ──


def get_model_from_llm(llm):
    """Extract the model object from a vLLM LLM instance.

    Tries multiple access paths across vLLM versions.
    """
    # vLLM v1 (0.8+)
    try:
        return llm.llm_engine.get_model()
    except (AttributeError, NotImplementedError):
        pass

    # Legacy path
    try:
        return llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass

    # V1 with multiprocessing disabled
    try:
        return llm.llm_engine.model_executor.model
    except AttributeError:
        pass

    raise RuntimeError(
        "Could not extract model from LLM. "
        "Ensure VLLM_ENABLE_V1_MULTIPROCESSING=0 is set."
    )


# ── Plugin registration ──


def register():
    """
    Plugin entry point — called by vLLM on startup.

    Patches cutlass_scaled_mm for input capture. Call configure_layer_count()
    or configure_from_model() after model loading for layer/proj identification.
    """
    global _patched

    if _patched:
        return

    capture_env = os.environ.get("VERILM_CAPTURE", "1")
    if capture_env == "0":
        logger.info("verilm loaded but capture disabled (VERILM_CAPTURE=0)")
        _capture_buffer.enabled = False
        _patched = True
        return

    try:
        import vllm._custom_ops as ops
    except ImportError:
        logger.warning("vllm._custom_ops not available — plugin not loaded")
        return

    # Re-check: the import above may have triggered the _startup.py import
    # hook, which already patched cutlass_scaled_mm during module loading.
    if _patched:
        return

    original = ops.cutlass_scaled_mm
    if original is _wrapped_cutlass_scaled_mm:
        logger.warning("verilm: cutlass_scaled_mm already wrapped, skipping")
        _patched = True
        return

    _real_kernel[0] = original
    ops.cutlass_scaled_mm = _wrapped_cutlass_scaled_mm
    _patched = True

    logger.info(
        "verilm: wrapped cutlass_scaled_mm for input capture "
        "(buffer max=%d, call-counting mode)",
        _capture_buffer._max_entries,
    )
