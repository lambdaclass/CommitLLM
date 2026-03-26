"""
Core capture logic for verified inference.

Wraps cutlass_scaled_mm to capture inputs and identifies each capture's
position in the transformer (layer index, projection name).

Captures only the o_proj input (post-attention 'a') and per-projection
scales. Skips _int_mm entirely — the biggest GPU overhead. Used with
commit_minimal_from_captures for retained-state (V4) commitments.

Layer/proj identity is derived from call counting (zero overhead) rather
than PyTorch model hooks. Both Llama and Qwen W8A8 models use fused
projections with a fixed 4-call pattern per layer:
  qkv_proj → o_proj → gate_up_proj → down_proj

The CUTLASS kernel runs unchanged — model output is bit-identical.

Synchronization (VERILM_SYNC_MODE env var):

  "global" (default):
    torch.cuda.synchronize() after inference — stalls entire GPU.

  "event":
    D2H copies on a dedicated CUDA stream. Event-based wait before
    drain — only blocks until copy completion, not all GPU work.

Both modes produce identical captured tensors and commitments.

CUDA graphs: vLLM must use enforce_eager=True for verified inference.
CUDA graphs replay GPU kernels without executing Python, so the capture
wrapper (counter, D2H copy, buffer append) never runs during replay.
This is a fundamental constraint of per-kernel capture.
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
_O_PROJ_IDX = 1  # o_proj position in PROJ_SEQUENCE

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

# Rust CaptureHook: handles counter/modulus/scale-storage in native code.
# Created by configure_layer_count() when in minimal mode. None = Python fallback.
_capture_hook = None

# Native C++ capture extension (JIT-compiled). When active, replaces the
# Python wrapper entirely — vLLM calls the C++ function directly.
# None = not loaded or not available.
_native_capture = None


def configure_layer_count(n_layers: int):
    """Set the number of transformer layers for call-counting identification.

    Must be called once after model loading. The wrapper uses this to derive
    layer index and projection name from the matmul call sequence.
    """
    global _n_layers, _calls_per_fwd, _configured, _capture_hook
    _n_layers = n_layers
    _calls_per_fwd = n_layers * PROJS_PER_LAYER
    _configured = True
    logger.info(
        "verilm: configured for %d layers, %d calls/fwd (pid=%d)",
        n_layers, _calls_per_fwd, os.getpid(),
    )

    # Create Rust capture hook — moves counter arithmetic,
    # scale storage, and proj identification out of Python bytecode.
    try:
        import verilm_rs
        _capture_hook = verilm_rs.CaptureHook(
            calls_per_fwd=_calls_per_fwd,
            projs_per_layer=PROJS_PER_LAYER,
            o_proj_idx=_O_PROJ_IDX,
        )
        logger.info("verilm: Rust CaptureHook active (pid=%d)", os.getpid())
    except (ImportError, AttributeError):
        logger.info("verilm: Rust CaptureHook unavailable, using Python path")


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


def activate_native_capture(device_hint_tensor=None):
    """Load the native C++ capture accumulator for fast scale storage + drain.

    Hybrid mode: the Python wrapper keeps calling the real kernel directly
    (fastest path), but uses C++ record() for scale storage and C++ drain()
    for bulk at::cat transfer. This avoids both:
      - c10 typed dispatcher overhead (~5-8us/call) from full-replace mode
      - Python torch.cat on 28K-element list overhead at drain time

    Must be called AFTER:
      - vllm._C is loaded (op registered in dispatcher)
      - configure_layer_count() has been called
      - init_pinned_slab() has been called (optional, for o_proj capture)

    Returns True if native accumulator is active, False on fallback.
    """
    global _native_capture

    if os.environ.get("VERILM_NO_NATIVE_ACC") == "1":
        logger.info("verilm: native accumulator disabled via VERILM_NO_NATIVE_ACC=1")
        return False

    if not _configured:
        logger.warning("verilm: activate_native_capture called before configure_layer_count")
        return False

    # JIT-compile the C++ extension (suppress compiler warnings from torch headers).
    try:
        from torch.utils.cpp_extension import load
        cpp_path = os.path.join(os.path.dirname(__file__), "csrc", "capture_native.cpp")
        if not os.path.exists(cpp_path):
            logger.info("verilm: native capture source not found at %s", cpp_path)
            return False

        _native_capture = load(
            name="capture_native",
            sources=[cpp_path],
            extra_cflags=["-w"],   # suppress all compiler warnings
            extra_cuda_cflags=["-w"] if torch.cuda.is_available() else [],
            verbose=False,
        )
        logger.info("verilm: native capture extension compiled (pid=%d)", os.getpid())
    except Exception as e:
        logger.warning("verilm: native capture compilation failed: %s", e)
        _native_capture = None
        return False

    # Initialize the accumulator (geometry + state reset).
    try:
        _native_capture.init(_calls_per_fwd, PROJS_PER_LAYER, _O_PROJ_IDX)
    except Exception as e:
        logger.warning("verilm: native capture init failed: %s", e)
        _native_capture = None
        return False

    # Set up pinned slab (if already initialized).
    buf = _capture_buffer
    if buf._pinned_slab is not None:
        _native_capture.init_slab(buf._pinned_slab)
        logger.info("verilm: native capture using existing pinned slab")

    # Hybrid mode: do NOT replace ops.cutlass_scaled_mm.
    # The Python wrapper calls the real kernel directly, then uses
    # native.record() + native.copy_slab() for C++ accumulation.
    logger.info(
        "verilm: native accumulator active — C++ scale storage + drain (pid=%d)",
        os.getpid(),
    )
    return True


# ── Capture buffer ──


class CaptureBuffer:
    """Lock-free buffer for captured matmul inputs.

    No lock needed: vLLM runs inference single-threaded per worker process.
    The buffer is a plain list for minimal append overhead.

    Sync modes (VERILM_SYNC_MODE env var):

      "global" (default):
        After inference completes, waits with torch.cuda.synchronize().
        Safe baseline — stalls the entire GPU but cannot race.

      "event":
        D2H copies run on a dedicated CUDA stream. A single event is
        recorded after each copy batch; wait_for_transfers() synchronizes
        on that event only. Does not stall unrelated GPU work.

    The sync mode is set once via set_sync_mode() during register().
    Both modes produce identical captured tensors.
    """

    def __init__(self, max_entries: int = 100_000):
        self._entries: List[CaptureTuple] = []
        self._max_entries = max_entries
        self.total_captured = 0
        self._enabled = True

        # Minimal-mode dedicated buffers: flat lists instead of tuples.
        self._minimal_o_inputs: List[torch.Tensor] = []
        self._minimal_call_count: int = 0

        # Scale references: GPU tensor refs collected during the wrapper
        # (plain list.append, near-zero overhead). At drain time, we stack
        # them into a single GPU tensor and do one bulk D2H transfer instead
        # of thousands of individual .item() calls.
        self._minimal_scales: list = []

        # Pinned CPU slab for o_proj D2H: avoids per-call tensor allocation.
        # Initialized by init_pinned_slab() after model config is known.
        self._pinned_slab: Optional[torch.Tensor] = None  # (capacity, hidden_dim), i8
        self._pinned_slab_idx: int = 0  # row write cursor
        self._slab_offsets: List[tuple] = []  # (start_row, end_row) per o_proj call

        # Sync infrastructure — initialized lazily by set_sync_mode("event").
        self._sync_mode = "global"
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._sync_event: Optional[torch.cuda.Event] = None
        self._transfers_pending = False

        # Request tracking
        self._current_request_id: Optional[str] = None
        self._current_prompt_ids: List[int] = []
        self._request_start: float = 0.0
        self._request_entries_start: int = 0

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        # Sync with native C++ capture (it has its own enabled flag).
        if _native_capture is not None:
            _native_capture.set_enabled(value)

    def set_sync_mode(self, mode: str):
        """Set synchronization mode. Call once after CUDA is initialized.

        "global": torch.cuda.synchronize() (safe default).
        "event":  dedicated copy stream + event sync.
        """
        if mode == "event":
            self._copy_stream = torch.cuda.Stream()
            self._sync_event = torch.cuda.Event()
            self._sync_mode = "event"
            logger.info(
                "verilm: sync mode = event (dedicated copy stream, pid=%d)",
                os.getpid(),
            )
        elif mode == "global":
            self._copy_stream = None
            self._sync_event = None
            self._sync_mode = "global"
            logger.info(
                "verilm: sync mode = global (torch.cuda.synchronize, pid=%d)",
                os.getpid(),
            )
        else:
            logger.warning(
                "verilm: unknown VERILM_SYNC_MODE=%s, using global", mode,
            )
        self._transfers_pending = False

    def copy_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy a GPU tensor to CPU using the configured sync strategy.

        Event mode: issues copy on the dedicated copy stream after
        synchronizing with the compute stream. Next compute kernels
        can overlap with the D2H transfer.

        Global mode: non-blocking copy on the current (compute) stream.
        Caller must use wait_for_transfers() before reading the result.
        """
        if self._sync_mode == "event" and self._copy_stream is not None:
            # Make copy stream wait for the tensor to be produced on compute stream.
            self._copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._copy_stream):
                cpu = tensor.to("cpu", non_blocking=True)
                self._sync_event.record()
            self._transfers_pending = True
            return cpu

        # Global mode: copy on compute stream, wait globally later.
        self._transfers_pending = True
        return tensor.to("cpu", non_blocking=True)

    def append(self, entry: CaptureTuple):
        self._entries.append(entry)
        self.total_captured += 1
        # Trim when we hit 2x max to avoid checking every call
        if len(self._entries) > self._max_entries * 2:
            self._entries = self._entries[-self._max_entries:]

    def drain(self) -> List[CaptureTuple]:
        """Remove and return all entries (full-mode captures).

        In event mode, warns if transfers haven't been waited on yet
        and auto-waits to prevent reading incomplete data.
        Also clears minimal-mode buffers (used as pre-request reset).
        """
        if self._transfers_pending:
            logger.warning(
                "CaptureBuffer.drain() called with pending transfers — "
                "auto-waiting (call wait_for_transfers() explicitly)"
            )
            self.wait_for_transfers()
        entries = self._entries
        self._entries = []
        # Also clear minimal-mode buffers (pre-request reset).
        self._minimal_o_inputs = []
        self._minimal_scales = []
        self._minimal_call_count = 0
        self._pinned_slab_idx = 0
        self._slab_offsets = []
        # Clear Rust hook state too (if active).
        if _capture_hook is not None:
            _capture_hook.drain_scales()
        # Clear native capture state (if active).
        if _native_capture is not None:
            _native_capture.drain_discard()
        return entries

    def drain_minimal(self):
        """Drain minimal-mode buffers. Returns (o_inputs, scales, call_count).

        o_inputs: list of CPU tensors, one per o_proj call (n_fwd * n_layers).
                  When pinned slab is active, these are views into contiguous
                  pinned memory (no per-call allocation was done).
        scales: numpy float32 array of pre-extracted scale values, one per
                matmul call. Bulk-transferred from GPU in a single D2H copy.
        call_count: total matmul calls captured (= len(scales)).
        """
        if self._transfers_pending:
            self.wait_for_transfers()

        # ── Rust hook or Python fallback path ──

        # If pinned slab was used, split it into per-call views matching
        # the same interface the consumer expects (list of 2D tensors).
        if self._pinned_slab is not None and self._pinned_slab_idx > 0:
            slab = self._pinned_slab[:self._pinned_slab_idx]
            # Reconstruct per-o_proj-call views from the contiguous slab.
            # The fallback list may also contain entries (e.g. if slab overflowed).
            o_inputs = self._minimal_o_inputs  # may have fallback entries
            if len(o_inputs) == 0:
                # Pure slab path: split into views using _slab_offsets
                o_inputs = []
                for start, end in self._slab_offsets:
                    o_inputs.append(slab[start:end])
            self._pinned_slab_idx = 0
            self._slab_offsets = []
        else:
            o_inputs = self._minimal_o_inputs

        # Get scales from Rust hook (fast path) or Python list (fallback).
        # Rust hook does torch.cat + .cpu() + .numpy() internally via PyO3
        # (avoids returning 28K Python tensor objects for C++/Python cat).
        hook = _capture_hook
        if hook is not None:
            scales, count = hook.drain_scales()  # returns (numpy_array, count)
        else:
            # Python fallback: normalize multi-element scales then stack.
            count = self._minimal_call_count
            if self._minimal_scales:
                reduced = []
                for s in self._minimal_scales:
                    reduced.append(s.max() if s.numel() > 1 else s.reshape(()))
                scales = torch.stack(reduced).cpu().numpy()
            else:
                import numpy as np
                scales = np.array([], dtype=np.float32)

        self._minimal_o_inputs = []
        self._minimal_scales = []
        self._minimal_call_count = 0
        return o_inputs, scales, count

    def init_pinned_slab(self, hidden_dim: int, capacity_rows: int = 32768):
        """Allocate a pinned CPU slab for o_proj D2H copies.

        Call once after configure_from_model(). The slab eliminates per-call
        CPU tensor allocation: each o_proj D2H copies directly into a slice
        of this pre-allocated pinned buffer.
        """
        self._pinned_slab = torch.empty(
            capacity_rows, hidden_dim, dtype=torch.int8, pin_memory=True,
        )
        self._pinned_slab_idx = 0
        self._slab_offsets: List[tuple] = []  # (start_row, end_row) per o_proj call
        logger.info(
            "verilm: pinned slab initialized (%d rows × %d, %.1f MB, pid=%d)",
            capacity_rows, hidden_dim,
            capacity_rows * hidden_dim / 1e6,
            os.getpid(),
        )

    def _slab_copy_o_proj(self, a: torch.Tensor):
        """Copy o_proj activation into pinned slab. No tensor allocation."""
        batch_sz = a.shape[0]
        idx = self._pinned_slab_idx
        end = idx + batch_sz

        # Grow if needed (rare: only if request exceeds initial capacity).
        if end > self._pinned_slab.shape[0]:
            new_cap = max(self._pinned_slab.shape[0] * 2, end + 1024)
            new_slab = torch.empty(
                new_cap, self._pinned_slab.shape[1],
                dtype=torch.int8, pin_memory=True,
            )
            if idx > 0:
                new_slab[:idx].copy_(self._pinned_slab[:idx])
            self._pinned_slab = new_slab
            logger.info("verilm: pinned slab grown to %d rows", new_cap)

        target = self._pinned_slab[idx:end]

        if self._sync_mode == "event" and self._copy_stream is not None:
            self._copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._copy_stream):
                target.copy_(a, non_blocking=True)
                self._sync_event.record()
        else:
            target.copy_(a, non_blocking=True)

        self._transfers_pending = True
        self._pinned_slab_idx = end
        self._slab_offsets.append((idx, end))

    def reset_counter(self):
        """Reset the call counter to zero (native, Rust hook, and Python global).

        Must be called before each inference request to realign the
        call-counting layer/proj identification. Without this, any
        extra cutlass_scaled_mm calls between requests (warmup, vLLM
        internal prefill scheduling, chunked prefill) cause permanent
        counter misalignment — the wrapper reads shape[0] from the
        wrong projection and derives incorrect batch sizes.
        """
        global _call_counter
        _call_counter = 0
        if _native_capture is not None:
            _native_capture.reset_counter()
        if _capture_hook is not None:
            _capture_hook.reset_counter()

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

    def wait_for_transfers(self):
        """Wait for all pending non-blocking D2H transfers to complete.

        Event mode: synchronizes on the last recorded event (dedicated
        copy stream only — does not stall unrelated GPU work).

        Global mode: torch.cuda.synchronize() (stalls entire device).
        """
        if not self._transfers_pending:
            return

        if self._sync_mode == "event" and self._sync_event is not None:
            self._sync_event.synchronize()
        else:
            torch.cuda.synchronize()

        self._transfers_pending = False

    def stats(self) -> dict:
        return {
            "buffered": len(self._entries),
            "total_captured": self.total_captured,
            "enabled": self.enabled,
            "sync_mode": self._sync_mode,
            "transfers_pending": self._transfers_pending,
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

_capture_mode = "minimal"


def _wrapped_cutlass_scaled_mm(
    a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None
):
    """
    Wrapper around cutlass_scaled_mm that captures inputs for verification.

    In "minimal" mode with Rust CaptureHook: counter/modulus/scale-storage
    handled in native code (~2-3µs vs ~9µs in Python bytecode).
    In "full" mode: captures x_i8 + acc_i32 (via _int_mm) for all projections.
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

    # ── Rust hook path (minimal mode) ──
    # CaptureHook handles counter, modulus, scale append in native Rust code.
    # Slab copy stays in Python (requires PyTorch tensor ops).
    # At drain time, hook.drain_scales() does torch.cat + .cpu() + .numpy()
    # internally via PyO3 and returns a numpy array directly.
    hook = _capture_hook
    if hook is not None:
        buf = _capture_buffer
        if not buf.enabled:
            return output
        is_o_proj = hook.record(scale_a)
        if is_o_proj:
            if buf._pinned_slab is not None:
                buf._slab_copy_o_proj(a)
            else:
                buf._minimal_o_inputs.append(buf.copy_to_cpu(a))
        return output

    # ── Python fallback (full mode, or Rust hook unavailable) ──
    buf = _capture_buffer
    if not buf.enabled:
        return output

    # Derive layer/proj index from call count (integer arithmetic only).
    if _configured:
        idx = _call_counter % _calls_per_fwd
        layer = idx // PROJS_PER_LAYER
        proj_idx = idx % PROJS_PER_LAYER
    else:
        layer = None
        proj_idx = -1

    _call_counter += 1

    # Python fallback for minimal mode (no Rust hook).
    buf._minimal_scales.append(scale_a)
    buf._minimal_call_count += 1

    if proj_idx == _O_PROJ_IDX:
        if buf._pinned_slab is not None:
            buf._slab_copy_o_proj(a)
        else:
            buf._minimal_o_inputs.append(buf.copy_to_cpu(a))

    buf.total_captured += 1
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

    Env vars:
        VERILM_CAPTURE: "0" to disable capture entirely.
        VERILM_SYNC_MODE: "global" (default) or "event" (dedicated copy stream).
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

    # Sync mode: "global" (torch.cuda.synchronize) or "event" (dedicated copy stream).
    sync_env = os.environ.get("VERILM_SYNC_MODE", "global")
    _capture_buffer.set_sync_mode(sync_env)

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
        "verilm: wrapped cutlass_scaled_mm (buffer max=%d)",
        _capture_buffer._max_entries,
    )
