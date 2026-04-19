"""
Deterministic attention hook for verified-attention mode.

Replaces vLLM's FlashAttention with the custom deterministic CUDA kernel
during decode (seqlen_q=1). The kernel produces bit-exact f32 output
matching the CPU Rust reference, enabling exact verification.

Arithmetic contract: FROZEN (2026-04-19). See docs/design/deterministic-attention-spec.md.
Do NOT modify the kernel arithmetic. Only adapt plumbing around it.

Usage:
    VERILM_ATTN_MODE=deterministic vllm serve ...

    Or programmatically:
        from verilm.det_attn import DeterministicAttentionHook
        hook = DeterministicAttentionHook()
        hook.install(model)
"""

import ctypes
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("verilm")


class DeterministicAttentionHook:
    """Hooks vLLM attention modules to use the deterministic CUDA kernel.

    Decode only (batch_size=1, seqlen_q=1). Prefill passes through to
    the stock attention backend unchanged.

    The hook captures post-RoPE Q, K, V for each layer and stores them
    for commitment and verification.
    """

    def __init__(self, lib_path: Optional[str] = None):
        self._handles: List = []
        self._lib = None
        self._lib_path = lib_path
        self.enabled = True

        # Per-token captures: list of (layer_idx, q_bf16, k_bf16, v_bf16, output_f32)
        # Drained after each forward pass.
        self.captures: List[dict] = []

        # Model geometry — set by install().
        self._n_q_heads = 0
        self._n_kv_heads = 0
        self._d_head = 0
        self._n_layers = 0

    def install(self, model) -> int:
        """Install forward hooks on all attention modules.

        Returns the number of hooks installed.
        """
        self._load_kernel_lib()

        # Extract model geometry.
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise RuntimeError("Model has no config — cannot determine attention geometry")

        self._n_q_heads = getattr(cfg, "num_attention_heads", 0)
        self._n_kv_heads = getattr(cfg, "num_key_value_heads", self._n_q_heads)
        self._d_head = getattr(cfg, "head_dim", 0)
        if self._d_head == 0 and self._n_q_heads > 0:
            hidden = getattr(cfg, "hidden_size", 0)
            self._d_head = hidden // self._n_q_heads

        if self._n_q_heads == 0 or self._d_head == 0:
            raise RuntimeError(
                f"Cannot determine attention geometry: "
                f"n_q_heads={self._n_q_heads}, d_head={self._d_head}"
            )

        # Precompute inv_sqrt_d — frozen, same bits to GPU and CPU.
        self._inv_sqrt_d = np.float32(
            1.0 / np.sqrt(np.float32(self._d_head))
        )

        installed = 0
        for name, mod in model.named_modules():
            # vLLM attention modules: look for the Attention wrapper
            # that contains .impl (the backend implementation).
            # Pattern: model.layers.N.self_attn.attn
            if not hasattr(mod, "impl"):
                continue
            impl_name = type(mod.impl).__name__.lower()
            if "attention" not in type(mod).__name__.lower():
                continue

            # Extract layer index from name.
            layer_idx = self._parse_layer_idx(name)
            if layer_idx is None:
                continue

            h = mod.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            self._handles.append(h)
            installed += 1

            if installed == 1:
                logger.info(
                    "verilm: deterministic attention — "
                    "n_q=%d, n_kv=%d, d_head=%d, inv_sqrt_d=%.8e (0x%08x)",
                    self._n_q_heads, self._n_kv_heads, self._d_head,
                    self._inv_sqrt_d,
                    self._inv_sqrt_d.view(np.uint32).item(),
                )

        self._n_layers = installed
        if installed == 0:
            raise RuntimeError(
                "No attention modules found — cannot install deterministic hook. "
                "Ensure model uses vLLM's Attention module with .impl backend."
            )

        logger.info(
            "verilm: installed deterministic attention hooks on %d layers",
            installed,
        )
        return installed

    def _load_kernel_lib(self):
        """Load the deterministic attention CUDA shared library."""
        if self._lib is not None:
            return

        # Search order: explicit path, bundled in package, /opt (Modal).
        search_paths = []
        if self._lib_path:
            search_paths.append(self._lib_path)

        pkg_dir = Path(__file__).parent
        search_paths.extend([
            str(pkg_dir / "libdet_attn.so"),
            str(pkg_dir.parent.parent / "kernels" / "libdet_attn.so"),
            "/opt/libdet_attn.so",
        ])

        for path in search_paths:
            if os.path.exists(path):
                self._lib = ctypes.CDLL(path)
                self._setup_ctypes()
                logger.info("verilm: loaded deterministic attention kernel from %s", path)
                return

        raise RuntimeError(
            f"Deterministic attention kernel not found. Searched: {search_paths}. "
            f"Compile with: nvcc -O2 --fmad=false -shared -Xcompiler -fPIC "
            f"-o libdet_attn.so deterministic_attention.cu"
        )

    def _setup_ctypes(self):
        """Set up ctypes signatures for the CUDA kernel."""
        self._lib.deterministic_attention_host.restype = ctypes.c_int
        self._lib.deterministic_attention_host.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),  # q_host
            ctypes.POINTER(ctypes.c_uint16),  # k_host
            ctypes.POINTER(ctypes.c_uint16),  # v_host
            ctypes.POINTER(ctypes.c_float),   # output_host
            ctypes.POINTER(ctypes.c_float),   # weights_host
            ctypes.c_int,                     # n_q_heads
            ctypes.c_int,                     # n_kv_heads
            ctypes.c_int,                     # d_head
            ctypes.c_int,                     # seq_len
            ctypes.c_float,                   # inv_sqrt_d
        ]

    @staticmethod
    def _parse_layer_idx(module_name: str) -> Optional[int]:
        """Extract layer index from module name like 'model.layers.5.self_attn.attn'."""
        parts = module_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def _make_attn_hook(self, layer_idx: int):
        """Create a forward hook for a specific attention layer.

        The hook intercepts the attention module's output during decode
        (batch=1) and replaces it with the deterministic kernel's output.

        vLLM's Attention.forward() signature (v1):
            forward(query, key, value, kv_cache, attn_metadata, ...) -> output

        For decode with paged attention:
            - query: [batch=1, n_q_heads * d_head] (post-RoPE)
            - key: [batch=1, n_kv_heads * d_head] (post-RoPE, current token)
            - value: [batch=1, n_kv_heads * d_head] (current token)
            - kv_cache: paged KV cache tensor
            - output: [batch=1, n_q_heads * d_head]

        We need all K,V for all seq positions. The current token's K,V
        are in the function args. Historical K,V are in the KV cache.
        We extract both and run our kernel.
        """
        def hook(module, args, output):
            if not self.enabled:
                return output

            # args: (query, key, value, kv_cache, attn_metadata, ...)
            if len(args) < 5:
                return output  # unexpected signature, pass through

            query, key, value, kv_cache, attn_metadata = args[:5]

            # Only intercept decode (batch=1, single query token).
            # During prefill, batch_size > 1 — pass through to stock attention.
            if query.shape[0] != 1:
                return output

            try:
                det_output, capture_data = self._run_deterministic(
                    layer_idx, query, key, value, kv_cache, attn_metadata,
                )
                self.captures.append(capture_data)
                return det_output
            except Exception as e:
                # Fail closed: do not silently fall back to stock attention.
                raise RuntimeError(
                    f"Deterministic attention failed on layer {layer_idx}: {e}. "
                    f"Verified-attention mode requires exact computation — "
                    f"no silent fallback to stock attention."
                ) from e

        return hook

    def _run_deterministic(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> Tuple[torch.Tensor, dict]:
        """Run deterministic attention kernel and return output + capture data.

        Extracts full K,V history from KV cache + current token,
        converts to bf16 numpy arrays, runs CUDA kernel, returns output
        as a torch tensor matching the expected output shape/dtype.
        """
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        d = self._d_head

        # --- Extract Q (current token, post-RoPE) ---
        # query shape: [1, n_q_heads * d_head], dtype depends on model (bf16/f16)
        q_flat = query.squeeze(0).contiguous().to(torch.bfloat16).cpu().numpy()
        q_bf16 = q_flat.view(np.uint16)  # bf16 as raw u16 bits

        # --- Extract full K, V from KV cache ---
        # vLLM paged attention stores K,V in a paged block table.
        # attn_metadata contains the block table and sequence lengths.
        # We need to reconstruct the full [seq_len, n_kv_heads, d_head] K and V.
        k_all, v_all, seq_len = self._extract_kv_from_cache(
            key, value, kv_cache, attn_metadata,
        )

        # k_all: [seq_len * n_kv * d] as u16, v_all: same
        k_bf16 = k_all
        v_bf16 = v_all

        # --- Run deterministic CUDA kernel ---
        output_f32 = np.zeros(n_q * d, dtype=np.float32)
        weights_f32 = np.zeros(n_q * seq_len, dtype=np.float32)

        rc = self._lib.deterministic_attention_host(
            q_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            k_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            v_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            output_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weights_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_q, n_kv, d, seq_len,
            float(self._inv_sqrt_d),
        )
        if rc != 0:
            raise RuntimeError(f"Deterministic CUDA kernel returned error code {rc}")

        # --- Convert output back to model dtype and device ---
        # Output: [1, n_q_heads * d_head] on the same device as query
        out_tensor = torch.from_numpy(output_f32).to(
            dtype=query.dtype, device=query.device,
        ).unsqueeze(0)

        # --- Capture data for commitment/verification ---
        capture_data = {
            "layer": layer_idx,
            "seq_len": seq_len,
            "q_bf16": q_bf16.copy(),
            "k_bf16": k_bf16.copy(),
            "v_bf16": v_bf16.copy(),
            "output_f32_bits": output_f32.view(np.uint32).copy(),
            "weight_f32_bits": weights_f32.view(np.uint32).copy(),
        }

        return out_tensor, capture_data

    def _extract_kv_from_cache(
        self,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Extract full K,V history from vLLM's paged KV cache.

        Returns (k_bf16_flat, v_bf16_flat, seq_len) where:
          k_bf16_flat: np.ndarray of uint16, shape [seq_len * n_kv * d]
          v_bf16_flat: same for V
          seq_len: total sequence length (including current token)

        The current token's K,V (from function args) are appended as the
        last position since the KV cache may not yet include them.
        """
        n_kv = self._n_kv_heads
        d = self._d_head

        # vLLM paged attention: kv_cache is [num_blocks, 2, block_size, n_kv_heads, d_head]
        # or split into separate k_cache, v_cache tensors depending on version.
        #
        # attn_metadata has:
        #   - block_tables: [batch, max_blocks] — block indices for each sequence
        #   - seq_lens or seq_lens_tensor: actual sequence lengths
        #
        # For decode (batch=1), we have one sequence.

        # Get sequence length (excluding current token which hasn't been cached yet).
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None:
            seq_lens_tensor = getattr(attn_metadata, "seq_lens_tensor", None)
            if seq_lens_tensor is not None:
                seq_lens = seq_lens_tensor.cpu().tolist()

        if seq_lens is None or len(seq_lens) == 0:
            raise RuntimeError("Cannot determine sequence length from attn_metadata")

        # seq_lens[0] includes the current token being processed.
        total_seq_len = int(seq_lens[0])
        cached_len = total_seq_len - 1  # positions already in cache

        # Get block table for this sequence.
        block_tables = getattr(attn_metadata, "block_tables", None)
        if block_tables is None:
            raise RuntimeError("Cannot find block_tables in attn_metadata")

        block_table = block_tables[0].cpu()  # [max_blocks]

        # Determine KV cache layout.
        # Common vLLM layout: kv_cache is a tuple (k_cache, v_cache) or
        # a single tensor [num_blocks, 2, block_size, n_kv_heads, d_head].
        if isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 2:
            k_cache, v_cache = kv_cache
        elif isinstance(kv_cache, torch.Tensor):
            if kv_cache.dim() == 5 and kv_cache.shape[1] == 2:
                k_cache = kv_cache[:, 0]  # [num_blocks, block_size, n_kv, d]
                v_cache = kv_cache[:, 1]
            else:
                raise RuntimeError(
                    f"Unexpected kv_cache shape: {kv_cache.shape}. "
                    f"Expected [num_blocks, 2, block_size, n_kv, d_head]"
                )
        else:
            raise RuntimeError(f"Unexpected kv_cache type: {type(kv_cache)}")

        block_size = k_cache.shape[1]

        # Gather cached K,V from blocks.
        k_positions = []
        v_positions = []
        for pos in range(cached_len):
            block_idx = int(block_table[pos // block_size])
            offset = pos % block_size
            k_positions.append(k_cache[block_idx, offset])  # [n_kv, d]
            v_positions.append(v_cache[block_idx, offset])

        # Append current token's K,V.
        # current_key shape: [1, n_kv * d], dtype bf16/f16
        cur_k = current_key.squeeze(0).view(n_kv, d)
        cur_v = current_value.squeeze(0).view(n_kv, d)
        k_positions.append(cur_k)
        v_positions.append(cur_v)

        # Stack to [seq_len, n_kv, d] and convert to bf16 numpy.
        k_full = torch.stack(k_positions).contiguous().to(torch.bfloat16).cpu()
        v_full = torch.stack(v_positions).contiguous().to(torch.bfloat16).cpu()

        k_bf16 = k_full.view(-1).numpy().view(np.uint16)
        v_bf16 = v_full.view(-1).numpy().view(np.uint16)

        return k_bf16, v_bf16, total_seq_len

    def drain(self) -> List[dict]:
        """Return and clear all captured attention data for this forward pass."""
        result = self.captures
        self.captures = []
        return result

    def remove(self):
        """Remove all installed hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._n_layers = 0

    @property
    def mode_name(self) -> str:
        return "deterministic"

    @property
    def geometry(self) -> dict:
        return {
            "n_q_heads": self._n_q_heads,
            "n_kv_heads": self._n_kv_heads,
            "d_head": self._d_head,
            "inv_sqrt_d": float(self._inv_sqrt_d),
            "inv_sqrt_d_bits": int(self._inv_sqrt_d.view(np.uint32).item()),
        }
