"""
PyTorch hooks for capturing data outside the cutlass_scaled_mm path.

Captures:
  - Input embeddings (token embedding lookup output)
  - Logits (LogitsProcessor output)
  - LP hidden (LogitsProcessor input hidden state — decode boundary)

These are needed for R_T but aren't INT8 matmuls, so they don't go
through the cutlass_scaled_mm wrapper in capture.py.

Note: vLLM's ParallelLMHead.forward() raises RuntimeError by design —
logits are computed via LogitsProcessor, not lm_head.forward(). We hook
LogitsProcessor instead.
"""

import logging
from typing import List, Optional

import torch

logger = logging.getLogger("verilm")


class EmbeddingLogitCapture:
    """Captures embeddings, logits, and pre-RMSNorm residuals via PyTorch forward hooks.

    Install on a vLLM model after loading. Captures are accumulated
    per-request and drained together with the matmul captures.
    """

    def __init__(self):
        self.embeddings: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []
        self.residuals: List[torch.Tensor] = []
        self._handles: list = []
        self.enabled = True

    def install(self, model) -> int:
        """Install hooks on embedding, LogitsProcessor, and RMSNorm modules.

        Returns the number of hooks installed.
        """
        installed = 0

        for name, mod in model.named_modules():
            # Embedding layer: usually model.embed_tokens or model.model.embed_tokens
            if name.endswith("embed_tokens"):
                h = mod.register_forward_hook(self._make_embed_hook())
                self._handles.append(h)
                installed += 1
                logger.info("verilm: installed embedding hook on %s", name)

            # LogitsProcessor: vLLM computes logits through this module,
            # not through lm_head.forward() (which raises RuntimeError).
            if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
                h = mod.register_forward_hook(self._make_logit_hook())
                self._handles.append(h)
                installed += 1
                logger.info("verilm: installed logit hook on %s (%s)", name, type(mod).__name__)

            # Pre-attention RMSNorm (input_layernorm): capture input = residual stream.
            # The input to RMSNorm IS the pre-normalization residual connection value.
            if "input_layernorm" in name and "post" not in name:
                h = mod.register_forward_hook(self._make_residual_hook())
                self._handles.append(h)
                installed += 1
                logger.info("verilm: installed residual hook on %s", name)

        return installed

    def _make_embed_hook(self):
        def hook(module, args, output):
            if self.enabled:
                self.embeddings.append(output.detach().to("cpu", non_blocking=True))
        return hook

    def _make_logit_hook(self):
        def hook(module, args, output):
            if self.enabled:
                self.logits.append(output.detach().to("cpu", non_blocking=True))
        return hook

    def _make_residual_hook(self):
        """Hook that captures the INPUT to RMSNorm (= pre-attention residual stream)."""
        def hook(module, args, output):
            if self.enabled and len(args) > 0:
                self.residuals.append(args[0].detach().to("cpu", non_blocking=True))
        return hook

    def drain(self):
        """Return and clear all captured embeddings, logits, and residuals."""
        result = {
            "embeddings": self.embeddings,
            "logits": self.logits,
            "residuals": self.residuals,
        }
        self.embeddings = []
        self.logits = []
        self.residuals = []
        return result

    def remove(self):
        """Remove all installed hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


class LPHiddenCapture:
    """Captures the hidden state at LogitsProcessor input per forward pass.

    This is the actual decode boundary: the pruned hidden state that
    LogitsProcessor receives as args[1], shape (batch, hidden_dim), bf16.

    The verifier uses this for exact token-identity verification via
    bf16 lm_head matmul instead of the i8→i32 quantized path.

    Empirical result: 192/192 exact on greedy decode (Qwen W8A8).
    See docs/design/decode-boundary-spike.md.

    Synchronous D2H: 7 KB/token is too small for async to matter, and
    async D2H risks the caching allocator recycling the GPU snapshot
    before the copy stream finishes reading (observed as exactly 2
    zeroed bf16 elements per affected capture).
    """

    def __init__(self):
        self.hiddens: List[torch.Tensor] = []
        self._handle = None
        self.enabled = True

    def install(self, model) -> bool:
        """Install forward hook on LogitsProcessor.

        Returns True if the hook was installed.
        """
        for name, mod in model.named_modules():
            if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
                self._handle = mod.register_forward_hook(self._hook)
                logger.info("verilm: installed LP hidden hook on %s (%s)", name, type(mod).__name__)
                return True
        logger.warning("verilm: could not find LogitsProcessor for LP hidden capture")
        return False

    def _hook(self, module, args, output):
        if self.enabled and len(args) >= 2 and isinstance(args[1], torch.Tensor):
            # args[1] is hidden_states, shape (batch, hidden_dim), dtype bf16.
            # Clone on GPU then synchronous D2H. Preserves native dtype.
            cpu_t = args[1].detach().contiguous().clone().cpu()
            self.hiddens.append(cpu_t)

    def wait_for_transfers(self):
        """No-op — D2H is synchronous. Kept for API compat with server.py."""
        pass

    def drain(self) -> List[torch.Tensor]:
        """Return and clear all captured LP hidden states."""
        result = self.hiddens
        self.hiddens = []
        return result

    def remove(self):
        """Remove the installed hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class FinalResidualCapture:
    """Captures the input to model.norm (final RMSNorm) per forward pass.

    This is the pre-normalization residual stream at the exact boundary
    between the approximate attention stack and the exact tail
    (RMSNorm -> lm_head -> sampling). The verifier uses this for exact
    token verification instead of the shell-replayed final hidden state.

    Uses a dedicated CUDA stream + event for non_blocking D2H copies,
    avoiding device-wide torch.cuda.synchronize() at drain time.
    """

    def __init__(self):
        self.residuals: List[torch.Tensor] = []
        self._handle = None
        self.enabled = True
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._sync_event: Optional[torch.cuda.Event] = None
        self._transfers_pending = False

    def install(self, model) -> bool:
        """Install forward hook on the final RMSNorm (model.norm).

        Returns True if the hook was installed.
        """
        for name, mod in model.named_modules():
            if name == "model.norm":
                self._handle = mod.register_forward_hook(self._hook)
                logger.info("verilm: installed final residual hook on %s", name)
                # Initialize dedicated copy stream + event for async D2H.
                if torch.cuda.is_available():
                    self._copy_stream = torch.cuda.Stream()
                    self._sync_event = torch.cuda.Event()
                return True
        logger.warning("verilm: could not find model.norm for final residual capture")
        return False

    def _hook(self, module, args, output):
        if self.enabled and len(args) > 0:
            t = args[0].detach().float()
            if self._copy_stream is not None:
                self._copy_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._copy_stream):
                    cpu_t = t.to("cpu", non_blocking=True)
                    self._sync_event.record()
            else:
                cpu_t = t.to("cpu", non_blocking=True)
            self.residuals.append(cpu_t)
            self._transfers_pending = True

    def wait_for_transfers(self):
        """Block until all non_blocking D2H copies are complete."""
        if self._transfers_pending and self._sync_event is not None:
            self._sync_event.synchronize()
        self._transfers_pending = False

    def drain(self) -> List[torch.Tensor]:
        """Return and clear all captured final residuals."""
        self.wait_for_transfers()
        result = self.residuals
        self.residuals = []
        return result

    def remove(self):
        """Remove the installed hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
