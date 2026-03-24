"""
Canonical sampler hook for verified sampled decoding.

Installs a PyTorch forward hook on vLLM's LogitsProcessor module to run
the protocol's canonical sampler (via the Rust verilm_rs bindings) and
force the chosen token by masking all other logits to -inf. This makes
vLLM's internal sampler a no-op — the token was already decided by VeriLM.

The canonical sampler pipeline:
    1. Temperature scaling
    2. Top-k filtering
    3. Softmax (f64)
    4. Top-p (nucleus) filtering
    5. ChaCha20Rng seeded per-token → categorical sample

Per-token seed: SHA256("vi-sample-v1" || batch_seed || token_index_le32)

Why a forward hook instead of SamplingParams.logits_processors?
vLLM v0.18+ uses msgspec.Struct for SamplingParams, which does not
accept a logits_processors kwarg. The forward hook works across versions.
"""

import logging

import torch

logger = logging.getLogger("verilm.sampler")


class CanonicalSamplerHook:
    """PyTorch forward hook that forces canonical sampled tokens.

    Installed on vLLM's LogitsProcessor module. When activated for a request,
    intercepts the logits output, runs the Rust canonical sampler, and masks
    everything except the chosen token to -inf. vLLM's sampler then
    deterministically picks the only surviving token.

    Lifecycle:
        hook = CanonicalSamplerHook()
        hook.install(model)          # once, at server startup
        hook.activate(seed, ...)     # before each generate()
        llm.generate(...)            # hook fires per decode step
        hook.deactivate()            # after generate()
    """

    def __init__(self):
        self._handle = None
        self._active = False
        self._batch_seed: bytes = b""
        self._temperature: float = 0.0
        self._top_k: int = 0
        self._top_p: float = 1.0
        self._call_count: int = 0

    def install(self, model) -> bool:
        """Install forward hook on the LogitsProcessor module.

        Must be installed AFTER capture hooks so that capture sees the
        original (unmasked) logits. This hook returns a cloned tensor
        to avoid race conditions with async D2H transfers from capture.

        Returns True if the hook was installed.
        """
        for name, mod in model.named_modules():
            if name == "logits_processor" or type(mod).__name__ == "LogitsProcessor":
                self._handle = mod.register_forward_hook(self._hook)
                logger.info("verilm: installed canonical sampler hook on %s", name)
                return True
        logger.warning("verilm: could not find LogitsProcessor for sampler hook")
        return False

    def activate(
        self,
        batch_seed: bytes,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        """Activate the canonical sampler for a new request."""
        assert len(batch_seed) == 32
        self._batch_seed = batch_seed
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._call_count = 0
        self._active = True

    def deactivate(self):
        """Deactivate after generate() completes."""
        self._active = False

    def remove(self):
        """Remove the hook entirely."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, output):
        """Forward hook: intercept logits and force canonical token.

        Returns a new (cloned) tensor with all logits set to -inf except
        the canonically chosen token. The original tensor is left unmodified
        so any prior capture hooks that initiated async D2H copies see the
        true logits.
        """
        if not self._active:
            return None  # pass through unchanged

        import verilm_rs

        token_index = self._call_count
        self._call_count += 1

        # Derive per-token seed via Rust (SHA256 domain-separated).
        token_seed = verilm_rs.derive_token_seed(self._batch_seed, token_index)

        # Extract the logits row being sampled (last row).
        # Prefill: output is (num_tokens, vocab_size), last row → first gen token.
        # Decode:  output is (1, vocab_size).
        if output.dim() == 2:
            sample_logits = output[-1]
        else:
            sample_logits = output

        # Run canonical sampler on CPU float32 — same code the verifier uses.
        logits_list = sample_logits.float().cpu().tolist()
        chosen = verilm_rs.canonical_sample(
            logits_list,
            self._temperature,
            self._top_k,
            self._top_p,
            token_seed,
        )

        # Clone to avoid mutating the tensor that capture hooks may be
        # async-copying to CPU.
        masked = output.clone()
        if masked.dim() == 2:
            masked[-1].fill_(float("-inf"))
            masked[-1, chosen] = 0.0
        else:
            masked.fill_(float("-inf"))
            masked[chosen] = 0.0

        return masked
