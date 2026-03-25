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
import os

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

    def __init__(self, minimal: bool = False):
        self._handle = None
        self._active = False
        self._batch_seed: bytes = b""
        self._temperature: float = 0.0
        self._top_k: int = 0
        self._top_p: float = 1.0
        self._call_count: int = 0
        # In minimal mode, no logit capture hook is installed, so we can
        # mask logits in-place (no clone needed to protect async D2H).
        self._minimal = minimal

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

    def _mask_logits(self, output, chosen):
        """Mask all logits to -inf except the chosen token.

        In minimal mode (no logit capture hook), modifies in-place.
        In full mode, clones first to protect async D2H from capture hooks.
        """
        if self._minimal:
            target = output
        else:
            target = output.clone()

        if target.dim() == 2:
            target[-1].fill_(float("-inf"))
            target[-1, chosen] = 0.0
        else:
            target.fill_(float("-inf"))
            target[chosen] = 0.0

        return target

    def _hook(self, module, args, output):
        """Forward hook: intercept logits and force canonical token.

        Greedy (temperature=0): GPU argmax, no CPU round-trip.
        Sampled: transfer logits to CPU as numpy bytes, sample in Rust.
        """
        if not self._active:
            return None  # pass through unchanged

        import verilm_rs

        timers = os.environ.get("VERILM_COMMIT_TIMERS", "0") == "1"
        if timers:
            import time
            t0 = time.monotonic()

        token_index = self._call_count
        self._call_count += 1

        # Extract the logits row being sampled (last row).
        if output.dim() == 2:
            sample_logits = output[-1]
        else:
            sample_logits = output

        # Greedy fast path: argmax on GPU, skip entire CPU round-trip.
        if self._temperature == 0.0:
            chosen = int(sample_logits.argmax().item())

            if timers:
                t_sample = time.monotonic()

            result = self._mask_logits(output, chosen)

            if timers:
                t_end = time.monotonic()
                if token_index < 3 or token_index % 64 == 0:
                    logger.info(
                        "verilm sampler [greedy]: argmax=%.3fms mask=%.3fms "
                        "total=%.3fms (token %d)",
                        (t_sample - t0) * 1000,
                        (t_end - t_sample) * 1000,
                        (t_end - t0) * 1000,
                        token_index,
                    )
            return result

        # Sampled decoding: CPU path via Rust canonical sampler.
        if timers:
            t_seed_start = time.monotonic()

        token_seed = verilm_rs.derive_token_seed(self._batch_seed, token_index)

        if timers:
            t_seed = time.monotonic()

        # Transfer logits to CPU as numpy array. extract_f32_vec on the Rust
        # side reads the buffer protocol directly — no .tolist() needed.
        logits_np = sample_logits.float().cpu().numpy()

        if timers:
            t_cpu = time.monotonic()

        chosen = verilm_rs.canonical_sample(
            logits_np,
            self._temperature,
            self._top_k,
            self._top_p,
            token_seed,
        )

        if timers:
            t_sample = time.monotonic()

        result = self._mask_logits(output, chosen)

        if timers:
            t_end = time.monotonic()
            if token_index < 3 or token_index % 64 == 0:
                logger.info(
                    "verilm sampler: seed=%.3fms cpu+sample=%.3fms "
                    "mask=%.3fms total=%.3fms (token %d)",
                    (t_seed - t_seed_start) * 1000,
                    (t_sample - t_seed) * 1000,
                    (t_end - t_sample) * 1000,
                    (t_end - t0) * 1000,
                    token_index,
                )
        return result
