"""
Verified inference HTTP server.

Serves /chat and /audit endpoints. Uses vLLM for inference, verilm for
capture, and verilm_rs (PyO3) for Rust commitment engine.

Architecture:
    User → HTTP → this server → vLLM (GPU) + verilm_rs (commitment)

Direct in-process integration — no separate server, no JSON serialization of captures, no HTTP hop.

Usage:
    # As a module (for Modal or direct deployment):
    from verilm.server import create_app
    app = create_app(llm, model)

    # Standalone:
    python -m verilm.server --model neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8
"""

import hashlib
import json
import logging
import os
import secrets
import time
from typing import Dict, List, Optional

logger = logging.getLogger("verilm.server")


class VerifiedInferenceServer:
    """Verified inference: capture + commit + audit in one process."""

    def __init__(self, llm, *, max_audit_entries: int = 10_000, ttl_secs: int = 600):
        from vllm import LLM

        from . import capture as cap
        from .capture import (
            configure_from_model,
            get_capture_buffer,
            get_model_from_llm,
        )
        from .hooks import EmbeddingLogitCapture
        from .trace import build_layer_traces

        self.llm = llm
        self.ttl_secs = ttl_secs

        # Prefix caching must be disabled for verified inference.
        # When enabled, vLLM reuses KV cache blocks from prior requests with
        # matching prefixes, processing fewer tokens through cutlass_scaled_mm.
        # This causes batch_size mismatches in our capture counting.
        cache_cfg = getattr(llm.llm_engine, "cache_config", None)
        if cache_cfg and getattr(cache_cfg, "enable_prefix_caching", False):
            logger.warning(
                "verilm: prefix caching is enabled — this will cause capture "
                "mismatches on repeated prompts. Use enable_prefix_caching=False."
            )

        # Configure capture from model.
        model = get_model_from_llm(llm)
        configure_from_model(model)

        # Install embedding/logit hooks (skip in minimal mode — data unused).
        if cap._capture_mode != "minimal":
            self.el_capture = EmbeddingLogitCapture()
            n_hooks = self.el_capture.install(model)
            logger.info("Installed %d embedding/logit hooks", n_hooks)
        else:
            self.el_capture = None
            logger.info("Minimal mode: skipping embedding/logit hooks")

        # Install final residual capture (always, even in minimal mode).
        # Captures the pre-final-norm residual for exact LM-head verification.
        from .hooks import FinalResidualCapture
        self.final_res_capture = FinalResidualCapture()
        if not self.final_res_capture.install(model):
            logger.warning(
                "Could not install final residual capture — "
                "lm_head verification will use shell-replayed state"
            )
            self.final_res_capture = None

        # Install canonical sampler hook on LogitsProcessor module.
        # Must be AFTER capture hooks so capture sees original (unmasked) logits.
        from .sampler import CanonicalSamplerHook
        self.sampler_hook = CanonicalSamplerHook()
        if not self.sampler_hook.install(model):
            raise RuntimeError(
                "Could not install canonical sampler hook — "
                "LogitsProcessor module not found in model"
            )

        # Manifest hashes.
        self._tokenizer_hash = self._compute_tokenizer_hash(llm)
        self._model_dir = self._resolve_model_dir(llm)
        self._weight_hash = self._compute_weight_hash_rw(llm)
        self._quant_hash = self._compute_quant_hash(model)
        self._system_prompt_hash = hashlib.sha256(b"").hexdigest()  # Default: empty prompt

        self.buf = get_capture_buffer()
        self._audit_store: Dict[str, dict] = {}
        self._max_audit_entries = max_audit_entries

        # Pre-load weight provider once for all future V4 audits.
        self._weight_provider = None
        if self._model_dir is not None:
            try:
                import verilm_rs
                logger.info("Pre-loading WeightProvider from %s...", self._model_dir)
                self._weight_provider = verilm_rs.WeightProvider(self._model_dir)
                logger.info("WeightProvider ready")
            except Exception as e:
                logger.warning("Could not pre-load WeightProvider: %s", e)

        # Verify the cached WeightProvider matches the committed R_W.
        if self._weight_provider is not None:
            provider_rw = self._weight_provider.weight_hash_hex()
            if provider_rw != self._weight_hash:
                raise RuntimeError(
                    f"WeightProvider R_W mismatch: provider={provider_rw}, "
                    f"manifest={self._weight_hash}. Model identity not bound."
                )
            logger.info("WeightProvider R_W bound: %s", provider_rw)

    def _compute_tokenizer_hash(self, llm) -> str:
        """SHA-256 of tokenizer vocab, used in manifest."""
        try:
            tokenizer = llm.get_tokenizer()
            vocab = tokenizer.get_vocab()
            vocab_str = json.dumps(vocab, sort_keys=True)
            return hashlib.sha256(vocab_str.encode()).hexdigest()
        except Exception as e:
            logger.warning("Could not hash tokenizer: %s", e)
            return "00" * 32

    def _compute_weight_hash_rw(self, llm) -> str:
        """Compute the paper's R_W: weight-chain hash over all INT8 weights.

        Uses verilm_rs.compute_weight_hash() which loads the safetensors
        checkpoint and hashes all weight matrices in canonical order.
        This is the real R_W from the paper, not a fingerprint.
        """
        try:
            import verilm_rs

            if self._model_dir is None:
                logger.warning("Could not resolve model directory for R_W computation")
                return "00" * 32

            logger.info("Computing R_W (weight-chain hash) from %s...", self._model_dir)
            weight_hash = verilm_rs.compute_weight_hash(self._model_dir)
            logger.info("R_W: %s", weight_hash)
            return weight_hash
        except Exception as e:
            logger.warning("Could not compute R_W: %s", e)
            return "00" * 32

    @staticmethod
    def _resolve_model_dir(llm) -> Optional[str]:
        """Resolve the local filesystem path to the model's safetensors."""
        try:
            # vLLM stores the model path in model_config.model or
            # model_config.tokenizer (which points to the same dir).
            model_id = llm.llm_engine.model_config.model
            # If it's already a local path, use it directly.
            if os.path.isdir(model_id):
                return model_id
            # Otherwise it's a HuggingFace model ID — resolve via snapshot.
            from huggingface_hub import snapshot_download
            return snapshot_download(model_id)
        except Exception:
            return None

    def _compute_quant_hash(self, model) -> str:
        """SHA-256 of quantization config, if present."""
        try:
            cfg = getattr(model, "config", None)
            quant_cfg = getattr(cfg, "quantization_config", None)
            if quant_cfg is None:
                return "00" * 32
            if hasattr(quant_cfg, "to_dict"):
                quant_dict = quant_cfg.to_dict()
            elif isinstance(quant_cfg, dict):
                quant_dict = quant_cfg
            else:
                quant_dict = {"repr": repr(quant_cfg)}
            return hashlib.sha256(
                json.dumps(quant_dict, sort_keys=True, default=str).encode()
            ).hexdigest()
        except Exception as e:
            logger.warning("Could not hash quant config: %s", e)
            return "00" * 32

    def configure_system_prompt(self, system_prompt: str):
        """Set the system prompt hash for manifest inclusion."""
        self._system_prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()

    def chat(
        self,
        prompt: str,
        max_tokens: int = 4,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> dict:
        """Run verified inference: generate → capture → commit.

        Supports both greedy (temperature=0) and sampled decoding.
        Sampled decoding uses a canonical ChaCha20-based sampler injected
        as a vLLM logits processor, ensuring exact reproducibility by the
        verifier. vLLM's own sampling knobs are neutralized — the
        canonical processor is the sole policy owner.
        """
        import verilm_rs
        from vllm import SamplingParams

        from . import capture as cap

        # Clear buffers and reset call counter to realign layer/proj counting.
        # Without the counter reset, any extra cutlass_scaled_mm calls between
        # requests (warmup, chunked prefill residue) cause permanent misalignment.
        self.buf.drain()
        self.buf.reset_counter()
        if self.el_capture is not None:
            self.el_capture.drain()
        if self.final_res_capture is not None:
            self.final_res_capture.drain()

        # Fresh random batch seed per request. The commitment includes
        # seed_commitment = SHA256(batch_seed); the batch_seed itself is
        # revealed only at audit time, preventing pre-computation attacks.
        seed = secrets.token_bytes(32)

        _chat_timers = os.environ.get("VERILM_COMMIT_TIMERS", "0") == "1"
        if _chat_timers:
            _ct0 = time.monotonic()

        # Activate the canonical sampler hook for this request.
        # The hook intercepts LogitsProcessor output, runs the Rust canonical
        # sampler, and masks all but the chosen token to -inf. vLLM then does
        # greedy argmax on the masked logits — the token is already decided.
        self.sampler_hook.activate(seed, temperature, top_k, top_p)
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        try:
            outputs = self.llm.generate([prompt], params)
        finally:
            self.sampler_hook.deactivate()
        output = outputs[0]

        generated_text = output.outputs[0].text
        gen_token_ids = list(output.outputs[0].token_ids)
        prompt_token_ids = list(output.prompt_token_ids)

        if _chat_timers:
            _ct_gen = time.monotonic()

        # Ensure all non-blocking GPU→CPU transfers completed.
        # 1. Capture buffer: event-based sync (only waits on the last D2H transfer).
        self.buf.wait_for_transfers()
        # 2. Hook D2H copies (final_res_capture, el_capture) use non_blocking=True
        #    without dedicated events. A device-wide sync ensures they're complete.
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if _chat_timers:
            _ct_sync = time.monotonic()

        # Drain captures (tensors are already on CPU).
        captures = self.buf.drain()
        el_data = self.el_capture.drain() if self.el_capture is not None else {}
        final_residuals_raw = self.final_res_capture.drain() if self.final_res_capture is not None else []

        if _chat_timers:
            _ct_drain = time.monotonic()

        n_layers = cap._n_layers
        calls_per_fwd = n_layers * cap.PROJS_PER_LAYER

        # Diagnostic: always log capture counts for debugging mismatch issues.
        all_token_ids_dbg = prompt_token_ids + gen_token_ids
        logger.info(
            "verilm: captures=%d, calls_per_fwd=%d, prompt_tokens=%d, "
            "gen_tokens=%d, all_tokens=%d, call_counter=%d",
            len(captures), calls_per_fwd, len(prompt_token_ids),
            len(gen_token_ids), len(all_token_ids_dbg), cap._call_counter,
        )

        if len(captures) == 0 or len(captures) % calls_per_fwd != 0:
            raise RuntimeError(
                f"Capture count {len(captures)} not a multiple of "
                f"calls_per_fwd {calls_per_fwd} "
                f"(prompt_tokens={len(prompt_token_ids)}, gen_tokens={len(gen_token_ids)}, "
                f"call_counter={cap._call_counter})"
            )

        # Derive forward pass batch sizes from capture tensor shapes.
        n_fwd = len(captures) // calls_per_fwd
        # In minimal mode, only o_proj has a tensor (index 1 in each layer's 4 calls).
        # In full mode, all captures have tensors at index [2].
        if cap._capture_mode == "minimal":
            # o_proj is call index 1 within each layer's 4-call group.
            fwd_batch_sizes = [
                captures[i * calls_per_fwd + 1][2].shape[0] for i in range(n_fwd)
            ]
        else:
            fwd_batch_sizes = [captures[i * calls_per_fwd][2].shape[0] for i in range(n_fwd)]

        all_token_ids = prompt_token_ids + gen_token_ids
        n_tokens = sum(fwd_batch_sizes)
        expected_traces = len(all_token_ids) - 1

        # EOS trailing forward pass trim: when generation stops early (EOS
        # before max_tokens), vLLM's async scheduler has already dispatched
        # one more decode step before the EOS sampling result is checked.
        # That extra forward pass processes the last gen token to produce
        # EOS logits but does not correspond to an emitted transcript token.
        # Trim it so the proof aligns to the actual transcript.
        if (n_tokens == expected_traces + 1
                and len(gen_token_ids) < max_tokens
                and fwd_batch_sizes[-1] == 1):
            logger.info(
                "verilm: trimming trailing EOS forward pass "
                "(gen=%d < max=%d, n_fwd %d→%d)",
                len(gen_token_ids), max_tokens, n_fwd, n_fwd - 1,
            )
            captures = captures[:-calls_per_fwd]
            n_fwd -= 1
            fwd_batch_sizes = fwd_batch_sizes[:-1]
            if len(final_residuals_raw) == n_fwd + 1:
                final_residuals_raw = final_residuals_raw[:-1]
            n_tokens = sum(fwd_batch_sizes)

        if n_tokens != expected_traces:
            raise RuntimeError(
                f"Token count ({n_tokens}) does not match expected "
                f"({expected_traces}). "
                f"Cannot commit with mismatched transcript. "
                f"captures={len(captures)}, calls_per_fwd={calls_per_fwd}, "
                f"n_fwd={n_fwd}, fwd_batch_sizes={fwd_batch_sizes}, "
                f"prompt_tokens={len(prompt_token_ids)}, gen_tokens={len(gen_token_ids)}"
            )

        # Build manifest from actual request parameters.
        # Logit-modifying parameters are pinned to their canonical defaults;
        # the verifier rejects anything else (no repetition/frequency/presence
        # penalty, no logit bias, no guided decoding in canonical sampler).
        manifest = {
            "tokenizer_hash": self._tokenizer_hash,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "eos_policy": "stop",
            "weight_hash": self._weight_hash,
            "quant_hash": self._quant_hash,
            "system_prompt_hash": self._system_prompt_hash,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": [],
            "guided_decoding": "",
            "stop_sequences": [],
            "max_tokens": max_tokens,
        }

        if cap._capture_mode == "minimal":
            use_packed = os.environ.get("VERILM_PACKED_COMMIT", "1") == "1"
            commit_fn = self._commit_minimal_packed if use_packed else self._commit_minimal
            state = commit_fn(
                captures, n_fwd, n_layers, calls_per_fwd, fwd_batch_sizes,
                all_token_ids, prompt, seed, manifest, final_residuals_raw,
            )
        else:
            state = self._commit_full(
                captures, fwd_batch_sizes, n_layers, el_data,
                all_token_ids, prompt, seed, manifest,
            )

        if _chat_timers:
            _ct_commit = time.monotonic()
            n_tok = len(gen_token_ids)
            logger.info(
                "verilm chat timers: generate=%.1fms sync=%.1fms drain=%.1fms "
                "commit=%.1fms total=%.1fms (%d tokens)",
                (_ct_gen - _ct0) * 1000,
                (_ct_sync - _ct_gen) * 1000,
                (_ct_drain - _ct_sync) * 1000,
                (_ct_commit - _ct_drain) * 1000,
                (_ct_commit - _ct0) * 1000,
                n_tok,
            )

        # Store for audit.
        import uuid
        request_id = str(uuid.uuid4())
        self._store_audit(request_id, state)

        commitment = json.loads(state.commitment_json())

        return {
            "request_id": request_id,
            "commitment": commitment,
            "token_ids": all_token_ids,
            "kv_roots": state.kv_roots_hex(),
            "generated_text": generated_text,
            "n_tokens": state.n_tokens(),
        }

    def _commit_full(self, captures, fwd_batch_sizes, n_layers, el_data,
                      all_token_ids, prompt, seed, manifest):
        """Full-trace commitment path (existing V1-V3)."""
        import verilm_rs
        from . import capture as cap

        # Pass numpy views directly — Rust reads via buffer protocol (no .tobytes() copy).
        capture_inputs = [c[2].numpy() for c in captures]
        capture_accs = [c[3].numpy() for c in captures]
        capture_scales = []
        for c in captures:
            s = c[4]
            if hasattr(s, 'numel') and s.numel() > 1:
                capture_scales.append(float(s.max().item()))
            elif hasattr(s, 'item'):
                capture_scales.append(float(s.item()))
            else:
                capture_scales.append(float(s))

        residuals = el_data.get("residuals")
        residual_bytes = None
        if residuals:
            residual_bytes = [r.float().numpy() for r in residuals]

        return verilm_rs.commit_from_captures(
            capture_inputs=capture_inputs,
            capture_accumulators=capture_accs,
            capture_scales=capture_scales,
            fwd_batch_sizes=fwd_batch_sizes,
            n_layers=n_layers,
            q_dim=cap._q_dim,
            kv_dim=cap._kv_dim,
            intermediate_size=cap._gate_up_half,
            level_c=True,
            token_ids=[int(t) for t in all_token_ids[1:]],
            prompt=prompt.encode(),
            sampling_seed=seed,
            manifest=manifest,
            residuals=residual_bytes,
        )

    def _commit_minimal(self, captures, n_fwd, n_layers, calls_per_fwd,
                         fwd_batch_sizes, all_token_ids, prompt, seed, manifest,
                         final_residuals_raw=None):
        """V4 retained-state commitment path (no _int_mm, no full traces).

        Extracts o_proj inputs (a_i8) and per-layer scales from the minimal
        capture buffer. Groups them for the Rust commit_minimal_from_captures.
        """
        import verilm_rs

        o_proj_inputs = []
        minimal_scales = []

        for fwd_i in range(n_fwd):
            for l_i in range(n_layers):
                base = fwd_i * calls_per_fwd + l_i * 4
                # Call order: qkv=0, o=1, gate_up=2, down=3
                qkv_cap = captures[base + 0]
                o_cap = captures[base + 1]
                gu_cap = captures[base + 2]
                down_cap = captures[base + 3]

                o_proj_inputs.append(o_cap[2].numpy())  # numpy view, Rust reads via buffer protocol

                # 4 scales per layer: [scale_x_attn, scale_a, scale_x_ffn, scale_h]
                for cap_entry in (qkv_cap, o_cap, gu_cap, down_cap):
                    s = cap_entry[4]
                    if hasattr(s, 'numel') and s.numel() > 1:
                        minimal_scales.append(float(s.max().item()))
                    elif hasattr(s, 'item'):
                        minimal_scales.append(float(s.item()))
                    else:
                        minimal_scales.append(float(s))

        # Organize per-token final residuals from per-forward-pass captures.
        # model.norm hook fires once per forward pass with shape (batch_sz, hidden_dim).
        # Split by fwd_batch_sizes to get one f32 vector per token position.
        final_residuals = None
        if final_residuals_raw:
            if len(final_residuals_raw) != len(fwd_batch_sizes):
                raise RuntimeError(
                    f"final_residual count ({len(final_residuals_raw)}) != "
                    f"forward pass count ({len(fwd_batch_sizes)}). "
                    f"Hook capture misaligned."
                )
            final_residuals = []
            for fwd_tensor, batch_sz in zip(final_residuals_raw, fwd_batch_sizes):
                for pos in range(batch_sz):
                    if fwd_tensor.dim() >= 2:
                        final_residuals.append(fwd_tensor[pos].numpy())
                    else:
                        final_residuals.append(fwd_tensor.numpy())

        return verilm_rs.commit_minimal_from_captures(
            o_proj_inputs=o_proj_inputs,
            scales=minimal_scales,
            n_layers=n_layers,
            fwd_batch_sizes=fwd_batch_sizes,
            token_ids=[int(t) for t in all_token_ids[1:]],
            prompt=prompt.encode(),
            sampling_seed=seed,
            manifest=manifest,
            weight_provider=self._weight_provider,
            final_residuals=final_residuals,
        )

    def _commit_minimal_packed(self, captures, n_fwd, n_layers, calls_per_fwd,
                               fwd_batch_sizes, all_token_ids, prompt, seed, manifest,
                               final_residuals_raw=None):
        """Packed V4 commit: passes contiguous buffers to Rust, avoiding
        per-entry Python→Rust crossing and intermediate Vec allocations.

        Drop-in replacement for _commit_minimal with the same return type.
        """
        import numpy as np
        import verilm_rs
        from . import capture as cap

        hidden_dim = cap._hidden_size
        timers = os.environ.get("VERILM_COMMIT_TIMERS", "0") == "1"
        if timers:
            import time as _t
            t0 = _t.monotonic()

        # Pack all o_proj activations into one contiguous numpy array.
        # Layout: fwd-major × layer-major, each segment is (batch_sz × hidden_dim) i8 bytes.
        # Collect views first, then single np.concatenate (avoids O(n²) bytearray realloc).
        a_arrays = []
        packed_scales = []

        for fwd_i in range(n_fwd):
            for l_i in range(n_layers):
                base = fwd_i * calls_per_fwd + l_i * 4
                # Call order: qkv=0, o=1, gate_up=2, down=3
                a_arrays.append(captures[base + 1][2].numpy().ravel())

                # 4 scales per (fwd, layer): [scale_x_attn, scale_a, scale_x_ffn, scale_h]
                for cap_entry in (captures[base + 0], captures[base + 1],
                                  captures[base + 2], captures[base + 3]):
                    s = cap_entry[4]
                    if hasattr(s, 'numel') and s.numel() > 1:
                        packed_scales.append(float(s.max().item()))
                    elif hasattr(s, 'item'):
                        packed_scales.append(float(s.item()))
                    else:
                        packed_scales.append(float(s))

        if timers:
            t_loop = _t.monotonic()

        packed_a = np.concatenate(a_arrays)

        if timers:
            t_pack_a = _t.monotonic()

        # Pack final residuals as contiguous f32 bytes, token-major.
        packed_fr = None
        fr_dim = 0
        if final_residuals_raw:
            if len(final_residuals_raw) != len(fwd_batch_sizes):
                raise RuntimeError(
                    f"final_residual count ({len(final_residuals_raw)}) != "
                    f"forward pass count ({len(fwd_batch_sizes)}). "
                    f"Hook capture misaligned."
                )
            fr_arrays = []
            for fwd_tensor, batch_sz in zip(final_residuals_raw, fwd_batch_sizes):
                t = fwd_tensor.numpy() if not isinstance(fwd_tensor, np.ndarray) else fwd_tensor
                if t.ndim >= 2:
                    fr_arrays.append(t[:batch_sz].ravel())
                else:
                    fr_arrays.append(t.ravel())
            packed_fr = np.concatenate(fr_arrays)
            fr_dim = final_residuals_raw[0].shape[-1]

        if timers:
            t_pack_fr = _t.monotonic()

        result = verilm_rs.commit_minimal_packed(
            packed_a=packed_a,
            packed_scales=packed_scales,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            fwd_batch_sizes=fwd_batch_sizes,
            token_ids=[int(t) for t in all_token_ids[1:]],
            prompt=prompt.encode(),
            sampling_seed=seed,
            manifest=manifest,
            weight_provider=self._weight_provider,
            packed_final_res=packed_fr,
            final_res_dim=fr_dim,
        )

        if timers:
            t_rust = _t.monotonic()
            n_tok = sum(fwd_batch_sizes)
            logger.info(
                "verilm commit timers: loop=%.1fms concat=%.1fms pack_fr=%.1fms rust=%.1fms "
                "total=%.1fms (%d tokens, %.2fms/tok)",
                (t_loop - t0) * 1000,
                (t_pack_a - t_loop) * 1000,
                (t_pack_fr - t_pack_a) * 1000,
                (t_rust - t_pack_fr) * 1000,
                (t_rust - t0) * 1000,
                n_tok,
                (t_rust - t0) * 1000 / max(n_tok, 1),
            )

        return result

    def audit(
        self,
        request_id: str,
        *,
        token_index: int,
        layer_indices: List[int],
        tier: str = "routine",
        binary: bool = True,
    ):
        """Open an audit proof.

        For full-trace (V1-V3) state: returns zstd-compressed binary.
        For V4 retained-state: returns binary (bincode+zstd) by default,
        or JSON string when binary=False (debug only).

        Args:
            request_id: from the /chat response.
            token_index: which token to audit.
            layer_indices: which layers to open. The verifier chooses.
            tier: "routine" (shell checks) or "full" (shell + attention replay).
            binary: if True (default), return bincode+zstd bytes. False for JSON debug.
        """
        entry = self._audit_store.get(request_id)
        if entry is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        if time.time() > entry["expires_at"]:
            del self._audit_store[request_id]
            raise KeyError(f"Audit state expired for: {request_id}")

        state = entry["state"]

        # V4 retained-state path.
        if hasattr(state, 'audit_v4'):
            if binary:
                return state.audit_v4_binary(token_index, layer_indices)
            return state.audit_v4(token_index, layer_indices)

        # V1-V3 full-trace path.
        return state.audit_stratified(token_index, layer_indices, tier)

    def _store_audit(self, request_id: str, state):
        """Store audit state with TTL."""
        # Evict expired entries.
        now = time.time()
        expired = [k for k, v in self._audit_store.items() if now > v["expires_at"]]
        for k in expired:
            del self._audit_store[k]

        # Evict oldest if over limit.
        while len(self._audit_store) >= self._max_audit_entries:
            oldest = min(self._audit_store, key=lambda k: self._audit_store[k]["expires_at"])
            del self._audit_store[oldest]

        self._audit_store[request_id] = {
            "state": state,
            "expires_at": now + self.ttl_secs,
        }


def create_app(llm, **kwargs):
    """Create a FastAPI app wrapping VerifiedInferenceServer."""
    from fastapi import FastAPI, Response
    from fastapi.responses import JSONResponse

    server = VerifiedInferenceServer(llm, **kwargs)
    app = FastAPI(title="VeriLM Verified Inference")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat")
    def chat(request: dict):
        try:
            result = server.chat(
                prompt=request.get("prompt", ""),
                max_tokens=request.get("n_tokens", 4),
                temperature=float(request.get("temperature", 0.0)),
                top_k=int(request.get("top_k", 0)),
                top_p=float(request.get("top_p", 1.0)),
            )
            return result
        except Exception as e:
            logger.exception("Chat error")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/audit")
    def audit(request: dict):
        try:
            if "token_index" not in request or "layer_indices" not in request:
                return JSONResponse(
                    {"error": "token_index and layer_indices are required"},
                    status_code=400,
                )
            use_binary = request.get("binary", True)
            result = server.audit(
                request_id=request["request_id"],
                token_index=request["token_index"],
                layer_indices=request.get("layer_indices", []),
                tier=request.get("tier", "routine"),
                binary=use_binary,
            )
            # Binary V4 or V1-V3 returns bytes.
            if isinstance(result, (bytes, memoryview)):
                return Response(
                    content=bytes(result),
                    media_type="application/octet-stream",
                )
            # JSON V4 returns string.
            if isinstance(result, str):
                return JSONResponse(
                    json.loads(result),
                    media_type="application/json",
                )
            return Response(
                content=result,
                media_type="application/octet-stream",
                headers={"Content-Encoding": "zstd"},
            )
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception as e:
            logger.exception("Audit error")
            return JSONResponse({"error": str(e)}, status_code=500)

    return app
