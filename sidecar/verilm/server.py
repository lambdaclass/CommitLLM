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
import time
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("verilm.server")


class VerifiedInferenceServer:
    """Verified inference: capture + commit + audit in one process."""

    def __init__(self, llm, *, max_audit_entries: int = 10_000, ttl_secs: int = 600):
        from vllm import LLM

        from .capture import (
            configure_from_model,
            get_capture_buffer,
            get_model_from_llm,
        )
        from .hooks import EmbeddingLogitCapture
        from .trace import build_layer_traces

        self.llm = llm
        self.ttl_secs = ttl_secs

        # Configure capture from model.
        model = get_model_from_llm(llm)
        configure_from_model(model)

        # Install embedding/logit hooks.
        self.el_capture = EmbeddingLogitCapture()
        n_hooks = self.el_capture.install(model)
        logger.info("Installed %d embedding/logit hooks", n_hooks)

        # Manifest hashes.
        self._tokenizer_hash = self._compute_tokenizer_hash(llm)
        self._weight_hash = self._compute_weight_hash_rw(llm)
        self._quant_hash = self._compute_quant_hash(model)
        self._system_prompt_hash = None  # Set via configure_system_prompt()

        self.buf = get_capture_buffer()
        self._audit_store: Dict[str, dict] = {}
        self._max_audit_entries = max_audit_entries

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

            # Get model path from vLLM. HuggingFace models are cached
            # under ~/.cache/huggingface/hub/ after download.
            model_dir = self._resolve_model_dir(llm)
            if model_dir is None:
                logger.warning("Could not resolve model directory for R_W computation")
                return "00" * 32

            logger.info("Computing R_W (weight-chain hash) from %s...", model_dir)
            weight_hash = verilm_rs.compute_weight_hash(model_dir)
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
    ) -> dict:
        """Run verified inference: generate → capture → commit.

        Only greedy decoding (temperature=0) is supported. Stochastic
        sampling cannot be faithfully replayed by the verifier because
        vLLM's sampler uses an internal RNG that does not match the
        protocol's canonical sampler. This restriction will be lifted
        once a canonical ChaCha20-based sampler is integrated.
        """
        import verilm_rs
        from vllm import SamplingParams

        from . import capture as cap
        from .trace import build_layer_traces

        # Enforce greedy-only until canonical stochastic replay exists.
        if temperature != 0.0:
            raise ValueError(
                "verilm: only greedy decoding (temperature=0) is supported. "
                "Stochastic sampling cannot be faithfully verified — the "
                "verifier's canonical sampler does not match vLLM's RNG."
            )

        # Clear buffers.
        self.buf.drain()
        self.el_capture.drain()

        # Greedy: deterministic seed derived from prompt.
        seed = hashlib.sha256(prompt.encode()).digest()

        # Generate.
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )
        outputs = self.llm.generate([prompt], params)
        output = outputs[0]

        generated_text = output.outputs[0].text
        gen_token_ids = list(output.outputs[0].token_ids)
        prompt_token_ids = list(output.prompt_token_ids)

        # Drain captures.
        captures = self.buf.drain()
        el_data = self.el_capture.drain()

        n_layers = cap._n_layers
        calls_per_fwd = n_layers * cap.PROJS_PER_LAYER
        if len(captures) == 0 or len(captures) % calls_per_fwd != 0:
            raise RuntimeError(
                f"Capture count {len(captures)} not a multiple of "
                f"calls_per_fwd {calls_per_fwd}"
            )

        n_tokens = len(captures) // calls_per_fwd

        # Build traces with KV cache for Level C attention verification.
        # Pass residuals from embedding/logit hooks for RMSNorm bridge verification.
        residuals = el_data.get("residuals")
        traces = build_layer_traces(
            captures, n_layers=n_layers, level_c=True,
            residuals=residuals if residuals else None,
        )

        # Token IDs must match trace count exactly.
        all_token_ids = prompt_token_ids + gen_token_ids
        if len(all_token_ids) != n_tokens:
            raise RuntimeError(
                f"Token ID count ({len(all_token_ids)}) does not match "
                f"trace count ({n_tokens}). Cannot commit with mismatched "
                f"transcript."
            )

        # Convert traces to list-of-list-of-dicts for verilm_rs.
        trace_dicts = []
        for token_layers in traces:
            layer_dicts = []
            for lt in token_layers:
                d = {
                    "x_attn": lt["x_attn"].to(torch.int8).cpu().numpy().tolist(),
                    "q": lt["q"].to(torch.int32).cpu().numpy().tolist(),
                    "k": lt["k"].to(torch.int32).cpu().numpy().tolist(),
                    "v": lt["v"].to(torch.int32).cpu().numpy().tolist(),
                    "a": lt["a"].to(torch.int8).cpu().numpy().tolist(),
                    "attn_out": lt["attn_out"].to(torch.int32).cpu().numpy().tolist(),
                    "x_ffn": lt["x_ffn"].to(torch.int8).cpu().numpy().tolist(),
                    "g": lt["g"].to(torch.int32).cpu().numpy().tolist(),
                    "u": lt["u"].to(torch.int32).cpu().numpy().tolist(),
                    "h": lt["h"].to(torch.int8).cpu().numpy().tolist(),
                    "ffn_out": lt["ffn_out"].to(torch.int32).cpu().numpy().tolist(),
                    "kv_cache_k": [t.to(torch.int8).cpu().numpy().tolist() for t in lt.get("kv_cache_k", [])],
                    "kv_cache_v": [t.to(torch.int8).cpu().numpy().tolist() for t in lt.get("kv_cache_v", [])],
                }
                # Residual stream for RMSNorm bridge verification.
                if "residual" in lt:
                    d["residual"] = lt["residual"].cpu().numpy().tolist()
                # Pass activation scales when available (production W8A8 path).
                if "qkv_scale" in lt:
                    d["scale_x_attn"] = float(lt["qkv_scale"].item()) if lt["qkv_scale"].numel() == 1 else float(lt["qkv_scale"].max().item())
                if "o_scale" in lt:
                    d["scale_a"] = float(lt["o_scale"].item()) if lt["o_scale"].numel() == 1 else float(lt["o_scale"].max().item())
                if "gu_scale" in lt:
                    d["scale_x_ffn"] = float(lt["gu_scale"].item()) if lt["gu_scale"].numel() == 1 else float(lt["gu_scale"].max().item())
                if "d_scale" in lt:
                    d["scale_h"] = float(lt["d_scale"].item()) if lt["d_scale"].numel() == 1 else float(lt["d_scale"].max().item())
                layer_dicts.append(d)
            trace_dicts.append(layer_dicts)

        # Build manifest.
        manifest = {
            "tokenizer_hash": self._tokenizer_hash,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "eos_policy": "stop",
            "weight_hash": self._weight_hash,
            "quant_hash": self._quant_hash,
            "system_prompt_hash": self._system_prompt_hash,
        }

        # Commit via Rust.
        state = verilm_rs.commit(
            traces=trace_dicts,
            token_ids=[int(t) for t in all_token_ids],
            prompt=prompt.encode(),
            sampling_seed=seed,
            manifest=manifest,
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

    def audit(
        self,
        request_id: str,
        *,
        token_index: int,
        layer_indices: List[int],
        tier: str = "routine",
    ) -> bytes:
        """Open a stratified audit proof. Returns zstd-compressed binary.

        Args:
            request_id: from the /chat response.
            token_index: which token to audit.
            layer_indices: which layers to open. The verifier chooses.
            tier: "routine" (shell checks) or "full" (shell + attention replay).
        """
        entry = self._audit_store.get(request_id)
        if entry is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        if time.time() > entry["expires_at"]:
            del self._audit_store[request_id]
            raise KeyError(f"Audit state expired for: {request_id}")

        state = entry["state"]
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
            proof_bytes = server.audit(
                request_id=request["request_id"],
                token_index=request["token_index"],
                layer_indices=request["layer_indices"],
                tier=request.get("tier", "routine"),
            )
            return Response(
                content=proof_bytes,
                media_type="application/octet-stream",
                headers={"Content-Encoding": "zstd"},
            )
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception as e:
            logger.exception("Audit error")
            return JSONResponse({"error": str(e)}, status_code=500)

    return app
