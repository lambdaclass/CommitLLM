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

        # Extract model geometry for manifest.
        cfg = getattr(model, "config", None)
        self._tokenizer_hash = self._compute_tokenizer_hash(llm)

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

    def chat(
        self,
        prompt: str,
        max_tokens: int = 4,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> dict:
        """Run verified inference: generate → capture → commit."""
        import verilm_rs
        from vllm import SamplingParams

        from . import capture as cap
        from .trace import build_layer_traces

        # Clear buffers.
        self.buf.drain()
        self.el_capture.drain()

        # Generate.
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else -1,
            top_p=top_p,
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
        traces = build_layer_traces(captures, n_layers=n_layers, level_c=True)

        # Token IDs: prompt + generated, truncated/padded to n_tokens.
        all_token_ids = prompt_token_ids + gen_token_ids
        token_ids = all_token_ids[:n_tokens]
        if len(token_ids) < n_tokens:
            token_ids.extend([0] * (n_tokens - len(token_ids)))

        # Sampling seed: random for non-greedy, deterministic for greedy.
        if temperature > 0:
            seed = os.urandom(32)
        else:
            seed = hashlib.sha256(prompt.encode()).digest()

        # Convert traces to list-of-list-of-dicts for verilm_rs.
        trace_dicts = []
        for token_layers in traces:
            layer_dicts = []
            for lt in token_layers:
                layer_dicts.append({
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
                })
            trace_dicts.append(layer_dicts)

        # Build manifest.
        manifest = {
            "tokenizer_hash": self._tokenizer_hash,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "eos_policy": "stop",
        }

        # Commit via Rust.
        state = verilm_rs.commit(
            traces=trace_dicts,
            token_ids=[int(t) for t in token_ids],
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
            "token_ids": token_ids,
            "kv_roots": state.kv_roots_hex(),
            "generated_text": generated_text,
            "n_tokens": state.n_tokens(),
        }

    def audit(
        self,
        request_id: str,
        challenge_indices: Optional[List[int]] = None,
    ) -> bytes:
        """Open proof for challenged tokens. Returns compact binary (zstd)."""
        entry = self._audit_store.get(request_id)
        if entry is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        if time.time() > entry["expires_at"]:
            del self._audit_store[request_id]
            raise KeyError(f"Audit state expired for: {request_id}")

        state = entry["state"]
        if challenge_indices is None:
            challenge_indices = list(range(state.n_tokens()))

        return state.open_compact(challenge_indices)

    def audit_json(
        self,
        request_id: str,
        challenge_indices: Optional[List[int]] = None,
    ) -> str:
        """Open proof for challenged tokens. Returns JSON string."""
        entry = self._audit_store.get(request_id)
        if entry is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        if time.time() > entry["expires_at"]:
            del self._audit_store[request_id]
            raise KeyError(f"Audit state expired for: {request_id}")

        state = entry["state"]
        if challenge_indices is None:
            challenge_indices = list(range(state.n_tokens()))

        return state.open_json(challenge_indices)

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
                temperature=request.get("temperature", 0.0),
                top_k=request.get("top_k", 0),
                top_p=request.get("top_p", 1.0),
            )
            return result
        except Exception as e:
            logger.exception("Chat error")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/audit")
    def audit(request: dict):
        try:
            request_id = request["request_id"]
            challenge_indices = request.get("challenge_indices")
            proof_bytes = server.audit(request_id, challenge_indices)
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
