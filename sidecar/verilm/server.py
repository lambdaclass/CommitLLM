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

        self.el_capture = None

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

        # Install LP hidden capture (decode boundary).
        # Captures LogitsProcessor input hidden state (bf16) for exact
        # token-identity verification via bf16 lm_head matmul.
        from .hooks import LPHiddenCapture
        self.lp_hidden_capture = LPHiddenCapture()
        if not self.lp_hidden_capture.install(model):
            logger.warning(
                "Could not install LP hidden capture — "
                "bf16 decode-boundary verification unavailable"
            )
            self.lp_hidden_capture = None

        # Install canonical sampler hook on LogitsProcessor module.
        # Must be AFTER capture hooks so capture sees original (unmasked) logits.
        from .sampler import CanonicalSamplerHook
        self.sampler_hook = CanonicalSamplerHook(minimal=True)
        if not self.sampler_hook.install(model):
            raise RuntimeError(
                "Could not install canonical sampler hook — "
                "LogitsProcessor module not found in model"
            )

        # Manifest hashes.
        self._tokenizer_hash = self._compute_tokenizer_hash(llm)
        self._chat_template_hash = self._compute_chat_template_hash(llm)
        self._model_dir = self._resolve_model_dir(llm)
        self._weight_hash = self._compute_weight_hash_rw(llm)
        self._quant_hash = self._compute_quant_hash(model)
        self._rope_config_hash = self._compute_rope_config_hash(model)
        self._rmsnorm_eps = self._extract_rmsnorm_eps(model)
        self._system_prompt_hash = hashlib.sha256(b"").hexdigest()  # Default: empty prompt
        self._bos_eos_policy = self._extract_bos_eos_policy(llm)
        self._truncation_policy = self._extract_truncation_policy(llm)
        self._special_token_policy = self._extract_special_token_policy(llm)
        self._padding_policy = self._extract_padding_policy(llm)
        self._detokenization_policy = self._extract_detokenization_policy(llm)
        self._adapter_hash = self._compute_adapter_hash(model)
        self._eos_token_id = self._extract_eos_token_id(llm)

        # Quantization fields (#5-7) — quant_family/scale_derivation deferred
        # until _weight_provider is available (initialized after this block).
        self._quant_block_size = self._extract_quant_block_size(model)

        # Architecture fields (#8).
        arch = self._extract_architecture(model)
        self._kv_dim = arch.get("kv_dim")
        self._ffn_dim = arch.get("ffn_dim")
        self._d_head = arch.get("d_head")
        self._n_q_heads = arch.get("n_q_heads")
        self._n_kv_heads = arch.get("n_kv_heads")
        self._rope_theta = arch.get("rope_theta")

        # Attention runtime semantics.
        self._attn_backend = self._extract_attn_backend(model)
        self._attn_dtype = self._extract_attn_dtype(model)

        self.buf = get_capture_buffer()

        # x_attn capture for committed KV transcript derivation.
        # When enabled, the QKV projection input (x_attn_i8) is captured
        # per layer per forward pass, allowing the Rust commit engine to
        # derive post-RoPE K/V via deterministic INT8 matmul + dequant.
        # On by default — required for kv_roots in the commitment.
        # Disable with VERILM_CAPTURE_X_ATTN=0.
        self.buf._capture_x_attn = os.environ.get("VERILM_CAPTURE_X_ATTN", "1") == "1"

        # Initialize pinned CPU slab for o_proj D2H (minimal mode).
        if cap._hidden_size > 0:
            self.buf.init_pinned_slab(cap._hidden_size)

        # Score witness: capture pre-RoPE K on GPU for attention score reconstruction.
        # Enabled via VERILM_SCORE_WITNESS=1 env var (opt-in).
        if os.environ.get("VERILM_SCORE_WITNESS", "0") == "1":
            cfg = getattr(model, "config", None)
            rope_theta = getattr(cfg, "rope_theta", 10000.0) if cfg else 10000.0
            rope_scaling = getattr(cfg, "rope_scaling", None) if cfg else None
            # Convert HF rope_scaling dict to plain dict if needed.
            rope_scaling_dict = dict(rope_scaling) if rope_scaling else None
            max_seq = int(os.environ.get(
                "VERILM_SCORE_WITNESS_MAX_SEQ",
                str(getattr(cfg, "max_position_embeddings", 4096) if cfg else 4096),
            ))
            self.buf.init_score_witness(
                n_layers=cap._n_layers,
                max_seq_len=max_seq,
                n_q_heads=cap._num_heads,
                n_kv_heads=cap._num_kv_heads,
                d_head=cap._head_dim,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling_dict,
            )

        # Try to activate native C++ capture wrapper.
        # Must be after configure_from_model() + init_pinned_slab().
        import torch
        device_hint = torch.empty(1, device="cuda")
        cap.activate_native_capture(device_hint_tensor=device_hint)

        self._audit_store: Dict[str, dict] = {}
        self._max_audit_entries = max_audit_entries

        # Cache timer flag once at init (avoid per-request os.environ read).
        self._chat_timers = os.environ.get("VERILM_COMMIT_TIMERS", "0") == "1"

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

        # Quantization fields deferred from above — need _weight_provider.
        if self._weight_provider is not None:
            self._quant_family = self._weight_provider.quant_family()
            self._scale_derivation = self._weight_provider.scale_derivation()
        else:
            self._quant_family = self._extract_quant_family(model)
            self._scale_derivation = self._extract_scale_derivation(model)

        # Detect verification profile to determine x_attn source policy.
        # When QKV Freivalds is enabled (Llama): prover MUST use bridge-derived
        # x_attn so QKV accumulators match the verifier's own derivation.
        # When QKV Freivalds is disabled (Qwen): captured x_attn is used for
        # score/corridor paths where bridge diverges from GPU fused norm_quant.
        self._model_type = getattr(getattr(model, "config", None), "model_type", None)
        self._supports_qkv_freivalds = self._detect_qkv_freivalds_support()
        x_attn_source = "bridge" if self._supports_qkv_freivalds else "captured"
        logger.info(
            "x_attn_source=%s (model_type=%s, quant=%s, qkv_freivalds=%s)",
            x_attn_source, self._model_type, self._quant_family,
            self._supports_qkv_freivalds,
        )

    def _detect_qkv_freivalds_support(self) -> bool:
        """Detect whether this model supports QKV Freivalds checks.

        Mirrors VerificationProfile::detect() in verilm-core/types.rs.
        When True, bridge-derived x_attn MUST be used for shell QKV so
        prover and verifier agree. When False, captured x_attn may be
        used (e.g. Qwen where bridge diverges from GPU).
        """
        mt = (self._model_type or "").lower()
        qf = (self._quant_family or "").upper()
        is_w8a8 = qf == "W8A8"
        if "qwen" in mt and is_w8a8:
            return False  # Qwen W8A8: bridge != GPU fused norm_quant
        if "llama" in mt and is_w8a8:
            return True   # Llama W8A8: bridge is accurate
        # Unknown model: default to True (safe — uses bridge, Freivalds works)
        return True

    def _compute_tokenizer_hash(self, llm) -> str:
        """SHA-256 of full tokenizer identity (vocab + normalizer + pre-tokenizer + added tokens).

        Hashes the canonical tokenizer.json representation via backend_tokenizer.to_str(),
        normalized through json.loads/json.dumps(sort_keys=True) for cross-platform determinism.
        This binds the complete tokenizer semantics — not just the vocabulary mapping.
        """
        try:
            tokenizer = llm.get_tokenizer()
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is not None:
                raw = backend.to_str()
                canonical = json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=True)
                return hashlib.sha256(canonical.encode()).hexdigest()
            # Fallback for tokenizers without a backend (rare/legacy).
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

    def _compute_chat_template_hash(self, llm) -> Optional[str]:
        """SHA-256 of the chat template string, if present."""
        try:
            tokenizer = llm.get_tokenizer()
            template = getattr(tokenizer, "chat_template", None)
            if template is None:
                return None
            return hashlib.sha256(template.encode()).hexdigest()
        except Exception as e:
            logger.warning("Could not hash chat template: %s", e)
            return None

    def _compute_rope_config_hash(self, model) -> Optional[str]:
        """SHA-256 of RoPE configuration (theta + scaling), if present."""
        try:
            cfg = getattr(model, "config", None)
            if cfg is None:
                return None
            rope_theta = getattr(cfg, "rope_theta", None)
            rope_scaling = getattr(cfg, "rope_scaling", None)
            if rope_theta is None and rope_scaling is None:
                return None
            rope_dict = {
                "rope_theta": rope_theta,
                "rope_scaling": rope_scaling,
            }
            return hashlib.sha256(
                json.dumps(rope_dict, sort_keys=True, default=str).encode()
            ).hexdigest()
        except Exception as e:
            logger.warning("Could not hash RoPE config: %s", e)
            return None

    @staticmethod
    def _extract_rmsnorm_eps(model) -> Optional[float]:
        """Extract RMSNorm epsilon from model config."""
        try:
            cfg = getattr(model, "config", None)
            if cfg is None:
                return None
            return getattr(cfg, "rms_norm_eps", None)
        except Exception:
            return None

    @staticmethod
    def _extract_bos_eos_policy(llm) -> str:
        """Derive BOS/EOS policy from tokenizer config.

        Checks whether the tokenizer adds BOS/EOS tokens automatically.
        Returns a canonical string like 'add_bos', 'add_bos_eos', or 'none'.
        """
        try:
            tokenizer = llm.get_tokenizer()
            add_bos = getattr(tokenizer, "add_bos_token", False)
            add_eos = getattr(tokenizer, "add_eos_token", False)
            if add_bos and add_eos:
                return "add_bos_eos"
            elif add_bos:
                return "add_bos"
            elif add_eos:
                return "add_eos"
            return "none"
        except Exception:
            return "none"

    @staticmethod
    def _extract_truncation_policy(llm) -> str:
        """Derive truncation policy from tokenizer/engine config.

        Inspects the tokenizer's truncation_side attribute and the engine's
        max_model_len to determine the canonical policy string.
        vLLM default is to raise an error if input exceeds max_model_len.
        """
        try:
            tokenizer = llm.get_tokenizer()
            truncation_side = getattr(tokenizer, "truncation_side", None)
            if truncation_side == "left":
                return "left"
            elif truncation_side == "right":
                return "right"
            return "error"
        except Exception:
            return "error"

    @staticmethod
    def _extract_padding_policy(llm) -> str:
        """Derive padding policy from tokenizer config.

        Inspects the tokenizer's padding_side attribute.
        vLLM default is no padding (individual sequences are not padded).
        """
        try:
            tokenizer = llm.get_tokenizer()
            padding_side = getattr(tokenizer, "padding_side", None)
            if padding_side == "left":
                return "left"
            elif padding_side == "right":
                return "right"
            return "none"
        except Exception:
            return "none"

    @staticmethod
    def _extract_quant_family(model) -> Optional[str]:
        """Derive quantization family from model config (e.g. 'W8A8', 'GPTQ')."""
        try:
            cfg = getattr(model, "config", None)
            quant_cfg = getattr(cfg, "quantization_config", None)
            if quant_cfg is None:
                return None
            if hasattr(quant_cfg, "to_dict"):
                d = quant_cfg.to_dict()
            elif isinstance(quant_cfg, dict):
                d = quant_cfg
            else:
                return None
            # Common keys: "quant_method" (GPTQ/AWQ), or infer from config shape
            method = d.get("quant_method")
            if method:
                return str(method).upper()
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_scale_derivation(model) -> Optional[str]:
        """Derive scale derivation method from quantization config."""
        try:
            cfg = getattr(model, "config", None)
            quant_cfg = getattr(cfg, "quantization_config", None)
            if quant_cfg is None:
                return None
            if hasattr(quant_cfg, "to_dict"):
                d = quant_cfg.to_dict()
            elif isinstance(quant_cfg, dict):
                d = quant_cfg
            else:
                return None
            # Look for scale derivation hints
            if d.get("is_marlin_format"):
                return "channel_absmax"
            if d.get("sym", True):
                return "absmax"
            return "zeropoint"
        except Exception:
            return None

    @staticmethod
    def _extract_quant_block_size(model) -> Optional[int]:
        """Derive quantization block size from config."""
        try:
            cfg = getattr(model, "config", None)
            quant_cfg = getattr(cfg, "quantization_config", None)
            if quant_cfg is None:
                return None
            if hasattr(quant_cfg, "to_dict"):
                d = quant_cfg.to_dict()
            elif isinstance(quant_cfg, dict):
                d = quant_cfg
            else:
                return None
            return d.get("group_size") or d.get("block_size")
        except Exception:
            return None

    @staticmethod
    def _extract_architecture(model) -> dict:
        """Extract architecture dimensions from model config."""
        result = {}
        try:
            cfg = getattr(model, "config", None)
            if cfg is None:
                return result
            # Head dimension
            d_head = getattr(cfg, "head_dim", None)
            if d_head is None:
                hidden = getattr(cfg, "hidden_size", None)
                n_heads = getattr(cfg, "num_attention_heads", None)
                if hidden and n_heads:
                    d_head = hidden // n_heads
            if d_head:
                result["d_head"] = int(d_head)
            # Head counts
            n_q = getattr(cfg, "num_attention_heads", None)
            if n_q:
                result["n_q_heads"] = int(n_q)
            n_kv = getattr(cfg, "num_key_value_heads", None)
            if n_kv:
                result["n_kv_heads"] = int(n_kv)
            # KV dim
            if d_head and n_kv:
                result["kv_dim"] = int(n_kv) * int(d_head)
            # FFN dim
            ffn = getattr(cfg, "intermediate_size", None)
            if ffn:
                result["ffn_dim"] = int(ffn)
            # RoPE theta
            rope_theta = getattr(cfg, "rope_theta", None)
            if rope_theta is not None:
                result["rope_theta"] = float(rope_theta)
        except Exception:
            pass
        return result

    @staticmethod
    def _extract_attn_backend(model) -> str:
        """Extract the attention backend from model config.

        Returns the attn_implementation string (e.g. "sdpa", "eager",
        "flash_attention_2"). Falls back to "unknown" if not detectable.
        """
        try:
            cfg = getattr(model, "config", None)
            if cfg is not None:
                # HuggingFace stores the resolved backend in _attn_implementation
                impl = getattr(cfg, "_attn_implementation", None)
                if impl is not None:
                    return str(impl)
                # Some models store it as attn_implementation
                impl = getattr(cfg, "attn_implementation", None)
                if impl is not None:
                    return str(impl)
        except Exception:
            pass

        # vLLM 0.8+: walk model layers to find the attention impl class name.
        try:
            for module in model.modules():
                cls_name = type(module).__name__
                if "Attention" not in cls_name:
                    continue
                # vLLM Attention layers have .impl with the backend instance
                impl = getattr(module, "impl", None)
                if impl is not None:
                    name = type(impl).__name__.lower()
                    if "flash" in name:
                        return "flash_attention_2"
                    if "sdpa" in name or "torch" in name:
                        return "sdpa"
                    if "eager" in name:
                        return "eager"
                    return name
        except Exception:
            pass

        # Fallback: vLLM global attention backend selector.
        try:
            from vllm.attention.selector import _Backend, get_global_forced_attn_backend
            forced = get_global_forced_attn_backend()
            if forced is not None:
                name = forced.name.lower()
                if "flash" in name:
                    return "flash_attention_2"
                if "sdpa" in name or "torch" in name:
                    return "sdpa"
                return name
        except (ImportError, Exception):
            pass

        return "unknown"

    @staticmethod
    def _extract_attn_dtype(model) -> str:
        """Extract the effective attention compute dtype.

        Returns the torch dtype string (e.g. "float16", "bfloat16").
        Falls back to "unknown" if not detectable.
        """
        try:
            cfg = getattr(model, "config", None)
            if cfg is not None:
                # Check torch_dtype on the config
                td = getattr(cfg, "torch_dtype", None)
                if td is not None:
                    import torch
                    if isinstance(td, torch.dtype):
                        return str(td).replace("torch.", "")
                    return str(td)
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _extract_special_token_policy(llm) -> str:
        """Derive special-token handling from tokenizer config.

        Returns a canonical string describing how special tokens in user
        input are processed during tokenization:
          - "encode": special tokens are encoded normally (default for most models)
          - "strip": special tokens are stripped before encoding
        """
        try:
            tokenizer = llm.get_tokenizer()
            # HuggingFace tokenizers with `added_tokens_encoder` map special
            # tokens to specific IDs. If the tokenizer has this and it's
            # non-empty, the policy is "encode" (they are recognized and
            # mapped to their canonical IDs).
            added = getattr(tokenizer, "added_tokens_encoder", {})
            if added:
                return "encode"
            return "encode"
        except Exception:
            return "encode"

    @staticmethod
    def _extract_detokenization_policy(llm) -> str:
        """Derive detokenization policy from tokenizer config.

        Inspects `clean_up_tokenization_spaces` which controls whether
        the tokenizer normalizes whitespace during decode.
        """
        try:
            tokenizer = llm.get_tokenizer()
            clean = getattr(tokenizer, "clean_up_tokenization_spaces", None)
            if clean is True:
                return "clean_spaces"
            return "default"
        except Exception:
            return "default"

    @staticmethod
    def _compute_adapter_hash(model) -> Optional[str]:
        """Hash adapter/LoRA identity if any adapter is loaded.

        Checks for vLLM LoRA modules and PEFT adapters. Returns None
        when no adapter is detected (base model only).
        """
        try:
            # vLLM LoRA support: check for lora_config on the model.
            lora_config = getattr(model, "lora_config", None)
            if lora_config is not None:
                config_dict = vars(lora_config) if hasattr(lora_config, "__dict__") else {"repr": repr(lora_config)}
                return hashlib.sha256(
                    json.dumps(config_dict, sort_keys=True, default=str).encode()
                ).hexdigest()
            # PEFT adapter: check for peft_config dict.
            peft_config = getattr(model, "peft_config", None)
            if peft_config is not None:
                serializable = {}
                for k, v in peft_config.items():
                    serializable[k] = vars(v) if hasattr(v, "__dict__") else str(v)
                return hashlib.sha256(
                    json.dumps(serializable, sort_keys=True, default=str).encode()
                ).hexdigest()
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_eos_token_id(llm) -> Optional[int]:
        """Extract the EOS token ID from the tokenizer.

        Required for min_tokens and ignore_eos enforcement in the verifier.
        """
        try:
            tokenizer = llm.get_tokenizer()
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                return int(eos_id)
            return None
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
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_tokens: int = 0,
        ignore_eos: bool = False,
    ) -> dict:
        """Run verified inference: generate → capture → commit.

        Defaults to sampled decoding (temperature=1.0) with the canonical
        ChaCha20-based sampler injected as a vLLM logits processor,
        ensuring exact reproducibility by the verifier. Greedy decoding
        is available via temperature=0. vLLM's own sampling knobs are
        neutralized — the canonical processor is the sole policy owner.
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
        if self.lp_hidden_capture is not None:
            self.lp_hidden_capture.drain()

        # Fresh random batch seed per request. The commitment includes
        # seed_commitment = SHA256(batch_seed); the batch_seed itself is
        # revealed only at audit time, preventing pre-computation attacks.
        seed = secrets.token_bytes(32)
        self._last_seed = seed  # Expose for diagnostics.

        _chat_timers = self._chat_timers
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
        # Each component uses its own dedicated CUDA stream + event.
        self.buf.wait_for_transfers()
        if self.final_res_capture is not None:
            self.final_res_capture.wait_for_transfers()
        if self.lp_hidden_capture is not None:
            self.lp_hidden_capture.wait_for_transfers()

        if _chat_timers:
            _ct_sync = time.monotonic()

        # Drain captures and hook data.
        el_data = self.el_capture.drain() if self.el_capture is not None else {}

        if _chat_timers:
            _ct_el_drain = time.monotonic()

        final_residuals_raw = self.final_res_capture.drain() if self.final_res_capture is not None else []
        lp_hidden_raw = self.lp_hidden_capture.drain() if self.lp_hidden_capture is not None else []

        if _chat_timers:
            _ct_fr_drain = time.monotonic()

        n_layers = cap._n_layers
        calls_per_fwd = n_layers * cap.PROJS_PER_LAYER
        all_token_ids = prompt_token_ids + gen_token_ids
        expected_traces = len(all_token_ids) - 1

        o_inputs, scales, call_count, x_attn_inputs, witnessed_scores = self.buf.drain_minimal()

        if _chat_timers:
            _ct_buf_drain = time.monotonic()

        if cap._native_capture is not None:
            _cc = cap._native_capture.get_call_counter()
        elif cap._capture_hook is not None:
            _cc = cap._capture_hook.call_counter
        else:
            _cc = cap._call_counter
        logger.info(
            "verilm: captures=%d (o_inputs=%d, scales=%d), calls_per_fwd=%d, "
            "prompt_tokens=%d, gen_tokens=%d, all_tokens=%d, call_counter=%d",
            call_count, len(o_inputs), len(scales), calls_per_fwd,
            len(prompt_token_ids), len(gen_token_ids),
            len(all_token_ids), _cc,
        )

        if call_count == 0 or call_count % calls_per_fwd != 0:
            raise RuntimeError(
                f"Capture count {call_count} not a multiple of "
                f"calls_per_fwd {calls_per_fwd} "
                f"(prompt_tokens={len(prompt_token_ids)}, gen_tokens={len(gen_token_ids)}, "
                f"call_counter={_cc})"
            )

        n_fwd = call_count // calls_per_fwd
        expected_o = n_fwd * n_layers
        if len(o_inputs) != expected_o:
            raise RuntimeError(
                f"o_inputs count ({len(o_inputs)}) != expected "
                f"({expected_o} = {n_fwd} fwd × {n_layers} layers). "
                f"Counter drift or missed o_proj append."
            )
        fwd_batch_sizes = [
            o_inputs[i * n_layers].shape[0] for i in range(n_fwd)
        ]
        # Per-row scales: prefill calls contribute batch_size values each.
        expected_scales = sum(
            bs * cap.PROJS_PER_LAYER * n_layers for bs in fwd_batch_sizes
        )
        if len(scales) != expected_scales:
            raise RuntimeError(
                f"scales count ({len(scales)}) != expected ({expected_scales}). "
                f"Scale buffer out of sync (fwd_batch_sizes={fwd_batch_sizes})."
            )
        n_tokens = sum(fwd_batch_sizes)

        # EOS trailing forward pass trim.
        if (n_tokens == expected_traces + 1
                and len(gen_token_ids) < max_tokens
                and fwd_batch_sizes[-1] == 1):
            logger.info(
                "verilm: trimming trailing EOS forward pass "
                "(gen=%d < max=%d, n_fwd %d→%d)",
                len(gen_token_ids), max_tokens, n_fwd, n_fwd - 1,
            )
            o_inputs = o_inputs[:-n_layers]
            scales = scales[:-calls_per_fwd]
            n_fwd -= 1
            fwd_batch_sizes = fwd_batch_sizes[:-1]
            if len(final_residuals_raw) == n_fwd + 1:
                final_residuals_raw = final_residuals_raw[:-1]
            if len(lp_hidden_raw) == n_fwd + 1:
                lp_hidden_raw = lp_hidden_raw[:-1]
            n_tokens = sum(fwd_batch_sizes)

        if _chat_timers:
            _ct_prep = time.monotonic()

        if n_tokens != expected_traces:
            raise RuntimeError(
                f"Token count ({n_tokens}) does not match expected "
                f"({expected_traces}). "
                f"Cannot commit with mismatched transcript. "
                f"calls_per_fwd={calls_per_fwd}, "
                f"n_fwd={n_fwd}, fwd_batch_sizes={fwd_batch_sizes}, "
                f"prompt_tokens={len(prompt_token_ids)}, gen_tokens={len(gen_token_ids)}"
            )

        # Build manifest from actual request parameters.
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
            "bad_word_ids": [],
            "guided_decoding": "",
            "stop_sequences": [],
            "max_tokens": max_tokens,
            # Four-spec fields.
            "chat_template_hash": self._chat_template_hash,
            "rope_config_hash": self._rope_config_hash,
            "rmsnorm_eps": self._rmsnorm_eps,
            "sampler_version": "chacha20-vi-sample-v1",
            # InputSpec fields.
            "bos_eos_policy": self._bos_eos_policy,
            "truncation_policy": self._truncation_policy,
            "special_token_policy": self._special_token_policy,
            "padding_policy": self._padding_policy,
            # DecodeSpec fields.
            "decode_mode": "greedy" if temperature == 0.0 else "sampled",
            # ModelSpec fields.
            "adapter_hash": self._adapter_hash,
            "attn_backend": self._attn_backend,
            "attn_dtype": self._attn_dtype,
            "quant_family": self._quant_family,
            "scale_derivation": self._scale_derivation,
            "quant_block_size": self._quant_block_size,
            "kv_dim": self._kv_dim,
            "ffn_dim": self._ffn_dim,
            "d_head": self._d_head,
            "n_q_heads": self._n_q_heads,
            "n_kv_heads": self._n_kv_heads,
            "rope_theta": self._rope_theta,
            # OutputSpec fields.
            "min_tokens": min_tokens,
            "ignore_eos": ignore_eos,
            "detokenization_policy": self._detokenization_policy,
            "eos_token_id": self._eos_token_id,
        }

        if _chat_timers:
            _ct_manifest = time.monotonic()

        n_prompt_tokens = len(prompt_token_ids)

        # Packed commit is faster but does not support x_attn (KV transcript)
        # or LP hidden (decode boundary). Fall back to non-packed when either is present.
        use_packed = os.environ.get("VERILM_PACKED_COMMIT", "1") == "1"
        has_x_attn = bool(x_attn_inputs)
        has_lp_hidden = bool(lp_hidden_raw)
        if use_packed and not has_x_attn and not has_lp_hidden:
            state = self._commit_minimal_packed(
                o_inputs, scales, n_fwd, n_layers, fwd_batch_sizes,
                all_token_ids, prompt, seed, manifest, final_residuals_raw,
                n_prompt_tokens,
            )
        else:
            state = self._commit_minimal(
                o_inputs, scales, n_fwd, n_layers, fwd_batch_sizes,
                all_token_ids, prompt, seed, manifest, final_residuals_raw,
                n_prompt_tokens,
                x_attn_inputs=x_attn_inputs if x_attn_inputs else None,
                lp_hidden_raw=lp_hidden_raw,
            )

        # Log LP hidden captures (decode boundary).
        # Store on self so callers (e.g. diagnostics) can access after chat().
        self._last_lp_hidden = lp_hidden_raw
        if lp_hidden_raw:
            lp_total_bytes = sum(t.nelement() * t.element_size() for t in lp_hidden_raw)
            logger.info(
                "verilm: LP hidden captured — %d forward passes, %.1f KB total (bf16)",
                len(lp_hidden_raw), lp_total_bytes / 1024,
            )

        # Attach witnessed scores if score witnessing was active.
        if witnessed_scores is not None:
            state.set_witnessed_scores(witnessed_scores, cap._num_heads)
            logger.info(
                "verilm: attached witnessed scores — %d layers, %.1f MB",
                len(witnessed_scores),
                sum(s.nbytes for s in witnessed_scores) / 1e6,
            )

        if _chat_timers:
            _ct_commit = time.monotonic()

        # Store for audit.
        import uuid
        request_id = str(uuid.uuid4())
        self._store_audit(request_id, state, generated_text)

        if _chat_timers:
            _ct_store = time.monotonic()

        commitment = json.loads(state.commitment_json())

        if _chat_timers:
            _ct_json = time.monotonic()

        response = {
            "request_id": request_id,
            "commitment": commitment,
            "token_ids": all_token_ids,
            "kv_roots": state.kv_roots_hex(),
            "generated_text": generated_text,
            "n_tokens": state.n_tokens(),
        }

        if _chat_timers:
            _ct_end = time.monotonic()
            n_tok = len(gen_token_ids)
            logger.info(
                "verilm chat timers: generate=%.1fms sync=%.1fms "
                "el_drain=%.1fms fr_drain=%.1fms buf_drain=%.1fms "
                "prep=%.1fms manifest=%.1fms commit=%.1fms "
                "store=%.1fms json=%.1fms response=%.1fms "
                "total=%.1fms (%d tokens)",
                (_ct_gen - _ct0) * 1000,
                (_ct_sync - _ct_gen) * 1000,
                (_ct_el_drain - _ct_sync) * 1000,
                (_ct_fr_drain - _ct_el_drain) * 1000,
                (_ct_buf_drain - _ct_fr_drain) * 1000,
                (_ct_prep - _ct_buf_drain) * 1000,
                (_ct_manifest - _ct_prep) * 1000,
                (_ct_commit - _ct_manifest) * 1000,
                (_ct_store - _ct_commit) * 1000,
                (_ct_json - _ct_store) * 1000,
                (_ct_end - _ct_json) * 1000,
                (_ct_end - _ct0) * 1000,
                n_tok,
            )

        return response

    def _commit_minimal(self, o_inputs, scales, n_fwd, n_layers,
                         fwd_batch_sizes, all_token_ids, prompt, seed, manifest,
                         final_residuals_raw=None, n_prompt_tokens=None,
                         x_attn_inputs=None, lp_hidden_raw=None):
        """V4 retained-state commitment path (no _int_mm, no full traces).

        Args:
            o_inputs: list of CPU tensors (one per o_proj call, n_fwd * n_layers).
            scales: numpy float32 array of per-row scale values. Per-row means
                    prefill calls contribute batch_size values per projection.
                    Already bulk-transferred from GPU at drain time.
            x_attn_inputs: optional list of CPU tensors (one per qkv_proj call,
                    n_fwd * n_layers). Used for precision corridor measurement.
            lp_hidden_raw: optional list of CPU tensors (one per decode step).
                    LP hidden at LogitsProcessor input, bf16, shape (1, hidden_dim).
        """
        import torch
        import verilm_rs
        import numpy as np

        o_proj_inputs = [inp.numpy() for inp in o_inputs]
        x_attn_np = [inp.numpy() for inp in x_attn_inputs] if x_attn_inputs else None

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

        # Organize LP hidden: LogitsProcessor hook captures (batch, hidden_dim) bf16
        # per forward pass. Expand by fwd_batch_sizes to get one entry per token,
        # matching the per-token indexing in open_v4.
        lp_hidden_list = None
        if lp_hidden_raw:
            if len(lp_hidden_raw) != len(fwd_batch_sizes):
                logger.warning(
                    "verilm: lp_hidden count (%d) != fwd count (%d), skipping",
                    len(lp_hidden_raw), len(fwd_batch_sizes),
                )
            else:
                lp_hidden_list = []
                for fwd_tensor, batch_sz in zip(lp_hidden_raw, fwd_batch_sizes):
                    for pos in range(batch_sz):
                        if fwd_tensor.dim() >= 2 and fwd_tensor.shape[0] >= batch_sz:
                            row = fwd_tensor[pos:pos+1].contiguous()
                        else:
                            row = fwd_tensor
                        arr = row.view(torch.int16).numpy().view(np.uint16).ravel()
                        lp_hidden_list.append(arr)

        return verilm_rs.commit_minimal_from_captures(
            o_proj_inputs=o_proj_inputs,
            scales=scales,
            n_layers=n_layers,
            fwd_batch_sizes=fwd_batch_sizes,
            token_ids=[int(t) for t in all_token_ids[1:]],
            prompt=prompt.encode(),
            sampling_seed=seed,
            manifest=manifest,
            weight_provider=self._weight_provider,
            final_residuals=final_residuals,
            n_prompt_tokens=n_prompt_tokens,
            x_attn_inputs=x_attn_np,
            lp_hidden_bf16=lp_hidden_list,
        )

    def _commit_minimal_packed(self, o_inputs, scales, n_fwd, n_layers,
                               fwd_batch_sizes, all_token_ids, prompt, seed, manifest,
                               final_residuals_raw=None, n_prompt_tokens=None):
        """Packed V4 commit: passes contiguous buffers to Rust, avoiding
        per-entry Python→Rust crossing and intermediate Vec allocations.

        Args:
            o_inputs: list of CPU tensors (one per o_proj call, n_fwd * n_layers).
            scales: numpy float32 array of per-row scale values. Per-row means
                    prefill calls contribute batch_size values per projection.
                    Already bulk-transferred from GPU at drain time.
        """
        import numpy as np
        import verilm_rs
        from . import capture as cap

        hidden_dim = cap._hidden_size
        timers = self._chat_timers
        if timers:
            import time as _t
            t0 = _t.monotonic()

        # o_inputs already ordered fwd-major × layer-major.
        a_arrays = [inp.numpy().ravel() for inp in o_inputs]

        if timers:
            t_numpy = _t.monotonic()
            # scales already extracted at drain time (GPU bulk D2H)
            t_scales = t_numpy

        packed_a = np.concatenate(a_arrays)

        if timers:
            t_concat = _t.monotonic()

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
            packed_scales=scales,
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
            n_prompt_tokens=n_prompt_tokens,
        )

        if timers:
            t_rust = _t.monotonic()
            n_tok = sum(fwd_batch_sizes)
            logger.info(
                "verilm commit timers: numpy=%.1fms scales=%.1fms concat=%.1fms "
                "pack_fr=%.1fms rust=%.1fms total=%.1fms (%d tokens, %.2fms/tok)",
                (t_numpy - t0) * 1000,
                (t_scales - t_numpy) * 1000,
                (t_concat - t_scales) * 1000,
                (t_pack_fr - t_concat) * 1000,
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
        deep_prefix: bool = False,
        use_captured_x_attn: Optional[bool] = None,
    ):
        """Open an audit proof.

        Returns binary (bincode+zstd) by default, or JSON string when
        binary=False (debug only).

        Args:
            request_id: from the /chat response.
            token_index: which token to audit.
            layer_indices: which layers to open. The verifier chooses.
            tier: "routine" (shell checks) or "full" (shell + attention replay).
            binary: if True (default), return bincode+zstd bytes. False for JSON debug.
            deep_prefix: if True, open prefix tokens for deep-prefix replay.
            use_captured_x_attn: DIAGNOSTIC ONLY. Overrides the automatic
                x_attn source selection. Do not set in production — the server
                chooses the correct source based on the verification profile.
                Ignored when QKV Freivalds is enabled (would cause mismatch).
        """
        entry = self._audit_store.get(request_id)
        if entry is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        if time.time() > entry["expires_at"]:
            del self._audit_store[request_id]
            raise KeyError(f"Audit state expired for: {request_id}")

        state = entry["state"]
        state.deep_prefix = deep_prefix

        # Automatic x_attn source selection based on verification profile.
        # - QKV Freivalds enabled (Llama): MUST use bridge-derived x_attn.
        #   Prover and verifier both derive x_attn from the residual chain,
        #   so QKV accumulators match for Freivalds checks.
        # - QKV Freivalds disabled (Qwen): use captured x_attn when available.
        #   Bridge diverges from GPU fused norm_quant; captured x_attn is
        #   needed for score/corridor paths.
        #
        # TODO(C): Once captured x_attn_i8 is bound into retained state and
        # Merkle leaf, captured x_attn can safely support QKV Freivalds too.
        if self._supports_qkv_freivalds:
            if use_captured_x_attn is True:
                logger.warning(
                    "Ignoring use_captured_x_attn=True: QKV Freivalds is enabled "
                    "for this model (%s). Using bridge-derived x_attn to ensure "
                    "prover/verifier agreement.",
                    self._model_type,
                )
            state.use_captured_x_attn = False
        else:
            # QKV Freivalds disabled — safe to use captured x_attn.
            if use_captured_x_attn is not None:
                state.use_captured_x_attn = use_captured_x_attn
            else:
                state.use_captured_x_attn = state.has_captured_x_attn()
        output_text = entry.get("output_text")

        if binary:
            return state.audit_v4_binary(token_index, layer_indices, output_text)
        return state.audit_v4(token_index, layer_indices, output_text)

    def _store_audit(self, request_id: str, state, output_text: str = ""):
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
            "output_text": output_text,
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
                temperature=float(request.get("temperature", 1.0)),
                top_k=int(request.get("top_k", 0)),
                top_p=float(request.get("top_p", 1.0)),
                min_tokens=int(request.get("min_tokens", 0)),
                ignore_eos=bool(request.get("ignore_eos", False)),
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
                deep_prefix=request.get("deep_prefix", False),
                # Diagnostic only — server.audit() enforces safe defaults
                # based on verification profile. Ignored when unsafe.
                use_captured_x_attn=request.get("use_captured_x_attn"),
            )
            # Binary response returns bytes.
            if isinstance(result, (bytes, memoryview)):
                return Response(
                    content=bytes(result),
                    media_type="application/octet-stream",
                )
            # JSON V4 returns a canonical JSON string produced by Rust.
            # Keep it opaque here so verifier-facing JSON paths can consume
            # the exact bytes instead of a Python parse/re-emit copy.
            if isinstance(result, str):
                return Response(
                    content=result,
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
