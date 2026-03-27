"""
Verification helpers for the VeriLM protocol.

Provides a tokenizer callback factory for canonical request→token
reconstruction inside verify_v4(). The callback replays the full
committed InputSpec: chat template, system prompt, BOS/EOS policy,
truncation, and special-token handling.
"""

import hashlib
from typing import Any, Callable, Dict, List, Optional


def make_tokenizer_fn(
    tokenizer: Any,
    system_prompt: str = "",
    max_model_len: Optional[int] = None,
) -> Callable[[bytes, Dict[str, Any]], List[int]]:
    """Build a tokenizer callback for verilm_rs.verify_v4().

    The returned callable has signature:
        fn(prompt: bytes, input_spec: dict) -> list[int]

    It replays the full committed InputSpec:
      - Verifies tokenizer_hash matches the expected tokenizer
      - Applies chat template if the tokenizer has one
      - Prepends system prompt if system_prompt_hash matches
      - Respects bos_eos_policy for add_special_tokens
      - Applies truncation_policy when max_model_len is set
      - Applies special_token_policy (encode or strip)

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer (or compatible).
        system_prompt: The system prompt text (empty string if none).
            Must match the system_prompt_hash committed in the InputSpec.
        max_model_len: Maximum model context length. Required for
            truncation_policy enforcement. None = no truncation.

    Returns:
        A callable suitable for the `tokenizer_fn` parameter of
        `verilm_rs.verify_v4()` and `verilm_rs.verify_v4_binary()`.

    Raises:
        Inside the callback: returns token IDs on success, raises
        RuntimeError on InputSpec mismatches (wrong tokenizer, wrong
        system prompt hash, etc.).

    Example:
        from transformers import AutoTokenizer
        from verilm.verify import make_tokenizer_fn
        import verilm_rs

        tok = AutoTokenizer.from_pretrained(
            "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"
        )
        tokenizer_fn = make_tokenizer_fn(tok, system_prompt="You are a helpful assistant.")

        result = verilm_rs.verify_v4(audit_json, key_json, tokenizer_fn=tokenizer_fn)
    """

    # Pre-compute hashes (same methods as server).
    _tokenizer_hash = _compute_tokenizer_hash(tokenizer)
    _system_prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()
    _chat_template = getattr(tokenizer, "chat_template", None)
    _chat_template_hash = (
        hashlib.sha256(_chat_template.encode()).hexdigest()
        if _chat_template else None
    )

    def _tokenize(prompt_bytes: bytes, input_spec: Dict[str, Any]) -> List[int]:
        text = prompt_bytes.decode("utf-8", errors="replace")

        # 1. Verify tokenizer identity.
        committed_tok_hash = input_spec.get("tokenizer_hash")
        if committed_tok_hash and committed_tok_hash != _tokenizer_hash:
            raise RuntimeError(
                f"tokenizer_hash mismatch: committed={committed_tok_hash} "
                f"local={_tokenizer_hash}"
            )

        # 2. Verify system prompt hash.
        committed_sp_hash = input_spec.get("system_prompt_hash")
        if committed_sp_hash and committed_sp_hash != _system_prompt_hash:
            raise RuntimeError(
                f"system_prompt_hash mismatch: committed={committed_sp_hash} "
                f"local={_system_prompt_hash}"
            )

        # 3. Verify chat template hash.
        committed_ct_hash = input_spec.get("chat_template_hash")
        if committed_ct_hash and committed_ct_hash != _chat_template_hash:
            raise RuntimeError(
                f"chat_template_hash mismatch: committed={committed_ct_hash} "
                f"local={_chat_template_hash}"
            )

        # 4. Apply chat template if the tokenizer has one.
        #    This is how vLLM processes prompts: if a chat template exists,
        #    it wraps the user message (and optional system prompt) before
        #    tokenization. The raw prompt from the server is the user message.
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": text})
            # apply_chat_template returns token IDs when tokenize=True,
            # or a formatted string when tokenize=False.
            # We need the string first, then tokenize with policy controls.
            templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text = templated

        # 5. Determine add_special_tokens from bos_eos_policy.
        bos_eos = input_spec.get("bos_eos_policy")
        if bos_eos == "none":
            add_special = False
        elif bos_eos in ("add_bos", "add_bos_eos", "add_eos", None):
            # When a chat template was applied, it typically already includes
            # BOS/EOS tokens in the template. HuggingFace tokenizers usually
            # set add_special_tokens=False internally when decoding a
            # chat-templated string. But since we're calling encode() on
            # the templated string, we follow the policy literally.
            #
            # For chat-templated text, the template itself controls BOS/EOS.
            # We set add_special=False to avoid double-adding.
            add_special = not bool(chat_template)
        else:
            raise RuntimeError(
                f"unknown bos_eos_policy='{bos_eos}' "
                f"(expected 'none', 'add_bos', 'add_bos_eos', 'add_eos', or absent)"
            )

        # 6. Apply special_token_policy.
        special_policy = input_spec.get("special_token_policy")
        if special_policy == "strip":
            # Strip known special tokens from the text before encoding.
            for tok_str in tokenizer.all_special_tokens:
                text = text.replace(tok_str, "")
        elif special_policy not in ("encode", "pass", None):
            raise RuntimeError(
                f"unknown special_token_policy='{special_policy}' "
                f"(expected 'strip', 'encode', 'pass', or absent)"
            )

        # 7. Tokenize.
        token_ids = tokenizer.encode(text, add_special_tokens=add_special)

        # 8. Apply truncation_policy.
        trunc_policy = input_spec.get("truncation_policy")
        if max_model_len and len(token_ids) > max_model_len:
            if trunc_policy == "left":
                token_ids = token_ids[-max_model_len:]
            elif trunc_policy == "right":
                token_ids = token_ids[:max_model_len]
            elif trunc_policy == "error":
                raise RuntimeError(
                    f"prompt has {len(token_ids)} tokens but max_model_len={max_model_len} "
                    f"and truncation_policy='error'"
                )
            else:
                raise RuntimeError(
                    f"unknown truncation_policy='{trunc_policy}' "
                    f"(expected 'left', 'right', 'error', or absent)"
                )
        elif trunc_policy not in ("left", "right", "error", None):
            raise RuntimeError(
                f"unknown truncation_policy='{trunc_policy}' "
                f"(expected 'left', 'right', 'error', or absent)"
            )

        return token_ids

    return _tokenize


def verify_detokenization(
    tokenizer: Any,
    token_ids: List[int],
    output_text: str,
    detokenization_policy: Optional[str] = None,
) -> List[str]:
    """Verify that output text matches detokenized token IDs under the committed policy.

    Decodes the given token IDs using the tokenizer, applies the committed
    detokenization_policy, and compares against the output text.

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer.
        token_ids: The generated token IDs (from the IO chain).
        output_text: The output text claimed by the prover.
        detokenization_policy: The committed policy string:
            - "default": decode with tokenizer defaults
            - "clean_spaces": decode with clean_up_tokenization_spaces=True
            - "raw": decode without cleanup
            - None: use tokenizer defaults (same as "default")

    Returns:
        List of failure descriptions (empty = pass).
    """
    failures = []

    if detokenization_policy == "clean_spaces":
        decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)
    elif detokenization_policy == "raw":
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    elif detokenization_policy in ("default", None):
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        raise RuntimeError(
            f"unknown detokenization_policy='{detokenization_policy}' "
            f"(expected 'default', 'clean_spaces', 'raw', or absent)"
        )

    if decoded != output_text:
        failures.append(
            f"detokenization mismatch (policy='{detokenization_policy}'): "
            f"decoded={decoded!r} vs claimed={output_text!r}"
        )

    return failures


def make_detokenizer_fn(
    tokenizer: Any,
) -> Callable[[List[int], Optional[str]], str]:
    """Build a detokenizer callback for verilm_rs.verify_v4().

    The returned callable has signature:
        fn(token_ids: list[int], policy: str|None) -> str

    It decodes token IDs back to text under the committed detokenization
    policy, suitable for the `detokenizer_fn` parameter of
    `verilm_rs.verify_v4()` and `verilm_rs.verify_v4_binary()`.

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer (or compatible).

    Returns:
        A callable suitable for the `detokenizer_fn` parameter.

    Example:
        from transformers import AutoTokenizer
        from verilm.verify import make_tokenizer_fn, make_detokenizer_fn
        import verilm_rs

        tok = AutoTokenizer.from_pretrained("neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8")
        tokenizer_fn = make_tokenizer_fn(tok, system_prompt="You are a helpful assistant.")
        detokenizer_fn = make_detokenizer_fn(tok)

        result = verilm_rs.verify_v4(
            audit_json, key_json,
            tokenizer_fn=tokenizer_fn,
            detokenizer_fn=detokenizer_fn,
        )
    """

    def _detokenize(token_ids: List[int], policy: Optional[str]) -> str:
        if policy == "clean_spaces":
            return tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)
        elif policy == "raw":
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        elif policy in ("default", None):
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            raise RuntimeError(
                f"unknown detokenization_policy='{policy}' "
                f"(expected 'default', 'clean_spaces', 'raw', or absent)"
            )

    return _detokenize


def _compute_tokenizer_hash(tokenizer: Any) -> str:
    """Compute SHA-256 hash of tokenizer vocabulary.

    Mirrors server._compute_tokenizer_hash: json.dumps(vocab, sort_keys=True).
    """
    import json

    try:
        vocab = tokenizer.get_vocab()
        vocab_str = json.dumps(vocab, sort_keys=True)
        return hashlib.sha256(vocab_str.encode()).hexdigest()
    except Exception:
        return hashlib.sha256(repr(tokenizer).encode()).hexdigest()
