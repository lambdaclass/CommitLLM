"""
Verification helpers for the VeriLM protocol.

Provides a tokenizer callback factory for canonical request→token
reconstruction inside verify_v4().
"""

from typing import Any, Dict, List


def make_tokenizer_fn(tokenizer: Any):
    """Build a tokenizer callback for verilm_rs.verify_v4().

    The returned callable has signature:
        fn(prompt: bytes, input_spec: dict) -> list[int]

    It uses the given HuggingFace tokenizer to encode raw prompt bytes
    into token IDs, respecting the InputSpec's bos_eos_policy.

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer (or compatible).

    Returns:
        A callable suitable for the `tokenizer_fn` parameter of
        `verilm_rs.verify_v4()` and `verilm_rs.verify_v4_binary()`.

    Example:
        from transformers import AutoTokenizer
        from verilm.verify import make_tokenizer_fn
        import verilm_rs

        tok = AutoTokenizer.from_pretrained("neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8")
        tokenizer_fn = make_tokenizer_fn(tok)

        result = verilm_rs.verify_v4(audit_json, key_json, tokenizer_fn=tokenizer_fn)
    """

    def _tokenize(prompt_bytes: bytes, input_spec: Dict[str, Any]) -> List[int]:
        text = prompt_bytes.decode("utf-8", errors="replace")

        # Respect bos_eos_policy from the committed InputSpec.
        policy = input_spec.get("bos_eos_policy")
        add_special = True
        if policy == "none":
            add_special = False
        elif policy == "add_bos":
            add_special = True  # tokenizer default usually adds BOS

        token_ids = tokenizer.encode(text, add_special_tokens=add_special)
        return token_ids

    return _tokenize
