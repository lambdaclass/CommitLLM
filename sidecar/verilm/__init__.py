"""
VeriLM sidecar plugin for vLLM.

Captures INT8 matmul inputs and INT32 accumulators for Freivalds verification.
Restructures captures into LayerTrace format for Merkle tree commitment.
Loaded automatically via vllm.general_plugins entry point.

Enable/disable:
    VLLM_PLUGINS=verilm_capture vllm serve ...
    VERILM_CAPTURE=1 vllm serve ...       # capture on
    VERILM_CAPTURE=0 vllm serve ...       # capture off (plugin still loaded, no overhead)

After model loading:
    from verilm import configure_from_model, get_model_from_llm
    model = get_model_from_llm(llm)
    configure_from_model(model)  # configures layer count + model geometry
"""

from .capture import (
    register,
    get_capture_buffer,
    configure_layer_count,
    configure_from_model,
    get_model_from_llm,
    CaptureBuffer,
)
from .trace import (
    build_layer_traces,
    split_qkv,
    split_gate_up,
    requantize_i32_to_i8,
)
from .hooks import EmbeddingLogitCapture
from .server import VerifiedInferenceServer, create_app

__all__ = [
    "register",
    "get_capture_buffer",
    "configure_layer_count",
    "configure_from_model",
    "get_model_from_llm",
    "CaptureBuffer",
    "build_layer_traces",
    "split_qkv",
    "split_gate_up",
    "requantize_i32_to_i8",
    "EmbeddingLogitCapture",
    "VerifiedInferenceServer",
    "create_app",
]
