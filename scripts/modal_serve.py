"""
Verified inference on Modal: vLLM + verilm + verilm_rs (PyO3).

Single GPU container, no sidecar process, no HTTP intermediary.
Capture -> Rust commitment happens in-process via PyO3 FFI.

Endpoints:
    POST /chat   -- prompt -> verified response + commitment
    POST /audit  -- request_id + challenge -> compact proof (zstd)
    GET  /health -- liveness

Usage:
    modal deploy scripts/modal_serve.py
    curl -X POST https://<app>--inference-chat.modal.run \
      -H 'Content-Type: application/json' \
      -d '{"prompt": "What is 2+2?", "n_tokens": 4}'
"""

import modal

app = modal.App("vi-verified-inference")

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "ca-certificates")
    # Install Rust for building verilm_rs (PyO3).
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VI_CAPTURE": "1",
    })
    .pip_install("vllm>=0.8", "torch", "numpy", "fastapi", "maturin")
    # Install verilm Python package (capture plugin).
    .add_local_dir("sidecar", remote_path="/opt/verilm", copy=True)
    .run_commands(
        "pip install -e /opt/verilm",
        "python -c \""
        "import site, os; "
        "d = site.getsitepackages()[0]; "
        "open(os.path.join(d, 'verilm_capture.pth'), 'w')"
        ".write('import verilm._startup\\n')\"",
    )
    # Build and install verilm_rs (PyO3 bridge to Rust commitment engine).
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin develop --release",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
class Inference:
    """Verified inference: vLLM + verilm capture + Rust commitment."""

    @modal.enter()
    def setup(self):
        import os
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        from vllm import LLM
        from verilm.server import VerifiedInferenceServer

        print(f"Loading {MODEL_ID}...")
        self.llm = LLM(
            model=MODEL_ID, dtype="auto",
            max_model_len=2048, enforce_eager=True,
        )
        self.server = VerifiedInferenceServer(self.llm)
        print("VerifiedInferenceServer ready")

    @modal.fastapi_endpoint(method="POST")
    def chat(self, request: dict):
        import traceback
        try:
            return self.server.chat(
                prompt=request.get("prompt", ""),
                max_tokens=request.get("n_tokens", 4),
            )
        except Exception as e:
            print(f"Chat error: {traceback.format_exc()}")
            return {"error": str(e)}

    @modal.fastapi_endpoint(method="POST")
    def audit(self, request: dict):
        from fastapi.responses import Response
        try:
            if "token_index" not in request or "layer_indices" not in request:
                return {"error": "token_index and layer_indices are required"}
            proof_bytes = self.server.audit(
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
            return {"error": str(e)}

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok", "model": MODEL_ID}
