"""
Custom vLLM worker that configures capture after model loading.

Use via worker_cls="verilm.worker.CaptureWorker" to get
layer/projection identification in TP worker subprocesses via call counting.
"""

import logging

logger = logging.getLogger("verilm")


def _get_worker_base():
    """Import the right Worker base class for the installed vLLM version."""
    # vLLM v1 (0.8+)
    try:
        from vllm.v1.worker.gpu_worker import Worker
        return Worker
    except ImportError:
        pass

    # Legacy
    from vllm.worker.worker import Worker
    return Worker


_WorkerBase = _get_worker_base()


class CaptureWorker(_WorkerBase):
    """Worker subclass that configures call-counting after model loading."""

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)

        try:
            from verilm.capture import configure_from_model
            model = self.model_runner.model
            configure_from_model(model)
        except Exception as e:
            logger.warning(
                "verilm: CaptureWorker failed to configure: %s", e
            )

        return result
