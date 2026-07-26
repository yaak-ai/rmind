from typing import TYPE_CHECKING, Any

from .onnx import OnnxInferenceBackend

if TYPE_CHECKING:
    from .tensorrt import TensorRTInferenceBackend

__all__ = ["OnnxInferenceBackend", "TensorRTInferenceBackend"]


def __getattr__(name: str) -> Any:
    # Lazily import the TensorRT backend so `OnnxInferenceBackend` (and its users,
    # e.g. flow_export_compare) don't require `tensorrt` to be importable.
    if name == "TensorRTInferenceBackend":
        from .tensorrt import TensorRTInferenceBackend  # noqa: PLC0415

        return TensorRTInferenceBackend
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
