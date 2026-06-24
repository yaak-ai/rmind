from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

import tensorrt as trt
import torch
from pydantic import AfterValidator, FilePath, InstanceOf, validate_call
from structlog import get_logger
from torch import Tensor

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class TensorRTTensorInfo(NamedTuple):
    name: str
    dtype: Any
    shape: tuple[int, ...]


class TensorRTInferenceBackend:
    @validate_call
    def __init__(
        self,
        *,
        path: FilePath,
        device: Annotated[str, AfterValidator(torch.device)]
        | InstanceOf[torch.device] = "cuda",
    ) -> None:
        self._path: Path = path
        self._device = device
        self._engine = None
        self._context = None
        self._output_tensor_info = []

    def on_predict_start(self) -> None:
        logger.debug("deserializing TensorRT engine", device=self._device)
        trt_logger = trt.Logger(trt.Logger.WARNING)  # ty:ignore[unresolved-attribute]
        runtime = trt.Runtime(trt_logger)  # ty:ignore[unresolved-attribute]
        self._engine = runtime.deserialize_cuda_engine(self._path.read_bytes())
        if self._engine is None:
            msg = "failed to deserialize TensorRT engine"
            raise RuntimeError(msg)

        self._context = self._engine.create_execution_context()

        self._output_tensor_info = []
        for tensor_index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(tensor_index)
            mode = self._engine.get_tensor_mode(name)
            tensor_info = TensorRTTensorInfo(
                name=name,
                dtype=self._engine.get_tensor_dtype(name),
                shape=tuple(self._engine.get_tensor_shape(name)),
            )
            logger.debug(
                "engine tensor",
                mode=mode.name,
                name=tensor_info.name,
                dtype=tensor_info.dtype.name,
                shape=tensor_info.shape,
            )

            if mode == trt.TensorIOMode.OUTPUT:  # ty:ignore[unresolved-attribute]
                self._output_tensor_info.append(tensor_info)

    def forward(self, input: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        if self._context is None:
            msg = "TensorRT execution context has not been initialized"
            raise RuntimeError(msg)

        input = {k: v.to(self._device).contiguous() for k, v in input.items()}
        output = {
            info.name: torch.empty(
                info.shape,
                dtype=getattr(torch, info.dtype.name.lower()),
                device=self._device,
            )
            for info in self._output_tensor_info
        }

        for name, tensor in input.items():
            if not self._context.set_input_shape(name, tuple(tensor.shape)):
                msg = f"failed to set TensorRT input shape for {name!r}: {tuple(tensor.shape)}"
                raise RuntimeError(msg)

        for tensors in (input, output):
            for name, tensor in tensors.items():
                self._context.set_tensor_address(name, tensor.data_ptr())

        cuda_stream = torch.cuda.current_stream(self._device).cuda_stream
        if not self._context.execute_async_v3(cuda_stream):
            msg = "TensorRT execution failed"
            raise RuntimeError(msg)

        return output
