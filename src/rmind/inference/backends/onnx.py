from collections.abc import Mapping, Sequence
from typing import Any

import onnxruntime as ort
import torch
from pydantic import FilePath, InstanceOf, validate_call
from structlog import get_logger
from torch import Tensor

logger = get_logger(__name__)


class OnnxInferenceBackend:
    @validate_call
    def __init__(
        self,
        *,
        path: FilePath | bytes,
        sess_options: InstanceOf[ort.SessionOptions] | None = None,
        providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        **kwargs: object,
    ) -> None:
        self._path = path
        self._sess_options = sess_options
        self._providers = providers
        self._provider_options = provider_options
        self._kwargs = kwargs

        self._session: Any | None = None
        self._output_names: list[str] = []

    def on_predict_start(self) -> None:
        logger.debug("instantiating inference session")
        self._session = ort.InferenceSession(
            path_or_bytes=self._path,
            sess_options=self._sess_options,
            providers=self._providers,
            provider_options=self._provider_options,
            **self._kwargs,
        )

        for session_input in self._session.get_inputs():
            logger.debug(
                "session input",
                name=session_input.name,
                type=session_input.type,
                shape=session_input.shape,
            )

        session_outputs = self._session.get_outputs()
        self._output_names = [node.name for node in session_outputs]
        for session_output in session_outputs:
            logger.debug(
                "session output",
                name=session_output.name,
                type=session_output.type,
                shape=session_output.shape,
            )

    def forward(self, input: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        if self._session is None:
            msg = "ONNX inference session has not been initialized"
            raise RuntimeError(msg)

        input_np = {k: v.cpu().numpy() for k, v in input.items()}
        output_np = self._session.run(self._output_names, input_np)
        output = [torch.from_numpy(x) for x in output_np]

        return dict(zip(self._output_names, output, strict=True))
