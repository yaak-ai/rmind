from collections.abc import Mapping
from typing import Annotated, Any, Protocol, override, runtime_checkable

import pytorch_lightning as pl
from pydantic import Field, InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Identity, Module


@runtime_checkable
class InferenceBackend(Protocol):
    def on_predict_start(self) -> None: ...
    def forward(self, input: Mapping[str, Tensor]) -> Mapping[str, Tensor]: ...


class LightningInferenceWrapper(pl.LightningModule):
    @validate_call
    def __init__(
        self,
        *,
        backend: InstanceOf[InferenceBackend],
        input_transform: Annotated[InstanceOf[Module], Field(default_factory=Identity)],
        output_transform: Annotated[
            InstanceOf[Module], Field(default_factory=Identity)
        ],
    ) -> None:
        super().__init__()

        self._backend = backend
        self._input_transform = input_transform
        self._output_transform = output_transform

    @override
    def on_predict_start(self) -> None:
        self._backend.on_predict_start()

    @override
    def predict_step(self, input: Any) -> TensorDict:
        input_transformed = self._input_transform(input)
        output = self._backend.forward(input_transformed)
        output_transformed = self._output_transform(output)

        return TensorDict(output_transformed).auto_batch_size_(1)
