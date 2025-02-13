from collections.abc import Callable, Mapping, Sequence
from typing import override

from optree import tree_map
from pydantic import ConfigDict, validate_call
from rbyte.batch import Batch
from rbyte.config import BaseModel
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module

from cargpt.components.episode import Modality

type Keys = Mapping[Modality, Mapping[str, tuple[str, ...]]]


class Transform(BaseModel):
    select: tuple[str | tuple[str, ...], ...]
    apply: Callable[[Tensor], Tensor]


class InputBuilder(Module):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, keys: Keys, transforms: Sequence[Transform]) -> None:
        super().__init__()

        self._keys = keys
        self._transforms = transforms

    @property
    def keys(self) -> Keys:
        return self._keys

    @override
    def forward(self, batch: Batch) -> TensorDict:
        input: TensorDict = (
            TensorDict(
                tree_map(
                    lambda k: batch.get(k, default=None),
                    self._keys,  # pyright: ignore[reportArgumentType]
                    is_leaf=lambda x: isinstance(x, tuple),
                ),
                device=batch.device,
            )
            .filter_non_tensor_data()
            .filter_empty_()
            .auto_batch_size_(batch_dims=2)
            .refine_names("b", "t")
        )

        for transform in self._transforms:
            input = input.update(
                input.select(*transform.select, strict=False).apply(
                    transform.apply, inplace=False
                )
            )

        return input.lock_()
