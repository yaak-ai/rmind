from functools import cached_property
from typing import ClassVar, Literal

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict
from tensordict import (
    NonTensorData,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    TensorClass,
    TensorDict,
)
from torch import Tensor


class BatchMeta(TensorClass, autocast=True):  # pyright: ignore[reportGeneralTypeIssues, reportCallIssue]
    sample_idx: Tensor | None = None
    input_id: NonTensorData | None = None  # pyright: ignore[reportUnknownVariableType]


class Batch(TensorClass, autocast=True):  # pyright: ignore[reportGeneralTypeIssues, reportCallIssue]
    data: TensorDict | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    meta: BatchMeta | None = None


BatchKeys = frozenset[
    Literal["data", "meta"]
    | tuple[Literal["data"], str]
    | tuple[Literal["meta"], Literal["sample_idx", "input_id"]]
]

BATCH_KEYS_DEFAULT = frozenset(("data", "meta"))



class BaseModel(_BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        ignored_types=(cached_property,),
    )
