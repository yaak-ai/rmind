from __future__ import annotations

from enum import Enum, EnumMeta
from typing import Any, ClassVar, Protocol, final, runtime_checkable

from typing_extensions import Self

import torch
from tensordict import TensorClass
from torch import Tensor
from torch.types import Number

from rmind.components.episode import Index


class AttentionMaskLegendMeta(EnumMeta):
    MEMBERS: ClassVar[set[str]] = {"DO_ATTEND", "DO_NOT_ATTEND"}

    def __new__(metacls, *args: Any, **kwargs: Any) -> type:
        cls = super().__new__(metacls, *args, **kwargs)
        if set(cls.__members__.keys()) != (required := metacls.MEMBERS):
            msg = f"{cls} must define exactly the following members: {required}"
            raise ValueError(msg)

        return cls


@runtime_checkable
class MaskValue(Protocol):
    value: Number


@runtime_checkable
class AttentionMaskLegend(Protocol):
    DO_ATTEND: MaskValue
    DO_NOT_ATTEND: MaskValue


@final
class TorchAttentionMaskLegend(Enum, metaclass=AttentionMaskLegendMeta):
    DO_ATTEND = False
    DO_NOT_ATTEND = True


@final
class WandbAttentionMaskLegend(Enum, metaclass=AttentionMaskLegendMeta):
    DO_ATTEND = 1.0
    DO_NOT_ATTEND = 0.0


class AttentionMask(TensorClass):
    mask: Tensor
    legend: AttentionMaskLegend

    def _set(self, *, src: Index, dest: Index, val: Number) -> Self:
        i = src.cat_from_tensordict(dim=-1).flatten()
        j = dest.cat_from_tensordict(dim=-1).flatten()
        grid = torch.meshgrid(i, j, indexing="ij")
        self.mask[grid] = val

        return self

    def do_attend(self, src: Index, dest: Index) -> Self:
        return self._set(src=src, dest=dest, val=self.legend.DO_ATTEND.value)

    def do_not_attend(self, src: Index, dest: Index) -> Self:
        return self._set(src=src, dest=dest, val=self.legend.DO_NOT_ATTEND.value)
