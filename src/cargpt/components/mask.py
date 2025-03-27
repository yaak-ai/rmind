from enum import Enum, EnumMeta

import torch
from jaxtyping import Float
from tensordict import TensorClass
from torch import Tensor
from typing_extensions import Self

from cargpt.components.episode import Index


class AttentionMaskLegend(EnumMeta):
    DO_ATTEND: float
    DO_NOT_ATTEND: float

    def __new__(metacls, cls, bases, classdict):
        if bases != (required := (float, Enum)):
            msg = f"required bases: {required}"
            raise ValueError(msg)

        if set(classdict._member_names) != (required := metacls.__annotations__.keys()):
            msg = f"required attrs: {required}"
            raise ValueError(msg)

        return super().__new__(metacls, cls, bases, classdict)


class XFormersAttentionMaskLegend(float, Enum, metaclass=AttentionMaskLegend):
    DO_ATTEND = 0.0
    DO_NOT_ATTEND = float("-inf")


class WandbAttentionMaskLegend(float, Enum, metaclass=AttentionMaskLegend):
    DO_ATTEND = 1.0
    DO_NOT_ATTEND = 0.0


class AttentionMask(TensorClass):
    data: Float[Tensor, "seq seq"]  # pyright: ignore[reportIncompatibleMethodOverride]
    legend: AttentionMaskLegend

    def _set(self, *, src: Index, dest: Index, val) -> Self:
        i = src.cat_from_tensordict(dim=-1).flatten()
        j = dest.cat_from_tensordict(dim=-1).flatten()
        grid = torch.meshgrid(i, j, indexing="ij")
        self.data[grid] = val

        return self

    def _do_attend(self, src: Index, dest: Index) -> Self:
        return self._set(src=src, dest=dest, val=self.legend.DO_ATTEND)

    def _do_not_attend(self, src: Index, dest: Index) -> Self:
        return self._set(src=src, dest=dest, val=self.legend.DO_NOT_ATTEND)

    def with_legend(self, legend: AttentionMaskLegend) -> Self:
        mask = self.clone(recurse=True)

        if self.legend is legend:
            return mask

        attend = mask.data == self.legend.DO_ATTEND
        do_not_attend = mask.data == self.legend.DO_NOT_ATTEND

        mask.legend = legend
        mask.data[attend] = mask.legend.DO_ATTEND
        mask.data[do_not_attend] = mask.legend.DO_NOT_ATTEND

        return mask
