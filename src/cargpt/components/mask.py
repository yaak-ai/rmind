from enum import Enum, EnumMeta, _EnumDict  # pyright: ignore[reportPrivateUsage]
from typing import Self

import torch
from tensordict import TensorClass
from torch import Tensor

from cargpt.components.episode import Index


class AttentionMaskLegend(EnumMeta):
    DO_ATTEND: float  # pyright: ignore[reportUninitializedInstanceVariable]
    DO_NOT_ATTEND: float  # pyright: ignore[reportUninitializedInstanceVariable]

    def __new__(metacls, cls: str, bases: tuple[type, ...], classdict: _EnumDict):  # noqa: ANN204
        if bases != (required := (float, Enum)):
            msg = f"required bases: {required}"
            raise ValueError(msg)

        if set(classdict._member_names) != (required := metacls.__annotations__.keys()):  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001
            msg = f"required attrs: {required}"
            raise ValueError(msg)

        return super().__new__(metacls, cls, bases, classdict)


class XFormersAttentionMaskLegend(float, Enum, metaclass=AttentionMaskLegend):
    DO_ATTEND: float = 0.0
    DO_NOT_ATTEND: float = float("-inf")


class WandbAttentionMaskLegend(float, Enum, metaclass=AttentionMaskLegend):
    DO_ATTEND: float = 1.0
    DO_NOT_ATTEND: float = 0.0


class AttentionMask(TensorClass):
    mask: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    legend: AttentionMaskLegend  # pyright: ignore[reportUninitializedInstanceVariable]

    def _set(self, *, src: Index, dest: Index, val: float) -> Self:
        i = src.cat_from_tensordict(dim=-1).flatten()
        j = dest.cat_from_tensordict(dim=-1).flatten()
        grid = torch.meshgrid(i, j, indexing="ij")
        self.mask[grid] = val

        return self

    def do_attend(self, src: Index, dest: Index) -> Self:
        return self._set(src=src, dest=dest, val=self.legend.DO_ATTEND)

    def do_not_attend(self, src: Index, dest: Index) -> Self:
        return self._set(src=src, dest=dest, val=self.legend.DO_NOT_ATTEND)

    def with_legend(self, legend: AttentionMaskLegend) -> Self:
        mask = self.clone(recurse=True)

        if self.legend is legend:
            return mask

        attend = mask.mask == self.legend.DO_ATTEND
        do_not_attend = mask.mask == self.legend.DO_NOT_ATTEND

        mask.legend = legend
        mask.mask[attend] = mask.legend.DO_ATTEND
        mask.mask[do_not_attend] = mask.legend.DO_NOT_ATTEND

        return mask
