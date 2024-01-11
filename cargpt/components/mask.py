from enum import Enum, EnumMeta

import more_itertools as mit
import torch
from jaxtyping import Float
from tensordict import tensorclass
from torch import Tensor
from typing_extensions import Self

from cargpt.components.episode import EpisodeIndex, Timestep


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


@tensorclass  # pyright: ignore
class AttentionMask:
    data: Float[Tensor, "seq seq"]
    legend: AttentionMaskLegend

    def _set(
        self,
        *,
        src: EpisodeIndex,
        dest: EpisodeIndex,
        val,
    ) -> Self:
        grid = torch.meshgrid(src.all_values, dest.all_values, indexing="ij")
        self.data[grid] = val

        return self

    def _do_attend(self, src: EpisodeIndex, dest: EpisodeIndex) -> Self:
        return self._set(src=src, dest=dest, val=self.legend.DO_ATTEND)

    def _do_not_attend(self, src: EpisodeIndex, dest: EpisodeIndex) -> Self:
        return self._set(src=src, dest=dest, val=self.legend.DO_NOT_ATTEND)

    def with_legend(self, legend: AttentionMaskLegend) -> Self:
        mask = self.clone(recurse=True)  # pyright: ignore
        if self.legend is legend:
            return mask

        attend = mask.data == self.legend.DO_ATTEND
        do_not_attend = mask.data == self.legend.DO_NOT_ATTEND

        mask.legend = legend
        mask.data[attend] = mask.legend.DO_ATTEND
        mask.data[do_not_attend] = mask.legend.DO_NOT_ATTEND

        return mask


@tensorclass  # pyright: ignore
class TimestepWiseCausalAttentionMask(AttentionMask):
    @classmethod
    def build(
        cls,
        *,
        index: EpisodeIndex,
        legend: AttentionMaskLegend,
    ) -> Self:
        device = index.device  # pyright: ignore
        mask = cls(
            data=torch.full((index.max + 1, index.max + 1), torch.nan),
            legend=legend,
            batch_size=[],
            device=device,
        )

        step_count = mit.one(index.batch_size)  # pyright: ignore
        for step in range(step_count):
            past, current, future = index[:step], index[step], index[step + 1 :]  # pyright: ignore
            mask = (
                mask._do_attend(current, current)
                ._do_attend(future, current)
                ._do_not_attend(past, current)
            )

        if mask.data.isnan().any().item():
            msg = "some values not set"
            raise RuntimeError(msg)

        return mask


@tensorclass  # pyright: ignore
class InverseDynamicsAttentionMask(TimestepWiseCausalAttentionMask):
    @classmethod
    def build(
        cls,
        *,
        index: EpisodeIndex,
        timestep: Timestep,
        legend: AttentionMaskLegend,
    ) -> Self:
        mask = super().build(index=index, legend=legend)

        observations, actions = timestep.observations, timestep.actions

        step_count = mit.one(index.batch_size)  # pyright: ignore
        for step in range(step_count):
            current, future = index[step], index[step + 1 :]  # pyright: ignore

            current_observations = current.select(*observations)
            current_actions = current.select(*actions)
            future_observations = future.select(*observations)

            mask = mask._do_not_attend(
                current_observations,
                current_actions,
            )._do_not_attend(
                future_observations,
                current_actions,
            )

        return mask


@tensorclass  # pyright: ignore
class NonCausalAttentionMask(AttentionMask):
    @classmethod
    def build(
        cls,
        *,
        index: EpisodeIndex,
        legend: AttentionMaskLegend,
    ) -> Self:
        device = index.device  # pyright: ignore
        return cls(
            data=torch.full(
                (index.max + 1, index.max + 1),
                fill_value=legend.DO_ATTEND,
                dtype=torch.float,
                device=device,
            ),
            legend=legend,
            batch_size=[],
            device=device,
        )
