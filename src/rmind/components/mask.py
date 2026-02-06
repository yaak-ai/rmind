from collections.abc import Mapping, Sequence
from enum import Enum, EnumMeta
from typing import Any, ClassVar, Protocol, final, runtime_checkable

import torch
from torch import Tensor
from torch.types import Number
from torch.utils._pytree import tree_leaves  # noqa: PLC2701

from rmind.components.base import TensorTree
from rmind.components.tokens import Modality, SummaryToken, TokenType


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


class AttentionMaskBuilder:
    @classmethod
    def build(  # noqa: PLR0914
        cls,
        *,
        index: TensorTree,
        timestep: dict[str, dict[str, Mapping[str, int]]],
        legend: type[AttentionMaskLegend] = TorchAttentionMaskLegend,
    ) -> Tensor:
        leaves = tree_leaves(index, lambda x: isinstance(x, Tensor))
        t = leaves[0].shape[0]
        length = t * sum(leaf.shape[1] for leaf in leaves)

        do_not_attend = legend.DO_NOT_ATTEND.value
        mask = torch.full(
            size=(length, length),
            fill_value=do_not_attend,
            dtype=torch.as_tensor(do_not_attend).dtype,
            device=leaves[0].device,  # for export path
        )

        obs_keys = tuple(
            (modality, name)
            for modality, names in timestep[TokenType.OBSERVATION.value].items()
            for name in names
        )
        action_keys = tuple(
            (modality, name)
            for modality, names in timestep[TokenType.ACTION.value].items()
            for name in names
        )
        foresight_keys = tuple(
            (Modality.FORESIGHT.value, name) for name in index[Modality.FORESIGHT.value]
        )
        obs_summary_key = (
            (Modality.SUMMARY.value, SummaryToken.OBSERVATION_SUMMARY.value),
        )
        obs_history_key = (
            (Modality.SUMMARY.value, SummaryToken.OBSERVATION_HISTORY.value),
        )
        action_summary_key = (
            (Modality.SUMMARY.value, SummaryToken.ACTION_SUMMARY.value),
        )

        for step in range(t):
            cur_obs = cls._select_indices(index, step=step, keys=obs_keys)
            cur_foresight = cls._select_indices(index, step=step, keys=foresight_keys)
            cur_obs_summary = cls._select_indices(
                index, step=step, keys=obs_summary_key
            )
            cur_obs_history = cls._select_indices(
                index, step=step, keys=obs_history_key
            )
            cur_actions = cls._select_indices(index, step=step, keys=action_keys)
            cur_action_summary = cls._select_indices(
                index, step=step, keys=action_summary_key
            )

            past_obs = cls._select_indices(index, step=slice(None, step), keys=obs_keys)
            past_foresight = cls._select_indices(
                index, step=slice(None, step), keys=foresight_keys
            )
            past_actions = cls._select_indices(
                index, step=slice(None, step), keys=action_keys
            )

            for src, dest in (
                (cur_obs, cur_obs),
                (cur_obs, past_obs),
                (cur_foresight, cur_obs),
                (cur_foresight, cur_foresight),
                (cur_foresight, past_obs),
                (cur_obs_summary, cur_foresight),
                (cur_obs_summary, cur_obs_summary),
                (cur_obs_summary, past_foresight),
                (cur_obs_history, cur_foresight),
                (cur_obs_history, cur_obs_history),
                (cur_obs_history, past_foresight),
                (cur_actions, cur_actions),
                (cur_actions, past_actions),
                (cur_action_summary, cur_actions),
                (cur_action_summary, cur_action_summary),
                (cur_action_summary, past_actions),
            ):
                cls._set(mask, src, dest, legend.DO_ATTEND.value)

        return mask

    @staticmethod
    def _select_indices(
        index: TensorTree, *, step: int | slice, keys: Sequence[tuple[str, str]]
    ) -> Tensor:
        leaves = tree_leaves(index, lambda x: isinstance(x, Tensor))
        chunks = [index[modality][name][step].reshape(-1) for modality, name in keys]  # ty:ignore[invalid-argument-type]
        if not chunks:
            return leaves[0].new_empty(0)

        return torch.cat(chunks)

    @staticmethod
    def _set(mask: Tensor, src: Tensor, dest: Tensor, val: Any) -> None:
        if src.numel() == 0 or dest.numel() == 0:
            return

        grid = torch.meshgrid(src, dest, indexing="ij")
        mask[grid] = val
