from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Protocol, TypedDict, final, override

import torch
from tensordict import tensorclass
from torch import Tensor
from torch.nn import Module
from torch.types import Number
from torch.utils._pytree import tree_leaves  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken, TensorTree, TokenType


class AttentionMaskLegendTree(TypedDict):
    DO_ATTEND: Tensor
    DO_NOT_ATTEND: Tensor


class AttentionMaskTree(TypedDict):
    mask: Tensor
    legend: AttentionMaskLegendTree


class AttentionMaskLegendProvider(Protocol):
    DO_ATTEND: Number
    DO_NOT_ATTEND: Number


@tensorclass
class AttentionMaskLegend:
    DO_ATTEND: Tensor
    DO_NOT_ATTEND: Tensor


@final
class TorchAttentionMaskLegend:
    DO_ATTEND = False
    DO_NOT_ATTEND = True


@final
class WandbAttentionMaskLegend:
    DO_ATTEND = 1.0
    DO_NOT_ATTEND = 0.0


@tensorclass
class AttentionMask:
    mask: Tensor
    legend: AttentionMaskLegend

    @classmethod
    def to_tensortree(
        cls, *, mask: Tensor, legend: type[AttentionMaskLegendProvider]
    ) -> AttentionMaskTree:
        return {
            "mask": mask,
            "legend": {
                "DO_ATTEND": torch.as_tensor(
                    legend.DO_ATTEND, dtype=mask.dtype, device=mask.device
                ),
                "DO_NOT_ATTEND": torch.as_tensor(
                    legend.DO_NOT_ATTEND, dtype=mask.dtype, device=mask.device
                ),
            },
        }

    @classmethod
    def from_tensortree(cls, tree: AttentionMaskTree) -> "AttentionMask":
        return cls(
            mask=torch.as_tensor(tree["mask"]),
            legend=AttentionMaskLegend(
                DO_ATTEND=torch.as_tensor(tree["legend"]["DO_ATTEND"]),
                DO_NOT_ATTEND=torch.as_tensor(tree["legend"]["DO_NOT_ATTEND"]),
            ),
        )


class AttentionMaskBuilder(Module, ABC):
    @abstractmethod
    def forward(
        self,
        *,
        index: TensorTree,
        timestep: dict[str, dict[str, Mapping[str, int]]],
        legend: type[AttentionMaskLegendProvider],
    ) -> Tensor: ...

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
    def _set(mask: Tensor, src: Tensor, dest: Tensor, val: Number | Tensor) -> None:
        if src.numel() == 0 or dest.numel() == 0:
            return

        grid = torch.meshgrid(src, dest, indexing="ij")
        mask[grid] = val


class CausalAttentionMaskBuilder(AttentionMaskBuilder):
    @override
    def forward(  # noqa: PLR0914
        self,
        *,
        index: TensorTree,
        timestep: dict[str, dict[str, Mapping[str, int]]],
        legend: type[AttentionMaskLegendProvider],
    ) -> Tensor:
        leaves = tree_leaves(index, lambda x: isinstance(x, Tensor))
        t = leaves[0].shape[0]
        length = t * sum(leaf.shape[1] for leaf in leaves)

        do_not_attend = legend.DO_NOT_ATTEND
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
            cur_obs = self._select_indices(index, step=step, keys=obs_keys)
            cur_foresight = self._select_indices(index, step=step, keys=foresight_keys)
            cur_obs_summary = self._select_indices(
                index, step=step, keys=obs_summary_key
            )
            cur_obs_history = self._select_indices(
                index, step=step, keys=obs_history_key
            )
            cur_actions = self._select_indices(index, step=step, keys=action_keys)
            cur_action_summary = self._select_indices(
                index, step=step, keys=action_summary_key
            )

            past_obs = self._select_indices(
                index, step=slice(None, step), keys=obs_keys
            )
            past_foresight = self._select_indices(
                index, step=slice(None, step), keys=foresight_keys
            )
            past_actions = self._select_indices(
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
                self._set(mask, src, dest, legend.DO_ATTEND)

        return mask
