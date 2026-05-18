import operator
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Protocol, Self, final, override

import torch
from einops import rearrange
from tensordict import tensorclass
from torch import Tensor
from torch.nn import Module
from torch.types import Number
from torch.utils._pytree import tree_leaves, tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken, TensorTree, TokenType


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


@tensorclass
class AttentionMask:
    mask_tensor: Tensor
    legend: AttentionMaskLegend

    @classmethod
    def from_tensor(
        cls, *, mask_tensor: Tensor, legend: type[AttentionMaskLegendProvider]
    ) -> Self:
        return cls(
            mask_tensor=torch.as_tensor(mask_tensor),
            legend=AttentionMaskLegend(
                DO_ATTEND=torch.as_tensor(
                    legend.DO_ATTEND, dtype=mask_tensor.dtype, device=mask_tensor.device
                ),
                DO_NOT_ATTEND=torch.as_tensor(
                    legend.DO_NOT_ATTEND,
                    dtype=mask_tensor.dtype,
                    device=mask_tensor.device,
                ),
            ),
        )


@tensorclass
class FactorizedAttentionMask:
    spatial: AttentionMask
    temporal: AttentionMask

    @classmethod
    def from_tensors(
        cls,
        *,
        spatial_mask_tensor: Tensor,
        temporal_mask_tensor: Tensor,
        legend: type[AttentionMaskLegendProvider],
    ) -> Self:
        return cls(
            spatial=AttentionMask.from_tensor(
                mask_tensor=spatial_mask_tensor, legend=legend
            ),
            temporal=AttentionMask.from_tensor(
                mask_tensor=temporal_mask_tensor, legend=legend
            ),
        )

    def effective_allow(self, num_timesteps: int | None = None) -> Tensor:
        spatial_allow = self.spatial.mask_tensor == self.spatial.legend.DO_ATTEND
        temporal_mask_tensor = (
            self.temporal.mask_tensor
            if num_timesteps is None
            else self.temporal.mask_tensor[:num_timesteps, :num_timesteps]
        )
        temporal_allow = temporal_mask_tensor == self.temporal.legend.DO_ATTEND

        effective_allow = (
            temporal_allow[:, :, None, None] & spatial_allow[None, None, :, :]
        )

        return rearrange(
            effective_allow,
            "t_src t_dest s_src s_dest -> (t_src s_src) (t_dest s_dest)",
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
        chunks = [index[modality][name][step].reshape(-1) for modality, name in keys]  # ty:ignore[invalid-argument-type, unresolved-attribute]
        if not chunks:
            return leaves[0].new_empty(0)

        return torch.cat(chunks)

    @staticmethod
    def _set(mask: Tensor, src: Tensor, dest: Tensor, val: Number | Tensor) -> None:
        if src.numel() == 0 or dest.numel() == 0:
            return

        grid = torch.meshgrid(src, dest, indexing="ij")
        mask[grid] = val


class FactorizedAttentionMaskBuilder(Module, ABC):
    @abstractmethod
    def forward(
        self,
        *,
        index: TensorTree,
        timestep: dict[str, dict[str, Mapping[str, int]]],
        legend: type[AttentionMaskLegendProvider],
    ) -> FactorizedAttentionMask: ...


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


class FactorizedCausalAttentionMaskBuilder(FactorizedAttentionMaskBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_attention_mask_builder = CausalAttentionMaskBuilder()

    @override
    def forward(
        self,
        *,
        index: TensorTree,
        timestep: dict[str, dict[str, Mapping[str, int]]],
        legend: type[AttentionMaskLegendProvider],
    ) -> FactorizedAttentionMask:
        leaves = tree_leaves(index, lambda x: isinstance(x, Tensor))
        t = leaves[0].shape[0]
        device = leaves[0].device

        spatial_mask_tensor = self.spatial_attention_mask_builder(
            index=tree_map(operator.itemgetter(slice(1)), index),
            timestep=timestep,
            legend=legend,
        )
        temporal_mask_tensor = torch.full(
            size=(t, t),
            fill_value=legend.DO_ATTEND,
            dtype=torch.as_tensor(legend.DO_ATTEND).dtype,
            device=device,
        )
        future_positions = torch.triu(
            torch.ones((t, t), dtype=torch.bool, device=device), diagonal=1
        )
        temporal_mask_tensor.masked_fill_(future_positions, legend.DO_NOT_ATTEND)

        return FactorizedAttentionMask.from_tensors(
            spatial_mask_tensor=spatial_mask_tensor,
            temporal_mask_tensor=temporal_mask_tensor,
            legend=legend,
        )
