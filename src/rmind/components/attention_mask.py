from typing import overload

import torch

from rmind.components.episode import (
    Episode,
    EpisodeExport,
    Index,
    Modality,
    SummaryToken,
    Timestep,
    TokenType,
)
from rmind.components.mask import AttentionMask, AttentionMaskLegend


def build_attention_mask(  # noqa: PLR0914
    index: Index, timestep: Timestep, *, legend: type[AttentionMaskLegend]
) -> AttentionMask:
    length: int = index.max(reduce=True).item() + 1
    mask = AttentionMask(
        mask=torch.full((length, length), legend.DO_NOT_ATTEND.value),
        legend=legend,
        device="cpu",
    )

    obs_keys = tuple(
        timestep.get(TokenType.OBSERVATION).keys(include_nested=True, leaves_only=True)
    )
    action_keys = tuple(
        timestep.get(TokenType.ACTION).keys(include_nested=True, leaves_only=True)
    )

    (t,) = index.batch_size
    for step in range(t):
        past, current = index[:step], index[step]

        cur_obs = current.select(*obs_keys)
        cur_foresight = current.select(Modality.FORESIGHT)
        cur_obs_summary = current.select((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_SUMMARY,
        ))
        cur_obs_history = current.select((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_HISTORY,
        ))
        cur_actions = current.select(*action_keys)
        cur_action_summary = current.select((
            Modality.SUMMARY,
            SummaryToken.ACTION_SUMMARY,
        ))

        past_obs = past.select(*obs_keys)
        past_foresight = past.select(Modality.FORESIGHT)
        past_actions = past.select(*action_keys)

        mask = (
            mask
            .do_attend(cur_obs, cur_obs)
            .do_attend(cur_obs, past_obs)
            .do_attend(cur_foresight, cur_obs)
            .do_attend(cur_foresight, cur_foresight)
            .do_attend(cur_foresight, past_obs)
            .do_attend(cur_obs_summary, cur_foresight)
            .do_attend(cur_obs_summary, cur_obs_summary)
            .do_attend(cur_obs_summary, past_foresight)
            .do_attend(cur_obs_history, cur_foresight)
            .do_attend(cur_obs_history, cur_obs_history)
            .do_attend(cur_obs_history, past_foresight)
            .do_attend(cur_actions, cur_actions)
            .do_attend(cur_actions, past_actions)
            .do_attend(cur_action_summary, cur_actions)
            .do_attend(cur_action_summary, cur_action_summary)
            .do_attend(cur_action_summary, past_actions)
        )

    return mask


@overload
def build_attention_mask_for_episode(
    episode: Episode, *, legend: type[AttentionMaskLegend]
) -> AttentionMask: ...


@overload
def build_attention_mask_for_episode(
    episode: EpisodeExport, *, legend: type[AttentionMaskLegend]
) -> AttentionMask: ...


def build_attention_mask_for_episode(
    episode: Episode | EpisodeExport, *, legend: type[AttentionMaskLegend]
) -> AttentionMask:
    index, timestep = episode.index, episode.timestep

    if isinstance(episode, EpisodeExport):
        index = Index.from_dict(index, batch_dims=1)
        timestep = Timestep.from_dict(timestep)  # ty:ignore[invalid-argument-type]

    return build_attention_mask(index, timestep, legend=legend)  # ty:ignore[invalid-argument-type]
