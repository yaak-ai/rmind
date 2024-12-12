from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import override

import numpy as np
import torch
from einops.layers.torch import Rearrange
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map

from cargpt.components.episode import (
    EpisodeBuilder,
    Index,
    Modality,
    SpecialToken,
    Timestep,
    TokenType,
)
from cargpt.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from cargpt.components.objectives.base import Objective, PredictionResultKey
from cargpt.utils.containers import ModuleDict


class RandomMaskedHindsightControlObjective(Objective):
    def __init__(
        self,
        *,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
        targets: DictConfig | None = None,
    ):
        super().__init__()

        self.heads = heads
        self.losses = losses
        self.targets = OmegaConf.to_container(targets) if targets else None

    @override
    def forward(
        self, inputs: TensorDict, episode_builder: EpisodeBuilder, encoder: Module
    ) -> TensorDict:
        if self.losses is None:
            raise RuntimeError

        _, t = inputs.batch_size
        masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()
        masked_observation_timestep_idx = np.random.choice(t, 1, replace=False).tolist()
        episode = episode_builder.build_episode(
            inputs,
            masked_action_timestep_idx=masked_action_timestep_idx,
            masked_observation_timestep_idx=masked_observation_timestep_idx,
        )
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        index = episode.index.select(*episode.timestep.keys(TokenType.ACTION))  # pyright: ignore[reportAttributeAccessIssue]
        embeddings = index[masked_action_timestep_idx].parse(embedding)
        logits = self.heads.forward(embeddings)
        targets = TensorDict(
            tree_map(lambda f: f(episode)[:, masked_action_timestep_idx], self.targets)
        )

        loss = self.losses(
            logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[]),
            targets.apply(Rearrange("b t 1 -> (b t)"), batch_size=[]),
        )

        return TensorDict({"loss": loss})

    @override
    def predict(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
        *,
        result_keys: AbstractSet[PredictionResultKey] | None,
    ) -> TensorDict:
        if result_keys is None:
            result_keys = frozenset(PredictionResultKey)

        b, t = inputs.batch_size
        result = TensorDict({}, batch_size=[b, t])

        masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()
        masked_observation_timestep_idx = np.random.choice(t, 1, replace=False).tolist()
        episode = episode_builder.build_episode(
            inputs,
            masked_action_timestep_idx=masked_action_timestep_idx,
            masked_observation_timestep_idx=masked_observation_timestep_idx,
        )

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.inputs.select(*self.heads.tree_paths())

        if result_keys & {
            PredictionResultKey.PREDICTION,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
        }:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
            index = episode.index.select(*episode.timestep.keys(TokenType.ACTION))  # pyright: ignore[reportAttributeAccessIssue]
            embeddings = index[masked_action_timestep_idx].parse(embedding)

            logits = self.heads.forward(embeddings)

            def timestep_padder(x):
                """insert NaN at all indices except `masked_action_timestep_idx`"""
                match x.shape:
                    case (b, _, *rest):
                        size = (b, t, *rest)

                    case _:
                        raise NotImplementedError

                out = torch.full(size=size, fill_value=torch.nan, device=x.device)
                out[:, masked_action_timestep_idx] = x
                return out

            if (result_key := PredictionResultKey.PREDICTION) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda k, v: episode_builder.tokenizers.get(k).invert(v),
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = logits.apply(lambda x: x.softmax(dim=-1)).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.softmax(dim=-1))
                    .apply(Rearrange("b t 1 d -> b t d"))  # pyright: ignore[reportAttributeAccessIssue]
                    .apply(timestep_padder, batch_size=[b, t])
                    .apply(
                        lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                        episode.tokenized,
                    )
                    .apply(lambda x: -torch.log(x))
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda k, v: episode_builder.tokenizers.get(k).invert(v),
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])
                    .apply(
                        lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                        episode.inputs,
                        nested_keys=True,
                    )
                )

        return result

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,  # pyright: ignore[reportGeneralTypeIssues]
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:  # pyright: ignore[reportGeneralTypeIssues]
        mask = AttentionMask(  # pyright: ignore[reportCallIssue]
            data=torch.full((index.max + 1, index.max + 1), legend.DO_ATTEND),  # pyright: ignore[reportCallIssue]
            legend=legend,  # pyright: ignore[reportCallIssue]
            batch_size=[],  # pyright: ignore[reportCallIssue]
            device=index.device,  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
        )

        (t,) = index.batch_size  # pyright: ignore[reportAttributeAccessIssue]
        for step in range(t):
            past, current, future = index[:step], index[step], index[step + 1 :]  # pyright: ignore[reportIndexIssue]
            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))
            past_actions = past.select(*timestep.keys(TokenType.ACTION))
            past_action_summary = past.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))
            future_actions = future.select(*timestep.keys(TokenType.ACTION))
            future_action_summary = future.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask._do_not_attend(current_actions, past_actions)
                ._do_not_attend(current_actions, past_action_summary)
                ._do_not_attend(current_actions, future_actions)
                ._do_not_attend(current_actions, future_action_summary)
                ._do_not_attend(current_action_summary, past_actions)
                ._do_not_attend(current_action_summary, past_action_summary)
                ._do_not_attend(current_action_summary, future_actions)
                ._do_not_attend(current_action_summary, future_action_summary)
            )

        return mask
