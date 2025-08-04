from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import override

import numpy as np
import torch
from einops.layers.torch import Rearrange
from optree import tree_map
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from rmind.components.episode import (
    Episode,
    Index,
    Modality,
    SpecialToken,
    Timestep,
    TokenType,
)
from rmind.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from rmind.components.objectives.base import Objective, PredictionResultKey, Targets
from rmind.utils import ModuleDict


class RandomMaskedHindsightControlObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets

    @override
    def forward(self, episode: Episode, encoder: Module) -> TensorDict:
        b, t = episode.input.batch_size
        device = episode.device

        masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()  # noqa: NPY002
        masked_observation_timestep_idx = np.random.choice(t, 1, replace=False).tolist()  # noqa: NPY002

        episode = episode.clone(recurse=True)
        episode.input_embeddings.select(
            *episode.timestep.keys_by_type[TokenType.ACTION]
        ).masked_fill_(
            torch.zeros((b, t), dtype=torch.bool, device=device).index_fill_(
                1,
                torch.tensor(masked_action_timestep_idx, device=device),
                True,  # noqa: FBT003
            ),
            -1.0,
        )

        episode.input_embeddings.select(
            *episode.timestep.keys_by_type[TokenType.OBSERVATION]
        ).masked_fill_(
            torch.zeros((b, t), dtype=torch.bool, device=device).index_fill_(
                1,
                torch.tensor(masked_observation_timestep_idx, device=device),
                True,  # noqa: FBT003
            ),
            -1.0,
        )

        mask = self.build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.embeddings_packed, mask=mask.mask)
        index = episode.index.select(*episode.timestep.keys_by_type[TokenType.ACTION])
        embeddings = index[masked_action_timestep_idx].parse(embedding)
        logits = self.heads.forward(embeddings)
        targets = TensorDict(
            tree_map(
                episode.get,
                self.targets,  # pyright: ignore[reportArgumentType]
                is_leaf=lambda x: isinstance(x, tuple),
            )
        ).auto_batch_size_(2)[:, masked_action_timestep_idx]

        loss = self.losses.forward(  # pyright: ignore[reportOptionalMemberAccess]
            logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[]),  # pyright: ignore[reportArgumentType]
            targets.apply(Rearrange("b t 1 -> (b t)"), batch_size=[]),
        )

        return TensorDict({"loss": loss})  # pyright: ignore[reportArgumentType]

    @override
    def predict(
        self,
        *,
        episode: Episode,
        encoder: Module,
        result_keys: AbstractSet[PredictionResultKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        b, t = episode.input.batch_size
        device = episode.device
        result = {}

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self.heads.tree_paths())

        if result_keys & {
            PredictionResultKey.PREDICTION_VALUE,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
            PredictionResultKey.SUMMARY_EMBEDDINGS,
        }:
            masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()  # noqa: NPY002
            masked_observation_timestep_idx = np.random.choice(  # noqa: NPY002
                t, 1, replace=False
            ).tolist()

            episode = episode.clone(recurse=True)
            episode.input_embeddings.select(
                *episode.timestep.keys_by_type[TokenType.ACTION]
            ).masked_fill_(
                torch.zeros((b, t), dtype=torch.bool, device=device).index_fill_(
                    1,
                    torch.tensor(masked_action_timestep_idx, device=device),
                    True,  # noqa: FBT003
                ),
                -1.0,
            )

            episode.input_embeddings.select(
                *episode.timestep.keys_by_type[TokenType.OBSERVATION]
            ).masked_fill_(
                torch.zeros((b, t), dtype=torch.bool, device=device).index_fill_(
                    1,
                    torch.tensor(masked_observation_timestep_idx, device=device),
                    True,  # noqa: FBT003
                ),
                -1.0,
            )

            mask = self.build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.embeddings_packed, mask=mask.mask)
            index = episode.index.select(
                *episode.timestep.keys_by_type[TokenType.ACTION]
            )
            embeddings = index[masked_action_timestep_idx].parse(embedding)

            logits = self.heads.forward(embeddings)

            def timestep_padder(x: Tensor) -> Tensor:
                """Insert NaN at all indices except `masked_action_timestep_idx`."""
                match x.shape:
                    case (b, _, *rest):
                        size = (b, t, *rest)

                    case _:
                        raise NotImplementedError

                out = torch.full(size=size, fill_value=torch.nan, device=x.device)
                out[:, masked_action_timestep_idx] = x
                return out

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess]
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
                        episode.input_tokens,
                    )
                    .apply(lambda x: -torch.log(x))
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess]
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])
                    .apply(
                        lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                        episode.input,
                        nested_keys=True,
                    )
                )

            if (result_key := PredictionResultKey.SUMMARY_EMBEDDINGS) in result_keys:
                result[result_key] = episode.index.select(Modality.SPECIAL)[[-1]].parse(
                    embedding
                )

        return TensorDict(result).auto_batch_size_(2)

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        length: int = index.max(reduce=True).item() + 1  # pyright: ignore[reportAttributeAccessIssue, reportAssignmentType]
        mask = AttentionMask(
            mask=torch.full((length, length), legend.DO_ATTEND),
            legend=legend,
            device=index.device,
        )

        (t,) = index.batch_size  # pyright: ignore[reportAssignmentType]
        for step in range(t):
            past, current, future = index[:step], index[step], index[step + 1 :]
            current_actions = current.select(*timestep.keys_by_type[TokenType.ACTION])
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))
            past_actions = past.select(*timestep.keys_by_type[TokenType.ACTION])
            past_action_summary = past.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))
            future_actions = future.select(*timestep.keys_by_type[TokenType.ACTION])
            future_action_summary = future.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask.do_not_attend(current_actions, past_actions)
                .do_not_attend(current_actions, past_action_summary)
                .do_not_attend(current_actions, future_actions)
                .do_not_attend(current_actions, future_action_summary)
                .do_not_attend(current_action_summary, past_actions)
                .do_not_attend(current_action_summary, past_action_summary)
                .do_not_attend(current_action_summary, future_actions)
                .do_not_attend(current_action_summary, future_action_summary)
            )

        return mask
