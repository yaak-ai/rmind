from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import TYPE_CHECKING, override

import torch
from einops import pack
from einops.layers.torch import Rearrange
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Embedding, Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map

from cargpt.components.disparity import DepthDecoder
from cargpt.components.episode import (
    Episode,
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
from cargpt.components.pose import PoseDecoder, Pose
from cargpt.utils.containers import ModuleDict
from cargpt.utils.functional import nan_padder

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class ForwardDynamicsPredictionObjective(Objective):
    def __init__(
        self,
        *,
        depth_decoder: DepthDecoder,
        pose_decoder: PoseDecoder,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
        targets: DictConfig | None = None,
    ):
        super().__init__()

        self.depth_decoder = depth_decoder
        self.pose_decoder = pose_decoder
        self.heads = heads
        self.losses = losses
        self.targets = OmegaConf.to_container(targets)

    @override
    def forward(
        self, inputs: TensorDict, episode_builder: EpisodeBuilder, encoder: Module
    ) -> TensorDict:
        if self.losses is None:
            raise RuntimeError

        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        # all but last timestep
        index = episode.index[:-1]  # pyright: ignore[reportIndexIssue]

        observations: TensorDict = index.select(
            *episode.timestep.keys(TokenType.OBSERVATION)
        ).parse(embedding)

        observation_summary: Float[Tensor, "b t 1 d"] = (
            index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        action_summary: Float[Tensor, "b t 1 d"] = (
            episode.index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        features: TensorDict = observations.apply(  # pyright: ignore[reportAssignmentType]
            # pack: (obs[0], obs_summary, action_summary), (obs[1], obs_summary, action_summary), ...
            lambda obs: pack(
                [
                    obs,
                    observation_summary.broadcast_to(obs.shape),
                    action_summary.broadcast_to(obs.shape),
                ],
                "b t p *",
            )[0]
        )

        logits = self.heads.forward(features)
        targets = TensorDict(
            tree_map(
                lambda f: f(episode)[:, 1:],  # all but first timestep
                self.targets,
            )
        )
        loss = self.losses(
            logits.apply(Rearrange("b t s d -> (b t s) d"), batch_size=[]),
            targets.apply(Rearrange("b t s ... -> (b t s) ..."), batch_size=[]),
        )

        loss["depth"] = self.depth_step(episode, embedding, index, action_summary)
        return TensorDict({"loss": loss})

    def depth_step(self, episode: Episode, embedding: Tensor, index, action_summary):
        # constants
        # TODO: extract it
        last_layer_n = "4"
        bs = [32, 6]

        depth_summary: Float[Tensor, "b t 1 d"] = (
            index.select(k := (Modality.SPECIAL, SpecialToken.DEPTH_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        pose_summary: Float[Tensor, "b t 1 d"] = (
            episode.index.select(k := (Modality.SPECIAL, SpecialToken.POSE_SUMMARY))
            .parse(embedding)
            .get(k)
        )
        image_features = (
            episode.index.select(k := Modality.IMAGE).parse(embedding).get(k)
        )
        losses = TensorDict({})

        # disparity
        obs_for_disp = image_features.apply(
            lambda obs: obs + depth_summary.broadcast_to(obs.shape)
        ).apply(Rearrange("... (h w) c -> ... c w h", h=10))

        features_disp = episode.auxilary_features[Modality.IMAGE][:, :-1].update(
            {(k, last_layer_n): obs_for_disp[k] for k in obs_for_disp.keys()},
            inplace=False,
        )

        out_disparity: Float[Tensor, "b t w h"] = features_disp.apply(
            self.disp_decoder, call_on_nested=True
        )  # [b, t, 576, 320]

        # pose
        obs_for_pose = (
            image_features.apply(
                lambda obs: obs
                + pose_summary.broadcast_to(obs.shape)
                + action_summary.broadcast_to(obs.shape)
            )
            .apply(
                lambda x: torch.cat([x[:, :-1], x[:, 1:]], dim=-1),
                batch_size=[bs[0], bs[1] - 1],
            )
            .apply(Rearrange("... (h w) c -> ... c w h", h=10))
        )

        out_pose: Pose = obs_for_pose.apply(self.pose_decoder)
        breakpoint()
        return None

        # (
        #     losses["photometric"],
        #     losses["geometric"],
        #     tgt_warped,
        #     projected_disparity,
        #     computed_disparity,
        #     valid_mask,
        #     self_mask,
        #     _,
        # ) = self.losses["depth"]["photometric"](
        #     tgt_img=episode.inputs['image'][1],
        #     ref_imgs=ref_imgs,
        #     tgt_disparity=tgt_disparity,
        #     ref_disparities=ref_disparities,
        #     poses=poses,
        # )

        # return  sum(
        #     self.hparams.loss.weights.get(k, 0) * v for k, v in losses.items()
        # )

    @override
    def predict(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
        *,
        result_keys: AbstractSet[PredictionResultKey] | None = None,
    ) -> TensorDict:
        if result_keys is None:
            result_keys = frozenset(PredictionResultKey)

        b, t = inputs.batch_size
        result = TensorDict({}, batch_size=[b, t])

        episode = episode_builder.build_episode(inputs)

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.inputs.select(
                *self.heads.tree_paths()
            ).exclude(Modality.IMAGE)

        if (result_key := PredictionResultKey.ATTENTION) in result_keys:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            attention = encoder.compute_attention_rollout(
                src=episode.packed_embeddings, mask=mask.data, drop_ratio=0.9
            )

            result[result_key] = (
                # from relevant tokens
                episode.index.select(  # pyright: ignore[reportAttributeAccessIssue]
                    (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
                    (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY),
                )
                .parse(attention, dim=1)
                # to all tokens
                .apply(lambda x: episode.index.parse(x, dim=3))
                .apply(
                    Rearrange("b t_from s_from t_to s_to -> b t_from t_to s_from s_to"),
                    batch_size=[b, t, t],
                )
            )

        if result_keys & {
            PredictionResultKey.PREDICTION,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
        }:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
            # all but last timestep
            index = episode.index[:-1]  # pyright: ignore[reportIndexIssue]

            observations: TensorDict = (
                index.select(*episode.timestep.keys(TokenType.OBSERVATION))
                .exclude(Modality.IMAGE)
                .parse(embedding)
            )

            observation_summary: Float[Tensor, "b t 1 d"] = (
                index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            action_summary: Float[Tensor, "b t 1 d"] = (
                index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            features: TensorDict = observations.apply(  # pyright: ignore[reportAssignmentType]
                # pack: (obs[0], obs_summary, action_summary), (obs[1], obs_summary, action_summary), ...
                lambda obs: pack(
                    [
                        obs,
                        observation_summary.broadcast_to(obs.shape),
                        action_summary.broadcast_to(obs.shape),
                    ],
                    "b t p *",
                )[0]
            )

            logits = self.heads.forward(features)

            timestep_padder = nan_padder(pad=(1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportAttributeAccessIssue]
                    .named_apply(
                        lambda k, v: episode_builder.tokenizers.get(k).invert(v),
                        nested_keys=True,
                    )
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = logits.apply(lambda x: x.softmax(dim=-1)).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                """Finds log prob of the correct token at each timestep."""
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
            data=torch.full((index.max + 1, index.max + 1), legend.DO_NOT_ATTEND),  # pyright: ignore[reportCallIssue]
            legend=legend,  # pyright: ignore[reportCallIssue]
            batch_size=[],  # pyright: ignore[reportCallIssue]
            device=index.device,  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        )

        (t,) = index.batch_size  # pyright: ignore[reportAttributeAccessIssue]
        for step in range(t):
            past, current = index[:step], index[step]  # pyright: ignore[reportIndexIssue]
            current_observations = current.select(*timestep.keys(TokenType.OBSERVATION))
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            current_observation_history = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))
            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask._do_attend(current, current)
                ._do_attend(current, past)
                ._do_not_attend(current_observations, current_actions)
                ._do_not_attend(current_observations, current_action_summary)
                ._do_not_attend(current_observation_summary, current_actions)
                ._do_not_attend(current_observation_summary, current_action_summary)
                ._do_not_attend(current_observation_history, current_actions)
                ._do_not_attend(current_observation_history, current_action_summary)
            )

        return mask
