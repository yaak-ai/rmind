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
from torch.nn import Module
from torch.nn import functional as F

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
from cargpt.components.pose import PoseDecoder, PoseLabeler
from cargpt.utils.camera import get_camera_config
from cargpt.utils.containers import ModuleDict
from cargpt.utils.functional import nan_padder

if TYPE_CHECKING:
    from jaxtyping import Float


class ForwardDynamicsPredictionObjective(Objective):
    def __init__(
        self,
        *,
        depth_decoder: DepthDecoder,
        pose_decoder: PoseDecoder,
        pose_labeler: PoseLabeler,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
        targets: DictConfig | None = None,
    ):
        super().__init__()

        self.depth_decoder = depth_decoder
        self.pose_decoder = pose_decoder
        self.pose_labeler = pose_labeler
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
        # img_emb_w = episode_builder.position_encoding.image.patch.col.num_embeddings
        # img_emb_h = episode_builder.position_encoding.image.patch.row.num_embeddings
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
                    action_summary[:, :-1].broadcast_to(obs.shape),
                ],
                "b t p *",
            )[0]
        )

        logits = self.heads.forward(features)
        # targets = TensorDict(
        #     tree_map(
        #         lambda f: f(episode)[:, 1:],  # all but first timestep
        #         self.targets,
        #     )
        # )
        # loss = self.losses(
        #     logits.apply(Rearrange("b t s d -> (b t s) d"), batch_size=[]),
        #     targets.apply(Rearrange("b t s ... -> (b t s) ..."), batch_size=[]),
        # )
        loss = {}

        # depth
        depth_metrics, loss["pose"] = self.depth_step(
            episode, embedding, action_summary
        )
        loss["depth"] = TensorDict({
            k: v.pop("total_loss") for k, v in depth_metrics.items()
        })

        return TensorDict({
            "loss": loss,
            "depth_metrics": depth_metrics.apply(
                Rearrange("(b t) ... -> b t ...", b=logits.shape[0]),
                batch_size=logits.shape,
            ),
        })

    def depth_step(
        self, episode: Episode, embedding: Tensor, action_summary
    ) -> tuple[TensorDict]:
        # constants
        # TODO: extract it
        last_layer_n = 4
        bs = episode.inputs.batch_size
        _img_emb_w, img_emb_h = 18, 10
        squeeze_bs = Rearrange("b t ... -> (b t)...")

        depth_summary: Float[Tensor, "b t 1 d"] = (
            episode.index.select(k := (Modality.SPECIAL, SpecialToken.DEPTH_SUMMARY))
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

        # pose ref -> tgt (temporal order input)
        pose: TensorDict = (
            image_features.named_apply(lambda k, v: pose_summary)
            .apply(Rearrange("... 1 c -> ... c"))
            .apply(
                lambda x: torch.cat([x[:, :-1], x[:, 1:]], dim=-1),
                batch_size=[bs[0], bs[1] - 1],
            )
            .apply(self.pose_decoder)
        )

        for k in episode.inputs["image"].keys():
            if "front" not in k:
                msg = "Speed Pose Labeler is implemented only for front cameras"
                raise NotImplementedError(msg)
        pose_labels = pose.named_apply(lambda k, v: self.pose_labeler(episode))

        loss_pose = (
            self.losses["depth"]["pose"](
                pose.apply(squeeze_bs, batch_size=[]),
                pose_labels.apply(squeeze_bs, batch_size=[]),
            )
            * 0.15
        )

        # disparity
        disparity_input_features = image_features.apply(
            lambda obs: obs + depth_summary.broadcast_to(obs.shape)
        ).apply(Rearrange("... (h w) c -> ... c h w", h=img_emb_h))

        disparity = (
            episode.auxilary_features[Modality.IMAGE]
            .update(
                {
                    (k, str(last_layer_n)): disparity_input_features[k]
                    for k in disparity_input_features.keys()
                },
                inplace=False,
            )
            .apply(self.depth_decoder, call_on_nested=True)
        )

        # TODO: get it from dataset when implemented in rbyte
        camera_model = TensorDict({
            k: get_camera_config(camera_name=k, batch_size=bs) for k in pose.keys()
        })

        tgt_img = episode.inputs["image"][:, 1:].apply(
            squeeze_bs, batch_size=[]
        )  # (t+1), to calculate loss with
        ref_img = episode.inputs["image"][:, :-1].apply(
            squeeze_bs, batch_size=[]
        )  # t,  to warp from
        tgt_disparity = disparity[:, 1:].apply(squeeze_bs, batch_size=[])
        ref_disparity = disparity[:, :-1].apply(squeeze_bs, batch_size=[])
        ref_tgt_pose = pose.apply(squeeze_bs, batch_size=[])

        squeeze_bs = Rearrange("b t ... -> (b t)...")
        loss_depth = self.losses["depth"]["photogeometry"](
            tgt_img, ref_img, tgt_disparity, ref_disparity, ref_tgt_pose, camera_model
        )
        return loss_depth, loss_pose

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
