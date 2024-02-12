from enum import StrEnum, auto
from functools import lru_cache
from typing import Annotated

import more_itertools as mit
import numpy as np
import pytorch_lightning as pl
import torch
from beartype.vale import Is
from einops import pack, rearrange
from einops.layers.torch import Rearrange
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from tensordict import TensorDict
from torch.nn import Module, ModuleDict
from wandb import Image
from yaak_datasets import Batch

from cargpt.components.episode import (
    Episode,
    EpisodeBuilder,
    Index,
    Modality,
    ModalityModuleDict,
    SpecialToken,
    Timestep,
    TokenType,
)
from cargpt.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    WandbAttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from cargpt.utils._wandb import LoadableFromArtifact


class Objective(StrEnum):
    FORWARD_DYNAMICS = auto()
    INVERSE_DYNAMICS = auto()
    RANDOM_MASKED_HINDSIGHT_CONTROL = auto()
    COPYCAT = auto()


ObjectiveModuleDict = Annotated[ModuleDict, Is[lambda d: d.keys() <= set(Objective)]]


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    hparams: AttributeDict

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.episode_builder: EpisodeBuilder = instantiate(self.hparams.episode_builder)
        self.encoder: Module = instantiate(self.hparams.encoder)
        self.objectives: ObjectiveModuleDict = instantiate(self.hparams.objectives)

    def _step(self, batch: Batch) -> TensorDict:
        inputs = self._build_input(batch)

        # TODO: currently this does full episode construction for each objective -- optimize?
        metrics = TensorDict(
            {
                name: objective(inputs, self.episode_builder, self.encoder)
                for name, objective in self.objectives.items()
            },
            batch_size=[],
            device=inputs.device,
        )

        if self.trainer.global_step == 0 and isinstance(self.logger, WandbLogger):
            for k, v in metrics.items():
                img = Image(v["mask"].with_legend(WandbAttentionMaskLegend).data)  # pyright: ignore
                self.logger.log_image(
                    f"masks/{k}",
                    [img],
                    step=self.trainer.global_step,
                )

        metrics = metrics.exclude(*((k, "mask") for k in metrics.keys()))  # pyright: ignore
        losses = metrics.select(*((k, "loss") for k in metrics.keys()))
        metrics[("loss", "total")] = sum(losses.values(True, True))

        return metrics

    def training_step(self, batch: Batch, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["train", *k]): v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
        })

        return metrics["loss", "total"]

    def validation_step(self, batch: Batch, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["val", *k]): v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
        })

        return metrics["loss", "total"]

    def _build_input(self, batch: Batch) -> TensorDict:
        frames = batch.frames
        meta = batch.meta
        shapes = [
            frames.get_item_shape(k)
            for k in frames.keys(include_nested=True, leaves_only=True)  # pyright: ignore
        ]

        # include timestep as batch dim
        batch_size = mit.one({(b, t) for (b, t, *_) in shapes})

        return TensorDict.from_dict(
            {
                Modality.IMAGE: frames,
                Modality.CONTINUOUS: {
                    "speed": meta["VehicleMotion_speed"],
                    "pedal": (
                        meta["VehicleMotion_gas_pedal_normalized"]
                        - meta["VehicleMotion_brake_pedal_normalized"]
                    ),
                    "steering_angle": meta["VehicleMotion_steering_angle_normalized"],
                },
                Modality.DISCRETE: {
                    "turn_signal": meta["VehicleState_turn_signal"],
                },
            },
            batch_size=batch_size,
            device=frames.device,
        )

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = cfg | {"scheduler": scheduler}

        return result


class ForwardDynamicsPredictionObjective(Module):
    def __init__(self, heads: ModalityModuleDict, loss: Module) -> None:
        # TODO
        raise NotImplementedError("update for new timestep structure")  # noqa: EM101

        super().__init__()
        self.heads = heads
        self.loss = loss

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        b, t = inputs.batch_size
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        index = episode.index.select(  # pyright: ignore
            (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
            (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY),
        )
        embeddings = index.parse(embedding)
        observations = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))
        actions = embeddings.get((Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))

        observation_action_pairs, _ = pack(
            [observations[:, :-1], actions[:, :-1]],
            "b t *",
        )

        logits = TensorDict(
            {
                (token, name): self.heads[token][name](observation_action_pairs)  # pyright: ignore
                for (token, name) in episode.timestep.keys(TokenType.OBSERVATION)
                if token in (Modality.CONTINUOUS, Modality.DISCRETE)
            },
            batch_size=[b, t - 1],
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[:, 1:]  # pyright: ignore

        logits = logits.apply(Rearrange("b t d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])
        loss = logits.apply(self.loss, labels)

        return TensorDict.from_dict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = AttentionMask(  # pyright: ignore
            data=torch.full((index.max + 1, index.max + 1), legend.DO_NOT_ATTEND),
            legend=legend,
            batch_size=[],
            device=index.device,  # pyright: ignore
        )

        (t,) = index.batch_size  # pyright: ignore
        for step in range(t):
            current, future = index[step], index[step + 1 :]  # pyright: ignore

            current_observations = current.select(*timestep.keys(TokenType.OBSERVATION))
            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            future_observations = future.select(*timestep.keys(TokenType.OBSERVATION))
            future_actions = future.select(*timestep.keys(TokenType.ACTION))
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))

            mask = (
                mask._do_attend(
                    current_observations,
                    current_observations,
                )
                ._do_attend(
                    current_actions,
                    current_actions,
                )
                ._do_attend(
                    current_actions,
                    current_observation_summary,
                )
                ._do_attend(
                    future_observations,
                    current_observation_summary,
                )
                ._do_attend(
                    future_actions,
                    current_observation_summary,
                )
            )

        return mask


class InverseDynamicsPredictionObjective(Module):
    def __init__(
        self,
        heads: ModalityModuleDict,
        loss: Module,
    ):
        # TODO
        raise NotImplementedError("update for new timestep structure")  # noqa: EM101

        super().__init__()
        self.heads = heads
        self.loss = loss

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        b, t = inputs.batch_size
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        index = episode.index.select((  # pyright: ignore
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))
        embeddings = index.parse(embedding)
        observations = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))

        # (o0, o1, o2, o3, ...) -> ((o0, o1), (o1, o2), (o2, o3), ...)
        observation_pairs = rearrange(
            [observations[:, :-1], observations[:, 1:]],
            "i b t 1 d -> b t (i d)",
        )

        logits = TensorDict(
            {
                (token, name): self.heads[token][name](observation_pairs)  # pyright: ignore
                for (token, name) in episode.timestep.keys(TokenType.ACTION)
                if token in (Modality.CONTINUOUS, Modality.DISCRETE)
            },
            batch_size=[b, t - 1],
        )
        labels = episode.tokenized.select(*logits.keys(True, True))[:, :-1]  # pyright: ignore

        logits = logits.apply(Rearrange("b t d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])
        loss = logits.apply(self.loss, labels)

        return TensorDict.from_dict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    def _build_attention_mask(cls, index: Index, timestep: Timestep) -> AttentionMask:
        return ForwardDynamicsPredictionObjective._build_attention_mask(index, timestep)


class RandomMaskedHindsightControlObjective(Module):
    def __init__(self, heads: ModalityModuleDict, loss: Module) -> None:
        # TODO
        raise NotImplementedError("update for new timestep structure")  # noqa: EM101

        super().__init__()
        self.heads = heads
        self.loss = loss

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
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
        index = episode.index.select(*episode.timestep.keys(TokenType.ACTION)).exclude(  # pyright: ignore
            Modality.SPECIAL
        )
        embeddings = index[masked_action_timestep_idx].parse(embedding)
        logits = embeddings.named_apply(
            lambda nested_key, tensor: self.heads[nested_key[0]][nested_key[1]](tensor),
            nested_keys=True,
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[
            :, masked_action_timestep_idx
        ]

        logits = logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t 1)"), batch_size=[])
        loss = logits.apply(self.loss, labels)

        return TensorDict.from_dict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = AttentionMask(  # pyright: ignore
            data=torch.full((index.max + 1, index.max + 1), legend.DO_NOT_ATTEND),
            legend=legend,
            batch_size=[],
            device=index.device,  # pyright: ignore
        )

        (t,) = index.batch_size  # pyright: ignore
        for step in range(t):
            current, future = index[step], index[step + 1 :]  # pyright: ignore

            current_observations = current.select(*timestep.keys(TokenType.OBSERVATION))
            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            future_observations = future.select(*timestep.keys(TokenType.OBSERVATION))
            future_actions = future.select(*timestep.keys(TokenType.ACTION))
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            future_observation_summary = future.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))

            mask = (
                mask._do_attend(
                    current_observations,
                    current_observations,
                )
                ._do_attend(
                    current_actions,
                    current_actions,
                )
                ._do_attend(
                    current_actions,
                    current_observation_summary,
                )
                ._do_attend(
                    future_observations,
                    current_observation_summary,
                )
                ._do_attend(
                    future_actions,
                    current_observation_summary,
                )
                ._do_attend(
                    current_observations,
                    future_observation_summary,
                )
                ._do_attend(
                    current_actions,
                    future_observation_summary,
                )
            )

        return mask


class CopycatObjective(Module):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)"""

    def __init__(self, memory_extraction: ModuleDict, policy: ModuleDict):
        super().__init__()

        self.memory_extraction = memory_extraction
        self.policy = policy

    def _memory_extraction_forward(
        self,
        episode: Episode,
        embeddings: TensorDict,
    ) -> TensorDict:
        observation_history = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_HISTORY,
        ))
        action_keys = episode.timestep.keys(TokenType.ACTION)

        pred = self.memory_extraction.head(observation_history)
        pred = TensorDict.from_dict(
            dict(zip(action_keys, pred.split(1, dim=-1))),
            batch_size=[],
        )
        pred = pred.apply(Rearrange("b 1 1 -> b"), batch_size=[])

        labels = episode.transformed.select(*action_keys).apply(
            lambda x: x[:, -1] - x[:, -2],
            batch_size=[],
        )

        return pred.apply(self.memory_extraction.loss, labels, batch_size=[])

    def _policy_forward(self, episode: Episode, embeddings: TensorDict) -> TensorDict:
        observation_history = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_HISTORY,
        )).detach()  # NOTE: equivalent to stop gradient layer in paper

        observation_summary = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))

        features = rearrange(
            [observation_summary, observation_history],
            "i b 1 d -> b 1 (i d)",
        )

        logits = TensorDict(
            {
                (modality, name): self.policy.heads[modality][name](features)  # pyright: ignore
                for (modality, name) in episode.timestep.keys(TokenType.ACTION)
                if modality is not Modality.SPECIAL
            },
            batch_size=[],
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[:, -1]  # pyright: ignore

        logits = logits.apply(Rearrange("b 1 d -> b d"), batch_size=[])
        labels = labels.apply(Rearrange("b 1 -> b"), batch_size=[])

        return logits.apply(self.policy.loss, labels)

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        _b, _t = inputs.batch_size
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        embeddings = (
            episode.index[-1]  # pyright: ignore
            .select(
                (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY),
                (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
            )
            .parse(embedding)
        )

        loss = TensorDict.from_dict({
            "memory_extraction": self._memory_extraction_forward(episode, embeddings),
            # NOTE: scale the policy loss so the losses start out around the same order of magnitude
            "policy": self._policy_forward(episode, embeddings).apply(
                lambda x: x * 0.1
            ),
        })

        return TensorDict.from_dict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ):
        mask = AttentionMask(  # pyright: ignore
            data=torch.full((index.max + 1, index.max + 1), legend.DO_NOT_ATTEND),
            legend=legend,
            batch_size=[],
            device=index.device,  # pyright: ignore
        )

        (t,) = index.batch_size  # pyright: ignore
        for step in range(t):
            past, current = index[:step], index[step]  # pyright: ignore
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

            past_observations = past.select(*timestep.keys(TokenType.OBSERVATION))

            mask = (
                mask._do_attend(
                    current_observations,
                    current_observations,
                )
                ._do_attend(
                    current_observation_summary,
                    current_observations,
                )
                ._do_attend(
                    current_observation_summary,
                    current_observation_summary,
                )
                ._do_attend(
                    current_observation_history,
                    current_observations,
                )
                ._do_attend(
                    current_observation_history,
                    current_observation_history,
                )
                ._do_attend(
                    current_actions,
                    current_actions,
                )
                ._do_attend(
                    current_actions,
                    current_observation_summary,
                )
                ._do_attend(
                    current_action_summary,
                    current_actions,
                )
                ._do_attend(
                    current_action_summary,
                    current_observation_summary,
                )
                ._do_attend(
                    current_action_summary,
                    current_action_summary,
                )
                ._do_attend(
                    current_observation_history,
                    past_observations,
                )
            )

        return mask
