from enum import Enum
from functools import lru_cache
from typing import Annotated

import more_itertools as mit
import numpy as np
import pytorch_lightning as pl
import torch
from beartype.vale import Is
from einops import pack, rearrange, reduce
from einops.layers.torch import Rearrange
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from tensordict import TensorDict
from torch.nn import Module, ModuleDict
from wandb import Image

from cargpt.components.episode import (
    EpisodeBuilder,
    EpisodeIndex,
    Timestep,
    Token,
    TokenModuleDict,
)
from cargpt.components.mask import (
    AttentionMask,
    InverseDynamicsAttentionMask,
    NonCausalAttentionMask,
    TimestepWiseCausalAttentionMask,
    WandbAttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from cargpt.utils._wandb import LoadableFromArtifact


class Objective(str, Enum):
    FORWARD_DYNAMICS = "forward_dynamics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    RANDOM_MASKED_HINDSIGHT_CONTROL = "random_masked_hindsight_control"


ObjectiveModuleDict = Annotated[ModuleDict, Is[lambda d: d.keys() <= set(Objective)]]


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    hparams: AttributeDict

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.episode_builder: EpisodeBuilder = instantiate(self.hparams.episode_builder)
        self.encoder: Module = instantiate(self.hparams.encoder)
        self.objectives: ObjectiveModuleDict = instantiate(self.hparams.objectives)

    def _step(self, batch: TensorDict) -> TensorDict:
        inputs = self._build_input(batch)

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
                img = Image(v["mask"].with_legend(WandbAttentionMaskLegend).data)
                self.logger.log_image(
                    f"masks/{k}",
                    [img],
                    step=self.trainer.global_step,
                )

        metrics = metrics.exclude(*((k, "mask") for k in metrics.keys()))  # pyright: ignore
        losses = metrics.select(*((k, "loss") for k in metrics.keys()))
        metrics[("loss", "total")] = torch.stack(
            tuple(losses.values(True, True))
        ).mean()

        return metrics

    def training_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["train", *k]): v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
        })

        return metrics["loss", "total"]

    def validation_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["val", *k]): v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
        })

        return metrics["loss", "total"]

    def _build_input(self, batch: TensorDict) -> TensorDict:
        meta = batch["meta"]
        return TensorDict.from_dict(
            {
                Token.IMAGE: batch["frames"],
                Token.CONTINUOUS: {
                    "speed": meta["VehicleMotion_speed"],
                    "pedal": (
                        meta["VehicleMotion_gas_pedal_normalized"]
                        - meta["VehicleMotion_brake_pedal_normalized"]
                    ),
                    "steering_angle": meta["VehicleMotion_steering_angle_normalized"],
                },
                Token.DISCRETE: {
                    "turn_signal": meta["VehicleState_turn_signal"],
                },
            },
            batch_size=batch.batch_size,
            device=batch.device,
        )

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = cfg | {"scheduler": scheduler}

        return result


class ForwardDynamicsPredictionObjective(Module):
    def __init__(self, head: Module, loss: Module) -> None:
        super().__init__()
        self.head = head
        self.loss = loss

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        embeddings = episode.index.parse(embedding)
        observations, _ = pack(
            [embeddings[k] for k in episode.timestep.observations],
            "b t * d",
        )
        actions, _ = pack([embeddings[k] for k in episode.timestep.actions], "b t * d")

        observations = reduce(observations, "b t s d -> b t d", "mean")
        actions = reduce(actions, "b t s d -> b t d", "mean")

        inputs, _ = pack([observations[:, :-1], actions[:, :-1]], "b t *")
        pred = self.head(inputs)
        labels = observations[:, 1:]
        loss = self.loss(pred, labels)

        return TensorDict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(cls, index: EpisodeIndex) -> AttentionMask:
        return TimestepWiseCausalAttentionMask.build(
            index=index,
            legend=XFormersAttentionMaskLegend,
        )


class InverseDynamicsPredictionObjective(Module):
    def __init__(
        self,
        heads: ModuleDict,
        loss: Module,
    ):
        super().__init__()
        self.heads = heads
        self.loss = loss

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        observation_index = episode.index.select(*episode.timestep.observations)  # pyright: ignore
        observations = observation_index.parse(embedding)
        observations, _ = pack(
            [observations[k] for k in episode.timestep.observations],
            "b t * d",
        )
        observations = reduce(observations, "b t s d -> b t 1 d", "mean")

        # (o0, o1, o2, o3, ...) -> ((o0, o1), (o1, o2), (o2, o3), ...)
        observations = rearrange(
            [observations[:, :-1], observations[:, 1:]],
            "i b t 1 d -> b t (i d)",
        )
        logits = TensorDict(
            {
                (t, n): self.heads[t][n](observations)  # pyright: ignore
                for (t, n) in episode.timestep.actions
            },
            batch_size=[observations.shape[0]],
        )
        logits = logits.apply(Rearrange("b t d -> (b t) d"), batch_size=[])

        labels = episode.labels.select(*episode.timestep.actions)
        labels = labels.apply(lambda x: x[:, :-1])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])

        loss = logits.apply(self.loss, labels)

        return TensorDict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls, index: EpisodeIndex, timestep: Timestep
    ) -> AttentionMask:
        return InverseDynamicsAttentionMask.build(
            index=index,
            timestep=timestep,
            legend=XFormersAttentionMaskLegend,
        )


class RandomMaskedHindsightControlObjective(Module):
    def __init__(self, heads: TokenModuleDict, loss: Module) -> None:
        super().__init__()
        self.heads = heads
        self.loss = loss

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        shapes = [
            inputs.get_item_shape(k)
            for k in inputs.keys(include_nested=True, leaves_only=True)  # pyright: ignore
        ]
        t = mit.one({t for (_, t, *_) in shapes})

        masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()
        masked_observation_timestep_idx = np.random.choice(t, 1, replace=False).tolist()
        episode = episode_builder.build_episode(
            inputs,
            masked_action_timestep_idx=masked_action_timestep_idx,
            masked_observation_timestep_idx=masked_observation_timestep_idx,
        )
        mask = self._build_attention_mask(episode.index)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        action_index = episode.index.select(*episode.timestep.actions)  # pyright: ignore
        action_embeddings = action_index[masked_action_timestep_idx].parse(embedding)

        logits = TensorDict(
            {
                (t, n): self.heads[t][n](emb)  # pyright: ignore
                for ((t, n), emb) in action_embeddings.items(
                    include_nested=True,
                    leaves_only=True,
                )
            },
            batch_size=action_embeddings.batch_size,
        )
        logits = logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[])

        labels = episode.labels.select(*episode.timestep.actions).apply(
            lambda lbl: lbl[:, masked_action_timestep_idx]
        )
        labels = labels.apply(Rearrange("b t 1 -> (b t 1)"), batch_size=[])

        loss = logits.apply(self.loss, labels)

        return TensorDict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(cls, index: EpisodeIndex) -> AttentionMask:
        return NonCausalAttentionMask.build(
            index=index,
            legend=XFormersAttentionMaskLegend,
        )
