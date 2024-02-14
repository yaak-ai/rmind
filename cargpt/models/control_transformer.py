from enum import Enum
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
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.optim import Adam
from wandb import Image

from cargpt.components.episode import (
    Episode,
    EpisodeBuilder,
    Index,
    Timestep,
    Token,
    TokenModuleDict,
)
from cargpt.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    WandbAttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from cargpt.utils._wandb import LoadableFromArtifact


class Objective(str, Enum):
    FORWARD_DYNAMICS = "forward_dynamics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    RANDOM_MASKED_HINDSIGHT_CONTROL = "random_masked_hindsight_control"
    CAUSAL_CONFUSION_ADVERSARIAL = "causal_confusion_adversarial"


ObjectiveModuleDict = Annotated[ModuleDict, Is[lambda d: d.keys() <= set(Objective)]]


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    hparams: AttributeDict

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.episode_builder: EpisodeBuilder = instantiate(self.hparams.episode_builder)
        self.encoder: Module = instantiate(self.hparams.encoder)
        self.objectives: ObjectiveModuleDict = instantiate(self.hparams.objectives)
        self.automatic_optimization = self.hparams.automatic_optimization

    def configure_optimizers(self):
        params = (
            list(self.episode_builder.parameters())
            + list(self.encoder.parameters())
            + list(self.objectives["causal_confusion_adversarial"].heads.parameters())
        )
        optimizer_all = instantiate(self.hparams.optimizer.all, params=params)
        optimizer_d = instantiate(
            self.hparams.optimizer.discriminator,
            params=self.objectives[
                "causal_confusion_adversarial"
            ].discriminator.parameters(),
        )
        return optimizer_all, optimizer_d

    def _step(self, batch: TensorDict) -> TensorDict:
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
                img = Image(v["mask"].with_legend(WandbAttentionMaskLegend).data)
                self.logger.log_image(
                    f"masks/{k}",
                    [img],
                    step=self.trainer.global_step,
                )

        metrics = metrics.exclude(
            *((k, "mask") for k in metrics.keys())
        )  # pyright: ignore
        losses = metrics.select(*((k, "loss") for k in metrics.keys()))
        metrics[("loss", "total")] = torch.stack(
            tuple(losses.values(True, True))
        ).mean()

        return metrics

    def training_step(self, batch: TensorDict, _batch_idx: int):
        opt_global, opt_d = self.optimizers()
        opt_global.zero_grad()
        opt_d.zero_grad()

        inputs = self._build_input(batch)
        episode = self.episode_builder.build_episode(inputs)
        objective = self.objectives["causal_confusion_adversarial"]
        mask = objective._build_attention_mask(episode.index, episode.timestep)

        embedding = self.encoder(
            src=episode.packed_embeddings,
            mask=mask.data,
        )

        index = episode.index.select(  # pyright: ignore
            (Token.SPECIAL, "observation_summary"),
            (Token.SPECIAL, "action_summary"),
        )
        embedding = index.parse(embedding)
        observations = embedding.get((Token.SPECIAL, "observation_summary"))
        actions = embedding.get((Token.SPECIAL, "action_summary"))

        observations_noised = observations + objective.noise_std * torch.randn(
            observations.shape
        ).to(observations.device)

        # update discriminator

        loss_d = objective.loss_d(observations_noised, actions, episode)
        loss = torch.stack(tuple(loss_d.values(True, True))).mean()

        self.manual_backward(loss, retain_graph=True)
        opt_d.step()

        opt_global.zero_grad()
        opt_d.zero_grad()

        # update generatror

        loss_g = objective.loss_g(observations, actions, episode)
        loss_d = objective.loss_d(observations_noised, actions, episode)
        loss_g_plus_d = loss_g.apply(lambda x, y: x + objective.weight * y, loss_d)
        loss = torch.stack(tuple(loss_g_plus_d.values(True, True))).mean()

        self.manual_backward(loss)
        opt_global.step()

        opt_global.zero_grad()
        opt_d.zero_grad()

        metrics = self._step(batch)
        metrics["causal_confusion_adversarial"] = TensorDict(
            {
                "loss_g": loss_g,
                "loss_d": loss_d,
                "loss_g_plus_d": loss_g_plus_d,
            },
            batch_size=[],
        )

        self.log_dict(
            {
                "/".join(["train", *k]): v
                for k, v in metrics.items(include_nested=True, leaves_only=True)
            }
        )

        loss_total = metrics["loss", "total"]
        self.manual_backward(loss_total)
        opt_global.step()

        return metrics["loss", "total"]

    def validation_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict(
            {
                "/".join(["val", *k]): v
                for k, v in metrics.items(include_nested=True, leaves_only=True)
            }
        )

        return metrics["loss", "total"]

    def _build_input(self, batch: TensorDict) -> TensorDict:
        frames = batch["frames"]
        meta = batch["meta"]
        shapes = [
            frames.get_item_shape(k)
            for k in frames.keys(include_nested=True, leaves_only=True)
        ]

        # include timestep as batch dim
        batch_size = mit.one({(b, t) for (b, t, *_) in shapes})

        return TensorDict(
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
            batch_size=batch_size,
            device=batch.device,
        )


class ForwardDynamicsPredictionObjective(Module):
    def __init__(self, heads: Module, loss: Module) -> None:
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
            (Token.SPECIAL, "observation_summary"),
            (Token.SPECIAL, "action_summary"),
        )
        embeddings = index.parse(embedding)
        observations = embeddings.get((Token.SPECIAL, "observation_summary"))
        actions = embeddings.get((Token.SPECIAL, "action_summary"))

        observation_action_pairs, _ = pack(
            [observations[:, :-1], actions[:, :-1]],
            "b t *",
        )

        logits = TensorDict(
            {
                (token, name): self.heads[token][name](
                    observation_action_pairs
                )  # pyright: ignore
                for (token, name) in episode.timestep.observations
                if token in (Token.CONTINUOUS, Token.DISCRETE)
            },
            batch_size=[b, t - 1],
        )

        labels = episode.labels.select(*logits.keys(True, True))[:, 1:]

        logits = logits.apply(Rearrange("b t d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])
        loss = logits.apply(self.loss, labels)

        return TensorDict({"loss": loss, "mask": mask}, batch_size=[])

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

            current_observations = current.select(*timestep.observations)
            current_actions = current.select(*timestep.actions)
            future_observations = future.select(*timestep.observations)
            future_actions = future.select(*timestep.actions)
            current_observation_summary = current.select(
                (
                    Token.SPECIAL,
                    "observation_summary",
                )
            )

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
        b, t = inputs.batch_size
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        index = episode.index.select(
            (Token.SPECIAL, "observation_summary")
        )  # pyright: ignore
        embeddings = index.parse(embedding)
        observations = embeddings.get((Token.SPECIAL, "observation_summary"))

        # (o0, o1, o2, o3, ...) -> ((o0, o1), (o1, o2), (o2, o3), ...)
        observation_pairs = rearrange(
            [observations[:, :-1], observations[:, 1:]],
            "i b t 1 d -> b t (i d)",
        )

        logits = TensorDict(
            {
                (token, name): self.heads[token][name](
                    observation_pairs
                )  # pyright: ignore
                for (token, name) in episode.timestep.actions
                if token in (Token.CONTINUOUS, Token.DISCRETE)
            },
            batch_size=[b, t - 1],
        )
        labels = episode.labels.select(*logits.keys(True, True))[:, :-1]

        logits = logits.apply(Rearrange("b t d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])
        loss = logits.apply(self.loss, labels)

        return TensorDict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    def _build_attention_mask(cls, index: Index, timestep: Timestep) -> AttentionMask:
        return ForwardDynamicsPredictionObjective._build_attention_mask(index, timestep)


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
        b, t = inputs.batch_size
        masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()
        masked_observation_timestep_idx = np.random.choice(t, 1, replace=False).tolist()
        episode = episode_builder.build_episode(
            inputs,
            masked_action_timestep_idx=masked_action_timestep_idx,
            masked_observation_timestep_idx=masked_observation_timestep_idx,
        )
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        index = episode.index.select(*episode.timestep.actions).exclude(
            Token.SPECIAL
        )  # pyright: ignore
        embeddings = index[masked_action_timestep_idx].parse(embedding)

        logits = TensorDict(
            {
                (token, name): self.heads[token][name](emb)  # pyright: ignore
                for ((token, name), emb) in embeddings.items(True, True)
            },
            batch_size=[b, len(masked_action_timestep_idx)],
        )

        labels = episode.labels.select(*logits.keys(True, True))[
            :, masked_action_timestep_idx
        ]

        logits = logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t 1)"), batch_size=[])
        loss = logits.apply(self.loss, labels)

        return TensorDict({"loss": loss, "mask": mask}, batch_size=[])

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

            current_observations = current.select(*timestep.observations)
            current_actions = current.select(*timestep.actions)
            future_observations = future.select(*timestep.observations)
            future_actions = future.select(*timestep.actions)
            current_observation_summary = current.select(
                (
                    Token.SPECIAL,
                    "observation_summary",
                )
            )
            future_observation_summary = future.select(
                (
                    Token.SPECIAL,
                    "observation_summary",
                )
            )

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


class CausalConfusionAdversarial(Module):
    def __init__(
        self,
        discriminator: Module,
        heads: Module,
        loss: Module,
        weight: float,
        noise_std: float,
    ) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.heads = heads
        self.loss = loss
        self.weight = weight
        self.noise_std = noise_std

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ):
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)

        embedding = encoder(
            src=episode.packed_embeddings,
            mask=mask.data,
        )

        index = episode.index.select(  # pyright: ignore
            (Token.SPECIAL, "observation_summary"),
            (Token.SPECIAL, "action_summary"),
        )
        embedding = index.parse(embedding)
        observations = embedding.get((Token.SPECIAL, "observation_summary"))
        actions = embedding.get((Token.SPECIAL, "action_summary"))

        loss = self.loss_g(observations, actions, episode)

        return TensorDict(
            {
                "loss": loss,
                "mask": mask,
            },
            batch_size=[],
        )

    def loss_g(
        self,
        observations: Tensor,
        actions: Tensor,
        episode: Episode,
    ) -> TensorDict:
        b, t = episode.embeddings.batch_size
        logits = TensorDict(
            {
                (token, name): self.heads[token][name](observations)  # pyright: ignore
                for (token, name) in episode.timestep.actions
                if token in (Token.CONTINUOUS, Token.DISCRETE)
            },
            batch_size=[b, t],
        )

        labels = episode.labels.select(*logits.keys(True, True))

        logits = logits.apply(Rearrange("b t 1 d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])
        return logits.apply(self.loss, labels)

    def loss_d(
        self,
        observations_noised: Tensor,
        actions: Tensor,
        episode: Episode,
    ):
        b, t = episode.embeddings.batch_size
        # observation_action_pairs, _ = pack(
        #     [observations_noised[:, 1:], actions[:, 1:]],
        #     "b t n *",
        # )
        logits = TensorDict(
            {
                (token, name): self.discriminator[token][name](
                    observations_noised[:, 1:]
                )  # pyright: ignore
                for (token, name) in episode.timestep.actions
                if token in (Token.CONTINUOUS, Token.DISCRETE)
            },
            batch_size=[b, t - 1],
        )

        labels = episode.labels.select(*logits.keys(True, True))[:, :-1]

        logits = logits.apply(Rearrange("b t 1 d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])
        return logits.apply(self.loss, labels)

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

            current_observations = current.select(*timestep.observations)
            current_actions = current.select(*timestep.actions)
            future_observations = future.select(*timestep.observations)
            future_actions = future.select(*timestep.actions)
            current_observation_summary = current.select(
                (
                    Token.SPECIAL,
                    "observation_summary",
                )
            )
            current_action_summary = current.select(
                (
                    Token.SPECIAL,
                    "action_summary",
                )
            )

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
