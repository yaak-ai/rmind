from enum import Enum
from functools import lru_cache
from typing import Annotated, List

import more_itertools as mit
import numpy as np
import pytorch_lightning as pl
import torch
from beartype.vale import Is
from einops import pack, reduce
from einops.layers.torch import Rearrange, Reduce
from hydra.utils import instantiate
from jaxtyping import Float
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from tensordict import TensorDict
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module, ModuleDict

from cargpt.components.episode import (
    Episode,
    EpisodeBuilder,
    EpisodeIndex,
    Timestep,
    Token,
    TokenModuleDict,
)
from cargpt.components.mask import (
    ForwardDynamicsAttentionMask,
    InverseDynamicsAttentionMask,
    RandomMaskedHindsightControlAttentionMask,
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
        batch_size, _device = batch.batch_size, batch.device

        inputs = self._build_input(batch)
        episode = self.episode_builder.build_episode(inputs)
        timestep = self.episode_builder.timestep

        masks = TensorDict({}, batch_size=[])
        embeddings = TensorDict({}, batch_size=batch_size)
        metrics = TensorDict({}, batch_size=[])

        episode_clone = episode.clone(recurse=True)  # pyright: ignore

        episode_clone.embeddings = self.episode_builder._apply_position_encoding(
            embeddings=episode_clone.embeddings,
            step_index=episode_clone.index[0],  # pyright: ignore
        )

        if objective := getattr(
            self.objectives, (k := Objective.FORWARD_DYNAMICS), None
        ):
            masks[k] = self._build_forward_dynamics_attention_mask(episode_clone.index)
            embeddings[k] = self.encoder(
                src=episode_clone.packed_embeddings,
                mask=masks[k].data,
            )
            metrics[k] = objective(episode_clone, embeddings.get(k))

        if objective := getattr(
            self.objectives, (k := Objective.INVERSE_DYNAMICS), None
        ):
            masks[k] = self._build_inverse_dynamics_attention_mask(
                episode_clone.index,
                timestep,
            )
            embeddings[k] = self.encoder(
                src=episode_clone.packed_embeddings,
                mask=masks[k].data,
            )

            metrics[k] = objective(
                episode_clone,
                embeddings.select(
                    Objective.INVERSE_DYNAMICS,
                    Objective.INVERSE_DYNAMICS,
                ),
            )

        if objective := getattr(
            self.objectives, (k := Objective.RANDOM_MASKED_HINDSIGHT_CONTROL), None
        ):
            num_timesteps = mit.one(episode.index.batch_size)  # pyright: ignore
            masked_action_timestep_idx = np.random.choice(
                num_timesteps, 2, replace=False
            ).tolist()
            masked_observation_timestep_idx = np.random.choice(
                num_timesteps, 1, replace=False
            ).tolist()

            episode_masked = episode.clone(recurse=True)  # pyright: ignore
            episode_masked.embeddings.select(*episode.timestep.actions)[
                :, masked_action_timestep_idx
            ] = -1.0
            episode_masked.embeddings.select(*episode.timestep.observations)[
                :, masked_observation_timestep_idx
            ] = -1.0

            episode_masked.embeddings = self.episode_builder._apply_position_encoding(
                embeddings=episode_masked.embeddings,
                step_index=episode_masked.index[0],  # pyright: ignore
            )

            masks[k] = self._build_random_masked_hindsight_control_attention_mask(
                episode_masked.index
            )

            embeddings[k] = self.encoder(
                src=episode_masked.packed_embeddings,
                mask=masks[k].data,
            )

            metrics[k] = objective(
                episode_masked, embeddings.get(k), masked_action_timestep_idx
            )

        for k, mask in masks.items():
            self.log_mask(k, mask)

        losses = metrics.select(
            *((k, "loss") for k in metrics.keys())
        )  # pyright: ignore
        # TODO: loss weights?
        metrics[("loss", "total")] = torch.stack(
            tuple(losses.values(True, True))
        ).mean()

        return metrics

    def training_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict(
            {
                "/".join(["train", *k]): v
                for k, v in metrics.items(include_nested=True, leaves_only=True)
            }
        )

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

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_forward_dynamics_attention_mask(
        cls, index: EpisodeIndex
    ) -> ForwardDynamicsAttentionMask:
        return ForwardDynamicsAttentionMask.build(
            index=index,
            legend=XFormersAttentionMaskLegend,
        )

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_inverse_dynamics_attention_mask(
        cls, index: EpisodeIndex, timestep: Timestep
    ) -> InverseDynamicsAttentionMask:
        return InverseDynamicsAttentionMask.build(
            index=index,
            timestep=timestep,
            legend=XFormersAttentionMaskLegend,
        )

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_random_masked_hindsight_control_attention_mask(
        cls, index: EpisodeIndex
    ) -> RandomMaskedHindsightControlAttentionMask:
        return RandomMaskedHindsightControlAttentionMask.build(
            index=index,
            legend=XFormersAttentionMaskLegend,
        )

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = cfg | {"scheduler": scheduler}

        return result

    def log_mask(self, name, mask):
        from wandb import Image  # noqa: PLC0415

        from cargpt.components.mask import WandbAttentionMaskLegend  # noqa: PLC0415

        if self.trainer.global_step == 0 and isinstance(self.logger, WandbLogger):
            img = Image(mask.with_legend(WandbAttentionMaskLegend).data)
            self.logger.log_image(name, [img])


class ForwardDynamicsPredictionObjective(Module):
    def __init__(self, head: Module, loss: Module) -> None:
        super().__init__()
        self.head = head
        self.loss = loss

    def forward(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
    ) -> TensorDict:
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

        return TensorDict.from_dict({"loss": loss})


class InverseDynamicsPredictionObjective(Module):
    def __init__(
        self,
        heads: ModuleDict,
        loss: Module,
        detokenizers: ModuleDict,
    ):
        super().__init__()
        self.heads = heads
        self.loss = loss
        self.detokenizers = detokenizers

    def forward(self, episode: Episode, embeddings: TensorDict) -> TensorDict:
        batch_size = embeddings.batch_size

        observation_index = episode.index.select(
            *episode.timestep.observations
        )  # pyright: ignore
        observation_index = TensorDict(
            {
                Objective.INVERSE_DYNAMICS: observation_index[:-1],
                Objective.INVERSE_DYNAMICS: observation_index[1:],
            },
            batch_size=torch.Size([mit.one(observation_index.batch_size) - 1]),
        )

        observations = embeddings.select(*observation_index.keys()).apply(
            lambda enc, idx: idx.parse(enc), observation_index
        )

        observations = TensorDict(
            {
                mask: pack([v[k] for k in episode.timestep.observations], "b t * d")[0]
                for mask, v in observations.items()
            },
            batch_size=observations.batch_size,
        )

        observations = observations.apply(Reduce("b t s d -> b t 1 d", "mean"))
        observations, _ = pack(
            [
                observations[Objective.INVERSE_DYNAMICS],
                observations[Objective.INVERSE_DYNAMICS],
            ],
            "b t *",
        )

        logits = TensorDict(
            {
                (t, n): self.heads[t][n](observations)  # pyright: ignore
                for (t, n) in episode.timestep.actions
            },
            batch_size=batch_size,
        )
        logits = logits.apply(Rearrange("b t d -> (b t) d"), batch_size=[])

        labels = episode.labels.select(*episode.timestep.actions)
        labels = labels.apply(lambda x: x[:, :-1])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])

        loss = logits.apply(self.loss, labels)

        with torch.no_grad():
            preds = logits.apply(lambda x: Categorical(logits=x).sample())
            pred_values = TensorDict(
                {k: v.apply(self.detokenizers[k]) for k, v in preds.items()},
                batch_size=preds.batch_size,
            )

            label_values = TensorDict(
                {k: v.apply(self.detokenizers[k]) for k, v in labels.items()},
                batch_size=labels.batch_size,
            )
            diff = pred_values.apply(
                lambda pred, lbl: (pred - lbl).abs().float().mean(),
                label_values,
                batch_size=[],
            )

        return TensorDict.from_dict({"loss": loss, "diff": diff})


class RandomMaskedHindsightControlObjective(Module):
    def __init__(self, heads: TokenModuleDict, loss: Module) -> None:
        super().__init__()
        self.heads = heads
        self.loss = loss

    def forward(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
        masked_action_timestep_idx: List[int],
    ) -> TensorDict:
        b, *_ = embedding.shape
        action_index = episode.index.select(
            *episode.timestep.actions
        )  # pyright: ignore
        action_embeddings = action_index[masked_action_timestep_idx].parse(embedding)

        logits = TensorDict(
            {
                (t, n): self.heads[t][n](emb)  # pyright: ignore
                for ((t, n), emb) in action_embeddings.items(
                    include_nested=True,
                    leaves_only=True,
                )
            },
            batch_size=[b],
        )
        logits = logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[])

        labels = episode.labels.select(*episode.timestep.actions).apply(
            lambda lbl: lbl[:, masked_action_timestep_idx]
        )
        labels = labels.apply(Rearrange("b t 1 -> (b t 1)"), batch_size=[])

        loss = logits.apply(self.loss, labels)

        return TensorDict.from_dict({"loss": loss})
