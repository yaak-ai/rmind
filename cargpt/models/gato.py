from typing import Optional

import more_itertools as mit
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from einops import repeat, rearrange
from hydra.utils import instantiate
from jaxtyping import Float, Int
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from torch.nn import Linear, ModuleDict
from torch.nn.functional import gelu, cross_entropy

from cargpt.utils.wandb import LoadableFromArtifact


class TransformerEncoderLayerGEGLU(torch.nn.TransformerEncoderLayer):
    """Replace the original linear transformation + ReLu with GEGLU.

    The paper: https://arxiv.org/pdf/2002.05202v1.pdf
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=lambda x: x,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        # Set activation to None, as GEGLU replaces both linear1 + activation
        self.linear1 = Linear(d_model, dim_feedforward * 2, **factory_kwargs)
        self.activation = None  # type: ignore[assignment]

    def _ff_block(self, x: Tensor) -> Tensor:
        # FFN_GEGLU eq. 6, https://arxiv.org/pdf/2002.05202v1.pdf
        x = self.linear1(x)
        xW, xV = x.chunk(2, dim=-1)
        geglu = gelu(xW) * xV
        # The original implementation with replacement
        x = self.linear2(self.dropout(geglu))
        return x


class Gato(pl.LightningModule, LoadableFromArtifact):
    """A Generalist Agent (Gato) https://arxiv.org/abs/2205.06175"""

    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        # for image embeddings
        logger.debug(
            "Instantiating image encoder",
            target=self.hparams.image_encoder._target_,  # type: ignore[union-attr]
        )
        self.image_encoder = instantiate(self.hparams.image_encoder)  # type: ignore[union-attr]
        logger.debug(
            "Instantiating sensor tokenizer",
            target=self.hparams.sensor_tokenizer.modules.continues._target_,  # type: ignore[union-attr]
        )
        self.sensor_tokenizer: ModuleDict = instantiate(self.hparams.sensor_tokenizer)  # type: ignore[union-attr]

        # for sensor embeddings
        logger.debug(
            "Instantiating sensor embeddings",
            target=self.hparams.sensor_embeddings._target_,  # type: ignore[union-attr]
        )
        self.sensor_encoder = instantiate(
            self.hparams.sensor_embeddings  # type: ignore[union-attr]
        )
        # position encoding
        logger.debug(
            "Instantiating patch row positional encodings",
            target=self.hparams.position_encoding.patch_row._target_,  # type: ignore[union-attr]
        )
        self.patch_row = instantiate(self.hparams.position_encoding.patch_row)  # type: ignore[union-attr]
        logger.debug(
            "Instantiating patch col positional encodings",
            target=self.hparams.position_encoding.patch_col._target_,  # type: ignore[union-attr]
        )
        self.patch_col = instantiate(self.hparams.position_encoding.patch_col)  # type: ignore[union-attr]
        logger.debug(
            "Instantiating local positional encodings",
            target=self.hparams.position_encoding.local._target_,  # type: ignore[union-attr]
        )
        self.local_position = instantiate(self.hparams.position_encoding.local)  # type: ignore[union-attr]
        logger.debug(
            "Instantiating action position encodings",
            target=self.hparams.position_encoding.action._target_,  # type: ignore[union-attr]
        )
        self.action_position = instantiate(self.hparams.position_encoding.action)  # type: ignore[union-attr]

        # network
        logger.debug(
            "Instantiating gato model",
            target=self.hparams.gpt._target_,  # type: ignore[union-attr]
        )  # type: ignore[union-attr]
        self.gpt = instantiate(self.hparams.gpt)  # type: ignore[union-attr]
        logger.debug(
            "Instantiating classifier layer",
            target=self.hparams.classifier._target_,  # type: ignore[union-attr]
        )
        self.classifier = instantiate(self.hparams.classifier)  # type: ignore[union-attr]

    def _image_embeddings_and_tokens(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        image_features = self.image_encoder(frames)
        _, D, H, W = image_features.shape
        image_features = rearrange(image_features, "(B T) D H W -> B T H W D", T=T)

        row_index = repeat(
            torch.arange(0, H, device=frames.device, dtype=torch.int),
            "H -> B T H W",
            B=B,
            T=T,
            W=W,
        )
        col_index = repeat(
            torch.arange(0, W, device=frames.device, dtype=torch.int),
            "W -> B T H W",
            B=B,
            T=T,
            H=H,
        )

        # BTHW -> BTHWD
        patch_pos_row: Float[Tensor, "B T H W D"] = self.patch_row(row_index)
        patch_pos_col: Float[Tensor, "B T H W D"] = self.patch_col(col_index)

        image_features += patch_pos_col + patch_pos_row
        # BTHWD -> BT(HW)D
        image_features = image_features.view(B, T, H * W, D)
        # for resnet image tokens are fake labels but for VQ-VAE
        # they would be index into the visual vocabulary
        image_tokens = self.hparams.masks.image * torch.ones(  # type: ignore[union-attr]
            B, T, H * W, device=image_features.device
        )

        return image_features, image_tokens

    def _metadata_embeddings_and_tokens(self, sample, keys=[]):
        embeddings = []
        tokens = []

        for key in keys:
            token: Int[Tensor, "b t"] = (
                self.sensor_tokenizer.continues(sample[key])
                if sample[key].dtype is torch.float64
                else sample[key]
            )
            token += self.hparams.tokens_shift[key]
            token = rearrange(token, "b t -> b t 1")
            embedding: Float[Tensor, "b t 1 e"] = self.sensor_encoder(token)

            embeddings.append(embedding)
            tokens.append(token)

        # cat on tokens
        embeddings = torch.cat(embeddings, 2)  # type: ignore[assignment]
        tokens = torch.cat(tokens, 2)  # type: ignore[assignment]

        # add local positional embedding
        b, t, n, e = embeddings.shape  # type: ignore[attr-defined]
        positions = torch.arange(0, t, dtype=torch.int, device=embeddings.device)  # type: ignore[attr-defined]
        positions_encoded = (
            self.local_position(positions).view(1, t, 1, e).repeat(b, 1, n, 1)
        )
        embeddings += positions_encoded

        return embeddings, tokens

    def _action_embeddings_and_tokens(self, sample, keys=[]):
        embeddings = []
        tokens = []

        for key in keys:
            token: Int[Tensor, "b t"] = (
                self.sensor_tokenizer.continues(sample[key])
                if sample[key].dtype is torch.float64
                else sample[key]
            )
            token += self.hparams.tokens_shift[key]
            token = rearrange(token, "b t -> b t 1")
            embedding: Float[Tensor, "b t 1 e"] = self.sensor_encoder(token)

            embeddings.append(embedding)
            tokens.append(token)

        # cat on tokens
        embeddings = torch.cat(embeddings, 2)  # type: ignore[assignment]
        tokens = torch.cat(tokens, 2)  # type: ignore[assignment]
        b, t, n, e = embeddings.shape  # type: ignore[attr-defined]

        position_encoded: Float[Tensor, "1 e"] = (
            self.action_position(torch.tensor([0], device=embeddings.device))  # type: ignore[attr-defined]
            .view(1, 1, 1, e)
            .repeat(b, t, n, 1)
        )

        embeddings += position_encoded

        return embeddings, tokens

    def _step(self, sample):
        # tokenization + embeddings
        image_embeddings, image_tokens = self._image_embeddings_and_tokens(
            sample["frames"]
        )
        metadata_embeddings, metadata_tokens = self._metadata_embeddings_and_tokens(
            sample, keys=self.hparams.metadata_keys
        )
        action_embeddings, action_tokens = self._action_embeddings_and_tokens(
            sample, keys=self.hparams.action_keys
        )

        observations = torch.cat([image_embeddings, metadata_embeddings], 2)
        observation_tokens = torch.cat([image_tokens, metadata_tokens], 2)

        b, t, o, d = observations.shape
        _, _, a, _ = action_embeddings.shape

        separator_tokens = (
            torch.tensor(
                [self.hparams.special_tokens.sep],  # type: ignore[union-attr]
                dtype=torch.int,
                device=observations.device,
            )
            .view(1, 1, 1)
            .repeat(b, t, 1)
        )
        separator: Float[Tensor, "b t 1 d"] = self.sensor_encoder(separator_tokens)

        # construct full sequence: [[o, |, a], [o, |, a], ...]
        episode = torch.cat([observations, separator, action_embeddings], dim=2)
        episode_labels = torch.cat(
            [observation_tokens, separator_tokens, action_tokens], dim=2
        )

        episode = episode.view(b, t * (o + 1 + a), d)
        episode_labels = episode_labels.view(b, t * (o + 1 + a))

        logits = self.forward(
            episode=episode[:, :-1],
        )

        labels = episode_labels[:, 1:]

        return logits, labels.to(torch.int64)

    def _compute_loss(self, logits, labels):
        b, t, c = logits.shape
        # flatten on batch dimension
        logits = logits.view(b * t, c)
        labels = labels.view(b * t)
        loss = cross_entropy(logits, labels, ignore_index=-1, reduction="sum")
        return loss

    def training_step(self, batch, batch_idx):
        sample = self.prepare_batch(batch)
        pred, tgt = self._step(sample)
        loss = self._compute_loss(pred, tgt)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=pred.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        sample = self.prepare_batch(batch)
        pred, tgt = self._step(sample)
        loss = self._compute_loss(pred, tgt)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=pred.shape[0],
        )

        # TODO: Logging to table
        return loss

    def forward(
        self,
        *,
        episode: Float[Tensor, "b to d"],
    ):
        _, m, _ = episode.shape
        episode_mask = torch.tril(torch.ones(m, m, device=episode.device)).float()
        episode_mask = episode_mask.masked_fill(episode_mask == 0.0, -torch.inf)

        features = self.gpt(
            src=episode,
            mask=episode_mask,
        )
        logits = self.classifier(features)

        return logits

    def prepare_batch(self, batch):
        clips = mit.one(batch["clips"].values())
        frames = clips["frames"].to(self.device)
        speed = clips["meta"]["VehicleMotion_speed"].to(self.device)
        steering = clips["meta"]["VehicleMotion_steering_angle_normalized"].to(
            self.device
        )
        gas = clips["meta"]["VehicleMotion_gas_pedal_normalized"].to(self.device)
        brake = clips["meta"]["VehicleMotion_brake_pedal_normalized"].to(self.device)
        turn = clips["meta"]["VehicleState_turn_signal"].to(self.device)

        sample = {
            "frames": frames,
            "VehicleMotion_speed": speed,
            "VehicleMotion_steering_angle_normalized": steering,
            "VehicleMotion_gas_pedal_normalized": gas,
            "VehicleMotion_brake_pedal_normalized": brake,
            "VehicleState_turn_signal": turn,
        }

        return sample

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (lr_scheduler_cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(lr_scheduler_cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = {**lr_scheduler_cfg, **{"scheduler": scheduler}}

        return result

    def on_validation_epoch_start(self):
        if isinstance(logger := self.logger, pl.loggers.WandbLogger) and isinstance(
            logger.experiment, wandb.wandb_run.Run
        ):
            self._tables = {}
            # TODO

    def on_validation_epoch_end(self):
        if hasattr(self, "_tables") and self.trainer.state.stage != "sanity_check":
            run: wandb.wandb_run.Run = self.logger.experiment  # type: ignore[union-attr]

            if table := self._tables.get(name := "outputs"):
                table.add_column("_step", list(map(int, table.get_index())))

                frame_table: Optional[
                    wandb.sdk.wandb_artifacts.ArtifactManifestEntry
                ] = None

                _table = (
                    table
                    if frame_table is None
                    else wandb.JoinedTable(frame_table, table, "_step")
                )

                artifact = wandb.Artifact(f"run-{run.id}-val_{name}", "run_table")
                artifact.add(_table, name)
                run.log_artifact(artifact)
