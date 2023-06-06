from collections import defaultdict
from typing import Any, Dict, List

import more_itertools as mit
import pytorch_lightning as pl
import torch
from loguru import logger
from einops import repeat, rearrange
from hydra.utils import instantiate
from jaxtyping import Float, Int
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from torch.nn import Linear, ModuleDict
from torch.nn.functional import gelu, cross_entropy, softmax

from cargpt.utils.wandb import LoadableFromArtifact, ValOutputsLoggingTableMixin


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


class Gato(pl.LightningModule, ValOutputsLoggingTableMixin, LoadableFromArtifact):
    """A Generalist Agent (Gato) https://arxiv.org/abs/2205.06175"""

    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        # for image embeddings
        logger.debug(
            "Instantiating image encoder",
            target=self.hparams.image_embedding._target_,  # type: ignore[union-attr]
        )
        self.image_embedding = instantiate(self.hparams.image_embedding)  # type: ignore[union-attr]
        self.tokenizers: ModuleDict = instantiate(self.hparams.sensor_tokenizers)
        self.sensor_detokenization = instantiate(self.hparams.sensor_detokenization)

        # for sensor embeddings
        logger.debug(
            "Instantiating sensor embeddings",
            target=self.hparams.sensor_embedding._target_,  # type: ignore[union-attr]
        )
        self.sensor_embedding = instantiate(self.hparams.sensor_embedding)  # type: ignore[union-attr]
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
            "Instantiating global position encodings",
            target=self.hparams.position_encoding.global_pos._target_,  # type: ignore[union-attr]
        )
        self.global_position = instantiate(self.hparams.position_encoding.global_pos)  # type: ignore[union-attr]
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

        self.action_keys: List[str] = self.hparams.action_keys  # type: ignore[assignment]
        self.metadata_keys: List[str] = self.hparams.metadata_keys  # type: ignore[assignment]

        self.val_table_main_columns = [
            f"{key}_{label}"
            for key in self.metadata_keys + self.action_keys
            for label in ("pred", "tgt")
        ]

    def _image_embeddings_and_tokens(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        image_features = self.image_embedding(frames)
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
        tokens_shift = torch.zeros(B, T, H * W, device=image_features.device)

        return image_features, image_tokens, tokens_shift

    def _metadata_embeddings_and_tokens(self, sample, keys=[]):
        embeddings = []
        tokens = []
        tokens_shift = []

        for key in keys:
            tokenizer = getattr(self.tokenizers, key)
            token: Int[Tensor, "b t"] = tokenizer(sample[key].clone())  # type: ignore[operator]
            token += self.hparams.tokens_shift[key]  # type: ignore[index]
            token = rearrange(token, "b t -> b t 1")
            embedding: Float[Tensor, "b t 1 e"] = self.sensor_embedding(token)
            token_shift = torch.ones_like(token) * self.hparams.tokens_shift[key]  # type: ignore[index]

            embeddings.append(embedding)
            tokens.append(token)
            tokens_shift.append(token_shift)

        # cat on tokens
        embeddings = torch.cat(embeddings, 2)  # type: ignore[assignment]
        tokens = torch.cat(tokens, 2)  # type: ignore[assignment]
        tokens_shift = torch.cat(tokens_shift, 2)  # type: ignore[assignment]

        return embeddings, tokens, tokens_shift

    def _action_embeddings_and_tokens(self, sample, keys=[]):
        embeddings = []
        tokens = []
        tokens_shift = []

        for key in keys:
            tokenizer = getattr(self.tokenizers, key)
            token: Int[Tensor, "b t"] = tokenizer(sample[key].clone())  # type: ignore[operator]
            token += self.hparams.tokens_shift[key]  # type: ignore[index]
            token = rearrange(token, "b t -> b t 1")
            embedding: Float[Tensor, "b t 1 e"] = self.sensor_embedding(token)
            token_shift = torch.ones_like(token) * self.hparams.tokens_shift[key]  # type: ignore[index]

            embeddings.append(embedding)
            tokens.append(token)
            tokens_shift.append(token_shift)

        # cat on tokens
        embeddings = torch.cat(embeddings, 2)  # type: ignore[assignment]
        tokens = torch.cat(tokens, 2)  # type: ignore[assignment]
        tokens_shift = torch.cat(tokens_shift, 2)  # type: ignore[assignment]
        b, t, n, e = embeddings.shape  # type: ignore[attr-defined]

        position_encoded: Float[Tensor, "b t n e"] = (
            self.action_position(torch.tensor([0], device=embeddings.device))  # type: ignore[attr-defined]
            .view(1, 1, 1, e)
            .repeat(b, t, n, 1)
        )

        embeddings += position_encoded

        return embeddings, tokens, tokens_shift

    def _step(self, sample):
        episode, episode_labels, episode_labels_shift = self._make_episode(sample)

        logits = self.forward(
            episode=episode[:, :-1],
        )

        labels = episode_labels[:, 1:]
        labels_shift = episode_labels_shift[:, 1:]

        return logits, labels.to(torch.int64), labels_shift.to(torch.int64)

    def _make_episode(self, sample):
        # tokenization + embeddings
        (
            image_embeddings,
            image_tokens,
            image_tokens_shift,
        ) = self._image_embeddings_and_tokens(sample["frames"])
        (
            metadata_embeddings,
            metadata_tokens,
            metadata_tokens_shift,
        ) = self._metadata_embeddings_and_tokens(sample, keys=self.metadata_keys)
        (
            action_embeddings,
            action_tokens,
            action_tokens_shift,
        ) = self._action_embeddings_and_tokens(sample, keys=self.action_keys)

        observations = torch.cat([image_embeddings, metadata_embeddings], 2)
        observation_tokens = torch.cat([image_tokens, metadata_tokens], 2)

        b, t, o, d = observations.shape
        _, _, a, _ = action_embeddings.shape

        # add local positional (along o) embedding to all observations image + metadata
        local_positions = torch.arange(0, o, dtype=torch.int, device=observations.device)  # type: ignore[attr-defined]
        local_positions_encoded = (
            self.local_position(local_positions).view(1, 1, o, d).repeat(b, t, 1, 1)
        )
        observations += local_positions_encoded

        separator_tokens = (
            torch.tensor(
                [self.hparams.special_tokens.sep],  # type: ignore[union-attr]
                dtype=torch.int,
                device=observations.device,
            )
            .view(1, 1, 1)
            .repeat(b, t, 1)
        )
        separator: Float[Tensor, "b t 1 d"] = self.sensor_embedding(separator_tokens)

        # construct full sequence: [[o, |, a], [o, |, a], ...]
        episode = torch.cat([observations, separator, action_embeddings], dim=2)
        episode_labels = torch.cat(
            [observation_tokens, separator_tokens, action_tokens], dim=2
        )
        episode_labels_shift = torch.cat(
            [
                image_tokens_shift,
                metadata_tokens_shift,
                separator_tokens,
                action_tokens_shift,
            ],
            dim=2,
        )

        b, t, s, d = episode.shape
        # add global positional (along t) embedding to all tokens
        global_positions = torch.arange(0, t, dtype=torch.int, device=episode.device)  # type: ignore[attr-defined]
        global_positions_encoded = (
            self.global_position(global_positions).view(1, t, 1, d).repeat(b, 1, s, 1)
        )
        episode += global_positions_encoded

        episode = episode.view(b, t * (o + 1 + a), d)
        episode_labels = episode_labels.view(b, t * (o + 1 + a))
        episode_labels_shift = episode_labels_shift.view(b, t * (o + 1 + a))

        return episode, episode_labels, episode_labels_shift

    def _compute_l1_diff(self, logits, tgt_labels, labels_shift):
        logits = logits.detach().clone()
        tgt_labels = tgt_labels.detach().clone()
        b, t, c = logits.shape
        # flatten on batch dimension
        logits = logits.view(b * t, c)
        tgt_labels = tgt_labels.view(b * t)
        labels_shift = labels_shift.view(b * t)
        # Kick out ignore_index labels (-1)
        labels_mask = torch.where(tgt_labels >= 0)
        #
        # Softmax and take max
        #
        pred_labels = torch.argmax(softmax(logits, dim=1), dim=1)
        # unshift pred, tgt labels to bring it to [0, 1024)
        pred_labels -= labels_shift
        tgt_labels -= labels_shift
        pred_labels = pred_labels[labels_mask]
        tgt_labels = tgt_labels[labels_mask]

        # reverse view
        parts = [len(self.metadata_keys), 1, len(self.action_keys)]
        view_reshape = (b, -1, sum(parts))

        pred_observations_labels, _, pred_actions_labels = torch.split(
            pred_labels.view(*view_reshape), parts, dim=2
        )
        tgt_observations_labels, _, tgt_actions_labels = torch.split(
            tgt_labels.view(*view_reshape), parts, dim=2
        )

        # detokenize (tokens to real values)
        pred_observations_values = torch.zeros_like(
            pred_observations_labels, dtype=torch.float
        )
        pred_actions_values = torch.zeros_like(pred_actions_labels, dtype=torch.float)
        tgt_observations_values = torch.zeros_like(
            tgt_observations_labels, dtype=torch.float
        )
        tgt_actions_values = torch.zeros_like(tgt_actions_labels, dtype=torch.float)

        for idx, key in enumerate(self.metadata_keys):
            inv_func = self.sensor_detokenization[key]
            pred_observations_values[:, :, idx] = inv_func(
                pred_observations_labels[:, :, idx]
            )
            tgt_observations_values[:, :, idx] = inv_func(
                tgt_observations_labels[:, :, idx]
            )

        for idx, key in enumerate(self.action_keys):
            inv_func = self.sensor_detokenization[key]
            pred_actions_values[:, :, idx] = inv_func(pred_actions_labels[:, :, idx])
            tgt_actions_values[:, :, idx] = inv_func(tgt_actions_labels[:, :, idx])

        # Calculate L1
        obs_diff = torch.abs(tgt_observations_values - pred_observations_values)
        action_diff = torch.abs(tgt_actions_values - pred_actions_values)

        l1_loss = {}
        values = {}
        for idx, key in enumerate(self.metadata_keys):
            l1_loss[key] = obs_diff[:, :, idx].mean()
            values[f"{key}_pred"] = pred_observations_values[:, :, idx]
            values[f"{key}_tgt"] = tgt_observations_values[:, :, idx]

        for idx, key in enumerate(self.action_keys):
            l1_loss[key] = action_diff[:, :, idx].mean()
            values[f"{key}_pred"] = pred_actions_values[:, :, idx]
            values[f"{key}_tgt"] = tgt_actions_values[:, :, idx]

        return l1_loss, values

    def _compute_loss(self, logits, labels):
        b, t, c = logits.shape
        # flatten on batch dimension
        logits = logits.view(b * t, c)
        labels = labels.view(b * t)
        loss = cross_entropy(logits, labels, ignore_index=-1)
        return loss

    def training_step(self, batch, batch_idx):
        sample = self.prepare_batch(batch)
        pred, tgt, tgt_shift = self._step(sample)
        loss_categorical = self._compute_loss(pred, tgt)
        diff_l1, _ = self._compute_l1_diff(pred, tgt, tgt_shift)

        metrics = {"train/loss": loss_categorical}
        metrics.update({f"diff/train_{key}": value for key, value in diff_l1.items()})

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=pred.shape[0],
        )
        return loss_categorical

    def validation_step(self, batch, batch_idx):
        sample = self.prepare_batch(batch)
        pred, tgt, tgt_shift = self._step(sample)
        loss_categorical = self._compute_loss(pred, tgt)
        diff_l1, numeric_values = self._compute_l1_diff(pred, tgt, tgt_shift)

        metrics = {"val/loss": loss_categorical}
        metrics.update({f"diff/val_{key}": value for key, value in diff_l1.items()})

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=pred.shape[0],
        )
        self._log_val_outputs_dict(outputs_dict=numeric_values)

        return loss_categorical

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        sample = self.prepare_batch(batch)
        full_episode, *_ = self._make_episode(sample)
        B, timesteps, *_ = sample["frames"].shape
        ts_len = int(full_episode.shape[1] / timesteps)

        predictions: Dict[int, Dict[int, Dict[str, Dict[str, int | float]]]] = {
            b: {ts: defaultdict(dict) for ts in range(timesteps)} for b in range(B)
        }

        actions_to_predict = len(self.action_keys)
        history = torch.tensor([], device=full_episode.device)
        for ts in range(timesteps):
            observations_start_idx = ts * ts_len
            actions_start_idx = (ts + 1) * ts_len - actions_to_predict
            next_observations_with_sep = full_episode[
                :, observations_start_idx:actions_start_idx, :
            ].clone()
            history = torch.cat([history, next_observations_with_sep], dim=1)
            for key in self.action_keys:
                logits = self.forward(episode=history).detach()
                token: Int[Tensor, "b 1"] = torch.argmax(
                    torch.softmax(logits[:, -1:, :], dim=-1), dim=-1
                )

                # embed prediction
                embedded: Float[Tensor, "b 1 e"] = self.sensor_embedding(token)
                position: Float[Tensor, "1 e"] = self.action_position(
                    torch.tensor([0], device=embedded.device)
                )
                embedded += repeat(
                    position, "1 e -> b t e", b=embedded.shape[0], t=embedded.shape[1]
                )

                # append to episode
                history = torch.concat([history, embedded], dim=1)

                # get real value
                token -= self.hparams.tokens_shift[key]  # type: ignore[index]
                prediction: Float[Tensor, "b 1"] = self.sensor_detokenization[key](
                    token.clone()
                )

                for b in range(B):
                    predictions[b][ts][key]["pred"] = prediction[b, 0].item()
                    predictions[b][ts][key]["gt"] = sample[key][b, ts].item()

        return predictions

    def forward(
        self,
        *,
        episode: Float[Tensor, "b to d"],
    ):
        _, m, _ = episode.shape
        episode_mask = torch.triu(
            torch.ones(m, m, device=episode.device) * float("-inf"), diagonal=1
        )

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

    def on_validation_epoch_end(self) -> None:
        self._finish_val_outputs_logging()
