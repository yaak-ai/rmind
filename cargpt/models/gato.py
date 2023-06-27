from collections import defaultdict
from typing import Any, Dict, List

import more_itertools as mit
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from hydra.utils import instantiate
from jaxtyping import Float, Int
from loguru import logger
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import Tensor
from torch.nn import Linear, ModuleDict
from torch.nn.functional import gelu

from cargpt.utils._wandb import LoadableFromArtifact, ValOutputsLoggingTableMixin


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
        self.linear1 = Linear(d_model, dim_feedforward * 2, **factory_kwargs)  # type: ignore
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
        logger.debug(
            "Instantiating image tokens",
            target=self.hparams.image_tokens._target_,  # type: ignore[union-attr]
        )
        self.image_tokens = instantiate(self.hparams.image_tokens)  # type: ignore[union-attr]
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

        # masking
        logger.debug(
            "Instantiating attention masking",
            target=self.hparams.attention_mask._target_,  # type: ignore[union-attr]
        )  # type: ignore[union-attr]
        self.attention_mask = instantiate(self.hparams.attention_mask)  # type: ignore[union-attr]
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
        logger.debug(
            "Instantiating regressor layer",
            target=self.hparams.classifier._target_,  # type: ignore[union-attr]
        )
        self.regressor = instantiate(self.hparams.regressor)  # type: ignore[union-attr]

        logger.debug(
            "Instantiating categorical loss",
            target=self.hparams.loss.categorical._target_,  # type: ignore[union-attr]
        )
        self.loss_categorical = instantiate(self.hparams.loss.categorical)  # type: ignore[union-attr]
        logger.debug(
            "Instantiating l1 loss",
            target=self.hparams.loss.l1._target_,  # type: ignore[union-attr]
        )
        self.loss_l1 = instantiate(self.hparams.loss.l1)  # type: ignore[union-attr]
        logger.debug(
            "Instantiating diff",
            target=self.hparams.diff._target_,  # type: ignore[union-attr]
        )
        self.diff = instantiate(self.hparams.diff)

        self.action_keys: List[str] = self.hparams.action_keys  # type: ignore[assignment]
        self.metadata_keys: List[str] = self.hparams.metadata_keys  # type: ignore[assignment]

        self.val_table_main_columns = [
            f"{key}_{label}"
            for key in self.metadata_keys + self.action_keys  # type: ignore
            for label in ("pred", "tgt")
        ]

    def _image_embeddings_and_tokens(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        image_features = self.image_embedding(frames)
        image_features, image_tokens = self.image_tokens(
            image_features,
            self.hparams.tokens_shift["ImageEncoder"],  # type: ignore
            self.sensor_embedding,
        )
        tokens_shift = torch.ones_like(image_tokens) * self.hparams.tokens_shift["ImageEncoder"]  # type: ignore[index]
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
        image_tokens = image_tokens.view(B, T, H * W)
        tokens_shift = tokens_shift.view(B, T, H * W)
        # Is ignore in L1 loss since only computed over metadata and actions values
        values = float("inf") * torch.ones(B, T, H * W, device=image_features.device)

        return image_features, image_tokens, tokens_shift, values

    def _metadata_embeddings_and_tokens(self, sample, keys=[]):
        embeddings = []
        tokens = []
        tokens_shift = []
        values = []

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
            values.append(sample[key].unsqueeze(-1))

        # cat on tokens
        embeddings = torch.cat(embeddings, 2)  # type: ignore[assignment]
        tokens = torch.cat(tokens, 2)  # type: ignore[assignment]
        tokens_shift = torch.cat(tokens_shift, 2)  # type: ignore[assignment]
        values = torch.cat(values, 2)  # type: ignore[assignment]

        return embeddings, tokens, tokens_shift, values

    def _action_embeddings_and_tokens(self, sample, keys=[]):
        embeddings = []
        tokens = []
        tokens_shift = []
        values = []

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
            values.append(sample[key].unsqueeze(-1))

        # cat on tokens
        embeddings = torch.cat(embeddings, 2)  # type: ignore[assignment]
        tokens = torch.cat(tokens, 2)  # type: ignore[assignment]
        tokens_shift = torch.cat(tokens_shift, 2)  # type: ignore[assignment]
        values = torch.cat(values, 2)  # type: ignore[assignment]
        b, t, n, e = embeddings.shape  # type: ignore[attr-defined]

        position_encoded: Float[Tensor, "b t n e"] = (
            self.action_position(torch.tensor([0], device=embeddings.device))  # type: ignore[attr-defined]
            .view(1, 1, 1, e)
            .repeat(b, t, n, 1)
        )

        embeddings += position_encoded

        return embeddings, tokens, tokens_shift, values

    def _step(self, sample):
        (
            episode,
            episode_labels,
            episode_labels_shift,
            episode_values,
            episode_mask,
        ) = self._make_episode(sample)

        logits, values = self.forward(
            episode=episode[:, :-1], episode_mask=episode_mask[:-1, :-1]
        )

        labels = episode_labels[:, 1:]
        labels_shift = episode_labels_shift[:, 1:]
        episode_values = episode_values[:, 1:]

        return (
            logits,
            values,
            labels.to(torch.int64),
            labels_shift.to(torch.int64),
            episode_values,
        )

    def _make_episode(self, sample):
        # tokenization + embeddings
        (
            image_embeddings,
            image_tokens,
            image_tokens_shift,
            image_values,
        ) = self._image_embeddings_and_tokens(sample["frames"])
        (
            metadata_embeddings,
            metadata_tokens,
            metadata_tokens_shift,
            metadata_values,
        ) = self._metadata_embeddings_and_tokens(sample, keys=self.metadata_keys)
        (
            action_embeddings,
            action_tokens,
            action_tokens_shift,
            action_values,
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
        episode_values = torch.cat(
            [
                image_values,
                metadata_values,
                float("inf") * torch.ones_like(separator_tokens),
                action_values,
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
        episode_values = episode_values.view(b, t * (o + 1 + a))

        episode_mask = self.attention_mask(
            image_embeddings, metadata_embeddings, separator, action_embeddings
        )

        return (
            episode,
            episode_labels,
            episode_labels_shift,
            episode_values,
            episode_mask,
        )

    @staticmethod
    def causal_attention_mask(
        image_embeddings, metadata_embeddings, separator, action_embeddings
    ):
        # causal masking fr fr
        _, t, n_i, _ = image_embeddings.shape
        _, _, m, _ = metadata_embeddings.shape
        _, _, s, _ = separator.shape
        _, _, a, _ = action_embeddings.shape
        seqlen = n_i + m + s + a
        episode_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=image_embeddings.device) * float("-inf"),
            diagonal=1,
        )

        return episode_mask

    @staticmethod
    def block_causal_sensor_attention_mask(
        image_embeddings, metadata_embeddings, separator, action_embeddings
    ):
        _, t, n_i, _ = image_embeddings.shape
        _, _, m, _ = metadata_embeddings.shape
        _, _, s, _ = separator.shape
        _, _, a, _ = action_embeddings.shape
        seqlen = n_i + m + s + a
        episode_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=image_embeddings.device) * float("-inf"),
            diagonal=1,
        )
        num_self_censor = m + s + a

        # Self masking
        for ts_row in range(0, t):
            row = seqlen * ts_row + n_i - 1
            for ts_col in range(0, ts_row + 1):
                col = seqlen * ts_col + n_i
                for i in range(num_self_censor):
                    episode_mask[row + i + 1, col + i] = 0

        return episode_mask

    @staticmethod
    def block_all_sensor_attention_mask(
        image_embeddings, metadata_embeddings, separator, action_embeddings
    ):
        _, t, n_i, _ = image_embeddings.shape
        _, _, m, _ = metadata_embeddings.shape
        _, _, s, _ = separator.shape
        _, _, a, _ = action_embeddings.shape
        seqlen = n_i + m + s + a
        episode_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=image_embeddings.device) * float("-inf"),
            diagonal=1,
        )
        num_self_censor = m + s + a

        # Self masking
        for ts_row in range(0, t):
            row = seqlen * ts_row + n_i - 1
            for ts_col in range(0, ts_row + 1):
                col = seqlen * ts_col + n_i
                episode_mask[
                    row: row + num_self_censor, col: col + num_self_censor
                ] = float("-inf")
                for i in range(num_self_censor):
                    episode_mask[row + i + 1, col + i] = 0

        return episode_mask

    @staticmethod
    def block_all_sensor_and_image_from_sensor_attention_mask(
        self, image_embeddings, metadata_embeddings, separator, action_embeddings
    ):
        _, t, n_i, _ = image_embeddings.shape
        _, _, m, _ = metadata_embeddings.shape
        _, _, s, _ = separator.shape
        _, _, a, _ = action_embeddings.shape
        seqlen = n_i + m + s + a
        episode_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=image_embeddings.device) * float("-inf"),
            diagonal=1,
        )
        num_self_censor = m + s + a

        # Self masking
        for ts_col in range(0, t):
            col = seqlen * ts_col + n_i
            episode_mask[:, col: col + num_self_censor] = float("-inf")
            for ts_row in range(ts_col, t):
                row = seqlen * ts_row + n_i - 1
                for i in range(num_self_censor):
                    episode_mask[row + i + 1, col + i] = 0

        return episode_mask

    def _compute_diff(self, logits, tgt_labels, labels_shift):
        return self.diff(
            logits.detach(),
            tgt_labels.detach(),
            labels_shift.detach(),
            self.sensor_detokenization,
            self.metadata_keys,
            self.action_keys,
        )

    def _compute_loss_categorical(self, logits, labels):
        b, t, c = logits.shape
        # flatten on batch dimension
        logits = logits.view(b * t, c)
        labels = labels.view(b * t)
        if self.hparams.ignore_image_tokens:
            # ignore_index takes care of masking classes
            return self.loss_categorical(logits, labels)

        # If non zero image tokens label expected compute separate loss for image
        w = self.hparams.loss.weights.image  # type: ignore[union-attr]
        mask = torch.bitwise_and(0 <= labels, labels < self.hparams.tokens_shift["ImageEncoder"])  # type: ignore[index]
        metadata_logits = logits[mask]
        metadata_labels = labels[mask]
        metadata_loss = self.loss_categorical(
            metadata_logits,
            metadata_labels,
        )
        image_logits = logits[~mask]
        image_labels = labels[~mask]
        image_loss = self.loss_categorical(
            image_logits,
            image_labels,
        )
        loss = w * image_loss + (1 - w) * metadata_loss

        return loss

    def _compute_loss_l1(self, pred, tgt):
        b, t, c = pred.shape

        # flatten on batch dimension
        pred = pred.clone().view(b * t * c)
        tgt = tgt.clone().view(b * t)

        loss = self.loss_l1(
            pred,
            tgt,
        )

        # Ignore tokens which have been set to inf, f.ex image and sep
        # Since the don't correspont to real values
        mask = tgt == float("inf")
        loss[mask] = 0

        loss_masked = loss.sum() / (~mask).sum()

        return loss_masked

    def training_step(self, batch, batch_idx):
        sample = self.prepare_batch(batch)
        pred, pred_values, tgt, tgt_shift, tgt_values = self._step(sample)
        loss_categorical = self._compute_loss_categorical(pred, tgt)
        loss_l1 = self._compute_loss_l1(pred_values, tgt_values)
        diff_l1, _ = self._compute_diff(pred, tgt, tgt_shift)

        loss = (
            self.hparams.loss.weights.categorical * loss_categorical  # type: ignore[union-attr]
            + self.hparams.loss.weights.l1 * loss_l1  # type: ignore[union-attr]
        )

        metrics = {"train/loss": loss_categorical, "train/loss_l1": loss_l1}
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
        return loss

    def validation_step(self, batch, batch_idx):
        sample = self.prepare_batch(batch)
        pred, pred_values, tgt, tgt_shift, tgt_values = self._step(sample)
        loss_categorical = self._compute_loss_categorical(pred, tgt)
        loss_l1 = self._compute_loss_l1(pred_values, tgt_values)
        diff_l1, numeric_values = self._compute_diff(pred, tgt, tgt_shift)

        loss = (
            self.hparams.loss.weights.categorical * loss_categorical  # type: ignore[union-attr]
            + self.hparams.loss.weights.l1 * loss_l1  # type: ignore[union-attr]
        )

        metrics = {"val/loss": loss_categorical, "val/loss_l1": loss_l1}
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

        return loss

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        start_timestep: int = 0,
    ) -> Any:
        sample = self.prepare_batch(batch)
        full_episode, *_, episode_mask = self._make_episode(sample)
        B, timesteps, *_ = sample["frames"].shape
        ts_len = int(full_episode.shape[1] / timesteps)

        actions_to_predict = len(self.action_keys)
        if start_timestep < 0:
            start_timestep = timesteps + start_timestep
        timesteps_to_predict = tuple(range(start_timestep, timesteps))
        history = full_episode[:, : (start_timestep * ts_len)].clone()

        predictions: Dict[int, Dict[int, Dict[str, Dict[str, int | float]]]] = {
            b: {ts: defaultdict(dict) for ts in timesteps_to_predict} for b in range(B)
        }

        m, n = episode_mask.shape
        for ts in timesteps_to_predict:
            observations_start_idx = ts * ts_len
            actions_start_idx = (ts + 1) * ts_len - actions_to_predict
            next_observations_with_sep = full_episode[
                :, observations_start_idx:actions_start_idx, :
            ].clone()
            history = torch.cat([history, next_observations_with_sep], dim=1)
            for key in self.action_keys:
                logits, _ = self.forward(
                    episode=history, episode_mask=episode_mask[:m, :n]
                )
                logits = logits.detach()
                token: Int[Tensor, "b 1"] = torch.argmax(
                    torch.softmax(logits[:, -1:, :], dim=-1), dim=-1
                )

                # embed prediction
                embedded: Float[Tensor, "b 1 e"] = self.sensor_embedding(token)
                b, t, _ = embedded.shape
                position: Float[Tensor, "1 e"] = self.action_position(
                    torch.tensor([0], device=embedded.device)
                )
                embedded += repeat(position, "1 e -> b t e", b=b, t=t)
                global_position: Float[Tensor, "1 e"] = self.global_position(
                    torch.tensor([ts]).to(self.device)
                )
                embedded += repeat(global_position, "1 e -> b t e", b=b, t=t)

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
        self, *, episode: Float[Tensor, "b to d"], episode_mask: Float[Tensor, "to to"]
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
        values = self.regressor(features)

        return logits, values

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
