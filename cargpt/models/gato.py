from collections import defaultdict
from typing import Any, Dict, List, Optional

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

from cargpt.utils._wandb import (
    LoadableFromArtifact,
    TrainValAttnMapLoggingMixin,
    ValOutputsLoggingTableMixin,
)


class HFGPT2(pl.LightningModule):
    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.llm = instantiate(self.hparams.llm)

    def forward(
        self,
        inputs_embeds: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
        episode_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Any:
        output = self.llm(
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

        return output

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()


class TorchGPT2(pl.LightningModule):
    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.llm = instantiate(self.hparams.llm)
        self.classifier = instantiate(self.hparams.classifier)
        self.loss_fn = instantiate(self.hparams.loss)

    def forward(
        self,
        inputs_embeds: Tensor,
        episode_mask: Tensor,
        return_dict: bool,
        output_hidden_states: bool,
        labels: Optional[Tensor] = None,
    ) -> Any:
        output = {}
        x = self.llm(src=inputs_embeds, mask=episode_mask)
        logits = self.classifier(x)
        output["logits"] = logits
        output["hidden_states"] = [x]
        # right shift logits

        if labels is not None:
            shifted_logits = logits[:, :-1, :].contiguous()
            b, t, c = shifted_logits.shape
            # left shift labels
            shifted_labels = labels[:, 1:].contiguous()
            # flatten on batch dimension
            logits_flattened = shifted_logits.view(b * t, c)
            labels_flattened = shifted_labels.view(b * t)
            loss = self.loss_fn(logits_flattened, labels_flattened)

            output["loss"] = loss

        return output

    def get_output_embeddings(self):
        return self.classifier


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


class Gato(
    pl.LightningModule,
    ValOutputsLoggingTableMixin,
    TrainValAttnMapLoggingMixin,
    LoadableFromArtifact,
):
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
        # for sensor dropout (only activated in training)
        logger.debug(
            "Instantiating sensor dropout",
            target=self.hparams.sensor_dropout._target_,  # type: ignore[union-attr]
        )
        self.sensor_dropout = instantiate(self.hparams.sensor_dropout)  # type: ignore[union-attr]

        # position encoding
        self.init_position_encodings()

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
            "Instantiating regressor layer",
            target=self.hparams.regressor._target_,  # type: ignore[union-attr]
        )
        self.regressor = instantiate(self.hparams.regressor)  # type: ignore[union-attr]

        # Losses
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
        # tie input sensor embeddings to LM Head of GPT2
        self.sensor_embedding.weight = self.gpt.get_output_embeddings().weight
        assert torch.all(
            self.sensor_embedding.weight == self.gpt.get_output_embeddings().weight
        )

    def init_position_encodings(self):
        self.have_position_encoding = self.hparams.have_position_encoding

        if self.have_position_encoding.patch:  # type: ignore[union-attr]
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
        if self.have_position_encoding.local:  # type: ignore[union-attr]
            logger.debug(
                "Instantiating local positional encodings",
                target=self.hparams.position_encoding.local._target_,  # type: ignore[union-attr]
            )
            self.local_position = instantiate(self.hparams.position_encoding.local)  # type: ignore[union-attr]
        if self.have_position_encoding.global_pos:  # type: ignore[union-attr]
            logger.debug(
                "Instantiating global position encodings",
                target=self.hparams.position_encoding.global_pos._target_,  # type: ignore[union-attr]
            )
            self.global_position = instantiate(self.hparams.position_encoding.global_pos)  # type: ignore[union-attr]
        if self.have_position_encoding.action:  # type: ignore[union-attr]
            logger.debug(
                "Instantiating action position encodings",
                target=self.hparams.position_encoding.action._target_,  # type: ignore[union-attr]
            )
            self.action_position = instantiate(self.hparams.position_encoding.action)  # type: ignore[union-attr]

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

        if self.have_position_encoding.patch:  # type: ignore[union-attr]
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

    def _metadata_embeddings_and_tokens(self, metadata):
        embeddings = []
        tokens = []
        tokens_shift = []
        values = []

        for k, v in metadata:
            tokenizer = getattr(self.tokenizers, k)
            token: Int[Tensor, "b t"] = tokenizer(v.clone())  # type: ignore[operator]
            token += self.hparams.tokens_shift[k]  # type: ignore[index]
            token = rearrange(token, "b t -> b t 1")
            embedding: Float[Tensor, "b t 1 e"] = self.sensor_embedding(token)
            token_shift = torch.ones_like(token) * self.hparams.tokens_shift[k]  # type: ignore[index]

            embeddings.append(embedding)
            tokens.append(token)
            tokens_shift.append(token_shift)
            values.append(v.unsqueeze(-1))

        # cat on tokens
        embeddings = torch.cat(embeddings, 2)  # type: ignore[assignment]
        tokens = torch.cat(tokens, 2)  # type: ignore[assignment]
        tokens_shift = torch.cat(tokens_shift, 2)  # type: ignore[assignment]
        values = torch.cat(values, 2)  # type: ignore[assignment]

        return embeddings, tokens, tokens_shift, values

    def _action_embeddings_and_tokens(self, actions):
        embeddings = []
        tokens = []
        tokens_shift = []
        values = []

        for k, v in actions:
            tokenizer = getattr(self.tokenizers, k)
            token: Int[Tensor, "b t"] = tokenizer(v.clone())  # type: ignore[operator]
            token += self.hparams.tokens_shift[k]  # type: ignore[index]
            token = rearrange(token, "b t -> b t 1")
            embedding: Float[Tensor, "b t 1 e"] = self.sensor_embedding(token)
            token_shift = torch.ones_like(token) * self.hparams.tokens_shift[k]  # type: ignore[index]

            embeddings.append(embedding)
            tokens.append(token)
            tokens_shift.append(token_shift)
            values.append(v.unsqueeze(-1))

        # cat on tokens
        embeddings = torch.cat(embeddings, 2)  # type: ignore[assignment]
        tokens = torch.cat(tokens, 2)  # type: ignore[assignment]
        tokens_shift = torch.cat(tokens_shift, 2)  # type: ignore[assignment]
        values = torch.cat(values, 2)  # type: ignore[assignment]
        b, t, n, e = embeddings.shape  # type: ignore[attr-defined]

        if self.have_position_encoding.action:  # type: ignore[union-attr]
            position_encoded: Float[Tensor, "b t n e"] = (
                self.action_position(torch.tensor([0], device=embeddings.device))  # type: ignore[attr-defined]
                .view(1, 1, 1, e)
                .repeat(b, t, n, 1)
            )

            embeddings += position_encoded

        return embeddings, tokens, tokens_shift, values

    def _step(self, batch: Any, is_training: bool = False):
        (
            episode,
            episode_labels,
            episode_labels_shift,
            episode_values,
            episode_mask,
        ) = self._make_episode(batch, is_training=is_training)

        episode_loss, episode_logits, episode_pred_values = self.forward(
            episode=episode, episode_labels=episode_labels, episode_mask=episode_mask
        )

        # left shift gt
        episode_gt_labels = episode_labels[:, 1:]
        episode_gt_labels_shift = episode_labels_shift[:, 1:]
        episode_gt_values = episode_values[:, 1:]
        # GPT2 parses whole sequence
        episode_logits = episode_logits[:, :-1]
        episode_pred_values = episode_pred_values[:, :-1]

        return (
            episode_loss,
            episode_logits,
            episode_pred_values,
            episode_gt_labels,
            episode_gt_labels_shift,
            episode_gt_values,
        )

    def add_special_tokens(
        self,
        observations,
        observation_tokens,
        observation_tokens_shift,
        observation_values,
        action_embeddings,
        action_tokens,
        action_tokens_shift,
        action_values,
    ):
        episode = []
        tokens = []
        shift = []
        values = []
        b, t, _, d = observations.shape

        if self.hparams.have_special_tokens.bos:  # type: ignore[union-attr]
            spl_tokens = (
                torch.tensor(
                    [self.hparams.special_tokens.bos],  # type: ignore[union-attr]
                    dtype=torch.int,
                    device=observations.device,
                )
                .view(1, 1, 1)
                .repeat(b, t, 1)
            )
            episode += [self.sensor_embedding(spl_tokens)]
            tokens += [spl_tokens]
            shift += [torch.zeros_like(spl_tokens)]
            values += [float("inf") * torch.ones_like(spl_tokens)]

        episode += [observations]
        tokens += [observation_tokens]
        shift += [observation_tokens_shift]
        values += [observation_values]

        if self.hparams.have_special_tokens.sep:  # type: ignore[union-attr]
            spl_tokens = (
                torch.tensor(
                    [self.hparams.special_tokens.sep],  # type: ignore[union-attr]
                    dtype=torch.int,
                    device=observations.device,
                )
                .view(1, 1, 1)
                .repeat(b, t, 1)
            )
            episode += [self.sensor_embedding(spl_tokens)]
            tokens += [spl_tokens]
            shift += [torch.zeros_like(spl_tokens)]
            values += [float("inf") * torch.ones_like(spl_tokens)]

        episode += [action_embeddings]
        tokens += [action_tokens]
        shift += [action_tokens_shift]
        values += [action_values]

        if self.hparams.have_special_tokens.eos:  # type: ignore[union-attr]
            spl_tokens = (
                torch.tensor(
                    [self.hparams.special_tokens.eos],  # type: ignore[union-attr]
                    dtype=torch.int,
                    device=observations.device,
                )
                .view(1, 1, 1)
                .repeat(b, t, 1)
            )
            episode += [self.sensor_embedding(spl_tokens)]
            tokens += [spl_tokens]
            shift += [torch.zeros_like(spl_tokens)]
            values += [float("inf") * torch.ones_like(spl_tokens)]

        return episode, tokens, shift, values

    def _make_episode(self, batch: Any, is_training: bool = False):
        frames = mit.only(batch["frames"].values())
        metadata = [(k, batch["meta"][k]) for k in self.metadata_keys]
        actions = [(k, batch["meta"][k]) for k in self.action_keys]
        # tokenization + embeddings
        (
            image_embeddings,
            image_tokens,
            image_tokens_shift,
            image_values,
        ) = self._image_embeddings_and_tokens(frames)
        (
            metadata_embeddings,
            metadata_tokens,
            metadata_tokens_shift,
            metadata_values,
        ) = self._metadata_embeddings_and_tokens(metadata)
        (
            action_embeddings,
            action_tokens,
            action_tokens_shift,
            action_values,
        ) = self._action_embeddings_and_tokens(actions)

        # https://arxiv.org/pdf/1905.11979.pdf
        # https://arxiv.org/pdf/1812.03079.pdf
        if is_training:
            embeddings = self.sensor_dropout([metadata_embeddings, action_embeddings])
            metadata_embeddings, action_embeddings = embeddings

        observations = torch.cat([image_embeddings, metadata_embeddings], 2)
        observation_tokens = torch.cat([image_tokens, metadata_tokens], 2)
        observation_tokens_shift = torch.cat(
            [image_tokens_shift, metadata_tokens_shift], 2
        )
        observation_values = torch.cat([image_values, metadata_values], 2)

        b, t, o, d = observations.shape
        _, _, a, _ = action_embeddings.shape

        if self.have_position_encoding.local:  # type: ignore[union-attr]
            # add local positional (along o) embedding to all observations image + metadata
            local_positions = torch.arange(0, o, dtype=torch.int, device=observations.device)  # type: ignore[attr-defined]
            local_positions_encoded = (
                self.local_position(local_positions).view(1, 1, o, d).repeat(b, t, 1, 1)
            )
            observations += local_positions_encoded

        (
            episode,
            episode_labels,
            episode_labels_shift,
            episode_values,
        ) = self.add_special_tokens(
            observations,
            observation_tokens,
            observation_tokens_shift,
            observation_values,
            action_embeddings,
            action_tokens,
            action_tokens_shift,
            action_values,
        )

        # construct full sequence: [[o, |, a], [o, |, a], ...]
        episode = torch.cat(episode, dim=2)
        episode_labels = torch.cat(episode_labels, dim=2)
        episode_labels_shift = torch.cat(episode_labels_shift, dim=2)
        episode_values = torch.cat(episode_values, dim=2)

        b, t, s, d = episode.shape
        if self.hparams.have_position_encoding.global_pos:  # type: ignore[union-attr]
            # add global positional (along t) embedding to all tokens
            # if using transformer_encoder we add it to embeddings if using hf_gp2 we pass it as position_ids param
            global_positions = torch.arange(0, t, dtype=torch.int, device=episode.device)  # type: ignore[attr-defined]
            global_positions_encoded = (
                self.global_position(global_positions)
                .view(1, t, 1, d)
                .repeat(b, 1, s, 1)
            )
            episode += global_positions_encoded

        episode = episode.view(b, t * (o + 1 + a), d)
        episode_labels = episode_labels.view(b, t * (o + 1 + a))
        episode_labels_shift = episode_labels_shift.view(b, t * (o + 1 + a))
        episode_values = episode_values.view(b, t * (o + 1 + a))

        episode_mask = self.attention_mask(episode.device)

        return (
            episode,
            episode_labels,
            episode_labels_shift,
            episode_values,
            episode_mask,
        )

    @staticmethod
    def causal_attention_mask(
        patch_row, patch_col, nun_metadata_keys, num_action_keys, clip_len, device
    ):
        # causal masking fr fr
        seqlen = (
            patch_row * patch_col + len(nun_metadata_keys) + 1 + len(num_action_keys)
        ) * clip_len
        episode_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=device) * float("-inf"),
            diagonal=1,
        )

        return episode_mask

    @staticmethod
    def block_causal_sensor_attention_mask(
        patch_row, patch_col, nun_metadata_keys, num_action_keys, clip_len, device
    ):
        episode_mask = Gato.causal_attention_mask(
            patch_row, patch_col, nun_metadata_keys, num_action_keys, clip_len, device
        )
        n_i = patch_row * patch_row
        num_self_censor = len(nun_metadata_keys) + 1 + len(num_action_keys)
        seqlen = n_i + num_self_censor

        # Self masking
        for ts_row in range(0, clip_len):
            row = seqlen * ts_row + n_i - 1
            for ts_col in range(0, ts_row + 1):
                col = seqlen * ts_col + n_i
                for i in range(num_self_censor):
                    episode_mask[row + i + 1, col + i] = 0

        return episode_mask

    @staticmethod
    def block_all_sensor_attention_mask(
        patch_row, patch_col, nun_metadata_keys, num_action_keys, clip_len, device
    ):
        episode_mask = Gato.causal_attention_mask(
            patch_row, patch_col, nun_metadata_keys, num_action_keys, clip_len, device
        )
        n_i = patch_row * patch_row
        num_self_censor = len(nun_metadata_keys) + 1 + len(num_action_keys)
        seqlen = n_i + num_self_censor

        # Self masking
        for ts_row in range(0, clip_len):
            row = seqlen * ts_row + n_i - 1
            for ts_col in range(0, ts_row + 1):
                col = seqlen * ts_col + n_i
                episode_mask[
                    row : row + num_self_censor, col : col + num_self_censor
                ] = float("-inf")
                for i in range(num_self_censor):
                    episode_mask[row + i + 1, col + i] = 0

        return episode_mask

    @staticmethod
    def block_all_sensor_and_image_from_sensor_attention_mask(
        patch_row, patch_col, nun_metadata_keys, num_action_keys, clip_len, device
    ):
        episode_mask = Gato.causal_attention_mask(
            patch_row, patch_col, nun_metadata_keys, num_action_keys, clip_len, device
        )
        n_i = patch_row * patch_row
        num_self_censor = len(nun_metadata_keys) + 1 + len(num_action_keys)
        seqlen = n_i + num_self_censor

        # Self masking
        for ts_col in range(0, clip_len):
            col = seqlen * ts_col + n_i
            episode_mask[:, col : col + num_self_censor] = float("-inf")
            for ts_row in range(ts_col, clip_len):
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
        loss_categorical, pred, pred_values, tgt, tgt_shift, tgt_values = self._step(
            batch, is_training=True
        )
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
        loss_categorical, pred, pred_values, tgt, tgt_shift, tgt_values = self._step(
            batch, is_training=True
        )
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
        verbose: bool = False,
    ) -> Any:
        full_episode, *_, episode_mask = self._make_episode(batch, is_training=True)
        B, timesteps, *_ = mit.only(batch["frames"].values()).shape  # pyright: ignore
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
            _, to, d = history.shape
            for idx, key in enumerate(self.action_keys):
                output = self.gpt(
                    inputs_embeds=history,
                    episode_mask=episode_mask[
                        : m - len(self.action_keys) + idx,
                        : n - len(self.action_keys) + idx,
                    ],
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits = output["logits"]
                logits = logits.detach()
                token: Int[Tensor, "b 1"] = torch.argmax(
                    torch.softmax(logits[:, -1:, :], dim=-1), dim=-1
                )

                # embed prediction
                embedded: Float[Tensor, "b 1 e"] = self.sensor_embedding(token)
                b, t, _ = embedded.shape
                if self.hparams.have_position_encoding.action:  # type: ignore[union-attr]
                    position: Float[Tensor, "1 e"] = self.action_position(
                        torch.tensor([0], device=embedded.device)
                    )
                    embedded += repeat(position, "1 e -> b t e", b=b, t=t)
                if self.hparams.have_position_encoding.global_pos:  # type: ignore[union-attr]
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

                log_message = ""
                for b in range(B):
                    pred = prediction[b, 0].item()
                    gt = batch["meta"][key][b, ts].item()
                    predictions[b][ts][key]["pred"] = pred
                    predictions[b][ts][key]["gt"] = gt
                    log_message += f"[{batch_idx}][{key}] gt:{gt:.3f} pred:{pred:.3f}"
                if verbose:
                    logger.info(log_message)

        return predictions

    def forward(
        self,
        *,
        episode: Float[Tensor, "b to d"],
        episode_labels: Int[Tensor, "b to"],
        episode_mask: Float[Tensor, "d d"],
    ):
        output = self.gpt(
            inputs_embeds=episode,
            labels=episode_labels.to(torch.long),
            episode_mask=episode_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        # last layer features
        features = output["hidden_states"][-1]
        values = self.regressor(features)

        loss = output["loss"]
        logits = output["logits"]

        return loss, logits, values

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (lr_scheduler_cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(lr_scheduler_cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = {**lr_scheduler_cfg, **{"scheduler": scheduler}}

        return result

    def on_validation_epoch_end(self) -> None:
        self._finish_val_outputs_logging()
