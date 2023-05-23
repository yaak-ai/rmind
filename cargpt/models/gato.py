from typing import Optional, Tuple

import more_itertools as mit
import pytorch_lightning as pl
import torch
import wandb
from einops import repeat, rearrange
from hydra.utils import instantiate
from jaxtyping import Float, Int
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from torch.nn import Linear, ModuleDict
from torch.nn.functional import gelu, log_softmax, relu

from cargpt.utils.wandb import LoadableFromArtifact


class TransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    def __setstate__(self, state):
        # This is a workaround to a strange pytorch behaviour (bug?)
        if "activation" not in state or "activation" not in state.get("_modules", {}):
            state["activation"] = relu
        # Yup, skip the direct superclass and use its parent
        super(torch.nn.TransformerDecoderLayer, self).__setstate__(state)


class TransformerDecoderLayerGEGLU(TransformerDecoderLayer):
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
        return self.dropout3(x)


class Gato(pl.LightningModule, LoadableFromArtifact):
    """A Generalist Agent (Gato) https://arxiv.org/abs/2205.06175"""

    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encodings: ModuleDict = instantiate(self.hparams.encodings)

        # for embeddings
        self.discrete_embedding = instantiate(
            self.hparams.embeddings.discrete_embedding  # type: ignore[union-attr]
        )
        self.local_position = instantiate(self.hparams.embeddings.local_position)  # type: ignore[union-attr]
        self.action_position = instantiate(self.hparams.embeddings.action_position)  # type: ignore[union-attr]

        # network
        self.transformer_decoder = instantiate(self.hparams.transformer_decoder)
        self.back_to_discrete = instantiate(self.hparams.back_to_discrete)

        # Special tokens
        self.separator_idx = 0
        self.start_token_idx = 1
        self.end_token_idx = 2

    def _step(
        self,
        speed,
        gas_pedal_normalized,
    ):
        speed_encoded: Int[Tensor, "b ts"] = self.encodings.continues_values(speed)  # type: ignore[operator]
        gas_pedal_normalized_encoded: Int[Tensor, "b ts"] = self.encodings.continues_values(  # type: ignore[operator]
            gas_pedal_normalized
        )

        # when more input data is added - concat vectors and pass as tensors encoded
        tensors_encoded: Int[Tensor, "b ts n"] = rearrange(
            speed_encoded[..., :-1], "b (ts 1) -> b ts 1"
        )
        discrete_actions_encoded: Int[Tensor, "b ts A"] = rearrange(
            gas_pedal_normalized_encoded[..., 1:], "b (ts 1) -> b ts 1"
        )

        pred = self.forward(
            tensors_encoded=tensors_encoded,
            discrete_actions_encoded=discrete_actions_encoded,
        )

        b, _, _ = pred.shape
        special_token = torch.ones(size=(b, 1)).to(tensors_encoded)
        ground_truth: Int[Tensor, "b l"] = torch.cat(
            [
                special_token * self.start_token_idx,
                tensors_encoded[:, -1],
                special_token * self.separator_idx,
                discrete_actions_encoded[:, -1],
                special_token * self.end_token_idx,
            ],
            dim=1,
        )
        ground_truth_mask: Int[Tensor, "b l"] = torch.ones_like(ground_truth)
        # The only thing we don't want to predict right now is separator
        # TODO: let it be configurabe from cfg if possible
        separator_position = tensors_encoded[:, -1].shape[1] + 1
        ground_truth_mask[:, separator_position] = 0

        return pred, ground_truth, ground_truth_mask

    def _compute_loss(self, predictions, ground_truth, ground_truth_mask):
        log_p = log_softmax(predictions, dim=-1)
        loss = -(
            torch.gather(
                input=log_p,
                dim=-1,
                index=ground_truth[:, :, None].to(torch.int64),
            ).squeeze(-1)
            * ground_truth_mask
        ).sum()
        return loss

    def training_step(self, batch, batch_idx):
        speed, gas_pedal_normalized = self.unpack_batch_for_predictions(batch)
        pred, ground_truth, ground_truth_mask = self._step(speed, gas_pedal_normalized)
        loss = self._compute_loss(pred, ground_truth, ground_truth_mask)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=speed.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        speed, gas_pedal_normalized = self.unpack_batch_for_predictions(batch)
        pred, ground_truth, ground_truth_mask = self._step(speed, gas_pedal_normalized)
        loss = self._compute_loss(pred, ground_truth, ground_truth_mask)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=speed.shape[0],
        )

        if hasattr(self, "_tables"):
            inverted: Float[Tensor, "b t"] = self.invert_predictions(pred)
            # TODO: figure out a more generic solution
            gas_pedal_normalized_pred: Float[Tensor, "b"] = inverted[:, 3]
            gas_pedal_normalized_gt: Float[Tensor, "b"] = gas_pedal_normalized[:, -1]
            speed_pred: Float[Tensor, "b"] = inverted[:, 1]
            speed_gt: Float[Tensor, "b"] = speed[:, -1]
            if table := self._tables.get("outputs"):
                data: Float[Tensor, "b tx"] = rearrange(
                    [  # type: ignore
                        gas_pedal_normalized_pred,
                        gas_pedal_normalized_gt,
                        speed_pred,
                        speed_gt,
                    ],
                    "x b -> b x",
                )

                for row in data.tolist():
                    table.add_data(*row)

        return loss

    def invert_predictions(
        self, predictions: Float[Tensor, "b t e"]
    ) -> Float[Tensor, "b t"]:
        discrete: Int[Tensor, "b t"] = torch.argmax(predictions, dim=-1)
        inverted = discrete.clone()
        for encoding in reversed(self.encodings.continues_values):  # type: ignore[arg-type]
            inverted = encoding.invert(inverted)
        return inverted

    def forward(
        self,
        *,
        tensors_encoded: Int[Tensor, "b ts 1"],
        discrete_actions_encoded: Int[Tensor, "b ts 1"],
    ):
        b, ts, _ = tensors_encoded.shape

        # encodings to embeddings - learnable!
        tensors_embeddings: Float[Tensor, "b ts 1 e"] = self.discrete_embedding(
            tensors_encoded
        )

        # construct observations and add local positional embedding
        observations = tensors_embeddings
        _, _, observations_tokens, _ = observations.shape
        positions = torch.arange(
            0, observations_tokens, dtype=torch.int, device=observations.device
        )
        positions = self.local_position(positions)
        positions_encoded = repeat(positions, "tokens e -> b ts tokens e", b=b, ts=ts)
        observations_embeddings = observations + positions_encoded

        # Prepare actions, use the same position always
        actions_embeddings: Float[Tensor, "b ts 1 e"] = self.discrete_embedding(
            discrete_actions_encoded
        )
        action_position_encoded: Float[Tensor, "1 e"] = self.action_position(
            torch.tensor([0], device=actions_embeddings.device)
        )
        _, _, actions_tokens, _ = actions_embeddings.shape
        actions_embeddings = actions_embeddings + repeat(
            action_position_encoded,
            "1 e -> b ts (tokens 1) e",
            b=b,
            ts=ts,
            tokens=actions_tokens,
        )
        # Get start and end token plus separator
        start_token, end_token, separator = self.discrete_embedding(
            torch.tensor(
                [self.start_token_idx, self.end_token_idx, self.separator_idx],
                device=actions_embeddings.device,
            )
        ).chunk(3, dim=0)

        # construct full sequence: [[o, |, a], [o, |, a], ...]
        # appendix B of the paper (https://arxiv.org/abs/2205.06175)
        full_sequence: Float[Tensor, "b ts L e"] = torch.concat(  # type: ignore[assignment]
            [
                observations_embeddings,
                repeat(separator, "1 e -> b ts 1 e", b=b, ts=ts),
                actions_embeddings,
            ],
            dim=2,  # concat tokens on each timestamp
        )

        # The last timestep s_l = [o_l | a_l] from full sequence is the one the decoder
        # should predict
        tgt_seq: Float[Tensor, "b L e"] = full_sequence[:, -1:].squeeze(1)
        tgt_seq = torch.concat(
            [
                repeat(start_token, "1 e -> b 1 e", b=b),
                tgt_seq,
                repeat(end_token, "1 e -> b 1 e", b=b),
            ],
            dim=1,
        )
        _, seq_tokens, _ = tgt_seq.shape
        tgt_mask = torch.tril(
            torch.ones(seq_tokens, seq_tokens, device=tgt_seq.device)
        ).float()
        # mask out separator which is just after observations
        separator_idx = observations_tokens + 1
        tgt_mask[separator_idx, separator_idx] = 0.0
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0.0, -torch.inf)

        pred = self.transformer_decoder(
            tgt=tgt_seq,
            memory=rearrange(full_sequence[:, :-1], "b ts L e -> b (ts L) e"),
            tgt_mask=tgt_mask,
        )
        pred = self.back_to_discrete(pred)
        return pred

    def unpack_batch_for_predictions(
        self, batch
    ) -> Tuple[Float[Tensor, "b t"], Float[Tensor, "b t"]]:
        clips = mit.one(batch["clips"].values())
        meta = clips["meta"]

        speed = meta["VehicleMotion_speed"].to(self.device)

        gas_pedal_normalized = meta["VehicleMotion_gas_pedal_normalized"].to(
            self.device
        )
        return (
            speed,
            gas_pedal_normalized,
        )

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (lr_scheduler_cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(lr_scheduler_cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = {**lr_scheduler_cfg, **{"scheduler": scheduler}}

        return result

    def on_validation_epoch_start(self):
        if (
            isinstance(logger := self.logger, pl.loggers.WandbLogger)
            and isinstance(logger.experiment, wandb.wandb_run.Run)
            and (log_cfg := self.hparams.get("log", {}).get("validation", {}))
        ):
            self._tables = {}

            if log_cfg.get(name := "outputs"):
                columns = [
                    "gas_pedal_normalized_pred",
                    "gas_pedal_normalized_gt",
                    "speed_pred",
                    "speed_gt",
                ]
                self._tables[name] = wandb.Table(columns=columns)

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
