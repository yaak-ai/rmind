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
        self.separator = torch.tensor([0.0])

        # for embeddings
        self.discrete_cont_embedding = instantiate(
            self.hparams.embeddings.discrete_cont_embedding  # type: ignore[union-attr]
        )
        self.local_position = instantiate(self.hparams.embeddings.local_position)  # type: ignore[union-attr]
        self.action_position = instantiate(self.hparams.embeddings.action_position)  # type: ignore[union-attr]

        # network
        self.transformer_decoder = instantiate(self.hparams.transformer_decoder)
        self.back_to_discrete = instantiate(self.hparams.back_to_discrete)

        self.predict_steps: int = self.hparams.predict_steps  # type: ignore[assignment]

    def _step(
        self,
        speed,
        gas_pedal_normalized,
    ):
        speed_encoded = self.encodings.continues_values(speed)  # type: ignore[operator]
        gas_pedal_normalized_encoded = self.encodings.continues_values(  # type: ignore[operator]
            gas_pedal_normalized
        )

        # when more input data is added - concat vectors and pass as tensors encoded
        pred, tgt_discrete_actions = self.forward(
            tensors_encoded=speed_encoded[..., :-1],
            discrete_actions_encoded=gas_pedal_normalized_encoded[..., 1:],
        )

        return pred, tgt_discrete_actions

    def _compute_loss(self, predictions, tgt_discrete_actions):
        log_p = log_softmax(predictions, dim=-1)
        # TODO: mask tokens when more than speed is available -> m(b,l) (2.3, eq 2)
        loss = -(
            torch.gather(
                input=log_p,
                dim=-1,
                index=tgt_discrete_actions[:, :, None].to(torch.int64),
            )
        ).sum()
        return loss

    def training_step(self, batch, batch_idx):
        speed, gas_pedal_normalized = self.unpack_batch_for_predictions(batch)
        pred, tgt_discrete_actions = self._step(speed, gas_pedal_normalized)
        loss = self._compute_loss(pred, tgt_discrete_actions)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        speed, gas_pedal_normalized = self.unpack_batch_for_predictions(batch)
        pred, tgt_discrete_actions = self._step(speed, gas_pedal_normalized)
        loss = self._compute_loss(pred, tgt_discrete_actions)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        if hasattr(self, "_tables"):
            speed_predicted: Float[Tensor, "b t"] = self.invert_predictions(pred)
            speed_gt: Float[Tensor, "b t"] = speed[..., -speed_predicted.shape[-1] :]
            if table := self._tables.get("outputs"):
                data: Float[Tensor, "b tx"] = rearrange(
                    [  # type: ignore
                        speed_predicted,
                        speed_gt,
                    ],
                    "x b t -> b (t x)",
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
        tensors_encoded: Int[Tensor, "b t"],
        discrete_actions_encoded: Int[Tensor, "b t"],
    ):
        # encodings to embeddings - learnable!
        tensors_embeddings: Float[Tensor, "b t e"] = self.discrete_cont_embedding(
            tensors_encoded
        )
        b, t, e = tensors_embeddings.shape

        # construct observations and add local positional embedding
        observations = tensors_embeddings
        positions = torch.range(0, t - 1, dtype=torch.int, device=observations.device)
        positions_encoded = repeat(self.local_position(positions), "t e -> b t e", b=b)
        observations_embeddings = observations + positions_encoded

        # Prepare actions, use the same position always
        actions_embeddings: Float[Tensor, "b t e"] = self.discrete_cont_embedding(
            discrete_actions_encoded
        )
        action_position_encoded: Float[Tensor, "1 e"] = self.action_position(
            torch.tensor([0], device=actions_embeddings.device)
        )
        actions_embeddings = actions_embeddings + repeat(
            action_position_encoded, "1 e -> b (t 1) e", b=b, t=t
        )

        # Make a separator of a correct shape, it's always the same
        separator = torch.zeros_like(actions_embeddings)

        # Make split point: the first step from tgt is always visible to the network
        # so the split is shifted by one
        split_point = t - (self.predict_steps + 1)

        # construct full sequence: [[o, |, a], [o, |, a], ...]
        # appendix B of the paper (https://arxiv.org/abs/2205.06175)
        full_sequence: Float[Tensor, "b L e"] = rearrange(  # type: ignore[assignment]
            [
                observations_embeddings[:, :split_point],
                separator[:, :split_point],
                actions_embeddings[:, :split_point],
            ],
            "x b t e -> b (t x) e",
        )  # this is like python's zip on the 1st dim

        # pass through decoder
        tgt_seq = actions_embeddings[:, split_point:, ...]
        _, m, _ = tgt_seq.shape
        tgt_mask = torch.tril(torch.ones(m, m, device=tgt_seq.device)).float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0.0, -torch.inf)

        pred = self.transformer_decoder(
            tgt=tgt_seq,
            memory=full_sequence,
            tgt_mask=tgt_mask,
        )
        pred = self.back_to_discrete(pred)

        # Reverse shift from split point
        pred = pred[:, :-1, :]
        discrete_actions_encoded = discrete_actions_encoded[:, (split_point + 1) :]
        return pred, discrete_actions_encoded

    def unpack_batch_for_predictions(
        self, batch
    ) -> Tuple[Float[Tensor, "b t"], Float[Tensor, "b t"],]:
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
                    f"{col}_{i}"
                    for i in range(self.predict_steps)
                    for col in ("speed_pred", "speed_gt")
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
