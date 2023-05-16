import more_itertools as mit
import pytorch_lightning as pl
import torch
from einops import repeat, rearrange
from hydra.utils import instantiate
from jaxtyping import Float, Int
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from torch.nn import ModuleDict
from torch.nn.functional import log_softmax, relu

from cargpt.utils.wandb import LoadableFromArtifact


class TransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    def __setstate__(self, state):
        # This is a workaround to a strange pytorch behaviour (bug?)
        if "activation" not in state or "activation" not in state.get("_modules", {}):
            state["activation"] = relu
        # Yup, skip the direct superclass and use its parent
        super(torch.nn.TransformerDecoderLayer, self).__setstate__(state)


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
            self.hparams.embeddings.discrete_cont_embedding
        )  # type: ignore[union-attr]
        self.local_position = instantiate(self.hparams.embeddings.local_position)  # type: ignore[union-attr]
        self.action_position = instantiate(self.hparams.embeddings.action_position)  # type: ignore[union-attr]

        # network
        self.transformer_decoder = instantiate(self.hparams.transformer_decoder)
        self.back_to_discrete = instantiate(self.hparams.back_to_discrete)

        self.predict_steps: int = self.hparams.predict_steps  # type: ignore[assignment]

    def training_step(self, batch, batch_idx):
        speed = self.unpack_batch_for_predictions(batch)

        speed_encoded = self.encodings.continues_values(speed)  # type: ignore[operator]

        # when more input data is added - concat vectors and pass as tensors encoded
        pred, tgt_discrete_actions = self.forward(
            tensors_encoded=speed_encoded[..., :-1],
            discrete_actions=speed_encoded[..., 1:],
        )

        # compute loss
        log_p = log_softmax(pred, dim=-1)
        # TODO: mask tokens when more than speed is available -> m(b,l) (2.3, eq 2)
        loss = -(
            torch.gather(
                input=log_p,
                dim=-1,
                index=tgt_discrete_actions[:, :, None].to(torch.int64),
            )
        ).sum()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def forward(
        self,
        *,
        tensors_encoded: Int[Tensor, "b t"],
        discrete_actions: Int[Tensor, "b t"],
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
        actions: Float[Tensor, "b t e"] = self.discrete_cont_embedding(discrete_actions)
        action_position_encoded: Float[Tensor, "1 e"] = self.action_position(
            torch.tensor([0], device=actions.device)
        )
        actions_encoded = actions + repeat(
            action_position_encoded, "1 e -> b (t 1) e", b=b, t=t
        )

        # Make a separator of a correct shape, it's always the same
        separator = torch.zeros_like(actions)

        # Make split point: the first step from tgt is always visible to the network
        # so the split is shifted by one
        split_point = t - (self.predict_steps + 1)

        # construct full sequence: [[o, |, a], [o, |, a], ...]
        # appendix B of the paper (https://arxiv.org/abs/2205.06175)
        full_sequence: Float[Tensor, "b L e"] = rearrange(
            [
                observations_embeddings[:, :split_point],
                separator[:, :split_point],
                actions_encoded[:, :split_point],
            ],
            "x b t e -> b (t x) e",
        )  # this is like python's zip on the 1st dim

        # pass through decoder
        tgt_seq = actions_encoded[:, split_point:, ...]
        _, m, _ = tgt_seq.shape
        tgt_mask = torch.tril(torch.ones(m, m, device=tgt_seq.device)).float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0.0, -torch.inf)

        pred = self.transformer_decoder(
            tgt=tgt_seq,
            memory=full_sequence,
            tgt_mask=tgt_mask,
        )
        pred = self.back_to_discrete(pred)
        return pred, discrete_actions[:, split_point:]

    def unpack_batch_for_predictions(self, batch) -> Float[Tensor, "b t"]:
        clips = mit.one(batch["clips"].values())
        speed = clips["meta"]["VehicleMotion_speed"].to(self.device)

        return speed

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (lr_scheduler_cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(lr_scheduler_cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = {**lr_scheduler_cfg, **{"scheduler": scheduler}}

        return result
