"""Zero: a ControlTransformer with a shared world-model latent stage.

Pipeline: ``episode -> encoder -> latent -> objectives``. After the encoder, a ``latent``
module produces a shared per-patch latent once; it is then passed (``latent=``) to every
objective's ``compute_metrics``/``predict``/``forward``. All of a Zero model's objectives are
latent-aware (they accept a ``latent`` kwarg).
"""

from typing import Any, override

from hydra.utils import instantiate
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tensordict import TensorDict
from torch.nn import Module

from rmind.components.base import TensorTree
from rmind.models.control_transformer import (
    INTERNAL_STEP_OUTPUT_KEY,
    ControlTransformer,
)


class Zero(ControlTransformer):
    latent: Module

    def __init__(self, *, latent: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # `latent` arrives as a raw config (the model is instantiated with _recursive_: false)
        self.hparams["latent"] = latent
        self.latent = instantiate(latent)

    @override
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        if (
            hasattr(self.episode_builder, "token_mask")
            and self.episode_builder.token_mask is not None
        ):
            self.episode_builder.token_mask.set_epoch(self.current_epoch)

        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )
        latent = self.latent(episode=episode, embedding=embedding)

        metrics = TensorDict({
            name: objective.compute_metrics(
                episode=episode,
                embedding=embedding,
                latent=latent,
            )
            for name, objective in self.objectives.items()
        })

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
        loss_total = losses.sum(reduce=True)
        metrics["loss", "total"] = loss_total

        self.log_dict(
            {
                "/".join(["train", *k]): v
                for k, v in metrics.detach().items(
                    include_nested=True, leaves_only=True
                )
                if not any(part.startswith("_") for part in k)
            },
            sync_dist=True,
        )

        outputs = {"loss": metrics["loss", "total"]} | metrics.select(
            *(
                (obj_name, "_artifacts")
                for obj_name, metric in metrics.items()
                if "_artifacts" in metric
            )
        ).to_dict()

        if self.current_epoch == 0 and batch_idx == 0:
            outputs[INTERNAL_STEP_OUTPUT_KEY] = {"episode": episode.detach()}

        return outputs

    @override
    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )
        latent = self.latent(episode=episode, embedding=embedding)

        metrics = TensorDict({
            name: objective.compute_metrics(
                episode=episode,
                embedding=embedding,
                latent=latent,
            )
            for name, objective in self.objectives.items()
        })

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
        loss_total = losses.sum(reduce=True)
        metrics["loss", "total"] = loss_total

        if not self.trainer.sanity_checking:
            self.log_dict(
                {
                    "/".join(["val", *k]): v
                    for k, v in metrics.items(include_nested=True, leaves_only=True)
                    if not any(part.startswith("_") for part in k)
                },
                sync_dist=True,
            )

        return {"loss": metrics["loss", "total"]} | metrics.select(
            *(
                (obj_name, "_artifacts")
                for obj_name, metric in metrics.items()
                if "_artifacts" in metric
            )
        ).to_dict()

    @override
    def predict_step(self, batch: dict[str, Any]) -> TensorDict:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )
        latent = self.latent(episode=episode, embedding=embedding)

        objectives_predictions = {
            name: objective.predict(
                episode=episode,
                embedding=embedding,
                keys=frozenset(self.prediction_config.objectives),
                tokenizers=self.episode_builder.tokenizers,
                latent=latent,
            )
            for name, objective in self.objectives.items()
        }
        return TensorDict(objectives_predictions).auto_batch_size_(1)

    @override
    def forward(self, batch: TensorTree) -> TensorTree | TensorDict:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )
        latent = self.latent(episode=episode, embedding=embedding)

        outputs = {
            name: objective(
                episode=episode,
                embedding=embedding,
                latent=latent,
            )
            for name, objective in self.objectives.items()
        }

        return TensorDict(outputs)
