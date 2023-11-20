from dataclasses import dataclass
from enum import Enum
from itertools import accumulate, pairwise
from typing import Iterable, Tuple

import more_itertools as mit
import pytorch_lightning as pl
import torch
from einops import pack, rearrange
from hydra.utils import instantiate
from jaxtyping import Float, Int
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from tensordict import TensorDict, dense_stack_tds, tensorclass
from tensordict.utils import NestedKey
from torch import Tensor
from torch.distributions import Categorical
from wandb import Image

from cargpt.utils._wandb import LoadableFromArtifact


class Token(str, Enum):
    IMAGE = "image"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class EpisodeConfig:
    observations: Tuple[Tuple[Token, str], ...]
    actions: Tuple[Tuple[Token, str], ...]

    def __init__(
        self,
        observations: Iterable[Iterable[str]],
        actions: Iterable[Iterable[str]],
    ):
        super().__init__()
        self.observations = tuple((Token(t), n) for (t, n) in observations)
        self.actions = tuple((Token(t), n) for (t, n) in actions)


@tensorclass  # pyright: ignore
class Episode:
    embeddings: Float[Tensor, "b s d"]
    labels: Int[Tensor, "b s"]
    index: TensorDict
    _keys: Tuple[NestedKey, ...]


class Gato(
    pl.LightningModule,
    LoadableFromArtifact,
):
    hparams: AttributeDict

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.transforms = instantiate(self.hparams.transforms)
        self.tokenizers = instantiate(self.hparams.tokenizers)
        self.detokenizers = instantiate(self.hparams.detokenizers)
        self.embeddings = instantiate(self.hparams.embeddings)
        self.position_encoding = instantiate(self.hparams.position_encoding)
        self.episode = instantiate(self.hparams.episode)
        self.encoder = instantiate(self.hparams.encoder)
        self.decoders = instantiate(self.hparams.decoders)
        self.loss = instantiate(self.hparams.loss)

        # TODO: tie continuous/discrete embeddings to corresponding decoders?

    def training_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict(
            {
                "/".join(["train", *k]): v
                for k, v in metrics.items(include_nested=True, leaves_only=True)
            }
        )

        return metrics["loss"]["total"]

    def validation_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict(
            {
                "/".join(["val", *k]): v
                for k, v in metrics.items(include_nested=True, leaves_only=True)
            }
        )

        return metrics["loss"]["total"]

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = cfg | {"scheduler": scheduler}

        return result

    def _step(self, batch: TensorDict) -> TensorDict:
        input = self._build_input(batch)
        episode = self._build_episode(input)

        if not hasattr(self, "attn_mask"):
            self.attn_mask = self._build_attention_mask(episode.index)

            if isinstance((logger := self.trainer.logger), WandbLogger):
                mask = self.attn_mask.clone()
                mask[mask == 0] = 1
                mask[mask == -torch.inf] = 0
                logger.experiment.log({"attention_mask": Image(mask)})

        encoder_output = self.encoder(src=episode.embeddings, mask=self.attn_mask)

        label_index = episode.index.select(*self.decoders.keys()).apply(
            lambda x: x[x > 0],
            batch_size=[],
        )

        labels = label_index.apply(
            lambda x: episode.labels.index_select(dim=1, index=x),
            batch_size=[episode.labels.shape[0]],
        )

        logit_index = label_index.apply(lambda x: x - 1)

        encoded = logit_index.apply(
            lambda x: encoder_output.index_select(dim=1, index=x),
            batch_size=[encoder_output.shape[0]],
        )

        logits = TensorDict.from_dict(
            {k: v.apply(self.decoders[k]) for k, v in encoded.items()},
            batch_size=encoded.batch_size,
            device=encoded.device,
        )

        loss = TensorDict.from_dict(
            {
                k: self.loss(
                    rearrange(logits[k], "b t d -> (b t) d"),
                    rearrange(labels[k], "b t -> (b t)"),
                )
                for k in logits.keys(include_nested=True, leaves_only=True)  # pyright: ignore
            }
        )

        loss["total"] = torch.stack(
            tuple(loss.values(include_nested=True, leaves_only=True))  # pyright: ignore
        ).mean()

        with torch.no_grad():
            diff = self._compute_diff(logits=logits, labels=labels)

        return TensorDict.from_dict({"loss": loss, "diff": diff})

    def _build_input(self, batch: TensorDict) -> TensorDict:
        meta = batch["meta"]
        input = TensorDict(
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
            names=["b"],
        )

        for t, transforms in self.transforms.items():
            for n, transform in transforms.items():
                input[(t, n)] = transform(input[(t, n)])

        return input

    def _build_episode(self, input: TensorDict) -> Episode:
        td_kwargs = {
            "batch_size": input.batch_size,
            "names": input.names,
            "device": input.device,
        }
        tokens = TensorDict(
            {
                (k := Token.CONTINUOUS): input[k].apply(self.tokenizers[k]),
                (k := Token.DISCRETE): input[k].apply(self.tokenizers[k]),
            },
            **td_kwargs,
        )

        embeddings = TensorDict(
            {
                (k := Token.IMAGE): input[k].apply(self.embeddings[k]),
                (k := Token.CONTINUOUS): tokens[k].apply(self.embeddings[k]),
                (k := Token.DISCRETE): tokens[k].apply(self.embeddings[k]),
            },
            **td_kwargs,
        )

        observation_embeddings, _ = pack(
            [embeddings[k] for k in self.episode.observations],
            "b t * d",
        )

        if pos_enc := getattr(self.position_encoding, "observations", None):
            _b, _t, s, _d = observation_embeddings.shape
            pos = torch.arange(s, device=observation_embeddings.device)
            observation_embeddings += pos_enc(pos)

        action_embeddings, _ = pack(
            [embeddings[k] for k in self.episode.actions],
            "b t * d",
        )

        if pos_enc := getattr(self.position_encoding, "actions", None):
            s = 1  # all actions share the same position encoding
            pos = torch.arange(s, device=action_embeddings.device)
            action_embeddings += pos_enc(pos)

        episode_embeddings, _ = pack(
            [observation_embeddings, action_embeddings],
            "b t * d",
        )

        _b, t, _s, _d = episode_embeddings.shape
        if pos_enc := getattr(self.position_encoding, "episode", None):
            pos = torch.arange(t, device=episode_embeddings.device)
            episode_embeddings += rearrange(pos_enc(pos), "t d -> t 1 d")

        labels = tokens.apply(lambda x: rearrange(x, "b t -> b t 1")).set(
            (k := Token.IMAGE),
            embeddings[k].apply(
                lambda x: torch.full(
                    x.shape[:-1],
                    self.loss.ignore_index,
                    device=embeddings.device,
                )
            ),
        )

        episode_keys = self.episode.observations + self.episode.actions
        episode_labels, episode_labels_ps = pack(
            [labels[k] for k in episode_keys],
            "b t *",
        )

        index_keys = episode_keys * t
        index_counts = tuple(map(mit.one, episode_labels_ps)) * t
        index_ranges = tuple(
            zip(
                index_keys,
                pairwise(accumulate(index_counts, initial=0)),
            )
        )
        episode_index = TensorDict(
            mit.map_reduce(
                index_ranges,
                keyfunc=lambda x: x[0],
                valuefunc=lambda x: torch.arange(*x[1]),
            ),
            batch_size=[t],
            names=["t"],
        )

        return Episode(  # pyright: ignore
            embeddings=rearrange(episode_embeddings, "b t s d -> b (t s) d"),
            labels=rearrange(episode_labels, "b t s -> b (t s)"),
            index=episode_index,
            _keys=episode_keys,
            batch_size=[],
            device=td_kwargs["device"],
        )

    @classmethod
    def _build_attention_mask(cls, index: TensorDict) -> Float[Tensor, "s s"]:
        DO_ATTEND = 0
        DO_NOT_ATTEND = -torch.inf

        max_idx = int(
            torch.tensor(
                tuple(
                    index.apply(torch.max, batch_size=[]).values(
                        include_nested=True, leaves_only=True
                    )
                ),
            )
            .max()
            .item()
        )

        mask = torch.full(
            (max_idx + 1, max_idx + 1),
            DO_NOT_ATTEND,
            device=index.device,
        )

        for step in range(mit.one(index.batch_size)):
            # non-image tokens attend to themselves from current and past steps
            for idxs in index.exclude("image").values(
                include_nested=True,
                leaves_only=True,
            ):
                mask[idxs[step:], idxs[step]] = DO_ATTEND

            # all tokens attend to all image tokens from current and past steps
            fut_all = torch.cat(
                tuple(index[step:].values(include_nested=True, leaves_only=True)),
                -1,
            ).flatten()

            cur_img = torch.cat(
                tuple(
                    index.select("image")[step].values(
                        include_nested=True, leaves_only=True
                    )
                ),
                -1,
            ).flatten()

            idxs = torch.meshgrid([fut_all.flatten(), cur_img.flatten()], indexing="ij")
            mask[idxs] = DO_ATTEND

        return mask

    def _compute_diff(
        self,
        *,
        logits: TensorDict,
        labels: TensorDict,
    ) -> TensorDict:
        preds = logits.apply(lambda x: Categorical(logits=x).sample())
        preds_labels = dense_stack_tds([preds, labels], dim=0)
        values = TensorDict.from_dict(
            {k: v.apply(self.detokenizers[k]) for k, v in preds_labels.items()},
            batch_size=preds_labels.batch_size,
        )

        return values.apply(
            lambda x: (x[0] - x[1]).abs().float().mean(),
            batch_size=[],
        )
