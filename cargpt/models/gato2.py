from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import accumulate, pairwise, starmap
from typing import Iterable, Tuple

import more_itertools as mit
import pytorch_lightning as pl
import torch
from cachetools import cached
from einops import pack, rearrange
from einops.layers.torch import Rearrange
from hydra.utils import instantiate
from jaxtyping import Float
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from tensordict import LazyStackedTensorDict, TensorDict, tensorclass
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
    embeddings: TensorDict
    labels: TensorDict
    index: TensorDict
    _keys: Tuple[NestedKey, ...]

    @cached_property
    def packed_embeddings(self):
        packed, _ = pack([self.embeddings[k] for k in self._keys], "b t * d")

        return packed


def _hash_tensordict(index: TensorDict) -> int:
    keys = sorted(index.keys(True, True))  # pyright: ignore
    items = tuple((k, tuple(index[k].flatten().tolist())) for k in keys)

    return hash(items)


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
            {"/".join(["train", *k]): v for k, v in metrics.items(True, True)}
        )

        return metrics["loss"]["total"]

    def validation_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({"/".join(["val", *k]): v for k, v in metrics.items(True, True)})

        return metrics["loss"]["total"]

    def predict_step(self, batch: TensorDict, _batch_idx: int) -> LazyStackedTensorDict:
        input = self._build_input(batch)
        episode = self._build_episode(input)
        attn_mask = self._build_attention_mask(episode.index)

        embeddings = rearrange(episode.packed_embeddings, "b t s d -> b (t s) d")
        _, seq_len, _ = embeddings.shape

        actions = self.episode.actions
        seq_prefix_len = seq_len - len(actions)
        if (expected := list(range(seq_prefix_len, seq_len))) != (
            actual := [episode.index[k][-1].item() for k in actions]
        ):
            msg = f"invalid action indices (expected: {expected}, actual: {actual})"
            raise RuntimeError(msg)

        prediction = TensorDict({}, batch_size=batch.batch_size, device=batch.device)

        for t, n in actions:
            encoder_output = self.encoder(
                src=embeddings[:, :seq_prefix_len],
                mask=attn_mask[:seq_prefix_len, :seq_prefix_len],
            )
            encoded = encoder_output[:, -1]
            logit = self.decoders[t](encoded)
            pred_token = logit.argmax(dim=-1)
            prediction[t, n] = self.detokenizers[t](pred_token)

            pred_emb = self.embeddings[t](pred_token)

            if pos_enc := getattr(self.position_encoding, "actions", None):
                pos = torch.arange(pos_enc.num_embeddings, device=pred_emb.device)
                pred_emb += pos_enc(pos)

            if pos_enc := getattr(self.position_encoding, "episode", None):
                pos = torch.arange(pos_enc.num_embeddings, device=pred_emb.device)
                pred_emb += pos_enc(pos[-1])

            embeddings[:, seq_prefix_len] = pred_emb
            seq_prefix_len += 1

        return torch.stack([input, prediction], dim=1)  # pyright: ignore

    def _step(self, batch: TensorDict) -> TensorDict:
        input = self._build_input(batch)
        episode = self._build_episode(input)

        attn_mask = self._build_attention_mask(episode.index)
        if self.trainer.global_step == 0 and isinstance(self.logger, WandbLogger):
            attn_mask_logged = torch.empty_like(attn_mask)
            attn_mask_logged[attn_mask == 0] = 1
            attn_mask_logged[attn_mask == -torch.inf] = 0
            self.logger.log_image("attention_mask", [Image(attn_mask_logged)], step=0)

        embeddings = rearrange(episode.packed_embeddings, "b t s d -> b (t s) d")
        encoder_output = self.encoder(src=embeddings, mask=attn_mask)

        label_index = episode.index.select(*self.decoders.keys())
        logit_index = label_index.apply(lambda idx: idx[idx > 0] - 1, batch_size=[])
        encoded = logit_index.apply(
            lambda idx: encoder_output[:, idx.flatten(), :],
            batch_size=[encoder_output.shape[0]],
        )

        logits = TensorDict.from_dict(
            {k: v.apply(self.decoders[k]) for k, v in encoded.items()},
            batch_size=encoded.batch_size,
            device=encoded.device,
        )

        labels = episode.labels.select(*self.decoders.keys()).apply(
            lambda lbl, idx: lbl[:, idx > 0],
            label_index,
        )

        loss = TensorDict.from_dict(
            {
                k: self.loss(
                    rearrange(logits[k], "b t d -> (b t) d"),
                    rearrange(labels[k], "b t -> (b t)"),
                )
                for k in logits.keys(True, True)  # pyright: ignore
            }
        )

        loss["total"] = torch.stack(tuple(loss.values(True, True))).mean()  # pyright: ignore

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

    def _build_input(self, batch: TensorDict) -> TensorDict:
        meta = batch["meta"]
        input = TensorDict.from_dict(
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

        for t, transforms in self.transforms.items():
            for n, transform in transforms.items():
                input[(t, n)] = transform(input[(t, n)])

        return input

    def _build_episode(self, input: TensorDict) -> Episode:
        tokens = TensorDict(
            {
                (k := Token.CONTINUOUS): input[k].apply(self.tokenizers[k]),
                (k := Token.DISCRETE): input[k].apply(self.tokenizers[k]),
            },
            batch_size=input.batch_size,
            device=input.device,
        ).apply(Rearrange("b t -> b t 1"))

        embeddings = TensorDict(
            {
                (k := Token.IMAGE): input[k].apply(self.embeddings[k]),
                (k := Token.CONTINUOUS): tokens[k].apply(self.embeddings[k]),
                (k := Token.DISCRETE): tokens[k].apply(self.embeddings[k]),
            },
            batch_size=input.batch_size,
            device=input.device,
        )

        step_keys = self.episode.observations + self.episode.actions
        step_index_counts = [embeddings.get_item_shape(k)[2] for k in step_keys]
        step_index_ranges = pairwise(accumulate(step_index_counts, initial=0))
        step_index = TensorDict(
            dict(zip(step_keys, starmap(torch.arange, step_index_ranges))),
            batch_size=[],
            device=embeddings.device,
        )
        step_len = sum(step_index_counts)
        step_count = mit.one({embeddings.get_item_shape(k)[1] for k in step_keys})
        index = step_index.apply(
            lambda idx: torch.stack([idx + i * step_len for i in range(step_count)]),
            batch_size=[step_count],
        )

        if pos_enc := getattr(self.position_encoding, "observations", None):
            embeddings.select(*self.episode.observations).apply(
                lambda emb, pos: emb + pos_enc(pos),  # pyright: ignore
                step_index,
                inplace=True,
            )

        if pos_enc := getattr(self.position_encoding, "actions", None):
            pos = torch.arange(pos_enc.num_embeddings, device=embeddings.device)
            pos_encd = pos_enc(pos)
            embeddings.select(*self.episode.actions).apply(
                lambda emb: emb + pos_encd,
                inplace=True,
            )

        # TODO: rename to timestep?
        if pos_enc := getattr(self.position_encoding, "episode", None):
            pos = torch.arange(pos_enc.num_embeddings, device=embeddings.device)
            pos_encd = rearrange(pos_enc(pos), "t d -> t 1 d")
            embeddings.apply(
                lambda emb: emb + pos_encd,
                inplace=True,
            )

        return Episode(  # pyright: ignore
            embeddings=embeddings,
            labels=tokens,
            index=index,
            _keys=step_keys,
            batch_size=[],
            device=input.device,
        )

    @staticmethod
    @cached(cache={}, key=_hash_tensordict)
    def _build_attention_mask(index: TensorDict) -> Float[Tensor, "s s"]:
        DO_ATTEND = 0
        DO_NOT_ATTEND = -torch.inf

        idx_max = max(index.apply(torch.max, batch_size=[]).values(True, True)).item()

        mask = torch.full(
            (idx_max + 1, idx_max + 1),
            DO_NOT_ATTEND,
            device=index.device,
        )

        for step in range(mit.one(index.batch_size)):
            # non-image tokens attend to themselves
            for idxs in index.exclude("image").values(True, True):
                mask[idxs[step], idxs[step]] = DO_ATTEND

            # all tokens attend to all image tokens from current and past steps
            fut_all = torch.cat(tuple(index[step:].values(True, True)), -1).flatten()

            cur_img = torch.cat(
                tuple(index.select("image")[step].values(True, True)), -1
            ).flatten()

            idxs = torch.meshgrid([fut_all.flatten(), cur_img.flatten()], indexing="ij")
            mask[idxs] = DO_ATTEND

        return mask

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = cfg | {"scheduler": scheduler}

        return result
