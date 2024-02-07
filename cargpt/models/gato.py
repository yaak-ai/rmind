from functools import lru_cache

import more_itertools as mit
import pytorch_lightning as pl
import torch
from einops import pack, rearrange
from einops.layers.torch import Rearrange
from hydra.utils import instantiate
from jaxtyping import Float
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from tensordict import LazyStackedTensorDict, TensorDict
from torch import Tensor
from torch.distributions import Categorical

from cargpt.components.episode import EpisodeBuilder, Index, TokenType
from cargpt.utils._wandb import LoadableFromArtifact


class Gato(pl.LightningModule, LoadableFromArtifact):
    hparams: AttributeDict

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.episode_builder: EpisodeBuilder = instantiate(self.hparams.episode_builder)
        self.detokenizers = instantiate(self.hparams.detokenizers)
        self.encoder = instantiate(self.hparams.encoder)
        self.decoders = instantiate(self.hparams.decoders)
        self.loss = instantiate(self.hparams.loss)

        for k, enabled in self.hparams.weight_tying.items():  # pyright: ignore
            if enabled:
                self.episode_builder.embeddings[k].weight = self.decoders[k].weight

    def training_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["train", *k]): v for k, v in metrics.items(True, True)
        })

        return metrics["loss", "total"]

    def validation_step(self, batch: TensorDict, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({"/".join(["val", *k]): v for k, v in metrics.items(True, True)})

        return metrics["loss", "total"]

    def predict_step(self, batch: TensorDict, _batch_idx: int) -> LazyStackedTensorDict:
        input = self._build_input(batch)
        episode = self.episode_builder.build_episode(input)
        attn_mask = self._build_attention_mask(episode.index)

        embeddings, _ = pack(
            [episode.embeddings[k] for k in episode.timestep.keys],
            "b t * d",
        )
        embeddings = rearrange(embeddings, "b t s d -> b (t s) d")
        _, seq_len, _ = embeddings.shape

        actions = self.episode_builder.timestep.actions
        seq_prefix_len = seq_len - len(actions)
        if (expected := list(range(seq_prefix_len, seq_len))) != (
            actual := [episode.index.get(k)[-1].item() for k in actions]  # pyright: ignore
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

            pred_emb = self.episode_builder.embeddings[t](pred_token)

            if pos_enc := getattr(
                self.episode_builder.position_encoding, "actions", None
            ):
                pos = torch.arange(pos_enc.num_embeddings, device=pred_emb.device)
                pred_emb += pos_enc(pos)

            if pos_enc := getattr(
                self.episode_builder.position_encoding, "timestep", None
            ):
                pos = torch.arange(pos_enc.num_embeddings, device=pred_emb.device)
                pred_emb += pos_enc(pos[-1])

            embeddings[:, seq_prefix_len] = pred_emb
            seq_prefix_len += 1

        return torch.stack([input, prediction], dim=1)  # pyright: ignore

    def _step(self, batch: TensorDict) -> TensorDict:
        input = self._build_input(batch)
        episode = self.episode_builder.build_episode(input)
        attn_mask = self._build_attention_mask(episode.index)

        if self.trainer.global_step == 0 and isinstance(self.logger, WandbLogger):
            attn_mask_logged = torch.empty_like(attn_mask)
            attn_mask_logged[attn_mask == 0] = 1
            attn_mask_logged[attn_mask == -torch.inf] = 0

            from wandb import Image  # noqa: PLC0415

            self.logger.log_image("attention_mask", [Image(attn_mask_logged)], step=0)

        embeddings, _ = pack(
            [episode.embeddings[k] for k in episode.timestep.keys],
            "b t * d",
        )
        embeddings = rearrange(embeddings, "b t s d -> b (t s) d")
        encoder_output = self.encoder(src=embeddings, mask=attn_mask)

        label_index = episode.index.select(*self.decoders.keys())._tensordict  # pyright: ignore
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

        loss = logits.apply(Rearrange("b t d -> (b t) d"), batch_size=[]).apply(
            self.loss,
            labels.apply(Rearrange("b t -> (b t)"), batch_size=[]),
        )

        loss = loss.to_tensordict().set(
            "total", torch.stack(tuple(loss.values(True, True))).mean()
        )

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
        return TensorDict.from_dict(
            {
                TokenType.IMAGE: batch["frames"],
                TokenType.CONTINUOUS: {
                    "speed": meta["VehicleMotion_speed"],
                    "pedal": (
                        meta["VehicleMotion_gas_pedal_normalized"]
                        - meta["VehicleMotion_brake_pedal_normalized"]
                    ),
                    "steering_angle": meta["VehicleMotion_steering_angle_normalized"],
                },
                TokenType.DISCRETE: {
                    "turn_signal": meta["VehicleState_turn_signal"],
                },
            },
            batch_size=batch.batch_size,
            device=batch.device,
        )

    @staticmethod
    @lru_cache(maxsize=None, typed=True)
    def _build_attention_mask(index: Index) -> Float[Tensor, "s s"]:
        DO_ATTEND = 0
        DO_NOT_ATTEND = -torch.inf

        mask = torch.full(
            (index.max + 1, index.max + 1),
            DO_NOT_ATTEND,
            device=index.device,  # pyright: ignore
        )

        index_td = index._tensordict  # pyright: ignore

        for step in range(mit.one(index_td.batch_size)):
            # non-image tokens attend to themselves
            for idxs in index_td.exclude(TokenType.IMAGE).values(True, True):
                mask[idxs[step], idxs[step]] = DO_ATTEND

            # all tokens attend to all image tokens from current and past steps
            fut_all = torch.cat(tuple(index_td[step:].values(True, True)), -1).flatten()

            cur_img = torch.cat(
                tuple(index_td.select(TokenType.IMAGE)[step].values(True, True)), -1
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
