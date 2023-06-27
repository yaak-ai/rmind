from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import more_itertools as mit
import numpy as np
import torch.nn
import wandb
from einops import rearrange, repeat
from jaxtyping import Float
from PIL.Image import Image, fromarray
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import Tensor, softmax
from wandb.sdk.interface.artifacts import Artifact
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from cargpt.visualization import Unnormalize


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(cls, name: str, **kwargs):
        get_artifact: Callable[..., Artifact] = (
            wandb.run.use_artifact
            if wandb.run is not None and not isinstance(wandb.run, RunDisabled)
            else wandb.Api().artifact
        )

        artifact = get_artifact(name, type="model")
        artifact_dir = artifact.download()
        ckpt_path = mit.one(Path(artifact_dir).glob("*.ckpt"))

        return cls.load_from_checkpoint(ckpt_path.as_posix(), **kwargs)  # type: ignore


class ValOutputsLoggingTableMixin:
    trainer: Trainer
    logger: WandbLogger
    hparams: AttributeDict

    @property
    def val_table_main_columns(self):
        return getattr(self, "_val_table_main_columns")

    @val_table_main_columns.setter
    def val_table_main_columns(self, columns: List[str]):
        setattr(self, "_val_table_main_columns", list(columns))

    @property
    def val_table(self):
        return getattr(self, "_val_table", None)

    @val_table.setter
    def val_table(self, wandb_table: wandb.Table):
        setattr(self, "_val_table", wandb_table)

    @val_table.deleter
    def val_table(self):
        delattr(self, "_val_table")

    def is_outputs_logging_active(self):
        return (
            isinstance(logger := self.logger, WandbLogger)
            and isinstance(logger.experiment, Run)
            and self.hparams.get("log", {}).get("validation", {}).get("outputs")
            and self.trainer.state.stage != "sanity_check"
        )

    def _init_val_outputs_logging(self, outputs_dict: Dict[str, Tensor]):
        if not self.is_outputs_logging_active():
            return

        if self.val_table is not None:
            return

        if set(outputs_dict.keys()) != set(self.val_table_main_columns):
            raise ValueError(
                f"different keys provided {list(outputs_dict)} than "
                f"declared in val_table_main_columns"
            )

        columns = []
        for key in self.val_table_main_columns:
            _, TS = outputs_dict[key].shape
            for ts in range(TS):
                columns.append(f"{key}_{ts}")

        self.val_table = wandb.Table(columns=columns)

    def _finish_val_outputs_logging(self):
        if not self.is_outputs_logging_active():
            return

        run: Run = self.logger.experiment

        assert self.val_table is not None
        self.val_table.add_column("_step", list(map(int, self.val_table.get_index())))
        artifact = wandb.Artifact(f"run-{run.id}-val_outputs", "run_table")
        artifact.add(self.val_table, "outputs")
        run.log_artifact(artifact)
        # Cleanup after epoch
        del self.val_table

    def _log_val_outputs_dict(self, outputs_dict: Dict[str, Float[Tensor, "b col ts"]]):
        if not self.is_outputs_logging_active():
            return

        self._init_val_outputs_logging(outputs_dict=outputs_dict)

        data: Float[Tensor, "b C"] = rearrange(
            [outputs_dict[column] for column in self.val_table_main_columns],
            "col b ts -> b (col ts)",
        )

        assert self.val_table is not None
        for row in data.tolist():
            self.val_table.add_data(*row)


class TrainAttnMapLoggingMixin:
    trainer: Trainer
    logger: WandbLogger
    hparams: AttributeDict
    gpt: torch.nn.Module
    global_step: int
    action_keys: List[str]
    metadata_keys: List[str]

    def is_attn_map_logging_active(self):
        return (
            isinstance(logger := self.logger, WandbLogger)
            and isinstance(logger.experiment, Run)
            and self.hparams.get("log", {}).get("training", {}).get("attention_maps")
            and self.trainer.state.stage
            in (RunningStage.TRAINING, RunningStage.VALIDATING)
        )

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        with torch.no_grad():
            for layer in self.gpt.layers:  # type: ignore
                x_in = layer.norm1(x) if layer.norm_first else x
                _, attn_map = layer.self_attn(
                    x_in, x_in, x_in, attn_mask=mask, need_weights=True
                )
                attention_maps.append(attn_map)
                x = layer(x)
        return attention_maps

    def _log_attn_maps(
        self,
        *,
        drive_ids,
        frame_idxs,
        frames,
        episode: Float[Tensor, "b to d"],
        episode_mask: Optional[Float[Tensor, "to to"]] = None,
        **kwargs,
    ):
        if not self.is_attn_map_logging_active():
            return

        if self.global_step % self.trainer.log_every_n_steps != 0:  # type: ignore
            return

        # Pick only one sample from batch - consistent with deephouse approach
        batch_idx = 0

        attn_maps: List[Float[Tensor, "to to"]] = self.get_attention_maps(
            episode[batch_idx],
            episode_mask,
        )

        frames = frames[batch_idx]
        images, metas_actions = self._postprocess_attn_maps(
            attn_maps=attn_maps,
            frames=frames,
            tokens=episode.shape[1],
        )

        drive_id = drive_ids[batch_idx]
        frame_idxs = frame_idxs[batch_idx].tolist()
        caption = f"{frame_idxs[0]}-{frame_idxs[-1]} [{drive_id}]"

        stage = self.trainer.state.stage
        data = {
            f"{stage}/attn_map/img-{self.action_keys[idx]}": [
                wandb.Image(img, caption=caption) for img in ts_images
            ]
            for idx, ts_images in enumerate(images)
        }
        data.update(
            {
                f"{stage}/attn_map/meta+actions-{self.action_keys[idx]}": [
                    wandb.Image(img, caption=caption) for img in ts_images
                ]
                for idx, ts_images in enumerate(metas_actions)
            }
        )
        data.update(
            {
                f"{stage}/attn_map/frames": [
                    wandb.Image(img, caption=f"{frame_idxs[idx]} [{drive_id}]")
                    for idx, img in enumerate(frames)
                ]
            }
        )
        self.logger.experiment.log(data=data, step=self.global_step)

    def _postprocess_attn_maps(
        self,
        attn_maps: List[Float[Tensor, "to to"]],
        frames: Float[Tensor, "ts c h w"],
        tokens: int,
    ) -> Tuple[List[List[Image]], List[List[Image]]]:
        timesteps, _, img_height, img_width = frames.shape
        params = self.hparams["log"]["training"]["attention_maps"]
        layer = params["layer"]
        norm = params.get("norm") or "softmax"
        strength = params.get("strength") or 1.0
        reverse = params.get("reverse") or False

        color = params.get("color") or [255, 255, 255]
        color = rearrange(torch.tensor(color).to(frames) / 255.0, "(c 1 1) -> c 1 1")

        mean = params["imgs_mean"]
        std = params["imgs_std"]
        unnormalize = Unnormalize(mean=mean, std=std)
        frames = unnormalize(frames.clone())

        actions_len = len(self.action_keys)
        metadata_len = len(self.metadata_keys)
        meta_sep_act_len = actions_len + 1 + metadata_len
        seq_len = (tokens + 1) // timesteps
        images_len = seq_len - meta_sep_act_len
        cols = params["cols"]
        rows = params["rows"]

        attn_map = attn_maps[layer].clone()
        last_meta_actions = attn_map[-actions_len:]

        if norm == "softmax":
            last_meta_actions = softmax(last_meta_actions, dim=1)
        elif norm == "max":
            last_meta_actions /= last_meta_actions.max(dim=1)[0][..., None]
        else:
            raise NotImplementedError(f"unknown norm for attention maps: {norm}")

        images = []
        metas_actions = []
        for row in last_meta_actions:
            row_images: List[Image] = []
            row_metas_actions: List[Image] = []

            for ts in range(timesteps):
                img_start_idx = ts * seq_len
                img_end_idx = img_start_idx + images_len
                mask = row[img_start_idx:img_end_idx].reshape(rows, cols).clone()
                mask = torch.nn.functional.interpolate(
                    input=mask[None, None, ...],
                    size=(img_height, img_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                if reverse:
                    mask = 1 - mask
                mask = mask * strength

                background = (1.0 - mask) * frames[ts]
                foreground = torch.ones_like(background) * color * mask
                img = background + foreground

                img = (img.permute(1, 2, 0) * 255).int().cpu().numpy().astype(np.uint8)
                row_images.append(fromarray(img))

                ma_start_idx = img_end_idx
                ma_end_idx = ma_start_idx + meta_sep_act_len
                ma = row[ma_start_idx:ma_end_idx]
                if len(ma) < meta_sep_act_len:
                    tmp = torch.zeros(meta_sep_act_len).to(ma)
                    tmp[: len(ma)] = ma
                    ma = tmp
                ma = ma[None, ...]
                ma = (ma * 255).int().cpu().numpy().astype(np.uint8)
                row_metas_actions.append(fromarray(ma))

            images.append(row_images)
            metas_actions.append(row_metas_actions)

        return images, metas_actions
