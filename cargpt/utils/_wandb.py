from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import more_itertools as mit
import numpy as np
import torch.nn
import wandb
from einops import rearrange
from jaxtyping import Float
from PIL.Image import Image, fromarray
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import Tensor, softmax
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from cargpt.visualization.utils import Unnormalize


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(cls, name: str, **kwargs):
        get_artifact = (
            wandb.run.use_artifact
            if wandb.run is not None and not isinstance(wandb.run, RunDisabled)
            else wandb.Api().artifact
        )

        artifact = get_artifact(name, type="model")
        artifact_dir = artifact.download()
        ckpt_path = mit.one(Path(artifact_dir).glob("*.ckpt")).absolute().as_posix()

        return cls.load_from_checkpoint(f"file://{ckpt_path}", **kwargs)  # pyright: ignore


class ValOutputsLoggingTableMixin:
    trainer: Trainer
    logger: WandbLogger
    hparams: AttributeDict

    @property
    def val_table_main_columns(self):
        return self._val_table_main_columns

    @val_table_main_columns.setter
    def val_table_main_columns(self, columns: List[str]):
        self._val_table_main_columns = list(columns)

    @property
    def val_table(self):
        return getattr(self, "_val_table", None)

    @val_table.setter
    def val_table(self, wandb_table: wandb.Table):
        self._val_table = wandb_table

    @val_table.deleter
    def val_table(self):
        delattr(self, "_val_table")

    def is_outputs_logging_active(self):
        return (
            isinstance(self.logger.experiment, Run)
            and self.hparams.get("log", {}).get("validation", {}).get("outputs")
            and self.trainer.state.stage != "sanity_check"
        )

    def _init_val_outputs_logging(self, outputs_dict: Dict[str, Tensor]):
        if not self.is_outputs_logging_active():
            return

        if self.val_table is not None:
            return

        if set(outputs_dict.keys()) != set(self.val_table_main_columns):
            msg = f"different keys provided {list(outputs_dict)} than declared in val_table_main_columns"
            raise ValueError(
                msg,
            )

        columns = []
        for key in self.val_table_main_columns:
            _, TS = outputs_dict[key].shape
            for ts in range(TS):
                columns.append(f"{key}_{ts}")  # noqa: PERF401

        self.val_table = wandb.Table(columns=columns)

    def _finish_val_outputs_logging(self):
        if not self.is_outputs_logging_active():
            return

        run: Run = self.logger.experiment

        assert self.val_table is not None
        self.val_table.add_column("_step", list(map(int, self.val_table.get_index())))
        artifact = wandb.Artifact(f"run-{run.id}-val_outputs", "run_table")
        artifact.add(
            self.val_table,
            "outputs",
        )  # pyright: ignore[reportUnusedCallResult]
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


class TrainValAttnMapLoggingMixin:
    trainer: Trainer
    logger: WandbLogger
    hparams: AttributeDict
    gpt: torch.nn.Module
    global_step: int
    action_keys: List[str]
    metadata_keys: List[str]
    sensor_detokenization: Dict[str, Callable]

    def is_attn_map_logging_active(self):
        try:
            _ = self._get_attn_map_logging_params()
        except KeyError:
            params_fine = False
        else:
            params_fine = True

        return params_fine and isinstance(self.logger.experiment, Run)

    def _should_attn_map_log_now(self, batch_idx):
        params: Dict[str, Any] = self._get_attn_map_logging_params()
        if self.trainer.state.stage == RunningStage.TRAINING:
            curr_step = self.global_step
        else:  # validation
            curr_step = batch_idx

        log_freq = params.get("log_freq") or 50
        return curr_step % log_freq == 0

    def _get_attn_map_logging_params(self):
        params: Dict[str, Any] = self.hparams["log"]
        if self.trainer.state.stage == RunningStage.TRAINING:
            params = params["training"]["attention_maps"]
        elif self.trainer.state.stage == RunningStage.VALIDATING:
            params = params["validation"]["attention_maps"]
        else:
            msg = "wrong stage"
            raise KeyError(msg)
        return params

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        with torch.no_grad():
            for layer in self.gpt.layers:  # type: ignore
                x_in = layer.norm1(x) if layer.norm_first else x
                _, attn_map = layer.self_attn(
                    x_in,
                    x_in,
                    x_in,
                    attn_mask=mask,
                    need_weights=True,
                )
                attention_maps.append(attn_map)
                x = layer(x)
        return attention_maps

    def _logits_to_real(self, logits: Float[Tensor, "to e"]) -> List[float]:
        action_logits = logits[-len(self.action_keys) :, ...]
        tokens = torch.argmax(torch.softmax(action_logits, dim=-1), dim=-1)

        out = []
        for token, action_key in zip(tokens, self.action_keys):
            t = token - self.hparams.tokens_shift[action_key]  # type: ignore
            prediction = self.sensor_detokenization[action_key](t.clone())
            out.append(prediction.item())
        return out

    def _log_attn_maps(
        self,
        *,
        batch,
        batch_idx: int,
        episode_values: Float[Tensor, "b to"],
        logits: Float[Tensor, "b to e"],
        episode: Float[Tensor, "b to d"],
        episode_mask: Optional[Float[Tensor, "to to"]] = None,
    ):
        if not self.is_attn_map_logging_active():
            return

        if not self._should_attn_map_log_now(batch_idx):
            return

        # Pick only one sample from batch - consistent with deephouse approach
        idx = 0

        attn_maps: List[Float[Tensor, "to to"]] = self.get_attention_maps(
            episode[idx],
            episode_mask,
        )

        camera = mit.only(batch["frames"].keys())

        frames: Float[Tensor, "ts c h w"] = batch["frames"][camera][idx]
        images, metas_actions = self._postprocess_attn_maps(
            attn_maps=attn_maps,
            frames=frames,
            tokens=episode.shape[1],
        )

        # Log images
        drive_id = (
            batch["meta"]["drive_id"][idx]
            .type(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
            .decode("ascii")
            .strip()
        )
        frame_idxs = batch["meta"][f"{camera}/ImageMetadata_frame_idx"][idx].tolist()
        gt_actions_values = episode_values[idx, -len(self.action_keys) :].tolist()
        pred_actions_values = self._logits_to_real(logits[idx])

        stage = self.trainer.state.stage
        data = {
            f"{stage}/attn_map/{prefix}-{self.action_keys[idx]}": [
                wandb.Image(
                    img,
                    caption=(
                        f"gt:{gt_actions_values[idx]:.4f}, "
                        f"pred: {pred_actions_values[idx]:.4f}, "
                        f"({frame_idxs[frame_idx]}) [{drive_id}]"
                    ),
                )
                for frame_idx, img in enumerate(ts_images)
            ]
            for prefix, values in (("img", images), ("meta+actions", metas_actions))
            for idx, ts_images in enumerate(values)
        }
        data.update(
            {
                f"{stage}/attn_map/frames": [
                    wandb.Image(img, caption=f"{frame_idxs[idx]} [{drive_id}]")
                    for idx, img in enumerate(frames)
                ],
            },
        )
        # Don't pass step, don't commit - let it be committed with logs
        self.logger.experiment.log(data=data, commit=False)

    def _postprocess_attn_maps(
        self,
        attn_maps: List[Float[Tensor, "to to"]],
        frames: Float[Tensor, "ts c h w"],
        tokens: int,
    ) -> Tuple[List[List[Image]], List[List[Image]]]:
        # Extract and prepare parameters
        params = self._get_attn_map_logging_params()
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

        # Extract and calculate dimensions
        timesteps, _, img_height, img_width = frames.shape
        actions_len = len(self.action_keys)
        metadata_len = len(self.metadata_keys)
        meta_sep_act_len = actions_len + 1 + metadata_len
        seq_len = (tokens + 1) // timesteps
        images_len = seq_len - meta_sep_act_len
        cols = params["cols"]
        rows = params["rows"]

        attn_map = attn_maps[layer].clone()
        last_meta_actions = attn_map[-actions_len:]

        # Normalize values using requested method
        if norm == "softmax":
            last_meta_actions = softmax(last_meta_actions, dim=1)
        elif norm == "max":
            last_meta_actions /= last_meta_actions.max(dim=1)[0][..., None]
        else:
            msg = f"unknown norm for attention maps: {norm}"
            raise NotImplementedError(msg)

        # Split attention map into multiple images
        images = []
        metas_actions = []
        for row in last_meta_actions:
            row_images: List[Image] = []
            row_metas_actions: List[Image] = []

            for ts in range(timesteps):
                # Get frame attention mixed with real frame
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

                # Get non-image tokens visualization
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
