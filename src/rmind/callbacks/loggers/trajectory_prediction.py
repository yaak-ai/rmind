from typing import Annotated, Any, final

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pydantic import AfterValidator, validate_call
from torch.utils._pytree import MappingKey, key_get  # noqa: PLC2701

from rmind.callbacks.safe import SafeCallback
from rmind.components.nn import ChunkFields
from rmind.utils.trajectory_camera import DEFAULT_CAMERA, densify, project
from wandb import Image

from .common import (
    _bind_hook_arguments,
    _figure_to_rgba,
    _get_wandb_loggers,
    _validate_hook,
)

_TIME_DIM = 1


def last_encoded_time_index(pl_module: pl.LightningModule) -> int:
    """Last encoded timestep on the raw batch's time axis (ChunkFields narrows the clip)."""
    for module in pl_module.episode_builder.input_transform.modules():  # ty:ignore[unresolved-attribute]
        if isinstance(module, ChunkFields) and module.dim == _TIME_DIM:
            return module.episode_length - 1
    return -1


@final
class WandbTrajectoryPredictionLogger(SafeCallback):
    """Ground-truth vs predicted trajectory for selected validation samples.

    Reads the trajectory objective's `_artifacts` (`target`, `predicted`: (b, P, 3) ego-frame
    poses from the last encoded timestep) and draws, per sample, the camera frame of that
    timestep with both trajectories projected onto the road, next to a top-down view.
    Samples are picked like the patch-similarity logger: `sample_id % every_n_sample == 0`.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        target_path: list[str | int],
        predicted_path: list[str | int],
        image_path: list[str | int],
        sample_id_path: list[str | int],
        every_n_sample: int,
        key_path: list[str | int] | None = None,
        objective: str = "trajectory",
        camera: dict[str, float] | None = None,
        fail_gracefully: bool = True,
        disable_on_error: bool = False,
    ) -> None:
        super().__init__(
            fail_gracefully=fail_gracefully, disable_on_error=disable_on_error
        )
        self._key = key
        self._target_path = tuple(map(MappingKey, target_path))
        self._predicted_path = tuple(map(MappingKey, predicted_path))
        self._image_path = tuple(map(MappingKey, image_path))
        self._sample_id_path = tuple(map(MappingKey, sample_id_path))
        self._every_n_sample = every_n_sample
        # training steps carry the latent keys instead of a decoded batch; decode the plotted rows here
        self._key_path = (
            tuple(map(MappingKey, key_path)) if key_path is not None else None
        )
        self._objective = objective
        # camera relative to the trajectory reference point (rear axle): metres / degrees
        self._camera = camera or DEFAULT_CAMERA
        self._when = when
        setattr(self, when, self._safe_hook(when, self._call))

    def _plot(
        self, image: np.ndarray, target: np.ndarray, predicted: np.ndarray, caption: str
    ) -> Image:
        fde = float(np.linalg.norm(predicted[-1, :2] - target[-1, :2]))
        fig, (a0, a1) = plt.subplots(
            1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.9, 1]}
        )
        h, w = image.shape[:2]
        a0.imshow(image)
        for poses, color, label in (
            (target, "white", "ground truth"),
            (predicted, "tab:orange", "prediction"),
        ):
            uv = project(densify(poses), self._camera, (w, h))
            if len(uv) > 1:
                a0.plot(uv[:, 0], uv[:, 1], "-", color=color, lw=2.5, label=label)
            pts = project(poses[:, :2], self._camera, (w, h))
            a0.scatter(
                pts[:, 0],
                pts[:, 1],
                s=14,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
        a0.set_xlim(0, w)
        a0.set_ylim(h, 0)
        a0.axis("off")
        a0.legend(loc="upper right", fontsize=8)
        for poses, color in ((target, "black"), (predicted, "tab:orange")):
            a1.plot(
                np.r_[0, poses[:, 0]],
                np.r_[0, poses[:, 1]],
                "-o",
                color=color,
                ms=3,
                lw=1.6,
            )
        ext = (
            max(
                np.abs(np.r_[0, target[:, 0], predicted[:, 0]]).max(),
                np.r_[0, target[:, 1], predicted[:, 1]].max(),
                2.0,
            )
            * 1.15
        )
        a1.set_xlim(-ext, ext)
        a1.set_ylim(-0.2 * ext, 1.8 * ext)
        a1.set_aspect("equal")
        a1.grid(alpha=0.3)
        a1.set_xlabel("x, right (m)")
        a1.set_ylabel("y, forward (m)")
        a1.set_title(f"final position error {fde:.2f} m", fontsize=10)
        fig.suptitle(caption, fontsize=10)
        fig.tight_layout()
        try:
            return Image(_figure_to_rgba(fig), caption=f"{caption}, FDE {fde:.2f} m")
        finally:
            plt.close(fig)

    def _decode(
        self, pl_module: pl.LightningModule, key: Any, rows: list[int], batch_size: int
    ) -> Any:
        """Greedy-decode only `rows` (eval mode, no grad) and place them in a batch-sized tensor."""
        objective = pl_module.objectives[self._objective]  # ty:ignore[not-subscriptable]
        was_training = objective.training
        objective.eval()
        try:
            poses, _ = objective.generate_from_key(key[rows])
        finally:
            objective.train(was_training)
        out = poses.new_zeros((batch_size, *poses.shape[1:]))
        out[rows] = poses
        return out

    def _call(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        loggers = _get_wandb_loggers(pl_module)
        if trainer.sanity_checking or not loggers or not trainer.is_global_zero:
            return
        bound = _bind_hook_arguments(
            self, self._when, trainer, pl_module, *args, **kwargs
        )
        batch, outputs = bound.get("batch"), bound.get("outputs")
        sample_ids = key_get(batch, self._sample_id_path)
        rows = (
            (sample_ids % self._every_n_sample == 0)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if not rows:
            return
        try:
            target = key_get(outputs, self._target_path)
        except (KeyError, TypeError):
            return  # this step carries no trajectory artifacts
        try:
            predicted = key_get(outputs, self._predicted_path)
        except (KeyError, TypeError):
            if self._key_path is None:
                return
            predicted = self._decode(
                pl_module, key_get(outputs, self._key_path), rows, len(sample_ids)
            )
        t = last_encoded_time_index(pl_module)
        images = key_get(batch, self._image_path)
        logged = [
            self._plot(
                images[i, t].detach().cpu().numpy(),
                target[i].detach().float().cpu().numpy(),
                predicted[i].detach().float().cpu().numpy(),
                f"sample {int(sample_ids[i])}, step {trainer.global_step}",
            )
            for i in rows
        ]
        for logger_ in loggers:
            logger_.log_image(key=self._key, images=logged, step=trainer.global_step)
