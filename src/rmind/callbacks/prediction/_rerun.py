from collections.abc import Sequence
from typing import Any, Literal, final, override

import pytorch_lightning as pl
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import BasePredictionWriter
from rbyte.viz.loggers import RerunLogger
from tensordict import TensorDict

from rmind.utils._camera_projection import project_trajectories_to_image


@final
class RerunPredictionWriter(BasePredictionWriter):
    @validate_call
    def __init__(
        self,
        logger: InstanceOf[RerunLogger],
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self._logger = logger

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        data = (
            TensorDict(batch=batch, prediction=prediction)
            .apply(lambda x: x.float() if x.is_floating_point() else x)
            .auto_batch_size_(1)  # ty:ignore[unresolved-attribute]
            .lock_()
        )

        self._logger.log(data)


@final
class DrivoRRerunPredictionWriter(BasePredictionWriter):
    """`RerunPredictionWriter` specialized for DrivoR's `trajectory`
    prediction: additionally projects the ego-frame `best_prediction`
    (best hypothesis), `prediction` (all `num_queries` hypotheses), and
    `ground_truth_xy` into `cam_front_left` pixel space before logging, so
    trajectories can be overlaid directly on the camera image (as
    `LineStrips2D`, children of the image entity) instead of floating in a
    disconnected 3D view. See `rmind.utils._camera_projection`.
    """

    @validate_call
    def __init__(
        self,
        logger: InstanceOf[RerunLogger],
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self._logger = logger

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # trajectory positions are `/100`-normalized (see
        # `rmind.components.drivor.trajectory_target`); the projection works
        # in meters, so undo that here.
        vehicles = [str(input_id).split("/")[0] for input_id in batch["meta"]["input_id"]]
        traj = prediction["trajectory"]

        uv = TensorDict(
            {
                "best_prediction_uv": project_trajectories_to_image(
                    traj["best_prediction"] * 100.0, vehicles
                ),
                "prediction_uv": project_trajectories_to_image(
                    traj["prediction"] * 100.0, vehicles
                ),
                "ground_truth_uv": project_trajectories_to_image(
                    traj["ground_truth_xy"] * 100.0, vehicles
                ),
            },
            batch_size=traj.batch_size,
        )
        prediction = prediction.clone(recurse=False)
        prediction["trajectory"] = traj.clone(recurse=False).update(uv)

        data = (
            TensorDict(batch=batch, prediction=prediction)
            .apply(lambda x: x.float() if x.is_floating_point() else x)
            .auto_batch_size_(1)  # ty:ignore[unresolved-attribute]
            .lock_()
        )

        self._logger.log(data)
