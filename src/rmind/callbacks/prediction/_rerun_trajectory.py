from collections.abc import Sequence
from typing import Any, Literal, final, override

import numpy as np
import pytorch_lightning as pl
import rerun as rr
from pydantic import validate_call
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict

from rmind.callbacks.loggers.trajectory_prediction import last_encoded_time_index
from rmind.callbacks.prediction._rerun import RerunPredictionWriter
from rmind.utils.trajectory_camera import densify

# negative radii are UI points: thin lines that stay thin at any zoom
_LINE_RADIUS = -0.75
_POINT_RADIUS = -1.5
_COLORS = {"ground_truth": (255, 255, 255), "prediction_value": (255, 140, 0)}

# top-down panel bounds, ego (rear axle) at the bottom centre. rerun's 2D y axis points down, so
# forward is drawn as -y. A static frame spans exactly these bounds and everything is clipped to it,
# so the panel's automatic zoom never changes.
_TOPDOWN_X_M = (-50.0, 50.0)  # right
_TOPDOWN_FWD_M = (0.0, 150.0)  # forward
_TOPDOWN_LO = (_TOPDOWN_X_M[0], -_TOPDOWN_FWD_M[1])
_TOPDOWN_HI = (_TOPDOWN_X_M[1], -_TOPDOWN_FWD_M[0])


def _topdown(xy: np.ndarray) -> np.ndarray:
    """Ego-frame (x right, y forward) metres -> top-down panel coordinates (forward up)."""
    return np.stack([xy[:, 0], -xy[:, 1]], -1)


def _clip(
    uv: np.ndarray, lo: tuple[float, float], hi: tuple[float, float]
) -> np.ndarray:
    """Points inside the rectangle [lo, hi] (the line is densified, so dropping is enough)."""
    ok = np.all((uv >= lo) & (uv <= hi), axis=-1)
    return uv[ok]


@final
class RerunTrajectoryWriter(BasePredictionWriter):
    """Ground-truth and predicted ego trajectories in a top-down rerun panel.

    Per sample, at the time of its last encoded frame (`camera`'s frame index / time stamp),
    `trajectory/<kind>` in metres: x -50..50, forward 0..150, ego at the bottom centre, fixed by a
    static frame of those bounds.

    `<kind>` is `ground_truth` (white) / `prediction_value` (orange), read from
    `prediction[objective][<kind>].value.trajectory` ((b, P, 3) poses: x right, y forward, theta).
    Logs into the recordings of the trainer's `RerunPredictionWriter` (one logger instance, so one
    recording per input; a second logger would open a separate recording in the viewer).
    """

    @validate_call
    def __init__(
        self,
        *,
        objective: str = "trajectory",
        camera: str = "cam_front_left",
        timelines: dict[str, Literal["sequence", "timestamp"]] | None = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self._objective = objective
        self._camera = camera
        self._frame_logged: set[str] = set()
        # timeline name -> kind; the defaults match the prediction loggers' time columns
        self._timelines = timelines or {
            f"batch/{camera}/frame_idx": "sequence",
            f"batch/{camera}/time_stamp": "timestamp",
        }

    def _time_values(self, data: dict[str, Any], t: int) -> dict[str, Any]:
        frame_idx = data[f"meta/ImageMetadata.{self._camera}/frame_idx"][:, t]
        time_stamp = data[f"meta/ImageMetadata.{self._camera}/time_stamp"][:, t]
        return {"sequence": frame_idx.tolist(), "timestamp": time_stamp.tolist()}

    @staticmethod
    def _log_topdown_frame(recording: rr.RecordingStream) -> None:
        recording.log(
            "trajectory/frame",
            rr.Boxes2D(
                mins=[_TOPDOWN_LO],
                sizes=[
                    [hi - lo for lo, hi in zip(_TOPDOWN_LO, _TOPDOWN_HI, strict=True)]
                ],
                colors=[(90, 90, 90)],
                radii=[_LINE_RADIUS],
            ),
            static=True,
        )
        # ego footprint, rear axle at the origin
        recording.log(
            "trajectory/ego",
            rr.Boxes2D(
                centers=_topdown(np.array([[0.0, 1.35]])),
                sizes=[[1.9, 4.6]],
                colors=[(160, 160, 160)],
                radii=[_LINE_RADIUS],
            ),
            static=True,
        )

    @staticmethod
    def _log_trajectory(
        recording: rr.RecordingStream, kind: str, poses: np.ndarray
    ) -> None:
        """One (P, 3) trajectory in the top-down panel."""
        color = _COLORS[kind]
        line = _clip(_topdown(densify(poses)), _TOPDOWN_LO, _TOPDOWN_HI)
        pts = _clip(_topdown(poses[:, :2]), _TOPDOWN_LO, _TOPDOWN_HI)
        recording.log(
            f"trajectory/{kind}",
            rr.LineStrips2D([line], colors=[color], radii=[_LINE_RADIUS]),
            rr.Points2D(pts, colors=[color], radii=[_POINT_RADIUS]),
        )

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
        objective = prediction.get(self._objective, None)
        if objective is None:
            return
        trajectories = {
            kind: objective[kind].value["trajectory"].float().cpu().numpy()
            for kind in _COLORS
            if kind in objective.keys()  # noqa: SIM118
        }
        if not trajectories:
            return

        logger = next(
            c.logger
            for c in trainer.callbacks
            if isinstance(c, RerunPredictionWriter)  # ty:ignore[unresolved-attribute]
        )
        data = batch["data"]
        t = last_encoded_time_index(pl_module)
        times = self._time_values(data, t)

        for i, input_id in enumerate(batch["meta"]["input_id"]):
            # the RerunLogger's per-input recording (created on first use)
            recording = logger._get_recording(str(input_id))  # noqa: SLF001
            if input_id not in self._frame_logged:
                self._log_topdown_frame(recording)
                self._frame_logged.add(input_id)
            for timeline, kind in self._timelines.items():
                value = times[kind][i]
                recording.set_time(
                    timeline,
                    **{
                        kind: np.datetime64(value, "us")
                        if kind == "timestamp"
                        else int(value)
                    },
                )
            for kind, poses in trajectories.items():
                self._log_trajectory(recording, kind, poses[i])
