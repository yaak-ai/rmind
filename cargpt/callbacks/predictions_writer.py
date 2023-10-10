import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import more_itertools as mit
import numpy as np
import parse
import pytorch_lightning as pl
import torch
import wandb
from einops import rearrange
from jaxtyping import Float
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from cargpt.visualization.trajectory import (
    draw_preds,
    draw_trajectory,
    smooth_predictions,
)


class ScoreWriter(BasePredictionWriter):
    def __init__(
        self,
        output_file: str | Path,
        incidents_file: str | Path,
        overwrite: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")

        self.output_file = Path(output_file)
        if self.output_file.exists() and not overwrite:
            raise ValueError(
                f"The output file {str(self.output_file.resolve())} exists!"
            )
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.incidents_file = Path(incidents_file)
        self.scores = []

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        score: Tensor,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        camera_name = mit.one(batch["frames"].keys())
        meta = batch["meta"]
        drive_ids = meta.pop("drive_id").detach()

        drive_ids = [
            drive_id.type(torch.uint8).cpu().numpy().tobytes().decode("ascii").strip()
            for drive_id in drive_ids
        ]
        frame_idxs = meta[f"{camera_name}/ImageMetadata_frame_idx"].tolist()
        score = score.detach().cpu().numpy()

        obj = {
            "batch_id": batch_idx,
            "score": f"{score:.6f}",
            "frame_idx": frame_idxs[0][-1],
            "drive_id": drive_ids,
        }
        self.scores.append(obj)

    def on_predict_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        with self.output_file.open("w") as pfile:
            json.dump(self.scores, pfile)

        with self.incidents_file.open("r") as pfile:
            incidents = json.load(pfile)

        drive_ids = [s["drive_id"] for s in self.scores]
        preds = [[s["frame_idx"], s["score"]] for s in self.scores]

        # Gehacked af
        drive_ids = list(mit.value_chain(*[s["drive_id"] for s in self.scores]))
        for incident in incidents:
            dongle_id = incident["dongle_id"]
            canonical_name = incident["canonical_name"]
            incident["drive_id"] = f"{dongle_id}/{canonical_name}"

        incidentds_in_drive = [
            incident for incident in incidents if incident["drive_id"] in drive_ids
        ]

        annotations = []

        for incident in incidentds_in_drive:
            annotation_start = incident["annotation_start"]
            annotation_end = incident["annotation_end"]
            start = parse.parse(
                "{:d} years {:d} mons {:d} days {:d} hours {:d} mins {:f} secs",
                annotation_start,
            )
            end = parse.parse(
                "{:d} years {:d} mons {:d} days {:d} hours {:d} mins {:f} secs",
                annotation_end,
            )
            # HH:MM:SS * FPS
            start_idx = (
                (start[3] * 60 + start[4]) * 60 + start[5]
            ) * 30  # pyright: ignore
            end_idx = ((end[3] * 60 + end[4]) * 60 + end[5]) * 30  # pyright: ignore
            annotation_idx = (start_idx + end_idx) // 2
            annotations.append([annotation_idx, 0.0])

        trainer.logger.log_table(  # pyright: ignore
            key="predict/SafetyScore", columns=["frame_idxs", "score"], data=preds  # type: ignore[union-attr]
        )
        trainer.logger.log_table(  # pyright: ignore
            key="predict/Incidents", columns=["frame_idxs", "score"], data=annotations  # type: ignore[union-attr]
        )


class FeatureWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")

        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and not overwrite:
            raise ValueError(
                f"The output file {str(self.output_dir.resolve())} exists!"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Tensor,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        camera_name = mit.one(batch["frames"].keys())
        meta = batch["meta"]
        drive_ids = meta.pop("drive_id").detach()

        for key in list(meta.keys()):
            match key.split("/"):
                case [_name, _key] if _name == camera_name:
                    meta.rename_key_(key, _key)

        drive_ids = [
            drive_id.type(torch.uint8).cpu().numpy().tobytes().decode("ascii").strip()
            for drive_id in drive_ids
        ]
        frame_idxs = meta["ImageMetadata_frame_idx"].tolist()

        obj = {
            "features": predictions.detach().cpu(),
            "frame_idx": frame_idxs,
            "drive_id": drive_ids,
        }
        torch.save(obj, f"{self.output_dir}/{batch_idx:06}.pt")

    def on_predict_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        pass


class VideoWriter(BasePredictionWriter):
    def __init__(
        self,
        video_file: Union[str, Path],
        predictions_file: Union[str, Path],
        fourcc: str = "vp09",
        fps: int = 30,
        overwrite: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")

        self.output_file = Path(video_file)
        self.predictions_file = Path(predictions_file)
        if self.output_file.exists() and not overwrite:
            raise ValueError(
                f"The output file {str(self.output_file.resolve())} exists!"
            )
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)

        self.fourcc = fourcc
        self.fps = fps
        self.video_writer = None
        self.predictions = []

    def __del__(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()

    def _set_video_writer(self, width: int, height: int) -> None:
        self.video_writer = cv2.VideoWriter(  # type: ignore
            str(self.output_file),
            cv2.VideoWriter_fourcc(*self.fourcc),  # type: ignore
            self.fps,
            (width, height),
        )

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: np.ndarray | List[np.ndarray],
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.video_writer is None:
            height, width, _ = predictions[0][0].shape
            self._set_video_writer(width, height)
        self.predictions.append(predictions)
        if not pl_module.logging.smooth_predictions:  # type: ignore[union-attr]
            vis, _ = predictions
            self.video_writer.write(vis[0, :, :, ::-1])  # type: ignore[attr-defined]

    def on_predict_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if pl_module.logging.smooth_predictions:  # type: ignore[union-attr]
            images, metadatas = zip(*self.predictions)

            # Numpy interpolate here
            # choese window_size from [1, 3, 5, 7]
            metadatas = smooth_predictions(
                metadatas, window_size=pl_module.logging.smooth_kernel_size  # type: ignore[union-attr]
            )

            for vis, metadata in zip(images, metadatas):
                pred_points_3d: Float[
                    Tensor, "f n 3"
                ] = pl_module.get_trajectory_3d_points(  # type: ignore[union-attr]
                    steps=pl_module.gt_steps,
                    time_interval=pl_module.gt_time_interval,
                    **metadata,
                )

                pred_points_2d: Float[Tensor, "f n 2"] = rearrange(
                    pl_module.camera.project(rearrange(pred_points_3d, "f n d -> (f n) 1 1 d")),  # type: ignore
                    "(f n) 1 1 d -> f n (1 1 d)",
                    f=1,
                )
                draw_trajectory(
                    vis,
                    pred_points_2d,
                    point_color=(0, 255, 0),
                    line_color=(0, 255, 0),
                )
                draw_preds(vis, metadata, line_color=(0, 255, 0))
                self.video_writer.write(vis[0, :, :, ::-1])  # type: ignore[attr-defined]

        if self.video_writer is not None:
            self.video_writer.release()

        with self.predictions_file.open("w") as pfile:
            images, metadatas = zip(*self.predictions)
            metadatas = [
                {key: value.item() for key, value in m.items()}
                for idx, m in enumerate(metadatas)
            ]
            json.dump(metadatas, pfile)


class CSVWriter(BasePredictionWriter):
    def __init__(
        self,
        output_file: Union[str, Path],
        overwrite: bool = False,
    ):
        super(CSVWriter, self).__init__(write_interval="batch")

        self.output_file = Path(output_file)
        if self.output_file.exists() and not overwrite:
            raise ValueError(
                f"The output file {str(self.output_file.resolve())} exists!"
            )
        self.has_header = False

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Dict[int, Dict[int, Dict[str, Dict[str, int | float]]]],
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.has_header:
            self._write_header(predictions)

        keys, labels = self._get_keys_and_labels(predictions)
        rows = []
        for b, batch_preds in predictions.items():
            for ts, ts_preds in batch_preds.items():
                row: List[Any] = [batch_idx, b, ts]
                row.extend([ts_preds[key][label] for key in keys for label in labels])
                rows.append("\t".join(map(str, row)))

        with self.output_file.open("a") as f:
            f.write("\n".join(rows))
            f.write("\n")

    def _write_header(
        self, predictions: Dict[int, Dict[int, Dict[str, Dict[str, int | float]]]]
    ) -> None:
        keys, labels = self._get_keys_and_labels(predictions)
        column_names = [f"{key}_{label}" for key in keys for label in labels]
        header = ["batch_idx", "batch_no", "timestamp"] + column_names
        with self.output_file.open("w") as f:
            f.write("\t".join(header))
            f.write("\n")
        self.has_header = True

    def _get_keys_and_labels(
        self, predictions: Dict[int, Dict[int, Dict[str, Dict[str, int | float]]]]
    ) -> Tuple[List[str], List[str]]:
        batch = list(predictions)[0]
        ts = list(predictions[batch])[0]
        keys = sorted(predictions[batch][ts])
        labels = sorted(predictions[batch][ts][keys[0]])
        return keys, labels

    def on_predict_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.has_header = False
