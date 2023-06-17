from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter


class VideoWriter(BasePredictionWriter):
    def __init__(
        self,
        output_file: Union[str, Path],
        fourcc: str = "vp09",
        fps: int = 30,
        overwrite: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")

        self.output_file = Path(output_file)
        if self.output_file.exists() and not overwrite:
            raise ValueError(
                f"The output file {str(self.output_file.resolve())} exists!"
            )
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.fourcc = fourcc
        self.fps = fps
        self.video_writer = None

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
            height, width, _ = predictions[0].shape
            self._set_video_writer(width, height)
        for vis in predictions:
            self.video_writer.write(vis[:, :, ::-1])  # type: ignore[attr-defined]

    def on_predict_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.video_writer is not None:
            self.video_writer.release()


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
