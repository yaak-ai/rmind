from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import cv2
import numpy as np
import pytorch_lightning as pl


class VideoWriter(pl.callbacks.BasePredictionWriter):
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
        self.video_writer = cv2.VideoWriter(
            str(self.output_file),
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.fps,
            (width, height),
        )

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: List[np.ndarray],
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
