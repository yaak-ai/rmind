from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, override

import more_itertools as mit
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict


class EmbeddingWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str | Path,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"],
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None = None,
        batch: Any = None,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> None:
        camera_name = mit.one(batch.frame.keys())
        obj = {
            "features": prediction.detach().cpu(),
            "frame_idx": batch.table[
                f"image_metadata.{camera_name}.frame_idx"
            ].tolist(),
            "drive_id": batch.meta.input_id,
            "camera_name": camera_name,
        }
        torch.save(obj, f"{self.output_dir}/{batch_idx:06}.pt")
