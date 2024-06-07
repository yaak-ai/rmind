from collections.abc import Sequence
from pathlib import Path
from typing import Any

import more_itertools as mit
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict
from typing_extensions import override

from cargpt.callbacks.index_writer import IndexWriter

class EmbeddingWriter(BasePredictionWriter):
    def __init__(self, output_dir: str | Path, index_writer: IndexWriter) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.index_writer = index_writer

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
        camera_name = mit.one(batch.frames.keys())
        meta = batch.meta
        drive_ids = meta.pop("drive_id").detach()
        drive_ids = [
            drive_id.type(torch.uint8).cpu().numpy().tobytes().decode("ascii").strip()
            for drive_id in drive_ids
        ]
        frame_idxs = meta[f"{camera_name}/ImageMetadata_frame_idx"].tolist()

        obj = {
            "features": prediction.detach().cpu(),
            "frame_idx": frame_idxs,
            "drive_id": drive_ids,
            "camera_name": camera_name,
        }
        torch.save(obj, f"{self.output_dir}/{batch_idx:06}.pt")
