from pathlib import Path
from typing import Any

import more_itertools as mit
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict.tensordict import TensorDict
from typing_extensions import override


class EmbeddingWriter(BasePredictionWriter):
    def __init__(self, output_dir: str | Path) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)

    @override
    def write_on_batch_end(
        self, predictions: TensorDict, batch: Any, batch_idx: int
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
            "features": predictions.detach().cpu(),
            "frame_idx": frame_idxs,
            "drive_id": drive_ids,
        }
        torch.save(obj, f"{self.output_dir}/{batch_idx:06}.pt")

    def on_predict_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pass
