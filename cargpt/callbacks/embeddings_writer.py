from pathlib import Path
from typing import Union

import more_itertools as mit
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor


class FeatureWriter(BasePredictionWriter):
    def __init__(
        self, application_id: str, output_dir: Union[str, Path], overwrite: bool = False
    ) -> None:
        super().__init__(write_interval="batch")

        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and not overwrite:
            msg = f"The output file {self.output_dir.resolve()!s} exists!"
            raise ValueError(msg)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: np.ndarray | List[np.ndarray] | Tensor,
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
                case [camera_name, _key]:
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
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pass
