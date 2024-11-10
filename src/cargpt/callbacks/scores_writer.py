import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, override

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict

from cargpt.components.episode import Modality

try:
    from rbyte.batch import Batch
except ImportError:
    from typing import Any

    Batch = Any


class ScoresPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        model_artifact: str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        write_frequency: int = 100,
    ) -> None:
        self.write_interval = write_interval
        self.write_frequency = write_frequency
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
        model_version = model_artifact.split("/")[-1]
        self.dir_to_save = Path(f"inference_results/{model_version}/{current_time}")
        super().__init__(write_interval)

    @override
    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.df = pd.DataFrame()

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None,
        batch: Batch,  # pyright: ignore[reportInvalidTypeForm]
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        data = prediction.auto_batch_size_(1).update({"batch": batch}).cpu()
        camera_name = "cam_front_left"  # maybe to extract it

        rows_batch = []
        for drive_id, sample in zip(
            data.get(k := ("batch", "meta", "input_id")).data,
            data.exclude(k),
            strict=True,
        ):  # over clips
            rows_clip = []
            for elem in sample.exclude(("batch", "meta")).auto_batch_size_(
                2
            ):  # over timesteps
                for k, v in elem.items(True, True):  # over elements of one timestep
                    frame_idx = elem[
                        "batch", "table", f"ImageMetadata.{camera_name}.frame_idx"
                    ].item()
                    time_stamp = elem[
                        "batch", "table", f"ImageMetadata.{camera_name}.time_stamp"
                    ].item()

                    rows_ts = {}
                    match k:
                        case (
                            "predictions",
                            *_module,
                            result_key,
                            (Modality.CONTINUOUS | Modality.DISCRETE),
                            name,
                        ) if not v.isnan().all():
                            if name not in rows_ts:
                                rows_ts[name] = {
                                    "frame_idx": frame_idx,
                                    "timestamp": time_stamp,
                                    "drive_id": drive_id,
                                    "name": name,
                                }
                            rows_ts[name][result_key.value] = v.item()

                        case _:
                            pass
                    rows_clip.extend(list(rows_ts.values()))
                rows_batch.extend(rows_clip)

        self.df = pd.concat([self.df, pd.DataFrame(rows_batch)], ignore_index=True)

        # TODO: not keep all the csv but add to the end
        if self.write_interval == "batch" and batch_idx % self.write_frequency == 0:
            write_to_csv(self.df, self.dir_to_save / "scores.csv")
            self.df = pd.DataFrame()

    @override
    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if len(self.df) > 0:
            write_to_csv(self.df, self.dir_to_save / "scores.csv")

        df = pd.read_csv(self.dir_to_save / "scores.csv")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")


def write_to_csv(df: pd.DataFrame, path_to_save: str | os.PathLike[str]):
    path = Path(path_to_save)
    if not path.exists():
        if not path.parent.exists():
            path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)
