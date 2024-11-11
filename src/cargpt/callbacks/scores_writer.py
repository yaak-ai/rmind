from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, override, Literal

import pytorch_lightning as pl
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from funcy import once_per
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict
import pandas as pd

from cargpt.components.episode import Modality
from cargpt.components.objectives.base import PredictionResultKey

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict
from typing_extensions import override

from cargpt.components.episode import Modality
from cargpt.components.objectives.base import PredictionResultKey

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
        curent_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
        model_version = model_artifact.split("/")[-1]
        self.dir_to_save = Path(f"inference_results/{model_version}/{curent_time}")
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
        batch: Batch,  # pyright: ignore[reportGeneralTypeIssues]
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
        ):  # over clips in batch
            frame_idx = sample[
                "batch", "table", f"ImageMetadata.{camera_name}.frame_idx"
            ][-1].item()
            time_stamp = sample[
                "batch", "table", f"ImageMetadata.{camera_name}.time_stamp"
            ][-1].item()
            # we take -1 since it is only where predicitons are stores
            rows = {}
            for k, v in (
                sample["predictions"].auto_batch_size_(1)[-1].items(True, True)
            ):  # over elements in the las timestep
                match k:
                    case (
                        _objective,
                        result_key,
                        (Modality.CONTINUOUS | Modality.DISCRETE),
                        name,
                    ) if not v.isnan().all():
                        if name not in rows:
                            rows[name] = {
                                "frame_idx": frame_idx,
                                "timestamp": time_stamp,
                                "drive_id": drive_id,
                                "name": name,
                            }
                        rows[name][result_key.value] = v.item()

                    case _:
                        pass
            rows_batch.extend(list(rows.values()))

        self.df = pd.concat([self.df, pd.DataFrame(rows_batch)], ignore_index=True)
        self.df["datetime"] = pd.to_datetime(self.df["timestamp"], unit="ns")

        # TODO: not keep all the csv but add to the end
        if self.write_interval == "batch" and batch_idx % self.write_frequency == 0:
            write_to_csv(self.df, self.dir_to_save / "scores.csv")
            self.df = pd.DataFrame()

    @override
    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if len(self.df) > 0:
            write_to_csv(self.df, self.dir_to_save / "scores.csv")

        # --- TODO: Incidents ---
        # df = pd.read_csv(self.dir_to_save / "scores.csv")

        # rows = []
        # for drive_id in df["drive_id"].unique():
        # incidents_list = metabase.request_incidents(drive_id=drive_id)
        # for dt, inc_type in incidents_list:
        #     rows.append({
        #         "drive_id": drive_id,
        #         "datetime": dt,
        #         "inc_type": inc_type,
        #     })

        # pd.DataFrame(rows).to_csv(self.dir_to_save / "incidents.csv", index=False)


def write_to_csv(df: pd.DataFrame, path_to_save: str | os.PathLike[str]):
    path = Path(path_to_save)
    if not path.exists():
        if not path.parent.exists():
            path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)
