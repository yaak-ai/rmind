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
from yaak_datasets import Batch
from yaak_datasets.utils.metabase import metabase

from cargpt.components.episode import Modality
from cargpt.components.objectives.common import PredictionResultKey

# TODO: rewrite pandas to polars. Oder?


class ScoresPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        model_artifact: str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        write_frequency: int = 1,
    ) -> None:
        self.write_interval = write_interval
        self.write_frequency = write_frequency
        curent_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
        if "ckpt" in model_artifact:
            model_version = model_artifact.split("/")[-2]
        else:
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
        data = TensorDict.from_dict({
            "meta": batch.meta,
            "predictions": prediction.to(pl_module.device)["predictions"],
        })

        # we need camera to get an access to time_stamp, but since they are all synced we can take any
        camera = next(iter(prediction["inputs"].get("image", default={}).keys()))  # pyright: ignore[reportAttributeAccessIssue]

        rows_batch = []
        for clip in data.cpu():
            drive_id = (
                clip["meta"]
                .pop("drive_id")
                .to(torch.uint8)
                .numpy()
                .tobytes()
                .decode("ascii")
            )

            rows_clip = []
            clip.auto_batch_size_()

            for elem in clip:
                timestamp = elem.get((
                    "meta",
                    f"{camera}/ImageMetadata_time_stamp",
                )).item()
                frame_idx = elem.get((
                    "meta",
                    f"{camera}/ImageMetadata_frame_idx",
                )).item()
                rows_ts = {}  # each row is a timestamp + some predicted value
                for nested_key, tensor in elem.items(True, True):
                    match nested_key:
                        case (
                            "predictions",
                            *_module,
                            result_key,
                            (Modality.CONTINUOUS | Modality.DISCRETE),
                            name,
                        ) if not tensor.isnan().all():
                            if name not in rows_ts:
                                rows_ts[name] = {
                                    "frame_idx": frame_idx,
                                    "timestamp": timestamp,
                                    "drive_id": drive_id,
                                    "name": name,
                                }
                            rows_ts[name][result_key.value] = (
                                tensor.tolist()
                                if result_key is PredictionResultKey.PREDICTION_PROBS
                                else tensor.item()
                            )

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

        rows = []
        for drive_id in df["drive_id"].unique():
            incidents_list = metabase.request_incidents(drive_id=drive_id)
            for dt, inc_type in incidents_list:
                rows.append({
                    "drive_id": drive_id,
                    "datetime": dt,
                    "inc_type": inc_type,
                })

        pd.DataFrame(rows).to_csv(self.dir_to_save / "incidents.csv", index=False)


def write_to_csv(df: pd.DataFrame, path_to_save: str | os.PathLike[str]) -> None:
    path = Path(path_to_save)
    if not path.exists():
        if not path.parent.exists():
            path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)
