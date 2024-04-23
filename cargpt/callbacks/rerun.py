from collections.abc import Sequence
from typing import Literal

import pytorch_lightning as pl
import rerun as rr
import rerun.blueprint as rrb
from einops.layers.torch import Rearrange
from funcy import once_per
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict
from typing_extensions import override
from yaak_datasets import Batch

from cargpt.components.episode import Modality
from cargpt.components.objectives.common import PredictionResultKey


class RerunPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        **rerun_init_kwargs,
    ) -> None:
        rerun_init_kwargs.setdefault("application_id", "prediction")
        rerun_init_kwargs.setdefault("spawn", True)
        rerun_init_kwargs.setdefault("strict", True)
        self.rerun_init_kwargs = rerun_init_kwargs

        super().__init__(write_interval)

    @override
    def on_predict_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        rr.init(**self.rerun_init_kwargs)

        # HACK: a lil monkeypatching never hurt nobody
        rr.log_once_per_entity_path = once_per("entity_path")(rr.log)  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def on_predict_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if hasattr(rr, "log_once_per_entity_path"):
            delattr(rr, "log_once_per_entity_path")

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        prediction = prediction.to(pl_module.device)
        inputs, predictions = prediction["inputs"], prediction["predictions"]
        inputs = inputs.update(
            inputs.select(Modality.IMAGE).apply(
                Rearrange("... c h w -> ... h w c"),  # CHW -> HWC for logging
            )
        )

        meta = batch.meta.exclude("drive_id")
        meta.batch_size = inputs.batch_size

        data = TensorDict.from_dict({
            "inputs": inputs,
            "meta": meta,
            "predictions": predictions,
        })

        # TODO: more robust?
        if batch_idx == 0:
            blueprint = self._build_blueprint(data)
            rr.send_blueprint(blueprint)

        for _batch in data.cpu():
            for timestep, elem in enumerate(_batch):
                for nested_key, tensor in elem.items(
                    include_nested=True,
                    leaves_only=True,
                ):
                    path = "/".join(nested_key)

                    match nested_key:
                        case ("meta", name):
                            match name.split("/"):
                                case (_, "ImageMetadata_frame_idx"):
                                    rr.set_time_sequence(path, tensor.item())

                                case (_, "ImageMetadata_time_stamp"):
                                    rr.set_time_nanos(path, tensor.item() * 1000)

                                case _:
                                    rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                        path,
                                        rr.SeriesLine(name=name),
                                        timeless=True,
                                    )
                                    rr.log(path, rr.Scalar(tensor))

                        case ("inputs", modality, name):
                            match modality:
                                case Modality.IMAGE:
                                    rr.log(path, rr.Image(tensor))

                                case Modality.CONTINUOUS | Modality.DISCRETE:
                                    rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                        path,
                                        rr.SeriesLine(name=name),
                                        timeless=True,
                                    )
                                    rr.log(path, rr.Scalar(tensor))

                                case _:
                                    raise NotImplementedError

                        case (
                            "predictions",
                            *_module,
                            PredictionResultKey.GROUND_TRUTH,
                            (Modality.CONTINUOUS | Modality.DISCRETE),
                            name,
                        ) if not tensor.isnan():
                            rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                path,
                                rr.SeriesLine(name=f"gt/{name}"),
                                timeless=True,
                            )
                            rr.log(path, rr.Scalar(tensor))

                        case (
                            "predictions",
                            *_module,
                            PredictionResultKey.PREDICTION,
                            (Modality.CONTINUOUS | Modality.DISCRETE),
                            name,
                        ) if not tensor.isnan():
                            rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                path,
                                rr.SeriesPoint(
                                    name=f"pred/{name}",
                                    marker="cross",
                                    marker_size=4,
                                ),
                                timeless=True,
                            )
                            rr.log(path, rr.Scalar(tensor))

                        case (
                            "predictions",
                            objective,
                            PredictionResultKey.ATTENTION,
                            Modality.SPECIAL,
                            token_from,
                            modality_to,
                            token_to,
                        ):
                            _path = ["attention", objective, token_from, token_to]
                            attn = tensor[timestep]
                            match modality_to:
                                case Modality.IMAGE:
                                    rr.log(_path, rr.Tensor(attn.view(10, 18)))

                                case _:
                                    rr.log(_path, rr.Tensor(attn))

                        case _:
                            pass

    @classmethod
    def _build_blueprint(cls, data: TensorDict) -> rrb.Blueprint:
        # TODO: attention
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Tabs(
                        *(
                            rrb.Spatial2DView(origin=f"/inputs/image/{k}", name=k)
                            for k in data[("inputs", "image")].keys()
                        ),
                        name="image",
                    ),
                ),
                rrb.Vertical(
                    rrb.Tabs(
                        *(
                            rrb.TimeSeriesView(origin=f"/predictions/{k}", name=k)
                            for k in data["predictions"].keys()
                        ),
                        name="objectives",
                    ),
                    rrb.Tabs(
                        *(
                            rrb.TimeSeriesView(origin=k, name=k)
                            for k in ("inputs", "meta")
                        ),
                    ),
                ),
            ),
        )
