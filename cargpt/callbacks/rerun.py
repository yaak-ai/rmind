from collections.abc import Sequence
from typing import Literal

import pytorch_lightning as pl
import rerun as rr
import rerun.blueprint as rrb
from einops.layers.torch import Rearrange
from funcy import once_per
from more_itertools import always_iterable
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict
from typing_extensions import override

from cargpt.components.episode import Modality
from cargpt.components.objectives.common import PredictionResultKey

try:
    from yaak_datasets import Batch
except ImportError:
    from typing import Any

    Batch = Any


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
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        rr.init(**self.rerun_init_kwargs)

        # HACK: a lil monkeypatching never hurt nobody
        rr.log_once_per_entity_path = once_per("entity_path")(rr.log)  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def on_predict_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
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
        batch: Batch,  # pyright: ignore[reportInvalidTypeForm]
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        prediction = prediction.to(pl_module.device)
        inputs, predictions = prediction["inputs"], prediction["predictions"]

        inputs = inputs.update(
            inputs.select(Modality.IMAGE).apply(
                Rearrange("... c h w -> ... h w c")  # CHW -> HWC for logging
            )
        )

        meta = batch.meta.exclude("drive_id")
        meta.batch_size = inputs.batch_size

        data = TensorDict.from_dict({
            "inputs": inputs,
            "meta": meta,
            "predictions": predictions,
        })

        cameras = data.get(("inputs", "image"), default={}).keys()  # pyright: ignore[reportAttributeAccessIssue]

        # TODO: more robust?
        if batch_idx == 0:
            blueprint = self._build_blueprint(data)
            rr.send_blueprint(blueprint)

        for _batch in data.cpu():
            for timestep, elem in enumerate(_batch):
                # pop and process time keys first
                for camera in cameras:
                    if v := elem.pop(
                        k := ("meta", f"{camera}/ImageMetadata_frame_idx"), default=None
                    ):
                        rr.set_time_sequence("/".join(k), v.item())

                    if v := elem.pop(
                        k := ("meta", f"{camera}/ImageMetadata_time_stamp"),
                        default=None,
                    ):
                        rr.set_time_nanos("/".join(k), v.item() * 1000)

                for nested_key, tensor in elem.items(True, True):
                    path = "/".join(nested_key)

                    match nested_key:
                        case ("meta", name):
                            rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                path, rr.SeriesLine(name=name), static=True
                            )
                            rr.log(path, rr.Scalar(tensor))

                        case ("inputs", modality, name):
                            match modality:
                                case Modality.IMAGE:
                                    rr.log(path, rr.Image(tensor))

                                case Modality.CONTINUOUS | Modality.DISCRETE:
                                    rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                        path, rr.SeriesLine(name=name), static=True
                                    )
                                    rr.log(path, rr.Scalar(tensor))

                                case _:
                                    raise NotImplementedError

                        case (
                            "predictions",
                            *_module,
                            result_key,
                            (Modality.CONTINUOUS | Modality.DISCRETE),
                            name,
                        ) if not tensor.isnan().all():
                            match result_key:
                                case PredictionResultKey.GROUND_TRUTH:
                                    rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                        path,
                                        rr.SeriesLine(name=f"gt/{name}"),
                                        static=True,
                                    )
                                    rr.log(path, rr.Scalar(tensor))

                                case PredictionResultKey.PREDICTION:
                                    rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                        path,
                                        rr.SeriesPoint(
                                            name=f"pred/{name}",
                                            marker="cross",
                                            marker_size=4,
                                        ),
                                        static=True,
                                    )
                                    rr.log(path, rr.Scalar(tensor))

                                case PredictionResultKey.PREDICTION_PROBS:
                                    rr.log(path, rr.BarChart(tensor))

                                case PredictionResultKey.SCORE_LOGPROB:
                                    rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                        path,
                                        rr.SeriesPoint(
                                            name=f"logp/{name}",
                                            marker="diamond",
                                            marker_size=4,
                                        ),
                                        static=True,
                                    )
                                    rr.log(path, rr.Scalar(tensor))

                                case PredictionResultKey.SCORE_L1:
                                    rr.log_once_per_entity_path(  # pyright: ignore[reportAttributeAccessIssue]
                                        path,
                                        rr.SeriesPoint(
                                            name=f"l1/{name}",
                                            marker="circle",
                                            marker_size=4,
                                        ),
                                        static=True,
                                    )
                                    rr.log(path, rr.Scalar(tensor))

                                case _:
                                    pass

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
                            rrb.Spatial2DView(
                                origin=f"/inputs/image/{camera}", name=camera
                            )
                            for camera in data[("inputs", "image")].keys()
                        ),
                        name="image",
                    ),
                    rrb.Tabs(
                        *(
                            rrb.TimeSeriesView(origin=k, name=k)
                            for k in ("inputs", "meta")
                        )
                    ),
                    name="inputs",
                ),
                rrb.Vertical(
                    rrb.Tabs(
                        *(
                            rrb.Vertical(
                                rrb.TimeSeriesView(
                                    origin=f"/predictions/{objective}",
                                    contents=[
                                        f"$origin/{result_key}/**"
                                        for result_key in (
                                            PredictionResultKey.PREDICTION,
                                            PredictionResultKey.GROUND_TRUTH,
                                        )
                                    ],
                                    name="Predictions",
                                ),
                                rrb.Tabs(
                                    rrb.BarChartView(
                                        origin=f"/predictions/{objective}/prediction_probs",
                                        name="Prediction Probs",
                                    ),
                                    rrb.TimeSeriesView(
                                        origin=f"/predictions/{objective}/score_l1",
                                        name="L1 score",
                                    ),
                                    rrb.TimeSeriesView(
                                        origin=f"/predictions/{objective}/score_logprob",
                                        name="Log(p) score",
                                    ),
                                ),
                                name=objective,
                            )
                            for objective in {
                                "/".join(always_iterable(module))
                                for (*module, _result_key, _modality, _name) in data[
                                    "predictions"
                                ].keys(True, True)
                            }
                        ),
                        name="predictions",
                    ),
                    name="outputs",
                ),
            )
        )
