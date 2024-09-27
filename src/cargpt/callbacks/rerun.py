from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, override

import pytorch_lightning as pl
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from funcy import once_per
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict

from cargpt.components.episode import Modality
from cargpt.components.objectives.base import PredictionResultKey

if TYPE_CHECKING:
    from rerun.blueprint.api import BlueprintLike

try:
    from rbyte.batch import Batch
except ImportError:
    from typing import Any

    Batch = Any


class RerunPredictionWriter(BasePredictionWriter):
    @override
    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._blueprint: BlueprintLike | None = None

        # HACK: a lil monkeypatching never hurt nobody
        rr.log_once_per_entity_path = once_per("entity_path")(rr.log)  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def on_predict_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._blueprint = None

        if hasattr(rr, "log_once_per_entity_path"):
            delattr(rr, "log_once_per_entity_path")

    @cache  # noqa: B019
    def _get_recording(self, *, application_id: str) -> rr.RecordingStream:
        recording = rr.new_recording(
            application_id=application_id, spawn=True, make_default=True
        )

        if self._blueprint is not None:
            rr.send_blueprint(self._blueprint, recording=recording)

        return recording

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

        if self._blueprint is None:
            self._blueprint = self._build_blueprint(data)

        for input_id, sample in zip(
            data.get(k := ("batch", "meta", "input_id")).data,
            data.exclude(k),
            strict=True,
        ):
            with self._get_recording(application_id=input_id):
                rr.set_time_sequence(
                    "/".join(k := ("batch", "meta", "sample_idx")), sample.get(k).item()
                )

                for timestep, elem in enumerate(
                    sample.exclude(("batch", "meta")).auto_batch_size_(2)
                ):
                    # log element time keys first
                    time_keys = []
                    for k, v in elem.items(True, True):
                        path = "/".join(k)

                        match k[-1].rsplit(".", 1)[-1]:
                            case "frame_idx":
                                rr.set_time_sequence(path, v.item())
                                time_keys.append(k)

                            case "time_stamp":
                                rr.set_time_nanos(path, v.item() * 1000)
                                time_keys.append(k)

                            case _:
                                pass

                    for k, v in elem.exclude(*time_keys).items(True, True):
                        path = "/".join(k)
                        style = None

                        match k:
                            case ("batch", "frame", *_) | (
                                "inputs",
                                Modality.IMAGE,
                                *_,
                            ):
                                match v.shape:
                                    case (3, _h, _w):
                                        v = rearrange(v, "c h w -> h w c")  # noqa: PLW2901

                                    case _:
                                        pass

                                entity = rr.Image(v)

                            case ("batch", "table", name) | (
                                "inputs",
                                Modality.CONTINUOUS | Modality.DISCRETE,
                                name,
                            ):
                                style = rr.SeriesLine(name=name)
                                entity = rr.Scalar(v)

                            case (
                                "predictions",
                                _objective,
                                result_key,
                                (Modality.CONTINUOUS | Modality.DISCRETE),
                                name,
                            ):
                                if v.isnan():
                                    continue
                                entity = rr.Scalar(v)

                                match result_key:
                                    case PredictionResultKey.GROUND_TRUTH:
                                        style = rr.SeriesLine(name=f"gt/{name}")

                                    case PredictionResultKey.PREDICTION:
                                        style = rr.SeriesPoint(
                                            name=f"pred/{name}",
                                            marker="cross",
                                            marker_size=4,
                                        )
                                    case PredictionResultKey.PREDICTION_PROBS:
                                        style = rr.SeriesPoint(
                                            name=f"prediction_probs/{name}",
                                            marker="plus",
                                            marker_size=4,
                                        )

                                    case PredictionResultKey.SCORE_LOGPROB:
                                        style = rr.SeriesPoint(
                                            name=f"logp/{name}",
                                            marker="diamond",
                                            marker_size=4,
                                        )

                                    case PredictionResultKey.SCORE_L1:
                                        style = rr.SeriesPoint(
                                            name=f"l1/{name}",
                                            marker="circle",
                                            marker_size=4,
                                        )

                                    case _:
                                        raise NotImplementedError(k)

                            case (
                                "predictions",
                                _objective,
                                PredictionResultKey.ATTENTION,
                                Modality.SPECIAL,
                                _token_from,
                                modality_to,
                                _token_to,
                            ):
                                attn = v[timestep]
                                match modality_to:
                                    case Modality.IMAGE:
                                        attn = attn.view(10, 18)

                                    case _:
                                        pass

                                entity = rr.Tensor(attn)

                            case _:
                                raise NotImplementedError(k)

                        if style is not None:
                            rr.log_once_per_entity_path(path, style, static=True)  # pyright: ignore[reportAttributeAccessIssue]

                        rr.log(path, entity)

    @classmethod
    def _build_blueprint(cls, data: TensorDict) -> rrb.Blueprint:
        return rrb.Blueprint(
            rrb.Horizontal(
                name="inference",
                contents=[
                    rrb.Vertical(
                        name="input",
                        contents=[
                            rrb.Tabs(
                                name="stages",
                                contents=[
                                    rrb.Vertical(
                                        name="raw",
                                        contents=[
                                            rrb.Tabs(
                                                name="frame",
                                                contents=[
                                                    rrb.Spatial2DView(
                                                        origin="/".join(k), name=k[-1]
                                                    )
                                                    for k in data.select((  # pyright: ignore[reportArgumentType]
                                                        "batch",
                                                        "frame",
                                                    )).keys(True, True)
                                                ],
                                            ),
                                            rrb.TimeSeriesView(
                                                origin="/batch/table", name="table"
                                            ),
                                        ],
                                    ),
                                    rrb.Vertical(
                                        name="transformed",
                                        contents=[
                                            rrb.Tabs(
                                                name="image",
                                                contents=[
                                                    rrb.Spatial2DView(
                                                        origin="/".join(k), name=k[-1]
                                                    )
                                                    for k in data.select((  # pyright: ignore[reportArgumentType]
                                                        "inputs",
                                                        Modality.IMAGE,
                                                    )).keys(True, True)
                                                ],
                                            ),
                                            rrb.TimeSeriesView(
                                                name="scalar",
                                                origin="/inputs",
                                                contents=[
                                                    f"$origin/{Modality.CONTINUOUS}/**",
                                                    f"$origin/{Modality.DISCRETE}/**",
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    rrb.Vertical(
                        name="output",
                        contents=[
                            rrb.Tabs(
                                name="objectives",
                                contents=[
                                    rrb.Vertical(
                                        name=objective,
                                        contents=[
                                            rrb.Tabs(
                                                name="scores",
                                                contents=[
                                                    rrb.TimeSeriesView(
                                                        origin=f"/predictions/{objective}/prediction_probs",
                                                        name="prediction_probs",
                                                    ),
                                                    rrb.TimeSeriesView(
                                                        origin=f"/predictions/{objective}/score_l1",
                                                        name="score_l1",
                                                    ),
                                                    rrb.TimeSeriesView(
                                                        origin=f"/predictions/{objective}/score_logprob",
                                                        name="score_logprob",
                                                    ),
                                                ],
                                            ),
                                            rrb.TimeSeriesView(
                                                name="predictions",
                                                origin=f"/predictions/{objective}",
                                                contents=[
                                                    f"$origin/{PredictionResultKey.PREDICTION}/**",
                                                    f"$origin/{PredictionResultKey.GROUND_TRUTH}/**",
                                                ],
                                            ),
                                        ],
                                    )
                                    for objective in data["predictions"].keys()  # pyright: ignore[reportAttributeAccessIssue]
                                ],
                            )
                        ],
                    ),
                ],
            )
        )
