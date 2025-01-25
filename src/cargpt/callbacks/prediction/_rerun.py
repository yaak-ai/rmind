from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import Any, ClassVar, final, override

import more_itertools as mit
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from loguru import logger
from pytorch_lightning.callbacks import BasePredictionWriter
from rbyte.batch import Batch
from rerun import (
    AsComponents,
    ChannelDatatype,
    ColorModel,
    ComponentBatchLike,
    ComponentColumn,
    Image,
    SeriesLine,
    SeriesPoint,
    TimeNanosColumn,
    TimeSequenceColumn,
)
from rerun._baseclasses import ComponentBatchMixin
from rerun.components import ImageBufferBatch, ImageFormat, MarkerShape, ScalarBatch
from tensordict import TensorDict

from cargpt.common import CameraName
from cargpt.components.episode import Modality
from cargpt.components.objectives.base import PredictionResultKey


@final
class RerunPredictionWriter(BasePredictionWriter):
    """
    Example:

    ```yaml
    _target_: cargpt.callbacks.RerunPredictionWriter
    write_interval: batch
    ```
    """

    MARKERS: ClassVar[Mapping[PredictionResultKey, MarkerShape]] = {
        PredictionResultKey.PREDICTION: MarkerShape.Cross,
        PredictionResultKey.PREDICTION_STD: MarkerShape.Asterisk,
        PredictionResultKey.SCORE_LOGPROB: MarkerShape.Diamond,
        PredictionResultKey.SCORE_L1: MarkerShape.Circle,
    }

    @override
    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._blueprint = None

    @override
    def on_predict_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._blueprint = None

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
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        data = (
            prediction.select("input", "predictions")
            .auto_batch_size_(1)
            .update({"batch": batch.to_tensordict().auto_batch_size_(1)})  # pyright: ignore[reportAttributeAccessIssue]
            .cpu()
        )

        time_keys = {TimeSequenceColumn: [], TimeNanosColumn: []}
        component_keys = []
        for k in data.keys(True, True):
            match k:
                case ("batch", "meta", *_):
                    pass

                case (*_, last) if last.endswith("idx"):
                    time_keys[TimeSequenceColumn].append(k)

                case (*_, last) if last.endswith("time_stamp"):
                    time_keys[TimeNanosColumn].append(k)

                case _:
                    component_keys.append(k)

        if self._blueprint is None:
            self._blueprint = self._build_blueprint(data)

        for input_id, elem in zip(data["batch", "meta", "input_id"], data, strict=True):
            with self._get_recording(application_id=input_id):
                rr.set_time_sequence(
                    "/".join(k := ("batch", "meta", "sample_idx")), elem.get(k).item()
                )

                times = list(
                    mit.flatten([
                        [
                            column("/".join(k), v.numpy())
                            for k, v in elem.select(*keys).items(True, True)
                        ]
                        for column, keys in time_keys.items()
                    ])
                )

                for k, v in elem.select(*component_keys).items(True, True):
                    entity_path = "/".join(k)
                    try:
                        static, components = self._build_entities(k, v.numpy())
                    except NotImplementedError as exc:
                        logger.warning("not implemented", exc=exc)
                    else:
                        rr.log(entity_path, static, static=True)
                        rr.send_columns(entity_path, times, components)

    @classmethod
    def _build_entities(
        cls, key: tuple[str, ...], array: npt.NDArray[Any]
    ) -> tuple[
        AsComponents | Iterable[ComponentBatchLike],
        Iterable[ComponentColumn | ComponentBatchMixin],
    ]:
        # TODO: configurize?
        match key:
            case (
                "batch",
                "data",
                CameraName.cam_front_center
                | CameraName.cam_front_left
                | CameraName.cam_front_right
                | CameraName.cam_left_forward
                | CameraName.cam_right_forward
                | CameraName.cam_left_backward
                | CameraName.cam_right_backward
                | CameraName.cam_rear,
            ) | ("input", Modality.IMAGE, *_):
                match array.shape:
                    case (*_, height, width, 3):
                        pass

                    case (*_, 3, height, width):
                        array = rearrange(array, "... c h w -> ... h w c")

                    case _:
                        raise NotImplementedError(key)

                return [
                    Image.indicator(),
                    ImageFormat(
                        height=height,
                        width=width,
                        color_model=ColorModel.RGB,
                        channel_datatype=ChannelDatatype.from_np_dtype(array.dtype),
                    ),
                ], [
                    ImageBufferBatch(
                        rearrange(array, "... h w c -> (...) (h w c)").view(np.uint8)
                    )
                ]

            case ("batch", "data", name) | (
                "input",
                Modality.CONTINUOUS | Modality.DISCRETE,
                name,
            ):
                return SeriesLine(name=name), [ScalarBatch(array)]

            case (
                "predictions",
                _objective,
                result_key,
                Modality.CONTINUOUS | Modality.DISCRETE,
                name,
            ):
                lengths = [np.prod(array.shape[1:])] * array.shape[0]

                match result_key:
                    case PredictionResultKey.GROUND_TRUTH:
                        return SeriesLine(name=f"{result_key}/{name}"), [
                            ScalarBatch(array).partition(lengths)
                        ]

                    case (
                        PredictionResultKey.PREDICTION
                        | PredictionResultKey.PREDICTION_STD
                        | PredictionResultKey.SCORE_LOGPROB
                        | PredictionResultKey.SCORE_L1
                    ):
                        return SeriesPoint(
                            name=f"{result_key}/{name}",
                            marker=cls.MARKERS[result_key],
                            marker_size=4,
                        ), [ScalarBatch(array).partition(lengths)]

                    # TODO: https://github.com/rerun-io/rerun/issues/8361

                    case _:
                        raise NotImplementedError(key)

            case _:
                raise NotImplementedError(key)

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
                                contents=[
                                    rrb.Vertical(
                                        name="batch",
                                        contents=[
                                            rrb.Tabs(
                                                contents=[
                                                    rrb.Spatial2DView(
                                                        origin="/".join(k), name=name
                                                    )
                                                    for k in data.select((
                                                        "batch",
                                                        "data",
                                                    )).keys(True, True)
                                                    if (name := k[-1]) in CameraName
                                                ]
                                            ),
                                            rrb.TimeSeriesView(origin="/batch/data"),
                                        ],
                                    ),
                                    rrb.Vertical(
                                        name="input",
                                        contents=[
                                            rrb.Tabs(
                                                name="image",
                                                contents=[
                                                    rrb.Spatial2DView(
                                                        origin="/".join(k), name=k[-1]
                                                    )
                                                    for k in data.select((
                                                        "input",
                                                        Modality.IMAGE,
                                                    )).keys(True, True)
                                                ],
                                            ),
                                            rrb.TimeSeriesView(
                                                name="scalar",
                                                origin="/input",
                                                contents=[
                                                    f"$origin/{Modality.CONTINUOUS}/**",
                                                    f"$origin/{Modality.DISCRETE}/**",
                                                ],
                                            ),
                                        ],
                                    ),
                                ]
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
                                                    rrb.BarChartView(
                                                        origin=f"/predictions/{objective}/{(name := PredictionResultKey.PREDICTION_PROBS)}",
                                                        name=name,
                                                    ),
                                                    *(
                                                        rrb.TimeSeriesView(
                                                            origin=f"/predictions/{objective}/{name}",
                                                            name=name,
                                                        )
                                                        for name in (
                                                            PredictionResultKey.PREDICTION_STD,
                                                            PredictionResultKey.SCORE_L1,
                                                            PredictionResultKey.SCORE_LOGPROB,
                                                        )
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
                                    for objective in data["predictions"].keys()
                                ],
                            )
                        ],
                    ),
                ],
            )
        )
