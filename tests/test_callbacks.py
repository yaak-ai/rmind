from typing import override
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch import Tensor, nn

from rmind.callbacks.freeze import ModuleFreezer
from rmind.callbacks.loggers import waypoints as waypoints_logger
from rmind.callbacks.loggers.waypoints import WandbWaypointsLogger
from rmind.callbacks.predict_metrics import PredictMetricsCallback
from rmind.components.objectives.base import ObjectivePredictionKey
from rmind.models.control_transformer import PredictionConfig
from rmind.utils.cluster import RuleBasedCluster


class ToyModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        self.decoder = nn.Linear(4, 2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@pytest.fixture
def module() -> ToyModule:
    return ToyModule()


@pytest.fixture(scope="module")
def trainer() -> pl.Trainer:
    return pl.Trainer(
        logger=False, enable_progress_bar=False, enable_model_summary=False
    )


def test_freeze_by_path(trainer: pl.Trainer, module: ToyModule) -> None:
    ModuleFreezer(paths={"encoder"}).setup(trainer, module, "fit")

    assert not any(p.requires_grad for p in module.encoder.parameters())
    assert not module.encoder.training
    assert all(p.requires_grad for p in module.decoder.parameters())
    assert module.decoder.training


def test_freeze_by_type(trainer: pl.Trainer, module: ToyModule) -> None:
    ModuleFreezer(types={"torch.nn.Linear"}).setup(trainer, module, "fit")  # ty:ignore[invalid-argument-type]

    for m in module.modules():
        if isinstance(m, nn.Linear):
            assert not any(p.requires_grad for p in m.parameters())
            assert not m.training


def test_missing_path_raises(trainer: pl.Trainer, module: ToyModule) -> None:
    cb = ModuleFreezer(paths={"does_not_exist"})
    with pytest.raises(AttributeError):
        cb.setup(trainer, module, "fit")


def test_waypoints_map_handles_basemap_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_http_error(*_args: object, **_kwargs: object) -> None:
        msg = "missing tile"
        raise waypoints_logger.requests.exceptions.HTTPError(msg)

    monkeypatch.setattr(waypoints_logger.ctx, "add_basemap", raise_http_error)

    image = WandbWaypointsLogger._plot_waypoints_on_map(  # noqa: SLF001
        wpts_xy=torch.tensor([[392000.0, 5810000.0], [392010.0, 5810010.0]]),
        crs="EPSG:25832",
    )

    assert image is not None


SPEED_BATCH = {
    "data": {
        "meta/VehicleMotion/speed": torch.tensor([
            [10.0, 20.0],
            [50.0, 60.0],
            [100.0, 110.0],
        ])
    }
}


# ---------------------------------------------------------------------------
# PredictMetricsCallback — fixtures
# ---------------------------------------------------------------------------


class FakePredictModule(pl.LightningModule):
    """Minimal LightningModule whose predict_step returns a fixed TensorDict."""

    def __init__(self) -> None:
        super().__init__()
        self.prediction_config = PredictionConfig()
        self._dummy = nn.Linear(1, 1)

    @override
    def predict_step(self, batch: dict) -> TensorDict:  # type: ignore[override]
        b = batch["data"]["meta/VehicleMotion/speed"].shape[0]
        return TensorDict({"score_l1": torch.rand(b)}, batch_size=[b])


@pytest.fixture
def fake_module() -> FakePredictModule:
    return FakePredictModule()


def _run_validation(
    cb: PredictMetricsCallback,
    trainer: pl.Trainer,
    module: FakePredictModule,
    batches: list[dict],
) -> None:
    cb.on_validation_epoch_start(trainer, module)
    for i, batch in enumerate(batches):
        cb.on_validation_batch_end(
            trainer, module, outputs=None, batch=batch, batch_idx=i
        )
    cb.on_validation_epoch_end(trainer, module)


# ---------------------------------------------------------------------------
# PredictMetricsCallback — tests
# ---------------------------------------------------------------------------


def test_predict_metrics_callback_logs_predict_prefix(
    fake_module: FakePredictModule, trainer: pl.Trainer
) -> None:
    logged: dict[str, Tensor] = {}
    mock_logger = MagicMock()
    mock_logger.log_metrics.side_effect = lambda m, **_: logged.update(m)

    cb = PredictMetricsCallback(prediction_config=PredictionConfig())

    with patch(
        "rmind.callbacks.predict_metrics._get_wandb_loggers", return_value=[mock_logger]
    ):
        _run_validation(cb, trainer, fake_module, [SPEED_BATCH])

    assert logged, "no metrics were logged"
    assert all(k.startswith("predict/") for k in logged)
    assert all(v.ndim == 0 for v in logged.values()), (
        "all logged values should be scalars"
    )


def test_predict_metrics_callback_logs_per_cluster(
    fake_module: FakePredictModule, trainer: pl.Trainer
) -> None:
    logged: dict[str, Tensor] = {}
    mock_logger = MagicMock()
    mock_logger.log_metrics.side_effect = lambda m, **_: logged.update(m)

    cb = PredictMetricsCallback(
        prediction_config=PredictionConfig(),
        cluster_fn=RuleBasedCluster(
            fields={"speed": {"key": "meta/VehicleMotion/speed", "reduce": "last"}},
            rules=[
                {"name": "slow", "when": {"speed": {"lt": 30.0}}},
                {"name": "urban", "when": {"speed": {"ge": 30.0, "lt": 90.0}}},
            ],
            default="fast",
        ),
    )

    with patch(
        "rmind.callbacks.predict_metrics._get_wandb_loggers", return_value=[mock_logger]
    ):
        _run_validation(cb, trainer, fake_module, [SPEED_BATCH])

    assert "predict/score_l1" in logged
    assert "predict/slow/score_l1" in logged
    assert "predict/urban/score_l1" in logged
    assert "predict/fast/score_l1" in logged


def test_predict_metrics_callback_swaps_prediction_config(
    fake_module: FakePredictModule, trainer: pl.Trainer
) -> None:
    original_config = PredictionConfig()
    fake_module.prediction_config = original_config
    seen_configs: list[PredictionConfig] = []

    original = fake_module.predict_step

    def capturing_predict_step(batch: dict) -> TensorDict:
        seen_configs.append(fake_module.prediction_config)
        return original(batch)

    fake_module.predict_step = capturing_predict_step  # type: ignore[method-assign]  # ty:ignore[invalid-assignment]

    override_config = PredictionConfig(objectives={ObjectivePredictionKey.SCORE_L1})
    cb = PredictMetricsCallback(prediction_config=override_config)

    with patch("rmind.callbacks.predict_metrics._get_wandb_loggers", return_value=[]):
        _run_validation(cb, trainer, fake_module, [SPEED_BATCH])

    assert seen_configs == [override_config], (
        "prediction_config should be swapped during loop"
    )
    assert fake_module.prediction_config is original_config, (
        "prediction_config should be restored after"
    )
