import pickle
from typing import Any, override
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch import nn

from rmind.callbacks.freeze import ModuleFreezer
from rmind.callbacks.loggers import waypoints as waypoints_logger
from rmind.callbacks.loggers.foresight_metrics import WandbForesightMetricsLogger
from rmind.callbacks.loggers.waypoints import WandbWaypointsLogger
from rmind.callbacks.predict_metrics import PredictMetricsCallback
from rmind.callbacks.safe import SafeCallback
from rmind.components.objectives.base import ObjectivePredictionKey
from rmind.models.control_transformer import PredictionConfig
from rmind.utils.cluster import RuleBasedCluster

_RETRY_CALL_COUNT = 2


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


class FailingBatchCallback(SafeCallback):
    def __init__(
        self,
        *,
        hook_style: str = "direct",
        fail_gracefully: bool = True,
        disable_on_error: bool = False,
    ) -> None:
        super().__init__(
            fail_gracefully=fail_gracefully, disable_on_error=disable_on_error
        )
        self.calls = 0
        self.msg = "logger failed"
        if hook_style == "dynamic":
            hook = "on_train_batch_end"
            setattr(self, hook, self._safe_hook(hook, self._call))

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        self._safe_call(
            "on_train_batch_end",
            self._call,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
        )

    def _call(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,  # noqa: ARG002
        outputs: object,  # noqa: ARG002
        batch: object,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        self.calls += 1
        raise RuntimeError(self.msg)


@pytest.mark.parametrize("hook_style", ["direct", "dynamic"])
def test_safe_callback_swallows_batch_hook_error(
    trainer: pl.Trainer, module: ToyModule, hook_style: str
) -> None:
    callback = FailingBatchCallback(hook_style=hook_style, disable_on_error=True)

    callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
    callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=1)

    assert callback.calls == 1


def test_safe_callback_can_retry_after_error(
    trainer: pl.Trainer, module: ToyModule
) -> None:
    callback = FailingBatchCallback()

    callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
    callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=1)

    assert callback.calls == _RETRY_CALL_COUNT


def test_safe_callback_can_fail_loudly(trainer: pl.Trainer, module: ToyModule) -> None:
    callback = FailingBatchCallback(fail_gracefully=False)

    with pytest.raises(RuntimeError, match="logger failed"):
        callback.on_train_batch_end(
            trainer, module, outputs=None, batch=None, batch_idx=0
        )


def test_safe_callback_dynamic_hook_is_picklable() -> None:
    # Spawn-based DDP launchers pickle the trainer (and therefore callbacks);
    # the dynamically-installed hook must survive a pickle round-trip.
    callback = FailingBatchCallback(hook_style="dynamic")

    pickle.loads(pickle.dumps(callback))


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
    logged: dict[str, float] = {}
    mock_logger = MagicMock()
    mock_logger.log_metrics.side_effect = lambda m, **_: logged.update(m)

    cb = PredictMetricsCallback(prediction_config=PredictionConfig())

    with patch(
        "rmind.callbacks.predict_metrics._get_wandb_loggers", return_value=[mock_logger]
    ):
        _run_validation(cb, trainer, fake_module, [SPEED_BATCH])

    assert logged, "no metrics were logged"
    assert all(k.startswith("predict/") for k in logged)
    assert all(isinstance(v, float) for v in logged.values()), (
        "all logged values should be scalars"
    )


def test_predict_metrics_callback_logs_per_cluster(
    fake_module: FakePredictModule, trainer: pl.Trainer
) -> None:
    logged: dict[str, float] = {}
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


# ---------------------------------------------------------------------------
# WandbForesightMetricsLogger
# ---------------------------------------------------------------------------

_FORESIGHT_SRC = "cam_front_left"
_FORESIGHT_PRED_KEY = "embeddings_predict"
_FORESIGHT_TGT_KEY = "embeddings_target"
# Both metrics are bounded: R² ≤ 1, search accuracy ∈ [0, 1]. When pred truly
# predicts target, both sit near 1, so we require > 0.9.
_FORESIGHT_SIGNAL_MIN = 0.9
# When pred and target are independent, both collapse toward their floor:
#   - search accuracy → chance, ~1/n_patches (here 1/32 ≈ 0.03), since the
#     nearest target patch to any pred patch is essentially uniform-random.
#   - R² → ~0 in the population, but this probe is fit *in-sample* (no held-out
#     split), so it can only overfit *upward*, never below 0. The inflation is
#     ~n_features/n_samples (≈ d/n). The test feeds n ≫ d (n/d ≈ 190 here), so
#     even the overfit R² stays well under 0.1.
# 0.3 sits comfortably between the signal (~1) and noise (<0.1) regimes, so it
# separates the two cases with margin to spare against seed-to-seed jitter.
_FORESIGHT_CHANCE_MAX = 0.3


class _StubTrainer:
    sanity_checking = False
    is_global_zero = True
    current_epoch = 0
    global_step = 0


class _LoggingModule:
    """Captures ``pl_module.log`` calls into a dict."""

    def __init__(self) -> None:
        self.logged: dict[str, float] = {}

    def log(self, name: str, value: float, **_: object) -> None:
        self.logged[name] = float(value)


def _make_foresight_callback(**overrides: object) -> WandbForesightMetricsLogger:
    kwargs: dict[str, Any] = {
        "key": "foresight",
        "image_sources": {_FORESIGHT_SRC: [_FORESIGHT_SRC]},
        "embeddings_predict": [_FORESIGHT_PRED_KEY],
        "embeddings_target": [_FORESIGHT_TGT_KEY],
        "hot_quantile": 0.25,
        # Surface impl errors instead of swallowing them in these tests.
        "fail_gracefully": False,
    }
    kwargs.update(overrides)
    return WandbForesightMetricsLogger(**kwargs)


def _foresight_outputs(pred_last: torch.Tensor, tgt_last: torch.Tensor) -> dict:
    """Wrap (B, P, d) embeddings into the (B, S=2, P, d) outputs the callback reads.

    ``tgt_prev`` (the T-2 slot) is random so motion stratification has a
    well-defined ranking; only the T-1 slot carries the pred→target signal.
    """
    b, p, d = pred_last.shape
    pred_prev = torch.randn(b, p, d)
    tgt_prev = torch.randn(b, p, d)
    pred_seq = torch.stack([pred_prev, pred_last], dim=1)
    tgt_seq = torch.stack([tgt_prev, tgt_last], dim=1)
    return {
        _FORESIGHT_PRED_KEY: {_FORESIGHT_SRC: pred_seq},
        _FORESIGHT_TGT_KEY: {_FORESIGHT_SRC: tgt_seq},
    }


def _run_foresight(
    cb: WandbForesightMetricsLogger, batches: list[dict]
) -> dict[str, float]:
    trainer = _StubTrainer()
    module = _LoggingModule()
    with patch(
        "rmind.callbacks.loggers.foresight_metrics._get_wandb_loggers",
        return_value=[object()],
    ):
        cb.on_validation_epoch_start(trainer, module)  # ty:ignore[invalid-argument-type]
        for i, batch in enumerate(batches):
            cb.on_validation_batch_end(trainer, module, batch, None, i)  # ty:ignore[invalid-argument-type]
        cb.on_validation_epoch_end(trainer, module)  # ty:ignore[invalid-argument-type]
    return module.logged


def test_foresight_logger_recovers_signal() -> None:
    """Identity pred→target ⇒ R² and search accuracy both ≈ 1."""
    torch.manual_seed(0)
    b, p, d = 8, 32, 4
    # target == pred at T-1: a perfect linear probe and a perfect cross-patch
    # search both fall out.
    batches = [
        _foresight_outputs(emb := torch.randn(b, p, d), emb.clone()) for _ in range(12)
    ]

    logged = _run_foresight(_make_foresight_callback(), batches)

    assert logged[f"foresight/{_FORESIGHT_SRC}/r2_hot"] > _FORESIGHT_SIGNAL_MIN
    assert logged[f"foresight/{_FORESIGHT_SRC}/search_hit_hot"] > _FORESIGHT_SIGNAL_MIN


def test_foresight_logger_rejects_noise() -> None:
    """Independent pred/target ⇒ R² (in-sample, n/d≫1) and search both ≈ chance."""
    torch.manual_seed(0)
    b, p, d = 8, 32, 4
    batches = [
        _foresight_outputs(torch.randn(b, p, d), torch.randn(b, p, d))
        for _ in range(12)
    ]

    logged = _run_foresight(_make_foresight_callback(), batches)

    assert logged[f"foresight/{_FORESIGHT_SRC}/r2_hot"] < _FORESIGHT_CHANCE_MAX
    assert logged[f"foresight/{_FORESIGHT_SRC}/search_hit_hot"] < _FORESIGHT_CHANCE_MAX


def test_foresight_logger_skips_nan_when_no_data() -> None:
    """No valid batches ⇒ nothing logged (NaN epochs are skipped, not logged)."""
    logged = _run_foresight(_make_foresight_callback(), [])

    assert logged == {}
