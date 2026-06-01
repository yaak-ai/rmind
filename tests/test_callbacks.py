import pickle
from typing import override

import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from rmind.callbacks.freeze import ModuleFreezer
from rmind.callbacks.loggers import waypoints as waypoints_logger
from rmind.callbacks.loggers.waypoints import WandbWaypointsLogger
from rmind.callbacks.safe import SafeCallback

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
        disable_on_error: bool = True,
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
    callback = FailingBatchCallback(hook_style=hook_style)

    callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
    callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=1)

    assert callback.calls == 1


def test_safe_callback_can_retry_after_error(
    trainer: pl.Trainer, module: ToyModule
) -> None:
    callback = FailingBatchCallback(disable_on_error=False)

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
