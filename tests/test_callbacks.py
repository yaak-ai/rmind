from typing import override

import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from rmind.callbacks.freeze import ModuleFreezer
from rmind.callbacks.loggers import waypoints as waypoints_logger
from rmind.callbacks.loggers.waypoints import WandbWaypointsLogger


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
