from typing import override

import pytest
import pytorch_lightning as pl
import structlog
import torch
from structlog.testing import capture_logs
from torch import nn

from rmind.callbacks.freeze import FreezeModules


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


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    structlog.reset_defaults()


def test_freeze_by_path(trainer: pl.Trainer, module: ToyModule) -> None:
    FreezeModules(paths=["encoder"]).setup(trainer, module, "fit")

    assert not any(p.requires_grad for p in module.encoder.parameters())
    assert not module.encoder.training
    assert all(p.requires_grad for p in module.decoder.parameters())
    assert module.decoder.training


def test_freeze_by_type(trainer: pl.Trainer, module: ToyModule) -> None:
    FreezeModules(types=["torch.nn.Linear"]).setup(trainer, module, "fit")

    for m in module.modules():
        if isinstance(m, nn.Linear):
            assert not any(p.requires_grad for p in m.parameters())
            assert not m.training


def test_missing_path_logs_error_and_skips(
    trainer: pl.Trainer, module: ToyModule
) -> None:
    cb = FreezeModules(paths=["does_not_exist", "encoder"])
    with capture_logs() as logs:
        cb.setup(trainer, module, "fit")

    assert any(e.get("log_level") == "error" for e in logs)
    assert not any(p.requires_grad for p in module.encoder.parameters())
