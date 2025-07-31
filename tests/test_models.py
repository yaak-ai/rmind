from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
from hydra import compose, initialize
from hydra.utils import instantiate
from pytest_lazy_fixtures import lf
from rbyte.batch import Batch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from rmind.callbacks.logit_bias import LogitBiasSetter
from rmind.datamodules import GenericDataModule
from rmind.models.control_transformer import ControlTransformer

CONFIG_PATH = "../config"


@pytest.fixture
def train_dataset(batch: Batch) -> TensorDict:
    return batch.to_tensordict()


@pytest.fixture
def val_dataset(batch: Batch) -> TensorDict:
    return batch.to_tensordict()


@pytest.fixture
def train_dataloader(train_dataset: TensorDict) -> DataLoader[Any]:
    return DataLoader(train_dataset, batch_size=1, collate_fn=TensorDict.to_dict)  # pyright: ignore[reportArgumentType]


@pytest.fixture
def val_dataloader(val_dataset: TensorDict) -> DataLoader[Any]:
    return DataLoader(val_dataset, batch_size=1, collate_fn=TensorDict.to_dict)  # pyright: ignore[reportArgumentType]


@pytest.fixture
def datamodule(
    train_dataloader: DataLoader[Any], val_dataloader: DataLoader[Any]
) -> pl.LightningDataModule:
    return GenericDataModule(train=train_dataloader, val=val_dataloader)


@pytest.fixture
def trainer() -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu",
        fast_dev_run=1,
        callbacks=[LogitBiasSetter()],
        precision="bf16-mixed",
    )


@pytest.fixture
def model_yaak_control_transformer_raw() -> ControlTransformer:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "model/yaak/control_transformer/raw",
            overrides=["+num_heads=4", "+num_layers=8", "+embedding_dim=512"],
        )

    return instantiate(cfg.model.yaak.control_transformer)


@pytest.mark.parametrize("model", [lf("model_yaak_control_transformer_raw")])
def test_resume_from_checkpoint(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    tmp_path: Path,
) -> None:
    trainer.fit(model, datamodule=datamodule)

    ckpt_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt_path)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
