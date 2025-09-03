from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
from hydra import compose, initialize
from hydra.utils import instantiate
from pytest_lazy_fixtures import lf
from rbyte.batch import Batch
from tensordict import TensorDict
from torch.nn import LayerNorm, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rmind.callbacks.logit_bias import LogitBiasSetter
from rmind.components.containers import ModuleDict
from rmind.components.nn import Embedding
from rmind.components.objectives import (
    ForwardDynamicsPredictionObjective,
    InverseDynamicsPredictionObjective,
    MemoryExtractionObjective,
    PolicyObjective,
    RandomMaskedHindsightControlObjective,
)
from rmind.config import HydraConfig
from rmind.datamodules import GenericDataModule
from rmind.models.control_transformer import ControlTransformer
from tests.conftest import (
    BRAKE_PEDAL_BINS,
    EMBEDDING_DIM,
    GAS_PEDAL_BINS,
    SPEED_BINS,
    STEERING_ANGLE_BINS,
)

CONFIG_PATH = "../config"


@pytest.fixture
def train_dataset(batch: Batch) -> TensorDict:
    return batch.to_tensordict()


@pytest.fixture
def val_dataset(batch: Batch) -> TensorDict:
    return batch.to_tensordict()


@pytest.fixture
def predict_dataset(batch: Batch) -> TensorDict:
    return batch.to_tensordict()


@pytest.fixture
def train_dataloader(train_dataset: TensorDict) -> DataLoader[Any]:
    return DataLoader(train_dataset, batch_size=1, collate_fn=TensorDict.to_dict)  # pyright: ignore[reportArgumentType]


@pytest.fixture
def val_dataloader(val_dataset: TensorDict) -> DataLoader[Any]:
    return DataLoader(val_dataset, batch_size=1, collate_fn=TensorDict.to_dict)  # pyright: ignore[reportArgumentType]


@pytest.fixture
def predict_dataloader(predict_dataset: TensorDict) -> DataLoader[Any]:
    return DataLoader(predict_dataset, batch_size=1, collate_fn=TensorDict.to_dict)  # pyright: ignore[reportArgumentType]


@pytest.fixture
def datamodule(
    train_dataloader: DataLoader[Any],
    val_dataloader: DataLoader[Any],
    predict_dataloader: DataLoader[Any],
) -> pl.LightningDataModule:
    return GenericDataModule(
        train=train_dataloader, val=val_dataloader, predict=predict_dataloader
    )


@pytest.fixture
def trainer() -> pl.Trainer:
    return pl.Trainer(
        devices=1,
        fast_dev_run=1,
        callbacks=[LogitBiasSetter()],
        precision="bf16-mixed",
        enable_progress_bar=False,
    )


@pytest.fixture
def objectives(
    inverse_dynamics_prediction_objective: InverseDynamicsPredictionObjective,
    forward_dynamics_prediction_objective: ForwardDynamicsPredictionObjective,
    random_masked_hindsight_control_objective: RandomMaskedHindsightControlObjective,
    memory_extraction_objective: MemoryExtractionObjective,
    policy_objective: PolicyObjective,
) -> ModuleDict:
    return ModuleDict({
        "inverse_dynamics": inverse_dynamics_prediction_objective,
        "forward_dynamics": forward_dynamics_prediction_objective,
        "random_masked_hindsight_control": random_masked_hindsight_control_objective,
        "memory_extraction": memory_extraction_objective,
        "policy_objective": policy_objective,
    })


@pytest.fixture
def optimizer() -> HydraConfig[Optimizer]:
    return HydraConfig[Optimizer](
        target="rmind.components.optimizers.SelectiveAdamW",
        lr=1e-5,  # pyright: ignore[reportCallIssue]
        betas=[0.9, 0.95],  # pyright: ignore[reportCallIssue]
        weight_decay=0.1,  # pyright: ignore[reportCallIssue]
        weight_decay_module_blacklist=[Embedding, LayerNorm],  # pyright: ignore[reportCallIssue]
    )


@pytest.fixture
def control_transformer(
    episode_builder: Module, objectives: ModuleDict, optimizer: HydraConfig[Optimizer]
) -> ControlTransformer:
    return ControlTransformer(
        episode_builder=episode_builder, objectives=objectives, optimizer=optimizer
    )


@pytest.fixture
def model_yaak_control_transformer_raw() -> ControlTransformer:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "model/yaak/control_transformer/raw",
            overrides=[
                "+num_heads=1",
                "+num_layers=1",
                f"+embedding_dim={EMBEDDING_DIM}",
                f"+speed_bins={SPEED_BINS}",
                f"+gas_pedal_bins={GAS_PEDAL_BINS}",
                f"+brake_pedal_bins={BRAKE_PEDAL_BINS}",
                f"+steering_angle_bins={STEERING_ANGLE_BINS}",
            ],
        )

    return instantiate(cfg.model.yaak.control_transformer)


@pytest.mark.parametrize("model", [lf("control_transformer")])
def test_fit(
    trainer: pl.Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule
) -> None:
    trainer.fit(model, datamodule=datamodule)


@pytest.mark.parametrize("model", [lf("control_transformer")])
def test_predict(
    trainer: pl.Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule
) -> None:
    trainer.predict(model, datamodule=datamodule, return_predictions=False)  # pyright: ignore[reportUnusedCallResult]


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
