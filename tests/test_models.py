from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from optree import tree_all, tree_map
from pytest_lazy_fixtures import lf
from rbyte.types import Batch
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

if TYPE_CHECKING:
    from tests.conftest import EmbeddingDims, NumBins

CONFIG_PATH = "../config"


@pytest.fixture
def train_dataset(batch: Batch) -> TensorDict:
    return batch.to_tensordict()  # ty:ignore[invalid-return-type]


@pytest.fixture
def val_dataset(batch: Batch) -> TensorDict:
    return batch.to_tensordict()  # ty:ignore[invalid-return-type]


@pytest.fixture
def predict_dataset(batch: Batch) -> TensorDict:
    return batch.to_tensordict()  # ty:ignore[invalid-return-type]


@pytest.fixture
def train_dataloader(train_dataset: TensorDict) -> DataLoader[Any]:
    return DataLoader(train_dataset, batch_size=1, collate_fn=TensorDict.to_dict)  # ty:ignore[invalid-argument-type]


@pytest.fixture
def val_dataloader(val_dataset: TensorDict) -> DataLoader[Any]:
    return DataLoader(val_dataset, batch_size=1, collate_fn=TensorDict.to_dict)  # ty:ignore[invalid-argument-type]


@pytest.fixture
def predict_dataloader(predict_dataset: TensorDict) -> DataLoader[Any]:
    return DataLoader(predict_dataset, batch_size=1, collate_fn=TensorDict.to_dict)  # ty:ignore[invalid-argument-type]


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
def trainer(device: torch.device) -> pl.Trainer:
    return pl.Trainer(
        accelerator=device.type,
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
        target="rmind.components.optimizers.SelectiveAdamW",  # ty:ignore[invalid-argument-type]
        lr=1e-5,  # ty:ignore[unknown-argument]
        betas=[0.9, 0.95],  # ty:ignore[unknown-argument]
        weight_decay=0.1,  # ty:ignore[unknown-argument]
        weight_decay_module_blacklist=[Embedding, LayerNorm],  # ty:ignore[unknown-argument]
    )


@pytest.fixture
def control_transformer(
    episode_builder: Module, objectives: ModuleDict, optimizer: HydraConfig[Optimizer]
) -> ControlTransformer:
    return ControlTransformer(
        episode_builder=episode_builder, objectives=objectives, optimizer=optimizer
    )


@pytest.fixture
def model_yaak_control_transformer_raw(
    request: pytest.FixtureRequest,
) -> ControlTransformer:
    embedding_dims: EmbeddingDims = request.getfixturevalue("embedding_dims")
    num_bins: NumBins = request.getfixturevalue("num_bins")

    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "model/yaak/control_transformer/raw",
            overrides=[
                "+num_heads=1",
                "+num_layers=1",
                f"+encoder_embedding_dim={embedding_dims.encoder}",
                f"+image_embedding_dim={embedding_dims.image}",
                f"+speed_bins={num_bins.speed}",
                f"+gas_pedal_bins={num_bins.gas_pedal}",
                f"+brake_pedal_bins={num_bins.brake_pedal}",
                f"+steering_angle_bins={num_bins.steering_angle}",
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
    trainer.predict(model, datamodule=datamodule, return_predictions=False)


@pytest.mark.parametrize("model", [lf("model_yaak_control_transformer_raw")])
def test_shared_encoder_state_dict(model: pl.LightningModule) -> None:
    state_dict = model.state_dict()
    encoder_keys = []
    objective_encoder_keys = []
    for k in state_dict:
        match k.split(".", maxsplit=3):
            case ["encoder", *_]:
                encoder_keys.append(k)

            case ["objectives", _objective, "encoder", *_]:
                objective_encoder_keys.append(k)

            case _:
                pass

    assert encoder_keys
    assert not objective_encoder_keys

    state_dict_ref = tree_map(torch.Tensor.clone, state_dict)  # ty:ignore[invalid-argument-type]
    model.load_state_dict(state_dict)
    state_dict_reload = model.state_dict()
    assert tree_all(tree_map(torch.equal, state_dict_ref, state_dict_reload))  # ty:ignore[invalid-argument-type]


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
    model = model.__class__.load_from_checkpoint(ckpt_path, strict=True)
    trainer.fit(model, datamodule=datamodule)
