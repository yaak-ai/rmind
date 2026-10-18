import torch
from torch import nn
import pytest
from omegaconf import OmegaConf

from rmind.components.rtc_flow_head import TokenTimeFlowHead
from rmind.scripts.export_d12_fm import checkpoint_export_safety
from rmind.scripts.train_d12_rtc import (
    SmokeCheckpointQuarantine,
    build_config,
    warmstart,
)


def test_smoke_checkpoint_stays_quarantined_after_reload_and_default_export(tmp_path):
    checkpoint = {"state_dict": {"weight": torch.ones(1)}}
    SmokeCheckpointQuarantine().on_save_checkpoint(None, None, checkpoint)
    path = tmp_path / "renamed.ckpt"
    torch.save(checkpoint, path)
    metadata = torch.load(path, weights_only=True, map_location="meta")
    assert checkpoint_export_safety(metadata, False) == {
        "test_only": True,
        "head_trained": False,
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {"test_only": True},
        {"rtc_test_fixture": {"deployable": False}},
        {"rtc_test_fixture": {"head_trained": False}},
    ],
)
def test_each_quarantine_marker_survives_default_export(metadata):
    assert checkpoint_export_safety(metadata, False)["test_only"] is True


@pytest.mark.parametrize(
    "metadata",
    [
        {"test_only": "false"},
        {"rtc_test_fixture": []},
        {"rtc_test_fixture": {"deployable": 0}},
        {"rtc_test_fixture": {"head_trained": "true"}},
    ],
)
def test_malformed_quarantine_metadata_rejected(metadata):
    with pytest.raises(ValueError):
        checkpoint_export_safety(metadata, False)


def test_legacy_export_and_explicit_test_only_behavior_remain_available():
    assert checkpoint_export_safety({}, False) == {
        "test_only": False,
        "head_trained": None,
    }
    assert checkpoint_export_safety({}, True)["test_only"] is True


@pytest.fixture
def baseline_config(tmp_path):
    config = OmegaConf.create({
        "model": {
            "cameras": ["cam_fork"],
            "action_horizon": 50,
            "fm_head": {"_target_": "legacy.Head"},
            "optimizer": {"lr": 1e-4, "betas": [0.9, 0.95], "weight_decay": 0.1},
            "action_mean": [0.1, 0.2, 0.3],
            "action_std": [0.4, 0.5, 0.6],
            "height_validity_mask": True,
        },
        "datamodule": {
            split: {"dataset": {"samples": {"inputs": {"job_dir": list(range(count))}}}}
            for split, count in (("train", 34), ("val", 4))
        },
        "trainer": {
            "precision": "bf16-mixed",
            "max_steps": 35000,
            "accumulate_grad_batches": 4,
            "logger": {"save_dir": "old"},
            "callbacks": [{}, {"dirpath": "old"}],
        },
        "paths": {"rbyte": {"cache": "old"}},
        "wandb": {"tags": ["fork_only"]},
    })
    OmegaConf.save(config, tmp_path / "resolved.yaml")
    return tmp_path, config


def test_config_preserves_dataset_normalization_and_optimizer_recipe(baseline_config):
    baseline, original = baseline_config
    config = build_config(baseline, baseline / "output")
    assert config.datamodule == original.datamodule
    assert config.model.action_mean == original.model.action_mean
    assert config.model.action_std == original.model.action_std
    assert config.model.height_validity_mask
    assert config.model.optimizer.betas == original.model.optimizer.betas
    assert config.model.optimizer.weight_decay == original.model.optimizer.weight_decay
    assert config.model.optimizer.lr == 3e-5
    assert config.model.fm_head.time_embedding_mode == "native"
    assert "time_frequency" not in config.model.fm_head
    assert config.model.fm_head.max_prefix == 6
    assert config.model.fm_head.action_horizon == 50
    assert config.model.fm_inference_steps == 5
    assert config.trainer.precision == "32-true"
    assert config.trainer.max_steps == 35000
    assert config.trainer.accumulate_grad_batches == 4
    assert OmegaConf.load(baseline / "resolved.yaml") == original


@pytest.mark.parametrize("failure", ["cameras", "horizon", "train_split", "val_split"])
def test_config_rejects_wrong_baseline(baseline_config, failure):
    baseline, config = baseline_config
    if failure == "cameras":
        config.model.cameras = ["cam_left_forward", "cam_right_forward"]
    elif failure == "horizon":
        config.model.action_horizon = 100
    else:
        split = "train" if failure == "train_split" else "val"
        config.datamodule[split].dataset.samples.inputs.job_dir.pop()
    OmegaConf.save(config, baseline / "resolved.yaml")
    with pytest.raises(ValueError):
        build_config(baseline, baseline / "output")


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.fm_head = TokenTimeFlowHead(4, 3, 8, channel_dim=16, num_layers=1)
        self.register_buffer("action_mean", torch.tensor([0.1, 0.2, 0.3]))
        self.register_buffer("action_std", torch.tensor([0.4, 0.5, 0.6]))


def test_warmstart_preserves_new_native_head_and_loads_every_shared_tensor():
    model = TinyPolicy()
    source = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("fm_head.")
    }
    source["encoder.weight"] += 3
    source["fm_head.old_mlp.weight"] = torch.ones(1)
    initial_head = {
        name: value.clone() for name, value in model.fm_head.state_dict().items()
    }
    audit = warmstart(model, source)
    assert audit["new_head_unchanged"]
    assert audit["all_shared_tensors_bitwise_equal"]
    assert audit["native_time_mode"] == "native"
    assert audit["ignored_source_head_keys"] == ["fm_head.old_mlp.weight"]
    torch.testing.assert_close(
        model.encoder.weight, source["encoder.weight"], atol=0, rtol=0
    )
    for name, value in model.fm_head.state_dict().items():
        torch.testing.assert_close(value, initial_head[name], atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "normalization"])
def test_warmstart_rejects_partial_or_changed_shared_state(failure):
    model = TinyPolicy()
    source = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("fm_head.")
    }
    if failure == "missing":
        del source["encoder.bias"]
    elif failure == "extra":
        source["unexpected"] = torch.zeros(1)
    elif failure == "shape":
        source["encoder.bias"] = torch.zeros(3)
    else:
        source["action_std"] += 0.1
    with pytest.raises(ValueError):
        warmstart(model, source)
