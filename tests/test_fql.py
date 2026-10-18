import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from rmind.components.fql import FQLConfig, FQLNoise, FQLPolicy
from rmind.datamodules.fql import FQLDataModule, FQLTransitions
from rmind.models.fql import FQLModel


def config() -> FQLConfig:
    return FQLConfig(
        observation_dim=4,
        action_dim=2,
        actor_hidden_dims=(8, 8),
        value_hidden_dims=(8, 8),
    )


def batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        "observations": torch.randn(5, 4, generator=generator),
        "next_observations": torch.randn(5, 4, generator=generator),
        "actions": torch.rand(5, 2, generator=generator) * 2 - 1,
        "rewards": torch.tensor([-1.0, -1.0, 0.0, -1.0, 0.0]),
        "masks": torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0]),
    }


def test_explicit_noise_loss_repeat_and_gradient_boundaries() -> None:
    policy = FQLPolicy(config())
    transitions = batch()
    noise = FQLNoise.draw(transitions["actions"], torch.Generator().manual_seed(9))
    total, losses = policy.losses(transitions, noise)
    assert torch.equal(total, policy.losses(transitions, noise)[0])
    for name, allowed in (
        ("q_loss", "actor"),
        ("distillation_loss", "actor"),
        ("prior_loss", "prior"),
        ("critic_loss", "critics"),
    ):
        policy.zero_grad(set_to_none=True)
        losses[name].backward(retain_graph=True)
        for parameter_name, parameter in policy.named_parameters():
            if parameter_name.split(".")[0] != allowed:
                assert parameter.grad is None
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in getattr(policy, allowed).parameters()
        )


def test_preoptimizer_targets_and_checkpoint_resume() -> None:
    policy = FQLPolicy(config())
    optimizer = torch.optim.Adam(
        [value for value in policy.parameters() if value.requires_grad], lr=3e-4
    )
    transitions = batch()
    noise = FQLNoise.draw(transitions["actions"], torch.Generator().manual_seed(7))
    initial = copy.deepcopy(policy.critics.state_dict())
    policy.losses(transitions, noise)[0].backward()
    policy.update_targets()
    optimizer.step()
    for name, tensor in policy.target_critics.state_dict().items():
        torch.testing.assert_close(tensor, initial[name], atol=1e-7, rtol=1e-7)
    restored = FQLPolicy(config())
    restored.load_state_dict(policy.state_dict())
    resumed = torch.optim.Adam(
        [value for value in restored.parameters() if value.requires_grad], lr=3e-4
    )
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for model, optim in ((policy, optimizer), (restored, resumed)):
        optim.zero_grad(set_to_none=True)
        model.losses(transitions, noise)[0].backward()
        model.update_targets()
        optim.step()
    for name, tensor in policy.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])


@pytest.mark.parametrize(
    ("field", "value"), [("actions", float("nan")), ("actions", 2.0), ("masks", 0.5)]
)
def test_reject_invalid_transition(field: str, value: float) -> None:
    transitions = batch()
    transitions[field].flatten()[0] = value
    policy = FQLPolicy(config())
    with pytest.raises(ValueError, match=r"finite|canonical"):
        policy.losses(transitions, FQLNoise.draw(transitions["actions"]))


def test_normalization_explicit_noise_and_lightning_checkpoint(tmp_path: Path) -> None:
    model = FQLModel(
        asdict(config()),
        normalization={
            "observation_mean": [1.0] * 4,
            "observation_std": [2.0] * 4,
            "action_low": [-2.0, -3.0],
            "action_high": [2.0, 5.0],
        },
    )
    transitions = batch()
    noise = torch.randn(5, 2)
    expected = (
        model.action_low / 2
        + model.action_high / 2
        + model.policy.sample_actions((transitions["observations"] - 1) / 2, noise)
        * (model.action_high / 2 - model.action_low / 2)
    )
    assert torch.equal(expected, model(transitions["observations"], noise))
    rows = [
        {key: value[index] for key, value in transitions.items()} for index in range(5)
    ]
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=DataLoader(rows, batch_size=5))
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    restored = FQLModel.load_from_checkpoint(path, weights_only=True)
    assert torch.equal(
        model(transitions["observations"], noise),
        restored(transitions["observations"], noise),
    )
    checkpoint = torch.load(path, weights_only=True)
    checkpoint.pop("fql_contract")
    with pytest.raises(ValueError, match="contract"):
        model.on_load_checkpoint(checkpoint)


def test_transition_data_requires_labels_and_disjoint_episodes(tmp_path: Path) -> None:
    arrays = {name: tensor.numpy() for name, tensor in batch().items()}
    train, validation = tmp_path / "train.npz", tmp_path / "validation.npz"
    np.savez(train, **arrays)
    with pytest.raises(ValueError, match="explicit"):
        FQLTransitions(str(train))
    np.savez(train, **arrays, episode_ids=np.ones(5, dtype=np.int64))
    np.savez(validation, **arrays, episode_ids=np.ones(5, dtype=np.int64))
    module = FQLDataModule(str(train), str(validation), batch_size=5)
    with pytest.raises(ValueError, match="overlap"):
        module.setup()
    np.savez(validation, **arrays, episode_ids=np.full(5, 2, dtype=np.int64))
    module.setup()
    assert len(next(iter(module.train_dataloader()))["actions"]) == module.batch_size


@pytest.mark.parametrize("field", ["next_actions", "prior", "times", "distillation"])
def test_invalid_explicit_noise_fails(field: str) -> None:
    transitions = batch()
    noise = FQLNoise.draw(transitions["actions"])
    values = {
        name: getattr(noise, name)
        for name in ("next_actions", "prior", "times", "distillation")
    }
    values[field] = values[field][:1]
    with pytest.raises(ValueError, match=r"shape|times"):
        FQLPolicy(config()).losses(transitions, FQLNoise(**values))


def test_zero_bootstrap_masks_remove_next_state_dependence() -> None:
    transitions = batch()
    transitions["masks"].zero_()
    policy = FQLPolicy(config())
    noise = FQLNoise.draw(transitions["actions"])
    before = policy.losses(transitions, noise)[1]["critic_loss"]
    transitions["next_observations"] *= 19
    after = policy.losses(transitions, noise)[1]["critic_loss"]
    assert torch.equal(before, after)


def test_checkpoint_normalization_metadata_disagreement_fails() -> None:
    model = FQLModel(asdict(config()))
    checkpoint = {"state_dict": copy.deepcopy(model.state_dict())}
    model.on_save_checkpoint(checkpoint)
    checkpoint["state_dict"]["action_high"][0] = 2
    with pytest.raises(ValueError, match="normalization"):
        model.on_load_checkpoint(checkpoint)


def test_optional_hydra_configuration_instantiates_without_default_changes() -> None:
    directory = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        composed = compose(
            config_name="train_fql",
            overrides=[
                "model.config.observation_dim=4",
                "model.config.action_dim=2",
                "model.config.actor_hidden_dims=[8,8]",
                "model.config.value_hidden_dims=[8,8]",
                "datamodule.train_path=/unused/train.npz",
                "datamodule.validation_path=/unused/val.npz",
            ],
        )
        model = instantiate(composed.model)
        module = instantiate(composed.datamodule)
    assert isinstance(model, FQLModel)
    assert isinstance(module, FQLDataModule)
    assert composed.trainer.precision == "32-true"
    assert composed.matmul_precision == "highest"


@pytest.mark.parametrize(
    "identities", [np.full(5, np.nan), np.full(5, ""), np.full(5, " "), np.ones(5)]
)
def test_invalid_episode_id_cannot_evade_disjoint_split(
    tmp_path: Path, identities: np.ndarray
) -> None:
    arrays = {name: tensor.numpy() for name, tensor in batch().items()}
    train, validation = tmp_path / "train.npz", tmp_path / "validation.npz"
    for path in (train, validation):
        np.savez(path, **arrays, episode_ids=identities)
    with pytest.raises(ValueError, match="Episode IDs"):
        FQLDataModule(str(train), str(validation), batch_size=5).setup()


def test_action_range_overflow_fails() -> None:
    with pytest.raises(ValueError, match="Normalization"):
        FQLModel(
            asdict(config()),
            normalization={"action_low": [-3e38] * 2, "action_high": [3e38] * 2},
        )


@pytest.mark.parametrize(
    "field", ["actor_layer_norm", "critic_layer_norm", "normalize_q_loss"]
)
def test_boolean_config_does_not_accept_strings(field: str) -> None:
    parameters = asdict(config())
    parameters[field] = "false"
    with pytest.raises(ValueError, match="booleans"):
        FQLConfig(**parameters)


def test_identity_normalization_preserves_small_canonical_actions_exactly() -> None:
    model = FQLModel(asdict(config()))
    transitions = batch()
    transitions["actions"][0] = torch.tensor([1e-8, -1e-8])
    normalized = model.normalized_batch(transitions)
    for name, value in transitions.items():
        assert torch.equal(value, normalized[name])
    noise = torch.randn_like(transitions["actions"])
    assert torch.equal(
        model(transitions["observations"], noise),
        model.policy.sample_actions(transitions["observations"], noise),
    )


@pytest.mark.parametrize("field", ["observations", "actions", "rewards"])
def test_complex_transition_values_are_not_silently_truncated(
    tmp_path: Path, field: str
) -> None:
    arrays = {name: tensor.numpy() for name, tensor in batch().items()}
    arrays[field] = arrays[field].astype(np.complex64) + 1j
    path = tmp_path / "complex.npz"
    np.savez(
        path, **arrays, episode_ids=np.ones(len(arrays["actions"]), dtype=np.int64)
    )
    with pytest.raises(ValueError, match="nonnumeric"):
        FQLTransitions(str(path))
