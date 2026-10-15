import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from rmind.components import fm_action_head
from rmind.components.fm_action_head import ActionHead
from rmind.data.d12 import ACTIONS, SIGNALS, episodes, join_fork_height
from rmind.models.patch_policy_continuous import PatchPolicyContinuous


def test_single_camera_ablation_preserves_data_and_training_recipe():
    directory = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        overrides = [
            "experiment=palletjack/patch_policy/d12_fm",
            "action_mean=[0,0,0]",
            "action_std=[1,1,1]",
        ]
        triple = compose(config_name="train", overrides=overrides)
        single = compose(
            config_name="train",
            overrides=[
                *overrides,
                "cameras=[cam_fork]",
                "num_cameras=1",
                "~image_remap.cam_left_forward",
                "~image_remap.cam_right_forward",
            ],
        )
    for section in ("datamodule", "trainer"):
        assert OmegaConf.to_container(
            single[section], resolve=True
        ) == OmegaConf.to_container(triple[section], resolve=True)
    for name in (
        "seed",
        "action_horizon",
        "action_step",
        "episode_length",
        "episode_step",
        "lr_total_steps",
        "lr_warmup_steps",
    ):
        assert single[name] == triple[name]
    assert single.model.encoder.tokens_per_frame == 149
    assert triple.model.encoder.tokens_per_frame == 437
    assert list(single.image_remap) == ["cam_fork"]


def test_vendored_flow_head_checksum():
    assert hashlib.sha256(Path(fm_action_head.__file__).read_bytes()).hexdigest() == (
        "8a99bd91f12807fc7e1ccee690d8ac4acc9e4c09121b369b435312f90aa65582"
    )


@pytest.mark.parametrize(
    "fixture,horizon,actions",
    [
        ("fm_fixture_bp_w10_s0_v1.1.pt", 10, 2),
        ("fm_fixture_robot_h50_v1.1.pt", 50, 13),
        ("fm_fixture_robot_h100_v1.1.pt", 100, 13),
    ],
)
def test_flow_parity(fixture, horizon, actions):
    path = Path("/nasa/max/bimod-study/wta/parity") / fixture
    if not path.exists():
        pytest.skip("external study fixture unavailable")
    data = torch.load(path, map_location="cpu", weights_only=False)
    head = ActionHead("fm", data["feat"], 1, data["D"]).eval()
    head.load_state_dict(data["state_dict"])
    policy = SimpleNamespace(
        fm_head=head,
        action_horizon=horizon,
        action_names=list(range(actions)),
        fm_inference_steps=4,
        fm_x0="sample",
    )
    inputs, reference = data["inputs"], data["ref"]
    features, noise, target = inputs["f"], inputs["x0"], inputs["target"]
    original_noise = noise.clone()
    with torch.no_grad():
        torch.testing.assert_close(
            head.velocity(features, inputs["xt"], inputs["t"]),
            reference["velocity"],
            atol=1e-5,
            rtol=0,
        )
        for steps in (1, 4, 10):
            for initial, suffix in [(noise, ""), (torch.zeros_like(noise), "_x0zero")]:
                actual = PatchPolicyContinuous.fm_sample(
                    policy, features, initial, steps
                )
                torch.testing.assert_close(
                    actual.flatten(1),
                    reference[f"euler{steps}{suffix}"],
                    atol=1e-5,
                    rtol=0,
                )
        time = inputs["t"][:, None]
        interpolated = (1 - time) * noise + time * target
        errors = (
            head.velocity(features, interpolated, inputs["t"]) - (target - noise)
        ).square()
        torch.testing.assert_close(
            errors.mean(), reference["cfm_loss"], atol=1e-5, rtol=0
        )
        mask = inputs["mask"]
        torch.testing.assert_close(
            (errors * mask).sum() / mask.sum(),
            reference["cfm_loss_masked"],
            atol=1e-5,
            rtol=0,
        )
    assert torch.equal(noise, original_noise)


def test_height_never_uses_future_stale_or_invalid_samples():
    times = pl.Series(
        "log_time", [0, 100_000_000, 200_000_000, 500_000_000], dtype=pl.Datetime("ns")
    )
    table = pl.DataFrame({"log_time": times})
    height = pl.DataFrame({
        "log_time": pl.Series([50_000_000, 150_000_000], dtype=pl.Datetime("ns")),
        "height_m": [1.15, 2.3],
        "sigma_m": [0.23, 0.1],
        "valid": [True, False],
    })
    actual = join_fork_height(table, height)["fork_height"].to_list()
    np.testing.assert_allclose(actual, [[0, 0, 0], [0.5, 0.1, 1], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(
        join_fork_height(table, None)["fork_height"].to_list(), np.zeros((4, 3))
    )
    valid_only = height.head(1)
    assert join_fork_height(table, valid_only)["fork_height"][-1].to_list() == [0, 0, 0]


@pytest.mark.parametrize("action_weights", [None, (1.0, 2.0, 1.0)])
def test_flow_loss_updates_trunk_and_head_and_roundtrips(tmp_path, action_weights):
    class TestEncoder(nn.Module):
        def forward(self, inputs, *, num_frames):
            return inputs

    policy = PatchPolicyContinuous(
        input_transform=nn.Identity(),
        image_encoder=nn.Identity(),
        patch_projection=nn.Linear(3, 8),
        speed_tokenizer=nn.Identity(),
        speed_embedding=nn.Linear(1, 8),
        encoder=TestEncoder(),
        heads=None,
        losses=None,
        targets={
            "continuous": {
                "traction": ("traction",),
                "steering": ("steering",),
                "fork1": ("fork1",),
            }
        },
        fm_head=ActionHead("fm", 8, 1, 12, hidden=32),
        action_horizon=4,
        action_mean=(0.1, -0.2, 0.3),
        action_std=(0.2, 0.3, 0.4),
        fm_action_weights=action_weights,
    )
    trunk = nn.Linear(5, 8)
    raw_features = torch.randn(2, 3, 5)
    policy._features = lambda batch: trunk(raw_features)
    policy.log_dict = lambda *args, **kwargs: None
    targets = {name: torch.randn(2, 3, 4) for name in ("traction", "steering", "fork1")}
    result = policy.training_step(targets, 0)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert trunk.weight.grad.abs().sum() > 0
    assert policy.fm_head.vel[0].weight.grad.abs().sum() > 0
    features = trunk(raw_features)
    torch.manual_seed(7)
    reference = policy.fm_sample(features) * policy.action_std + policy.action_mean
    torch.manual_seed(7)
    actual = policy._predict(features)
    for index, name in enumerate(targets):
        torch.testing.assert_close(actual["continuous", name], reference[..., index])
    torch.save(policy.state_dict(), tmp_path / "weights.pt")
    saved = torch.load(tmp_path / "weights.pt", weights_only=True)
    torch.testing.assert_close(saved["action_mean"], policy.action_mean)
    torch.testing.assert_close(saved["action_std"], policy.action_std)


@pytest.mark.parametrize("horizon", [50, 100])
def test_30hz_actions_with_5hz_observations(horizon):
    count = 180
    table = pl.DataFrame({
        "input_id": ["job"] * count,
        "cam_fork/frame_idx": np.arange(count),
        **{name: np.arange(count, dtype=np.float32) for name in SIGNALS},
        "fork_height": [[0.5, 0.1, 1.0]] * count,
    })
    samples = episodes(
        table,
        cameras=("cam_fork",),
        episode_length=6,
        step=6,
        stride=1,
        horizon=horizon,
        action_step=1,
        require_contiguous=True,
    )
    frames = samples["cam_fork/frame_idx"].to_numpy()
    np.testing.assert_array_equal(frames[0], [0, 6, 12, 18, 24, 30])
    for name in ACTIONS:
        actions = samples[name].to_numpy()
        assert actions.shape[1:] == (6, horizon)
        for position, frame_index in enumerate(frames[0]):
            np.testing.assert_array_equal(
                actions[0, position], np.arange(frame_index, frame_index + horizon)
            )
    assert samples["fork_height"].to_numpy().shape[1:] == (6, 3)


def test_action_windows_do_not_bridge_dropped_rows():
    indices = np.concatenate([np.arange(5), np.arange(6, 15)])
    table = pl.DataFrame({
        "input_id": ["job"] * len(indices),
        "cam_fork/frame_idx": indices,
        **{name: indices.astype(np.float32) for name in SIGNALS},
    })
    samples = episodes(
        table,
        cameras=("cam_fork",),
        episode_length=2,
        step=2,
        stride=1,
        horizon=4,
        action_step=1,
        require_contiguous=True,
    )
    assert samples["cam_fork/frame_idx"].to_numpy()[:, 0].min() == 6
