import math
from typing import Literal

import pytest
import torch
from torch import Tensor
from torch.nn import GELU, Identity, Sequential

from rmind.components.augment import YawRotateTrajectory
from rmind.components.nn import Linear, Remapper, TrajectoryTarget
from rmind.components.vq import ResidualVQ
from rmind.models.trajectory_tokenizer import TrajectoryTokenizer

NUM_POSES = 5
HIDDEN_DIM = 32
LATENT_DIM = 16
CODEBOOK_SIZE = 8
NUM_QUANTIZERS = 2


def _make_tokenizer(
    representation: Literal["absolute", "delta"],
    input_transform: torch.nn.Module | None = None,
) -> TrajectoryTokenizer:
    dim = NUM_POSES * 3
    return TrajectoryTokenizer(
        input_transform=input_transform or Identity(),
        encoder=Sequential(
            Linear(dim, HIDDEN_DIM), GELU(), Linear(HIDDEN_DIM, LATENT_DIM)
        ),
        quantizer=ResidualVQ(
            dim=LATENT_DIM,
            codebook_size=CODEBOOK_SIZE,
            num_quantizers=NUM_QUANTIZERS,
            kmeans_init=False,
        ),
        decoder=Sequential(
            Linear(LATENT_DIM, HIDDEN_DIM), GELU(), Linear(HIDDEN_DIM, dim)
        ),
        num_poses=NUM_POSES,
        representation=representation,
    )


def _random_poses(n: int) -> Tensor:
    """Plausible forward-driving poses: increasing y, small lateral x, heading
    near +-pi on some samples to exercise wrapping."""
    step = torch.rand(n, NUM_POSES, 1) * 4.0
    lateral = torch.randn(n, NUM_POSES, 1) * 0.3
    xy = torch.cat([lateral, step], dim=-1).cumsum(dim=-2)
    theta = torch.randn(n, 1, 1) * 0.1 + torch.linspace(3.0, 3.3, NUM_POSES).view(
        1, -1, 1
    )
    theta = torch.atan2(torch.sin(theta), torch.cos(theta))
    return torch.cat([xy, theta], dim=-1)


@pytest.mark.parametrize("representation", ["absolute", "delta"])
def test_representation_round_trip(
    representation: Literal["absolute", "delta"],
) -> None:
    tokenizer = _make_tokenizer(representation)
    poses = _random_poses(64)

    x = tokenizer.to_representation(poses)
    assert x.shape == (64, NUM_POSES * 3)

    back = tokenizer.from_representation(x)
    torch.testing.assert_close(back[..., :2], poses[..., :2], atol=1e-4, rtol=0)
    heading_diff = torch.atan2(
        torch.sin(back[..., 2] - poses[..., 2]), torch.cos(back[..., 2] - poses[..., 2])
    )
    assert heading_diff.abs().max() < 1e-4  # noqa: PLR2004


def test_delta_heading_increments_are_wrapped() -> None:
    tokenizer = _make_tokenizer("delta")
    # crosses +pi -> -pi between poses 1 and 2: the raw increment is ~-2pi
    theta = torch.tensor([3.1, 3.13, -3.13, -3.1, -3.07])
    poses = torch.cat([torch.zeros(1, NUM_POSES, 2), theta.view(1, -1, 1)], dim=-1)
    delta_theta = tokenizer.to_representation(poses).view(NUM_POSES, 3)[:, 2]
    assert delta_theta[1:].abs().max() < 0.1  # noqa: PLR2004
    assert math.isclose(delta_theta[0].item(), 3.1, abs_tol=1e-6)


@pytest.mark.parametrize("representation", ["absolute", "delta"])
def test_step_updates_stats_and_backprops(
    representation: Literal["absolute", "delta"],
) -> None:
    tokenizer = _make_tokenizer(representation)
    tokenizer.train()
    poses = _random_poses(256)

    total, metrics = tokenizer._step({"context": {"trajectory_target": poses}})  # noqa: SLF001
    assert torch.isfinite(total)
    total.backward()
    assert tokenizer.encoder[0].weight.grad is not None

    assert tokenizer.norm_count.item() == 256  # noqa: PLR2004
    mean, std = tokenizer._norm_stats  # noqa: SLF001
    x = tokenizer.to_representation(poses)
    torch.testing.assert_close(mean, x.mean(0), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(std, x.std(0, unbiased=False), atol=1e-4, rtol=1e-4)

    for key in ("ade", "fde", "heading_err_deg", "perplexity/q0"):
        assert torch.isfinite(metrics[key])

    # stats are frozen outside of training
    tokenizer.eval()
    tokenizer._step({"context": {"trajectory_target": poses}})  # noqa: SLF001
    assert tokenizer.norm_count.item() == 256  # noqa: PLR2004


@pytest.mark.parametrize("representation", ["absolute", "delta"])
def test_forward_invert_shapes(representation: Literal["absolute", "delta"]) -> None:
    tokenizer = _make_tokenizer(representation)
    tokenizer.train()
    tokenizer._step({"context": {"trajectory_target": _random_poses(128)}})  # noqa: SLF001
    tokenizer.eval()

    poses = _random_poses(12).view(3, 4, NUM_POSES, 3)
    codes = tokenizer(poses)
    assert codes.shape == (3, 4, NUM_QUANTIZERS)
    assert codes.dtype == torch.long

    decoded = tokenizer.invert(codes)
    assert decoded.shape == (3, 4, NUM_POSES, 3)
    assert decoded[..., 2].abs().max() <= math.pi + 1e-6


def test_input_transform_matches_config_pipeline() -> None:
    """Mirrors config/model/yaak/trajectory_tokenizer/raw.yaml: Remapper ->
    TrajectoryTarget at a single anchor over a (b, P+1) clip."""
    input_transform = Sequential(
        Remapper(
            paths={
                "continuous": {"speed": ("data", "speed")},
                "context": {
                    "heading": ("data", "heading"),
                    "time_stamp": ("data", "time_stamp"),
                },
            }
        ),
        TrajectoryTarget(
            speed=("continuous", "speed"),
            heading=("context", "heading"),
            time_stamp=("context", "time_stamp"),
            out=("context", "trajectory_target"),
            episode_length=1,
            num_poses=NUM_POSES,
        ),
    )
    tokenizer = _make_tokenizer("delta", input_transform=input_transform)
    tokenizer.train()

    b, t = 8, NUM_POSES + 1
    epoch_us = 1_700_000_000_000_000
    batch = {
        "data": {
            "speed": torch.full((b, t), 36.0),  # 10 m/s
            "heading": torch.full((b, t), 90.0),
            "time_stamp": epoch_us + torch.arange(t).expand(b, t) * 333_333,
        }
    }
    poses = tokenizer._gather_poses(batch)  # noqa: SLF001
    assert poses.shape == (b, NUM_POSES, 3)
    # straight ahead at 10 m/s, ~1/3 s per pose: forward is local +y
    torch.testing.assert_close(
        poses[0, :, 1], torch.arange(1, NUM_POSES + 1) * 3.33333, atol=1e-3, rtol=0
    )
    assert poses[..., 0].abs().max() < 1e-4  # noqa: PLR2004

    total, _ = tokenizer._step(batch)  # noqa: SLF001
    assert torch.isfinite(total)


@pytest.mark.parametrize("k", range(1, NUM_QUANTIZERS + 1))
def test_prefix_decode_masks_later_levels(k: int) -> None:
    """Setting levels k.. to -1 decodes the sum of the first k levels' codebook
    vectors (docs/trajectory_tokenizer_codebook_plan.md §3.4)."""
    tokenizer = _make_tokenizer("absolute")
    tokenizer.train()
    tokenizer._step({"context": {"trajectory_target": _random_poses(128)}})  # noqa: SLF001
    tokenizer.eval()

    codes = tokenizer(_random_poses(32))
    masked = codes.clone()
    masked[..., k:] = -1
    z_q = tokenizer.quantizer.lookup(masked)

    reference = torch.stack([
        tokenizer.quantizer.codebook(level)[codes[:, level]] for level in range(k)
    ]).sum(dim=0)
    torch.testing.assert_close(z_q, reference)
    torch.testing.assert_close(tokenizer.decode(z_q), tokenizer.decode(reference))


def test_yaw_rotation_gives_lateral_first_pose_variance() -> None:
    """Dead-reckoned targets have x == 0 at pose 0 (std 0 -> clamped to 1e-6),
    so an unaugmented tokenizer blows up on rotated targets. Training through
    `YawRotateTrajectory` gives that feature real variance."""
    poses = _random_poses(512)
    poses[:, 0, 0] = 0.0
    batch = {"context": {"trajectory_target": poses[:, None]}}
    rotator = YawRotateTrajectory(fx=389.4, margin_px=32, walk_frames=32)

    plain = _make_tokenizer("absolute", input_transform=Identity()).train()
    plain._step({"context": {"trajectory_target": poses}})  # noqa: SLF001
    assert plain._norm_stats[1][0] <= 1e-6  # noqa: PLR2004, SLF001

    augmented = _make_tokenizer("absolute", input_transform=rotator).train()
    torch.manual_seed(0)
    total, _ = augmented._step(batch)  # noqa: SLF001
    assert torch.isfinite(total)
    assert augmented._norm_stats[1][0] > 0.01  # noqa: PLR2004, SLF001
