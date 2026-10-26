"""Unit tests for `rmind.components.flow_matching.FlowMatchingTrajectoryHead`."""

from typing import Any

import torch

from rmind.components.flow_matching import (
    FlowMatchingTrajectoryHead,
    sinusoidal_time_embedding,
)

IN_CHANNELS = 8
NUM_POSES = 4


def _make_head(**kwargs: Any) -> FlowMatchingTrajectoryHead:
    return FlowMatchingTrajectoryHead(
        in_channels=IN_CHANNELS, num_poses=NUM_POSES, hidden_channels=(32, 32), **kwargs
    )


def test_time_embedding_shape_and_range() -> None:
    embedding = sinusoidal_time_embedding(torch.rand(2, 3), 16)
    assert embedding.shape == (2, 3, 16)
    assert embedding.abs().max() <= 1.0


def test_sample_shapes_and_deterministic_mode() -> None:
    head = _make_head().eval()
    features = torch.randn(2, 3, IN_CHANNELS)

    samples = head.sample(features, 5)
    assert samples.shape == (2, 3, 5, NUM_POSES, 3)

    deterministic = head.sample(features, 5, noise=False)
    torch.testing.assert_close(
        deterministic, deterministic[..., :1, :, :].expand_as(deterministic)
    )
    torch.testing.assert_close(head(features), deterministic[..., 0, :, :])


def test_normalizer_accumulates_only_in_training_then_freezes() -> None:
    warmup_steps = 2
    head = _make_head(normalizer_warmup_steps=warmup_steps)
    targets = [torch.randn(6, NUM_POSES, 3) * 5.0 + 3.0 for _ in range(3)]
    features = torch.randn(6, IN_CHANNELS)

    head.eval()
    head.loss(features, targets[0])
    assert head.stats_steps == 0

    head.train()
    for target in targets:
        head.loss(features, target)
    assert head.stats_steps == warmup_steps  # frozen: the third batch is ignored

    seen = torch.cat(targets[:warmup_steps])
    mean, std = head._normalizer()  # noqa: SLF001
    torch.testing.assert_close(mean, seen.mean(dim=0))
    torch.testing.assert_close(std, seen.std(dim=0, unbiased=False))


def test_normalizer_survives_state_dict_round_trip() -> None:
    head = _make_head().train()
    head.loss(torch.randn(6, IN_CHANNELS), torch.randn(6, NUM_POSES, 3) * 4.0)

    restored = _make_head()
    restored.load_state_dict(head.state_dict())
    for a, b in zip(head._normalizer(), restored._normalizer(), strict=True):  # noqa: SLF001
        torch.testing.assert_close(a, b)


def test_learns_feature_conditioned_trajectories_in_target_units() -> None:
    """Two contexts with distinct, meter-scale trajectories: the flow must
    learn to tell them apart from the features and land in target units."""
    torch.manual_seed(0)
    head = _make_head(normalizer_warmup_steps=10, num_inference_steps=10)
    features = torch.eye(2, IN_CHANNELS)
    progress = torch.linspace(1.0, 4.0, NUM_POSES)
    targets = torch.stack([
        torch.stack([5.0 * progress, 0.5 * progress, 0.02 * progress], dim=-1),
        torch.stack([3.0 * progress, -1.0 * progress, -0.1 * progress], dim=-1),
    ])
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)

    head.train()
    for _ in range(600):
        optimizer.zero_grad()
        head.loss(features.repeat(32, 1), targets.repeat(32, 1, 1)).backward()
        optimizer.step()

    head.eval()
    predicted = head(features)
    torch.testing.assert_close(predicted, targets, atol=0.3, rtol=0.05)
