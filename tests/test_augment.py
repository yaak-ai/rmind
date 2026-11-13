import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import Identity, Sequential
from torch.testing import assert_close
from torchvision.transforms.v2 import CenterCrop, Resize

from rmind.components.augment import (
    ClipPhotometric,
    ConditionalSpeedShift,
    FixedFrameMask,
    YawShiftCrop,
    rotate_ego_frame,
)
from rmind.components.containers import ModuleDict
from rmind.components.dead_reckoning import (
    dead_reckon_future_trajectory,
    rolling_dead_reckoned_trajectory,
)
from rmind.models.patch_policy import PatchPolicy

FX = 389.4  # fx_native 1298 px at 1920 -> 576 px wide


def _sql_ego_frame(world_xy: torch.Tensor, heading_deg: float) -> torch.Tensor:
    """`ST_Rotate(geom, radians(heading))`: counter-clockwise rotation in (E, N)."""
    h = math.radians(heading_deg)
    e, n = world_xy[..., 0], world_xy[..., 1]
    return torch.stack(
        [e * math.cos(h) - n * math.sin(h), e * math.sin(h) + n * math.cos(h)], dim=-1
    )


def test_straight_target_bends_left_when_camera_yawed_right() -> None:
    straight = torch.tensor([[0.0, 5.0], [0.0, 10.0]], dtype=torch.float64)
    psi = torch.tensor(math.radians(3.0), dtype=torch.float64)

    rotated = rotate_ego_frame(straight, psi)

    assert (rotated[:, 0] < 0).all()  # recorded path is now to the camera's left
    assert_close(rotated.norm(dim=-1), straight.norm(dim=-1))


def test_rotation_matches_dead_reckoning_and_waypoint_conventions() -> None:
    """Yawing the ego by +psi must equal re-projecting the same world path
    with heading + psi -- the convention both `dead_reckoning` and the
    `waypoints/xy_normalized` SQL use."""
    t = 8
    heading_deg = 30.0 + torch.linspace(0, 12, t, dtype=torch.float64)  # curving right
    speed = torch.full((t,), 36.0, dtype=torch.float64)
    stamp = torch.arange(t, dtype=torch.float64) / 3
    psi_deg = 4.0

    ego, theta = dead_reckon_future_trajectory(
        speed_kmh=speed, heading_deg=heading_deg, time_stamp_s=stamp
    )
    # independent world-frame integration (compass: E += v sin h, N += v cos h)
    h = torch.deg2rad(heading_deg[:-1])
    step = speed[:-1] / 3.6 * stamp.diff()
    world = torch.stack([step * h.sin(), step * h.cos()], dim=-1).cumsum(0)

    assert_close(ego, _sql_ego_frame(world, heading_deg[0].item()))
    expected = _sql_ego_frame(world, heading_deg[0].item() + psi_deg)
    assert_close(rotate_ego_frame(ego, torch.tensor(math.radians(psi_deg))), expected)
    assert_close(
        theta - math.radians(psi_deg),
        torch.deg2rad(heading_deg[1:] - heading_deg[0] - psi_deg),
    )


def _batch(b: int = 4, t: int = 6, h: int = 324, w: int = 576) -> dict:
    image = torch.randint(0, 256, (b, t, h, w, 3), dtype=torch.uint8)
    target = torch.randn(b, t, 5, 3)
    target[..., 2] = target[..., 2].clamp(-1, 1)
    return {
        "image": {"cam_front_left": image},
        "context": {"trajectory_target": target, "waypoints": torch.randn(b, t, 10, 2)},
        "continuous": {"speed": torch.rand(b, t, 1)},
    }


def _module(*, probability: float = 0.5) -> YawShiftCrop:
    return YawShiftCrop(
        crop_width=512, fx=FX, cameras=["cam_front_left"], probability=probability
    )


def _center(image: torch.Tensor) -> torch.Tensor:
    return CenterCrop((320, 512))(image.movedim(-1, -3))


def test_eval_is_identity() -> None:
    batch = _batch()

    assert _module(probability=1.0).eval()(batch) is batch


def test_zero_probability_matches_eval_pipeline_bit_for_bit() -> None:
    batch = _batch()

    out = _module(probability=0.0).train()(batch)

    assert torch.equal(
        _center(out["image"]["cam_front_left"]),
        _center(batch["image"]["cam_front_left"]),
    )
    assert torch.equal(
        out["context"]["trajectory_target"], batch["context"]["trajectory_target"]
    )
    assert torch.equal(out["context"]["waypoints"], batch["context"]["waypoints"])
    assert out["continuous"] is batch["continuous"]


def test_image_shift_and_labels_agree() -> None:
    """Camera yawed right -> scene shifts left in the crop -> target bends left."""
    b, t, w = 2, 3, 576
    image = torch.zeros(b, t, 324, w, 3, dtype=torch.uint8)
    marker = w // 2
    image[..., marker, :] = 255
    straight = torch.zeros(b, t, 5, 3)
    straight[..., 1] = torch.arange(1, 6, dtype=torch.float32) * 3
    batch = {
        "image": {"cam_front_left": image},
        "context": {"trajectory_target": straight, "waypoints": straight[..., :2]},
    }
    module = _module(probability=1.0).train()
    psi = torch.full((b, t), math.radians(2.0))
    module.sample_psi = lambda *_a, **_k: psi  # ty:ignore[invalid-assignment]

    out = module(batch)

    shift = round(math.radians(2.0) * FX)
    column = out["image"]["cam_front_left"][0, 0, 0, :, 0].nonzero().item()
    assert column == marker - 32 - shift  # centre crop would put it at marker - 32
    realised = shift / FX
    target = out["context"]["trajectory_target"]
    assert (target[..., 0] < 0).all()
    assert_close(target[..., 2], torch.full_like(target[..., 2], -realised))
    assert_close(target[..., 0], -straight[..., 1] * math.sin(realised))
    assert_close(out["context"]["waypoints"], target[..., :2])


def test_psi_is_a_smooth_bounded_walk_per_clip() -> None:
    module = _module(probability=0.5)
    max_rad = 32 / FX
    torch.manual_seed(0)

    psi = module.sample_psi(4096, 16, max_rad=max_rad, device=torch.device("cpu"))

    assert psi.abs().max() <= max_rad + 1e-6
    selected = psi.ne(0).any(dim=1)
    assert abs(selected.float().mean() - 0.5) < 0.05  # noqa: PLR2004
    assert torch.equal(psi[~selected], torch.zeros_like(psi[~selected]))
    steps = psi[selected].diff(dim=1).abs()
    assert steps.max() < math.radians(0.5) * 6  # no frame-to-frame jumps


@pytest.mark.parametrize("training", [True, False])
def test_photometric_shape_range_and_eval_identity(*, training: bool) -> None:
    x = torch.rand(3, 4, 3, 64, 64)
    module = ClipPhotometric(blur_p=1.0, noise_p=1.0, jpeg_p=1.0).train(training)

    out = module(x)

    if training:
        assert out.shape == x.shape
        assert out.min() >= 0
        assert out.max() <= 1
        assert not torch.equal(out, x)
    else:
        assert out is x


def test_photometric_parameters_constant_within_clip() -> None:
    x = torch.full((3, 4, 3, 8, 8), 0.5)
    module = ClipPhotometric(
        contrast=0.0, saturation=0.0, hue=0.0, blur_p=0.0, noise_p=0.0, jpeg_p=0.0
    ).train()

    out = module(x)

    per_frame = out.mean(dim=(-3, -2, -1))  # (b, t)
    assert_close(per_frame, per_frame[:, :1].expand_as(per_frame))
    assert per_frame[:, 0].unique().numel() == per_frame.shape[0]


def test_export_finds_modality_dict_behind_augmentation() -> None:
    modalities = ModuleDict(modules={"image": Identity()})
    owner = SimpleNamespace(
        input_transform=Sequential(Identity(), Identity(), _module(), modalities)
    )

    assert PatchPolicy.modality_transforms(owner) is modalities  # ty:ignore[invalid-argument-type]


def test_photometric_under_bf16_autocast() -> None:
    x = torch.rand(2, 3, 3, 32, 32)
    module = ClipPhotometric(blur_p=1.0, noise_p=1.0, jpeg_p=1.0).train()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(x)

    assert out.dtype == x.dtype


@pytest.mark.parametrize("psi_deg", [-4.7, 0.0, 4.7])
def test_frame_mask_hides_hood_and_arcs_at_any_shift(psi_deg: float) -> None:
    """Paint the hood band and arcs black on a white frame, run the R1 image
    path at the largest shifts: nothing dark may survive outside the mask."""
    mask = FixedFrameMask(cameras=["cam_front_left"], size=(224, 224))
    frame = torch.full((1, 1, 324, 576, 3), 255, dtype=torch.uint8)
    rows = torch.arange(324)[:, None].float()
    cols = torch.arange(576)[None, :].float()
    hood_row, arc_depth, arc_radius = 255, 35, 240
    fixed = (rows >= hood_row) | (
        rows < arc_depth * (1 - ((cols - 288) / arc_radius) ** 2)
    )
    frame[0, 0][fixed] = 0
    crop = _module(probability=1.0).train()
    crop.sample_psi = lambda *_a, **_k: torch.full((1, 1), math.radians(psi_deg))  # ty:ignore[invalid-assignment]

    shifted = crop({"image": {"cam_front_left": frame}})["image"]["cam_front_left"]
    image = Resize((224, 224))(_center(shifted).float() / 255) - 0.5  # white -> +0.5
    out = mask({"image": {"cam_front_left": image}})["image"]["cam_front_left"]

    keep = mask.keep[0, 0, 0].bool()
    assert torch.equal(out[..., ~keep], torch.zeros_like(out[..., ~keep]))
    assert (out[..., keep] > 0.49).all()  # noqa: PLR2004
    assert 0.2 < (~keep).float().mean() < 0.35  # noqa: PLR2004


def _speed_shift_inputs(
    speed_kmh: torch.Tensor, *, episode_length: int, num_poses: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(speed (b, t, 1), trajectory_target (b, t, P, 3))` as the input
    transform would produce them, from a full-clip `(b, T)` speed profile at
    3 Hz on a straight road.
    """
    b, t = speed_kmh.shape
    heading = torch.zeros(b, t)
    time_stamp_us = (torch.arange(t, dtype=torch.float64) * 1e6 / 3).expand(b, -1)
    target = rolling_dead_reckoned_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading,
        time_stamp_us=time_stamp_us,
        episode_length=episode_length,
        num_poses=num_poses,
    )
    return speed_kmh[:, :episode_length, None], target


def test_speed_shift_flags_braking_and_accelerating_frames() -> None:
    # clip 0: braking 36 -> 0 km/h, clip 1: launching from 3.6 km/h,
    # clip 2: cruising, clip 3: standing
    t = 12
    ramp = torch.arange(t, dtype=torch.float32)
    speed = torch.stack([
        (36.0 - 3.6 * ramp).clamp(min=0.0),
        3.6 + 3.6 * ramp,
        torch.full((t,), 50.0),
        torch.zeros(t),
    ])  # 3.6 km/h per 1/3 s tick = 3 m/s^2
    module = ConditionalSpeedShift(min_accel=1.0, min_speed_kmh=2.0)
    v, target = _speed_shift_inputs(speed, episode_length=6)
    slowing, speeding_up = module.flags(v, target)

    assert slowing[0].all()
    assert not speeding_up[0].any()
    assert speeding_up[1].all()
    assert not slowing[1].any()
    assert not (slowing[2:] | speeding_up[2:]).any()


def test_speed_shift_is_rotation_invariant() -> None:
    speed = (36.0 - 3.6 * torch.arange(12.0)).clamp(min=0.0)[None]
    module = ConditionalSpeedShift()
    v, target = _speed_shift_inputs(speed, episode_length=6)
    psi = torch.full(target.shape[:2], 0.1)
    rotated = torch.cat(
        [rotate_ego_frame(target[..., :2], psi), target[..., 2:]], dim=-1
    )
    for a, b in zip(module.flags(v, target), module.flags(v, rotated), strict=True):
        assert torch.equal(a, b)


def test_speed_shift_boosts_braking_and_cuts_accelerating_speed() -> None:
    t = 12
    ramp = torch.arange(t, dtype=torch.float32)
    speed = torch.stack([
        (36.0 - 3.6 * ramp).clamp(min=0.0),
        3.6 + 3.6 * ramp,
        torch.full((t,), 50.0),
    ])
    v, target = _speed_shift_inputs(speed, episode_length=6)
    boost, cut = (1.2, 1.4), (0.6, 0.8)
    module = ConditionalSpeedShift(
        probability=1.0, boost_scale=boost, cut_scale=cut
    ).train()
    out = module(v, target)

    ratio = out / v
    assert ((ratio[0] >= boost[0]) & (ratio[0] <= boost[1])).all()
    assert ((ratio[1] >= cut[0]) & (ratio[1] <= cut[1])).all()
    assert torch.equal(out[2], v[2])
    # one factor per clip
    assert torch.allclose(ratio[0], ratio[0, 0].expand_as(ratio[0]))

    assert torch.equal(module.eval()(v, target), v)
    assert torch.equal(ConditionalSpeedShift(probability=0.0).train()(v, target), v)
