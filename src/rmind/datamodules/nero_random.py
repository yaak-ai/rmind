"""Structured synthetic nero-arms batches at the EXACT contract §8 shapes.

This exists for smoke runs and tests, not for research. Two properties matter:

1. **The shapes are the real ones** (contract §8): 3 cameras, `(T, 2, 60)` state,
   `(T, H, 2, 60)` action, `(3, 13)` camera conditioning, `(3, 3, H, W)` goal
   images, `(2,)` `side_valid`. A smoke run on the old driving shapes proves
   nothing.
2. **The poses are structured SE(3), not uniform noise.** A residual-VQ fitted to
   uniform-random 9D vectors has nothing to compress: translation and rotation
   error both sit at the data std and "rotation is garbage" becomes
   indistinguishable from the real silent failure of §5.5. So trajectories are
   smooth SE(3) walks driven by a ~14-DOF latent (see `_side_trajectory`), with
   contract-plausible magnitudes (§4), canonicalised quaternions (§5), and an
   action chunk that is the genuine future of the state trajectory (§6.2).

Images are uniform noise -- there is no synthetic-vision claim here, only a
shape/dtype/throughput claim.
"""

from collections.abc import Iterator
from typing import Any, final, override

import torch
from torch import Tensor

from rmind.data.nero import (
    BIMANUAL_DIM,
    CAMERA_COND_DIM,
    NUM_SIDES,
    SIDE_DIM,
    canonicalize_quat,
    pose_quat_to_9d,
    quat_to_rot6d,
)

__all__ = [
    "CAMERA_NAMES",
    "CAMERA_RESOLUTIONS",
    "NeroRandomDataset",
    "nero_random_batch",
]

#: contract §8
CAMERA_NAMES: tuple[str, ...] = ("base", "side_left", "side_right")
#: contract §7.2 -- DIFFERENT resolutions and aspect ratios per camera
CAMERA_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "base": (1920, 1080),
    "side_left": (1280, 800),
    "side_right": (1280, 800),
}

#: contract §4: start-pose spread std ~[0.037, 0.021, 0.009] m, within-episode
#: motion range ~[0.32, 0.19, 0.06] m -- i.e. episodes START in a narrow region
#: and MOVE a lot. That ratio is what makes the space compressible at all, so it
#: is reproduced here rather than sampling starts uniformly.
_START_XYZ = torch.tensor([0.30, 0.10, 0.05])
_START_SPREAD_XYZ = torch.tensor([0.037, 0.021, 0.009])
_STEP_XYZ = torch.tensor([0.010, 0.006, 0.002])  # per 30 Hz frame
#: nominal fingertip offsets from the hub (a hand in a neutral grasp posture)
_FINGER_NOMINAL = torch.tensor([
    [0.04, -0.03, 0.02],  # thumb
    [0.08, -0.01, 0.01],  # index
    [0.08, 0.01, 0.00],  # middle
    [0.07, 0.03, -0.01],  # ring
    [0.06, 0.05, -0.02],  # little
])
#: per-episode fingertip jitter. Deliberately SMALL: on a real hand the
#: fingertip offset is determined by the finger's curl, not independently random
#: per episode. Large i.i.d. jitter here would make the 15 finger translation
#: dims incompressible and the translation metric uninformative.
_FINGER_SPREAD = 0.001
#: direction a fingertip travels as its curl scalar goes 0 -> 1, and the axis it
#: rotates about. One scalar per finger drives both -- fingers are kinematically
#: coupled, which is what makes the 45 finger dims compressible.
_FINGER_CURL_DIRECTION = torch.tensor([
    [-0.030, 0.020, -0.025],
    [-0.045, 0.005, -0.030],
    [-0.048, 0.000, -0.032],
    [-0.042, -0.005, -0.028],
    [-0.035, -0.010, -0.024],
])
_FINGER_AXIS = torch.tensor([
    [0.0, 1.0, 0.3],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, -0.1],
    [0.0, 1.0, -0.2],
])


def _axis_angle_quat(axis: Tensor, angle: Tensor) -> Tensor:
    """`(..., 3)` unit axis and `(...)` angle -> canonical quaternion `(..., 4)`."""
    # NOTE: a NEW name -- `axis` is typically an `expand`ed view, on which an
    # in-place op is either an error or silent corruption.
    unit = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    half = angle[..., None] / 2
    return canonicalize_quat(torch.cat([unit * half.sin(), half.cos()], dim=-1))


def _smooth_walk(
    steps: int, dim: int, generator: torch.Generator, *, scale: float
) -> Tensor:
    """`(steps, dim)` zero-mean smooth random walk (cumulative Gaussian increments)."""
    return (scale * torch.randn(steps, dim, generator=generator)).cumsum(0)


def _side_trajectory(steps: int, generator: torch.Generator) -> Tensor:
    """`(steps, 60)` per-side state trajectory in the contract §6.1 layout.

    ⚠️ Generated from a LOW-DIMENSIONAL latent, deliberately. A human hand has
    ~20 DOF; the contract's 60-dim 9D encoding is a redundant overparameterisation
    of that, and it is exactly this redundancy that a VQ codebook exploits.
    Sampling all 60 dims independently would make the action space incompressible
    -- the tokenizer would score at the mean baseline and the reported numbers
    would say nothing about the recipe. The latent here is:

        6  arm pose (translation walk + orientation walk)
        3  hub orientation walk
        5  per-finger curl (one scalar per finger, driving both the fingertip
           offset and its orientation -- i.e. fingers are kinematically coupled
           to their curl, as on a real hand)
    """
    # --- arm in world (contract §4 magnitudes)
    arm_start = _START_XYZ + _START_SPREAD_XYZ * torch.randn(3, generator=generator)
    arm_t = arm_start + _smooth_walk(steps, 3, generator, scale=1.0) * _STEP_XYZ
    arm_axis = torch.randn(3, generator=generator)
    arm_angle = 0.35 * torch.randn(1, generator=generator) + _smooth_walk(
        steps, 1, generator, scale=0.03
    ).squeeze(-1)
    arm_q = _axis_angle_quat(arm_axis.expand(steps, 3), arm_angle)
    columns: list[Tensor] = [pose_quat_to_9d(torch.cat([arm_t, arm_q], dim=-1))]

    # --- fingers, each driven by ONE curl scalar
    curl = 0.6 * torch.rand(5, generator=generator)[None, :] + _smooth_walk(
        steps, 5, generator, scale=0.02
    )
    for finger in range(5):
        c = curl[:, finger : finger + 1]
        translation = (
            _FINGER_NOMINAL[finger]
            + c * _FINGER_CURL_DIRECTION[finger]
            + _FINGER_SPREAD * torch.randn(1, 3, generator=generator)
        )
        rotation = _axis_angle_quat(
            _FINGER_AXIS[finger].expand(steps, 3), c.squeeze(-1) * 1.2
        )
        columns.append(pose_quat_to_9d(torch.cat([translation, rotation], dim=-1)))

    # --- hub orientation (rotation only)
    hub_axis = torch.randn(3, generator=generator)
    hub_angle = 0.25 * torch.randn(1, generator=generator) + _smooth_walk(
        steps, 1, generator, scale=0.02
    ).squeeze(-1)
    columns.append(
        quat_to_rot6d(_axis_angle_quat(hub_axis.expand(steps, 3), hub_angle))
    )

    return torch.cat(columns, dim=-1)


def nero_random_batch(  # noqa: PLR0913
    *,
    batch_size: int = 2,
    episode_length: int = 6,
    action_horizon: int = 6,
    image_size: int = 224,
    both_sides: bool = True,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """One batch at the contract §8 shapes, images already letterboxed to a square.

    `both_sides=False` reproduces the dummy recording: `side_valid = [False, True]`
    (left absent) with the invalid side's state and action zeroed, which is what
    the contract §6.1 mandates and what the policy's mask path must handle.
    """
    generator = torch.Generator().manual_seed(seed)
    steps = episode_length + action_horizon

    trajectories = torch.stack([
        torch.stack(
            [_side_trajectory(steps, generator) for _ in range(NUM_SIDES)], dim=1
        )
        for _ in range(batch_size)
    ])  # (b, steps, 2, 60)

    state = trajectories[:, :episode_length]  # (b, T, 2, 60)
    action = torch.stack(
        [
            trajectories[:, t + 1 : t + 1 + action_horizon]
            for t in range(episode_length)
        ],
        dim=1,
    )  # (b, T, H, 2, 60) -- §6.2: action(t) = future states [t+1 .. t+H]

    valid = torch.ones(batch_size, NUM_SIDES, dtype=torch.bool)
    if not both_sides:
        valid[:, 0] = False
        state = state * valid[:, None, :, None]  # noqa: PLR6104
        action = action * valid[:, None, None, :, None]  # noqa: PLR6104

    images = {
        f"image.{name}": torch.randint(
            0,
            256,
            (batch_size, episode_length, 3, image_size, image_size),
            dtype=torch.uint8,
        )
        for name in CAMERA_NAMES
    }

    batch: dict[str, Any] = {
        **images,
        "goal.image": torch.randint(
            0,
            256,
            (batch_size, len(CAMERA_NAMES), 3, image_size, image_size),
            dtype=torch.uint8,
        ),
        # ⚠️ randomised on purpose: with contract §7 `placeholder: true` the real
        # camera_cond is a CONSTANT (zero intrinsics, identity extrinsics), which
        # would leave this path untested. Randomising exercises it.
        "camera_cond": torch.randn(batch_size, len(CAMERA_NAMES), CAMERA_COND_DIM),
        "state.pose": state,
        "side_valid": valid,
        "action.future_state": action,
        "goal.xyz": torch.randn(batch_size, NUM_SIDES, 3) * _START_XYZ,
        "align_residual_ms": torch.rand(batch_size, episode_length) * 3.0,
    }
    return {k: v.to(device) for k, v in batch.items()}


@final
class NeroRandomDataset(torch.utils.data.IterableDataset):
    """Infinite stream of `nero_random_batch` batches (already collated)."""

    def __init__(
        self, *, num_batches: int = 1000, seed: int = 0, **kwargs: Any
    ) -> None:
        super().__init__()
        self.num_batches = num_batches
        self.seed = seed
        self.kwargs = kwargs

    def __len__(self) -> int:
        return self.num_batches

    @override
    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(self.num_batches):
            yield nero_random_batch(seed=self.seed + i, **self.kwargs)


@final
class NeroRandomDataLoader:
    """Trivial `DataLoader`-shaped wrapper (the dataset already yields batches)."""

    def __init__(
        self, *, num_batches: int = 1000, seed: int = 0, **kwargs: Any
    ) -> None:
        self.dataset = NeroRandomDataset(num_batches=num_batches, seed=seed, **kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.dataset)


assert SIDE_DIM * NUM_SIDES == BIMANUAL_DIM  # noqa: S101
