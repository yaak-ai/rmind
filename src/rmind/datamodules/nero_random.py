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
    state_9d_to_quat,
)

__all__ = [
    "CAMERA_NAMES",
    "CAMERA_RESOLUTIONS",
    "DEPTH_CAMERA_NAMES",
    "DEPTH_GRIDS",
    "NeroRandomDataset",
    "nero_random_batch",
]

#: contract §8
CAMERA_NAMES: tuple[str, ...] = ("base", "side_left", "side_right")
#: contract §7.2 -- DIFFERENT native resolutions and aspect ratios per camera
CAMERA_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "base": (1920, 1080),
    "side_left": (1280, 800),
    "side_right": (1280, 800),
}
#: ⚠️ what rbyte actually delivers: each camera ISOTROPICALLY downscaled to its
#: OWN grid `(H, W)`, so the three image tensors do NOT share H/W. Unifying them
#: is rmind's job (`LetterboxResize`), and a consumer that assumes a uniform grid
#: across cameras breaks here rather than silently.
CAMERA_GRIDS: dict[str, tuple[int, int]] = {
    "base": (270, 480),
    "side_left": (300, 480),
    "side_right": (300, 480),
}

#: contract §21.3: the depth stream exists on the OVERHEAD `base` device only --
#: it looks down at the workspace so disparity maps to table position, while the
#: SR side cameras sit near-tangent and localise on-plane objects poorly.
DEPTH_CAMERA_NAMES: tuple[str, ...] = ("base",)
#: ⚠️ contract §21.11: the recorder runs `StereoDepth` at **640x400** and §21.9
#: forbids resizing the stream (a resize scales `fx` but not the stored disparity
#: values, so `fx * baseline / disparity` would be wrong by the resize factor).
#: So the depth tensor is LARGER than the RGB tensors, which are downscaled per
#: §8 -- surprising on first read, and deliberate. Model-side letterboxing to the
#: ViT grid is still fine, because the policy consumes standardised DISPARITY and
#: never performs the metric conversion (§22.3).
DEPTH_GRIDS: dict[str, tuple[int, int]] = {"base": (400, 640)}
#: contract §21.1 default mode; §21.10 DECLARES it per episode, never infers it.
DEPTH_MAX_DISPARITY = 95

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


#: Fraction of synthetic pixels left without a measurement. Generated as
#: connected blobs, not i.i.d. pixels -- stereo drops out in regions at depth
#: discontinuities, which is exactly where a grasped object is (§22.4).
DEPTH_INVALID_FRACTION = 0.15


def _disparity_field(
    shape: tuple[int, ...], height: int, width: int
) -> tuple[Tensor, Tensor]:
    """`(*shape, 1, H, W)` synthetic uint8 disparity plus its validity mask.

    Unlike the RGB streams (pure noise -- there is no synthetic-vision claim
    anywhere in this module) this one is a **smooth low-frequency field**, for a
    specific reason: `DisparityStandardizer` is fitted on it, and the mean/std of
    uniform noise over 0..95 are a fixed constant that would make the fitted
    statistics vacuous. A smooth field at least varies per sample the way a real
    depth map does.

    Invalid regions are generated as **blobs**, not i.i.d. pixels, because that
    is how stereo actually fails -- it drops out in connected regions at depth
    discontinuities (§22.4). Per contract §21.4 the emitted disparity is exactly
    `0` wherever the mask is False, so the two are self-consistent; the policy
    still consumes the explicit mask rather than recomputing it, because a real
    loader may mark additional pixels invalid (confidence threshold, LR check)
    that are not zero.
    """
    low = (8, 12)

    def _smooth(scale: float, offset: float) -> Tensor:
        coarse = torch.rand(*shape, 1, *low) * scale + offset
        return torch.nn.functional.interpolate(
            coarse.reshape(-1, 1, *low),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(*shape, 1, height, width)

    disparity = _smooth(1.0, 0.0).mul(DEPTH_MAX_DISPARITY - 1).add(1).round()
    valid = _smooth(1.0, 0.0) > DEPTH_INVALID_FRACTION
    disparity = torch.where(valid, disparity, torch.zeros_like(disparity))
    return disparity.to(torch.uint8), valid


def nero_random_batch(  # noqa: PLR0913
    *,
    batch_size: int = 2,
    episode_length: int = 6,
    action_horizon: int = 6,
    grids: dict[str, tuple[int, int]] | None = None,
    both_sides: bool = True,
    depth: bool = False,
    depth_grids: dict[str, tuple[int, int]] | None = None,
    depth_present: float = 1.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """One batch in the shapes rbyte actually emits.

    Note what is deliberately NOT normalised away here, because the model has to
    cope with all three:

    * the three cameras are on **different grids** (`CAMERA_GRIDS`);
    * `goal.image.*` is **three separate keys**, each on its own grid;
    * state and action are the contract §5.2 **storage form, 46 per side**
      (quaternions), not the 60-dim 9D form -- the expansion is the model's job.

    `both_sides=False` reproduces the dummy recording: `side_valid = [False, True]`
    (left absent) with the invalid side's state and action zeroed, which is what
    the contract §6.1 mandates and what the policy's mask path must handle.

    ⚠️ **`depth=False` is the default and it is the REALISTIC case** (contract
    §22.5): none of the 104 existing episodes carry depth and the recorder's
    `--depth` is off by default, so a depth-enabled model trains on a mixture
    where the stream is frequently missing. `depth=False` emits **no**
    `disparity.*` key at all -- the "key absent from the batch" case, which is
    different from, and more common than, `depth_present < 1` (the key exists but
    some samples in the batch have no measurement). The policy must survive both.

    The depth keys are drawn **strictly last**, after every other tensor, so that
    enabling depth cannot shift the global RNG stream for the images, poses or
    `camera_cond`. Without that, a depth-on/depth-off comparison would be
    confounded by a different synthetic dataset rather than by the model.
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

    # back to the 46-dim storage form the loader emits
    state_quat = state_9d_to_quat(state)
    action_quat = state_9d_to_quat(action)

    images: dict[str, Tensor] = {}
    for name in CAMERA_NAMES:
        h, w = (grids or CAMERA_GRIDS)[name]
        images[f"image.{name}"] = torch.randint(
            0, 256, (batch_size, episode_length, 3, h, w), dtype=torch.uint8
        )
        # ⚠️ THREE SEPARATE goal keys, not one stacked tensor: rbyte cannot index
        # one stream by several columns and the final frame index differs per
        # camera (199 vs 200 in the dummy).
        images[f"goal.image.{name}"] = torch.randint(
            0, 256, (batch_size, 3, h, w), dtype=torch.uint8
        )

    batch: dict[str, Any] = {
        **images,
        # ⚠️ randomised on purpose: with contract §7 `placeholder: true` the real
        # camera_cond is a CONSTANT (zero intrinsics, identity extrinsics), which
        # would leave this path untested. Randomising exercises it.
        "camera_cond": torch.randn(batch_size, len(CAMERA_NAMES), CAMERA_COND_DIM),
        "state.pose": state_quat,
        "side_valid": valid,
        "action.future_state": action_quat,
        # §6.2 reserves this as an alias and rbyte currently materialises it as a
        # byte-identical duplicate (~199 MB of a ~470 MB TensorDict). The policy
        # reads exactly ONE of the two, so the duplicate is never paid for here.
        "action.commanded": action_quat,
        "goal.xyz": torch.randn(batch_size, NUM_SIDES, 3) * _START_XYZ,
        "align_residual_ms": torch.rand(batch_size, episode_length) * 3.0,
    }

    # ⚠️ EVERYTHING BELOW IS DRAWN LAST, ON PURPOSE. See the docstring: keeping
    # the depth draws after every other tensor is what makes `depth=False`
    # byte-identical to the pre-depth datamodule.
    if depth:
        for name in DEPTH_CAMERA_NAMES:
            h, w = (depth_grids or DEPTH_GRIDS)[name]
            disparity, valid = _disparity_field((batch_size, episode_length), h, w)
            batch[f"disparity.{name}"] = disparity
            batch[f"disparity_valid.{name}"] = valid
        # §22.2: depth's own 13-dim conditioning vector, built from the MONO
        # intrinsics at the recorded depth resolution (§21.11) and the mono
        # camera's own extrinsics -- a genuinely different camera from `base`'s
        # RGB `CAM_A` (96.0 deg HFOV against 73.7 deg), which is exactly why it
        # cannot be a 4th channel on the RGB image (§22.1).
        batch["disparity_cond"] = torch.randn(
            batch_size, len(DEPTH_CAMERA_NAMES), CAMERA_COND_DIM
        )
        # §22.5: per-sample availability. Distinct from the per-PIXEL
        # `disparity_valid.*` mask above -- this one says "this episode has a
        # depth stream at all", that one says "this pixel got a measurement".
        batch["depth_valid"] = torch.rand(batch_size) < depth_present

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
