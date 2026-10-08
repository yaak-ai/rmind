"""Synthetic robot-native nero batches in the EXACT rbyte `NeroRobotWindowGrouper` schema.

For tests, the export gates and smoke runs ONLY -- no claim about real data.
Keys and shapes match what rbyte emits per sample (`T` frames on the 10 Hz grid,
`H = chunk_size` 30 Hz steps, `S = 2` sides, `A = 13`):

| key                       | shape / dtype                |
|---------------------------|------------------------------|
| `image.{camera}`          | `(T, 3, h, w)` uint8         |
| `state`                   | `(T, S, A)` float32          |
| `action.chunk`            | `(T, H, S, A)` float32       |
| `action.is_pad`           | `(T, H)` bool                |
| `side_valid`              | `(S,)` bool, `[True, False]` |
| `hand.current/pos_err/pos`| `(T, 6)` float32             |
| `hand.tip`                | `(T, 10)` float32            |
| `hand.age`                | `(T,)` float32               |
| `hand.motor_ok/tip_ok`    | `(T,)` bool                  |
| `camera_cond`             | `(3, 13)` float32 (zeros = placeholder) |

BIMANUAL (`bimanual=True`, rbyte `nero-bimanual-26`): `side_valid [True, True]`,
the right side's state / chunk filled from an independent draw (its own contact
time, fingers and hand stream; same padding pattern), and the hand blocks per
side as `hand.left.*` / `hand.right.*` instead of `hand.*`. The left side and
the images are BIT-identical to the single-arm batch of the same seed (the right
side comes from a separate generator), so `bimanual=False` is unchanged.

THE HAND-DEPENDENT TASK (`hand_dependent=True`). Each sample has a hidden
"contact" time; the fingers start closing `grasp_delay_steps` (0.5 s) AFTER it,
to a per-finger plateau (850-1000 counts) from an open atom (~50 counts). The
only input that shows the contact before the fingers move is the hand token's
`current` group (it steps up at contact) -- images are noise and the state's
`hand_prev` only moves once the closing has begun. So a policy that reads the
hand token can place the close onset inside the chunk; one that cannot only
learns the marginal. This is what the reliance metrics are expected to detect
(acceptance: hand ablation delta > 0 on this task).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, cast, final, override

import torch
from torch import Tensor

__all__ = ["NeroRobotRandomDataLoader", "nero_robot_batch"]

CAMERAS = ("base", "side_left", "side_right")
#: the model grid: rbyte's `TransformedSource(NeroImagePreprocess)` emits it
IMAGE_HW = (140, 224)
OPEN_ATOM = 0.05
SAG = 0.02  # steady-state arm tracking offset (command - measured), rad


def _smooth(noise: Tensor, width: int) -> Tensor:
    kernel = torch.ones(1, 1, width) / width
    x = noise.transpose(-1, -2).reshape(-1, 1, noise.shape[-2])
    out = torch.nn.functional.conv1d(x, kernel, padding=width // 2)[
        ..., : noise.shape[-2]
    ]
    return out.reshape(noise.shape[0], noise.shape[-1], -1).transpose(-1, -2)


def nero_robot_batch(  # noqa: PLR0913, PLR0914
    *,
    batch_size: int = 2,
    num_frames: int = 6,
    chunk_size: int = 100,
    frame_stride: int = 3,
    image_hw: tuple[int, int] | Mapping[str, tuple[int, int]] = IMAGE_HW,
    hand_dependent: bool = True,
    grasp_delay_steps: int = 15,
    hand_invalid_frac: float = 0.1,
    pad_tail: bool = True,
    images: bool = True,
    bimanual: bool = False,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """One collated batch (see the module docstring)."""
    g = torch.Generator().manual_seed(seed)
    b, t, h, s = batch_size, num_frames, chunk_size, frame_stride
    steps = (t - 1) * s + h  # 30 Hz steps covered by the window's chunks

    # --- arm: smooth random walk around a pose; measured lags the command
    arm = _smooth(torch.randn(b, steps + 2, 7, generator=g) * 0.08, 15).cumsum(1)
    arm += torch.randn(b, 1, 7, generator=g) * 0.5
    command_arm = arm[:, 2:]
    measured_arm = arm[:, :-2] - SAG

    # --- fingers: open atom -> close after contact (+ delay) -> plateau
    contact = torch.randint(0, (t - 1) * s + 1, (b,), generator=g)
    if not hand_dependent:
        contact = torch.full((b,), steps + 10)
    plateau = 0.85 + 0.15 * torch.rand(b, 6, generator=g)
    k = torch.arange(steps)
    ramp = ((k[None] - (contact[:, None] + grasp_delay_steps)) / 6.0).clamp(0, 1)
    fingers = OPEN_ATOM + ramp[..., None] * (plateau[:, None] - OPEN_ATOM)
    fingers = (fingers * 1000).round() / 1000  # integer counts / 1000
    command = torch.cat([command_arm, fingers], dim=-1)  # (b, steps, 13)
    hand_prev = torch.cat([fingers[:, :1], fingers[:, :-1]], dim=1)

    # --- per frame: state and chunk (frame i sits at 30 Hz step i * s)
    idx = torch.arange(t) * s
    state = torch.zeros(b, t, 2, 13)
    state[:, :, 0, :7] = measured_arm[:, idx]
    state[:, :, 0, 7:] = hand_prev[:, idx]
    window = idx[:, None] + torch.arange(h)[None]  # (t, h)
    chunk = torch.zeros(b, t, h, 2, 13)
    chunk[:, :, :, 0] = command[:, window]
    is_pad = torch.zeros(b, t, h, dtype=torch.bool)
    if pad_tail:
        # the episode "ends" inside the last frame's chunk for half the samples:
        # hold the last command and flag the tail
        for i in range(0, b, 2):
            end = (t - 1) * s + h // 2  # absolute step of the last real command
            pad = window > end
            is_pad[i] = pad
            held = command[i, end]
            chunk[i, :, :, 0] = torch.where(pad[..., None], held, chunk[i, :, :, 0])

    # --- hand token blocks (newest sample): current steps up at contact
    contact_frame = (idx[None] >= contact[:, None]).float()  # (b, t)
    current = contact_frame[..., None] * (0.4 + 0.2 * torch.rand(b, 1, 6, generator=g))
    current += 0.01 * torch.randn(b, t, 6, generator=g)
    pos = hand_prev[:, idx] - 0.01
    pos_err = hand_prev[:, idx] - pos
    motor_ok = torch.rand(b, t, generator=g) >= hand_invalid_frac
    age = (torch.rand(b, t, generator=g) * 0.8).float()
    zero = ~motor_ok[..., None]
    batch: dict[str, Any] = {
        "state": state,
        "action.chunk": chunk,
        "action.is_pad": is_pad,
        "side_valid": torch.tensor([True, False]).expand(b, 2).clone(),
        "hand.current": current.masked_fill(zero, 0.0),
        "hand.pos_err": pos_err.masked_fill(zero, 0.0),
        "hand.pos": pos.masked_fill(zero, 0.0),
        "hand.tip": torch.zeros(b, t, 10),
        "hand.age": age.masked_fill(~motor_ok, 0.0),
        "hand.motor_ok": motor_ok,
        "hand.tip_ok": torch.zeros(b, t, dtype=torch.bool),
        "camera_cond": torch.zeros(b, len(CAMERAS), 13),
    }
    if images:
        for camera in CAMERAS:
            hh, ww = (
                cast("Mapping[str, tuple[int, int]]", image_hw)[camera]
                if isinstance(image_hw, Mapping)
                else image_hw
            )
            batch[f"image.{camera}"] = torch.randint(
                0, 256, (b, t, 3, hh, ww), dtype=torch.uint8, generator=g
            )
    if bimanual:
        batch = _add_right_side(
            batch,
            nero_robot_batch(
                batch_size=b,
                num_frames=t,
                chunk_size=h,
                frame_stride=s,
                hand_dependent=hand_dependent,
                grasp_delay_steps=grasp_delay_steps,
                hand_invalid_frac=hand_invalid_frac,
                pad_tail=pad_tail,
                images=False,
                seed=seed + RIGHT_SEED_OFFSET,
            ),
        )
    return {k: v.to(device) for k, v in batch.items()}


#: the right side's independent draw (any offset that no test seed collides with)
RIGHT_SEED_OFFSET = 1_000_003


def _add_right_side(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """A single-arm batch + another one's LEFT side as the right side."""
    out = {k: v for k, v in left.items() if not k.startswith("hand.")}
    out["state"] = left["state"].clone()
    out["state"][:, :, 1] = right["state"][:, :, 0]
    out["action.chunk"] = left["action.chunk"].clone()
    out["action.chunk"][..., 1, :] = right["action.chunk"][..., 0, :]
    out["side_valid"] = torch.ones_like(left["side_valid"])
    for side, source in (("left", left), ("right", right)):
        for key, value in source.items():
            if key.startswith("hand."):
                out[f"hand.{side}.{key.removeprefix('hand.')}"] = value
    return out


class _Dataset(torch.utils.data.IterableDataset):
    def __init__(self, *, num_batches: int, seed: int, **kwargs: Any) -> None:
        super().__init__()
        self.num_batches, self.seed, self.kwargs = num_batches, seed, kwargs

    def __len__(self) -> int:
        return self.num_batches

    @override
    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(self.num_batches):
            yield nero_robot_batch(seed=self.seed + i, **self.kwargs)


@final
class NeroRobotRandomDataLoader:
    """`DataLoader`-shaped wrapper (the dataset already yields collated batches)."""

    def __init__(
        self, *, num_batches: int = 1000, seed: int = 0, **kwargs: Any
    ) -> None:
        self.dataset = _Dataset(num_batches=num_batches, seed=seed, **kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.dataset)
