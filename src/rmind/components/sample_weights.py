"""Per-sample draw weights for the train sampler.

`speed_change_weights` over-samples clips in which the logged ego speed
changes: stops, launches and hard acceleration/braking. These are the
situations behind most closed-loop speed failures of the route-free PAI
finetune (front collisions from not braking, rear hits from being too slow);
see /nasa/alex/docs/5U0ELM6G_IMPROVEMENT_PLAN.md, experiment B2.

A weight function takes the built dataset and returns one non-negative weight
per sample; `DistributedTorchDataNodeDataLoader(sample_weights=...)` then
draws with replacement in proportion to it.
"""

from typing import Protocol

import torch
from rbyte.dataloader import BatchIndexableDataset
from structlog import get_logger
from torch import Tensor

logger = get_logger(__name__)

KMH_TO_MS = 1.0 / 3.6


class SampleWeightFn(Protocol):
    def __call__(self, dataset: BatchIndexableDataset) -> Tensor: ...


def speed_change_flags(  # noqa: PLR0913
    speed_kmh: Tensor,
    *,
    dt: float,
    accel_window_s: float = 1.0,
    accel_threshold: float = 1.0,
    stop_speed: float = 1.0,
    moving_speed: float = 3.0,
) -> dict[str, Tensor]:
    """Flag clips with a speed change.

    Args:
        speed_kmh: `(n, clip_length)` logged ego speed, km/h.
        dt: seconds between consecutive clip frames.
        accel_window_s: span of the finite difference used for the
            acceleration, which smooths per-frame speed noise.
        accel_threshold: |acceleration| (m/s^2) that counts as a speed change.
        stop_speed: speed (m/s) below which the ego counts as stopped.
        moving_speed: speed (m/s) above which the ego counts as moving.

    Returns:
        Boolean `(n,)` masks: `accel` (|a| above threshold somewhere), `stop_launch`
        (both stopped and moving within the clip) and `any` (either).
    """
    v = speed_kmh.float() * KMH_TO_MS
    lag = max(1, round(accel_window_s / dt))
    if v.shape[1] > lag:
        accel = (v[:, lag:] - v[:, :-lag]).abs().amax(dim=1) / (lag * dt)
    else:
        accel = torch.zeros(v.shape[0])
    accel_flag = accel > accel_threshold
    stop_launch = (v.amin(dim=1) < stop_speed) & (v.amax(dim=1) > moving_speed)
    return {
        "accel": accel_flag,
        "stop_launch": stop_launch,
        "any": accel_flag | stop_launch,
    }


def speed_change_weights(  # noqa: PLR0913
    dataset: BatchIndexableDataset,
    *,
    target_fraction: float = 0.5,
    speed_key: str = "meta/VehicleMotion/speed",
    dt: float = 1.0 / 3.0,
    accel_window_s: float = 1.0,
    accel_threshold: float = 2.0,
    stop_speed: float = 1.0,
    moving_speed: float = 3.0,
) -> Tensor:
    """Weights so that flagged speed-change clips make up `target_fraction` of draws.

    Flagged and unflagged clips keep uniform weights within their group. If the
    flagged share already reaches `target_fraction` (or either group is empty),
    all weights are 1 and sampling is unchanged.

    The 2 m/s^2 default is chosen for PAI: at 1 m/s^2, 42% of the 2k-slice clips
    are already flagged, so re-weighting to 50% barely changes the draw; at
    2 m/s^2 (plus stops/launches) 17% are flagged, i.e. ~2.9x up-weighting.

    Raises:
        ValueError: if `target_fraction` is outside (0, 1).
    """
    if not 0.0 < target_fraction < 1.0:
        msg = f"target_fraction must be in (0, 1), got {target_fraction}"
        raise ValueError(msg)

    speed = dataset.data[speed_key]  # ty: ignore[unresolved-attribute]
    flags = speed_change_flags(
        speed,
        dt=dt,
        accel_window_s=accel_window_s,
        accel_threshold=accel_threshold,
        stop_speed=stop_speed,
        moving_speed=moving_speed,
    )
    flagged = flags["any"]
    n = flagged.numel()
    n_flagged = int(flagged.sum())
    share = n_flagged / max(n, 1)

    weights = torch.ones(n, dtype=torch.double)
    if 0 < n_flagged < n and share < target_fraction:
        weights[flagged] = target_fraction / share
        weights[~flagged] = (1.0 - target_fraction) / (1.0 - share)
    effective = float(weights[flagged].sum() / weights.sum()) if n else 0.0

    logger.info(
        "speed_change_weights",
        samples=n,
        flagged=n_flagged,
        share_accel=round(float(flags["accel"].float().mean()), 4),
        share_stop_launch=round(float(flags["stop_launch"].float().mean()), 4),
        share_before=round(share, 4),
        share_after=round(effective, 4),
    )
    return weights
