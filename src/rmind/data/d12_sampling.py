"""Training-only hold/transition-balanced sampling; validation stays unweighted."""

from typing import Any, cast

import numpy as np
import torch
from rbyte.dataloader import NodeDataLoader
from rbyte.dataset import Dataset
from structlog import get_logger
from torch.utils.data import WeightedRandomSampler

CATEGORIES = ("ordinary", "hold", "lift_lower", "steering_change", "start_stop")
ACTION_FIELDS = 3
ACTION_RANK = 3
NEAR_HORIZON = 12
ACTIVE_THRESHOLD = 0.05
STEERING_CHANGE_THRESHOLD = 0.15
logger = get_logger(__name__)


def sampling_weights(
    actions: np.ndarray, balanced_fraction: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Mix uniform and inverse-frequency sampling over action categories.

    Raises:
        ValueError: If actions or the balanced fraction are invalid.
    """
    actions = np.asarray(actions)
    if (
        actions.ndim != ACTION_RANK
        or actions.shape[-1] != ACTION_FIELDS
        or not len(actions)
    ):
        msg = "expected nonempty N,H,3 actions"
        raise ValueError(msg)
    if not np.isfinite(actions).all() or not 0 <= balanced_fraction <= 1:
        msg = "invalid actions or mixture fraction"
        raise ValueError(msg)
    near = actions[:, :NEAR_HORIZON]
    active = np.abs(near[..., [0, 2]]) > ACTIVE_THRESHOLD
    labels = np.zeros(len(actions), dtype=np.int64)
    labels[~active.any(axis=(1, 2))] = 1
    labels[active[..., 1].any(axis=1)] = 2
    labels[np.ptp(near[..., 1], axis=1) >= STEERING_CHANGE_THRESHOLD] = 3
    labels[(active[:, 1:] != active[:, :-1]).any(axis=(1, 2))] = 4
    counts = np.bincount(labels, minlength=len(CATEGORIES))
    balanced = 1 / (np.count_nonzero(counts) * counts[labels])
    weights = (1 - balanced_fraction) / len(actions) + balanced_fraction * balanced
    return weights, labels


def balanced_loader(
    dataset: Dataset, balanced_fraction: float = 0.5, seed: int = 1337, **kwargs: Any
) -> NodeDataLoader[Any]:
    actions = np.stack(
        [
            cast("torch.Tensor", dataset.data[name])[:, -1].numpy()
            for name in ("traction", "steering", "fork1")
        ],
        axis=-1,
    )
    weights, labels = sampling_weights(actions, balanced_fraction)
    logger.info(
        "hold_transition_balanced",
        balanced_fraction=balanced_fraction,
        counts={
            name: int((labels == index).sum()) for index, name in enumerate(CATEGORIES)
        },
        probability_mass={
            name: float(weights[labels == index].sum())
            for index, name in enumerate(CATEGORIES)
        },
    )
    sampler = WeightedRandomSampler(
        weights.tolist(),
        len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    kwargs.pop("shuffle", None)
    return NodeDataLoader(dataset=dataset, sampler=sampler, **kwargs)
