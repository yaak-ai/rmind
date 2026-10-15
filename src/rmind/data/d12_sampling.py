"""Training-only hold/transition-balanced sampling; validation stays unweighted."""

import json

import numpy as np
import torch
from rbyte.dataloader import NodeDataLoader
from torch.utils.data import WeightedRandomSampler

CATEGORIES = ("ordinary", "hold", "lift_lower", "steering_change", "start_stop")


def sampling_weights(actions, balanced_fraction=0.5):
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[-1] != 3 or not len(actions):
        raise ValueError("expected nonempty N,H,3 actions")
    if not np.isfinite(actions).all() or not 0 <= balanced_fraction <= 1:
        raise ValueError("invalid actions or mixture fraction")
    near = actions[:, :12]
    active = np.abs(near[..., [0, 2]]) > 0.05
    labels = np.zeros(len(actions), dtype=np.int64)
    labels[~active.any(axis=(1, 2))] = 1
    labels[active[..., 1].any(axis=1)] = 2
    labels[np.ptp(near[..., 1], axis=1) >= 0.15] = 3
    labels[(active[:, 1:] != active[:, :-1]).any(axis=(1, 2))] = 4
    counts = np.bincount(labels, minlength=len(CATEGORIES))
    balanced = 1 / (np.count_nonzero(counts) * counts[labels])
    weights = (1 - balanced_fraction) / len(actions) + balanced_fraction * balanced
    return weights, labels


def balanced_loader(dataset, balanced_fraction=0.5, seed=1337, **kwargs):
    actions = np.stack(
        [
            dataset.data[name][:, -1].numpy()
            for name in ("traction", "steering", "fork1")
        ],
        axis=-1,
    )
    weights, labels = sampling_weights(actions, balanced_fraction)
    print(
        json.dumps({
            "sampling": "hold_transition_balanced",
            "balanced_fraction": balanced_fraction,
            "counts": {
                name: int((labels == index).sum())
                for index, name in enumerate(CATEGORIES)
            },
            "probability_mass": {
                name: float(weights[labels == index].sum())
                for index, name in enumerate(CATEGORIES)
            },
        }),
        flush=True,
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights),
        len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    kwargs.pop("shuffle", None)
    return NodeDataLoader(dataset=dataset, sampler=sampler, **kwargs)
