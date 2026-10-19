"""Explicit labeled-transition input; no inference of rewards from failed recordings."""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

MATRIX_RANK = 2


def episode_identities(values: np.ndarray) -> frozenset[str]:
    if values.dtype.kind not in {"U", "S"} and not np.issubdtype(
        values.dtype, np.integer
    ):
        message = (
            "Episode IDs must be stable strings or integer identities, not floats/NaN"
        )
        raise ValueError(message)
    identities = values.astype(str).tolist()
    if any(not value.strip() for value in identities):
        message = "Episode IDs must be nonempty stable strings"
        raise ValueError(message)
    return frozenset(identities)


def validate_bootstrap_discounts(arrays: dict[str, np.ndarray]) -> None:
    if "bootstrap_discounts" not in arrays:
        return
    discounts = arrays["bootstrap_discounts"]
    if (
        not np.isfinite(discounts).all()
        or not ((discounts >= 0) & (discounts <= 1)).all()
        or (discounts[arrays["masks"] == 0] != 0).any()
    ):
        message = "Invalid explicit macro bootstrap discounts"
        raise ValueError(message)


def validate_transition_fields(arrays: dict[str, np.ndarray], count: int) -> None:
    if (
        arrays["observations"].ndim != MATRIX_RANK
        or arrays["actions"].ndim != MATRIX_RANK
    ):
        message = "FQL accepts vector observations and flat action vectors only"
        raise ValueError(message)
    if arrays["next_observations"].shape != arrays["observations"].shape:
        message = "Current/next observation shape mismatch"
        raise ValueError(message)
    scalar_fields = ["rewards", "masks", "episode_ids"]
    if "bootstrap_discounts" in arrays:
        scalar_fields.append("bootstrap_discounts")
    if any(arrays[name].shape != (count,) for name in scalar_fields):
        message = "Rewards, masks and episode IDs must be one-dimensional"
        raise ValueError(message)
    if not np.isin(arrays["masks"], (0, 1)).all():
        message = "Bootstrap masks must be explicitly binary"
        raise ValueError(message)
    validate_bootstrap_discounts(arrays)


class FQLTransitions(Dataset[dict[str, Tensor]]):
    def __init__(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "masks",
                "episode_ids",
            }
            if not required.issubset(archive.files):
                message = "Transition archive requires explicit observations/actions/rewards/masks/episode_ids"
                raise ValueError(message)
            arrays = {name: np.array(archive[name], copy=True) for name in required}
            if "bootstrap_discounts" in archive.files:
                arrays["bootstrap_discounts"] = np.array(
                    archive["bootstrap_discounts"], copy=True
                )
        count = len(arrays["actions"])
        if count == 0 or any(len(array) != count for array in arrays.values()):
            message = "Transition arrays must have equal nonzero lengths"
            raise ValueError(message)
        validate_transition_fields(arrays, count)
        self.episode_ids = episode_identities(arrays.pop("episode_ids"))
        self.tensors = {}
        for name, array in arrays.items():
            if (
                not (
                    np.issubdtype(array.dtype, np.integer)
                    or np.issubdtype(array.dtype, np.floating)
                )
                or not np.isfinite(array).all()
            ):
                message = f"Nonfinite or nonnumeric transition field {name}"
                raise ValueError(message)
            tensor = torch.as_tensor(array, dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                message = "Transition values overflow float32"
                raise ValueError(message)
            self.tensors[name] = tensor

    def __len__(self) -> int:
        return len(self.tensors["actions"])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {name: values[index] for name, values in self.tensors.items()}


class FQLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        validation_path: str,
        batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if (
            batch_size <= 0
            or Path(train_path).resolve() == Path(validation_path).resolve()
        ):
            message = "Require positive batch size and distinct transition archives"
            raise ValueError(message)
        self.train_path, self.validation_path = train_path, validation_path
        self.batch_size, self.seed = batch_size, seed

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.train_data = FQLTransitions(self.train_path)
        self.validation_data = FQLTransitions(self.validation_path)
        if self.train_data.episode_ids & self.validation_data.episode_ids:
            message = "Training and validation episode identities overlap"
            raise ValueError(message)
        if len(self.train_data) < self.batch_size:
            message = "Training set has no full batch"
            raise ValueError(message)

    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data, batch_size=self.batch_size, shuffle=False
        )
