"""Explicit labeled-transition input; no inference of rewards from failed recordings."""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

MATRIX_RANK = 2


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
        count = len(arrays["actions"])
        if count == 0 or any(len(array) != count for array in arrays.values()):
            message = "Transition arrays must have equal nonzero lengths"
            raise ValueError(message)
        if (
            arrays["observations"].ndim != MATRIX_RANK
            or arrays["actions"].ndim != MATRIX_RANK
        ):
            message = (
                "FQL currently accepts vector observations and single actions only"
            )
            raise ValueError(message)
        if arrays["next_observations"].shape != arrays["observations"].shape:
            message = "Current/next observation shape mismatch"
            raise ValueError(message)
        if any(
            arrays[name].shape != (count,)
            for name in ("rewards", "masks", "episode_ids")
        ):
            message = "Rewards, masks and episode IDs must be one-dimensional"
            raise ValueError(message)
        if not np.isin(arrays["masks"], (0, 1)).all():
            message = "Bootstrap masks must be explicitly binary"
            raise ValueError(message)
        self.episode_ids = frozenset(arrays.pop("episode_ids").tolist())
        self.tensors = {}
        for name, array in arrays.items():
            if (
                not np.issubdtype(array.dtype, np.number)
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
