from typing import override

import pytorch_lightning as pl
from pydantic import InstanceOf, validate_call
from torch.utils.data import DataLoader


class GenericDataModule[T](pl.LightningDataModule):
    @validate_call
    def __init__(
        self,
        train: InstanceOf[DataLoader[T]] | None = None,
        val: InstanceOf[DataLoader[T]] | None = None,
        test: InstanceOf[DataLoader[T]] | None = None,
        predict: InstanceOf[DataLoader[T]] | None = None,
    ) -> None:
        super().__init__()

        self._train: DataLoader[T] | None = train
        self._val: DataLoader[T] | None = val
        self._test: DataLoader[T] | None = test
        self._predict: DataLoader[T] | None = predict

    @override
    def train_dataloader(self) -> DataLoader[T] | None:
        return self._train

    @override
    def val_dataloader(self) -> DataLoader[T] | None:
        return self._val

    @override
    def test_dataloader(self) -> DataLoader[T] | None:
        return self._test

    @override
    def predict_dataloader(self) -> DataLoader[T] | None:
        return self._predict
