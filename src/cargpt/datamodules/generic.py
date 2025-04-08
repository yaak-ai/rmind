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

        self._train = train
        self._val = val
        self._test = test
        self._predict = predict

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
