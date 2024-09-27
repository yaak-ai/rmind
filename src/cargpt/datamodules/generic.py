from typing import Any, override

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class GenericDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: DataLoader[Any] | None = None,
        val: DataLoader[Any] | None = None,
        test: DataLoader[Any] | None = None,
        predict: DataLoader[Any] | None = None,
    ) -> None:
        super().__init__()

        self._train = train
        self._val = val
        self._test = test
        self._predict = predict

    @override
    def train_dataloader(self):
        return self._train

    @override
    def val_dataloader(self):
        return self._val

    @override
    def test_dataloader(self):
        return self._test

    @override
    def predict_dataloader(self):
        return self._predict
