from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class GenericDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: Optional[DataLoader] = None,
        val: Optional[DataLoader] = None,
        test: Optional[DataLoader] = None,
        predict: Optional[DataLoader] = None,
    ) -> None:
        super().__init__()

        self._train = train
        self._val = val
        self._test = test
        self._predict = predict

    def train_dataloader(self):
        return self._train

    def val_dataloader(self):
        return self._val

    def test_dataloader(self):
        return self._test

    def predict_dataloader(self):
        return self._predict
