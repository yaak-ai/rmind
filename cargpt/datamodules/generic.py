from typing import Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from typing_extensions import override


class CustomizedDataModule(pl.LightningDataModule):
    def __init__(self, **cfg) -> None:
        super().__init__()

        for name in ["train", "val", "test", "predict"]:
            if cfg_ := cfg.get(name, None):
                self._dataset = instantiate(cfg_["dataset"])
                dataloader = instantiate(
                    cfg_["dataloader"],
                    dataset=self._dataset,
                    sampler=instantiate(cfg_["sampler"])(dataset=self._dataset)
                    if "sampler" in cfg_
                    else None,
                )
                self.__setattr__(name, dataloader)
            else:
                self.__setattr__(name, None)

    @override
    def train_dataloader(self):
        return self.train

    @override
    def val_dataloader(self):
        return self.val

    @override
    def test_dataloader(self):
        return self.test

    @override
    def predict_dataloader(self):
        return self.predict


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
