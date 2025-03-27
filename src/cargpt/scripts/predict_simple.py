import multiprocessing as mp
import sys

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.export import export

import cargpt
from cargpt.components.objectives.base import PredictionResultKey
from cargpt.models.control_transformer import ControlTransformer
from cargpt.utils.logging import setup_logging


@hydra.main(version_base=None)
def predict_simple(cfg: DictConfig):
    logger.debug("instantiating model", target=cfg.model._target_)
    model = instantiate(cfg.model)

    # logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    # datamodule = instantiate(cfg.datamodule)

    # logger.debug("instantiating trainer", target=cfg.trainer._target_)
    # trainer = instantiate(cfg.trainer)

    logger.debug("starting prediction")
    exportable_model = ExportableModule(model=model)
    exportable_model.to(model.device)
    res = exportable_model(input_d={})
    breakpoint()

    # return trainer.predict(model=model, datamodule=datamodule, return_predictions=False)


class ExportableModule(pl.LightningModule):
    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        model.eval()
        self.policy = model.objectives["policy"]
        self.encoder = model.encoder
        self.episode_builder = model.episode_builder
        self._device = "cuda:0"
        self.example_input_array = TensorDict(
            {
                "continuous": TensorDict(
                    {
                        "brake_pedal": torch.zeros(
                            (1, 6, 1), device=self._device, dtype=torch.float64
                        ),
                        "gas_pedal": torch.zeros(
                            (1, 6, 1), device=self._device, dtype=torch.float64
                        ),
                        "speed": torch.zeros(
                            (1, 6, 1), device=self._device, dtype=torch.float64
                        ),
                        "steering_angle": torch.zeros(
                            (1, 6, 1), device=self._device, dtype=torch.float64
                        ),
                    },
                    batch_size=[1, 6],
                    device=self._device,
                ),
                "discrete": TensorDict(
                    {
                        "turn_signal": torch.zeros(
                            (1, 6, 1), device=self._device, dtype=torch.int64
                        )
                    },
                    batch_size=[1, 6],
                    device=self._device,
                ),
                "image": TensorDict(
                    {
                        "cam_front_left": torch.zeros(
                            (1, 6, 3, 320, 576),
                            device=self._device,
                            dtype=torch.float32,
                        )
                    },
                    batch_size=[1, 6],
                    device=self._device,
                ),
            },
            batch_size=[1, 6],
            device=self._device,
        )

    def forward(self, input_d: dict = {}):
        return self.policy.predict(
            self._build_input(input_d),
            episode_builder=self.episode_builder,
            encoder=self.encoder,
            result_keys=frozenset((
                PredictionResultKey.GROUND_TRUTH,
                PredictionResultKey.PREDICTION,
                PredictionResultKey.PREDICTION_STD,
                PredictionResultKey.PREDICTION_PROBS,
                PredictionResultKey.SCORE_LOGPROB,
                PredictionResultKey.SCORE_L1,
            )),
        )

    def _build_input(self, input_d: dict):
        input: TensorDict = self.example_input_array.clone()
        input.update(input_d)
        return input


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("spawn", force=True)
    mp.set_forkserver_preload(["torch"])
    setup_logging()

    predict_simple()


if __name__ == "__main__":
    main()
