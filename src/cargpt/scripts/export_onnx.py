import multiprocessing as mp
import sys

import cargpt
import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict

from cargpt.components.objectives.base import PredictionResultKey
from cargpt.models.control_transformer import ControlTransformer
from cargpt.utils.logging import setup_logging
from torch.export import export


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

    def forward(self, cam_front_left, speed):
        input_d = {
            ("image", "cam_front_left"): cam_front_left,
            ("continuous", "speed"): speed,
        }

        return self.policy.predict_simple(
            self._build_input(input_d),
            # input_d,
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

    def _build_input(self, input_d):
        input = self.example_input_array
        for k, v in input_d.items():
            input[k] = v
        return input


@hydra.main(version_base=None)
def export_onnx(cfg: DictConfig):
    logger.debug("instantiating model", target=cfg.model._target_)
    model: ControlTransformer = instantiate(cfg.model)
    exportable_model = ExportableModule(model=model)
    exportable_model.to(model.device)

    # class InputDataClass:
    #     cam_front_left:

    # onnx_path = Path(cfg.model.artifact L
    onnx_path = "/home/nikita/carGPT/model.onnx"

    cam_front_left = torch.ones(1, 6, 3, 320, 576).to(model.device)
    speed = torch.ones(1, 6, 1).to(model.device)
    input_sample = (cam_front_left, speed)
    input_names = ["cam_front_left", "speed"]
    export(exportable_model, args=input_sample)
    breakpoint()
    # exportable_model(cam_front_left, speed)
    # torch.onnx.dynamo_export(exportable_model, args=input_sample)
    # torch.onnx.export(
    #     exportable_model,
    #     args=input_sample,
    #     f=str(onnx_path),
    #     input_names=input_names,
    #     mode="eager",
    # )
    # torch.compile(exportable_model)
    # # from torch import package
    # torch.onnx.dynamo_export()
    # exportable_model.to_torchscript(
    #     "model.pt", method="trace", example_inputs=(cam_front_left, speed)
    # )

    # package_name = "cargpt"
    # resource_name = "model.pkl"
    # path = "tmp/cargpt.pt"
    # intern_modules = [
    #     "cargpt",
    #     "torchvision",
    #     "einops",
    #     "xformers",
    #     "omegaconf",
    #     "lightning_fabric",
    #     "hydra",
    #     "pytorch_lightning",
    #     "jaxtyping",
    #     "tensordict",
    #     "torchaudio",
    #     "polars",
    #     "triton",
    #     "defusesxml",
    #     "lightning_utilities",
    #     "numpy",
    #     "torchmetrics",
    # ]
    # extern_modules = [
    #     "io",
    #     "sys",
    #     "yaml",
    #     "loguru",
    #     "more_itertools",
    #     "typing_extensions",
    #     "fsspec",
    #     "antlr4",
    #     "PIL",
    #     "packaging",
    # ]
    # with package.PackageExporter(path) as exp:
    #     for module in intern_modules:
    #         exp.intern(f"{module}.**")
    #     for module in extern_modules:
    #         exp.extern(f"{module}.**")
    #     exp.save_pickle(package_name, resource_name, model)


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("spawn", force=True)
    mp.set_forkserver_preload(["rbyte"])
    setup_logging()

    export_onnx()


if __name__ == "__main__":
    main()
