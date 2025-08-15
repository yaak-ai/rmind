from pathlib import Path
from typing import Annotated

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import BeforeValidator, InstanceOf
from pydantic.dataclasses import dataclass
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from structlog import get_logger
from tensordict import TensorDict
from torch.testing import make_tensor

from rmind.config import HydraConfig

logger = get_logger(__name__)


@dataclass
class Config:
    model: HydraConfig[LightningModule]
    path: Path
    device: Annotated[InstanceOf[torch.device], BeforeValidator(torch.device)]


# TODO: hydra  # noqa: FIX002
BATCH = TensorDict({  # pyright: ignore[reportArgumentType]
    "data": {
        "cam_front_left": make_tensor(
            (1, 6, 324, 576, 3), dtype=torch.uint8, device="cpu", low=0, high=256
        ),
        "meta/VehicleMotion/brake_pedal_normalized": make_tensor(
            (1, 6), dtype=torch.float32, device="cpu", low=0.0, high=1.0
        ),
        "meta/VehicleMotion/gas_pedal_normalized": make_tensor(
            (1, 6), dtype=torch.float32, device="cpu", low=0.0, high=1.0
        ),
        "meta/VehicleMotion/steering_angle_normalized": make_tensor(
            (1, 6), dtype=torch.float32, device="cpu", low=-1.0, high=1.0
        ),
        "meta/VehicleMotion/speed": make_tensor(
            (1, 6), dtype=torch.float32, device="cpu", low=0.0, high=130.0
        ),
        "meta/VehicleState/turn_signal": make_tensor(
            (1, 6), dtype=torch.int64, device="cpu", low=0, high=3
        ),
        "waypoints/waypoints_normalized": make_tensor(
            (1, 6, 10, 2), dtype=torch.float32, device="cpu", low=0.0, high=20.0
        ),
    }
})


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # pyright: ignore[reportCallIssue]

    logger.debug("instantiating", target=config.model.target)
    model = config.model.instantiate()
    model = model.eval().to(config.device)
    logger.debug(f"instantiated\n{ModelSummary(model)}")  # noqa: G004

    logger.debug("exporting")
    batch = BATCH.to(config.device).to_dict()
    exported_program = torch.export.export(model, (batch,), strict=True)

    logger.debug("compiling")
    path = torch._inductor.aoti_compile_and_package(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        exported_program, package_path=config.path
    )

    logger.debug("compiled", path=path.resolve().as_posix())  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    main()
