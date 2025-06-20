#! /usr/bin/env python

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from torch.utils._pytree import tree_leaves, tree_map

logger = get_logger(__name__)


@hydra.main(version_base=None)
def export(cfg: DictConfig) -> None:
    with bound_contextvars(stage="py"):
        model = instantiate(cfg.model)
        model.eval()
        logger.debug("model instantiated", target=cfg.model._target_)

        dataset = instantiate(cfg.dataset)
        batch = dataset[[0]]
        data = batch.data

        input = {
            "image": {"cam_front_left": data["cam_front_left"]},
            "continuous": {
                "speed": data["meta/VehicleMotion/speed"],
                "gas_pedal": data["meta/VehicleMotion/gas_pedal_normalized"],
                "brake_pedal": data["meta/VehicleMotion/brake_pedal_normalized"],
                "steering_angle": data["meta/VehicleMotion/steering_angle_normalized"],
            },
            "discrete": {"turn_signal": data["meta/VehicleState/turn_signal"]},
            "context": {"waypoints": data["waypoints/waypoints_normalized"]},
        }

        output_model = model.forward(input)
        logger.debug("forward pass", output=output_model)

    with bound_contextvars(stage="export"):
        exported_program = torch.export.export(mod=model, args=(input,), strict=True)
        logger.info("exported", program=str(exported_program))

    with bound_contextvars(stage="aoti"):
        output_path = torch._inductor.aoti_compile_and_package(exported_program)
        logger.debug("compiled", path=output_path)

        package = torch._inductor.aoti_load_package(output_path)
        logger.debug("loaded")

        with torch.inference_mode():
            output_package = package(input)

    if not all(
        tree_leaves(
            tree_map(
                lambda x, y: torch.allclose(x, y, atol=1e-5),
                output_model,
                output_package,
            )
        )
    ):
        msg = "this aint it chief"
        breakpoint()
        raise RuntimeError(msg)


if __name__ == "__main__":
    export()
