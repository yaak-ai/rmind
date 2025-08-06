from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger
from tensordict import TensorDict
from torch.nn import Module
from torch.testing import make_tensor
from torch.utils._pytree import key_get, keystr, tree_flatten_with_path  # noqa: PLC2701

if TYPE_CHECKING:
    from rmind.models.control_transformer import ControlTransformer

logger = get_logger(__name__)


def create_dummy_inputs(device: torch.device) -> tuple[Any, ...]:
    """Create dummy inputs matching the expected input format for the model."""

    # Create dummy batch similar to conftest.py
    dummy_batch = {
        "data": TensorDict(
            {
                "cam_front_left": make_tensor(
                    (1, 6, 324, 576, 3),
                    dtype=torch.uint8,
                    device=device,
                    low=0,
                    high=256,
                ),
                "meta/ImageMetadata.cam_front_left/frame_idx": make_tensor(
                    (1, 6), dtype=torch.int32, device=device, low=0
                ),
                "meta/ImageMetadata.cam_front_left/time_stamp": make_tensor(
                    (1, 6), dtype=torch.int64, device=device, low=0
                ),
                "meta/VehicleMotion/brake_pedal_normalized": make_tensor(
                    (1, 6), dtype=torch.float32, device=device, low=0.0, high=1.0
                ),
                "meta/VehicleMotion/gas_pedal_normalized": make_tensor(
                    (1, 6), dtype=torch.float32, device=device, low=0.0, high=1.0
                ),
                "meta/VehicleMotion/steering_angle_normalized": make_tensor(
                    (1, 6), dtype=torch.float32, device=device, low=-1.0, high=1.0
                ),
                "meta/VehicleMotion/speed": make_tensor(
                    (1, 6), dtype=torch.float32, device=device, low=0.0, high=130.0
                ),
                "meta/VehicleState/turn_signal": make_tensor(
                    (1, 6), dtype=torch.int64, device=device, low=0, high=3
                ),
                "waypoints/waypoints_normalized": make_tensor(
                    (1, 6, 10, 2),
                    dtype=torch.float32,
                    device=device,
                    low=0.0,
                    high=20.0,
                ),
            },
            batch_size=[1],
            device=device,
        ),
        "batch_size": [1],
        "device": device,
    }

    return (dummy_batch,)


def verify_output(
    output: Any,
    reference_items: list[tuple[tuple[Any, ...], Any]],
    rtol: float | None = 0.0,
    atol: float | None = 0.0,
) -> None:
    """Verify that the output matches the reference items."""
    for kp, expected in reference_items:
        actual = key_get(output, kp)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            check_dtype=True,
            msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
        )


def export_model_aoti(
    model: Module, dummy_inputs: tuple[Any, ...], output_path: Path, strict: bool = True
) -> None:
    """Export model using AOT (Ahead of Time) compilation."""
    logger.info("Starting AOT export", model_type=type(model).__name__)

    model = model.eval()

    with torch.inference_mode():
        # Get reference output
        logger.debug("Computing reference output")
        reference_output = model(*dummy_inputs)
        reference_items, _ = tree_flatten_with_path(reference_output)

        # Test export mode output
        logger.debug("Testing export mode")
        torch.compiler._is_exporting_flag = True  # pyright: ignore[reportPrivateUsage]
        try:
            export_output = model(*dummy_inputs)
            breakpoint()

            # Verify export mode output matches
            verify_output(export_output, reference_items)
        finally:
            torch.compiler._is_exporting_flag = False  # pyright: ignore[reportPrivateUsage]

        # Export the model
        logger.info("Exporting model with `torch.export` to", output_path=output_path)
        exported = torch.export.export(model, args=dummy_inputs, strict=strict)

        # Verify exported model output
        logger.debug("Verifying exported model output")
        exported_output = exported.module()(*dummy_inputs)
        verify_output(exported_output, reference_items)

        # Compile and package with AOT
        logger.info("Compiling with AOT Inductor", output_path=output_path)
        package_path = torch._inductor.aoti_compile_and_package(exported)  # pyright: ignore[reportPrivateUsage]

        # Move package to desired location
        if output_path != package_path:
            import shutil

            shutil.move(package_path, output_path)
            logger.info("Package moved", from_path=package_path, to_path=output_path)

        # Verify final package
        logger.debug("Verifying final package")
        package = torch._inductor.aoti_load_package(output_path)  # pyright: ignore[reportPrivateUsage]
        package_output = package(*dummy_inputs)

        verify_output(package_output, reference_items, rtol=None, atol=None)

        logger.info("AOT export completed successfully", output_path=output_path)


@hydra.main(version_base=None)
def export(cfg: DictConfig) -> None:
    """Main export function using Hydra configuration."""
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    logger.info("Instantiating model", target=cfg.model._target_)
    model: ControlTransformer = instantiate(cfg.model)

    device = model.device
    assert device is not None, "Device is not set"

    logger.info("Creating dummy inputs")
    dummy_inputs = create_dummy_inputs(device)

    output_dir = Path(cfg.get("output_dir", "exports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model.__class__.__name__.lower()}.so"
    logger.info("Output path", output_path=output_path)

    # # Export model
    export_model_aoti(
        model=model,
        dummy_inputs=dummy_inputs,
        output_path=output_path,
        strict=cfg.get("export_strict", True),
    )


if __name__ == "__main__":
    import logging

    logging.getLogger("xformers").setLevel(logging.ERROR)

    export()
