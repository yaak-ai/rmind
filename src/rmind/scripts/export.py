import shutil
from pathlib import Path

import torch
from structlog import get_logger
from torch.utils._pytree import tree_flatten_with_path

from rmind.export.fixtures import batch, batch_dict
from rmind.export.utils import init_model, verify_output

logger = get_logger(__name__)

# NOTE: if pos_emb != 6, set_seed() is necessary in front of each model call


def main() -> None:
    torch.set_float32_matmul_precision("high")
    output_dir = Path("exported")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "control_transformer.so"

    logger.info("Starting AOT export")
    dummy_input = batch_dict(batch())

    logger.info("Computing reference output")
    model = init_model().eval()

    with torch.inference_mode():
        logger.info("Computing export mode output")
        out_orig = model(dummy_input)
        out_orig_items, _ = tree_flatten_with_path(out_orig)
        torch.compiler._is_exporting_flag = True  # pyright: ignore[reportPrivateUsage]
        out_export = model(dummy_input)
        torch.compiler._is_exporting_flag = False  # pyright: ignore[reportPrivateUsage]
        verify_output(out_export, out_orig_items)

        logger.info("Exporting model with `torch.export`", output_path=output_path)
        exported = torch.export.export(model, args=(dummy_input,), strict=True)

        logger.debug("Verifying exported model output")
        exported_output = exported.module()(dummy_input)
        verify_output(exported_output, out_orig_items)

        logger.info("Compiling with AOT Inductor", output_path=output_path)
        package_path = torch._inductor.aoti_compile_and_package(exported)  # pyright: ignore[reportPrivateUsage]
        package_path = Path(package_path)

        # Move package to desired location
        if output_path != package_path:
            shutil.move(str(package_path), str(output_path))  # pyright: ignore[reportUnusedCallResult]
            logger.info("Package moved", from_path=package_path, to_path=output_path)

        logger.debug("Verifying final package")
        package = torch._inductor.aoti_load_package(output_path)  # pyright: ignore[reportPrivateUsage]
        package_output = package(dummy_input)
        verify_output(package_output, out_orig_items, rtol=None, atol=None)

    logger.info("AOT export completed successfully", output_path=output_path)


if __name__ == "__main__":
    main()
