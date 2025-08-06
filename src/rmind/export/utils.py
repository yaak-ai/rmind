"""Utilities for model export operations."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from structlog import get_logger
from torch.nn import Module
from torch.utils._pytree import key_get, keystr, tree_flatten_with_path

from rmind.export.fixtures import (
    batch,
    batch_dict,
    control_transformer,
    encoder,
    episode,
    episode_builder,
    objectives,
    policy_mask,
    policy_objective,
    tokenizers,
)

logger = get_logger(__name__)


def set_seeds(seed_value: int = 42) -> None:
    """Set all random seeds for reproducible results."""
    _ = torch.manual_seed(seed_value)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Force single-threaded CPU execution for determinism
    torch.set_num_threads(1)

    # Optional: GPU determinism settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    model: Module,
    dummy_inputs: tuple[Any, ...],
    output_path: Path,
    strict: bool = True,
    verify_outputs: bool = True,
) -> None:
    """Export model using AOT (Ahead of Time) compilation.

    Args:
        model: The PyTorch module to export
        dummy_inputs: Tuple of dummy inputs for the model
        output_path: Path where to save the compiled package
        strict: Whether to use strict mode for torch.export
        verify_outputs: Whether to verify outputs match between stages
    """
    logger.info("Starting AOT export", model_type=type(model).__name__)

    model = model.eval()

    with torch.inference_mode():
        reference_items = []

        if verify_outputs:
            # Get reference output
            logger.debug("Computing reference output")
            set_seeds()
            reference_output = model(*dummy_inputs)
            reference_items, _ = tree_flatten_with_path(reference_output)

            # Test export mode output
            logger.debug("Testing export mode")
            set_seeds()
            torch.compiler._is_exporting_flag = True  # pyright: ignore[reportPrivateUsage]
            try:
                export_output = model(*dummy_inputs)
                verify_output(export_output, reference_items)
            finally:
                torch.compiler._is_exporting_flag = False  # pyright: ignore[reportPrivateUsage]

        # Export the model
        logger.info("Exporting model with `torch.export`", output_path=output_path)
        exported = torch.export.export(model, args=dummy_inputs, strict=strict)

        if verify_outputs:
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

        if verify_outputs:
            # Verify final package
            logger.debug("Verifying final package")
            package = torch._inductor.aoti_load_package(output_path)  # pyright: ignore[reportPrivateUsage]
            package_output = package(*dummy_inputs)
            verify_output(package_output, reference_items, rtol=None, atol=None)

        logger.info("AOT export completed successfully", output_path=output_path)


# TODO: change hparams of encoder
def init_model() -> Module:
    batch_dict_ = batch_dict(batch())
    episode_builder_ = episode_builder(tokenizers())
    episode_ = episode(episode_builder_, batch_dict_)
    policy_mask_ = policy_mask(episode_)
    encoder_ = encoder()
    objectives_ = objectives(policy_objective(encoder_, policy_mask_))
    return control_transformer(episode_builder_, objectives_)
