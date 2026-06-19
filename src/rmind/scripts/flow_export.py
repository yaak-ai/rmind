"""Export the flow policy's deterministic inference graph (ONNX / torch.export).

`PolicyObjective.exportable()` -> `ExportablePolicy` is

    forward(condition_tokens[B,S,D], noise[B,K,H,A_model]) -> raw draws[B,K,H,A_raw]

= K decoder rollouts (fixed NFE) + the inverse action transform, with the
winner-take-all readout deliberately left to host-side postprocessing (out of the
graph). This loads a trained checkpoint, builds that graph, and writes a
deployable artifact — ONNX by default, or a torch.export `.pt2`.

Noise is an INPUT (the host supplies it), so the artifact is a pure function of
its inputs; K and the sampler step count are frozen at export time.

Usage:
    just export-policy inference=yaak/control_transformer/policy \\
        model.artifact=yaak/action-flow/model-<run>:vN \\
        '+export.out=artifacts/policy.onnx' '+export.draws=32'

Options (+export.*): out (required; .onnx or .pt2), draws K (default 32),
condition_tokens S (default 12 = 2 summary + 10 waypoint tokens), batch B
(default 1), opset (ONNX, default 18).
"""

import multiprocessing as mp
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger

from rmind.components.objectives.policy import PolicyObjective
from rmind.components.transformer import FlowActionDecoder

logger = get_logger(__name__)


def _condition_dim(decoder: FlowActionDecoder) -> int:
    """The decoder's condition (encoder-embedding) dim, from its input projection."""
    projection = decoder.condition_projection
    if isinstance(projection, torch.nn.Linear):
        return projection.in_features
    return decoder.action_projection.out_features  # Identity => dim_model


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: PLR0914
    opts = cfg.get("export") or {}
    out = Path(str(opts["out"]))
    draws = int(opts.get("draws", 32))
    condition_tokens = int(opts.get("condition_tokens", 12))
    batch = int(opts.get("batch", 1))
    opset = int(opts.get("opset", 18))
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))

    model = instantiate(cfg.model).to("cpu").eval()
    objective = model.objectives["policy"]
    if not isinstance(objective, PolicyObjective):
        msg = f"policy objective is not a PolicyObjective: {type(objective)}"
        raise TypeError(msg)
    exportable = objective.exportable().eval()

    dim = _condition_dim(objective.decoder)
    horizon = objective.decoder.action_horizon
    action_dim = objective.decoder.action_dim
    condition = torch.randn(batch, condition_tokens, dim)
    noise = torch.randn(batch, draws, horizon, action_dim)
    with torch.inference_mode():
        out_shape = tuple(exportable(condition, noise).shape)
    logger.info(
        "built exportable graph",
        condition=(batch, condition_tokens, dim),
        noise=(batch, draws, horizon, action_dim),
        draws=out_shape,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".pt2":
        torch.export.save(torch.export.export(exportable, (condition, noise)), str(out))
    else:
        torch.onnx.export(
            exportable,
            (condition, noise),
            str(out),
            dynamo=True,
            opset_version=opset,
            input_names=["condition_tokens", "noise"],
            output_names=["action_draws"],
        )
    logger.info("wrote export", path=str(out), format=out.suffix.lstrip("."))


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
