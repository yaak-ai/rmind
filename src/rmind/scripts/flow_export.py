"""Export the flow policy's full inference graph to ONNX / torch.export.

`ControlTransformer.exportable_policy()` -> `ExportableControlPolicy` is

    forward(batch, noise[B,K,H,A_model]) -> raw action draws[B,K,H,A_raw]

= episode builder + encoder + K decoder rollouts (fixed NFE) + inverse action
transform. The raw sensor `batch` goes in and the K candidate action chunks come
out; the winner-take-all readout is deliberately left to host-side postprocessing
(out of the graph). This loads a trained checkpoint, builds that graph, and writes
a deployable artifact — ONNX by default, or a torch.export `.pt2` — so the whole
perception+policy stack ships in one file for edge deployment.

Noise is an INPUT (the host supplies it), so the artifact is a pure function of
its inputs; K and the sampler step count are frozen at export time.

Usage:
    just export-policy model.artifact=yaak/action-flow/model-<run>:vN \\
        '+export.out=artifacts/policy.onnx' '+export.draws=32'

Options (+export.*): out (required; .onnx or .pt2), draws K (default 32),
batch B (default 1), opset (ONNX, default 18).

Graph size / speed knobs — the sampler loop unrolls, so these change the node
count (and on-device latency), frozen into the artifact:
  flow_sampling_steps (NFE; default = the decoder's trained value),
  flow_sampling_method (euler|heun; heun is 2 evals/step). draws is a batch dim —
  it does NOT change the graph.

optimize (default false): the onnxscript optimizer is superlinear on the large
  unrolled graph, so `optimize=true` can take tens of minutes — optimize offline
  instead. verify (default true): ORT-vs-eager parity check at export.
  external_data (default false).
"""

import multiprocessing as mp
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.fx.experimental._config as _fx_config  # noqa: PLC2701
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger
from torch.nn import Module

from rmind.components.objectives.policy import PolicyObjective
from rmind.components.transformer import FlowActionDecoder
from rmind.utils.patch import monkeypatched

# tensordict's global dicts mutated during export tracing cause a spurious
# "pending unbacked symbol u0" error even though the exported graph is valid
# (same workaround as scripts/export_onnx.py and tests/test_export.py).
_fx_config.soft_pending_unbacked_not_found_error = True  # ty:ignore[invalid-assignment]

logger = get_logger(__name__)


def _apply_sampler_overrides(opts: Any, decoder: FlowActionDecoder) -> None:
    """Optionally override the sampler (NFE / method) at export time.

    The sampler loop unrolls into the graph, so fewer steps (or euler over heun,
    which is 2 evals/step) shrink the node count and cut on-device latency — at
    the cost of a coarser ODE solve. Frozen into the artifact, so this is an
    accuracy<->speed choice; validate the sampled actions.

    Raises:
        ValueError: if `flow_sampling_steps` <= 0 or the method is not euler/heun.
    """
    if "flow_sampling_steps" in opts:
        steps = int(opts["flow_sampling_steps"])
        if steps <= 0:
            msg = f"flow_sampling_steps must be positive, got {steps}"
            raise ValueError(msg)
        decoder.flow_sampling_steps = steps
    if "flow_sampling_method" in opts:
        method = str(opts["flow_sampling_method"])
        decoder._validate_sampling_method(method)  # noqa: SLF001
        decoder.flow_sampling_method = method
    logger.info(
        "sampler config",
        flow_sampling_steps=decoder.flow_sampling_steps,
        flow_sampling_method=decoder.flow_sampling_method,
    )


def build_exportable(cfg: DictConfig) -> tuple[Module, tuple[Any, ...]]:
    """Load the checkpoint and build the eager inference graph + example inputs.

    Shared by `main` (which exports it) and `flow_export_compare` (which runs it
    eagerly to check parity against the saved ONNX). Returns
    `(exportable, (batch, noise))`; the attention-mask cache is warmed so the
    graph is both export- and eager-ready.

    Raises:
        KeyError: if `cfg` has no `input` (the dummy batch to trace with).
        TypeError: if the model's "policy" objective is not a `PolicyObjective`.
    """
    opts = cfg.get("export") or {}
    draws = int(opts.get("draws", 32))
    batch = int(opts.get("batch", 1))

    if "input" not in cfg:
        msg = (
            "policy export needs an `input` config (a dummy batch to trace); "
            "use --config-name export/policy_onnx.yaml"
        )
        raise KeyError(msg)

    model = instantiate(cfg.model).to("cpu").eval()
    objective = model.objectives["policy"]
    if not isinstance(objective, PolicyObjective):
        msg = f"policy objective is not a PolicyObjective: {type(objective)}"
        raise TypeError(msg)

    _apply_sampler_overrides(opts, objective.decoder)

    noise = torch.randn(
        batch, draws, objective.decoder.action_horizon, objective.decoder.action_dim
    )
    dummy_batch = instantiate(cfg.input, _recursive_=True, _convert_="all")
    exportable = model.exportable_policy().eval()  # ty:ignore[unresolved-attribute]

    # Warm the (non-trace-friendly) attention-mask cache before torch.export; an
    # eager forward must run first (see episode.py). Two passes mirror the
    # is_exporting toggle in scripts/export_onnx.py.
    with torch.inference_mode():
        for flag in (False, True):
            with monkeypatched(
                obj=torch.compiler, name="_is_exporting_flag", patch=flag
            ):
                exportable(dummy_batch, noise)
    return exportable, (dummy_batch, noise)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    opts = cfg.get("export") or {}
    out = Path(str(opts["out"]))
    opset = int(opts.get("opset", 18))
    # optimize default OFF: the sampler loop unrolls (flow_sampling_steps x decoder
    # stack), so the full graph is large (tens of thousands of nodes) and the
    # onnxscript optimizer/rewriter is superlinear on it — `optimize=true` can run
    # for tens of minutes. Emit the unoptimized graph fast and optimize offline
    # (or via the ORT session's graph optimization at load).
    optimize = bool(opts.get("optimize", False))
    verify = bool(opts.get("verify", True))
    external_data = bool(opts.get("external_data", False))
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))

    exportable, example_args = build_exportable(cfg)

    with torch.inference_mode():
        out_shape = tuple(exportable(*example_args).shape)
    logger.info("built exportable graph", draws=out_shape)

    out.parent.mkdir(parents=True, exist_ok=True)
    # torch.export first (as in export_onnx.py / test_onnx_export): the tensordict
    # batch input lowers more reliably via an ExportedProgram.
    exported_program = torch.export.export(exportable, example_args, strict=True)
    if out.suffix == ".pt2":
        torch.export.save(exported_program, str(out))
    else:
        torch.onnx.export(
            model=exported_program,
            f=str(out),
            dynamo=True,
            opset_version=opset,
            output_names=["action_draws"],
            external_data=external_data,
            optimize=optimize,
            verify=verify,
        )
    logger.info(
        "wrote export",
        path=str(out),
        format=out.suffix.lstrip("."),
        optimize=optimize,
        verify=verify,
    )


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
