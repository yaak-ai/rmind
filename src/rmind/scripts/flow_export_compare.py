"""Compare a saved policy ONNX artifact against the eager PyTorch model.

The flow noise is a graph INPUT (not sampled inside the graph), so parity is
DETERMINISTIC: feed identical inputs to both sides and the outputs must match to
float tolerance — there is no RNG mismatch to reconcile. This rebuilds the eager
graph from the checkpoint (with the SAME args used to export), then on CPU and (if
available) GPU runs both the eager model and an ONNX Runtime session on one shared
(batch, noise), reporting:

  * a per-action-channel discrepancy table (max/mean abs, max rel, pass/fail), and
  * a latency table (ms per inference pass) for eager vs ONNX on each device.

Exits non-zero if any channel is out of tolerance.

Usage (point it at the file `flow_export` wrote, with the same args):
    just compare-policy-onnx model.artifact=yaak/action-flow/model-<run>:vN \\
        '+export.out=artifacts/policy.onnx'

Knobs (+compare.*): onnx (path, default = export.out), rtol (default 1e-2),
atol (default 1e-3), warmup (default 3), iters (default 20).
"""

import multiprocessing as mp
import time
from collections.abc import Callable
from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import DictConfig
from structlog import get_logger
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_map  # noqa: PLC2701

from rmind.inference.backends import OnnxInferenceBackend
from rmind.scripts.flow_export import build_exportable

logger = get_logger(__name__)

_NOISE_RANK = 4  # (batch, draws, horizon, action_dim)


def _graph_noise_shape(onnx_path: Path) -> list[int | None] | None:
    """The saved graph's `noise` input shape (concrete dims as ints, dynamic as None).

    Read cheaply from the ONNX proto (no session, no weights). K (and batch) are
    frozen into the artifact at export, so the eager side must build noise to
    match — otherwise ONNX Runtime rejects the feed on shape mismatch.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    inputs = list(model.graph.input)
    if not inputs:
        return None
    noise = next((i for i in inputs if i.name == "noise"), inputs[-1])
    return [
        d.dim_value if d.HasField("dim_value") else None
        for d in noise.type.tensor_type.shape.dim
    ]


def _match_noise_shape_to_graph(cfg: DictConfig, onnx_path: Path) -> None:
    """Set cfg.export.{batch,draws} to the saved graph's frozen noise dims.

    K (draws) and batch are baked into the artifact at export, so the eager side
    must build noise to match regardless of the config defaults — otherwise ONNX
    Runtime rejects the feed on shape mismatch.
    """
    shape = _graph_noise_shape(onnx_path)
    if shape is None or len(shape) != _NOISE_RANK:
        return
    graph_batch, graph_draws = shape[0], shape[1]
    if not (isinstance(graph_batch, int) and isinstance(graph_draws, int)):
        return
    if graph_batch == cfg.export.batch and graph_draws == cfg.export.draws:
        return
    logger.info(
        "matching eager noise to the saved graph",
        graph_noise=shape,
        batch=graph_batch,
        draws=graph_draws,
    )
    cfg.export.batch = graph_batch
    cfg.export.draws = graph_draws


def build_feed(onnx_path: Path, example_args: tuple) -> dict[str, Tensor]:
    """Feed dict keyed by the saved graph's OWN input names.

    Reads the input names from the artifact and zips them positionally with the
    flattened example tensors — `torch.export` preserves the flattened arg order as
    the graph input order, so this lines up regardless of how the artifact was
    named at export (robust to pre/post-`input_names` exports). ONNX Runtime then
    rejects any genuine shape mismatch on `run`.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    names = [i.name for i in model.graph.input]
    leaves, _ = tree_flatten(example_args)
    tensors = [leaf for leaf in leaves if isinstance(leaf, Tensor)]
    return dict(zip(names, tensors, strict=True))


def _action_channels(exportable: torch.nn.Module, width: int) -> list[str]:
    """Names of the raw output-action channels, else positional fallbacks."""
    objective = getattr(exportable, "objective", None)
    keys = getattr(objective, "action_keys", None)
    if keys and len(keys) == width:
        return list(keys)
    return [f"ch{i}" for i in range(width)]


def _to_device(example_args: tuple, device: str) -> tuple:
    return tree_map(
        lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, example_args
    )


def _bench(fn: Callable[[], object], *, warmup: int, iters: int, cuda: bool) -> float:
    """Mean milliseconds per call, syncing CUDA around the timed region."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def _channel_rows(
    eager: np.ndarray,
    onnx_out: np.ndarray,
    channels: list[str],
    tol: tuple[float, float],
) -> tuple[list[list[str]], bool]:
    """Per-channel (max/mean abs, max rel, pass) rows; caller prepends the device."""
    rtol, atol = tol
    rows: list[list[str]] = []
    ok_all = True
    for c, name in enumerate(channels):
        e, o = eager[..., c], onnx_out[..., c]
        diff = np.abs(e - o)
        denom = np.maximum(np.abs(e), np.abs(o))
        rel = np.where(denom > 0, diff / denom, 0.0)
        ok = bool(np.allclose(e, o, rtol=rtol, atol=atol))
        ok_all &= ok
        rows.append([
            name,
            f"{diff.max():.2e}",
            f"{diff.mean():.2e}",
            f"{rel.max():.2e}",
            "yes" if ok else "NO",
        ])
    return rows, ok_all


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:  # noqa: PLR0914
    opts = cfg.get("export") or {}
    copts = cfg.get("compare") or {}
    onnx_path = Path(str(copts.get("onnx") or opts["out"]))
    rtol = float(copts.get("rtol", 1e-2))
    atol = float(copts.get("atol", 1e-3))
    warmup = int(copts.get("warmup", 3))
    iters = int(copts.get("iters", 20))
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))
    torch.manual_seed(0)  # reproducible example inputs
    ort.set_default_logger_severity(3)  # errors only (hide CastLike constant-fold spam)

    _match_noise_shape_to_graph(cfg, onnx_path)
    exportable, example_args = build_exportable(cfg)

    torch_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ort_by_device = {"cpu": "CPUExecutionProvider"}
    if torch.cuda.is_available():
        if "CUDAExecutionProvider" in ort.get_available_providers():
            ort_by_device["cuda"] = "CUDAExecutionProvider"
        else:
            logger.info(
                "ORT CUDA provider unavailable (install onnxruntime-gpu); "
                "skipping onnx-gpu",
                available=sorted(ort.get_available_providers()),
            )

    latency_rows: list[list[str]] = []
    eager_by_device: dict[str, np.ndarray] = {}
    for device in torch_devices:
        model = exportable.to(device).eval()
        args = _to_device(example_args, device)
        eager_by_device[device] = model(*args).detach().cpu().numpy()
        ms = _bench(
            lambda model=model, args=args: model(*args),
            warmup=warmup,
            iters=iters,
            cuda=device == "cuda",
        )
        latency_rows.append(["eager", device, f"{ms:.2f}"])
    exportable.to("cpu")

    onnx_by_device: dict[str, np.ndarray] = {}
    feed = build_feed(onnx_path, example_args)
    for device, provider in ort_by_device.items():
        backend = OnnxInferenceBackend(path=onnx_path, providers=[provider])
        backend.on_predict_start()
        onnx_by_device[device] = backend.forward(feed)["action_draws"].numpy()
        ms = _bench(
            lambda backend=backend: backend.forward(feed),
            warmup=warmup,
            iters=iters,
            cuda=False,
        )
        latency_rows.append(["onnx", device, f"{ms:.2f}"])

    reference = eager_by_device["cpu"]
    channels = _action_channels(exportable, reference.shape[-1])
    parity_rows: list[list[str]] = []
    all_ok = True
    for device, onnx_out in onnx_by_device.items():
        eager = eager_by_device.get(device, reference)
        rows, ok = _channel_rows(eager, onnx_out, channels, (rtol, atol))
        parity_rows.extend([device, *row] for row in rows)
        all_ok &= ok

    parity = _fmt_table(
        ["device", "channel", "max_abs", "mean_abs", "max_rel", "ok"], parity_rows
    )
    latency = _fmt_table(["backend", "device", "ms/pass"], latency_rows)
    verdict = "PARITY OK" if all_ok else "PARITY FAILED"
    print(  # noqa: T201
        f"\nonnx vs eager  (rtol={rtol}, atol={atol}; noise is a graph input, so "
        f"outputs are deterministic)\n"
        f"artifact: {onnx_path}\n\n{parity}\n\n"
        f"latency — one forward pass over the exported batch\n{latency}\n\n"
        f"{verdict}\n"
    )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
