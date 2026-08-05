"""ONNX-level parity between the `concat` decoder step and its `split*` variants.

`tests/test_causal_frame.py` gates the restructure in eager PyTorch. This gates
the thing that actually ships: two ONNX graphs, exported from the same seed and
therefore holding **the same weights**, run on the same inputs through ONNX
Runtime. It is the check that the export path -- `torch.export` decomposition,
the ONNX optimizer, the transposed cache layout -- did not change the answer.

    python -m rmind.scripts.decoder_only_cache_parity \
        --reference decoder_small_n6.onnx \
        --variant decoder_small_n6_split.onnx decoder_small_n6_splitkt.onnx

Both cache states are checked, because they fail differently:

* `warm` -- every slot valid. Exercises the online-softmax merge proper.
* `cold` -- every slot `MASK_BIAS`, with large garbage K/V behind it. This is the
  state a merge gets wrong: `exp(row_max - shared_max)` must underflow to exactly
  zero and annihilate a finite-but-meaningless past numerator, rather than
  produce `0 * inf`.

A `split*` graph is a **drop-in replacement** only if the K cache is written in
its own layout; `split_kt` holds `past_k` pre-transposed, so this script
transposes the shared input for it -- exactly what the host must do (§7 hazard 1:
`set_tensor_address` will not catch it).
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from structlog import get_logger

from rmind.components.transformer.causal_frame import MASK_BIAS

logger = get_logger(__name__)

CACHE_INPUTS = ("inputs_past_k", "inputs_past_v", "inputs_cache_bias")


def graph_input_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    model = onnx.load(path, load_external_data=False)
    return {
        i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
        for i in model.graph.input
    }


def make_inputs(
    shapes: dict[str, tuple[int, ...]], *, state: str, seed: int = 17
) -> dict[str, np.ndarray]:
    """Deterministic inputs; `past_k` is generated in the graph's OWN layout."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name, shape in shapes.items():
        if name == "inputs_image":
            out[name] = rng.random(shape, dtype=np.float32)
        elif name == "inputs_speed":
            out[name] = (rng.random(shape, dtype=np.float32) * 130).astype(np.float32)
        elif name == "inputs_waypoints":
            out[name] = (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float32)
        elif name == "inputs_cache_bias":
            fill = 0.0 if state == "warm" else MASK_BIAS
            out[name] = np.full(shape, fill, dtype=np.float32)
        elif name in {"inputs_rope_cos", "inputs_rope_sin"}:
            out[name] = rng.standard_normal(shape).astype(np.float32) * 0.5
        else:  # past_k / past_v
            scale = 1.0 if state == "warm" else 50.0  # garbage behind a cold mask
            out[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
    return out


def align_cache(
    inputs: dict[str, np.ndarray], shapes: dict[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """Re-lay-out the shared cache tensors for the variant's declared shapes.

    Handles both layout knobs, so one reference run can be scored against every
    variant: `split_kt` wants `past_k` transposed, and `per_layer_cache` wants the
    stacked cache split into `past_k_0 …` with the layer dimension dropped.

    Raises:
        ValueError: if a declared cache shape is neither the reference's, nor its
            transpose, nor a per-layer slice of it -- i.e. the variant is not a
            re-layout of the same cache.
    """
    aligned = {k: v for k, v in inputs.items() if k not in CACHE_INPUTS}
    aligned["inputs_cache_bias"] = inputs["inputs_cache_bias"]
    for side in ("inputs_past_k", "inputs_past_v"):
        stacked = inputs[side]
        per_layer = f"{side}_0" in shapes
        blocks = list(stacked) if per_layer else [stacked]
        names = [f"{side}_{i}" for i in range(len(blocks))] if per_layer else [side]
        for name, block in zip(names, blocks, strict=True):
            want, got = shapes[name], block.shape
            if want == got:
                aligned[name] = block
            elif want == (*got[:-2], got[-1], got[-2]):
                aligned[name] = np.ascontiguousarray(block.swapaxes(-1, -2))
            else:
                msg = f"{name}: cannot reconcile variant shape {want} with {got}"
                raise ValueError(msg)
    return aligned


def run(path: Path, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(None, inputs), strict=True))


def compare(
    reference: dict[str, np.ndarray], variant: dict[str, np.ndarray]
) -> dict[str, tuple[float, float]]:
    """`{output: (max_abs_diff, max_abs_diff / reference_scale)}`.

    Deliberately NOT per-element relative error: on near-zero K/V entries that is
    alarming and meaningless (the same false alarm `torch.onnx.export(verify=True)`
    raises on `new_k`). Compare against the tensor's own scale instead.
    """
    report: dict[str, tuple[float, float]] = {}
    for name, want in reference.items():
        got = variant[name]
        if got.shape != want.shape:  # split_kt emits new_k pre-transposed
            got = got.swapaxes(-1, -2)
        diff = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
        scale = float(np.abs(want).max()) or 1.0
        report[name] = (diff, diff / scale)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--variant", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="max abs diff / tensor scale allowed; float32 accumulation over "
        "16k keys in a different order is ~1e-6 relative, so this is loose by "
        "two orders of magnitude and still 3 orders below a code flip (~0.1)",
    )
    args = parser.parse_args()

    ref_shapes = graph_input_shapes(args.reference)
    failures: list[str] = []
    for state in ("warm", "cold"):
        inputs = make_inputs(ref_shapes, state=state)
        reference = run(args.reference, inputs)
        for variant_path in args.variant:
            shapes = graph_input_shapes(variant_path)
            got = run(variant_path, align_cache(inputs, shapes))
            report = compare(reference, got)
            for name, (absolute, relative) in report.items():
                verdict = "ok" if relative <= args.tol else "FAIL"
                if verdict == "FAIL":
                    failures.append(f"{variant_path.name}/{state}/{name}")
                logger.info(
                    verdict,
                    state=state,
                    variant=variant_path.name,
                    output=name,
                    abs_diff=f"{absolute:.3e}",
                    vs_scale=f"{relative:.3e}",
                )
            finite = all(np.isfinite(v).all() for v in got.values())
            if not finite:
                failures.append(f"{variant_path.name}/{state}/non-finite")
                logger.error(
                    "non-finite output", variant=variant_path.name, state=state
                )

    if failures:
        msg = f"parity failed: {failures}"
        raise SystemExit(msg)
    logger.info("parity ok", variants=[p.name for p in args.variant])


if __name__ == "__main__":
    main()
