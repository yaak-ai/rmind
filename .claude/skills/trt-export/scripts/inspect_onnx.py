#!/usr/bin/env python
"""Report the four things about an ONNX export that decide the whole TRT pipeline.

  input size + Resize count -> does the HOST have to deliver the exact size?
  first ops (Sub/Div)       -> is ImageNet normalization already in the graph?
                               (yes => serve with --image-norm unit)
  ArgMax count              -> codebook/VQ decoding in-graph => fp16 is dangerous
  Sin/Cos                   -> RoPE => strongly fp16-fragile (DINOv3 has it, v2 doesn't)

Usage:  inspect_onnx.py MODEL.onnx [MODEL2.onnx ...]
Needs:  onnx   (e.g. `uv run --with onnx python inspect_onnx.py ...`)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import onnx

WATCH = ("Resize", "ArgMax", "Sin", "Cos", "Softmax", "LayerNormalization")


def dims(t) -> list:
    return [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]


def report(path: Path) -> None:
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph
    ops = [n.op_type for n in g.node]
    c = Counter(ops)

    print(f"\n=== {path.name}")
    for i in g.input:
        print(f"  IN   {i.name:44s} {dims(i)}")
    for o in g.output:
        print(f"  OUT  {o.name:44s} {dims(o)}")
    print(f"  nodes {len(ops)}   " + "  ".join(f"{k}={c.get(k, 0)}" for k in WATCH))
    print(f"  first ops: {ops[:10]}")

    # the actionable reading of the above
    notes = []
    cam = next((i for i in g.input if "cam" in i.name.lower()), None)
    if cam is not None:
        d = dims(cam)
        if len(d) >= 2 and all(isinstance(x, int) for x in d[-2:]):
            hw = f"{d[-2]}x{d[-1]}"
            notes.append(f"image input {hw}" + ("; NO Resize in graph -> host must "
                         f"deliver exactly {hw}" if not c.get("Resize") else ""))
    if ops[:2] == ["Sub", "Div"]:
        notes.append("Sub/Div first -> ImageNet norm IS in-graph -> serve --image-norm unit")
    if c.get("ArgMax"):
        notes.append(f"{c['ArgMax']} ArgMax -> codebook decode in-graph -> verify parity by "
                     "DECISION changes, prefer fp32")
    if c.get("Sin") or c.get("Cos"):
        notes.append("Sin/Cos present -> RoPE -> strongly fp16-fragile")
    for n in notes:
        print(f"  ** {n}")


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        if p.exists():
            report(p)
        else:
            print(f"\n=== {p}  MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
