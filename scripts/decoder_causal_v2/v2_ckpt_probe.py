# This file is an operational script, deliberately exempt from the repo's lint
# profile. `T201` is the reason that matters: ruff runs here with `fix = true,
# unsafe-fixes = true`, and left to itself it DELETES every `print` -- which is
# this script's entire product. Its output IS the measurement. Removing this
# header silently guts the file (it already did once, mid-run).
# ruff: noqa: T201, INP001, ANN001, ANN201,
# ruff: noqa: PLR2004, E402
"""Confirm model-do8m9ot8:v2 is the checkpoint the task claims BEFORE exporting.

Everything downstream (latency, margins, parity) inherits the architecture, so a
mismatch in epoch/step/window/depth must stop the run rather than be discovered
in the parity table.
"""

import sys

import torch
import wandb

ART = sys.argv[1] if len(sys.argv) > 1 else "yaak/rmind/model-do8m9ot8:v2"

api = wandb.Api()
art = api.artifact(ART, type="model")
print(f"artifact      : {art.name}  ({art.id})")
print(f"version       : {art.version}")
print(f"created       : {art.created_at}")
print(f"source run    : {art.logged_by().name if art.logged_by() else '?'}")
d = art.download()
print(f"downloaded to : {d}")

import pathlib

ckpts = sorted(pathlib.Path(d).glob("*.ckpt"))
print(f"files         : {[p.name for p in ckpts]}")
ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
print(f"epoch         : {ck.get('epoch')}")
print(f"global_step   : {ck.get('global_step')}")
hp = ck.get("hyper_parameters", {})


def walk(o, prefix="", depth=0):
    if depth > 4:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                walk(v, f"{prefix}.{k}" if prefix else str(k), depth + 1)
            elif any(
                s in str(k)
                for s in (
                    "window",
                    "rope",
                    "num_layers",
                    "num_heads",
                    "dim",
                    "fusion_norm",
                    "attention_impl",
                    "_target_",
                    "teacher_force",
                    "quantiz",
                    "code",
                )
            ):
                print(f"  hp {prefix}.{k} = {v}")


walk(hp)

sd = ck["state_dict"]
print(f"n params      : {sum(v.numel() for v in sd.values()) / 1e6:.2f} M (state_dict)")
trunk = [k for k in sd if "transformer" in k or "trunk" in k]
print(f"trunk tensors : {len(trunk)}")
for k in trunk[:6]:
    print(f"   {k} {tuple(sd[k].shape)}")
nblocks = len({
    k.split(".")[3]
    for k in trunk
    if len(k.split(".")) > 4 and k.split(".")[3].isdigit()
})
print(f"blocks (heuristic): {nblocks}")
