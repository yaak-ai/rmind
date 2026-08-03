"""Classifier-free-guidance (CFG) amplification sweep for map-context PatchPolicy.

The model was trained with `max_speed_dropout=0.3` (token -> UNKNOWN), i.e. it
carries exactly the conditional / unconditional pair CFG needs. This script
runs, on IDENTICAL cached val batches:

  l_g = l_u + w * (l_c - l_u)          (per quantizer, LOGIT level)
  codes_g = argmax(l_g)                (the model's own argmax decode)
  chunk   = tokenizer.invert(codes_g) + _offset(offsets, codes_g)

where `l_u` / `offsets_u` come from the UNKNOWN (override=None) forward and
`l_c` / `offsets_c` from the conditional forward (max_speed_override=<kmh>).
Three offset policies are decoded and dumped so the action deltas can be
attributed:

- `chunk`        offsets_c at guided codes  (primary: "conditional model, guided codes")
- `chunk_offu`   offsets_u at guided codes  (pure code-flip effect, offsets held fixed)
- `chunk_offcfg` offsets_u + w*(offsets_c - offsets_u) at guided codes (full CFG)

Anchors: w=0 must reproduce the baseline codes exactly (arithmetic check);
w=1 must reproduce the plain-override run bit-for-bit (plumbing check).

Usage (kitkat worktree, PYTHONPATH pointing at it):
  python -m rmind.scripts... / python diag_results/eval_v0/cfg_sweep.py \
      --ckpt artifacts/model-0nr1ydjm:v1/model.ckpt \
      --batch-cache caches/eval_v0_val_batches.pt --micro-batch 8 \
      --out diag_results/eval_v0/cfg_sweep_mv_v1.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from rmind.models.patch_policy import PatchPolicy
from rmind.scripts.map_probe import _to_device, override_name, run_override

WEIGHTS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
CONDS = (30.0, 5.0, 100.0, -1.0)


def _slice_tree(node, s, e):
    if isinstance(node, dict):
        return {k: _slice_tree(v, s, e) for k, v in node.items()}
    if isinstance(node, torch.Tensor):
        return node[s:e]
    return node


@torch.no_grad()
def heads(model: PatchPolicy, batch: dict, override: float | None):
    """Last-frame code logits (b, g, c) and offset table (b, g, c, a)."""
    model.max_speed_override = override
    features, _ = model._features(batch, require_chunk=False)  # noqa: SLF001
    return model._heads(features[:, -1])  # noqa: SLF001


@torch.no_grad()
def decode(model: PatchPolicy, offsets: Tensor, codes: Tensor) -> Tensor:
    offset = model._offset(offsets, codes)  # noqa: SLF001
    return (model.tokenizer.invert(codes) + offset).unflatten(
        -1,
        (-1, model.tokenizer._action_features),  # noqa: SLF001
    )


def main() -> None:  # noqa: PLR0914, PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-cache", required=True)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    batches = torch.load(args.batch_cache, weights_only=False)
    print(f"loaded {len(batches)} cached val batches", flush=True)

    model = PatchPolicy.load_from_checkpoint(
        args.ckpt, map_location="cpu", weights_only=False
    )
    model.sample_codes = False  # deterministic argmax decoding
    if model.max_speed_tokenizer is None:
        raise SystemExit("not a map-context checkpoint")
    model = model.to(device).eval()
    print("offset_scale =", model.offset_scale, flush=True)
    print("input_transform:", repr(model.tokenizer.input_transform), flush=True)

    acc: dict[str, list] = {}

    def put(key: str, value: Tensor) -> None:
        acc.setdefault(key, []).append(value.cpu())

    anchor_w1 = 0.0
    anchor_w0_codes = 0

    for bi, batch in enumerate(batches):
        bsz = batch["data"]["meta/VehicleMotion/speed"].shape[0]
        step = args.micro_batch or bsz
        for s in range(0, bsz, step):
            batch_d = _to_device(_slice_tree(batch, s, s + step), device)

            lu, ou = heads(model, batch_d, None)
            base_codes = lu.argmax(-1)
            put("base/logits", lu.float())
            put("base/codes", base_codes)
            put("base/chunk", decode(model, ou, base_codes).float())

            for cond in CONDS:
                cname = override_name(cond)
                lc, oc = heads(model, batch_d, cond)
                put(f"{cname}/logits", lc.float())
                for w in WEIGHTS:
                    lg = lu + w * (lc - lu)
                    cg = lg.argmax(-1)
                    tag = f"{cname}/w{w:g}"
                    put(f"{tag}/codes", cg)
                    put(f"{tag}/chunk", decode(model, oc, cg).float())
                    put(f"{tag}/chunk_offu", decode(model, ou, cg).float())
                    og = ou + w * (oc - ou)
                    put(f"{tag}/chunk_offcfg", decode(model, og, cg).float())
                    if w == 0.0:
                        anchor_w0_codes += int((cg != base_codes).sum())
                # w=1 plumbing anchor vs the plain-override path (first batch only:
                # it costs a second backbone pass per condition)
                if bi == 0:
                    plain = run_override(model, batch_d, cond)
                    c1 = lc.argmax(-1)
                    d = (
                        (decode(model, oc, c1).float().cpu() - plain["chunk"])
                        .abs()
                        .max()
                    )
                    anchor_w1 = max(anchor_w1, float(d))
            del batch_d
        torch.cuda.empty_cache()
        print(f"batch {bi + 1}/{len(batches)} done", flush=True)

    arrays: dict[str, np.ndarray] = {
        k: torch.cat(v).numpy() for k, v in acc.items()
    }
    arrays["speed_last"] = (
        torch.cat(
            [b["data"]["meta/VehicleMotion/speed"][:, -1].reshape(-1) for b in batches]
        )
        .float()
        .numpy()
    )
    arrays["anchor_w1_max_chunk_diff"] = np.asarray(anchor_w1)
    arrays["anchor_w0_code_mismatches"] = np.asarray(anchor_w0_codes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(
        f"saved {args.out}: {len(arrays)} arrays, "
        f"n={arrays['speed_last'].shape[0]}; "
        f"anchor w=1 max|chunk diff|={anchor_w1:g}; "
        f"anchor w=0 code mismatches={anchor_w0_codes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
