"""Compute the val entropy sidecar foresight_mm/cache/val_entropy.pt.

Reads the EXISTING pooled val cache (foresight_mm/cache/val, 18833 samples,
shuffle=False order) and writes a single .pt with, per the contract:

  sample_id    (18833,)   int64  (cache order)
  entropy_h    (18833,5)  fp32   mean-pairwise-L2 neighbor-future entropy, H=1..5
  quintile_h1  (18833,)   int8   0..4 quintile of entropy_h[:,0]
  quintile_h5  (18833,)   int8   0..4 quintile of entropy_h[:,4]
  neighbor_idx (18833,32) int32  anti-leakage kNN indices into cache order

kNN is IDENTICAL to foresight_multimodality_analysis.py (reused by import):
k=32, gap=90 sample-order units (~30 s), context = standardized
[ctx_foresight_pooled ‖ ctx_action_summary], eligible iff different drive OR
|order gap| > 90.

Validation (runs after writing, all asserted):
  - quintile counts balanced
  - entropy_h mean per H vs phase0 reference (16.52 -> 17.34)
  - anti-leakage rule on 100 random anchors

Usage:
  uv run python src/rmind/scripts/foresight_val_entropy_sidecar.py [--device cpu]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from rmind.scripts import foresight_phase0_common as com
from rmind.scripts.foresight_multimodality_analysis import (
    eligible_knn,
    entropy_maps,
    quintile_assign,
)

K = 32
GAP = 90.0
PHASE0_MEAN_REF = (16.52, 17.34)  # overall mean H=1 -> H=5 from phase0_d0_d4.json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=com.DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--out",
        type=Path,
        default=com.REPO_ROOT / "foresight_mm" / "cache" / "val_entropy.pt",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="validation-anchor RNG seed")
    args = ap.parse_args()

    t0 = time.perf_counter()
    data = com.load_cache(args.cache_dir)
    n = data["sample_id"].shape[0]
    drives, drive_names = com.drive_codes(data["input_id"])
    order, order_src = com.sample_order(drives, data.get("frame_idx"))
    print(f"[entropy] n={n}, drives={drive_names}, order source: {order_src}", flush=True)

    c, _, _ = com.standardize(
        torch.cat(
            [data["ctx_foresight_pooled"].float(), data["ctx_action_summary"].float()],
            dim=1,
        )
    )
    nn_idx, n_excl = eligible_knn(c, drives, order, K, GAP, args.device)
    mean_excl = float((n_excl - 1).float().mean())
    print(f"[entropy] mean excluded candidates/anchor (beyond self) = {mean_excl:.1f}", flush=True)

    mpd, _ = entropy_maps(nn_idx, data["fut_dino_pooled"].float(), args.device)

    sidecar = {
        "sample_id": data["sample_id"].to(torch.int64),
        "entropy_h": mpd.float(),
        "quintile_h1": quintile_assign(mpd[:, 0]).to(torch.int8),
        "quintile_h5": quintile_assign(mpd[:, 4]).to(torch.int8),
        "neighbor_idx": nn_idx.to(torch.int32),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sidecar, args.out)
    print(f"[entropy] wrote {args.out} in {time.perf_counter() - t0:.0f}s", flush=True)

    # ---- validation --------------------------------------------------------- #
    for qk in ("quintile_h1", "quintile_h5"):
        counts = torch.bincount(sidecar[qk].long(), minlength=5).tolist()
        print(f"[entropy] {qk} counts: {counts}", flush=True)
        assert min(counts) > 0 and max(counts) - min(counts) <= max(2, n // 1000), (
            f"{qk} quintiles unbalanced: {counts}"
        )
    means = sidecar["entropy_h"].mean(0).tolist()
    print(f"[entropy] entropy_h means H=1..5: {[f'{m:.3f}' for m in means]}", flush=True)
    assert all(b >= a for a, b in zip(means, means[1:], strict=False)), (
        f"entropy_h means not monotone: {means}"
    )
    assert abs(means[0] - PHASE0_MEAN_REF[0]) < 0.1, (means[0], PHASE0_MEAN_REF[0])
    assert abs(means[4] - PHASE0_MEAN_REF[1]) < 0.1, (means[4], PHASE0_MEAN_REF[1])

    g = torch.Generator().manual_seed(args.seed)
    anchors = torch.randperm(n, generator=g)[:100]
    for a in anchors.tolist():
        nb = sidecar["neighbor_idx"][a].long()
        ok = (drives[nb] != drives[a]) | ((order[nb] - order[a]).abs() > GAP)
        assert bool(ok.all()), (
            f"anti-leakage violated at anchor {a}: "
            f"{nb[~ok].tolist()} within gap on same drive"
        )
        assert a not in nb.tolist(), f"self-neighbor at anchor {a}"
    print("[entropy] validation OK: quintiles balanced, means match phase0, "
          "anti-leakage holds on 100 random anchors", flush=True)


if __name__ == "__main__":
    main()
