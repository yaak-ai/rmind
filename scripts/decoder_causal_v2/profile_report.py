# This file is an operational script, deliberately exempt from the repo's lint
# profile. `T201` is the reason that matters: ruff runs here with `fix = true,
# unsafe-fixes = true`, and left to itself it DELETES every `print` -- which is
# this script's entire product. Its output IS the measurement. Removing this
# header silently guts the file (it already did once, mid-run).
# ruff: noqa: T201, INP001, ANN001,
"""Attribute TRT profile JSON time to the kernels that decide the fp16 verdict.

A raw `grep -c _gemm_mha_v2` is NOT a kernel count: detailed verbosity emits
hash-suffixed tactic names that recur several times in one dump, and an engine
built at default verbosity has them partly stripped, so counts are comparable
neither across engines nor across files (docs §12.4/§12.8). The defensible
figure is per-instance attributed time from the profile JSON, which is what this
prints: instances, summed ms, and share of the step.

Three things are read out, all load-bearing:
  * `_gemm_mha_v2`   -- the fused MHA. Without it the fp16 speedup is not real.
  * `cat`/`concat`   -- expected at 0.00 ms. The KV Concat has not vanished, it
                        has been ABSORBED into the fused kernel; that is exactly
                        why removing it to "save a copy" would take the fusion.
  * `Reformatting`   -- the cache-conversion copies a plain --fp16 build pays
                        because its cache bindings stay fp32 (§12.7).
"""

import json
import sys
from pathlib import Path


def report(path: Path) -> None:
    entries = json.loads(path.read_text(encoding="utf-8"))
    rows = [e for e in entries if isinstance(e, dict) and "name" in e]
    key = "averageMs" if rows and "averageMs" in rows[0] else "medianMs"
    total = sum(float(r.get(key, 0.0)) for r in rows)
    print(f"\n===== {path.name} =====")
    print(f"  layers: {len(rows)}   total {key}: {total:.3f} ms")

    def bucket(label: str, pred) -> None:
        hit = [r for r in rows if pred(r["name"])]
        t = sum(float(r.get(key, 0.0)) for r in hit)
        share = 100.0 * t / total if total else 0.0
        print(f"  {label:<34s} {len(hit):3d} instances  {t:7.3f} ms  {share:5.1f} %")

    bucket("_gemm_mha_v2 (fused MHA)", lambda n: "_gemm_mha_v2" in n)
    bucket("any *mha* kernel", lambda n: "mha" in n.lower())
    bucket(
        "cat/concat standalone", lambda n: "concat" in n.lower() or "cat_" in n.lower()
    )
    bucket("Reformatting CopyNode", lambda n: "Reformatting" in n)

    ref = sorted(
        (r for r in rows if "Reformatting" in r["name"]),
        key=lambda r: -float(r.get(key, 0.0)),
    )[:4]
    for r in ref:
        print(f"      {float(r.get(key, 0.0)):7.3f} ms  {r['name'][:78]}")
    top = sorted(rows, key=lambda r: -float(r.get(key, 0.0)))[:5]
    print("  -- 5 most expensive layers --")
    for r in top:
        print(f"      {float(r.get(key, 0.0)):7.3f} ms  {r['name'][:78]}")


for p in sys.argv[1:]:
    q = Path(p)
    if q.exists():
        report(q)
    else:
        print(f"\n===== {p} MISSING =====")
