"""Render the per-checkpoint argmax-vs-sampled comparison tables.

Usage:
    python render_tables.py <label>=<json> [<label>=<json> ...]

Emits markdown for: the headline scalar table, the sampled/argmax ratio,
the partial-vs-full-window buckets under argmax, the tails, and the
per-position argmax-recon curve. Reads the JSON written by
`rmind.scripts.patch_policy_eval --json-out`.
"""

import json
import sys


def load(argv: list[str]) -> dict[str, dict]:
    out = {}
    for a in argv:
        label, path = a.split("=", 1)
        with open(path) as fh:
            out[label] = json.load(fh)
    return out


def pct(new: float, old: float) -> str:
    return f"{(new / old - 1) * 100:+.1f}%"


def main() -> None:
    runs = load(sys.argv[1:])
    labels = list(runs)
    base, last = labels[0], labels[-1]

    def sc(d: dict, k: str) -> float:
        return d["val_scalars"][k]

    def qmean(d: dict, suffix: str = "") -> float:
        return sum(sc(d, f"val/policy/loss/code_{q}{suffix}") for q in range(4)) / 4

    def buckets(d: dict, name: str) -> tuple[float, float]:
        return d["by_position"]["partial 0-14"][name], d["by_position"]["full 15-31"][name]

    print("### Headline val scalars (full clip37 val split)\n")
    rows = [
        ("num_samples", lambda d: d["num_samples"], "{:.0f}"),
        ("code focal smoothed (mean q0-3)", lambda d: qmean(d), "{:.4f}"),
        ("code focal UNSMOOTHED (mean)", lambda d: qmean(d, "_unsmoothed"), "{:.4f}"),
        *[
            (f"  code_{q} smoothed", (lambda q: lambda d: sc(d, f"val/policy/loss/code_{q}"))(q), "{:.4f}")
            for q in range(4)
        ],
        *[
            (
                f"  code_{q} unsmoothed",
                (lambda q: lambda d: sc(d, f"val/policy/loss/code_{q}_unsmoothed"))(q),
                "{:.4f}",
            )
            for q in range(4)
        ],
        ("offset L1 (teacher-forced)", lambda d: sc(d, "val/policy/loss/offset"), "{:.6f}"),
        ("recon L1 ARGMAX (served)", lambda d: sc(d, "val/policy/metric/offset_argmax_recon"), "{:.5f}"),
        ("recon L1 sampled (logged)", lambda d: sc(d, "val/policy/metric/offset_sampled_recon"), "{:.5f}"),
        (
            "sampled / argmax ratio",
            lambda d: sc(d, "val/policy/metric/offset_sampled_recon")
            / sc(d, "val/policy/metric/offset_argmax_recon"),
            "{:.3f}x",
        ),
        ("top-1 code acc (mean over q)", lambda d: d["by_position"]["all (wandb)"]["top1_acc"], "{:.4f}"),
        ("joint code acc (all 4 q)", lambda d: d["by_position"]["all (wandb)"]["joint_acc"], "{:.4f}"),
        ("p(GT)", lambda d: d["by_position"]["all (wandb)"]["p_gt"], "{:.4f}"),
        ("entropy (nats, uniform=2.77)", lambda d: d["by_position"]["all (wandb)"]["entropy"], "{:.4f}"),
    ]
    print("| metric | " + " | ".join(labels) + f" | {last} vs {base} |")
    print("| --- |" + " --- |" * (len(labels) + 1))
    for name, fn, fmt in rows:
        vals = []
        for lb in labels:
            try:
                vals.append(fmt.format(fn(runs[lb])))
            except (KeyError, ZeroDivisionError):
                vals.append("—")
        try:
            delta = pct(fn(runs[last]), fn(runs[base]))
        except (KeyError, ZeroDivisionError):
            delta = "—"
        print(f"| {name} | " + " | ".join(vals) + f" | {delta} |")

    print("\n### Partial (0-14) vs full (15-31) window, ARGMAX and code metrics\n")
    bnames = ["code_focal", "code_plain", "top1_acc", "joint_acc", "p_gt", "entropy", "offset", "recon_argmax", "recon_sampled"]
    print("| metric | " + " | ".join(f"{lb} partial | {lb} full | {lb} full-vs-partial" for lb in labels) + " |")
    print("| --- |" + " --- |" * (3 * len(labels)))
    for nm in bnames:
        cells = []
        for lb in labels:
            try:
                p, f = buckets(runs[lb], nm)
                fmt = "{:.6f}" if "offset" in nm else "{:.4f}"
                cells += [fmt.format(p), fmt.format(f), pct(f, p)]
            except KeyError:
                cells += ["—", "—", "—"]
        print(f"| {nm} | " + " | ".join(cells) + " |")

    print("\n### Tails\n")
    print("| series | " + " | ".join(f"{lb} mean | {lb} p50 | {lb} p95 | {lb} p99 | {lb} max" for lb in labels) + " |")
    print("| --- |" + " --- |" * (5 * len(labels)))
    for series in ["argmax_recon/all", "argmax_recon/last", "sampled_recon/all", "sampled_recon/last", "offset/all", "offset/last"]:
        cells = []
        for lb in labels:
            t = runs[lb]["tails"].get(series)
            cells += [f"{t[k]:.4f}" for k in ("mean", "p50", "p95", "p99", "max")] if t else ["—"] * 5
        print(f"| `{series}` | " + " | ".join(cells) + " |")

    print("\n### Per-position argmax recon and code loss\n")
    positions = [k for k in runs[base]["by_position"] if k.startswith("t=")] + ["all (wandb)", "last (=bsln)", "partial 0-14", "full 15-31"]
    print("| position | " + " | ".join(f"{lb} recon_argmax | {lb} code_focal | {lb} top1" for lb in labels) + " |")
    print("| --- |" + " --- |" * (3 * len(labels)))
    for p in positions:
        cells = []
        for lb in labels:
            r = runs[lb]["by_position"].get(p)
            cells += [f"{r['recon_argmax']:.4f}", f"{r['code_focal']:.4f}", f"{r['top1_acc']:.4f}"] if r else ["—"] * 3
        print(f"| {p} | " + " | ".join(cells) + " |")

    print("\n### Per-cluster L1 at the last readout (argmax)\n")
    cl = runs[base]["clusters"]
    print("| cluster | n | " + " | ".join(f"{lb} argm_gas | {lb} argm_brake | {lb} argm_steer" for lb in labels) + " |")
    print("| --- | --- |" + " --- |" * (3 * len(labels)))
    for c in cl:
        cells = []
        for lb in labels:
            r = runs[lb]["clusters"].get(c)
            cells += [f"{r['argmax_gas']:.4f}", f"{r['argmax_brake']:.4f}", f"{r['argmax_steer']:.4f}"] if r else ["—"] * 3
        print(f"| {c} | {cl[c]['n']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
