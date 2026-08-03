"""Analyze the CFG sweep dump (cfg_sweep.py output) -> markdown tables.

Usage: python diag_results/eval_v0/cfg_analyze.py <npz> [--out <md>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CONDS = ("30", "5(WALK)", "100", "-1(UNLIMITED)")
WS = (0, 1, 2, 3, 5, 8, 12)
ACT = ("gas", "brake", "steer")
# normalized-space validity bounds for the three continuous fields
BOUNDS = ((0.0, 1.0), (0.0, 1.0), (-1.0, 1.0))


def fmt(x: float) -> str:
    return f"{x:+.4f}" if abs(x) >= 1e-4 else f"{x:+.1e}"


def excess(chunk: np.ndarray) -> np.ndarray:
    """Per-(sample, step, field) distance outside the valid action box."""
    ex = []
    for i, (lo, hi) in enumerate(BOUNDS):
        x = chunk[..., i]
        ex.append(np.maximum(0.0, lo - x) + np.maximum(0.0, x - hi))
    return np.stack(ex, -1)


def artifact_frac(chunk: np.ndarray) -> float:
    """Fraction of samples with ANY horizon step out of the valid action box.

    NOTE: near-saturated even for the UNKNOWN baseline (the offset head is
    unbounded -- `offset_scale=None` -- so tiny negative gas/brake values are
    routine decoder slop), hence `mat_viol` below is the informative indicator.
    """
    return float((excess(chunk) > 0).any((1, 2)).mean())


def mat_viol(chunk: np.ndarray) -> float:
    """Fraction of (sample, step, field) slots violating the box by > 0.05."""
    return float((excess(chunk) > 0.05).mean())


def mean_excess(chunk: np.ndarray) -> float:
    return float(excess(chunk).mean())


def paired(cur: np.ndarray, base: np.ndarray, i: int) -> str:
    """Mean +- 1 paired SE of the per-sample (horizon-mean) delta in field i."""
    x = cur[..., i].mean(-1) - base[..., i].mean(-1)
    se = x.std(ddof=1) / np.sqrt(x.shape[0])
    return f"{x.mean():+.4f} ± {se:.4f}"


def turn_signal(chunk: np.ndarray) -> np.ndarray:
    return np.digitize(chunk[..., 3] * 2, [0.5, 1.5])


def main() -> None:  # noqa: PLR0914, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.npz)
    speed = d["speed_last"]
    base_chunk = d["base/chunk"]  # (n, h, 4)
    base_codes = d["base/codes"]  # (n, g)
    n, h, _ = base_chunk.shape
    g = base_codes.shape[1]
    fast = speed > 50
    L: list[str] = []

    L += [
        "# CFG amplification sweep — armMV v1 (model-0nr1ydjm:v1), map-context PatchPolicy",
        "",
        f"`{args.npz}` — {n} val samples (24x32 cached batches, micro-batch 8, kitkat),",
        "argmax decoding, last-frame readout, action chunk horizon "
        f"{h}, {g} quantizers x 16 codes.",
        "",
        "Guided logits per quantizer: `l_g = l_u + w*(l_c - l_u)`, `l_u` = UNKNOWN "
        "(override=None; the cached val batches carry all-NaN `max_speed`, so this IS "
        "the unconditional path — verified bitwise in the earlier NaN-flood check), "
        "`l_c` = `max_speed_override=<cond>`. Guided codes = `argmax(l_g)`; actions = "
        "the model's own decode `tokenizer.invert(codes_g) + _offset(offsets, codes_g)`, "
        "i.e. the real offset head at the guided codes (the decode path re-enters "
        "cleanly — no fallback needed).",
        "",
        "Offset policies: **primary** = conditional `offsets_c` at guided codes; "
        "`offu` = unconditional offsets at guided codes (pure code-flip effect); "
        "`offcfg` = `offsets_u + w*(offsets_c - offsets_u)` (full CFG, incl. offsets).",
        "",
        f"Anchors: w=0 code mismatches vs baseline = "
        f"{int(d['anchor_w0_code_mismatches'])} (must be 0); "
        f"w=1 max |chunk - plain_override_chunk| = "
        f"{float(d['anchor_w1_max_chunk_diff']):g} (must be 0).",
        "",
        "All actions are in the tokenizer's NORMALIZED space (gas/brake ~[0,1], "
        "steer ~[-1,1]); deltas are means over samples x horizon vs the UNKNOWN "
        "baseline.",
        "",
    ]

    # ---------- main per-condition tables ----------
    for cond in CONDS:
        L += [
            f"## cond = {cond}",
            "",
            "| w | flip% | q0 | q1 | q2 | q3 | Δgas (±SE) | Δbrake (±SE) | Δsteer | "
            "Δgas·flipped | Δbrake·flipped | TS flip% | viol>0.05 % |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for w in WS:
            tag = f"{cond}/w{w}"
            codes = d[f"{tag}/codes"]
            chunk = d[f"{tag}/chunk"]
            flip = codes != base_codes
            dch = chunk - base_chunk
            any_flip = flip.any(-1)
            row = [
                str(w),
                f"{flip.mean() * 100:.2f}",
                *[f"{flip[:, q].mean() * 100:.2f}" for q in range(g)],
                paired(chunk, base_chunk, 0),
                paired(chunk, base_chunk, 1),
                fmt(dch[..., 2].mean()),
            ]
            if any_flip.any():
                row += [
                    fmt(dch[any_flip][..., 0].mean()),
                    fmt(dch[any_flip][..., 1].mean()),
                ]
            else:
                row += ["n/a", "n/a"]
            ts = (turn_signal(chunk) != turn_signal(base_chunk)).mean() * 100
            row += [f"{ts:.2f}", f"{mat_viol(chunk) * 100:.2f}"]
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    L += [
        "Baseline (UNKNOWN) reference: mean gas "
        f"{base_chunk[..., 0].mean():.4f}, brake {base_chunk[..., 1].mean():.4f}, "
        f"steer {base_chunk[..., 2].mean():.4f}; "
        f"material violations (>0.05 outside the box) "
        f"{mat_viol(base_chunk) * 100:.2f}% of slots, mean excess "
        f"{mean_excess(base_chunk):.5f}; any-violation (incl. hairline) "
        f"{artifact_frac(base_chunk) * 100:.1f}% of samples — the baseline itself, "
        "so only the >0.05 column tracks CFG-induced degeneracy.",
        "",
    ]

    # ---------- headline: 30 vs 100 on speed > 50 ----------
    L += [
        "## Headline — guided-30 minus guided-100 on speed > 50 km/h "
        f"(n={int(fast.sum())})",
        "",
        "A policy that READS the token should show **negative** Δgas and "
        "**positive** Δbrake here. Paired per-sample contrast (same samples, "
        "mean over horizon); +- is 1 paired SE.",
        "",
        "| w | Δgas (30-100) | Δbrake (30-100) | Δsteer (30-100) | "
        "flip% 30 | flip% 100 |",
        "|---|---|---|---|---|---|",
    ]
    for w in WS:
        c30 = d[f"30/w{w}/chunk"][fast]
        c100 = d[f"100/w{w}/chunk"][fast]
        cells = []
        for i in range(3):
            x = c30[..., i].mean(-1) - c100[..., i].mean(-1)
            se = x.std(ddof=1) / np.sqrt(x.shape[0])
            cells.append(f"{x.mean():+.4f} ± {se:.4f}")
        f30 = (d[f"30/w{w}/codes"][fast] != base_codes[fast]).mean() * 100
        f100 = (d[f"100/w{w}/codes"][fast] != base_codes[fast]).mean() * 100
        L.append(
            f"| {w} | " + " | ".join(cells) + f" | {f30:.2f} | {f100:.2f} |"
        )
    L.append("")

    # WALK vs 100 for completeness (the only strongly responsive class)
    L += [
        f"## WALK(5) minus 100 on speed > 50 km/h (n={int(fast.sum())})",
        "",
        "| w | Δgas | Δbrake | Δsteer |",
        "|---|---|---|---|",
    ]
    for w in WS:
        cw = d[f"5(WALK)/w{w}/chunk"][fast]
        c100 = d[f"100/w{w}/chunk"][fast]
        cells = []
        for i in range(3):
            x = cw[..., i].mean(-1) - c100[..., i].mean(-1)
            se = x.std(ddof=1) / np.sqrt(x.shape[0])
            cells.append(f"{x.mean():+.4f} ± {se:.4f}")
        L.append(f"| {w} | " + " | ".join(cells) + " |")
    L.append("")

    # ---------- per-sample direction split ----------
    L += [
        "## Per-sample direction split (material moves, |Δ| > 0.05 on the "
        "horizon-mean pedal, vs the UNKNOWN baseline)",
        "",
        "Population means can hide opposite per-sample moves, so: share of the "
        "768 samples whose gas rises / brake rises / both rise materially, plus "
        "the share of (sample, step) slots with simultaneously high pedals "
        "(gas>0.1 AND brake>0.1) — a physical-incoherence indicator "
        "(baseline: "
        f"{(((base_chunk[..., 0] > 0.1) & (base_chunk[..., 1] > 0.1)).mean() * 100):.2f}%).",
        "",
        "| cond | w | gas↑ only % | brake↑ only % | both↑ % | gas↓ % | "
        "both-pedals-high % |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond in CONDS:
        for w in WS:
            c = d[f"{cond}/w{w}/chunk"]
            dg = c[..., 0].mean(-1) - base_chunk[..., 0].mean(-1)
            db = c[..., 1].mean(-1) - base_chunk[..., 1].mean(-1)
            up_g, up_b = dg > 0.05, db > 0.05
            hi = ((c[..., 0] > 0.1) & (c[..., 1] > 0.1)).mean() * 100
            L.append(
                f"| {cond} | {w} | {(up_g & ~up_b).mean() * 100:.2f} | "
                f"{(up_b & ~up_g).mean() * 100:.2f} | {(up_g & up_b).mean() * 100:.2f} | "
                f"{(dg < -0.05).mean() * 100:.2f} | {hi:.2f} |"
            )
    L.append("")

    # ---------- offset-policy attribution ----------
    L += [
        "## Offset-policy attribution (mean Δgas / Δbrake vs baseline, all samples)",
        "",
        "| cond | w | primary (offsets_c) | offu (code-flip only) | offcfg (full CFG) |",
        "|---|---|---|---|---|",
    ]
    for cond in CONDS:
        for w in WS:
            tag = f"{cond}/w{w}"
            cells = []
            for key in ("chunk", "chunk_offu", "chunk_offcfg"):
                dch = d[f"{tag}/{key}"] - base_chunk
                cells.append(
                    f"{dch[..., 0].mean():+.4f} / {dch[..., 1].mean():+.4f}"
                )
            L.append(f"| {cond} | {w} | " + " | ".join(cells) + " |")
    L.append("")

    L += [
        "## Artifact indicators — material box violations (% of "
        "(sample, step, field) slots more than 0.05 outside the valid range)",
        "",
        "| cond | w | primary | offu | offcfg | mean excess (primary) |",
        "|---|---|---|---|---|---|",
    ]
    for cond in CONDS:
        for w in WS:
            tag = f"{cond}/w{w}"
            cells = [
                f"{mat_viol(d[f'{tag}/{key}']) * 100:.2f}"
                for key in ("chunk", "chunk_offu", "chunk_offcfg")
            ]
            cells.append(f"{mean_excess(d[f'{tag}/chunk']):.5f}")
            L.append(f"| {cond} | {w} | " + " | ".join(cells) + " |")
    L.append("")

    # ---------- critical-w geometry (why 30/50/100 barely move) ----------
    L += [
        "## Critical-w geometry (from the dumped logits, no decode involved)",
        "",
        "Per (sample, quantizer): the smallest w at which some code overtakes the "
        "UNKNOWN top-1. Percentiles are over slots where a finite crossing exists.",
        "",
        "| cond | finite-crossing slots % | crit-w p1 | p5 | p25 | p50 |",
        "|---|---|---|---|---|---|",
    ]
    # this table is computed from logits only; it agrees with the decoded flip
    # rates above (e.g. cond 30: p1 = 1.8 vs 0.52% flips at w=1), so the two
    # independent paths -- offline logit geometry and the GPU decode -- match.
    u = d["base/logits"].astype(np.float64)
    bk = u.argmax(-1)
    for cond in CONDS:
        cl = d[f"{cond}/logits"].astype(np.float64)
        uk = np.take_along_axis(u, bk[..., None], -1)
        ck = np.take_along_axis(cl, bk[..., None], -1)
        a = u - uk
        bb = (cl - ck) - a
        with np.errstate(divide="ignore", invalid="ignore"):
            wc = np.where(bb > 0, -a / bb, np.inf)
        np.put_along_axis(wc, bk[..., None], np.inf, -1)
        wmin = wc.min(-1)
        fin = np.isfinite(wmin) & (wmin > 0)
        ps = [np.percentile(wmin[fin], p) for p in (1, 5, 25, 50)]
        L.append(
            f"| {cond} | {np.isfinite(wmin).mean() * 100:.1f} | "
            + " | ".join(f"{x:.2f}" for x in ps)
            + " |"
        )
    L.append("")

    L += [
        "## Verdict",
        "",
        "CFG is mechanically live and correctly plumbed — w=0 reproduces the "
        "baseline codes exactly, w=1 is bit-identical to the plain-override run, "
        "and guided code flips grow monotonically with w (cond 30: 0.5% -> 6.0%, "
        "UNLIMITED: 1.2% -> 12.2%, WALK: 16.5% -> 77.7% of (sample, quantizer) "
        "slots) — so the weak conditioning CAN be amplified into real, decodable "
        "action changes.",
        "But the amplified direction is the WRONG one: guided-30 minus guided-100 "
        "on speed>50 frames grows from +0.0011 ± 0.0006 to +0.0070 ± 0.0028 gas "
        "(more throttle in the slower zone, ~2.5 paired SE from zero) while brake "
        "stays at noise (+0.0007 ± 0.0007), i.e. CFG scales up the token's "
        "regime association, not speed compliance — the eval_v0 defect grows "
        "≈6x from w=1 to w=12 instead of reversing.",
        "The only strongly responsive class, WALK, does shift population brake "
        "up with w (Δbrake +0.017 → +0.036 while Δgas recedes from its w=5 peak "
        "+0.047 → +0.037), but per sample it is a diffuse regime split, not "
        "compliance: at w=12, 35% of samples get materially MORE throttle vs "
        "17% more brake (both rise on only 0.7%), simultaneous "
        "high-gas-and-high-brake slots appear where the baseline had none "
        "(0 → 1.1%), and all of it sits past 60-78% code flips, 20% "
        "turn-signal flips and material out-of-box action slots rising from ~0 "
        "to 1.0% (2.7% if offsets are guided too) — decode degeneracy.",
        "The logit geometry says this is structural rather than a bad choice of "
        "w: the median (sample, quantizer) needs w ≈ 168 (cond 30) or w ≈ 230 "
        "(cond 100) — p25 still 57 / 96 — before any code overtakes the UNKNOWN "
        "top-1, so the numeric speed classes are ~1.5-2 orders of magnitude too "
        "weak for guidance in the usable range, and the offset head is "
        "irrelevant here (primary ≈ "
        "offsets-frozen variant to 4 decimals; extrapolating offsets only adds "
        "artifacts, `offset_scale=None` so nothing saturates them).",
        "**No usable w exists** — every w that moves actions moves them the wrong "
        "way or into degeneracy, so a serving-side CFG trick cannot rescue zone "
        "conditioning on this checkpoint; the levers stay the training-side ones "
        "(compliance-conditioned targets, rare-class oversampling, stronger aux "
        "supervision).",
        "",
    ]

    text = "\n".join(L)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
