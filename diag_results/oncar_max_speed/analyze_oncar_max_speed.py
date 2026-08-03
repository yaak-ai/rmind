#!/usr/bin/env python3
"""On-car max-speed (zone limit) conditioning analysis.

Question: when the max-speed input was changed during a drive, did the model's
gas/brake predictions change at all?

Data (all copied into ./raw/, see analysis.md for provenance):
  * ``<ts>_predictions.jsonl``  one row per control tick: gas/brake/steering/
    turn_signal predictions, ``engine`` (= "plan:i/H nNsS" + "*" on a fresh
    plan), ``inference_ms``, ``ai_control_enabled``, ``mcm_sent``.
  * ``<ts>.json``               event log (a JSON *list*): ai_policy_enabled,
    human_override, mcm_engaged, ...
  * ``<ts>_vehicle.jsonl``      CAN feedback at ~60 Hz: achieved ``speed``
    (km/h), gas_pedal, brake_pedal.  Only committed for the two analysed
    sessions (the others are 3-15 MB each); ``--speed`` degrades gracefully.
  * ``zellij_pane_t2_scrollback.txt``  the ONLY surviving record of the drivr
    process stdout, recovered from a live zellij pane's in-memory scrollback.
    ``Zone max-speed set from UI  max_speed=<v>`` lines here are the only
    source for the per-tick max_speed setting -- it is NOT a column in
    predictions.jsonl, not an event in the event log, and not a rerun entity.

Run: python3 analyze_oncar_max_speed.py [--raw raw]
"""

from __future__ import annotations

import argparse
import bisect
import calendar
import glob
import json
import math
import os
import random
import re
import statistics as st
from collections import defaultdict

# --- the car (delta-devcar / delta-emc1.kit) logs UTC; predictions
# `timestamp_iso` and structlog `...Z` lines are therefore the same clock. ---
ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})\.(\d+)Z?")


def iso_to_epoch_utc(s: str) -> float:
    """Parse '2026-08-03T13:49:44.736267Z' / '...736' as UTC epoch seconds."""
    m = ISO_RE.match(s)
    if not m:
        raise ValueError(f"unparseable timestamp: {s!r}")
    date, hms, frac = m.groups()
    y, mo, d = (int(x) for x in date.split("-"))
    h, mi, sec = (int(x) for x in hms.split(":"))
    base = calendar.timegm((y, mo, d, h, mi, sec, 0, 0, 0))
    return base + float("0." + frac)


# ---------------------------------------------------------------- loading ---

def load_sessions(raw: str) -> dict:
    """Return {session_id: {"pred": [...], "events": [...], "veh": [...]}}."""
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(raw, "*_predictions.jsonl"))):
        sid = os.path.basename(path).replace("_predictions.jsonl", "")
        rows = [json.loads(line) for line in open(path) if line.strip()]
        if not rows:
            continue
        ev_path = os.path.join(raw, f"{sid}.json")
        events = []
        if os.path.exists(ev_path):
            try:
                loaded = json.load(open(ev_path))
                events = loaded if isinstance(loaded, list) else []
            except json.JSONDecodeError:
                events = []
        veh_path = os.path.join(raw, f"{sid}_vehicle.jsonl")
        veh = []
        if os.path.exists(veh_path):
            veh = [json.loads(line) for line in open(veh_path) if line.strip()]
            veh.sort(key=lambda r: r["timestamp"])
        out[sid] = {"pred": rows, "events": events, "veh": veh, "path": path}
    return out


def join_speed(pred: list, veh: list, tol: float = 0.25) -> None:
    """Attach nearest-in-time achieved ``speed`` (km/h) to each prediction row.

    Nearest-neighbour rather than interpolation: the CAN stream is ~60 Hz, so
    the nearest sample is <=8 ms away in practice; ``tol`` only guards against
    gaps (dropouts / stream not yet started).
    """
    if not veh:
        return
    ts = [r["timestamp"] for r in veh]
    for r in pred:
        i = bisect.bisect_left(ts, r["timestamp"])
        best, bestdt = None, tol
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(ts):
                dt = abs(ts[j] - r["timestamp"])
                if dt <= bestdt:
                    best, bestdt = veh[j], dt
        if best is not None:
            r["speed"] = best.get("speed")
            r["gas_pedal_actual"] = best.get("gas_pedal")
            r["brake_pedal_actual"] = best.get("brake_pedal")


def parse_max_speed_changes(scrollback: str) -> list:
    """Recover (epoch, value) of every ``Zone max-speed set from UI`` line.

    This is the ONLY recoverable source of the max_speed setting.  drivr logs
    the POST unconditionally (drivr.py:3181-3182, no changed-value guard), so a
    line proves the value was *set*, not that it differed from the prior state.
    """
    out = []
    if not os.path.exists(scrollback):
        return out
    pat = re.compile(r"^(\S+Z)\s+Zone max-speed set from UI\s+max_speed=(\S+)")
    for line in open(scrollback, errors="replace"):
        m = pat.match(line)
        if m:
            raw = m.group(2)
            val = None if raw in ("None", "null") else float(raw)
            out.append((iso_to_epoch_utc(m.group(1)), val))
    return out


def scrollback_coverage(scrollback: str) -> tuple:
    """(first, last) epoch of timestamped structlog lines in the pane dump.

    Bounds *where* a missing max-speed line is real evidence of "no change":
    outside this window the scrollback ring buffer has already discarded the
    output, so absence proves nothing.
    """
    if not os.path.exists(scrollback):
        return (None, None)
    stamps = []
    for line in open(scrollback, errors="replace"):
        m = ISO_RE.match(line)
        if m:
            stamps.append(iso_to_epoch_utc(m.group(0)))
    return (min(stamps), max(stamps)) if stamps else (None, None)


# ------------------------------------------------------------- statistics ---

def fresh_plans(rows: list) -> list:
    """Rows that start a fresh plan (inference actually ran on this tick)."""
    return [r for r in rows if r["engine"].rstrip().endswith("*")]


def plan_blocks(rows: list) -> list:
    """Group ticks into plans: a new block starts at each fresh-plan row.

    Ticks inside one plan are *steps of a single forward pass*, so they are not
    independent samples.  Every CI/bootstrap below resamples plans, not ticks.
    """
    blocks, cur = [], []
    for r in rows:
        if r["engine"].rstrip().endswith("*") and cur:
            blocks.append(cur)
            cur = []
        cur.append(r)
    if cur:
        blocks.append(cur)
    return blocks


def block_bootstrap_mean_ci(blocks: list, key: str, n_boot: int = 5000,
                            seed: int = 1337) -> tuple:
    """(mean, lo, hi) 95% CI for the tick-mean of `key`, resampling plans."""
    vals = [[r[key] for r in b if r.get(key) is not None] for b in blocks]
    vals = [v for v in vals if v]
    if not vals:
        return (float("nan"),) * 3
    flat = [x for v in vals for x in v]
    mean = st.mean(flat)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        f = [x for v in pick for x in v]
        means.append(st.mean(f))
    means.sort()
    return (mean, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def mde(blocks_a: list, blocks_b: list, key: str) -> float:
    """Minimum detectable difference in tick-means (alpha=.05, power=.80).

    Uses the *plan* as the unit of analysis: n = number of plans, sigma = SD of
    the per-plan mean.  This is the honest sample size -- treating each tick as
    independent would understate the MDE by ~sqrt(ticks_per_plan).
    """
    def per_plan(bs):
        return [st.mean([r[key] for r in b if r.get(key) is not None])
                for b in bs if any(r.get(key) is not None for r in b)]
    a, b = per_plan(blocks_a), per_plan(blocks_b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sd = math.sqrt((st.variance(a) + st.variance(b)) / 2)
    return 2.80 * sd * math.sqrt(1 / len(a) + 1 / len(b))


SPEED_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 1e9)]


def bin_label(lo, hi):
    return f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+"


def speed_binned(rows: list) -> dict:
    """{bin_label: [rows]} using the joined achieved speed (km/h)."""
    out = defaultdict(list)
    for r in rows:
        s = r.get("speed")
        if s is None:
            continue
        for lo, hi in SPEED_BINS:
            if lo <= s < hi:
                out[bin_label(lo, hi)].append(r)
                break
    return out


# ----------------------------------------------------------------- report ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"))
    ap.add_argument("--no-speed", action="store_true", help="skip vehicle join")
    args = ap.parse_args()

    sessions = load_sessions(args.raw)
    scroll = os.path.join(args.raw, "zellij_pane_t2_scrollback.txt")
    changes = parse_max_speed_changes(scroll)
    cov0, cov1 = scrollback_coverage(scroll)

    if not args.no_speed:
        for s in sessions.values():
            join_speed(s["pred"], s["veh"])

    def iso(t):
        import datetime
        return datetime.datetime.utcfromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]

    print("=" * 78)
    print("1. RECOVERED max_speed SETTING TIMELINE (sole source: pane scrollback)")
    print("=" * 78)
    if cov0:
        print(f"scrollback covers   {iso(cov0)} .. {iso(cov1)} UTC "
              f"({cov1 - cov0:.0f} s).  OUTSIDE this window the setting is UNRECOVERABLE.")
    for t, v in changes:
        print(f"  {iso(t)}  max_speed set to {v!r}"
              + ("  (UNKNOWN / NaN fill)" if v is None else " km/h"))
    if not changes:
        print("  (none found)")

    print()
    print("=" * 78)
    print("2. SESSION TIMELINE")
    print("=" * 78)
    print("max_speed column: 'verified' only inside the scrollback window; every")
    print("other session's setting is UNRECOVERABLE (not logged to any artifact).")
    hdr = ("%-16s %-19s %5s %5s %-6s %7s %7s %7s %7s %8s %7s %-10s" %
           ("session", "window UTC", "ticks", "plans", "plancfg", "gas_mn",
            "gas_md", "brk_mn", "brk_md", "speed_mn", "inf_ms", "max_speed"))
    print(hdr)
    print("-" * len(hdr))
    for sid, s in sessions.items():
        rows = s["pred"]
        gas = [r["gas"] for r in rows]
        brk = [r["brake"] for r in rows]
        sp = [r["speed"] for r in rows if r.get("speed") is not None]
        inf = [r["inference_ms"] for r in fresh_plans(rows)]
        cfg = "/".join(sorted({r["engine"].rstrip("*").split()[-1] for r in rows}))
        t0, t1 = rows[0]["timestamp"], rows[-1]["timestamp"]
        covered = cov0 is not None and t0 >= cov0 and t1 <= cov1
        if covered:
            prior = [v for t, v in changes if t <= t0]
            ms = ("%g (verified)" % prior[-1]) if prior and prior[-1] is not None \
                else ("UNKNOWN(verified)" if prior else "unrecoverable")
        else:
            ms = "unrecoverable"
        print("%-16s %-19s %5d %5d %-6s %7.4f %7.4f %7.4f %7.4f %8s %7s %-10s" % (
            sid, f"{rows[0]['timestamp_iso'][11:19]}-{rows[-1]['timestamp_iso'][11:19]}",
            len(rows), len(fresh_plans(rows)), cfg,
            st.mean(gas), st.median(gas), st.mean(brk), st.median(brk),
            ("%.2f" % st.mean(sp)) if sp else "n/a",
            ("%.0f" % st.median(inf)) if inf else "n/a", ms))

    print()
    print("=" * 78)
    print("3. PRE/POST SPLIT AT EACH RECOVERED max_speed CHANGE")
    print("=" * 78)
    print("A within-drive test needs ticks on BOTH sides of a change.")
    for t, v in changes:
        print(f"\n  change @ {iso(t)} -> {v!r}")
        for sid, s in sessions.items():
            rows = s["pred"]
            pre = [r for r in rows if r["timestamp"] < t]
            post = [r for r in rows if r["timestamp"] >= t]
            if not pre and not post:
                continue
            print("    %-16s ticks pre/post %4d/%4d   plans pre/post %3d/%3d" % (
                sid, len(pre), len(post), len(fresh_plans(pre)), len(fresh_plans(post))))
            first_fresh = next((r for r in fresh_plans(rows)
                                if r["timestamp"] >= t), None)
            if first_fresh:
                print("      first fresh plan after the change: %s (+%.1f s) "
                      "-- the earliest tick the new input could reach the model"
                      % (first_fresh["timestamp_iso"][11:23],
                         first_fresh["timestamp"] - t))

    print()
    print("=" * 78)
    print("4. SPEED-BINNED gas/brake COMPARISON (context control)")
    print("=" * 78)
    pairs = [("20260803_134521", "20260803_134900")]
    for a, b in pairs:
        if a not in sessions or b not in sessions:
            continue
        print(f"\n  {a} (max_speed UNKNOWN)  vs  {b} (max_speed = 10.0 km/h, verified)")
        ba, bb = speed_binned(sessions[a]["pred"]), speed_binned(sessions[b]["pred"])
        hdr2 = ("  %-9s %5s %5s %5s %5s %8s %8s %9s %9s %8s %8s %9s %9s" %
                ("speed bin", "nA", "nB", "plA", "plB", "gasA", "gasB",
                 "dgas", "MDE_gas", "brkA", "brkB", "dbrake", "MDE_brk"))
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for lo, hi in SPEED_BINS:
            k = bin_label(lo, hi)
            ra, rb = ba.get(k, []), bb.get(k, [])
            if not ra and not rb:
                continue
            pa, pb = plan_blocks(ra), plan_blocks(rb)
            ga = st.mean([r["gas"] for r in ra]) if ra else float("nan")
            gb = st.mean([r["gas"] for r in rb]) if rb else float("nan")
            bka = st.mean([r["brake"] for r in ra]) if ra else float("nan")
            bkb = st.mean([r["brake"] for r in rb]) if rb else float("nan")
            mg = mde(pa, pb, "gas") if (ra and rb) else float("nan")
            mb = mde(pa, pb, "brake") if (ra and rb) else float("nan")
            # The PLAN is the unit of analysis (ticks inside a plan are steps of
            # one forward pass), so the smallness flag counts plans, not ticks.
            flag = ("  << underpowered (<15 plans/arm)"
                    if (len(pa) < 15 or len(pb) < 15) else "")
            print("  %-9s %5d %5d %5d %5d %8.4f %8.4f %+9.4f %9.4f %8.4f %8.4f "
                  "%+9.4f %9.4f%s"
                  % (k, len(ra), len(rb), len(pa), len(pb), ga, gb, gb - ga, mg,
                     bka, bkb, bkb - bka, mb, flag))

        print("\n  pooled (all speeds), 95%% CI by plan-block bootstrap:")
        for sid in (a, b):
            blocks = plan_blocks(sessions[sid]["pred"])
            for key in ("gas", "brake"):
                mean, lo, hi = block_bootstrap_mean_ci(blocks, key)
                print("    %-16s %-5s mean %.4f  [%.4f, %.4f]  (%d plans)"
                      % (sid, key, mean, lo, hi, len(blocks)))
        for key in ("gas", "brake"):
            m = mde(plan_blocks(sessions[a]["pred"]),
                    plan_blocks(sessions[b]["pred"]), key)
            print("    MDE %-5s (alpha=.05, power=.80, unit=plan): %.4f" % (key, m))

        # --- Is the brake gap mechanical rather than conditioning? ------------
        # The two sessions use different CLI plan configs: A is n6s0 (serves
        # steps [0,6) of a 6-step plan, so plan:1/6 first) and B is n5s1 (serves
        # [1,6), so plan:2/6 first -- step 1 of EVERY plan is skipped).  If
        # brake is front-loaded within a plan, B mechanically shows less brake
        # no matter what the max-speed input is.
        print("\n  brake/gas by step-index-within-plan (the plan-config confound,"
              "\n  A=%s n6s0 vs B=%s n5s1):" % (a, b))
        print("    %-6s %6s %8s %8s   %6s %8s %8s"
              % ("plan:i", "nA", "gasA", "brkA", "nB", "gasB", "brkB"))
        step_re = re.compile(r"plan:(\d+)/")
        per = {}
        for tag, sid in (("A", a), ("B", b)):
            d = defaultdict(list)
            for r in sessions[sid]["pred"]:
                m = step_re.search(r["engine"])
                if m:
                    d[int(m.group(1))].append(r)
            per[tag] = d
        for i in sorted(set(per["A"]) | set(per["B"])):
            ra, rb = per["A"].get(i, []), per["B"].get(i, [])
            def f(rows, key):
                return ("%8.4f" % st.mean([r[key] for r in rows])) if rows else "       -"
            print("    %-6s %6d %s %s   %6d %s %s"
                  % (f"{i}/6", len(ra), f(ra, "gas"), f(ra, "brake"),
                     len(rb), f(rb, "gas"), f(rb, "brake")))

    print()
    print("=" * 78)
    print("5. ENGINE FINGERPRINT (inference_ms, fresh-plan rows only)")
    print("=" * 78)
    print("  The engine used is NOT recorded in any artifact.  TRT precision")
    print("  changes GPU compute time, so inference_ms clusters identify it.")
    groups = [("max-speed checkout", sorted(sessions))]
    ctrl = sorted(glob.glob(os.path.join(args.raw, "control_receding", "*_predictions.jsonl")))
    print("  %-46s %5s %8s %8s %8s" % ("source", "n", "median", "p10", "p90"))
    for _, sids in groups:
        for sid in sids:
            v = sorted(r["inference_ms"] for r in fresh_plans(sessions[sid]["pred"]))
            if v:
                print("  %-46s %5d %8.1f %8.1f %8.1f" % (
                    sid, len(v), st.median(v), v[int(.1 * len(v))], v[int(.9 * len(v))]))
    for p in ctrl:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        v = sorted(r["inference_ms"] for r in fresh_plans(rows))
        if v:
            print("  %-46s %5d %8.1f %8.1f %8.1f" % (
                "control/" + os.path.basename(p)[:34], len(v), st.median(v),
                v[int(.1 * len(v))], v[int(.9 * len(v))]))

    print()
    print("=" * 78)
    print("6. PLUMBING EVIDENCE (grep of the recovered stdout)")
    print("=" * 78)
    pats = ["max_speed_binding", "declares no max-speed", "maxspeed engine without",
            "Zone max-speed set from UI"]
    for f in sorted(glob.glob(os.path.join(args.raw, "zellij_pane_*.txt"))):
        txt = open(f, errors="replace").read()
        hits = {p: txt.count(p) for p in pats}
        launches = sorted(set(re.findall(r"drivr\.app\.drivr --model \S+[^\n]*", txt)))
        print(f"  {os.path.basename(f)}: {hits}")
        for l in launches:
            print(f"     launch seen in ps: {l.strip()}")

    print()
    print("=" * 78)
    print("7. POWER VERDICT vs THE OFFLINE REFERENCE EFFECT SIZES")
    print("=" * 78)
    # diag_results/eval_v0/override_probe_mv_v1.md, model-0nr1ydjm:v1 (= the
    # checkpoint behind maxspeed_aux-0nr1ydjm-v1.trt), 768 val samples, argmax.
    ref = {"10 km/h": 0.0004, "30 km/h": 0.0008, "100 km/h": 0.0001,
           "WALK (5)": 0.0167, "UNLIMITED": 0.0005}
    a, b = "20260803_134521", "20260803_134900"
    if a in sessions and b in sessions:
        m = mde(plan_blocks(sessions[a]["pred"]), plan_blocks(sessions[b]["pred"]), "gas")
        print("  on-car MDE for Delta-gas (best available pair, unit=plan): %.4f" % m)
        print("  %-12s %10s %10s   detectable on-car?" % ("offline cond", "dgas", "MDE/dgas"))
        for k, v in sorted(ref.items(), key=lambda kv: -kv[1]):
            print("  %-12s %+10.4f %10.0fx   %s" % (
                k, v, m / v, "NO" if v < m else "yes"))
        print("  The setting actually used on-car was 10 km/h, whose offline dgas")
        print("  (+0.0004) is ~%.0fx below what this drive could resolve." % (m / ref["10 km/h"]))


if __name__ == "__main__":
    main()
