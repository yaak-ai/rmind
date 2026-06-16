#! /usr/bin/env python
"""Self-contained HTML experiment report: metrics table + pred-vs-GT overlays.

The autonomous-experimentation review surface: one HTML file comparing N runs
on (a) a key-metric table (best-per-column highlighted), (b) interactive
pred-vs-GT overlays of the ACTUAL action values per drive (toggle runs via the
legend), (c) zoomed small-multiples of the top maneuver segments, (d) per-run
cards with motivation/verdict. Input: a manifest JSON listing runs:

    {
      "title": "...",
      "runs": [
        {"name": "exp01_control", "parquet": "reports/parquets/x.parquet",
         "wandb": "https://wandb.ai/...", "tags": "lds, cached",
         "notes": "motivation ...", "verdict": "baseline",
         "metrics": {"spike meanK": 0.326, ...}},   # optional extras
        ...
      ]
    }

Core metrics (overall/flat/spike steering chunk-L1, gas/brake L1) are computed
from the parquets directly; the optional "metrics" dict adds eval-only numbers
(oracle/coverage/bimodality) into the table.

Usage:
    uv run python -m rmind.scripts.flow_report manifest.json [out.html]
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GT = "policy/ground_truth/value/continuous/"
PR = "policy/prediction_value/value/continuous/"
FI = "batch/data/meta/ImageMetadata.cam_front_left/frame_idx"
ID = "batch/meta/input_id"
CHANNELS = ("steering_angle", "gas_pedal", "brake_pedal")
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _load(parquet: str) -> dict:
    df = pl.read_parquet(parquet)
    out = {
        "frame": np.stack(df[FI].to_numpy())[:, -1],  # current frame
        "drive": np.array(df[ID].to_list()),
    }
    for ch in CHANNELS:
        out[f"gt_{ch}"] = np.stack(df[GT + ch].to_numpy())
        out[f"pr_{ch}"] = np.stack(df[PR + ch].to_numpy())
    return out


def _core_metrics(d: dict) -> dict[str, float]:
    gs, ps = d["gt_steering_angle"], d["pr_steering_angle"]
    ok = np.isfinite(gs).all(1) & np.isfinite(ps).all(1)
    gs, ps = gs[ok], ps[ok]
    fe = np.abs(ps - gs).mean(1)
    spike = (np.abs(gs) > 0.5).any(1)
    flat = (np.abs(gs) < 0.05).all(1)
    m = {
        "steer L1": float(fe.mean()),
        "steer flat": float(fe[flat].mean()),
        "steer SPIKE": float(fe[spike].mean()) if spike.any() else float("nan"),
        "steer corr": float(np.corrcoef(ps.mean(1), gs.mean(1))[0, 1]),
    }
    for ch, label in (("gas_pedal", "gas L1"), ("brake_pedal", "brake L1")):
        g, p = d[f"gt_{ch}"][ok], d[f"pr_{ch}"][ok]
        m[label] = float(np.abs(p - g).mean())
    return m


def _metric_table(runs: list[dict], all_metrics: list[dict]) -> str:
    keys: list[str] = []
    for m in all_metrics:
        keys.extend(k for k in m if k not in keys)
    # lower-is-better for all except corr-like keys
    rows = []
    best: dict[str, float] = {}
    for k in keys:
        vals = [m.get(k) for m in all_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            continue
        best[k] = max(vals) if "corr" in k or "%" in k else min(vals)
    head = "<tr><th>run</th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
    for run, m in zip(runs, all_metrics, strict=True):
        cells = []
        for k in keys:
            v = m.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                cls = " class='best'" if v == best.get(k) else ""
                cells.append(f"<td{cls}>{v:.4f}</td>")
            else:
                cells.append("<td>—</td>")
        rows.append(f"<tr><td class='name'>{run['name']}</td>{''.join(cells)}</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def _overlay_fig(runs: list[dict], data: list[dict], channel: str) -> go.Figure:
    drives = sorted(set(data[0]["drive"].tolist()))
    fig = make_subplots(
        rows=len(drives), cols=1, shared_xaxes=False, vertical_spacing=0.04,
        subplot_titles=[d.split("/")[0] + " " + d.split("--")[0].split("/")[-1] for d in drives],
    )
    for r, drive in enumerate(drives, start=1):
        m0 = data[0]["drive"] == drive
        order = np.argsort(data[0]["frame"][m0])
        fig.add_trace(
            go.Scattergl(
                x=data[0]["frame"][m0][order],
                y=data[0][f"gt_{channel}"][m0][:, 0][order],
                mode="lines", line={"color": "crimson", "width": 1.6},
                name="GT", legendgroup="GT", showlegend=(r == 1),
            ),
            row=r, col=1,
        )
        for i, (run, d) in enumerate(zip(runs, data, strict=True)):
            mi = d["drive"] == drive
            oi = np.argsort(d["frame"][mi])
            fig.add_trace(
                go.Scattergl(
                    x=d["frame"][mi][oi], y=d[f"pr_{channel}"][mi][:, 0][oi],
                    mode="markers", marker={"size": 3, "color": COLORS[i % len(COLORS)]},
                    opacity=0.75, name=run["name"], legendgroup=run["name"],
                    showlegend=(r == 1),
                ),
                row=r, col=1,
            )
    fig.update_layout(
        height=260 * len(drives), template="plotly_dark",
        title=f"{channel} — first predicted step vs GT (click legend to toggle runs)",
        legend={"orientation": "h", "y": 1.01},
        margin={"l": 40, "r": 20, "t": 80, "b": 30},
    )
    return fig


def _spike_zoom_fig(runs: list[dict], data: list[dict], n_seg: int = 6) -> go.Figure:
    gt = data[0]["gt_steering_angle"][:, 0]
    frame, drive = data[0]["frame"], data[0]["drive"]
    spike = np.abs(np.nan_to_num(gt)) > 0.5
    segs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for i in np.flatnonzero(spike):
        key = (drive[i], int(frame[i]) // 400)
        if key not in seen:
            seen.add(key)
            segs.append((drive[i], int(frame[i])))
        if len(segs) >= n_seg:
            break
    rows = (len(segs) + 1) // 2
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[f"{d.split('-HQ')[0]} @ {f}" for d, f in segs],
        vertical_spacing=0.10,
    )
    for s, (dr, fc) in enumerate(segs):
        r, c = s // 2 + 1, s % 2 + 1
        for i, (run, d) in enumerate(zip(runs, data, strict=True)):
            m = (d["drive"] == dr) & (np.abs(d["frame"].astype(int) - fc) < 300)
            o = np.argsort(d["frame"][m])
            if s == 0 or True:
                fig.add_trace(
                    go.Scattergl(
                        x=d["frame"][m][o], y=d["pr_steering_angle"][m][:, 0][o],
                        mode="lines+markers", marker={"size": 3},
                        line={"width": 1, "color": COLORS[i % len(COLORS)]},
                        name=run["name"], legendgroup=run["name"], showlegend=(s == 0),
                    ),
                    row=r, col=c,
                )
        m0 = (data[0]["drive"] == dr) & (np.abs(data[0]["frame"].astype(int) - fc) < 300)
        o0 = np.argsort(data[0]["frame"][m0])
        fig.add_trace(
            go.Scattergl(
                x=data[0]["frame"][m0][o0], y=data[0]["gt_steering_angle"][m0][:, 0][o0],
                mode="lines", line={"color": "crimson", "width": 2},
                name="GT", legendgroup="GT", showlegend=(s == 0),
            ),
            row=r, col=c,
        )
    fig.update_layout(
        height=300 * rows, template="plotly_dark",
        title="maneuver zoom — steering, top spike segments (±300 frames)",
        legend={"orientation": "h", "y": 1.02},
        margin={"l": 40, "r": 20, "t": 90, "b": 30},
    )
    return fig


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "reports/report.html")
    runs = manifest["runs"]
    data = [_load(r["parquet"]) for r in runs]
    metrics = [{**_core_metrics(d), **r.get("metrics", {})} for r, d in zip(runs, data, strict=True)]

    cards = "".join(
        f"""<div class="card">
        <h3>{r['name']} <span class="verdict v-{r.get('verdict', 'na')}">{r.get('verdict', '')}</span></h3>
        <p class="meta">{r.get('tags', '')} · <a href="{r.get('wandb', '#')}">wandb</a></p>
        <p>{r.get('notes', '')}</p></div>"""
        for r in runs
    )
    steer = _overlay_fig(runs, data, "steering_angle").to_html(full_html=False, include_plotlyjs="cdn")
    gas = _overlay_fig(runs, data, "gas_pedal").to_html(full_html=False, include_plotlyjs=False)
    zoom = _spike_zoom_fig(runs, data).to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{manifest.get('title', 'experiment report')}</title>
<style>
 body {{ background:#111; color:#ddd; font-family: -apple-system, 'Segoe UI', sans-serif; margin: 24px; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 36px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
 table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0; }}
 th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: right; }}
 th {{ background: #1c1c1c; }} td.name {{ text-align: left; font-weight: 600; }}
 td.best {{ background: #14391f; color: #7be08f; font-weight: 700; }}
 .cards {{ display: flex; gap: 14px; flex-wrap: wrap; }}
 .card {{ background: #1a1a1a; border: 1px solid #2c2c2c; border-radius: 8px; padding: 12px 16px; max-width: 380px; }}
 .card h3 {{ margin: 0 0 6px; font-size: 14px; }}
 .meta {{ color: #888; font-size: 12px; margin: 2px 0 8px; }}
 .card p {{ font-size: 12.5px; line-height: 1.45; margin: 4px 0; }}
 .verdict {{ font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 6px; }}
 .v-baseline {{ background: #1d3a5f; }} .v-positive {{ background: #14512a; }}
 .v-null {{ background: #4d4d22; }} .v-negative {{ background: #5c1f1f; }} .v-na {{ background: #333; }}
 a {{ color: #6fb3ff; }}
</style></head><body>
<h1>{manifest.get('title', 'experiment report')}</h1>
<h2>runs</h2><div class="cards">{cards}</div>
<h2>key metrics (held-out val drives; green = best per column)</h2>
{_metric_table(runs, metrics)}
<h2>steering — pred vs GT per drive</h2>{steer}
<h2>maneuver zoom</h2>{zoom}
<h2>gas — pred vs GT per drive</h2>{gas}
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
