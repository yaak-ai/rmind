"""Step 1: is the action distribution actually saturating, and where?"""
import numpy as np, polars as pl, sys
sys.path.insert(0, "src")
from rmind.scripts.trajectory_action_controller import read_ticks, stitch_horizon, SPEED_BANDS

TICKS = "outputs/2026-07-16/09-34-08/predictions/yaak/alex-tmp/model-4vqatiom:v4.ticks.parquet"
ticks = read_ticks(TICKS)
print(f"{ticks.height:,} ticks, {ticks['input_id'].n_unique()} drives")

a = stitch_horizon(ticks, horizon=2, action_steps=2, with_gnss=False)
v0 = a["speed"][:, 1]          # speed at the tick the action is taken
dv = a["speed"][:, 2] - a["speed"][:, 1]
gas, brake, steer = (a[f][:, 0] for f in ("gas_pedal", "brake_pedal", "steering_angle"))
print(f"{len(v0):,} (action, dv) pairs\n")

qs = [50, 75, 90, 95, 99, 99.9]
for name, x in (("gas", gas), ("brake", brake), ("steer", steer)):
    print(f"{name:6s} min={x.min():.3f} max={x.max():.3f} mean={x.mean():.4f} "
          f"frac>0={100*(np.abs(x)>1e-6).mean():5.1f}%  "
          + "  ".join(f"p{q}={np.percentile(np.abs(x), q):.3f}" for q in qs))
print()

# --- saturation: does dv still respond to the pedal above a threshold?
def slope(x, y):
    if len(x) < 50 or x.std() < 1e-6: return np.nan
    return np.polyfit(x, y, 1)[0]

print("brake -> dv slope by (speed band, brake decile-ish bin). "
      "slope ~ 0 => that pedal range is unidentifiable from the outcome")
BINS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]
hdr = " ".join(f"[{lo:.2f},{hi:.2f})".rjust(13) for lo, hi in zip(BINS[:-1], BINS[1:]))
print(f"{'band':>10s} {hdr}")
for lo, hi in SPEED_BANDS:
    m = (v0 >= lo) & (v0 < hi) & (brake > 0.05)
    cells = []
    for blo, bhi in zip(BINS[:-1], BINS[1:]):
        mm = m & (brake >= blo) & (brake < bhi)
        cells.append(f"{slope(brake[mm], dv[mm]):+7.2f}({mm.sum()//1000:4d}k)" if mm.sum() >= 50 else "        -    ")
    print(f"{lo:4d}-{hi:<5d} " + " ".join(c.rjust(13) for c in cells))
print()

print("mass above candidate clip thresholds, by speed band (% of rows with the pedal beyond c)")
CLIPS = [0.2, 0.3, 0.4, 0.5, 0.6]
for name, x in (("gas", gas), ("brake", brake), ("|steer|", np.abs(steer))):
    print(f"  {name}")
    print("      band  " + "  ".join(f"c={c:.1f}".rjust(8) for c in CLIPS) + "     n")
    for lo, hi in SPEED_BANDS:
        m = (v0 >= lo) & (v0 < hi)
        print(f"  {lo:4d}-{hi:<5d} " + "  ".join(f"{100*(x[m]>c).mean():7.2f}%" for c in CLIPS) + f"  {m.sum():>9,}")
