"""Is the steering/gas tail identifiable, or does it saturate like brake?"""
import numpy as np, sys
sys.path.insert(0, "src")
from rmind.scripts.trajectory_action_controller import read_ticks, stitch_horizon, SPEED_BANDS, DEFAULT_TICKS

a = stitch_horizon(read_ticks(DEFAULT_TICKS), horizon=2, action_steps=2, with_gnss=False)
v0 = a["speed"][:, 1]
dv = a["speed"][:, 2] - a["speed"][:, 1]
dh = np.deg2rad((a["heading"][:, 2] - a["heading"][:, 1] + 180) % 360 - 180)
gas, steer = a["gas_pedal"][:, 0], a["steering_angle"][:, 0]

def slope(x, y, n=50):
    return np.polyfit(x, y, 1)[0] if len(x) >= n and x.std() > 1e-6 else np.nan

def table(name, act, resp, bins, extra=""):
    print(f"\n{name} -> {extra} slope, by (speed band, magnitude bin). "
          "|slope| ~ 0 => saturated / unidentifiable")
    print(f"{'band':>10s} " + " ".join(f"[{lo:.2f},{hi:.2f})".rjust(14) for lo, hi in zip(bins[:-1], bins[1:])))
    for lo, hi in SPEED_BANDS:
        m = (v0 >= lo) & (v0 < hi)
        cells = []
        for blo, bhi in zip(bins[:-1], bins[1:]):
            mm = m & (np.abs(act) >= blo) & (np.abs(act) < bhi)
            s = slope(np.abs(act[mm]), np.sign(act[mm]) * resp[mm] if name.startswith("|steer|") else resp[mm])
            cells.append(f"{s:+8.3f}({mm.sum()//1000:4d}k)" if mm.sum() >= 50 else "       -      ")
        print(f"{lo:4d}-{hi:<5d} " + " ".join(c.rjust(14) for c in cells))

table("|steer|", steer, dh, [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.01], extra="signed dheading (rad/tick)")
table("gas", gas, dv, [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.01], extra="dv (km/h/tick)")
