"""Finding #10's information floor, recomputed as a function of the clip level.

If clipping is physically free (an action beyond `c` produces the same
outcome), then the part of the floor it removes was never resolvable error --
it was label noise being counted as error.
"""
import numpy as np, sys
sys.path.insert(0, "src")
from rmind.scripts.trajectory_action_controller import (
    read_ticks, stitch_horizon, drive_split, _path_length_m, _conditional_l1_floor,
    DEFAULT_TICKS, CONTINUOUS_RAW_COLS,
)

a = stitch_horizon(read_ticks(DEFAULT_TICKS), horizon=30, action_steps=1, with_gnss=False)
_, val_idx, _, _ = drive_split(a["input_id"], seed=7, val_frac=0.1)
v = val_idx.numpy()
speed = a["speed"][v, 0]
path = _path_length_m(a["speed"][v], a["time_stamp"][v])
brake = a["brake_pedal"][v]; steer = a["steering_angle"][v]

near = speed < 5.0
stat = near & (path < 0.5)     # zero trajectory: horizon-invariant
mover = near & (path >= 0.5)
print(f"val rows {len(v):,}; near-stop {near.sum():,} ({100*near.mean():.1f}%); "
      f"stationary {stat.sum():,} ({100*stat.sum()/near.sum():.1f}% of near-stop), "
      f"moving {mover.sum():,}\n")

print("BRAKE, stationary near-stop group -- oracle floor vs clip level")
print(f"{'clip c':>8} {'floor L1':>10} {'vs c=inf':>10} {'rows at c':>10} {'target med':>11} {'target p90':>11}")
for c in (np.inf, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15):
    x = np.minimum(brake[stat], c)
    f, _ = _conditional_l1_floor(values=x, speed_kmh=speed[stat], speed_bin=0.5)
    base = base_ if c is not np.inf else (base_ := f)
    print(f"{c:>8.2f} {f:>10.4f} {100*(f/base_-1):>+9.1f}% "
          f"{100*(brake[stat] > c).mean():>9.1f}% {np.median(x):>11.4f} {np.percentile(x,90):>11.4f}")

print("\nSame, but for the near-stop rows that DO move (trajectory is informative there)")
for c in (np.inf, 0.4, 0.3, 0.2):
    x = np.minimum(brake[mover], c)
    f, _ = _conditional_l1_floor(values=x, speed_kmh=speed[mover], speed_bin=0.5)
    print(f"  c={c:<6.2f} speed-only baseline L1 {f:.4f}   clipped rows {100*(brake[mover] > c).mean():.1f}%")

print("\nSTEERING, stationary near-stop group (|steer| clip)")
for c in (np.inf, 0.5, 0.4, 0.3, 0.2):
    x = np.clip(steer[stat], -c, c)
    f, _ = _conditional_l1_floor(values=x, speed_kmh=speed[stat], speed_bin=0.5)
    print(f"  c={c:<6.2f} floor L1 {f:.4f}   clipped rows {100*(np.abs(steer[stat]) > c).mean():.1f}%")
