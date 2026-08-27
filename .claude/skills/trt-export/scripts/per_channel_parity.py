"""Per-ACTION-CHANNEL parity over all 200 trials, mixed engine vs rmind-native reference.

max|d| over the whole (6 x 4) chunk conflates turn_signal with the control channels.
Channels: 0=gas_pedal 1=brake_pedal 2=steering_angle 3=turn_signal (continuous norm = Identity).
"""
import json, sys, pathlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_trt_parity import TRTRunner, build_inputs

NAMES = ["gas_pedal", "brake_pedal", "steering_angle", "turn_signal"]
CACHE = None  # --cache
ENGINES = {}  # --engines
FRAMES = None  # --frames
TOL = 0.02

def load_frames(spec):
    import cv2
    out = []
    for fp in [pathlib.Path(x) for x in spec.split(",") if x.strip()]:
        img = cv2.imread(str(fp))
        if img is None: continue
        h = img.shape[0]; side = min(h, 320); top = max((h - side)//2, 0)
        out.append(cv2.cvtColor(img[top:top+side,:], cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
    return out

class _S:
    def __init__(s, d): s.name, s.shape, s.type = d["name"], d["shape"], d["type"]
class SS:
    def __init__(s, x): s._x = x
    def get_inputs(s): return s._x

import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--cache", required=True, help="reference npz (parity_matrix --ref-cache format)")
_ap.add_argument("--engines", required=True, help="comma-separated name=path pairs, or bare paths")
_ap.add_argument("--frames", required=True)
_ap.add_argument("--tol", type=float, default=0.02)
_a = _ap.parse_args()
CACHE, FRAMES, TOL = _a.cache, _a.frames, _a.tol
ENGINES = {}
for _e in _a.engines.split(","):
    _e = _e.strip()
    if not _e: continue
    ENGINES[_e.split("=",1)[0] if "=" in _e else pathlib.Path(_e).name] = _e.split("=",1)[-1]

z = np.load(CACHE, allow_pickle=True); meta = json.loads(str(z["meta"]))
action = meta["action"]; ai = meta["out_names"].index(action); N = meta["trials"]
sess = SS([_S(s) for s in meta["inputs"]]); frames = load_frames(FRAMES)
runners = {k: TRTRunner(pathlib.Path(v)) for k, v in ENGINES.items()}

stats = {k: dict(mx=np.zeros(4), dec_any=0, dec_ctrl=0, sm=np.zeros(4)) for k in runners}
rng = np.random.default_rng(1234)
for t in range(N):
    feed = build_inputs(sess, rng, frames[t % len(frames)] if frames else None)
    nat = np.asarray(z[f"t{t}_{ai}"], dtype=np.float64).reshape(-1, 4)
    for k, r in runners.items():
        got = np.asarray(r(feed)[action], dtype=np.float64).reshape(-1, 4)
        d = np.abs(got - nat)
        s = stats[k]
        s["mx"] = np.maximum(s["mx"], d.max(axis=0))
        s["sm"] += d.mean(axis=0)
        if (d > TOL).any(): s["dec_any"] += 1
        if (d[:, :3] > TOL).any(): s["dec_ctrl"] += 1

print(f"\nvs rmind-native reference, n={N}, tol={TOL}\n")
for k, s in stats.items():
    print(f"### {k}")
    print(f"  {'channel':<16}{'max|d|':>10}{'mean|d|':>11}")
    for c, nm in enumerate(NAMES):
        print(f"  {nm:<16}{s['mx'][c]:10.4f}{s['sm'][c]/N:11.5f}")
    print(f"  decision changes ANY channel      : {s['dec_any']}/{N}")
    print(f"  decision changes CONTROL only     : {s['dec_ctrl']}/{N}   "
          f"(gas/brake/steering; excludes turn_signal)")
    print()
