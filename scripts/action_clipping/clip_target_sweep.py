"""Same clip sweep, repeated over 5 training seeds.

The split is fixed (seed 7, identical rows for every run); only the `_fit`
seed -- init + minibatch stream -- varies. Within one seed, baseline and each
variant share init and sampling order, so the comparison is PAIRED: the seed
effect cancels and what is left is the effect of the clipped target.
"""
import json, sys, numpy as np, torch
sys.path.insert(0, "src")
from rmind.scripts.trajectory_action_controller import (
    read_ticks, stitch_horizon, dead_reckon_stitched, drive_split, _fit,
    _horizon_features, DEFAULT_TICKS, SPEED_BANDS, CONTINUOUS_RAW_COLS, VAL_CHUNK_ROWS,
)

HORIZON, STEPS, SPLIT_SEED = 30, 20_000, 7
FIT_SEEDS = [7, 11, 23, 42, 101]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

a = stitch_horizon(read_ticks(DEFAULT_TICKS), horizon=HORIZON, action_steps=1, with_gnss=False)
position, heading = dead_reckon_stitched(a)
speed_now = torch.from_numpy(a["speed"][:, 0]).float()
features = _horizon_features(position=position, heading=heading, speed_now=speed_now, horizon=HORIZON)
del position, heading
raw = {f: torch.from_numpy(a[f]).float() for f in CONTINUOUS_RAW_COLS}
turn_gt = torch.from_numpy(a["turn_signal"]).long()
train_idx, val_idx, *_ = drive_split(a["input_id"], seed=SPLIT_SEED, val_frac=0.1)
val_speed = a["speed"][val_idx.numpy(), 0]
del a
LOW = torch.from_numpy(speed_now.numpy() < 5.0)
features = features.to(DEV)

def clipped(field, c, low_only=False):
    t = {f: v.clone() for f, v in raw.items()}
    lim = torch.where(LOW, c, 10.0) if low_only else torch.full_like(t[field], c)
    t[field] = t[field].clamp(-lim, lim) if field == "steering_angle" else t[field].clamp(max=lim)
    return t

CONFIGS = {
    "baseline": raw,
    "brake<=0.4": clipped("brake_pedal", 0.4),
    "brake<=0.3": clipped("brake_pedal", 0.3),
    "brake<=0.2": clipped("brake_pedal", 0.2),
    "gas<=0.3": clipped("gas_pedal", 0.3),
    "|steer|<=0.3@v<5": clipped("steering_angle", 0.3, low_only=True),
}
truth = {f: raw[f][val_idx].numpy() for f in CONTINUOUS_RAW_COLS}
turn_truth = turn_gt[val_idx].numpy()

def evaluate(model):
    model, out, turn = model.to(DEV).eval(), {f: [] for f in CONTINUOUS_RAW_COLS}, []
    with torch.no_grad():
        for chunk in val_idx.split(VAL_CHUNK_ROWS):
            o = model(features[chunk])
            for f in CONTINUOUS_RAW_COLS:
                out[f].append(o["continuous"][f][..., 0].cpu())
            turn.append(o["turn_signal"].argmax(-1).cpu())
    pred = {f: torch.cat(v).numpy() for f, v in out.items()}
    row = {"turn_acc": float((torch.cat(turn).numpy() == turn_truth).mean())}
    for f in CONTINUOUS_RAW_COLS:
        e = np.abs(pred[f] - truth[f])
        row[f] = float(e.mean())
        row[f + "|band"] = {f"{lo}-{hi}": float(e[(val_speed >= lo) & (val_speed < hi)].mean())
                            for lo, hi in SPEED_BANDS}
    for c in (0.4, 0.3, 0.2):  # matched brake metric: clip both sides
        e = np.abs(np.minimum(pred["brake_pedal"], c) - np.minimum(truth["brake_pedal"], c))
        row[f"brake@{c}"] = float(e.mean())
        row[f"brake@{c}|band"] = {f"{lo}-{hi}": float(e[(val_speed >= lo) & (val_speed < hi)].mean())
                                  for lo, hi in SPEED_BANDS}
    lim = np.where(val_speed < 5.0, 0.3, 10.0)
    e = np.abs(np.clip(pred["steering_angle"], -lim, lim) - np.clip(truth["steering_angle"], -lim, lim))
    row["steer@0.3@v<5"] = float(e.mean())
    row["steer@0.3@v<5|band"] = {f"{lo}-{hi}": float(e[(val_speed >= lo) & (val_speed < hi)].mean())
                                 for lo, hi in SPEED_BANDS}
    return row

results = []
for seed in FIT_SEEDS:
    for label, targets in CONFIGS.items():
        fit = _fit(features=features, targets=targets, turn_gt=turn_gt, train_idx=train_idx,
                   val_idx=val_idx, hidden_size=128, steps=STEPS, batch_size=4096, lr=1e-3,
                   log_every=20_000, seed=seed, label=f"{label}/s{seed}", device=DEV)
        results.append({"label": label, "seed": seed} | evaluate(fit.model))
        json.dump(results, open("/tmp/clip_seeds.json", "w"), indent=2)
        print(f"== {label} s{seed} done", flush=True)
print("ALLDONE")
