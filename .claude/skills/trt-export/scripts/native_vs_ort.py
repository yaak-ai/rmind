"""Baseline validation: rmind PyTorch NATIVE vs ONNX-Runtime fp32 on identical inputs.

Every precision result so far was scored against the ONNX fp32 reference. That is only a
valid baseline if the ONNX export is itself faithful to rmind native. This measures that
link directly, on the exact harness inputs (same rng seed 1234, same frame cycling, same
build_inputs), and writes a NATIVE reference cache in parity_matrix.py's npz format so the
TRT engines can afterwards be re-scored against native instead of transitively.

Native side uses PatchPolicy.load_for_export -- the same wrapper the ONNX was traced from
(sample_codes=False -> argmax decode; in-model crop/resize replaced by ImageNet Normalize,
so it consumes the same pre-windowed [0,1] (1,6,3,H,W) frames the ONNX does).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_trt_parity import build_inputs  # noqa: E402


def load_frames(spec):
    """Byte-identical to parity_matrix.load_frames."""
    import cv2
    frames = []
    for fp in [Path(x) for x in spec.split(",") if x.strip()]:
        if not fp.exists():
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h = img.shape[0]
        side = min(h, 320)
        top = max((h - side) // 2, 0)
        frames.append(cv2.cvtColor(img[top:top + side, :], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    return frames


def leaves(obj, path=()):
    """Yield (path, tensor) for every tensor leaf in a nested mapping."""
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif hasattr(obj, "items"):
        for k, v in obj.items():
            yield from leaves(v, (*path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from leaves(v, (*path, i))


def assign(obj, path, value):
    cur = obj
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value


def get_action(out):
    """Pull policy.joint_actions out of whatever container forward returned."""
    try:
        return out["policy", "joint_actions"]
    except Exception:
        pass
    for p, t in leaves(out):
        if any("joint_action" in str(k) for k in p):
            return t
    raise SystemExit(f"could not find joint_actions in output: {type(out)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--image-hw", type=int, required=True)
    ap.add_argument("--repo", default="/home/max/pr265")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--frames",
                    default="/tmp/native_raw.png,/tmp/frame_video1.jpg,/tmp/frame_video0_.jpg")
    ap.add_argument("--code-tol", type=float, default=0.02)
    ap.add_argument("--native-cache", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)

    import onnxruntime as ort
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    print(f"ONNX  : {a.onnx.name}")
    print(f"native: {a.artifact}  (image_hw={a.image_hw})")

    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    action = next(n for n in out_names if "joint_actions" in n)
    specs = [dict(name=i.name, shape=list(i.shape), type=i.type) for i in sess.get_inputs()]
    print(f"  onnx inputs : {[s['name'] for s in specs]}")
    print(f"  onnx outputs: {out_names}  action='{action}'")

    # Build the model AND the batch structure from the very config the ONNX was traced
    # from, so the native call is the same callable with the same input tree.
    with initialize_config_dir(config_dir=f"{a.repo}/config", version_base=None):
        cfg = compose(config_name="export_onnx", overrides=[
            "export=yaak/patch_policy/finetuned",
            f"model.artifact={a.artifact}",
            f"image_hw={a.image_hw}",
        ])
    model = instantiate(cfg.model)
    model = model.eval()
    args = instantiate(cfg.args, _recursive_=True, _convert_="all")
    batch = args[0]
    print(f"  sample_codes={getattr(model, 'sample_codes', '?')} (must be False for argmax)")
    tl = list(leaves(batch))
    print(f"  native batch leaves: {[('/'.join(map(str, p)), tuple(t.shape)) for p, t in tl]}")

    # map ONNX feed -> native leaf by tensor rank (one 5-dim cam, one 3-dim speed, one 4-dim wp)
    by_ndim_native = {}
    for p, t in tl:
        by_ndim_native.setdefault(t.dim(), []).append(p)
    onnx_by_ndim = {}
    for s in specs:
        onnx_by_ndim.setdefault(len(s["shape"]), []).append(s["name"])
    mapping = {}
    for nd, names in onnx_by_ndim.items():
        paths = by_ndim_native.get(nd, [])
        if len(names) != 1 or len(paths) != 1:
            raise SystemExit(f"ambiguous rank-{nd} mapping: onnx={names} native={paths}")
        mapping[names[0]] = paths[0]
    print("  input mapping (onnx -> native):")
    for k, v in mapping.items():
        print(f"    {k}  ->  {'/'.join(map(str, v))}")

    frames = load_frames(a.frames)
    print(f"  {len(frames)} real camera frames" if frames else "  NOISE inputs (not acceptable)")

    rng = np.random.default_rng(1234)   # identical to parity_matrix / verify_trt_parity
    dec = 0
    maxabs = 0.0
    sumabs = 0.0
    per_trial = []
    cache = {}
    t0 = time.time()
    for t in range(a.trials):
        feed = build_inputs(sess, rng, frames[t % len(frames)] if frames else None)

        ort_out = dict(zip(out_names, sess.run(out_names, feed)))

        for oname, npath in mapping.items():
            assign(batch, npath, torch.from_numpy(np.ascontiguousarray(feed[oname])))
        with torch.no_grad():
            nat = get_action(model(batch)).detach().cpu().numpy()

        ref = np.asarray(ort_out[action], dtype=np.float64)
        got = np.asarray(nat, dtype=np.float64)
        d = np.abs(got - ref)
        maxabs = max(maxabs, float(d.max()))
        sumabs += float(d.mean())
        changed = bool((d > a.code_tol).any())
        dec += int(changed)
        per_trial.append((t, float(d.max()), changed))
        if changed:
            print(f"    t{t}: DECISION DIFF max|d|={d.max():.6f}", flush=True)

        # native reference cache, parity_matrix.py npz layout
        for i, n in enumerate(out_names):
            cache[f"t{t}_{i}"] = nat if n == action else np.asarray(ort_out[n])

        if (t + 1) % 20 == 0:
            print(f"    {t+1}/{a.trials}  {time.time()-t0:.0f}s  dec={dec} max|d|={maxabs:.2e}",
                  flush=True)

    cache["meta"] = json.dumps(dict(onnx=a.onnx.name, trials=a.trials, out_names=out_names,
                                    action=action, inputs=specs, reference="rmind-native-fp32",
                                    artifact=a.artifact))
    np.savez_compressed(a.native_cache, **cache)

    print()
    print(f"RESULT  native vs ONNX-fp32, n={a.trials}, tol={a.code_tol}")
    print(f"  decision changes : {dec}/{a.trials}")
    print(f"  max |d|          : {maxabs:.3e}")
    print(f"  mean |d|         : {sumabs / a.trials:.3e}")
    worst = sorted(per_trial, key=lambda x: -x[1])[:5]
    print(f"  worst trials     : {[(t, round(m, 6)) for t, m, _ in worst]}")
    print(f"  native cache     : {a.native_cache}")
    if dec == 0:
        print("  VERDICT: ONNX fp32 is faithful to rmind native at decision level ->"
              " every parity result scored against it holds transitively.")
    else:
        print("  VERDICT: ONNX fp32 DIVERGES from native -> re-score the engines against"
              " the native cache before trusting any fp16 number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
