#!/usr/bin/env python
"""Parity matrix: one ORT fp32 reference pass per trial, compared against MANY engines.

ORT CPU on a 440 MB ViT is the slow part, so running it once per trial and fanning
out to every engine is ~N times faster than calling verify_trt_parity.py per engine.

Judged by DECISION changes on policy.joint_actions (|d| > --code-tol on any channel),
exactly like scripts/verify_trt_parity.py.  If the ONNX exposes ArgMax code indices
as extra outputs, per-ArgMax flip counts are reported too.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_trt_parity import TRTRunner, build_inputs  # noqa: E402


def load_frames(spec):
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
        frames.append(cv2.cvtColor(img[top:top+side, :], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--engines", default="", help="comma-separated engine paths")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--frames",
                    default="/tmp/native_raw.png,/tmp/frame_video1.jpg,/tmp/frame_video0_.jpg")
    ap.add_argument("--code-tol", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ref-only", action="store_true",
                    help="compute + write the ORT reference cache and exit (no engines needed). "
                         "Lets the slow CPU reference run on a big x86 host while the Jetson "
                         "build host stays idle for TRT builds.")
    ap.add_argument("--ref-cache", type=Path, default=None,
                    help="npz of ORT reference outputs; written on first use, reused after "
                         "(ORT CPU on a 440 MB ViT is the slow part). Feeds are regenerated "
                         "deterministically from --trials/--frames/seed, so they need no cache.")
    a = ap.parse_args()

    frames_g = load_frames(a.frames)

    class SpecSess:
        """Stand-in for an ORT session: only build_inputs' .get_inputs() is needed."""
        def __init__(self, specs):
            self._specs = specs
        def get_inputs(self):
            return self._specs

    class _Spec:
        def __init__(self, name, shape, type_):
            self.name, self.shape, self.type = name, shape, type_

    cache_ok = a.ref_cache is not None and a.ref_cache.exists()
    if cache_ok:
        z = np.load(a.ref_cache, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if meta["trials"] < a.trials or meta["onnx"] != a.onnx.name:
            print(f"  [ref-cache stale: {meta['onnx']} trials={meta['trials']}] recomputing")
            cache_ok = False
    if cache_ok:
        out_names = meta["out_names"]
        action = meta["action"]
        codes = [n for n in out_names if n != action]
        sess = SpecSess([_Spec(s["name"], s["shape"], s["type"]) for s in meta["inputs"]])
        refs = [dict(zip(out_names, [z[f"t{t}_{i}"] for i in range(len(out_names))]))
                for t in range(a.trials)]
        print(f"ONNX {a.onnx.name}: using cached ORT reference "
              f"({a.trials} trials) action='{action}' code-probes={codes}")
    else:
        import onnxruntime as ort
        rsess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
        out_names = [o.name for o in rsess.get_outputs()]
        action = next(n for n in out_names if "joint_actions" in n)
        codes = [n for n in out_names if n != action]
        print(f"ONNX {a.onnx.name}: action='{action}' code-probes={codes}")
        print("  running ORT fp32 reference (slow, cached afterwards)...", flush=True)
        rng0 = np.random.default_rng(1234)
        refs = []
        t0r = time.time()
        for t in range(a.trials):
            feed = build_inputs(rsess, rng0, frames_g[t % len(frames_g)] if frames_g else None)
            refs.append(dict(zip(out_names, rsess.run(out_names, feed))))
            if (t + 1) % 10 == 0:
                print(f"    ORT {t+1}/{a.trials}  {time.time()-t0r:.0f}s", flush=True)
        specs = [dict(name=i.name, shape=list(i.shape), type=i.type) for i in rsess.get_inputs()]
        sess = SpecSess([_Spec(s["name"], s["shape"], s["type"]) for s in specs])
        if a.ref_cache is not None:
            d = {f"t{t}_{i}": np.asarray(refs[t][n])
                 for t in range(a.trials) for i, n in enumerate(out_names)}
            d["meta"] = json.dumps(dict(onnx=a.onnx.name, trials=a.trials,
                                        out_names=out_names, action=action, inputs=specs))
            np.savez_compressed(a.ref_cache, **d)
            print(f"  wrote ORT reference cache {a.ref_cache}")
        del rsess

    if a.ref_only:
        print("--ref-only: reference cache written, exiting without engine comparison")
        return 0

    engines = {}
    for p in [x.strip() for x in a.engines.split(",") if x.strip()]:
        pp = Path(p)
        if not pp.exists():
            print(f"  [skip] missing {pp}")
            continue
        try:
            engines[pp.name] = TRTRunner(pp)
            print(f"  loaded {pp.name}")
        except SystemExit as e:
            print(f"  [skip] {pp.name}: {e}")
    if not engines:
        raise SystemExit("no engines loaded")

    frames = frames_g
    print(f"{len(frames)} real camera frames" if frames else "NOISE inputs (not acceptable)")

    stats = {k: dict(dec=0, maxabs=0.0, sumabs=0.0, n=0,
                     flips={c: 0 for c in codes}) for k in engines}
    rng = np.random.default_rng(1234)   # same seed as verify_trt_parity.py
    t0 = time.time()
    for t in range(a.trials):
        feed = build_inputs(sess, rng, frames[t % len(frames)] if frames else None)
        ref = refs[t]
        for name, run in engines.items():
            got = run(feed)
            g = got.get(action)
            if g is None:
                g = next(iter(got.values()))
            d = np.abs(np.asarray(g).astype(np.float64)
                       - np.asarray(ref[action]).astype(np.float64))
            s = stats[name]
            s["maxabs"] = max(s["maxabs"], float(d.max()))
            s["sumabs"] += float(d.mean())
            s["n"] += 1
            changed = bool((d > a.code_tol).any())
            s["dec"] += int(changed)
            det = []
            for c in codes:
                if c not in got:
                    continue
                r = np.asarray(ref[c]).astype(np.int64).ravel()
                e = np.asarray(got[c]).astype(np.int64).ravel()[:r.size]
                nd = int((r != e).sum())
                if nd:
                    s["flips"][c] += 1
                    det.append(f"{c}={nd}/{r.size}(ref{r.tolist()}!=trt{e.tolist()})")
            if changed or det:
                print(f"  t{t:02d} {name}: dec={changed} max|d|={d.max():.4f} "
                      f"{' '.join(det)}")
        if (t + 1) % 10 == 0:
            print(f"  ... {t+1}/{a.trials} trials, {time.time()-t0:.0f}s", flush=True)

    print(f"\n{'engine':46s} {'decisions':>12s} {'max|d|':>10s} {'mean|d|':>10s}  verdict")
    results = {}
    for name, s in stats.items():
        frac = s["dec"] / max(s["n"], 1)
        ok = s["dec"] == 0 and s["maxabs"] < 0.05
        print(f"{name:46s} {s['dec']:5d}/{s['n']:<6d} {s['maxabs']:10.6f} "
              f"{s['sumabs']/max(s['n'],1):10.6f}  {'PASS' if ok else 'FAIL'}")
        results[name] = dict(decision_changes=s["dec"], trials=s["n"],
                             max_abs=s["maxabs"], mean_abs=s["sumabs"]/max(s["n"],1),
                             pass_=ok, flips=s["flips"])
    if codes:
        print(f"\nper-ArgMax flip counts (trials with >=1 differing code index):")
        hdr = "  " + f"{'engine':46s}" + "".join(f"{c:>12s}" for c in codes)
        print(hdr)
        for name, s in stats.items():
            print("  " + f"{name:46s}" + "".join(f"{s['flips'][c]:>12d}" for c in codes))
    if a.out:
        a.out.write_text(json.dumps(dict(onnx=str(a.onnx), trials=a.trials,
                                         code_tol=a.code_tol, results=results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
