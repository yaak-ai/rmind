#!/usr/bin/env python
"""Decision-margin analysis: how close is each ArgMax to flipping, in fp16 units?

For every ArgMax in the graph we expose its input tensor and measure
    margin = top1 - top2
and compare it to the fp16 representation spacing at that magnitude.  A margin
below ~1 fp16 ULP means the decision is not resolvable in fp16 at all; a margin
of a few ULPs means accumulated fp16 error can flip it.

Run on a FAILING and a PASSING checkpoint of the same architecture to test
whether checkpoint-dependent fp16 failures are explained by margin geometry.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_probe(src: Path, dst: Path):
    m = onnx.load(str(src))
    g = m.graph
    have = {o.name for o in g.output}
    probes = []
    for n in g.node:
        if n.op_type != "ArgMax":
            continue
        t = n.input[0]
        if t not in have:
            g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
            probes.append((n.name, t))
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
              location=dst.name + ".w", size_threshold=1024)
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="comma-separated models to compare")
    ap.add_argument("--labels", default=None, help="comma-separated labels")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--frames",
                    default="/tmp/native_raw.png,/tmp/frame_video1.jpg,/tmp/frame_video0_.jpg")
    ap.add_argument("--workdir", default="/tmp/marginprobe")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import cv2, onnxruntime as ort
    from verify_trt_parity import build_inputs

    frames = []
    for fp in [Path(x) for x in a.frames.split(",") if x.strip()]:
        img = cv2.imread(str(fp)) if fp.exists() else None
        if img is None:
            continue
        h = img.shape[0]
        side = min(h, 320)
        top = max((h - side) // 2, 0)
        frames.append(cv2.cvtColor(img[top:top+side, :], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    print(f"{len(frames)} real frames")

    models = [Path(x) for x in a.onnx.split(",") if x.strip()]
    labels = (a.labels.split(",") if a.labels else [m.stem[-24:] for m in models])
    allres = {}
    for mp, lab in zip(models, labels):
        dst = Path(a.workdir) / (lab.replace("/", "_") + ".probe.onnx")
        probes = make_probe(mp, dst)
        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        names = [t for _, t in probes]
        print(f"\n##### {lab}   probes={[(n,t) for n,t in probes]}", flush=True)
        rng = np.random.default_rng(1234)
        acc = {t: [] for t in names}
        for i in range(a.trials):
            feed = build_inputs(sess, rng, frames[i % len(frames)] if frames else None)
            outs = sess.run(names, feed)
            for t, arr in zip(names, outs):
                A = np.asarray(arr, dtype=np.float64)
                A = A.reshape(-1, A.shape[-1])          # (groups, candidates)
                srt = np.sort(A, axis=1)
                top1, top2 = srt[:, -1], srt[:, -2]
                margin = top1 - top2
                ulp = np.spacing(np.abs(top1).astype(np.float16)).astype(np.float64)
                ulp = np.where(ulp == 0, np.spacing(np.float16(1e-4)), ulp)
                acc[t].append(np.stack([margin, np.abs(top1), margin / ulp], 1))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.trials}", flush=True)
        res = {}
        for t in names:
            M = np.concatenate(acc[t], 0)          # (n, 3): margin, |top1|, margin/ulp
            r = dict(n=int(M.shape[0]),
                     margin_median=float(np.median(M[:, 0])),
                     margin_p05=float(np.percentile(M[:, 0], 5)),
                     margin_min=float(M[:, 0].min()),
                     top1_median=float(np.median(M[:, 1])),
                     ulps_median=float(np.median(M[:, 2])),
                     ulps_p05=float(np.percentile(M[:, 2], 5)),
                     ulps_min=float(M[:, 2].min()),
                     frac_under_1ulp=float((M[:, 2] < 1).mean()),
                     frac_under_4ulp=float((M[:, 2] < 4).mean()),
                     frac_under_16ulp=float((M[:, 2] < 16).mean()))
            res[t] = r
            print(f"  {t:12s} n={r['n']:4d} |top1|~{r['top1_median']:8.3f} "
                  f"margin med={r['margin_median']:.5f} p05={r['margin_p05']:.5f} "
                  f"min={r['margin_min']:.6f} | fp16 ULPs: med={r['ulps_median']:8.1f} "
                  f"p05={r['ulps_p05']:7.2f} min={r['ulps_min']:6.3f} "
                  f"| <1ULP {100*r['frac_under_1ulp']:.1f}% <4ULP {100*r['frac_under_4ulp']:.1f}% "
                  f"<16ULP {100*r['frac_under_16ulp']:.1f}%")
        allres[lab] = res
    if a.out:
        Path(a.out).write_text(json.dumps(allres, indent=2))


if __name__ == "__main__":
    main()
