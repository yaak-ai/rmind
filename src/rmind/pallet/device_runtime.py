"""On-device pallet perception: boxes + loaded/empty, 12 entry-face keypoints, 3D pose.

Self-contained: needs numpy and OpenCV, plus either TensorRT engines (Jetson) or
onnxruntime; the floor-free pose uses the bundled `rmind.pallet.pose_free` (numpy / cv2).
Bundle layout written by `scripts.pallet_export_device`:

    detector.onnx    DINOv3 ViT-S encoder + box head, RGB [0, 1] 1x3x288x512 -> heat, size,
                     offset, p_loaded (sigmoids applied)
    keypoints.onnx   HRNet-W18, RGB [0, 1] Bx3xHxW crops -> 12 heatmaps at stride 4
    pipeline.json    resolutions, crop shape, thresholds, keypoint names, pallet layout
    rmind/pallet/    geometry.py, pose.py, pose_free.py (camera model + PnP)

Per frame: resize to the detector input, decode boxes (heatmap peaks); for the widest
boxes (`min_kp_box_px`, at most `max_kp`) crop box + margin, run the keypoint net, decode
peaks back to frame pixels, and solve the pallet pose (6-DoF PnP refined upright: level
pallet, height free, so stacks and racks work). Positions are in the camera frame
(forward, left, up in metres), yaw in degrees.

    python device_runtime.py --bundle . --video in.mp4 --out out.mp4 [--backend trt|onnx]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LOOPS = (
    (
        "pallet_bottom_left",
        "pallet_bottom_right",
        "pallet_top_right",
        "pallet_top_left",
    ),
    (
        "pocket_l_bottom_left",
        "pocket_l_bottom_right",
        "pocket_l_top_right",
        "pocket_l_top_left",
    ),
    (
        "pocket_r_bottom_left",
        "pocket_r_bottom_right",
        "pocket_r_top_right",
        "pocket_r_top_left",
    ),
)


class OnnxModel:
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort  # ruff: ignore[import-outside-top-level]

        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input: x})


class TrtModel:
    """A TensorRT engine; buffers as torch CUDA tensors (torch is on the Jetson image)."""

    def __init__(self, path: Path) -> None:
        import tensorrt as trt  # ruff: ignore[import-outside-top-level]
        import torch  # ruff: ignore[import-outside-top-level]

        self.torch = torch
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.inputs = [
            n
            for n in self.names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.buffers = {
            n: torch.empty(
                tuple(self.engine.get_tensor_shape(n)),
                dtype=torch.float32,
                device="cuda",
            )
            for n in self.names
        }
        for n, b in self.buffers.items():
            self.context.set_tensor_address(n, b.data_ptr())
        self.stream = torch.cuda.Stream()

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        buf = self.buffers[self.inputs[0]]
        n = x.shape[0]
        buf[:n].copy_(self.torch.from_numpy(x))
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return [self.buffers[o].cpu().numpy() for o in self.outputs]


@dataclass
class Pallet:
    box: list[float]
    score: float
    p_loaded: float
    keypoints: np.ndarray | None = None  # (12, 2) frame px
    confidence: np.ndarray | None = None  # (12,)
    pose: dict | None = None
    extra: dict = field(default_factory=dict)


def decode_boxes(
    heat, size, offset, load, threshold: float, k: int, width: int
) -> list[Pallet]:
    h = heat[0, 0]
    _gh, gw = h.shape
    cell = width / gw
    peaks = h * (cv2.dilate(h, np.ones((3, 3), np.float32)) == h)
    flat = peaks.ravel()
    order = np.argsort(-flat)[:k]
    out = []
    for f in order:
        s = float(flat[f])
        if s < threshold:
            break
        i, j = divmod(int(f), gw)
        cx, cy = j + offset[0, 0, i, j], i + offset[0, 1, i, j]
        w, hh = np.exp(size[0, 0, i, j]), np.exp(size[0, 1, i, j])
        box = [
            (cx - w / 2) * cell,
            (cy - hh / 2) * cell,
            (cx + w / 2) * cell,
            (cy + hh / 2) * cell,
        ]
        out.append(
            Pallet(
                [float(v) for v in box],
                s,
                float(load[0, 0, i, j]) if load is not None else float("nan"),
            )
        )
    return out


def rect_crop(
    box: list[float], margin: float, max_aspect: float
) -> tuple[float, float, float, float]:
    """Same as `rmind.pallet.keypoints.RectCrop.around`."""
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    w, h = max(w, h / max_aspect, 1.0), max(h, w / max_aspect, 1.0)
    return (x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - h / 2, w, h


def extract(
    rgb: np.ndarray, crop: tuple[float, float, float, float], shape: tuple[int, int]
) -> np.ndarray:
    """Bilinear crop resampled at pixel centres, zero outside the frame (as grid_sample)."""
    x0, y0, w, h = crop
    mx = ((np.arange(shape[0], dtype=np.float32) + 0.5) * (w / shape[0]) + x0)[
        None, :
    ].repeat(shape[1], 0)
    my = ((np.arange(shape[1], dtype=np.float32) + 0.5) * (h / shape[1]) + y0)[
        :, None
    ].repeat(shape[0], 1)
    return cv2.remap(
        rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def decode_keypoints(heat: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """`(B, K, H, W)` -> crop pixels `(B, K, 2)` and peak value; parabola refinement on log."""
    b, k, h, w = heat.shape
    flat = heat.reshape(b, k, -1)
    idx = flat.argmax(-1)
    conf = np.take_along_axis(flat, idx[..., None], -1)[..., 0]
    log = np.log(np.clip(flat, 1e-6, None))
    py, px = idx // w, idx % w

    def offset(i, step, n):
        c = np.take_along_axis(log, idx[..., None], -1)[..., 0]
        lo = np.take_along_axis(log, (idx - step * (i > 0))[..., None], -1)[..., 0]
        hi = np.take_along_axis(log, (idx + step * (i < n - 1))[..., None], -1)[..., 0]
        curv = lo - 2 * c + hi
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(curv < -1e-6, 0.5 * (lo - hi) / curv, 0.0)
        return np.clip(d, -0.5, 0.5)

    x = px + offset(px, 1, w)
    y = py + offset(py, w, h)
    return (np.stack([x, y], -1) + 0.5) * stride, conf


class Pipeline:
    def __init__(self, bundle: Path, backend: str = "onnx") -> None:
        self.cfg = json.loads((bundle / "pipeline.json").read_text())
        make = (
            (lambda n: TrtModel(bundle / f"{n}.engine"))
            if backend == "trt"
            else (lambda n: OnnxModel(bundle / f"{n}.onnx"))
        )
        self.detector = make("detector")
        self.keypoints = make("keypoints")
        self.names = self.cfg["keypoint_names"]
        self.layout = np.array(self.cfg["layout_m"], dtype=float)
        sys.path.insert(0, str(bundle))
        try:
            from rmind.pallet.geometry import CAM_FORK
            from rmind.pallet.pose_free import solve_6dof, solve_upright  # ruff: ignore[import-outside-top-level]

            self.pose = (CAM_FORK, solve_6dof, solve_upright)
        except ImportError:  # no pose without the bundled camera model
            self.pose = None

    def __call__(self, bgr: np.ndarray) -> list[Pallet]:
        c = self.cfg
        _h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        dw, dh = c["detector_input"]
        x = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[
            None
        ]
        heat, size, offset, load = self.detector(np.ascontiguousarray(x))
        pallets = decode_boxes(
            heat, size, offset, load, c["box_threshold"], c["max_boxes"], w
        )
        near = sorted(
            (p for p in pallets if p.box[2] - p.box[0] >= c["min_kp_box_px"]),
            key=lambda p: p.box[0] - p.box[2],
        )
        near = near[: c["max_kp"]]
        shape = tuple(c["crop_shape"])
        for i in range(0, len(near), c["kp_batch"]):
            chunk = near[i : i + c["kp_batch"]]
            crops = [
                rect_crop(p.box, c["crop_margin"], c["crop_max_aspect"]) for p in chunk
            ]
            batch = np.zeros((c["kp_batch"], 3, shape[1], shape[0]), np.float32)
            for j, cr in enumerate(crops):
                batch[j] = extract(rgb, cr, shape).transpose(2, 0, 1)
            (hm,) = self.keypoints(batch)
            px, conf = decode_keypoints(hm[: len(chunk)], c["stride"])
            for p, cr, q, cf in zip(chunk, crops, px, conf, strict=True):
                p.keypoints = q * np.array([
                    cr[2] / shape[0],
                    cr[3] / shape[1],
                ]) + np.array(cr[:2])
                p.confidence = cf
                p.pose = self.solve(p.keypoints, cf > c["kp_confidence"])
        return pallets

    def solve(self, pixels: np.ndarray, visible: np.ndarray) -> dict | None:
        if self.pose is None or visible.sum() < 4:
            return None
        cam, solve_6dof, solve_upright = self.pose
        s6 = solve_6dof(pixels, visible, self.layout, camera=cam)
        s = (
            solve_upright(pixels, visible, self.layout, s6, camera=cam)
            if s6 is not None
            else None
        )
        s = s or s6
        if s is None:
            return None
        if s.rms_px > self.cfg.get(
            "max_pose_rms_px", 8.0
        ):  # keypoints do not fit one pallet
            return None
        f, l, u = (float(v) for v in s.centre)
        return {
            "forward_m": f,
            "left_m": l,
            "up_m": u,
            "yaw_deg": float(np.degrees(s.yaw)),
            "rms_px": float(s.rms_px),
        }


def colour(score: float, threshold: float) -> tuple[int, int, int]:
    t = min(max((score - threshold) / max(1 - threshold, 1e-6), 0.0), 1.0)
    return (0, int(510 * t), 255) if t < 0.5 else (0, 255, int(255 * (2 - 2 * t)))  # ruff: ignore[magic-value-comparison]


def draw(
    bgr: np.ndarray, pallets: list[Pallet], names: list[str], threshold: float
) -> np.ndarray:
    v = bgr.copy()
    for p in pallets:
        b = [int(x) for x in p.box]
        c = colour(p.score, threshold)
        cv2.rectangle(v, b[:2], b[2:], c, 3)
        tag = f"{'LOADED' if p.p_loaded >= 0.5 else 'empty'} {p.score:.2f}"  # ruff: ignore[magic-value-comparison]
        if p.pose:
            tag += f"  {p.pose['forward_m']:.1f}m {p.pose['yaw_deg']:+.0f}deg h{p.pose['up_m']:+.2f}"
        cv2.putText(v, tag, (b[0], max(20, b[1] - 8)), 0, 0.7, (0, 0, 0), 4)
        cv2.putText(v, tag, (b[0], max(20, b[1] - 8)), 0, 0.7, c, 2)
        if p.keypoints is not None:
            ok = p.confidence > 0.3  # ruff: ignore[magic-value-comparison]
            for loop in LOOPS:
                idx = [names.index(n) for n in loop]
                for a, d in zip(idx, idx[1:] + idx[:1], strict=True):
                    if ok[a] and ok[d]:
                        cv2.line(
                            v,
                            tuple(int(t) for t in p.keypoints[a]),
                            tuple(int(t) for t in p.keypoints[d]),
                            (255, 0, 255),
                            2,
                        )
            for q, o in zip(p.keypoints, ok, strict=True):
                if o:
                    cv2.circle(v, (int(q[0]), int(q[1])), 4, (255, 0, 255), -1)
    return v


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, default=Path(__file__).parent)
    parser.add_argument("--backend", choices=["onnx", "trt"], default="onnx")
    parser.add_argument("--video", type=Path, help="input video")
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=[],
        help="input images instead of a video",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, required=True, help="annotated .mp4 (video / images)"
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--json", type=Path, default=None, help="per-frame detections")
    args = parser.parse_args()
    pipe = Pipeline(args.bundle, args.backend)
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

        def frames():
            for i in range(args.frames * args.stride):
                ok, f = cap.read()
                if not ok:
                    return
                if i % args.stride == 0:
                    yield f"{args.start + i}", f

    else:

        def frames():
            for p in args.images:
                yield p.name, cv2.imread(str(p))

    writer, log, times = None, [], []
    for name, frame in frames():
        t = time.perf_counter()
        pallets = pipe(frame)
        times.append(time.perf_counter() - t)
        view = cv2.resize(
            draw(frame, pallets, pipe.names, pipe.cfg["box_threshold"]), (1280, 720)
        )
        if writer is None:
            writer = cv2.VideoWriter(
                str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720)
            )
        writer.write(view)
        log.append({
            "frame": name,
            "pallets": [
                {
                    "box": p.box,
                    "score": p.score,
                    "p_loaded": p.p_loaded,
                    "keypoints": None if p.keypoints is None else p.keypoints.tolist(),
                    "pose": p.pose,
                }
                for p in pallets
            ],
        })
    if writer is not None:
        writer.release()
    if args.json:
        args.json.write_text(json.dumps(log))


if __name__ == "__main__":
    main()
