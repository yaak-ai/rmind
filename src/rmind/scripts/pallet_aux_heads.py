"""Early check: can light heads on the policy's FROZEN DINOv3 learn what the teachers do?

Encoder: `vit_small_patch16_dinov3.lvd1689m` (pre-norm intermediates, `TimmBackbone`
`out_indices`), at `--res` (256x144 = the policy input, 512x288 = a 2x pass). `--k 1`
reads layer 10 (the policy's layer); `--k 4` reads layers 8-11, each through its own
1x1 conv, summed. Recomputed every batch, so crops / flips / colour jitter augment.

Heads, trained jointly from scratch (AdamW, one-cycle):

    det    CenterNet of `pallet_proposer_multi.Head` at 2x the token grid: centre heat
           (focal), log box size + sub-cell offset (L1), loaded/empty logit (BCE).
           Real `train` frames only.
    kp     a square RoI (the box + 15 % margin, as the teacher's crop) pooled from
           the fused token grid (`roi_align`, 16x16) -> convs -> 32x32 logits per
           entry-face keypoint -> soft-argmax; L1 in RoI units on VISIBLE keypoints.
           Real `train` keypoints + synthetic `train` + real pseudo-labels (not the
           test day). Trained on jittered GT boxes, scored at the GT box.
    pose   the same RoI trunk, mean+max pooled, with the box geometry in the frame
           -> MLP -> entry-face centre (forward, left, up; m) + yaw / 30 deg; L1.
           Synthetic `train` only (no geometric augmentation there: a crop would
           change the effective intrinsics the pose depends on).

Scores: detection (`pallet_proposer_multi.score`) on real `test` and real `val`;
keypoint error in full-res px on real `val` (teacher: ~2.4 px median); pose error on
synthetic `val`, by GT forward range. The synthetic camera is pinhole with borrowed
intrinsics, real `cam_fork` is fisheye: synthetic pose numbers do not transfer as is.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from structlog import get_logger
from torch import Tensor, nn
from torchvision.ops import roi_align

from rmind.pallet.schema import KEYPOINTS
from rmind.pallet.tracking import start
from rmind.scripts import pallet_proposer_multi as multi
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER, MEAN, STD, focal_loss

logger = get_logger(__name__)
WIDTH, HEIGHT = 1920, 1080
STORE_W, STORE_H = 1024, 576  # frames kept on the GPU at this size (uint8)
LABELS = Path("/nasa/team-space/nikita/pallet-detector/labels/v5")
REAL = LABELS / "real_v5.jsonl"
SYNTH = LABELS / "synthetic_pallet-niantic-20261028-warehouse-labels-v2.jsonl"
PSEUDO = Path("/nasa/team-space/nikita/pallet-detector-tresor/pseudo_labels_v1.json")
TEST_DAY = "2026-10-12"
LAYERS = {1: [LAYER], 4: [8, 9, 10, 11]}
NK = len(KEYPOINTS)
ROI = 16  # RoI grid pooled from the token map
MARGIN = 0.15
YAW_SCALE = 30.0
RANGES = ((0.0, 2.0), (2.0, 4.0), (4.0, 99.0))


def _mirror(name: str) -> str:
    swap = {"left": "right", "right": "left", "l": "r", "r": "l"}
    return "_".join(swap.get(part, part) for part in name.split("_"))


FLIP = [KEYPOINTS.index(_mirror(n)) for n in KEYPOINTS]


# --- data ----------------------------------------------------------------------------


def _pallet(p: dict) -> dict:
    kp, kv = np.zeros((NK, 2), np.float32), np.zeros(NK, bool)
    for i, n in enumerate(KEYPOINTS):
        v = p.get("keypoints", {}).get(n)
        if v is not None and (len(v) < 3 or v[2] == "visible"):
            kp[i], kv[i] = v[:2], True
    pose = p.get("pose")
    return {
        "box": p.get("box"),
        "unsure": bool(p.get("unsure")),
        "load": {True: "loaded", False: "empty"}.get(p.get("loaded")),
        "kp": kp,
        "kv": kv,
        "pose": None
        if pose is None
        else np.array([*pose["entry_centre_m"], pose["yaw_deg"]], np.float32),
    }


def load_sets() -> dict[str, dict[str, list[dict]]]:
    """{set: {image path: [pallet]}} for real train/val/test, synthetic train/val, pseudo."""
    sets: dict[str, dict[str, list[dict]]] = {}
    for prefix, path in (("real", REAL), ("synth", SYNTH)):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            sets.setdefault(f"{prefix}_{r['split']}", {})[r["image"]] = [
                _pallet(p) for p in r["pallets"]
            ]
    kps = json.loads(PSEUDO.read_text(encoding="utf-8"))
    boxes = json.loads(PSEUDO.with_suffix(".boxes.json").read_text())
    pseudo: dict[str, list[dict]] = {}
    for key, points in kps.items():
        image = key.split("#")[0]
        if Path(image).stem.startswith(TEST_DAY):  # keep the detector's test day unseen
            continue
        pseudo.setdefault(image, []).append(
            _pallet({
                "box": boxes[key],
                "keypoints": {n: [*uv, "visible"] for n, uv in points.items()},
            })
        )
    sets["pseudo"] = pseudo
    return sets


def _read(path: str) -> Tensor:
    from PIL import Image

    im = (
        Image
        .open(path)
        .convert("RGB")
        .resize((STORE_W, STORE_H), Image.Resampling.BILINEAR)
    )
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).contiguous()


class Store:
    """Every frame at 1024x576 uint8 on the device; batches at the encoder resolution."""

    def __init__(self, paths: list[str], device: str, res: tuple[int, int]) -> None:
        with ThreadPoolExecutor(16) as pool:
            frames = list(pool.map(_read, paths))
        self.index = {p: i for i, p in enumerate(paths)}
        self.frames = torch.stack(frames).to(device)
        self.res, self.device = res, device
        self.mean, self.std = MEAN.to(device), STD.to(device)

    def batch(
        self,
        names: list[str],
        labels: dict[str, list[dict]],
        rng: np.random.Generator | None,
        *,
        geometric: bool = True,
    ) -> tuple[Tensor, list[list[dict]]]:
        xs, ls = [], []
        for n in names:
            img = self.frames[self.index[n]].float() / 255
            pallets = [dict(p) for p in labels.get(n, [])]
            if rng is not None:
                if geometric:
                    img, pallets = _crop_flip(img, pallets, rng)
                img = (img * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)).clamp(0, 1)
            x = F.interpolate(
                img[None],
                size=(self.res[1], self.res[0]),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )[0]
            xs.append((x - self.mean) / self.std)
            ls.append(pallets)
        return torch.stack(xs), ls


def _crop_flip(
    img: Tensor, pallets: list[dict], rng: np.random.Generator
) -> tuple[Tensor, list[dict]]:
    s = rng.uniform(0.75, 1.0)
    cw, ch = WIDTH * s, HEIGHT * s
    x0, y0 = rng.uniform(0, WIDTH - cw), rng.uniform(0, HEIGHT - ch)
    k = STORE_W / WIDTH
    img = img[:, int(y0 * k) : int((y0 + ch) * k), int(x0 * k) : int((x0 + cw) * k)]
    sx, sy = WIDTH / cw, HEIGHT / ch
    flip = rng.random() < 0.5
    if flip:
        img = img.flip(-1)
    out = []
    for p in pallets:
        if p["box"] is None:
            continue
        b = p["box"]
        b = [(b[0] - x0) * sx, (b[1] - y0) * sy, (b[2] - x0) * sx, (b[3] - y0) * sy]
        if b[2] <= 0 or b[0] >= WIDTH or b[3] <= 0 or b[1] >= HEIGHT:
            continue
        b = [max(b[0], 0), max(b[1], 0), min(b[2], WIDTH), min(b[3], HEIGHT)]
        kp = (p["kp"] - np.array([x0, y0], np.float32)) * np.array([sx, sy], np.float32)
        kv = (
            p["kv"]
            & (kp[:, 0] >= 0)
            & (kp[:, 0] < WIDTH)
            & (kp[:, 1] >= 0)
            & (kp[:, 1] < HEIGHT)
        )
        if flip:
            b = [WIDTH - b[2], b[1], WIDTH - b[0], b[3]]
            kp = kp[FLIP] * np.array([-1, 1], np.float32) + np.array(
                [WIDTH, 0], np.float32
            )
            kv = kv[FLIP]
        out.append(p | {"box": b, "kp": kp, "kv": kv})
    return img, out


# --- model ---------------------------------------------------------------------------


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(16, cout),
        nn.GELU(),
    )


class AuxHeads(nn.Module):
    def __init__(self, k: int, dim: int = 384) -> None:
        super().__init__()
        self.proj = (
            nn.ModuleList(nn.Conv2d(dim, dim, 1) for _ in range(k)) if k > 1 else None
        )
        self.det = multi.Head(dim)
        self.roi = nn.Sequential(
            nn.Conv2d(dim, 256, 1), _block(256, 256), _block(256, 256)
        )
        self.kp = nn.Sequential(_block(256, 128), nn.Conv2d(128, NK, 3, padding=1))
        self.pose = nn.Sequential(
            nn.Linear(512 + 6, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 4),
        )

    def fuse(self, feats: list[Tensor]) -> Tensor:
        if self.proj is None:
            return feats[0]
        return sum(p(f) for p, f in zip(self.proj, feats, strict=True))  # ty: ignore[invalid-return-type]

    def rois(self, fmap: Tensor, rois: Tensor, scale: float) -> tuple[Tensor, Tensor]:
        """`(N, 5)` [batch index, x0, y0, side, side] full-res RoIs -> keypoints (N, K, 2)
        in RoI units [0, 1] and the RoI trunk's pooled descriptor (N, 512)."""
        boxes = torch.cat(
            [rois[:, :3], rois[:, 1:3] + rois[:, 3:5]], dim=1
        )  # [b, x0, y0, x1, y1]
        x = roi_align(
            fmap, boxes, ROI, spatial_scale=scale, sampling_ratio=2, aligned=True
        )
        x = self.roi(x)
        y = self.kp(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        )
        n, k, h, w = y.shape
        prob = y.flatten(2).softmax(-1).view(n, k, h, w)
        gx = (torch.arange(w, device=y.device) + 0.5) / w
        gy = (torch.arange(h, device=y.device) + 0.5) / h
        uv = torch.stack(
            [(prob.sum(2) * gx).sum(-1), (prob.sum(3) * gy).sum(-1)], dim=-1
        )
        return uv, torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)


def square_rois(boxes: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    """(N, 4) boxes -> (N, 3) square RoIs [x0, y0, side], jittered when training."""
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    side = np.maximum(w, h) * (1 + 2 * MARGIN)
    if rng is not None:  # a detector's box is off by a few % in centre and scale
        cx += rng.normal(0, 0.05, len(cx)) * w
        cy += rng.normal(0, 0.05, len(cy)) * np.maximum(h, 0.3 * w)
        side *= np.exp(rng.normal(0, 0.1, len(side)))
    return np.stack([cx - side / 2, cy - side / 2, side], axis=1)


def geometry(boxes: np.ndarray) -> np.ndarray:
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    return np.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / 2 / WIDTH,
            (boxes[:, 1] + boxes[:, 3]) / 2 / HEIGHT,
            w / WIDTH,
            h / HEIGHT,
            np.log(np.maximum(w, 1) / WIDTH),
            np.log(np.maximum(h, 1) / HEIGHT),
        ],
        axis=1,
    ).astype(np.float32)


def roi_inputs(
    pallets: list[list[dict]], rng: np.random.Generator | None, offset: int = 0
) -> dict[str, np.ndarray]:
    """Every boxed pallet with a visible keypoint or a pose -> RoI tensors (numpy)."""
    rows = [
        (b + offset, p)
        for b, ps in enumerate(pallets)
        for p in ps
        if p["box"] is not None and (p["kv"].any() or p["pose"] is not None)
    ]
    if not rows:
        return {}
    boxes = np.array([p["box"] for _, p in rows], np.float32)
    sq = square_rois(boxes, rng).astype(np.float32)
    kp = np.stack([p["kp"] for _, p in rows])
    uv = (kp - sq[:, None, :2]) / sq[:, None, 2:]
    kv = np.stack([p["kv"] for _, p in rows]) & (uv >= 0).all(-1) & (uv <= 1).all(-1)
    has_pose = np.array([p["pose"] is not None for _, p in rows])
    pose = np.stack([
        p["pose"] if p["pose"] is not None else np.zeros(4, np.float32) for _, p in rows
    ])
    return {
        "rois": np.concatenate(
            [np.array([[b] for b, _ in rows], np.float32), sq, sq[:, 2:]], axis=1
        ),
        "sq": sq,
        "uv": uv.astype(np.float32),
        "kv": kv,
        "geom": geometry(boxes),
        "pose": pose,
        "has_pose": has_pose,
    }


def pose_out(model: AuxHeads, desc: Tensor, geom: Tensor) -> Tensor:
    return model.pose(torch.cat([desc, geom], dim=1))


# --- train / eval --------------------------------------------------------------------


class Runner:
    def __init__(self, store: Store, sets: dict, k: int, device: str) -> None:
        from rmind.components.timm_backbone import TimmBackbone

        self.store, self.sets, self.k, self.device = store, sets, k, device
        res = store.res
        self.enc = (
            TimmBackbone(ENCODER, out_indices=LAYERS[k], img_size=[res[1], res[0]])
            .to(device)
            .eval()
        )
        self.gh, self.gw = res[1] // 16 * 2, res[0] // 16 * 2
        self.scale = res[0] / WIDTH / 16  # full-res px -> token grid
        self.model = AuxHeads(k).to(device)

    def features(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            feats = self.enc.model(x.to(self.device))
        return self.model.fuse(feats)

    def train(self, steps: int, batch: int, lr: float, log) -> None:
        model, dev = self.model, self.device
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=steps, pct_start=0.05
        )
        rng = np.random.default_rng(0)
        real, synth, pseudo = (
            self.sets["real_train"],
            self.sets["synth_train"],
            self.sets["pseudo"],
        )
        names = {
            s: sorted(d)
            for s, d in (("real", real), ("synth", synth), ("pseudo", pseudo))
        }
        for step in range(steps):
            model.train()
            xr, lr_ = self.store.batch(
                list(rng.choice(names["real"], batch)), real, rng
            )
            xs, ls = self.store.batch(
                list(rng.choice(names["synth"], batch)), synth, rng, geometric=False
            )
            xp, lp = self.store.batch(
                list(rng.choice(names["pseudo"], batch)), pseudo, rng
            )
            fmap = self.features(torch.cat([xr, xs, xp]))
            # detection + loaded/empty: real frames only
            t = [multi.targets(p, self.gh, self.gw) for p in lr_]
            t = {key: torch.stack([v[key] for v in t]).to(dev) for key in t[0]}
            heat, size, offset, load = model.det(fmap[:batch])
            m = t["mask"][:, None].expand_as(size)
            norm = m.sum().clamp(min=1)
            loss_det = (
                focal_loss(heat, t["heat"], t["ignore"])
                + 0.2 * F.l1_loss(size[m], t["size"][m], reduction="sum") / norm
                + 2 * F.l1_loss(offset[m], t["offset"][m], reduction="sum") / norm
            )
            lm = t["load_mask"]
            loss_load = (
                F.binary_cross_entropy_with_logits(load[lm], t["load"][lm])
                if lm.any()
                else heat.sum() * 0
            )
            r = roi_inputs([*lr_, *ls, *lp], rng)
            loss_kp = loss_pose = torch.zeros((), device=dev)
            if r:
                uv, desc = model.rois(
                    fmap, torch.from_numpy(r["rois"]).to(dev), self.scale
                )
                kv = torch.from_numpy(r["kv"]).to(dev)
                if kv.any():
                    loss_kp = F.l1_loss(uv[kv], torch.from_numpy(r["uv"]).to(dev)[kv])
                hp = torch.from_numpy(r["has_pose"]).to(dev)
                if hp.any():
                    pred = pose_out(
                        model, desc[hp], torch.from_numpy(r["geom"]).to(dev)[hp]
                    )
                    gt = torch.from_numpy(r["pose"]).to(dev)[hp]
                    gt = torch.cat([gt[:, :3], gt[:, 3:] / YAW_SCALE], dim=1)
                    loss_pose = F.smooth_l1_loss(pred, gt, beta=0.05)
            loss = loss_det + 0.5 * loss_load + 20 * loss_kp + loss_pose
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            log(
                step,
                {
                    "loss": loss.item(),
                    "det": loss_det.item(),
                    "load": loss_load.item(),
                    "kp": loss_kp.item(),
                    "pose": loss_pose.item(),
                },
            )

    @torch.no_grad()
    def predict(
        self, labels: dict[str, list[dict]], chunk: int = 16
    ) -> dict[str, dict]:
        """Per image: detected boxes (thresholds applied later) and, at each GT box,
        keypoints (full-res px) and pose."""
        self.model.eval()
        out = {}
        names = sorted(labels)
        for i in range(0, len(names), chunk):
            part = names[i : i + chunk]
            x, ls = self.store.batch(part, labels, None)
            fmap = self.features(x)
            dets = multi.decode(self.model.det(fmap), threshold=0.05)
            r = roi_inputs(ls, None)
            per = {
                n: {"boxes": d, "kp": [], "pose": []}
                for n, d in zip(part, dets, strict=True)
            }
            if r:
                uv, desc = self.model.rois(
                    fmap, torch.from_numpy(r["rois"]).to(self.device), self.scale
                )
                pose = pose_out(
                    self.model, desc, torch.from_numpy(r["geom"]).to(self.device)
                )
                pose[:, 3] *= YAW_SCALE
                px = uv.cpu().numpy() * r["sq"][:, None, 2:] + r["sq"][:, None, :2]
                gt_px = r["uv"] * r["sq"][:, None, 2:] + r["sq"][:, None, :2]
                for j, b in enumerate(r["rois"][:, 0].astype(int)):
                    per[part[b]]["kp"].append((px[j], gt_px[j], r["kv"][j]))
                    if r["has_pose"][j]:
                        per[part[b]]["pose"].append((
                            pose[j].cpu().numpy(),
                            r["pose"][j],
                        ))
            out |= per
        return out


def det_metrics(pred: dict, labels: dict, thresholds: list[float]) -> dict:
    return {
        f"@{t}": multi.score(
            {n: [b for b in v["boxes"] if b[4] >= t] for n, v in pred.items()},
            {n: [p for p in ps if p["box"] is not None] for n, ps in labels.items()},
        )
        for t in thresholds
    }


def kp_metrics(pred: dict) -> dict:
    err = np.concatenate(
        [
            np.linalg.norm(px - gt, axis=1)[kv]
            for v in pred.values()
            for px, gt, kv in v["kp"]
        ]
        or [np.zeros(0)]
    )
    pallets = sum(bool(kv.any()) for v in pred.values() for _, _, kv in v["kp"])
    if not len(err):
        return {}
    return {
        "median_px": float(np.median(err)),
        "p90_px": float(np.quantile(err, 0.9)),
        "mean_px": float(err.mean()),
        "keypoints": len(err),
        "pallets": pallets,
    }


def box_prior(sets: dict) -> dict:
    """Baseline with no image content: the mean RoI layout of the real training keypoints
    (hand labels + pseudo-labels), placed in the GT box of each real `val` pallet."""
    train = roi_inputs([*sets["real_train"].values(), *sets["pseudo"].values()], None)
    kv = train["kv"]
    mean = np.stack([train["uv"][kv[:, i], i].mean(0) for i in range(NK)])
    val = roi_inputs(list(sets["real_val"].values()), None)
    px = mean[None] * val["sq"][:, None, 2:] + val["sq"][:, None, :2]
    gt = val["uv"] * val["sq"][:, None, 2:] + val["sq"][:, None, :2]
    return kp_metrics({"": {"kp": list(zip(px, gt, val["kv"], strict=True))}})


def pose_metrics(pred: dict) -> dict:
    rows = [(p, g) for v in pred.values() for p, g in v["pose"]]
    if not rows:
        return {}
    p, g = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    pos = np.linalg.norm(p[:, :3] - g[:, :3], axis=1) * 100
    yaw = np.abs(p[:, 3] - g[:, 3])
    out = {}
    for name, sel in [("all", np.ones(len(g), bool))] + [
        (
            f"{lo:g}-{hi:g}m" if hi < 99 else f">{lo:g}m",
            (g[:, 0] >= lo) & (g[:, 0] < hi),
        )
        for lo, hi in RANGES
    ]:
        if sel.any():
            out[name] = {
                "n": int(sel.sum()),
                "pos_cm_median": float(np.median(pos[sel])),
                "pos_cm_p90": float(np.quantile(pos[sel], 0.9)),
                "forward_cm_median": float(
                    np.median(np.abs(p[sel, 0] - g[sel, 0])) * 100
                ),
                "lateral_cm_median": float(
                    np.median(np.abs(p[sel, 1] - g[sel, 1])) * 100
                ),
                "yaw_deg_median": float(np.median(yaw[sel])),
                "yaw_deg_p90": float(np.quantile(yaw[sel], 0.9)),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--res", type=int, nargs=2, default=[256, 144], help="encoder input W H"
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 4], choices=sorted(LAYERS)
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--batch", type=int, default=8, help="frames per source per step"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res = tuple(args.res)
    sets = load_sets()
    paths = sorted({p for d in sets.values() for p in d})
    t0 = time.time()
    store = Store(paths, args.device, res)
    logger.info(
        "data",
        seconds=round(time.time() - t0),
        frames={s: len(d) for s, d in sets.items()},
        pallets={s: sum(len(v) for v in d.values()) for s, d in sets.items()},
    )
    prior = box_prior(sets)
    logger.info("kp box prior (real val)", **prior)
    (args.out / "kp_box_prior.json").write_text(json.dumps(prior, indent=1))
    for k in args.k:
        tag = f"{res[0]}x{res[1]}_k{k}"
        run = start(
            "aux-heads",
            vars(args) | {"k": k, "layers": LAYERS[k]},
            ["aux-heads", f"res:{res[0]}x{res[1]}", f"k:{k}"],
            notes=f"frozen {ENCODER} layers {LAYERS[k]} at {res[0]}x{res[1]}: det+load, RoI keypoints, RoI pose heads",
            enabled=not args.no_wandb,
        )
        runner = Runner(store, sets, k, args.device)
        t0 = time.time()

        def log(step: int, losses: dict, run=run, t0=t0, tag=tag) -> None:
            if run is not None and step % 25 == 0:
                run.log({f"train/{a}": v for a, v in losses.items()} | {"step": step})
            if step % 500 == 0:
                logger.info(
                    "train",
                    cfg=tag,
                    step=step,
                    s=round(time.time() - t0),
                    **{a: round(v, 4) for a, v in losses.items()},
                )

        runner.train(args.steps, args.batch, args.lr, log)
        train_s = time.time() - t0
        results: dict = {
            "config": {
                "res": list(res),
                "k": k,
                "layers": LAYERS[k],
                "steps": args.steps,
                "train_seconds": round(train_s),
            }
        }
        for split in ("real_test", "real_val"):
            pred = runner.predict(sets[split])
            results[f"det_{split}"] = det_metrics(pred, sets[split], args.thresholds)
            if split == "real_val":
                results["kp_real_val"] = kp_metrics(pred)
        pred = runner.predict(sets["synth_val"])
        results["kp_synth_val"] = kp_metrics(pred)
        results["pose_synth_val"] = pose_metrics(pred)
        logger.info(
            "result",
            cfg=tag,
            **{
                "det_test@0.3": {
                    a: round(v, 3) for a, v in results["det_real_test"]["@0.3"].items()
                },
                "det_val@0.3": {
                    a: round(v, 3) for a, v in results["det_real_val"]["@0.3"].items()
                },
                "kp_real_val": results["kp_real_val"],
                "pose": results["pose_synth_val"].get("all"),
            },
        )
        (args.out / f"metrics_{tag}.json").write_text(json.dumps(results, indent=1))
        torch.save(
            {
                "state_dict": runner.model.state_dict(),
                "encoder": ENCODER,
                "layers": LAYERS[k],
                "res": res,
            },
            args.out / f"heads_{tag}.pt",
        )
        if run is not None:
            run.summary.update(_flat(results))
            run.finish()
        del runner
        torch.cuda.empty_cache()


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for a, v in d.items():
        if isinstance(v, dict):
            out |= _flat(v, f"{prefix}{a}/")
        elif isinstance(v, (int, float)):
            out[f"{prefix}{a}"] = v
    return out


if __name__ == "__main__":
    main()
