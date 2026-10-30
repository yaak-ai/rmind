"""Proposer head trained on the hand-labelled multi-pallet boxes, k-fold cross-validated.

The 50 frames of `labels/multi_labels_v1.json` (every pallet boxed around its base and
entry face) are split into `--folds` contiguous blocks in time; each fold trains on the
others and is scored on its own block, so every frame is scored by a model that never
saw it or its neighbours. Positives (`--positives all`, default): every labelled pallet,
on the floor, in racks or on a stack, with or without a visible face; only `unsure`
boxes are ignored. Choosing which pallet to approach is a later step, not the
detector's. `--positives floor-face` restores the first convention (only floor pallets
with a visible face; the rest ignored).

Features: the frozen policy encoder (`vit_small_patch16_dinov3.lvd1689m`, layer 10,
pre-norm) at `--res` (256x144 = the policy input; 512x288 = a 2x pass), recomputed per
batch so crops, flips and colour jitter can augment the 40 training frames. The head
is the CenterNet head of `pallet_proposer_head` with its output grid at 2x the token
grid (60 px cells at 256x144, 30 px at 512x288).

Scores, pooled over folds, on the positive pallets:
    recall50   a predicted box at IoU >= 0.5
    centre     a predicted box contains the pallet centre and its width is within 2x
               (what a proposer needs: the keypoint crop has margin)
    precision  predicted boxes that contain a labelled pallet's centre (any pallet)
Grounding DINO (proposer cache) is scored the same way.
"""

import argparse
import json
import operator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from structlog import get_logger
from torch import Tensor, nn

from rmind.pallet.tracking import log_model, start, use_labels
from rmind.scripts.pallet_proposer_head import (
    ENCODER,
    LAYER,
    MEAN,
    STD,
    focal_loss,
    iou,
)

logger = get_logger(__name__)
WORKSPACE = Path("/nasa/team-space/nikita/pallet-detector")
WIDTH, HEIGHT = 1920, 1080


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(16, cout),
        nn.GELU(),
    )


class Head(nn.Module):
    """Outputs (heat, size, offset) and, with `load`, a loaded-vs-empty logit per cell."""

    def __init__(self, dim: int = 384, width: int = 256, *, load: bool = True) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(dim, width, 1), _block(width, width), _block(width, width)
        )
        self.up = _block(width, 128)
        self.heat, self.size, self.offset = (
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Conv2d(128, 2, 3, padding=1),
            nn.Conv2d(128, 2, 3, padding=1),
        )
        self.load = nn.Conv2d(128, 1, 3, padding=1) if load else None
        nn.init.constant_(self.heat.bias, -2.19)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        x = self.up(
            F.interpolate(
                self.trunk(x), scale_factor=2, mode="bilinear", align_corners=False
            )
        )
        out = (self.heat(x), self.size(x), self.offset(x))
        return out if self.load is None else (*out, self.load(x))


def make_head(state_dict: dict | None = None) -> Head:
    """A head matching `state_dict` (older checkpoints have no load output)."""
    head = Head(load=state_dict is None or "load.weight" in state_dict)
    if state_dict is not None:
        head.load_state_dict(state_dict)
    return head


POSITIVES = "all"


def is_positive(p: dict) -> bool:
    """A box the detector must find (and is scored on); the rest is ignored."""
    if p.get("unsure"):
        return False
    return POSITIVES == "all" or (p["floor"] and p["face"])


def targets(pallets: list[dict], gh: int, gw: int) -> dict[str, Tensor]:
    cell = WIDTH / gw
    heat, ignore = torch.zeros(1, gh, gw), torch.zeros(1, gh, gw, dtype=torch.bool)
    size, offset, mask = (
        torch.zeros(2, gh, gw),
        torch.zeros(2, gh, gw),
        torch.zeros(gh, gw, dtype=torch.bool),
    )
    load, load_mask = torch.zeros(1, gh, gw), torch.zeros(1, gh, gw, dtype=torch.bool)
    ys, xs = torch.meshgrid(
        torch.arange(gh) + 0.5, torch.arange(gw) + 0.5, indexing="ij"
    )
    for p in pallets:
        x0, y0, x1, y1 = (v / cell for v in p["box"])
        if not is_positive(p):
            ignore[0, int(y0) : int(np.ceil(y1)), int(x0) : int(np.ceil(x1))] = True
            continue
        cx, cy, w, h = (
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            max(x1 - x0, 0.1),
            max(y1 - y0, 0.1),
        )
        sx, sy = max(0.5, w / 6), max(0.5, h / 3)
        heat[0] = torch.maximum(
            heat[0],
            torch.exp(-((xs - cx) ** 2 / (2 * sx**2) + (ys - cy) ** 2 / (2 * sy**2))),
        )
        i, j = min(int(cy), gh - 1), min(int(cx), gw - 1)
        heat[0, i, j] = 1.0
        size[:, i, j] = torch.tensor([np.log(w), np.log(h)])
        offset[:, i, j] = torch.tensor([cx - j, cy - i])
        mask[i, j] = True
        if p.get("load") in {"loaded", "empty"}:
            load[0, i, j] = float(p["load"] == "loaded")
            load_mask[0, i, j] = True
    ignore &= heat < 0.5
    return {
        "heat": heat,
        "ignore": ignore,
        "size": size,
        "offset": offset,
        "mask": mask,
        "load": load,
        "load_mask": load_mask,
    }


def decode(
    out: tuple[Tensor, ...], threshold: float, k: int = 20
) -> list[list[list[float]]]:
    """Head output -> per image [x0, y0, x1, y1, score(, p_loaded)] in full-res pixels."""
    heat, size, offset = out[:3]
    p_load = out[3].sigmoid() if len(out) > 3 else None
    _, _, _gh, gw = heat.shape
    cell = WIDTH / gw
    p = heat.sigmoid()
    peaks = p * (F.max_pool2d(p, 3, stride=1, padding=1) == p)
    out = []
    for b in range(p.shape[0]):
        scores, flat = peaks[b, 0].flatten().topk(k)
        boxes = []
        for s, f in zip(scores.tolist(), flat.tolist(), strict=True):
            if s < threshold:
                break
            i, j = divmod(f, gw)
            cx, cy = j + offset[b, 0, i, j].item(), i + offset[b, 1, i, j].item()
            w, h = size[b, 0, i, j].exp().item(), size[b, 1, i, j].exp().item()
            boxes.append([
                (cx - w / 2) * cell,
                (cy - h / 2) * cell,
                (cx + w / 2) * cell,
                (cy + h / 2) * cell,
                s,
                *([p_load[b, 0, i, j].item()] if p_load is not None else []),
            ])
        out.append(boxes)
    return out


def _contains(box: list[float], pt: tuple[float, float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def score(pred: dict[str, list], test: dict[str, list]) -> dict[str, float]:
    n = hit = cen = n_pred = good = 0
    load_n, load_ok = {"loaded": 0, "empty": 0}, {"loaded": 0, "empty": 0}
    for name, pallets in test.items():
        p = pred.get(name, [])
        centres = [
            ((q["box"][0] + q["box"][2]) / 2, (q["box"][1] + q["box"][3]) / 2)
            for q in pallets
        ]
        for q in pallets:
            if not is_positive(q):
                continue
            g, c, w = (
                q["box"],
                ((q["box"][0] + q["box"][2]) / 2, (q["box"][1] + q["box"][3]) / 2),
                q["box"][2] - q["box"][0],
            )
            n += 1
            hit += any(iou(g, b) >= 0.5 for b in p)
            found = [b for b in p if _contains(b, c) and 0.5 <= (b[2] - b[0]) / w <= 2]
            cen += bool(found)
            if found and len(found[0]) > 5 and q.get("load") in {"loaded", "empty"}:
                best = max(found, key=operator.itemgetter(4))
                load_n[q["load"]] += 1
                load_ok[q["load"]] += (best[5] >= 0.5) == (q["load"] == "loaded")
        n_pred += len(p)
        good += sum(any(_contains(b, c) for c in centres) for b in p)
    rec, prec = cen / max(n, 1), good / max(n_pred, 1)
    out = {
        "recall50": hit / max(n, 1),
        "centre": rec,
        "precision": prec,
        "f1": 2 * rec * prec / max(rec + prec, 1e-9),
        "boxes_per_frame": n_pred / max(len(test), 1),
    }
    if sum(load_n.values()):  # loaded/empty accuracy on the pallets that were found
        out |= {
            "load_acc": sum(load_ok.values()) / sum(load_n.values()),
            "load_acc_loaded": load_ok["loaded"] / max(load_n["loaded"], 1),
            "load_acc_empty": load_ok["empty"] / max(load_n["empty"], 1),
            "load_n": sum(load_n.values()),
        }
    return out


class Frames:
    """Full-res frames -> augmented (or plain) encoder input at the chosen resolution."""

    def __init__(self, images: Path, names: list[str], res: tuple[int, int]) -> None:
        from PIL import Image

        self.res = res
        # uint8 in memory (a float frame is 25 MB); absolute names (synthetic sets) ignore `images`
        self.full = {
            n: torch.from_numpy(
                np.array(Image.open(images / n).convert("RGB"))
            ).permute(2, 0, 1)
            for n in names
        }

    def image(self, name: str) -> Tensor:
        return self.full[name].float() / 255

    def batch(
        self, names: list[str], labels: dict[str, list], rng: np.random.Generator | None
    ) -> tuple[Tensor, list[list[dict]]]:
        xs, ls = [], []
        for n in names:
            img, pallets = self.image(n), [dict(p) for p in labels.get(n, [])]
            if rng is not None:
                s = rng.uniform(0.75, 1.0)
                cw, ch = WIDTH * s, HEIGHT * s
                x0, y0 = rng.uniform(0, WIDTH - cw), rng.uniform(0, HEIGHT - ch)
                img = img[:, int(y0) : int(y0 + ch), int(x0) : int(x0 + cw)]
                k = WIDTH / img.shape[2], HEIGHT / img.shape[1]
                pallets = [
                    p
                    | {
                        "box": [
                            (p["box"][0] - x0) * k[0],
                            (p["box"][1] - y0) * k[1],
                            (p["box"][2] - x0) * k[0],
                            (p["box"][3] - y0) * k[1],
                        ]
                    }
                    for p in pallets
                ]
                pallets = [
                    p
                    for p in pallets
                    if p["box"][2] > 0
                    and p["box"][0] < WIDTH
                    and p["box"][3] > 0
                    and p["box"][1] < HEIGHT
                ]
                for p in pallets:
                    b = p["box"]
                    p["box"] = [
                        max(b[0], 0),
                        max(b[1], 0),
                        min(b[2], WIDTH),
                        min(b[3], HEIGHT),
                    ]
                if rng.random() < 0.5:
                    img = img.flip(-1)
                    pallets = [
                        p
                        | {
                            "box": [
                                WIDTH - p["box"][2],
                                p["box"][1],
                                WIDTH - p["box"][0],
                                p["box"][3],
                            ]
                        }
                        for p in pallets
                    ]
                img = (img * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)).clamp(0, 1)
            x = F.interpolate(
                img[None],
                size=(self.res[1], self.res[0]),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )[0]
            xs.append((x - MEAN) / STD)
            ls.append(pallets)
        return torch.stack(xs), ls


class Loco(Frames):
    """LOCO (TUM, CC0): 4.7k logistics images, every pallet boxed. All of them are positives
    (floor and racks alike: the head should first learn what a pallet is). Images are
    brought to the 1920x1080 frame: 16:9 ones scaled, 4:3 ones centre-cropped to 16:9."""

    def __init__(self, root: Path, res: tuple[int, int]) -> None:
        self.res = res
        d = json.loads((root / "loco-all-v1.json").read_text())
        pallet = next(c["id"] for c in d["categories"] if c["name"] == "pallet")
        paths = {p.name: p for p in root.glob("dataset/**/*.jpg")}
        boxes: dict[int, list] = {}
        for a in d["annotations"]:
            if a["category_id"] == pallet:
                boxes.setdefault(a["image_id"], []).append(a["bbox"])
        self.items, self.labels = {}, {}
        for im in d["images"]:
            if im["file_name"] not in paths or im["id"] not in boxes:
                continue
            w, h = im["width"], im["height"]
            ch = w * 9 / 16
            y0 = (h - ch) / 2
            k = WIDTH / w
            pallets = []
            for x, y, bw, bh in boxes[im["id"]]:
                b = [x * k, (y - y0) * k, (x + bw) * k, (y + bh - y0) * k]
                if b[3] > 0 and b[1] < HEIGHT:
                    pallets.append({
                        "box": [b[0], max(b[1], 0), b[2], min(b[3], HEIGHT)],
                        "floor": True,
                        "face": True,
                    })
            self.items[im["file_name"]] = (paths[im["file_name"]], y0)
            self.labels[im["file_name"]] = pallets
        self.names = sorted(self.items)

    def image(self, name: str) -> Tensor:
        from PIL import Image

        path, y0 = self.items[name]
        im = Image.open(path).convert("RGB")
        im = im.crop((0, round(y0), im.width, round(y0 + im.width * 9 / 16))).resize(
            (WIDTH, HEIGHT), Image.Resampling.BILINEAR
        )
        return torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255


def train_head(
    head: nn.Module,
    sample,
    steps: int,
    lr: float,
    features,
    gh: int,
    gw: int,
    device: str,
    log,
) -> float:
    """`steps` of AdamW + one-cycle on batches from `sample(rng) -> (x, pallets)`; returns the last loss."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.05
    )
    loss = torch.zeros(())
    for step in range(steps):
        head.train()
        x, ls = sample()
        t = [targets(p, gh, gw) for p in ls]
        t = {key: torch.stack([v[key] for v in t]).to(device) for key in t[0]}
        out = head(features(x))
        heat, size, offset = out[:3]
        m = t["mask"][:, None].expand_as(size)
        norm = m.sum().clamp(min=1)
        loss = (
            focal_loss(heat, t["heat"], t["ignore"])
            + 0.2 * F.l1_loss(size[m], t["size"][m], reduction="sum") / norm
            + 2 * F.l1_loss(offset[m], t["offset"][m], reduction="sum") / norm
        )
        if (
            len(out) > 3 and t["load_mask"].any()
        ):  # loaded vs empty at each pallet centre
            lm = t["load_mask"]
            loss += 0.5 * F.binary_cross_entropy_with_logits(out[3][lm], t["load"][lm])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        log(step, float(loss))
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--labels", type=Path, default=WORKSPACE / "labels/multi_labels_v1.json"
    )
    parser.add_argument("--images", type=Path, default=WORKSPACE / "images/multi")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/nasa/team-space/nikita/pallet-detector-ab/proposer-cache-20261027/cache"
        ),
    )
    parser.add_argument(
        "--res", type=int, nargs=2, default=[256, 144], help="encoder input W H"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help='{"train": [...], "test": [...]} file: train once, score on "test" (no k-fold)',
    )
    parser.add_argument(
        "--steps", type=int, default=1500, help="fine-tuning steps per fold"
    )
    parser.add_argument(
        "--loco",
        type=Path,
        default=None,
        help="LOCO root: pretrain the head on it first",
    )
    parser.add_argument("--pretrain-steps", type=int, default=20000)
    parser.add_argument(
        "--init-head",
        type=Path,
        default=None,
        help="start every fold from this pretrained head (skips --loco)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="train one head on all frames (final.pt) instead of k-fold",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--positives", choices=["all", "floor-face"], default="all")
    parser.add_argument(
        "--extra-train",
        type=Path,
        nargs="*",
        default=[],
        help="more labelled frames (e.g. synthetic multi_train.json, absolute image paths) "
        "added to every fold's training set, never scored",
    )
    parser.add_argument(
        "--extra-share",
        type=float,
        default=0.5,
        help="share of each training batch drawn from --extra-train",
    )
    parser.add_argument(
        "--extra-test",
        type=Path,
        default=None,
        help="a separate test set (e.g. synthetic multi_test.json) each head is also scored on",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from rmind.components.timm_backbone import TimmBackbone

    global POSITIVES  # ruff: ignore[global-statement]
    POSITIVES = args.positives
    res = tuple(args.res)
    run = start(
        "train-proposer",
        vars(args),
        [
            "proposer",
            "dinov3-head",
            "multi-labels",
            "kfold",
            f"res:{res[0]}x{res[1]}",
            f"positives:{args.positives}",
            *(["loco-pretrain"] if args.loco else []),
            *args.tags,
        ],
        notes=f"head on frozen {ENCODER} layer {LAYER} at {res[0]}x{res[1]}, trained on hand-labelled multi-pallet boxes, {args.folds}-fold CV by time",
        enabled=not args.no_wandb,
    )
    use_labels(run, args.labels, "multi-labels")
    labels = json.loads(args.labels.read_text())
    names = sorted(
        labels, key=lambda n: (n[:-4].rsplit("_", 1)[0], int(n[:-4].rsplit("_", 1)[1]))
    )
    if args.split is not None:  # one fixed train / test split instead of k-fold
        test_set = set(json.loads(args.split.read_text())["test"])
        folds = [np.array([i for i, n in enumerate(names) if n in test_set])]
    else:
        folds = np.array_split(np.arange(len(names)), args.folds)
    extra: dict[str, list] = {}
    for f in args.extra_train:
        extra |= json.loads(f.read_text())
    extra_names = sorted(extra)
    extra_test = json.loads(args.extra_test.read_text()) if args.extra_test else {}
    frames = Frames(args.images, names + extra_names + sorted(extra_test), res)
    train_labels = labels | extra
    logger.info(
        "frames",
        labelled=len(names),
        extra=len(extra_names),
        extra_test=len(extra_test),
    )

    def pick(rng: np.random.Generator, pool: list[str]) -> list[str]:
        """A training batch: each item from --extra-train with probability --extra-share."""
        return [
            str(rng.choice(extra_names))
            if extra_names and rng.random() < args.extra_share
            else str(rng.choice(pool))
            for _ in range(args.batch)
        ]

    def predict(
        head: nn.Module, frame_names: list[str]
    ) -> dict[float, dict[str, list]]:
        """Boxes per threshold, in chunks (a whole test set at high res does not fit)."""
        head.eval()
        out: dict[float, dict[str, list]] = {t: {} for t in args.thresholds}
        for i in range(0, len(frame_names), 16):
            chunk = frame_names[i : i + 16]
            x, _ = frames.batch(chunk, extra_test, None)
            with torch.no_grad():
                o = head(features(x))
            for t in args.thresholds:
                out[t] |= dict(zip(chunk, decode(o, t), strict=True))
        return out

    extra_scores: list[dict[float, dict[str, float]]] = []
    enc = (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
        .to(args.device)
        .eval()
    )
    gh, gw = res[1] // 16 * 2, res[0] // 16 * 2

    def features(x: Tensor) -> Tensor:
        with torch.no_grad():
            return enc(x.to(args.device))

    pretrained = None
    if args.loco is not None:
        loco = Loco(args.loco, res)
        logger.info(
            "loco",
            images=len(loco.names),
            pallets=sum(len(v) for v in loco.labels.values()),
        )
        head = Head().to(args.device)
        rng = np.random.default_rng(100)

        def log(s, v):  # ruff: ignore[missing-return-type-private-function]
            return (
                run.log({"pretrain/loss": v, "pretrain/step": s})
                if run is not None and s % 100 == 0
                else None,
                logger.info("pretrain", step=s, loss=round(v, 4))
                if s % 1000 == 0
                else None,
            )

        train_head(
            head,
            lambda: loco.batch(
                list(rng.choice(loco.names, args.batch)), loco.labels, rng
            ),
            args.pretrain_steps,
            args.lr,
            features,
            gh,
            gw,
            args.device,
            log,
        )
        pretrained = {k: v.detach().clone() for k, v in head.state_dict().items()}
        torch.save(
            {
                "state_dict": pretrained,
                "encoder": ENCODER,
                "layer": LAYER,
                "res": res,
                "data": "loco",
            },
            args.out / "pretrained_loco.pt",
        )
        # LOCO alone, no fine-tuning, on all 50 frames
        head.eval()
        x, _ = frames.batch(names, labels, None)
        with torch.no_grad():
            out = head(features(x))
        for thr in args.thresholds:
            logger.info(
                "loco-only",
                threshold=thr,
                **{
                    a: round(v, 3)
                    for a, v in score(
                        dict(zip(names, decode(out, thr), strict=True)), labels
                    ).items()
                },
            )
        loco_only = {
            f"loco_only@{thr}": score(
                dict(zip(names, decode(out, thr), strict=True)), labels
            )
            for thr in args.thresholds
        }
    elif (
        args.init_head is not None
    ):  # a head already pretrained (e.g. pretrained_loco.pt)
        pretrained = torch.load(args.init_head, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        head = Head().to(args.device).eval()
        head.load_state_dict(pretrained, strict=False)
        x, _ = frames.batch(names, labels, None)
        with torch.no_grad():
            out = head(features(x))
        loco_only = {
            f"loco_only@{thr}": score(
                dict(zip(names, decode(out, thr), strict=True)), labels
            )
            for thr in args.thresholds
        }

    if (
        args.final
    ):  # one head on every labelled frame, for use (no cross-validation score)
        head = Head().to(args.device)
        if pretrained is not None:
            head.load_state_dict(pretrained, strict=False)
        rng = np.random.default_rng(0)
        train_head(
            head,
            lambda: frames.batch(pick(rng, names), train_labels, rng),
            args.steps,
            args.lr if pretrained is None else args.lr / 3,
            features,
            gh,
            gw,
            args.device,
            lambda _step, _loss: None,
        )
        if extra_test:
            for t, pr in predict(head, sorted(extra_test)).items():
                logger.info(
                    "extra-test",
                    threshold=t,
                    **{a: round(b, 3) for a, b in score(pr, extra_test).items()},
                )
        path = args.out / "final.pt"
        torch.save(
            {
                "state_dict": head.state_dict(),
                "encoder": ENCODER,
                "layer": LAYER,
                "res": res,
                "fold": -1,
                "test": [],
            },
            path,
        )
        log_model(
            run,
            path,
            "proposer-head",
            {
                "targets": "multi-labels",
                "res": f"{res[0]}x{res[1]}",
                "final": True,
                "labels": args.labels.name,
            },
        )
        if run is not None:
            run.finish()
        logger.info("final", path=str(path))
        return

    preds: dict[float, dict[str, list]] = {t: {} for t in args.thresholds}
    for k, test_idx in enumerate(folds):
        test_names = [names[i] for i in test_idx]
        train_names = [n for n in names if n not in test_names]
        head = Head().to(args.device)
        if pretrained is not None:
            head.load_state_dict(pretrained, strict=False)
        rng = np.random.default_rng(k)

        def log(s, v, k=k):  # ruff: ignore[missing-return-type-private-function]
            return (
                run.log({f"fold{k}/loss": v, "step": s})
                if run is not None and s % 50 == 0
                else None
            )

        loss = train_head(
            head,
            lambda rng=rng, names_=train_names: frames.batch(
                pick(rng, names_), train_labels, rng
            ),
            args.steps,
            args.lr if pretrained is None else args.lr / 3,
            features,
            gh,
            gw,
            args.device,
            log,
        )
        head.eval()
        x, _ = frames.batch(test_names, labels, None)
        with torch.no_grad():
            out = head(features(x))
        for thr in args.thresholds:
            preds[thr] |= dict(zip(test_names, decode(out, thr), strict=True))
        fold_metrics = score(
            preds[0.3] if 0.3 in preds else preds[args.thresholds[0]],
            {n: labels[n] for n in test_names},
        )
        logger.info(
            "fold",
            fold=k,
            test=len(test_names),
            loss=round(loss, 4),
            **{a: round(v, 3) for a, v in fold_metrics.items()},
        )
        if extra_test:
            extra_scores.append({
                t: score(pr, extra_test)
                for t, pr in predict(head, sorted(extra_test)).items()
            })
            logger.info(
                "extra-test",
                fold=k,
                **{
                    a: round(b, 3)
                    for a, b in extra_scores[-1][args.thresholds[0]].items()
                },
            )
        torch.save(
            {
                "state_dict": head.state_dict(),
                "encoder": ENCODER,
                "layer": LAYER,
                "res": res,
                "fold": k,
                "test": test_names,
            },
            args.out / f"fold{k}.pt",
        )

    # every metric is on the frames that were scored (all of them in k-fold, the test set otherwise)
    scored = [names[i] for f in folds for i in f]
    labels = {n: labels[n] for n in scored}
    results = {f"head@{t}": score(preds[t], labels) for t in args.thresholds}
    if pretrained is not None and args.split is None:
        results |= loco_only
    for t in args.thresholds if extra_scores else ():  # mean over the fold heads
        results[f"extra_test_head@{t}"] = {
            a: float(np.mean([e[t][a] for e in extra_scores if a in e[t]]))
            for a in extra_scores[0][t]
        }
    cached = {}
    for job in {n[:-4].rsplit("_", 1)[0] for n in scored}:
        d = torch.load(args.cache / f"{job}.pt", weights_only=False, map_location="cpu")
        cached |= {
            f"{job}_{fr:06d}.jpg": b
            for fr, b in zip(d["frame"], d["boxes"], strict=True)
        }
    for t in (0.25, 0.35, 0.45):
        results[f"gdino@{t}"] = score(
            {n: [b for b in cached.get(n, []) if b[4] >= t] for n in scored}, labels
        )
    for k, v in results.items():
        logger.info("result", model=k, **{a: round(b, 3) for a, b in v.items()})
    (args.out / "metrics.json").write_text(json.dumps(results, indent=1))
    (args.out / "predictions.json").write_text(
        json.dumps({str(t): p for t, p in preds.items()})
    )
    if run is not None:
        run.summary.update({
            f"{m}/{a}": b for m, v in results.items() for a, b in v.items()
        })
        log_model(
            run,
            args.out / f"fold{len(folds) - 1}.pt",
            "proposer-head",
            {
                "targets": "multi-labels",
                "res": f"{res[0]}x{res[1]}",
                "kfold": args.folds,
            },
        )
        run.finish()


if __name__ == "__main__":
    main()
