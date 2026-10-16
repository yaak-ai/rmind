#! /usr/bin/env python
"""Log the W1 policy readout query's attention over image patches, for one or more drives.

For every 6-frame window of a drive (the ``yaak/rmind_zero/drahve`` inference pipeline: same
data, same chunk-stripped model), runs the policy's image stage by hand and records, for the
readout query, the head-averaged cross-attention over the 16x16 DINO patches of the current
frame I(T-1):

* ``attn_l{k}``     per decoder layer k (softmax over the 256 patches, sums to 1)
* ``attn_rollout``  residual-augmented rollout over all layers (Abnar & Zuidema 2020):
                    R <- 0.5 R + 0.5 C~_l ; R <- (0.5 S~_l + 0.5 I) R ; rows renormalized,
                    where ~ = head fusion (``--head-fusion`` mean|max|min) followed by
                    discarding the lowest ``--discard-ratio`` fraction of each query's
                    weights and renormalizing. Fusion + discard follow the widely used
                    vit-explain implementation (Gildenblat 2020); the paper itself uses mean
                    fusion and no discard (``--discard-ratio 0``). Per-layer ``attn_l{k}``
                    maps stay raw (head mean, nothing discarded).
* ``attn_rollout_q`` / ``attn_l{k}_q``  the same rollout / per-layer maps for EVERY policy query
                    (N_q x P; ``readout_query`` marks the one whose features are decoded)
* summaries: normalized entropy (readout and per query), top-k patches, sky / hood mass

W1 queries first read the latent through ``policy.latent_decoder`` (K=V=L(T-2)); the image
stage ``policy.decoder`` then scores them against the image patches (``key_source="image"``).
W0 (no latent_decoder, ``key_source="latent"``): the readout scores the latent L(T-2) as keys and
reads values I(T-1). L is the per-patch FWD output, so key slot k is patch k and the maps are
spatial in the same 16x16 grid.

Outputs, per drive ``<out>/<vehicle>__<drive>/``:
    attention.mp4       video, one frame per window: model view | readout rollout overlay on top,
                        readout per-layer overlays, then every query's rollout (4 per row, with its
                        entropy), status strip (frame, speed, steer, turn signal, predicted chunk,
                        entropy). Encoded with the system ``ffmpeg``.
    attention.parquet   the same maps as data, one row per window (list[float] per map)
    mean_rollout.npy    (16, 16) mean rollout map over the drive
and ``<out>/summary.parquet`` with one row per drive.

Usage:
    uv run --extra train --extra predict python -m rmind.scripts.policy_attention \\
        --artifact yaak/rmind/model-yh6qrp22:v9 --out outputs/policy_attention \\
        /nasa/drives/yaak/data/G1-00428/2024-04-23--07-00-31 [more drive dirs ...]

    # many drives: one directory per line
    ... --drives-file drives.txt

    # local checkpoint instead of a W&B download
    ... --ckpt /path/to/model.ckpt
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
from PIL import Image, ImageDraw
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from structlog import get_logger

if TYPE_CHECKING:
    from torch import Tensor

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
CAMERA = "cam_front_left"
FRAME_IDX = f"meta/ImageMetadata.{CAMERA}/frame_idx"
TIME_STAMP = f"meta/ImageMetadata.{CAMERA}/time_stamp"
OBS_KEYS = {
    "speed": "meta/VehicleMotion/speed",
    "gas": "meta/VehicleMotion/gas_pedal_normalized",
    "brake": "meta/VehicleMotion/brake_pedal_normalized",
    "steering": "meta/VehicleMotion/steering_angle_normalized",
    "turn_signal": "meta/VehicleState/turn_signal",
}
PRED_CHANNELS = ("gas_pedal", "brake_pedal", "steering_angle")
SKY_ROWS = (0, 1)
HOOD_ROWS = (13, 14, 15)
TOP_K = 5
QUERIES_PER_ROW = 4
# Model image geometry (W1 config): CenterCrop(320, 576) of the 324x576 JPG, then Resize(256, 256).
CROP_HW = (320, 576)


def compose_cfg(drive_dir: Path, artifact: str, cache_dir: Path) -> Any:
    slug = f"{drive_dir.parent.name}__{drive_dir.name}"
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="predict.yaml",
            overrides=[
                "inference=yaak/rmind_zero/drahve",
                f"model.artifact={artifact}",
                f"drive_dir={drive_dir}",
                # one samples run_folder per drive so sequential drives never share storage
                f"datamodule.predict.dataset.samples.run_folder={cache_dir / slug}",
            ],
        )


def load_model(cfg: Any, ckpt: Path | None, device: str) -> torch.nn.Module:
    if ckpt is None:
        model = instantiate(cfg.model)
    else:
        from rmind.models.zero import Zero  # noqa: PLC0415

        model = Zero.load_from_checkpoint(
            ckpt, hparams_jq=cfg.model.hparams_jq, strict=False, map_location="cpu"
        )
    return model.eval().to(device)


def policy_qkv(policy: Any, episode: Any, latent: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(queries, key, value) exactly as PolicyObjective._features feeds policy.decoder."""
    from rmind.components.base import Modality  # noqa: PLC0415
    from rmind.components.objectives.base import PATCHES  # noqa: PLC0415

    patches = episode.get(PATCHES)[:, -1:]
    value = policy.value_norm(patches) if policy.value_norm is not None else patches
    lat = latent[:, -2:-1]
    lat = policy.key_norm(lat) if policy.key_norm is not None else lat
    queries = episode.embeddings.get((Modality.UTILITY, "policy"))[:, -1:]
    if policy.latent_decoder is not None:
        queries = policy.latent_decoder({"query": queries, "key": lat, "value": lat})
    key = lat if policy.key_source == "latent" else value
    return queries, key, value


def decoder_attention(policy: Any, queries: Tensor, key: Tensor, value: Tensor) -> tuple[list[Tensor], list[Tensor | None]]:
    """Per-layer per-head cross (b, h, N_q, P) and self (b, h, N_q, N_q) attention.

    Re-runs CrossAttentionDecoderBlock.forward step by step (eval: dropout off) so layer k
    sees the evolving query stream.
    """
    b, _, n_q, d = queries.shape
    p = key.shape[2]
    x = queries.reshape(b, n_q, d)
    k = key.reshape(b, p, d)
    v = value.reshape(b, p, d)
    cross, self_ = [], []
    for blk in policy.decoder.decoder.layers:
        out, w = blk.cross_attn(
            query=blk.cross_attn_norm(x), key=k, value=v,
            need_weights=True, average_attn_weights=False,
        )
        x = x + out
        cross.append(w.float())
        if blk.use_self_attn:
            xn = blk.self_attn_norm(x)
            out, w = blk.self_attn(
                query=xn, key=xn, value=xn, need_weights=True, average_attn_weights=False
            )
            x = x + out
            self_.append(w.float())
        else:
            self_.append(None)
        x = x + blk.mlp(blk.mlp_norm(x))
    return cross, self_


def fuse_heads(w: Tensor, how: str) -> Tensor:
    """(b, h, n, m) -> (b, n, m)."""
    if how == "mean":
        return w.mean(1)
    if how == "max":
        return w.amax(1)
    if how == "min":
        return w.amin(1)
    msg = f"unknown head fusion {how!r}"
    raise ValueError(msg)


def discard_low(a: Tensor, ratio: float) -> Tensor:
    """Zero the lowest ``ratio`` fraction of each row (at least one entry kept), renormalize."""
    if ratio <= 0:
        return a / a.sum(-1, keepdim=True)
    keep = max(1, a.shape[-1] - int(a.shape[-1] * ratio))
    idx = a.topk(keep, dim=-1).indices
    a = torch.zeros_like(a).scatter(-1, idx, a.gather(-1, idx))
    return a / a.sum(-1, keepdim=True).clamp_min(1e-12)


def rollout(
    cross: list[Tensor], self_: list[Tensor | None], *,
    head_fusion: str = "mean", discard_ratio: float = 0.0, discard_self: bool = False,
) -> Tensor:
    """Residual-augmented attention flow over per-head weights, (b, N_q, P); rows sum to 1."""
    eye = torch.eye(cross[0].shape[2], device=cross[0].device)
    r = None
    for c, s in zip(cross, self_, strict=True):
        c = discard_low(fuse_heads(c, head_fusion), discard_ratio)
        r = c if r is None else 0.5 * r + 0.5 * c
        if s is not None:
            s = fuse_heads(s, head_fusion)
            s = discard_low(s, discard_ratio) if discard_self else s
            st = 0.5 * s + 0.5 * eye
            r = (st / st.sum(-1, keepdim=True)) @ r
        r = r / r.sum(-1, keepdim=True)
    return r


def row_mass(w: Tensor, rows: tuple[int, ...], grid: int) -> Tensor:
    return w.reshape(*w.shape[:-1], grid, grid)[..., list(rows), :].sum((-1, -2))


@torch.no_grad()
def process_batch(model: Any, batch: dict, args: argparse.Namespace) -> list[dict]:
    from rmind.components.objectives.base import ObjectivePredictionKey  # noqa: PLC0415

    policy = model.objectives["policy"]
    episode = model.episode_builder(batch)
    embedding = model.encoder(src=episode.embeddings_flattened, mask=episode.attention_mask)
    latent = model.latent(episode=episode, embedding=embedding)
    e = len(episode.index)

    queries, key, value = policy_qkv(policy, episode, latent)
    cross, self_ = decoder_attention(policy, queries, key, value)
    ro = policy.readout
    layers_all = [c.mean(1) for c in cross]  # raw head-mean, (b, N_q, P) each
    layers = [w[:, ro] for w in layers_all]  # readout, (b, P) each
    roll_all = rollout(
        cross, self_, head_fusion=args.head_fusion,
        discard_ratio=args.discard_ratio, discard_self=args.discard_self,
    )  # every query, (b, N_q, P)
    roll = roll_all[:, ro]  # (b, P)
    p = roll.shape[-1]
    grid = math.isqrt(p)

    pred = policy.predict(
        episode=episode, embedding=embedding, latent=latent,
        keys=frozenset({ObjectivePredictionKey.PREDICTION_VALUE}),
        tokenizers=model.episode_builder.tokenizers,
    )[ObjectivePredictionKey.PREDICTION_VALUE].value

    ent_all = -(roll_all.clamp_min(1e-12) * roll_all.clamp_min(1e-12).log()).sum(-1) / math.log(p)
    ent = ent_all[:, ro]
    topv, topi = roll.topk(TOP_K, dim=-1)
    sq = grid * grid == p
    data, meta = batch["data"], batch["meta"]
    input_ids = meta["input_id"]
    cpu = lambda t: t.detach().float().cpu().numpy()  # noqa: E731

    rows = []
    for i in range(roll.shape[0]):
        row = {
            "input_id": input_ids[i] if isinstance(input_ids, (list, tuple)) else str(input_ids),
            "frame_idx": int(data[FRAME_IDX][i, e - 1]),
            "time_stamp": data[TIME_STAMP][i, e - 1].item() if torch.is_tensor(data[TIME_STAMP]) else None,
            **{name: float(data[k][i, e - 1]) for name, k in OBS_KEYS.items()},
            **{f"pred_{c}": cpu(pred["continuous", c][i]).tolist() for c in PRED_CHANNELS},
            "pred_turn_signal": pred["discrete", "turn_signal"][i].long().cpu().tolist()
            if ("discrete", "turn_signal") in pred.keys(include_nested=True) else None,
            "attn_rollout": cpu(roll[i]).tolist(),
            **{f"attn_l{k}": cpu(w[i]).tolist() for k, w in enumerate(layers)},
            "entropy_rollout": float(ent[i]),
            # every policy query (readout = index `readout_query`), (N_q, P) as nested lists
            "readout_query": int(ro),
            "attn_rollout_q": cpu(roll_all[i]).tolist(),
            **{f"attn_l{k}_q": cpu(w[i]).tolist() for k, w in enumerate(layers_all)},
            "entropy_rollout_q": cpu(ent_all[i]).tolist(),
            "top_patches": topi[i].tolist(),
            "top_weights": cpu(topv[i]).tolist(),
        }
        if sq:
            row["mass_sky"] = float(row_mass(roll[i], SKY_ROWS, grid))
            row["mass_hood"] = float(row_mass(roll[i], HOOD_ROWS, grid))
        rows.append(row)
    return rows, data[CAMERA][:, e - 1]


def turbo(x: np.ndarray) -> np.ndarray:
    """Turbo colormap (polynomial fit, Mikhailov 2019) for x in [0, 1] -> uint8 RGB."""
    x = np.clip(x, 0.0, 1.0)[..., None] ** np.arange(6)
    c = np.array([
        [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
        [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
        [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
    ])
    return (np.clip(x @ c.T, 0, 1) * 255).astype(np.uint8)


def model_view(img: np.ndarray) -> np.ndarray:
    """The region the model sees: center-crop to CROP_HW (it is then squashed to 256x256)."""
    h, w = img.shape[:2]
    ch, cw = min(CROP_HW[0], h), min(CROP_HW[1], w)
    top, left = (h - ch) // 2, (w - cw) // 2
    return img[top : top + ch, left : left + cw]


def overlay(view: np.ndarray, w: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Heatmap of patch weights w (P,) stretched over the model view (patch grid -> crop)."""
    grid = math.isqrt(w.size)
    h, wd = view.shape[:2]
    s = w.reshape(grid, grid)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    rows = (np.arange(h) * grid // h)[:, None]
    cols = (np.arange(wd) * grid // wd)[None, :]
    heat = turbo(s[rows, cols])
    return (view * (1 - alpha) + heat * alpha).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 8 + 7 * len(text), 16), fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def fit(img: np.ndarray, width: int) -> np.ndarray:
    h = round(img.shape[0] * width / img.shape[1])
    return np.asarray(Image.fromarray(img).resize((width, h), Image.BILINEAR))


TS_NAMES = {0: "off", 1: "left", 2: "right"}


def render_frame(row: dict, frame: np.ndarray, n_layers: int) -> np.ndarray:
    """[model view | readout rollout] / [readout per layer] / [rollout per query] / status strip."""
    view = model_view(frame)
    top = np.concatenate([label(view, "model view"),
                          label(overlay(view, np.asarray(row["attn_rollout"])), "readout rollout")], 1)
    width = top.shape[1]
    thumb_w = width // n_layers
    thumbs = [label(fit(overlay(view, np.asarray(row[f"attn_l{k}"])), thumb_w), f"layer {k}")
              for k in range(n_layers)]
    mid = np.concatenate(thumbs, 1)
    mid = np.pad(mid, ((0, 0), (0, width - mid.shape[1]), (0, 0)))
    # every policy query's rollout, QUERIES_PER_ROW per row; the readout is the one decoded
    q_maps = row["attn_rollout_q"]
    q_w = width // QUERIES_PER_ROW
    q_thumbs = [
        label(fit(overlay(view, np.asarray(w)), q_w),
              f"q{j}{' readout' if j == row['readout_query'] else ''} H={row['entropy_rollout_q'][j]:.2f}")
        for j, w in enumerate(q_maps)
    ]
    q_thumbs += [np.zeros_like(q_thumbs[0])] * (-len(q_thumbs) % QUERIES_PER_ROW)
    q_rows = [np.concatenate(q_thumbs[r : r + QUERIES_PER_ROW], 1)
              for r in range(0, len(q_thumbs), QUERIES_PER_ROW)]
    queries = np.concatenate(q_rows, 0)
    queries = np.pad(queries, ((0, 0), (0, width - queries.shape[1]), (0, 0)))
    pred = lambda c: np.mean(row[f"pred_{c}"])  # noqa: E731
    pts = row.get("pred_turn_signal")
    lines = [
        f"{row['input_id']}  frame {row['frame_idx']}  speed {row['speed']:.1f}  "
        f"gas {row['gas']:.2f}  brake {row['brake']:.2f}  steer {row['steering']:+.3f}  "
        f"signal {TS_NAMES.get(int(row['turn_signal']), row['turn_signal'])}",
        f"pred (chunk mean)  gas {pred('gas_pedal'):.2f}  brake {pred('brake_pedal'):.2f}  "
        f"steer {pred('steering_angle'):+.3f}  signal {TS_NAMES.get(max(set(pts), key=pts.count), '?') if pts else '-'}"
        f"   |  entropy {row['entropy_rollout']:.3f}"
        + (f"  sky {row['mass_sky']:.2f}  hood {row['mass_hood']:.2f}" if "mass_sky" in row else ""),
    ]
    strip = Image.new("RGB", (width, 40), (20, 20, 20))
    d = ImageDraw.Draw(strip)
    for j, t in enumerate(lines):
        d.text((6, 4 + 17 * j), t, fill=(235, 235, 235))
    out = np.concatenate([top, mid, queries, np.asarray(strip)], 0)
    return np.pad(out, ((0, out.shape[0] % 2), (0, out.shape[1] % 2), (0, 0)))  # even dims for yuv420p


class VideoWriter:
    """Raw RGB frames -> H.264 mp4 through the system ffmpeg (no Python video deps)."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path, self.fps, self.proc = path, fps, None

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None:
            h, w = frame.shape[:2]
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(self.path)],
                stdin=subprocess.PIPE,
            )
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


def run_drive(model: Any, cfg: Any, drive_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict | None:
    slug = f"{drive_dir.parent.name}__{drive_dir.name}"
    ddir = out_dir / slug
    ddir.mkdir(parents=True, exist_ok=True)
    cfg.datamodule.predict.batch_size = args.batch_size
    loader = instantiate(cfg.datamodule.predict)

    n_layers = len(model.objectives["policy"].decoder.decoder.layers)
    video = VideoWriter(ddir / "attention.mp4", args.fps)

    all_rows: list[dict] = []
    for b_idx, batch in enumerate(loader):
        if args.max_batches is not None and b_idx >= args.max_batches:
            break
        batch = to_device(batch, args.device)
        rows, frames = process_batch(model, batch, args)
        for row in rows:  # predict_drive's input_id is the constant "drive"
            row["input_id"] = slug
        all_rows.extend(rows)
        for row, frame in zip(rows, frames, strict=True):
            img = frame.detach().cpu().numpy()
            if img.shape[0] == 3:  # CHW -> HWC
                img = img.transpose(1, 2, 0)
            video.write(render_frame(row, img.astype(np.uint8), n_layers))
        if b_idx % 20 == 0:
            logger.info("batch", drive=slug, batch=b_idx, windows=len(all_rows))
        del batch, frames

    video.close()
    if not all_rows:
        logger.warning("no windows", drive=slug)
        return None
    df = pl.DataFrame(all_rows).sort("frame_idx")
    df.write_parquet(ddir / "attention.parquet")
    roll = np.asarray(df["attn_rollout"].to_list(), dtype=np.float32)
    grid = math.isqrt(roll.shape[1])
    np.save(ddir / "mean_rollout.npy", roll.mean(0).reshape(grid, grid))
    summary = {
        "drive": slug, "windows": len(df),
        "head_fusion": args.head_fusion, "discard_ratio": args.discard_ratio,
        "entropy_mean": df["entropy_rollout"].mean(),
        **({"mass_sky_mean": df["mass_sky"].mean(), "mass_hood_mean": df["mass_hood"].mean()}
           if "mass_sky" in df.columns else {}),
    }
    logger.info("drive done", **summary, video=str(ddir / "attention.mp4"))
    return summary


def to_device(x: Any, device: str) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drives", nargs="*", type=Path, help="drive directories (<data>/<vehicle>/<drive>)")
    ap.add_argument("--drives-file", type=Path, help="text file, one drive directory per line")
    ap.add_argument("--artifact", default="yaak/rmind/model-yh6qrp22:v9")
    ap.add_argument("--ckpt", type=Path, help="local model.ckpt (skips the W&B download)")
    ap.add_argument("--out", type=Path, default=Path("outputs/policy_attention"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".rbyte_cache/policy_attention"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, help="per drive; for quick checks")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--head-fusion", choices=("mean", "max", "min"), default="mean",
                    help="how heads are combined in the rollout (vit-explain)")
    ap.add_argument("--discard-ratio", type=float, default=0.9,
                    help="rollout: drop this fraction of the lowest cross-attention weights per query "
                         "per layer, then renormalize (vit-explain default 0.9; 0 = paper rollout)")
    ap.add_argument("--discard-self", action="store_true",
                    help="also discard in the query self-attention (off: only N_q queries there)")
    ap.add_argument("--fps", type=float, default=3.0, help="video fps; windows are ~3 Hz, so 3 = real time")
    args = ap.parse_args()
    drives = list(args.drives)
    if args.drives_file:
        drives += [Path(s.strip()) for s in args.drives_file.read_text().splitlines() if s.strip() and not s.startswith("#")]
    if not drives:
        ap.error("no drives given")
    args.drives = [d.resolve() for d in drives]
    return args


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = None
    summaries = []
    for drive_dir in args.drives:
        missing = [f for f in ("metadata.log", "waypoints.json", "headings_denoised.json",
                               f"frames/{CAMERA}.pii.mp4/576x324") if not (drive_dir / f).exists()]
        if missing:
            logger.warning("skipping drive, missing inputs", drive=str(drive_dir), missing=missing)
            continue
        cfg = compose_cfg(drive_dir, args.artifact, args.cache_dir.resolve())
        if model is None:
            model = load_model(cfg, args.ckpt, args.device)
            policy = model.objectives["policy"]
            logger.info("model loaded", artifact=args.ckpt or args.artifact,
                        latent_decoder=policy.latent_decoder is not None,
                        key_source=policy.key_source, readout=policy.readout,
                        layers=len(policy.decoder.decoder.layers))
        if (s := run_drive(model, cfg, drive_dir, args.out, args)) is not None:
            summaries.append(s)
    if summaries:
        pl.DataFrame(summaries).write_parquet(args.out / "summary.parquet")
        print(pl.DataFrame(summaries))


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
