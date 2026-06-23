"""Standalone residual-VQ tokenizer for waypoints (VQ-BeT style).

Applies the same residual vector-quantizer idea as `rmind.models.action_tokenizer`
(https://arxiv.org/pdf/2403.03181) to the future-path *waypoints* instead of the
ego action chunk. Each timestep's waypoints are a ``[10, 2]`` ego-normalized path;
we flatten to a 20-d vector, encode -> residual-VQ -> decode, and train the
autoencoder with an L1 reconstruction loss + the VQ commitment loss.

It then visualizes what the learned codebooks represent:

  1. per-quantizer code sweep   -- decode every codebook entry of each residual
                                   level (others held at their modal code) and
                                   overlay the resulting paths: coarse -> fine.
  2. reconstruction overlay     -- real paths vs their tokenized reconstruction.
  3. code usage / perplexity    -- occupancy histogram per quantizer.

Data is pulled through the existing rbyte datamodule (`datamodule=yaak/train`),
so waypoints get the exact ego-centering / heading-rotation / 1e-2 scaling the
real model sees. The extracted waypoint buffer is cached to disk so re-runs skip
the (slow) first-time dataset build.

Usage:
    just generate-config            # once, to materialize config/dataset/*
    uv run python scripts/waypoint_tokenizer.py --drives 6 --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from structlog import get_logger
from torch import Tensor, nn
from torch.nn import functional as F

import rmind  # noqa: F401  registers the "eval" OmegaConf resolver
from rmind.components.vq import ResidualVQ

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from omegaconf import DictConfig

logger = get_logger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")
NUM_WAYPOINTS = 10
WAYPOINT_DIM = 2
INPUT_DIM = NUM_WAYPOINTS * WAYPOINT_DIM  # flattened path
WAYPOINTS_KEY = ("data", "waypoints/xy_normalized")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _trim_drives(ds_cfg: DictConfig, k: int) -> list[str]:
    """Keep only the first `k` drives so the first dataset build stays tractable."""
    drives = list(ds_cfg.samples.inputs.input_id)[:k]
    keep = set(drives)
    for key in list(ds_cfg.samples.inputs.keys()):
        ds_cfg.samples.inputs[key] = list(ds_cfg.samples.inputs[key])[:k]
    sources = ds_cfg.streams.cam_front_left.sources
    for drive in list(sources.keys()):
        if drive not in keep:
            del sources[drive]
    return drives


def make_synthetic_waypoints(n: int) -> Tensor:
    """Generate `[n, 10, 2]` synthetic ego-frame paths (arcs of varying curvature).

    A fast, dependency-free stand-in for the real dataset: each path is a constant-
    curvature arc starting at the ego origin and extending forward (+y), with a
    multimodal mix of straight / left / right turns and varying length. Useful for
    smoke-testing the tokenizer and visualizations without the slow dataset build.
    """
    g = torch.Generator().manual_seed(0)
    # mixture of driving modes: (curvature mean, weight)
    modes = torch.tensor([-0.020, -0.008, 0.0, 0.0, 0.008, 0.020])
    pick = modes[torch.randint(len(modes), (n,), generator=g)]
    kappa = pick + 0.002 * torch.randn(n, generator=g)  # signed curvature [1/m]
    floor = 1e-5  # keep curvature away from exactly straight (avoids 1/kappa blow-up)
    kappa = torch.where(kappa.abs() < floor, torch.full_like(kappa, floor), kappa)
    length = 30.0 + 70.0 * torch.rand(n, generator=g)  # path length [m]
    s = torch.linspace(0, 1, NUM_WAYPOINTS).unsqueeze(0) * length.unsqueeze(1)  # (n,10)
    # constant-curvature arc from the origin heading +y; radius = 1/kappa (signed)
    r = (1.0 / kappa).unsqueeze(1)
    theta = s * kappa.unsqueeze(1)
    x = r * (1 - torch.cos(theta))
    y = r * torch.sin(theta)
    path = torch.stack([x, y], dim=-1) / 100.0  # to normalized units (~/100)
    path += 0.003 * torch.randn(path.shape, generator=g)
    return path


def load_waypoints(
    *, drives: int, max_samples: int, cache: Path, synthetic: int = 0
) -> Tensor:
    """Return a `[N, 10, 2]` buffer of ego-normalized waypoint paths."""
    if synthetic:
        logger.info("using synthetic waypoints", n=synthetic)
        return make_synthetic_waypoints(synthetic)

    if cache.exists():
        buf = torch.load(cache)
        logger.info("loaded cached waypoints", n=buf.shape[0], cache=str(cache))
        return buf

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=yaak/control_transformer/pretrain",
                "datamodule=yaak/train",
            ],
        )
    OmegaConf.set_struct(cfg, False)  # noqa: FBT003  allow trimming drive lists below

    dm_cfg = cfg.datamodule
    selected = _trim_drives(dm_cfg.train.dataset, drives)
    logger.info("building dataset", drives=selected)
    dm_cfg.train.batch_size = 64
    dm_cfg.train.num_workers = 2
    dm_cfg.train.pin_memory = False  # pin_memory probes CUDA; we only extract to CPU
    if "val" in dm_cfg:
        del dm_cfg.val  # we only need the train split; skip the separate val build

    datamodule = instantiate(dm_cfg)
    loader = datamodule.train_dataloader()

    collected: list[Tensor] = []
    total = 0
    for batch in loader:
        wpts = batch[WAYPOINTS_KEY[0]][WAYPOINTS_KEY[1]]  # (b, t, 10, 2)
        wpts = wpts.reshape(-1, NUM_WAYPOINTS, WAYPOINT_DIM).float().cpu()
        collected.append(wpts)
        total += wpts.shape[0]
        if total >= max_samples:
            break

    buf = torch.cat(collected, dim=0)[:max_samples].contiguous()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buf, cache)
    logger.info("cached waypoints", n=buf.shape[0], cache=str(cache))
    return buf


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class WaypointTokenizer(nn.Module):
    """Residual-VQ autoencoder over flattened `[10, 2]` waypoint paths."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        codebook_size: int = 16,
        num_quantizers: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quantizer = ResidualVQ(
            dim=latent_dim, codebook_size=codebook_size, num_quantizers=num_quantizers
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, INPUT_DIM),
        )
        # standardization stats, filled by `fit_normalizer`
        self.register_buffer("mean", torch.zeros(INPUT_DIM))
        self.register_buffer("std", torch.ones(INPUT_DIM))

    def fit_normalizer(self, paths: Tensor) -> None:
        flat = paths.reshape(-1, INPUT_DIM)
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp_min(1e-6))

    def _norm(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def _denorm(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def step(self, paths: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        x = self._norm(paths.reshape(-1, INPUT_DIM))
        z = self.encoder(x)
        codes, z_q, vq = self.quantizer(z)
        x_hat = self.decoder(z + (z_q - z).detach())  # straight-through
        recon = F.l1_loss(x_hat, x)
        return recon, {"codes": codes, "commit": vq["commit"]}

    @torch.no_grad()
    def encode(self, paths: Tensor) -> Tensor:
        z = self.encoder(self._norm(paths.reshape(-1, INPUT_DIM)))
        codes, _, _ = self.quantizer(z)
        return codes

    @torch.no_grad()
    def decode(self, codes: Tensor) -> Tensor:
        z_q = self.quantizer.lookup(codes)
        x = self._denorm(self.decoder(z_q))
        return x.reshape(*codes.shape[:-1], NUM_WAYPOINTS, WAYPOINT_DIM)


def train(  # noqa: PLR0913
    model: WaypointTokenizer,
    paths: Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    vq_weight: float,
    device: str,
) -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    n = paths.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        data = paths.to(device)[perm]
        recon_sum = commit_sum = 0.0
        steps = 0
        for i in range(0, n - batch_size + 1, batch_size):
            chunk = data[i : i + batch_size]
            recon, aux = model.step(chunk)
            loss = recon + vq_weight * aux["commit"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            recon_sum += recon.item()
            commit_sum += aux["commit"].item()
            steps += 1
        model.eval()  # don't let the diagnostic encode update the VQ's EMA codebooks
        perp = model.quantizer.perplexity(model.encode(data))
        model.train()
        logger.info(
            "epoch",
            n=f"{epoch + 1}/{epochs}",
            recon=round(recon_sum / steps, 4),
            commit=round(commit_sum / steps, 4),
            perplexity=[round(p, 1) for p in perp.tolist()],
        )


# --------------------------------------------------------------------------- #
# visualizations
# --------------------------------------------------------------------------- #
def _modal_codes(model: WaypointTokenizer, paths: Tensor, device: str) -> Tensor:
    codes = model.encode(paths.to(device))  # (N, G)
    g = codes.shape[-1]
    modal = torch.stack([codes[:, q].mode().values for q in range(g)])
    return modal.to(device)  # (G,)


def _plot_path(ax: Axes, path: Tensor, **kw: object) -> None:
    path = path.cpu()
    ax.plot(path[:, 0], path[:, 1], marker="o", markersize=3, **kw)


def viz_code_sweep(
    model: WaypointTokenizer, base: Tensor, out: Path, device: str
) -> None:
    g = model.quantizer.num_quantizers
    c = model.quantizer.codebook_size
    fig, axes = plt.subplots(1, g, figsize=(4 * g, 4), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for q in range(g):
        ax = axes[0][q]
        codes = base.repeat(c, 1)  # (C, G)
        codes[:, q] = torch.arange(c, device=device)
        paths = model.decode(codes)  # (C, 10, 2)
        for v in range(c):
            _plot_path(ax, paths[v], color=cmap(v / max(c - 1, 1)), alpha=0.8)
        ax.plot(0, 0, "r*", markersize=12, zorder=10)
        ax.set_title(f"quantizer {q}: sweep all {c} codes")
        ax.set_aspect("equal")
        ax.grid(visible=True, alpha=0.3)
    fig.suptitle("Per-quantizer codebook sweep (others held at modal code)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote figure", path=str(out))


def viz_reconstruction(
    model: WaypointTokenizer, paths: Tensor, out: Path, device: str, k: int = 16
) -> None:
    idx = torch.randperm(paths.shape[0])[:k]
    sample = paths[idx].to(device)
    codes = model.encode(sample)
    recon = model.decode(codes)
    cols = 4
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for j in range(rows * cols):
        ax = axes[j // cols][j % cols]
        if j >= k:
            ax.axis("off")
            continue
        _plot_path(ax, sample[j], color="black", label="original")
        _plot_path(ax, recon[j], color="deeppink", linestyle="--", label="recon")
        ax.plot(0, 0, "b*", markersize=10)
        ax.set_aspect("equal")
        ax.grid(visible=True, alpha=0.3)
        code_str = ",".join(str(int(x)) for x in codes[j].tolist())
        ax.set_title(f"codes=[{code_str}]", fontsize=8)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Waypoint reconstruction: original vs tokenized")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote figure", path=str(out))


def viz_code_usage(
    model: WaypointTokenizer, paths: Tensor, out: Path, device: str
) -> None:
    codes = model.encode(paths.to(device))  # (N, G)
    g = model.quantizer.num_quantizers
    c = model.quantizer.codebook_size
    perp = model.quantizer.perplexity(codes)
    fig, axes = plt.subplots(1, g, figsize=(4 * g, 3.5), squeeze=False)
    for q in range(g):
        ax = axes[0][q]
        counts = torch.bincount(codes[:, q].cpu(), minlength=c).float()
        counts /= counts.sum().clamp_min(1.0)
        ax.bar(range(c), counts.numpy(), color="steelblue")
        ax.set_title(f"quantizer {q}  (perplexity={perp[q]:.1f}/{c})")
        ax.set_xlabel("code index")
        ax.set_ylabel("frequency")
    fig.suptitle("Codebook usage per residual quantizer")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote figure", path=str(out))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drives", type=int, default=6)
    p.add_argument(
        "--synthetic",
        type=int,
        default=0,
        metavar="N",
        help="skip the dataset and train on N synthetic arc paths (smoke test / no-data fallback)",
    )
    p.add_argument("--max-samples", type=int, default=200_000)
    p.add_argument("--num-quantizers", type=int, default=4)
    p.add_argument("--codebook-size", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vq-weight", type=float, default=1.0)
    p.add_argument("--cache", type=Path, default=Path("artifacts/waypoints_buffer.pt"))
    p.add_argument("--out", type=Path, default=Path("artifacts/waypoint_tokenizer"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(1337)
    args.out.mkdir(parents=True, exist_ok=True)

    paths = load_waypoints(
        drives=args.drives,
        max_samples=args.max_samples,
        cache=args.cache,
        synthetic=args.synthetic,
    )

    model = WaypointTokenizer(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
    )
    model.fit_normalizer(paths)

    train(
        model,
        paths,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        vq_weight=args.vq_weight,
        device=args.device,
    )

    ckpt = args.out / "waypoint_tokenizer.pt"
    torch.save(model.state_dict(), ckpt)

    model.eval()
    base = _modal_codes(model, paths, args.device)
    viz_code_sweep(model, base, args.out / "code_sweep.png", args.device)
    viz_reconstruction(model, paths, args.out / "reconstruction.png", args.device)
    viz_code_usage(model, paths, args.out / "code_usage.png", args.device)


if __name__ == "__main__":
    main()
