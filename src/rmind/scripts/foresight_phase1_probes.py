"""Phase-1 frozen-feature probes for the foresight multimodality investigation.

Trains FD-head-architecture probes on the GRID CACHE (see schema contract in
foresight_mm/notes/phase1.md): context = encoder outputs at the foresight slots
of timestep 5 (UNpooled, (n,256,384)) + action_summary; targets = ground-truth
raw DINO patch grids of clip frames 6/8/10 (horizons H=1/3/5).

Subcommands
-----------
p1     Oracle-bits probe: baseline head vs head conditioned on the TRUE target's
       MiniBatchKMeans cluster id (M modes). The relative loss drop upper-bounds
       the mode-averaging tax recoverable with log2(M) oracle bits.
       -> foresight_mm/results/p1_h{H}_m{M}.json
p2     K-head trajectory-WTA probe: K learned mode embeddings + horizon
       embeddings, evolving winner-take-all (top-j annealed 8->4->2->1) with a
       TRAJECTORY-level winner (argmin over k of the loss MEAN over horizons,
       shared across horizons - the load-bearing design).
       -> foresight_mm/results/p2_k{K}.json
smoke  End-to-end self-test on a tiny synthetic grid cache with planted
       bimodality (no real caches touched).

Probe head (mirrors ForwardDynamicsPredictionObjective.compute_metrics,
src/rmind/components/objectives/forward_dynamics.py L49-87):
  query   = mask_vec tiled to (b,256,384) [+ condition embedding] + patch_pos_embed
  context = Linear_{768->384}(cat(ctx_foresight_grid, action_summary bcast)) (b,256,384)
  pred    = CrossAttentionDecoderHead({query, context})  (3D input path)
All probe modules are initialized from the PT checkpoint's FD objective weights
(loaded once, map_location=cpu) and are trainable.

Loss: local per-sample GramAnchoringLoss variant (math copied from
src/rmind/components/loss.py L101-132; per-sample sim/gram exposed before the
final batch means; weight_sim=1, weight_gram=100, patches=256). Parity vs the
stock loss is asserted at startup (<1e-5).

Usage (real caches, run from repo root):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \
      src/rmind/scripts/foresight_phase1_probes.py p1 --horizon 5 --n-modes 16
  ... p1 --horizon 1 --n-modes 16
  ... p2 --k 8 --horizons 1,3,5 \
      --p1-json foresight_mm/results/p1_h1_m16.json foresight_mm/results/p1_h5_m16.json
Smoke:
  uv run python src/rmind/scripts/foresight_phase1_probes.py smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = REPO_ROOT / "artifacts" / "model-t5cmm8om_latest" / "model.ckpt"
DEFAULT_TRAIN_CACHE = REPO_ROOT / "foresight_mm" / "cache" / "train096101_g"
DEFAULT_VAL_CACHE = REPO_ROOT / "foresight_mm" / "cache" / "val_g"
DEFAULT_ENTROPY = REPO_ROOT / "foresight_mm" / "cache" / "val_entropy.pt"
DEFAULT_RESULTS = REPO_ROOT / "foresight_mm" / "results"

N_PATCHES = 256
D = 384
WEIGHT_SIM = 1.0
WEIGHT_GRAM = 100.0
SHARD_SIZE = 2048  # grid-cache schema contract
HORIZON_TO_GRID_IDX = {1: 0, 3: 1, 5: 2}  # gt_grid stores clip frames 6, 8, 10
WTA_EPS = 0.05

CACHE_KEYS = ["ctx_foresight_grid", "ctx_action_summary", "gt_grid", "gt_pooled", "sample_id"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Tensor):
        obj = obj.detach().cpu()
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# --------------------------------------------------------------------------- #
# per-sample GramAnchoringLoss
# --------------------------------------------------------------------------- #
def gram_loss_multi(
    pred: Tensor, target: Tensor, *, weight_sim: float = WEIGHT_SIM, weight_gram: float = WEIGHT_GRAM
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-sample GramAnchoringLoss for K predictions sharing one target.

    pred (b, K, p, d), target (b, p, d) -> (total, sim, gram), each (b, K).
    Math copied from rmind.components.loss.GramAnchoringLoss.forward (L101-132)
    with the final batch `.mean()`s removed; target-side quantities (uniqueness
    weights, gram_gt) computed once per sample and broadcast over K.
    """
    target = target.detach()
    eps = 1e-6
    p = target.shape[-2]

    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)

    # target-driven within-frame patch uniqueness weights (loss.py L113-118)
    frame_sim = torch.einsum("bpd,bqd->bpq", target_n, target_n).clamp_min(0.0)
    eye = torch.eye(p, dtype=torch.bool, device=target.device)
    weights = 1.0 / (frame_sim.masked_fill(eye, 0.0).sum(dim=-1) / (p - 1) + eps)
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)  # (b, p)

    # sim term (loss.py L120-121, per-sample: sum over patches, no batch mean)
    patch_loss = (pred - target[:, None]).pow(2).mean(dim=-1)  # (b, K, p)
    sim = (weights[:, None] * patch_loss).sum(dim=-1)  # (b, K)

    # gram term (loss.py L127-130, per-sample)
    gram_pred = torch.einsum("bkpd,bkqd->bkpq", pred_n, pred_n)
    gram_gt = torch.einsum("bpd,bqd->bpq", target_n, target_n)
    pair_weights = torch.einsum("bp,bq->bpq", weights, weights)
    gram = (pair_weights[:, None] * (gram_pred - gram_gt[:, None]).pow(2)).sum(dim=(-2, -1))

    return weight_sim * sim + weight_gram * gram, sim, gram


def gram_loss_per_sample(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(b, p, d) x (b, p, d) -> per-sample (total, sim, gram), each (b,)."""
    total, sim, gram = gram_loss_multi(pred[:, None], target)
    return total[:, 0], sim[:, 0], gram[:, 0]


def validate_loss_parity(device: torch.device) -> float:
    """Assert mean of per-sample losses == stock GramAnchoringLoss (<1e-5)."""
    from rmind.components.loss import GramAnchoringLoss

    g = torch.Generator().manual_seed(1234)
    b = 6
    x = torch.randn(b * N_PATCHES, D, generator=g).to(device)
    y = (torch.randn(b * N_PATCHES, D, generator=g) * 2.0 + 0.3).to(device)
    stock = GramAnchoringLoss(patches=N_PATCHES, weight_sim=WEIGHT_SIM, weight_gram=WEIGHT_GRAM)
    ref = stock(x, y).item()
    per, _, _ = gram_loss_per_sample(x.view(b, N_PATCHES, D), y.view(b, N_PATCHES, D))
    diff = abs(per.mean().item() - ref)
    assert diff < 1e-5, f"per-sample loss parity failed: |{per.mean().item()} - {ref}| = {diff}"
    print(f"[loss-parity] per-sample mean vs stock GramAnchoringLoss: diff={diff:.2e} OK", flush=True)
    return diff


# --------------------------------------------------------------------------- #
# FD probe head (PT-initialized replica of the FD objective's decode path)
# --------------------------------------------------------------------------- #
def load_fd_probe_init(ckpt_path: str | Path) -> dict[str, Any]:
    """Load the PT checkpoint ONCE (cpu) and extract the FD-head init tensors.

    Returns proj/head/patch_pos_embed state dicts + the (UTILITY,'mask')
    embedding VALUE as used by the FD objective: episode.embeddings.get(
    (UTILITY,'mask')) = projections.utility(embeddings.utility(0)) + role,
    where the role embedding is ZERO for utility because it is not in the
    timestep tuple (episode.py L229-240 falls back to torch.zeros_like).
    Verified below by recomputing through real nn modules vs manual matmul.
    """
    sd = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)["state_dict"]
    fd = "objectives.forward_dynamics."
    head_prefix = fd + "heads.foresight.cam_front_left."
    head_sd = {
        k.removeprefix(head_prefix): v.clone().float()
        for k, v in sd.items()
        if k.startswith(head_prefix)
    }
    assert len(head_sd) == 38, f"expected 38 FD head tensors, got {len(head_sd)}"
    proj_sd = {
        "weight": sd[fd + "projections.foresight.cam_front_left.0.weight"].clone().float(),
        "bias": sd[fd + "projections.foresight.cam_front_left.0.bias"].clone().float(),
    }
    assert proj_sd["weight"].shape == (D, 2 * D)
    ppe_sd = {
        "row_embed.weight": sd[fd + "patch_pos_embed.row_embed.weight"].clone().float(),
        "col_embed.weight": sd[fd + "patch_pos_embed.col_embed.weight"].clone().float(),
    }

    # (UTILITY,'mask') query-seed value, manual matmul
    e = sd["episode_builder.embeddings.utility.weight"][0].float()  # (384,)
    w = sd["episode_builder.projections.utility.weight"].float()
    b = sd["episode_builder.projections.utility.bias"].float()
    mask_vec = e @ w.T + b  # (384,)

    # verification: apply through the same module classes the episode builder
    # uses (Embedding -> Linear); role addition for utility is zeros.
    with torch.no_grad():
        emb = nn.Embedding(1, D)
        emb.weight.copy_(sd["episode_builder.embeddings.utility.weight"].float())
        lin = nn.Linear(D, D)
        lin.weight.copy_(w)
        lin.bias.copy_(b)
        mask_vec_mod = lin(emb(torch.tensor([0])))[0]
    diff = (mask_vec - mask_vec_mod).abs().max().item()
    assert diff < 1e-6, f"mask_vec module-vs-manual mismatch: {diff}"
    print(
        f"[init] FD probe init loaded from {ckpt_path}: head tensors={len(head_sd)}, "
        f"mask_vec norm={mask_vec.norm().item():.4f} (module parity {diff:.1e})",
        flush=True,
    )
    return {"head_sd": head_sd, "proj_sd": proj_sd, "ppe_sd": ppe_sd, "mask_vec": mask_vec}


class FDProbeHead(nn.Module):
    """FD-objective decode-path replica, PT-initialized, fully trainable."""

    def __init__(self, init: dict[str, Any]) -> None:
        super().__init__()
        from rmind.components.position_encoding import PatchPositionEmbedding2D
        from rmind.components.transformer.decoder import (
            CrossAttentionDecoder,
            CrossAttentionDecoderHead,
        )

        self.projection = nn.Linear(2 * D, D)
        self.projection.load_state_dict(init["proj_sd"])
        # config: raw.yaml L426-444 (dim 384, 2 layers, 4 heads, dropout 0.1)
        self.head = CrossAttentionDecoderHead(
            decoder=CrossAttentionDecoder(dim_model=D, num_layers=2, num_heads=4),
            output_projection=nn.Linear(D, D),
        )
        self.head.load_state_dict(init["head_sd"])
        self.patch_pos_embed = PatchPositionEmbedding2D(grid_size=(16, 16), embedding_dim=D)
        self.patch_pos_embed.load_state_dict(init["ppe_sd"])
        self.mask_vec = nn.Parameter(init["mask_vec"].clone().view(1, 1, D))

    def encode_context(self, ctx_grid: Tensor, action_summary: Tensor) -> Tensor:
        """(b,256,384) fp32 + (b,384) -> projected context (b,256,384).

        Mirrors forward_dynamics.py L63-68: per-patch concat
        [foresight_slot_out || action_summary_out] -> Linear(768->384).
        """
        b, p, d = ctx_grid.shape
        feats = torch.cat([ctx_grid, action_summary[:, None, :].expand(b, p, d)], dim=-1)
        return self.projection(feats)

    def decode(self, context: Tensor, cond: Tensor | None = None) -> Tensor:
        """context (B,256,384), cond (B,384)|None -> predicted patches (B,256,384).

        Mirrors forward_dynamics.py L69-87 with the 3D input path of
        CrossAttentionDecoderHead (decoder.py L141-142): query = mask token
        tiled to all patches [+ condition] + patch_pos_embed.
        """
        bsz = context.shape[0]
        query = self.mask_vec.expand(bsz, N_PATCHES, D)
        if cond is not None:
            query = query + cond[:, None, :]
        query = self.patch_pos_embed(query)
        return self.head({"query": query, "context": context})

    def forward(self, ctx_grid: Tensor, action_summary: Tensor, cond: Tensor | None = None) -> Tensor:
        return self.decode(self.encode_context(ctx_grid, action_summary), cond)


# --------------------------------------------------------------------------- #
# grid-cache streaming
# --------------------------------------------------------------------------- #
class GridCacheStream:
    """Shard-streaming reader over a grid cache (fp16 on disk -> fp32 on GPU).

    Shuffles shard order + within-shard order per epoch (seeded); yields dict
    batches of cpu fp16 tensors. Partial batches occur at shard boundaries
    (2048 % bs != 0) - acceptable for probe training, exact for eval.
    """

    def __init__(self, cache_dir: str | Path, *, batch_size: int, shuffle: bool, seed: int = 0) -> None:
        self.dir = Path(cache_dir)
        self.shards = sorted(self.dir.glob("shard_*.pt"))
        if not self.shards:
            msg = f"no shard_*.pt under {self.dir}"
            raise FileNotFoundError(msg)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.n_samples = int(meta["n_samples"])
        assert int(meta["n_shards"]) == len(self.shards), (
            f"meta n_shards={meta['n_shards']} != {len(self.shards)} files in {self.dir}"
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        # schema contract: shards of SHARD_SIZE, last shard = remainder
        if len(self.shards) == 1:
            self.shard_sizes = [self.n_samples]
        else:
            rest = self.n_samples - SHARD_SIZE * (len(self.shards) - 1)
            assert 0 < rest <= SHARD_SIZE, f"shard-size contract violated in {self.dir}"
            self.shard_sizes = [SHARD_SIZE] * (len(self.shards) - 1) + [rest]
        self.batches_per_epoch = sum(math.ceil(s / batch_size) for s in self.shard_sizes)

    def batches(self, epoch: int = 0):
        rng = np.random.default_rng(self.seed * 100_003 + epoch)
        shard_order = rng.permutation(len(self.shards)) if self.shuffle else np.arange(len(self.shards))
        for si in shard_order:
            shard = torch.load(self.shards[si], map_location="cpu", weights_only=False)
            n = shard["sample_id"].shape[0]
            idx = torch.from_numpy(rng.permutation(n)) if self.shuffle else torch.arange(n)
            for i0 in range(0, n, self.batch_size):
                sel = idx[i0 : i0 + self.batch_size]
                yield {k: shard[k][sel] for k in CACHE_KEYS}
            del shard


def load_keys_all_shards(cache_dir: str | Path, keys: list[str]) -> dict[str, Tensor]:
    """Concat small keys across all shards (loads each shard once, drops grids)."""
    shards = sorted(Path(cache_dir).glob("shard_*.pt"))
    parts: dict[str, list[Tensor]] = {k: [] for k in keys}
    for s in shards:
        shard = torch.load(s, map_location="cpu", weights_only=False)
        for k in keys:
            parts[k].append(shard[k])
        del shard
    return {k: torch.cat(v) for k, v in parts.items()}


def load_entropy_sidecar(path: str | Path, val_sample_ids: Tensor) -> dict[str, Tensor]:
    """Join {quintile_h1, quintile_h5} onto val cache order via sample_id."""
    path = Path(path)
    if not path.exists():
        msg = (
            f"entropy sidecar {path} not found - required for per-quintile "
            "stratification (schema: foresight_mm/cache/val_entropy.pt)"
        )
        raise FileNotFoundError(msg)
    sc = torch.load(path, map_location="cpu", weights_only=False)
    lut = {int(s): i for i, s in enumerate(sc["sample_id"].tolist())}
    pos, missing = [], 0
    for s in val_sample_ids.tolist():
        i = lut.get(int(s), -1)
        pos.append(i)
        missing += i < 0
    pos_t = torch.tensor(pos, dtype=torch.long)
    out = {}
    for key in ("quintile_h1", "quintile_h5"):
        q = sc[key].to(torch.long)[pos_t.clamp_min(0)]
        q[pos_t < 0] = -1
        out[key] = q
    if missing:
        print(f"[entropy] WARNING: {missing}/{len(pos)} val samples missing from sidecar", flush=True)
    print(f"[entropy] joined sidecar {path} onto {len(pos)} val samples ({missing} missing)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# wandb (graceful fallback)
# --------------------------------------------------------------------------- #
class WandbLogger:
    def __init__(self, name: str, config: dict[str, Any], enabled: bool) -> None:
        self.run = None
        if not enabled:
            print("[wandb] disabled (--no-wandb)", flush=True)
            return
        try:
            import wandb

            self.run = wandb.init(
                entity="yaak",
                project="rmind",
                name=name,
                tags=["foresight_mm", "phase1"],
                job_type="probe",
                config=config,
            )
            print(f"[wandb] run: {self.run.url}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] init failed ({e!r}); falling back to JSON-only", flush=True)
            self.run = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=step)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] log failed ({e!r}); disabling", flush=True)
            self.run = None

    @property
    def url(self) -> str | None:
        return self.run.url if self.run is not None else None

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:  # noqa: BLE001, S110
                pass


# --------------------------------------------------------------------------- #
# shared training utilities
# --------------------------------------------------------------------------- #
def cosine_lambda(total_steps: int, floor: float = 0.1):
    def fn(step: int) -> float:
        t = min(step, total_steps) / max(total_steps, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))

    return fn


def assign_clusters(pooled: Tensor, centers: Tensor) -> Tensor:
    """(b,384) fp32, (M,384) fp32 -> (b,) int64 nearest-center id."""
    return torch.cdist(pooled, centers).argmin(dim=1)


def quintile_means(values: Tensor, quintiles: Tensor) -> dict[str, dict[str, float]]:
    out = {}
    for q in range(5):
        m = quintiles == q
        out[str(q)] = {
            "n": int(m.sum()),
            "mean": float(values[m].mean()) if m.any() else None,
        }
    return out


def fit_kmeans(pooled: np.ndarray, m: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=m, random_state=seed, batch_size=4096, n_init=10, max_no_improvement=50
    ).fit(pooled)
    sizes = np.bincount(km.labels_, minlength=m)
    info = {
        "inertia": float(km.inertia_),
        "cluster_sizes_train": sizes.tolist(),
        "n_iter": int(km.n_iter_),
    }
    return km.cluster_centers_.astype(np.float32), info


# --------------------------------------------------------------------------- #
# P1: oracle bits
# --------------------------------------------------------------------------- #
def run_p1(args: argparse.Namespace) -> dict[str, Any]:
    seed_all(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.set_float32_matmul_precision("high")
    validate_loss_parity(dev)

    horizon, m = args.horizon, args.n_modes
    gidx = HORIZON_TO_GRID_IDX[horizon]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"p1_h{horizon}_m{m}.json"

    init = load_fd_probe_init(args.ckpt)

    # --- kmeans on TRAIN gt_pooled at this horizon ------------------------- #
    train_small = load_keys_all_shards(args.train_cache, ["gt_pooled", "sample_id"])
    train_pooled = train_small["gt_pooled"][:, horizon - 1].float().numpy()
    centers_np, km_info = fit_kmeans(train_pooled, m, args.seed)
    centers = torch.from_numpy(centers_np).to(dev)
    centers_path = results_dir / f"p1_h{horizon}_m{m}_centers.pt"
    torch.save({"centers": torch.from_numpy(centers_np), "horizon": horizon, "seed": args.seed}, centers_path)
    print(f"[p1] kmeans M={m} on {len(train_pooled)} train gt_pooled[h={horizon}]: "
          f"inertia={km_info['inertia']:.1f} sizes={km_info['cluster_sizes_train']}", flush=True)

    # --- data ---------------------------------------------------------------- #
    train_stream = GridCacheStream(args.train_cache, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_stream = GridCacheStream(args.val_cache, batch_size=args.batch_size, shuffle=False)
    val_small = load_keys_all_shards(args.val_cache, ["sample_id"])
    quints = load_entropy_sidecar(args.entropy, val_small["sample_id"])
    qkey = f"quintile_h{horizon}"
    val_quint = quints[qkey]

    # --- heads (identical seeds/schedule; trained jointly on the same batches) #
    torch.manual_seed(args.seed)
    head_base = FDProbeHead(init).to(dev)
    torch.manual_seed(args.seed)
    head_orac = FDProbeHead(init).to(dev)
    cond_emb = nn.Embedding(m, D).to(dev)
    nn.init.zeros_(cond_emb.weight)

    params = (
        list(head_base.parameters()) + list(head_orac.parameters()) + list(cond_emb.parameters())
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * train_stream.batches_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(total_steps))

    wb = WandbLogger(
        f"phase1-p1-h{horizon}-m{m}" + ("-smoke" if args.smoke_tag else ""),
        {**{kk: str(v) if isinstance(v, Path) else v for kk, v in vars(args).items() if kk != "func"},
         "total_steps": total_steps, "n_train": train_stream.n_samples,
         "n_val": val_stream.n_samples},
        enabled=not args.no_wandb,
    )

    def _fwd(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        ctx = batch["ctx_foresight_grid"].to(dev, non_blocking=True).float()
        act = batch["ctx_action_summary"].to(dev, non_blocking=True).float()
        tgt = batch["gt_grid"][:, gidx].to(dev, non_blocking=True).float()
        pooled = batch["gt_pooled"][:, horizon - 1].to(dev, non_blocking=True).float()
        cid = assign_clusters(pooled, centers)
        pred_b = head_base(ctx, act)
        pred_o = head_orac(ctx, act, cond_emb(cid))
        loss_b, sim_b, gram_b = gram_loss_per_sample(pred_b, tgt)
        loss_o, sim_o, gram_o = gram_loss_per_sample(pred_o, tgt)
        del sim_b, gram_b, sim_o, gram_o
        return loss_b, loss_o, ctx, act, tgt

    @torch.no_grad()
    def _eval(epoch: int) -> dict[str, Any]:
        head_base.eval()
        head_orac.eval()
        lb, lo, sids = [], [], []
        for batch in val_stream.batches():
            loss_b, loss_o, *_ = _fwd(batch)
            lb.append(loss_b.cpu())
            lo.append(loss_o.cpu())
            sids.append(batch["sample_id"])
        lb_t, lo_t = torch.cat(lb), torch.cat(lo)
        sid_t = torch.cat(sids)
        assert torch.equal(sid_t, val_small["sample_id"]), "val order drifted"
        l_base, l_orac = float(lb_t.mean()), float(lo_t.mean())
        rel = (l_base - l_orac) / l_base if l_base != 0 else float("nan")
        per_q = {}
        for q in range(5):
            mask = val_quint == q
            if not mask.any():
                per_q[str(q)] = {"n": 0}
                continue
            qb, qo = float(lb_t[mask].mean()), float(lo_t[mask].mean())
            per_q[str(q)] = {
                "n": int(mask.sum()),
                "L_base": qb,
                "L_oracle": qo,
                "rel_drop": (qb - qo) / qb if qb != 0 else None,
            }
        res = {"epoch": epoch, "L_base": l_base, "L_oracle": l_orac, "rel_drop": rel, "per_quintile": per_q}
        res["_per_sample"] = {"L_base": lb_t, "L_oracle": lo_t}
        return res

    train_curve, val_curve = [], []
    step = 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        head_base.train()
        head_orac.train()
        ep_b, ep_o, nb = 0.0, 0.0, 0
        for batch in train_stream.batches(epoch):
            loss_b, loss_o, *_ = _fwd(batch)
            loss = loss_b.mean() + loss_o.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            lb_i = loss_b.detach().mean().item()
            lo_i = loss_o.detach().mean().item()
            ep_b += lb_i
            ep_o += lo_i
            nb += 1
            if step % args.log_every == 0:
                wb.log(
                    {"p1/train/loss_base": lb_i, "p1/train/loss_oracle": lo_i,
                     "p1/train/lr": sched.get_last_lr()[0]},
                    step=step,
                )
        train_curve.append({"epoch": epoch, "loss_base": ep_b / nb, "loss_oracle": ep_o / nb,
                            "lr": sched.get_last_lr()[0]})
        ev = _eval(epoch)
        ev.pop("_per_sample")
        val_curve.append(ev)
        wb.log({"p1/val/L_base": ev["L_base"], "p1/val/L_oracle": ev["L_oracle"],
                "p1/val/rel_drop": ev["rel_drop"], "epoch": epoch}, step=step)
        print(f"[p1 h={horizon}] epoch {epoch}: train base={train_curve[-1]['loss_base']:.4f} "
              f"oracle={train_curve[-1]['loss_oracle']:.4f} | val base={ev['L_base']:.4f} "
              f"oracle={ev['L_oracle']:.4f} rel_drop={ev['rel_drop']:.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    final = val_curve[-1]
    result = {
        "task": "p1",
        "horizon": horizon,
        "n_modes": m,
        "quintile_key": qkey,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
        "n_train": train_stream.n_samples,
        "n_val": val_stream.n_samples,
        "total_steps": total_steps,
        "kmeans": {**km_info, "centers_path": str(centers_path)},
        "train_curve": train_curve,
        "val_curve": val_curve,
        "final": final,
        "wandb_run": wb.url,
        "elapsed_s": time.perf_counter() - t0,
    }
    out_json.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
    print(f"[p1] wrote {out_json}", flush=True)
    wb.finish()
    return result


# --------------------------------------------------------------------------- #
# P2: K-head trajectory WTA
# --------------------------------------------------------------------------- #
def wta_j(step: int, total_steps: int, k: int) -> int:
    """Evolving-WTA top-j schedule: K -> K/2 -> K/4 -> 1 at 25/50/75% of steps."""
    stage = min(3, int(4 * step / max(total_steps, 1)))
    return [k, max(k // 2, 1), max(k // 4, 1), 1][stage]


def run_p2(args: argparse.Namespace) -> dict[str, Any]:
    seed_all(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.set_float32_matmul_precision("high")
    validate_loss_parity(dev)

    k = args.k
    horizons = [int(h) for h in args.horizons.split(",")]
    assert all(h in HORIZON_TO_GRID_IDX for h in horizons), f"horizons must be in {list(HORIZON_TO_GRID_IDX)}"
    gidxs = [HORIZON_TO_GRID_IDX[h] for h in horizons]
    n_h = len(horizons)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"p2_k{k}.json"

    init = load_fd_probe_init(args.ckpt)
    train_stream = GridCacheStream(args.train_cache, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_stream = GridCacheStream(args.val_cache, batch_size=args.batch_size, shuffle=False)
    val_small = load_keys_all_shards(args.val_cache, ["sample_id"])
    quints = load_entropy_sidecar(args.entropy, val_small["sample_id"])

    # P1 baseline references (optional)
    p1_ref: dict[str, Any] = {}
    for pth in args.p1_json or []:
        try:
            j = json.loads(Path(pth).read_text())
            p1_ref[f"h{j['horizon']}"] = {
                "L_base": j["final"]["L_base"],
                "L_oracle": j["final"]["L_oracle"],
                "per_quintile": j["final"]["per_quintile"],
                "path": str(pth),
            }
        except Exception as e:  # noqa: BLE001
            print(f"[p2] could not read p1 json {pth}: {e!r}", flush=True)

    torch.manual_seed(args.seed)
    head = FDProbeHead(init).to(dev)
    cond_emb = nn.Embedding(k, D).to(dev)
    nn.init.normal_(cond_emb.weight, std=0.02)
    hor_emb = nn.Embedding(n_h, D).to(dev)
    nn.init.zeros_(hor_emb.weight)

    params = list(head.parameters()) + list(cond_emb.parameters()) + list(hor_emb.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * train_stream.batches_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(total_steps))

    wb = WandbLogger(
        f"phase1-p2-k{k}" + ("-smoke" if args.smoke_tag else ""),
        {**{kk: str(v) if isinstance(v, Path) else v for kk, v in vars(args).items() if kk != "func"},
         "total_steps": total_steps, "n_train": train_stream.n_samples, "n_val": val_stream.n_samples},
        enabled=not args.no_wandb,
    )

    def _losses_kh(batch: dict[str, Tensor], *, want_pooled: bool = False):
        """Per-(sample, k, horizon) losses; horizons looped, (K) folded into batch.

        Returns loss_kh (b, K, n_h) and optionally pooled preds (b, K, n_h, D).
        """
        ctx = batch["ctx_foresight_grid"].to(dev, non_blocking=True).float()
        act = batch["ctx_action_summary"].to(dev, non_blocking=True).float()
        b = ctx.shape[0]
        context = head.encode_context(ctx, act)  # (b, p, d)
        context_k = context[:, None].expand(b, k, N_PATCHES, D).reshape(b * k, N_PATCHES, D)
        losses, pooled_preds = [], []
        for hi, gi in enumerate(gidxs):
            cond = (cond_emb.weight + hor_emb.weight[hi][None, :])[None].expand(b, k, D).reshape(b * k, D)
            pred = head.decode(context_k, cond)  # (b*k, p, d)
            tgt = batch["gt_grid"][:, gi].to(dev, non_blocking=True).float()
            total, _sim, _gram = gram_loss_multi(pred.view(b, k, N_PATCHES, D), tgt)
            losses.append(total)  # (b, k)
            if want_pooled:
                pooled_preds.append(pred.view(b, k, N_PATCHES, D).mean(dim=2))
        loss_kh = torch.stack(losses, dim=-1)  # (b, k, n_h)
        pooled = torch.stack(pooled_preds, dim=2) if want_pooled else None  # (b, k, n_h, d)
        return loss_kh, pooled

    @torch.no_grad()
    def _eval(epoch: int) -> dict[str, Any]:
        head.eval()
        acc = {"loss_kh": [], "pooled_min_ade": [], "diversity": [], "winner": [], "sid": []}
        for batch in val_stream.batches():
            loss_kh, pooled = _losses_kh(batch, want_pooled=True)
            gt_pooled = batch["gt_pooled"].to(dev).float()  # (b, 5, d), H=1..5
            gt_sel = torch.stack([gt_pooled[:, h - 1] for h in horizons], dim=1)  # (b, n_h, d)
            ade = (pooled - gt_sel[:, None]).norm(dim=-1)  # (b, k, n_h)
            pd_pair = torch.cdist(
                pooled.permute(0, 2, 1, 3).reshape(-1, k, D),
                pooled.permute(0, 2, 1, 3).reshape(-1, k, D),
            ).view(loss_kh.shape[0], n_h, k, k)
            div = pd_pair.sum(dim=(-2, -1)) / (k * (k - 1))  # (b, n_h) mean pairwise L2
            acc["loss_kh"].append(loss_kh.cpu())
            acc["pooled_min_ade"].append(ade.min(dim=1).values.cpu())
            acc["diversity"].append(div.cpu())
            acc["winner"].append(loss_kh.mean(dim=-1).argmin(dim=1).cpu())
            acc["sid"].append(batch["sample_id"])
        loss_kh = torch.cat(acc["loss_kh"])  # (n, k, n_h)
        min_ade = torch.cat(acc["pooled_min_ade"])  # (n, n_h)
        diversity = torch.cat(acc["diversity"])  # (n, n_h)
        winner = torch.cat(acc["winner"])  # (n,)
        assert torch.equal(torch.cat(acc["sid"]), val_small["sample_id"]), "val order drifted"

        n = loss_kh.shape[0]
        min_k = loss_kh.min(dim=1).values  # (n, n_h)
        mean_k = loss_kh.mean(dim=1)  # (n, n_h)
        win_rate = torch.bincount(winner, minlength=k).float() / n
        live_thresh = 1.0 / (4 * k)
        live_modes = int((win_rate > live_thresh).sum())

        res: dict[str, Any] = {
            "epoch": epoch,
            "minK_loss_per_h": {f"h{h}": float(min_k[:, i].mean()) for i, h in enumerate(horizons)},
            "meanK_loss_per_h": {f"h{h}": float(mean_k[:, i].mean()) for i, h in enumerate(horizons)},
            "min_ade_per_h": {f"h{h}": float(min_ade[:, i].mean()) for i, h in enumerate(horizons)},
            "diversity_per_h": {f"h{h}": float(diversity[:, i].mean()) for i, h in enumerate(horizons)},
            "win_rate": win_rate.tolist(),
            "live_modes": live_modes,
            "live_thresh": live_thresh,
        }
        # per-quintile stratification (quintile of matching horizon for the
        # h1/h5 loss comparisons; quintile_h5 for trajectory-level stats)
        for qk in ("quintile_h1", "quintile_h5"):
            qv = quints[qk]
            per_q = {}
            for q in range(5):
                mask = qv == q
                if not mask.any():
                    per_q[str(q)] = {"n": 0}
                    continue
                wq = torch.bincount(winner[mask], minlength=k).float() / int(mask.sum())
                per_q[str(q)] = {
                    "n": int(mask.sum()),
                    "minK_loss_per_h": {f"h{h}": float(min_k[mask, i].mean()) for i, h in enumerate(horizons)},
                    "meanK_loss_per_h": {f"h{h}": float(mean_k[mask, i].mean()) for i, h in enumerate(horizons)},
                    "min_ade_per_h": {f"h{h}": float(min_ade[mask, i].mean()) for i, h in enumerate(horizons)},
                    "diversity_per_h": {f"h{h}": float(diversity[mask, i].mean()) for i, h in enumerate(horizons)},
                    "live_modes": int((wq > live_thresh).sum()),
                    "win_rate": wq.tolist(),
                }
            res[f"per_{qk}"] = per_q
        return res

    train_curve, val_curve = [], []
    step = 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        head.train()
        ep_loss, ep_minh, nb = 0.0, 0.0, 0
        for batch in train_stream.batches(epoch):
            loss_kh, _ = _losses_kh(batch)
            traj = loss_kh.mean(dim=-1)  # (b, k) trajectory-level (shared across horizons)
            j = wta_j(step, total_steps, k)
            order = traj.argsort(dim=1)
            w = torch.full_like(traj, WTA_EPS)
            w.scatter_(1, order[:, :j], 1.0)
            # winner-weighted (sim AND gram identically: weights apply to the
            # combined per-(sample,k) loss which already includes the gram term)
            loss = ((w * traj).sum(dim=1) / w.sum(dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            l_i = loss.detach().item()
            mh_i = traj.detach().min(dim=1).values.mean().item()
            ep_loss += l_i
            ep_minh += mh_i
            nb += 1
            if step % args.log_every == 0:
                wb.log({"p2/train/wta_loss": l_i, "p2/train/minK_traj": mh_i,
                        "p2/train/j": j, "p2/train/lr": sched.get_last_lr()[0]}, step=step)
        train_curve.append({"epoch": epoch, "wta_loss": ep_loss / nb, "minK_traj": ep_minh / nb,
                            "j_end": wta_j(step - 1, total_steps, k), "lr": sched.get_last_lr()[0]})
        ev = _eval(epoch)
        val_curve.append(ev)
        wb.log({f"p2/val/minK_loss_{hk}": v for hk, v in ev["minK_loss_per_h"].items()}
               | {f"p2/val/min_ade_{hk}": v for hk, v in ev["min_ade_per_h"].items()}
               | {f"p2/val/diversity_{hk}": v for hk, v in ev["diversity_per_h"].items()}
               | {"p2/val/live_modes": ev["live_modes"], "epoch": epoch}, step=step)
        print(f"[p2 k={k}] epoch {epoch}: train wta={train_curve[-1]['wta_loss']:.4f} | "
              f"val minK={ev['minK_loss_per_h']} live_modes={ev['live_modes']} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    final = val_curve[-1]
    # gate-G1 recovery vs P1 (if references provided):
    # recovery(h) = (L_base - L_minK) / (L_base - L_oracle)
    recovery = {}
    for hk, ref in p1_ref.items():
        if hk in final["minK_loss_per_h"]:
            l_mink = final["minK_loss_per_h"][hk]
            gap = ref["L_base"] - ref["L_oracle"]
            recovery[hk] = {
                "p1_L_base": ref["L_base"],
                "p1_L_oracle": ref["L_oracle"],
                "p2_minK": l_mink,
                "oracle_gap": gap,
                "recovery_frac": (ref["L_base"] - l_mink) / gap if gap > 0 else None,
            }
    result = {
        "task": "p2",
        "k": k,
        "horizons": horizons,
        "args": {kk: str(v) if isinstance(v, (Path, list)) else v for kk, v in vars(args).items() if kk != "func"},
        "n_train": train_stream.n_samples,
        "n_val": val_stream.n_samples,
        "total_steps": total_steps,
        "wta": {"eps": WTA_EPS, "j_schedule": "K -> K/2 -> K/4 -> 1 at 25/50/75% of steps",
                "winner": "trajectory-level argmin over k of mean-over-horizons loss"},
        "p1_baseline": p1_ref,
        "gate_g1_recovery": recovery,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "final": final,
        "wandb_run": wb.url,
        "elapsed_s": time.perf_counter() - t0,
    }
    out_json.write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
    print(f"[p2] wrote {out_json}", flush=True)
    wb.finish()
    return result


# --------------------------------------------------------------------------- #
# smoke: synthetic grid cache + fake entropy sidecar + end-to-end p1/p2
# --------------------------------------------------------------------------- #
SYNTH_D_LAT = 4


def synth_params(seed: int) -> dict[str, Tensor]:
    """Generative matrices SHARED between the train and val synthetic caches
    (train/val must be draws from the SAME distribution)."""
    g = torch.Generator().manual_seed(seed)
    d_lat = SYNTH_D_LAT
    m0 = torch.randn(D, generator=g)
    return {
        "a_ctx": torch.randn(d_lat, D, generator=g) / d_lat**0.5,
        "a_act": torch.randn(d_lat, D, generator=g) / d_lat**0.5,
        "patch_sig": 0.3 * torch.randn(N_PATCHES, D, generator=g),
        "c_fut": 0.05 * torch.randn(d_lat, D, generator=g) / d_lat**0.5,
        "m0": m0 / m0.norm(),
    }


def generate_synthetic_grid_cache(
    out_dir: Path,
    *,
    n: int,
    seed: int,
    sample_id_offset: int,
    params: dict[str, Tensor],
    truth_path: Path | None = None,
) -> dict[str, Tensor]:
    """Write a schema-conforming grid cache with planted bimodality.

    Planted structure: for a marked ~50% subset ("fork"), the future grids at
    every horizon carry an additive mode component s * gap * m0 with sign s
    iid +-1 per sample, NOT predictable from the context. gap grows with the
    horizon. The pooled futures are therefore bimodal along m0 in the fork
    subset -> MiniBatchKMeans cluster ids recover the mode sign (oracle bit),
    while the baseline head can only regress the between-modes mean.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.pt"):
        old.unlink()
    g = torch.Generator().manual_seed(seed)
    d_lat = SYNTH_D_LAT
    z = torch.randn(n, d_lat, generator=g)
    fork = torch.rand(n, generator=g) < 0.5
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()

    a_ctx, a_act = params["a_ctx"], params["a_act"]
    patch_sig, c_fut, m0 = params["patch_sig"], params["c_fut"], params["m0"]

    ctx_grid = (z @ a_ctx)[:, None, :] + patch_sig[None] + 0.05 * torch.randn(n, N_PATCHES, D, generator=g)
    ctx_act = z @ a_act + 0.05 * torch.randn(n, D, generator=g)

    def fut_grid(h: int) -> Tensor:
        gap = 2.0 + 0.6 * h
        offset = fork.float() * s * gap  # (n,)
        base = (z @ c_fut)[:, None, :] + 0.3 * patch_sig[None]
        return base + offset[:, None, None] * m0[None, None, :] + 0.1 * torch.randn(
            n, N_PATCHES, D, generator=g
        )

    grids = {h: fut_grid(h) for h in (1, 2, 3, 4, 5)}
    gt_grid = torch.stack([grids[1], grids[3], grids[5]], dim=1)  # (n, 3, p, d)
    gt_pooled = torch.stack([grids[h].mean(dim=1) for h in (1, 2, 3, 4, 5)], dim=1)  # (n, 5, d)

    sample_id = torch.arange(n, dtype=torch.int64) + sample_id_offset
    shard = {
        "ctx_foresight_grid": ctx_grid.half(),
        "ctx_action_summary": ctx_act.half(),
        "gt_grid": gt_grid.half(),
        "gt_pooled": gt_pooled.half(),
        "ctx_foresight_pooled": ctx_grid.mean(dim=1).half(),
        "speed": (30 + 5 * torch.randn(n, 11, generator=g)).half(),
        "gas": torch.rand(n, 11, generator=g).half() * 0.2,
        "brake": torch.rand(n, 11, generator=g).half() * 0.1,
        "steer": (0.5 + 0.05 * torch.randn(n, 11, generator=g)).half(),
        "sample_id": sample_id,
        "frame_idx": torch.arange(n, dtype=torch.int64) * 10,
        "input_id": [f"Niro900-HQ/2026-01-01--00-00-{i % 60:02d}" for i in range(n)],
    }
    n_shards = 0
    for i0 in range(0, n, SHARD_SIZE):
        i1 = min(i0 + SHARD_SIZE, n)
        part = {
            k: (v[i0:i1].clone() if isinstance(v, Tensor) else v[i0:i1]) for k, v in shard.items()
        }
        torch.save(part, out_dir / f"shard_{n_shards:05d}.pt")
        n_shards += 1
    (out_dir / "meta.json").write_text(
        json.dumps({"n_samples": n, "n_shards": n_shards, "dataset": "SYNTHETIC-phase1-smoke",
                    "notes": f"planted bimodal fork subset ({int(fork.sum())}/{n}), seed={seed}"}, indent=2) + "\n"
    )
    truth = {"fork": fork, "s": s, "m0": m0, "sample_id": sample_id}
    if truth_path is not None:
        torch.save(truth, truth_path)
    print(f"[smoke] synthetic cache {out_dir}: n={n}, fork={int(fork.sum())}", flush=True)
    return truth


def generate_fake_entropy_sidecar(path: Path, truth: dict[str, Tensor], seed: int) -> None:
    """Fake val_entropy.pt: fork samples -> quintile 4, others spread 0..3."""
    g = torch.Generator().manual_seed(seed + 1)
    n = truth["sample_id"].shape[0]
    fork = truth["fork"]
    quint = torch.randint(0, 4, (n,), generator=g).to(torch.int8)
    quint[fork] = 4
    entropy = torch.rand(n, 5, generator=g) * 5 + fork.float()[:, None] * 10
    torch.save(
        {
            "sample_id": truth["sample_id"].clone(),
            "entropy_h": entropy.float(),
            "quintile_h1": quint.clone(),
            "quintile_h5": quint.clone(),
            "neighbor_idx": torch.randint(0, n, (n, 32), generator=g).to(torch.int32),
        },
        path,
    )
    print(f"[smoke] fake entropy sidecar -> {path} (fork subset = quintile 4)", flush=True)


def run_smoke(args: argparse.Namespace) -> None:
    base = REPO_ROOT / "foresight_mm" / "cache" / "synthetic_phase1"
    results_dir = REPO_ROOT / "foresight_mm" / "results" / "synthetic" / "phase1"
    results_dir.mkdir(parents=True, exist_ok=True)
    train_dir, val_dir = base / "train_g", base / "val_g"
    entropy_path = base / "val_entropy.pt"

    params = synth_params(args.seed + 999)
    generate_synthetic_grid_cache(train_dir, n=512, seed=args.seed, sample_id_offset=0, params=params)
    truth_val = generate_synthetic_grid_cache(
        val_dir, n=256, seed=args.seed + 7, sample_id_offset=100_000, params=params,
        truth_path=base / "val_truth.pt",
    )
    generate_fake_entropy_sidecar(entropy_path, truth_val, args.seed)

    common = dict(
        train_cache=str(train_dir), val_cache=str(val_dir), entropy=str(entropy_path),
        results_dir=str(results_dir), ckpt=args.ckpt, device=args.device,
        epochs=2, batch_size=8, lr=1e-3, seed=args.seed, log_every=10,
        no_wandb=True, smoke_tag=True,
    )
    p1_args = argparse.Namespace(**common, horizon=5, n_modes=16)
    p1_res = run_p1(p1_args)

    p2_args = argparse.Namespace(
        **{**common, "batch_size": 8},
        k=8, horizons="1,3,5",
        p1_json=[str(results_dir / "p1_h5_m16.json")],
    )
    p2_res = run_p2(p2_args)

    # ------------------------- assertions ---------------------------------- #
    p1_json = json.loads((results_dir / "p1_h5_m16.json").read_text())
    p2_json = json.loads((results_dir / "p2_k8.json").read_text())

    q4_p1 = p1_json["final"]["per_quintile"]["4"]
    assert q4_p1["n"] > 0, "no planted-subset (quintile 4) samples"
    assert q4_p1["rel_drop"] > 0, f"oracle drop not positive on planted subset: {q4_p1}"

    q4_p2 = p2_json["final"]["per_quintile_h5"]["4"]
    mink_h5_q4 = q4_p2["minK_loss_per_h"]["h5"]
    base_h5_q4 = q4_p1["L_base"]
    assert mink_h5_q4 < base_h5_q4, (
        f"p2 min-over-K ({mink_h5_q4:.4f}) not < p1 single-head baseline "
        f"({base_h5_q4:.4f}) on planted subset"
    )
    live = p2_json["final"]["live_modes"]
    assert live >= 2, f"expected >=2 live modes, got {live}"

    print("\n===== SMOKE SUMMARY =====", flush=True)
    print(json.dumps({
        "p1_h5_overall": {kk: p1_json["final"][kk] for kk in ("L_base", "L_oracle", "rel_drop")},
        "p1_h5_planted_q4": q4_p1,
        "p2_minK_loss_per_h": p2_json["final"]["minK_loss_per_h"],
        "p2_planted_q4_minK_h5_vs_p1_base": {"p2_minK": mink_h5_q4, "p1_base": base_h5_q4},
        "p2_live_modes": live,
        "p2_win_rate": p2_json["final"]["win_rate"],
        "p2_diversity_per_h": p2_json["final"]["diversity_per_h"],
        "gate_g1_recovery": p2_json.get("gate_g1_recovery"),
    }, indent=2), flush=True)
    print("SMOKE PASS", flush=True)
    del p1_res, p2_res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--train-cache", default=str(DEFAULT_TRAIN_CACHE))
    sp.add_argument("--val-cache", default=str(DEFAULT_VAL_CACHE))
    sp.add_argument("--entropy", default=str(DEFAULT_ENTROPY))
    sp.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    sp.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--epochs", type=int, default=8)
    sp.add_argument("--batch-size", type=int, default=48)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--log-every", type=int, default=20)
    sp.add_argument("--no-wandb", action="store_true")
    sp.add_argument("--smoke-tag", action="store_true", help=argparse.SUPPRESS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp1 = sub.add_parser("p1", help="oracle-bits probe")
    _add_common(sp1)
    sp1.add_argument("--horizon", type=int, choices=[1, 3, 5], required=True)
    sp1.add_argument("--n-modes", type=int, default=16)
    sp1.set_defaults(func=run_p1)

    sp2 = sub.add_parser("p2", help="K-head trajectory-WTA probe")
    _add_common(sp2)
    sp2.add_argument("--k", type=int, default=8)
    sp2.add_argument("--horizons", default="1,3,5")
    sp2.add_argument("--p1-json", nargs="*", default=None,
                     help="p1 result JSONs for baseline comparison / gate-G1 recovery")
    sp2.set_defaults(func=run_p2)

    sps = sub.add_parser("smoke", help="synthetic end-to-end self-test (no real caches)")
    sps.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    sps.add_argument("--device", default="cuda")
    sps.add_argument("--seed", type=int, default=42)
    sps.set_defaults(func=run_smoke)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    import sys

    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
