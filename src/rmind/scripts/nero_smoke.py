"""Smoke harness for the nero-arms causal patch policy and its action tokenizer.

Runs three stages, all on structured synthetic data at the **real** contract §8
shapes (3 cameras, 120-dim bimanual action, camera-conditioning, goal images):

``budget``
    Compute the flattened sequence length, check `head_dim % 8 == 0`, and probe
    which SDPA backend the trunk actually gets. This is the pre-flight from
    PR #265: at head_dim 20 the fused kernels are inadmissible and the math
    fallback materialises the full `(B, H, L, L)` score matrix and OOMs.
``tokenizer``
    Fit the VQ-BeT `NeroPoseTokenizer`, reporting translation error in **mm** and
    rotation error in **degrees**, separately (contract §5.5). Writes the
    checkpoint and the versioned standardisation artifact.
``policy``
    Train `NeroPatchPolicy` for a few hundred steps against that tokenizer;
    report the loss curve, step time and peak memory.

Usage::

    uv run python -m rmind.scripts.nero_smoke --stage all --out /tmp/nero-smoke
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

from rmind.data.nero import PoseStandardizer, state_quat_to_9d
from rmind.datamodules.nero_random import CAMERA_NAMES, nero_random_batch

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
#: steps discarded before the peak-memory counter is reset (allocator warmup)
WARMUP_STEPS = 4


def _cfg(experiment: str, overrides: list[str] | None = None) -> Any:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="train",
            overrides=[f"experiment={experiment}", *(overrides or [])],
        )


def _to(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# --------------------------------------------------------------------- budget


def stage_budget(cfg: Any) -> dict[str, Any]:
    tokens_per_frame = cfg.num_cameras * cfg.num_patches + 1
    sequence_length = cfg.episode_length * tokens_per_frame
    head_dim = cfg.policy_embedding_dim // cfg.num_heads
    patch = 14
    grid = (cfg.image_height // patch, cfg.image_width // patch)
    if grid[0] * grid[1] != cfg.num_patches:
        msg = (
            f"num_patches {cfg.num_patches} != the {grid} grid implied by "
            f"{cfg.image_height}x{cfg.image_width} at patch {patch}"
        )
        raise ValueError(msg)

    report = {
        "patch_grid": list(grid),
        "tokens_per_frame": tokens_per_frame,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "head_dim_div8": head_dim % 8 == 0,
        "head_dim_even_for_rope": head_dim % 2 == 0,
        # what the dense score matrix WOULD cost if a math fallback is taken
        "dense_scores_gib_per_sample_bf16": (
            cfg.num_heads * sequence_length**2 * 2 / 1024**3
        ),
    }
    if not report["head_dim_div8"]:
        msg = (
            f"head_dim {head_dim} is not divisible by 8: fused SDPA is inadmissible and "
            "the math fallback materialises the full (B, H, L, L) matrix (PR #265)"
        )
        raise ValueError(msg)

    # empirical: is the memory-efficient backend actually admissible for the
    # mask this trunk passes? Reasoning about it is not enough.
    if torch.cuda.is_available():
        q = torch.randn(
            1, cfg.num_heads, 512, head_dim, device="cuda", dtype=torch.bfloat16
        )
        mask = torch.zeros(512, 512, device="cuda", dtype=torch.bool)
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, q, q, attn_mask=~mask
                )
            report["efficient_attention_admissible"] = True
        except RuntimeError as error:  # pragma: no cover - backend-dependent
            report["efficient_attention_admissible"] = False
            report["efficient_attention_error"] = str(error)[:200]
    return report


# ------------------------------------------------------------------ tokenizer


def _chunk_pool(cfg: Any, *, batches: int, device: torch.device) -> torch.Tensor:
    """`(n, H, features)` valid per-side chunks drawn from the synthetic stream."""
    pool: list[torch.Tensor] = []
    for i in range(batches):
        batch = nero_random_batch(
            batch_size=cfg.batch_size,
            episode_length=cfg.episode_length,
            action_horizon=cfg.action_horizon,
            grids=dict.fromkeys(CAMERA_NAMES, (8, 8)),  # images unused here
            both_sides=cfg.both_sides,
            seed=i,
        )
        # storage form (46/side) -> model-facing 9D (60/side), the same
        # conversion the policy and the tokenizer do at their input boundary
        action = state_quat_to_9d(batch["action.future_state"])  # (b, T, H, 2, 60)
        valid = batch["side_valid"]
        b, t, h, s, f = action.shape
        chunks = action.permute(0, 1, 3, 2, 4).reshape(-1, h, f)
        mask = valid[:, None, :].expand(b, t, s).reshape(-1)
        pool.append(chunks[mask])
    return torch.cat(pool).to(device)


def stage_tokenizer(  # noqa: PLR0914
    cfg: Any, out: Path, *, steps: int, device: torch.device
) -> dict[str, Any]:
    torch.manual_seed(0)
    model = instantiate(cfg.model)

    train_pool = _chunk_pool(cfg, batches=256, device=device)
    val_pool = _chunk_pool(cfg, batches=8, device=device)

    # ⚠️ contract §5.4: statistics from the TRAIN split only.
    standardizer = PoseStandardizer.from_samples(
        train_pool.cpu(), source="smoke-train-split"
    )
    artifact = standardizer.save(out / "pose_standardizer.json")
    model.standardizer = standardizer
    model = model.to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    curve: list[dict[str, float]] = []
    batch_size = 256

    for step in range(steps):
        index = torch.randint(0, train_pool.shape[0], (batch_size,), device=device)
        chunks = train_pool[index]
        raw = chunks.flatten(-2, -1)
        a = model._normalize(raw)  # noqa: SLF001
        z = model.encoder(a)
        _codes, z_q, vq = model.quantizer(z)
        a_hat = model.decoder(z + (z_q - z).detach())
        recon = torch.nn.functional.l1_loss(a_hat, a)
        loss = recon + model.vq_weight * (
            vq["codebook"] + model.commitment_weight * vq["commit"]
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(1, steps // 20) == 0 or step == steps - 1:
            with torch.no_grad():
                metrics = model.reconstruction_metrics(a_hat.detach(), raw)
            curve.append({
                "step": step,
                "loss": loss.item(),
                "recon": recon.item(),
                **{k: v.item() for k, v in metrics.items()},
            })
            print(  # noqa: T201
                f"[tokenizer] step {step:5d} loss {loss.item():.4f} "
                f"recon {recon.item():.4f} "
                f"trans {metrics['translation_mm'].item():.2f} mm "
                f"rot {metrics['rotation_deg'].item():.2f} deg"
            )

    model.eval()
    with torch.no_grad():
        raw = val_pool.flatten(-2, -1)
        codes = model(val_pool)
        # round-trip exactly as the policy does it: codes -> chunk (no offset)
        reconstruction = model.invert(codes)
        held_out = {
            k: v.item()
            for k, v in model.reconstruction_metrics(reconstruction, raw).items()
        }
        # the UNQUANTIZED autoencoder, i.e. the ceiling the code path is measured
        # against: the gap between the two is the cost of the codebook, and the
        # gap between this and the mean baseline is what the encoder learned
        a = model._normalize(raw)  # noqa: SLF001
        z = model.encoder(a)
        ae = {
            k: v.item()
            for k, v in model.reconstruction_metrics(model.decoder(z), raw).items()
        }
        # baseline: predicting the train-split MEAN chunk, so the numbers above
        # are falsifiable rather than merely small
        mean_chunk = model._normalize(train_pool.flatten(-2, -1)).mean(0, keepdim=True)  # noqa: SLF001
        baseline = {
            k: v.item()
            for k, v in model.reconstruction_metrics(
                mean_chunk.expand_as(a), raw
            ).items()
        }

    torch.save(
        {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)},
        out / "pose_tokenizer.ckpt",
    )
    return {
        "curve": curve,
        "held_out_code_only": held_out,
        "held_out_autoencoder": ae,
        "held_out_mean_baseline": baseline,
        "standardizer_artifact": str(artifact),
        "num_train_chunks": int(train_pool.shape[0]),
    }


# --------------------------------------------------------------------- policy


def stage_policy(  # noqa: C901, PLR0913, PLR0914, PLR0915
    cfg: Any,
    out: Path,
    *,
    steps: int,
    device: torch.device,
    tokenizer_ckpt: Path | None,
    batch_size: int,
    both_sides: bool = True,
    overfit_one_batch: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(0)
    model = instantiate(cfg.model)

    if tokenizer_ckpt is not None and tokenizer_ckpt.exists():
        payload = torch.load(tokenizer_ckpt, map_location="cpu", weights_only=False)
        missing = model.tokenizer.load_state_dict(payload["state_dict"], strict=False)
        print(f"[policy] tokenizer loaded ({missing})")  # noqa: T201
    model.tokenizer.requires_grad_(False).eval()  # noqa: FBT003
    model = model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, betas=(0.9, 0.95)
    )

    curve: list[dict[str, float]] = []
    step_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    fixed = (
        _to(
            nero_random_batch(
                batch_size=batch_size,
                episode_length=cfg.episode_length,
                action_horizon=cfg.action_horizon,
                both_sides=both_sides,
                seed=0,
            ),
            device,
        )
        if overfit_one_batch
        else None
    )

    for step in range(steps):
        batch = fixed or _to(
            nero_random_batch(
                batch_size=batch_size,
                episode_length=cfg.episode_length,
                action_horizon=cfg.action_horizon,
                both_sides=both_sides,
                seed=step,
            ),
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            metrics = model._compute_metrics(batch)  # noqa: SLF001
            loss = metrics["policy", "loss"].sum(reduce=True)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - started)

        if step == WARMUP_STEPS and device.type == "cuda":  # discard warmup allocations
            torch.cuda.reset_peak_memory_stats()

        flat = {
            "/".join(k): v.item()
            for k, v in metrics.detach().items(include_nested=True, leaves_only=True)
        }
        curve.append({"step": step, "loss": loss.item(), **flat})
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            extra = ""
            if "policy/metric/translation_mm" in flat:
                extra = (
                    f" trans {flat['policy/metric/translation_mm']:.1f} mm "
                    f"rot {flat['policy/metric/rotation_deg']:.1f} deg"
                )
            print(  # noqa: T201
                f"[policy] step {step:4d} loss {loss.item():.4f} "
                f"offset {flat['policy/loss/offset']:.4f}{extra}"
            )

    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    tokens_per_frame = cfg.num_cameras * cfg.num_patches + 1

    def _mean(key: str, lo: int, hi: int) -> float:
        return statistics.fmean(c[key] for c in curve[lo:hi])

    window = max(1, steps // 10)
    summary = {
        "batch_size": batch_size,
        "both_sides": both_sides,
        "overfit_one_batch": overfit_one_batch,
        # ⚠️ the trunk gradient-CHECKPOINTS while training
        # (rmind.components.transformer.utils.run_layer_stack), so peak memory
        # is a checkpointed figure -- memory traded for recompute.
        "gradient_checkpointing": True,
        "sequence_length": cfg.episode_length * tokens_per_frame,
        "trainable_params_m": trainable / 1e6,
        "total_params_m": total / 1e6,
        "peak_memory_gib": peak,
        "median_step_s": (
            statistics.median(step_times[WARMUP_STEPS + 1 :])
            if len(step_times) > WARMUP_STEPS + 1
            else None
        ),
        "loss_first_window": _mean("loss", 0, window),
        "loss_last_window": _mean("loss", -window, len(curve)),
        "offset_first_window": _mean("policy/loss/offset", 0, window),
        "offset_last_window": _mean("policy/loss/offset", -window, len(curve)),
        "curve": curve,
    }
    name = "policy_curve"
    if overfit_one_batch:
        name += "_overfit"
    if not both_sides:
        name += "_right_only"
    _ = (out / f"{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--stage", default="all", choices=["all", "budget", "tokenizer", "policy"]
    )
    _ = parser.add_argument("--out", default="/tmp/nero-smoke")  # noqa: S108
    _ = parser.add_argument("--tokenizer-steps", type=int, default=3000)
    _ = parser.add_argument("--policy-steps", type=int, default=300)
    _ = parser.add_argument("--batch-size", type=int, default=8)
    _ = parser.add_argument("--lr", type=float, default=None)
    # right-only, i.e. the dummy recording's `side_valid = [False, True]`
    _ = parser.add_argument("--right-only", action="store_true")
    # train on ONE fixed batch: the discriminating check that the whole path
    # (readout -> per-side embedding -> shared head -> tokenizer targets ->
    # gradients) can learn at all, independently of whether fresh noise images
    # carry any signal
    _ = parser.add_argument("--overfit-one-batch", action="store_true")
    _ = parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    overrides = [] if args.lr is None else [f"lr={args.lr}"]
    report: dict[str, Any] = {}
    policy_cfg = _cfg("yaak/nero_arms/causal", overrides)

    if args.stage in {"all", "budget"}:
        report["budget"] = stage_budget(policy_cfg)
        print(json.dumps(report["budget"], indent=2))  # noqa: T201

    if args.stage in {"all", "tokenizer"}:
        report["tokenizer"] = stage_tokenizer(
            _cfg("yaak/nero_arms/tokenizer"),
            out,
            steps=args.tokenizer_steps,
            device=device,
        )

    if args.stage in {"all", "policy"}:
        report["policy"] = stage_policy(
            policy_cfg,
            out,
            steps=args.policy_steps,
            device=device,
            tokenizer_ckpt=out / "pose_tokenizer.ckpt",
            batch_size=args.batch_size,
            both_sides=not args.right_only,
            overfit_one_batch=args.overfit_one_batch,
        )

    _ = (out / "report.json").write_text(
        json.dumps(
            report, indent=2, default=lambda o: OmegaConf.to_container(o, resolve=True)
        )
    )
    print(f"\nwrote {out / 'report.json'}")  # noqa: T201


if __name__ == "__main__":
    main()
