"""Smoke harness for the robot-native nero patch policy (contract v3).

    python -m rmind.scripts.nero_robot_smoke --stage budget --tokenizer-ckpt T.ckpt
    python -m rmind.scripts.nero_robot_smoke --stage tokenizer --out OUT [--real]
    python -m rmind.scripts.nero_robot_smoke --stage policy --out OUT --relative-mode hand

Stages (one GPU job at a time; check nvidia-smi first):

``budget``
    Build the policy from its experiment, compare the trunk's configured
    `tokens_per_frame` with the token block the model ACTUALLY builds (a stale
    value tiles RoPE/intra-frame embeddings wrong without raising), check the
    head_dim rules and ASSERT the offset head is <= 5M parameters (pitfall 2: a
    per-code 100x13 table is ~340M).
``tokenizer``
    Fit a `NeroChunkTokenizer` (Lightning, `max_steps`) on synthetic chunks or,
    with `--real`, on the rbyte train split, save the checkpoint, and run the
    playbook report on the holdout.
``policy``
    Train the policy for `--steps` on synthetic hand-dependent batches
    (`--overfit` = one fixed batch) or on the real rbyte split, then report the
    loss drop, token-norm ratios, NaN/alarm counters and the hand-reliance
    deltas on a held-out synthetic batch.

Smoke only: nothing here is a claim about real-data performance.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

import rmind  # noqa: F401  (registers the `eval` resolver)
from rmind.datamodules.nero_robot_random import nero_robot_batch

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
OFFSET_HEAD_BUDGET = 5_000_000


def _cfg(experiment: str, overrides: list[str]) -> Any:
    from hydra import compose, initialize_config_dir  # noqa: PLC0415

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=[f"experiment={experiment}", *overrides])


def _to(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _bimanual(cfg: Any) -> bool:
    """A sided hand token (`hand_sides`) reads `hand.{left,right}.*`: feed it
    both-valid bimanual synthetic batches."""
    return bool(cfg.get("hand_sides"))


def _model(cfg: Any) -> Any:
    from hydra.utils import instantiate  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    return instantiate(OmegaConf.to_container(cfg.model, resolve=True))


# --------------------------------------------------------------------- budget


def stage_budget(cfg: Any) -> dict[str, Any]:
    model = _model(cfg).eval()
    hw = (int(cfg.image_height), int(cfg.image_width))
    batch = nero_robot_batch(batch_size=1, num_frames=2, image_hw=hw, seed=0, bimanual=_bimanual(cfg))
    with torch.no_grad():
        built = int(model._frame_tokens(batch).shape[-2])  # noqa: SLF001
    configured = int(model.encoder.tokens_per_frame)
    n_hand = int(cfg.use_hand_token) * max(1, len(cfg.get("hand_sides") or ()))
    arithmetic = 1 + n_hand + int(cfg.num_cameras) * int(cfg.num_patches)
    head_dim = int(cfg.policy_embedding_dim) // int(cfg.num_heads)
    offset_params = sum(p.numel() for p in model.offset_head.parameters())
    report = {
        "tokens_per_frame_built": built,
        "tokens_per_frame_configured": configured,
        "tokens_per_frame_arithmetic": arithmetic,
        "sequence_length": int(cfg.episode_length) * configured,
        "window": int(cfg.window),
        "head_dim": head_dim,
        "offset_head_params": offset_params,
        "code_head_params": sum(p.numel() for p in model.code_head.parameters()),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "trainable_params_m": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    }
    if not built == configured == arithmetic:
        msg = f"tokens_per_frame disagreement: {report}"
        raise ValueError(msg)
    if head_dim % 8 or head_dim % 2:
        msg = f"head_dim {head_dim} must be divisible by 8 (fused SDPA) and even (RoPE)"
        raise ValueError(msg)
    if offset_params > OFFSET_HEAD_BUDGET:
        msg = f"offset head has {offset_params} params > {OFFSET_HEAD_BUDGET}"
        raise ValueError(msg)
    return report


# ------------------------------------------------------------------ tokenizer


def stage_tokenizer(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    import pytorch_lightning as pl  # noqa: PLC0415
    from hydra.utils import instantiate  # noqa: PLC0415

    from rmind.scripts import nero_tokenizer_report as tr  # noqa: PLC0415

    experiment = "yaak/nero_robot/tokenizer" if args.real else "yaak/nero_robot/tokenizer_synthetic"
    overrides = [
        f"relative_mode={args.relative_mode}",
        f"num_quantizers={args.num_quantizers}",
        f"lr_total_steps={args.steps}",
        f"lr_warmup_steps={max(1, args.steps // 20)}",
        *args.override,
    ]
    cfg = _cfg(experiment, overrides)
    model = _model(cfg)
    dm = instantiate(cfg.datamodule)
    trainer = pl.Trainer(
        max_steps=args.steps,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        limit_val_batches=0,
        enable_progress_bar=False,
    )
    t0 = time.perf_counter()
    trainer.fit(model, train_dataloaders=dm.train_dataloader())
    ckpt = out / f"tokenizer_{args.relative_mode}_q{args.num_quantizers}.ckpt"
    trainer.save_checkpoint(ckpt)
    model = model.cpu().eval()
    ns = argparse.Namespace(
        synthetic=not args.real,
        experiment=experiment,
        override=overrides,
        max_batches=args.report_batches,
    )
    report = tr.report(
        model,
        tr.collect(model, tr.batches_for(ns, "train"), args.report_batches),
        tr.collect(model, tr.batches_for(ns, "val"), args.report_batches),
    )
    report |= {"ckpt": str(ckpt), "train_seconds": time.perf_counter() - t0, "steps": args.steps}
    (out / f"tokenizer_report_{args.relative_mode}_q{args.num_quantizers}.json").write_text(
        json.dumps(report, indent=1) + "\n"
    )
    return report


# --------------------------------------------------------------------- policy


def _policy_batches(args: argparse.Namespace, cfg: Any) -> Any:
    if args.real:
        from hydra.utils import instantiate  # noqa: PLC0415

        dm = instantiate(cfg.datamodule)
        loader = dm.train_dataloader()
        if args.overfit:  # one fixed real batch (windows from 1-2 episodes)
            fixed = next(iter(loader))
            while True:
                yield fixed
        while True:
            yield from loader
    hw = (int(cfg.image_height), int(cfg.image_width))
    step = 0
    while True:
        yield nero_robot_batch(
            batch_size=args.batch_size,
            num_frames=int(cfg.episode_length),
            image_hw=hw,
            seed=0 if args.overfit else step,
            bimanual=_bimanual(cfg),
        )
        step += 1


def stage_policy(args: argparse.Namespace, out: Path) -> dict[str, Any]:  # noqa: PLR0914, PLR0915
    experiment = "yaak/nero_robot/causal" if args.real else "yaak/nero_robot/synthetic"
    overrides = [f"relative_mode={args.relative_mode}", *args.override]
    cfg = _cfg(experiment, overrides)
    torch.manual_seed(0)
    device = torch.device(args.device)
    model = _model(cfg).to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    curve: list[dict[str, float]] = []
    times: list[float] = []
    nonfinite = 0
    alarms: dict[str, float] = {}
    batches = _policy_batches(args, cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        batch = _to(next(batches), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        norms: dict[str, torch.Tensor] = {}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            metrics = model.compute_metrics(batch, token_norms=norms)
            loss = metrics["policy", "loss"].sum(reduce=True)
        if not torch.isfinite(loss):
            nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        flat = {
            "/".join(k): float(v)
            for k, v in metrics.detach().items(include_nested=True, leaves_only=True)
        }
        for k, v in flat.items():
            if "alarm/" in k:
                alarms[k] = alarms.get(k, 0.0) + v
        ratio_state = float(norms["state"] / norms["patch"])
        ratio_hand = float(norms["hand"] / norms["patch"]) if "hand" in norms else math.nan
        curve.append({
            "step": step,
            "loss": float(loss),
            "offset": flat["policy/loss/offset"],
            "code_acc_joint": flat.get("policy/metric/code_acc_joint", math.nan),
            "offset_to_code_norm": flat.get("policy/metric/offset_to_code_norm", math.nan),
            "state_patch_ratio": ratio_state,
            "hand_patch_ratio": ratio_hand,
        })
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(  # noqa: T201
                f"[policy:{args.relative_mode}] step {step:4d} loss {float(loss):.4f} "
                f"offset {flat['policy/loss/offset']:.4f} state/patch {ratio_state:.2f} "
                f"hand/patch {ratio_hand:.2f}"
            )
    model.eval()
    hw = (int(cfg.image_height), int(cfg.image_width))
    with torch.no_grad():
        held = _to(
            nero_robot_batch(batch_size=args.batch_size, num_frames=int(cfg.episode_length), image_hw=hw, seed=123_456, bimanual=_bimanual(cfg)),
            device,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reliance = {k: float(v) for k, v in model.reliance_metrics_for(held).items()} if model.use_hand else {}
            if args.overfit:
                train_batch = _to(nero_robot_batch(batch_size=args.batch_size, num_frames=int(cfg.episode_length), image_hw=hw, seed=0, bimanual=_bimanual(cfg)), device)
                reliance_train = {k: float(v) for k, v in model.reliance_metrics_for(train_batch).items()} if model.use_hand else {}
            else:
                reliance_train = {}
    window = max(1, len(curve) // 10)
    first = statistics.fmean(c["loss"] for c in curve[:window])
    last = statistics.fmean(c["loss"] for c in curve[-window:])
    summary = {
        "experiment": experiment,
        "relative_mode": args.relative_mode,
        "overfit": args.overfit,
        "real": args.real,
        "steps": len(curve),
        "nonfinite_steps": nonfinite,
        "loss_first_window": first,
        "loss_last_window": last,
        "loss_drop_x": first / max(last, 1e-9),
        "offset_first": curve[0]["offset"],
        "offset_last_window": statistics.fmean(c["offset"] for c in curve[-window:]),
        "offset_to_code_norm_last_window": statistics.fmean(
            c["offset_to_code_norm"] for c in curve[-window:]
        ),
        "offset_code_conditioning": bool(getattr(model, "offset_code_conditioning", False)),
        "state_patch_ratio_last": curve[-1]["state_patch_ratio"],
        "hand_patch_ratio_last": curve[-1]["hand_patch_ratio"],
        "alarm_steps": alarms,
        "reliance_heldout": reliance,
        "reliance_train_batch": reliance_train,
        "median_step_s": statistics.median(times[3:]) if len(times) > 3 else None,  # noqa: PLR2004
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
        "tokens_per_frame": int(model.encoder.tokens_per_frame),
        "episode_length": int(cfg.episode_length),
        "window": int(cfg.window),
        "curve": curve,
    }
    name = f"policy_{'real' if args.real else 'synth'}_{args.relative_mode}{'_overfit' if args.overfit else ''}"
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1) + "\n")
    if args.save_ckpt:
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}, out / f"{name}.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["budget", "tokenizer", "policy"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--relative-mode", default="none", choices=["none", "hand", "all"])
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--report-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--save-ckpt", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    if args.stage == "budget":
        experiment = "yaak/nero_robot/causal" if args.real else "yaak/nero_robot/synthetic"
        result = stage_budget(_cfg(experiment, args.override))
    elif args.stage == "tokenizer":
        result = stage_tokenizer(args, args.out)
        result = {k: result[k] for k in ("relative_mode", "bits", "gates", "dct_rate_matched", "ckpt")}
    else:
        result = stage_policy(args, args.out)
        result = {k: v for k, v in result.items() if k != "curve"}
    print(json.dumps(result, indent=1, default=str))  # noqa: T201


if __name__ == "__main__":
    main()
