import argparse
from pathlib import Path
from typing import cast

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from rmind.components.transformer import FlowActionDecoder
from rmind.components.transformer.decoder import FlowSamplingMethod
from rmind.scripts.flow_oracle import (
    FLOW_SAMPLING_METHODS,
    Metrics,
    Snapshot,
    SyntheticOracle,
    resolve_device,
    save_snapshot_gif,
    trace_sample_path,
)


def load_checkpoint(
    path: Path,
    device: torch.device,
    sampling_steps: int | None = None,
    sampling_method: FlowSamplingMethod | None = None,
) -> tuple[SyntheticOracle, FlowActionDecoder, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    oracle = SyntheticOracle(
        condition_tokens=cfg["condition_tokens"],
        condition_dim=cfg["condition_dim"],
        horizon=cfg["horizon"],
        action_dim=cfg["action_dim"],
    ).to(device)
    oracle.load_state_dict(ckpt["oracle_state_dict"])

    decoder = FlowActionDecoder(
        condition_dim=cfg["condition_dim"],
        dim_model=cfg["decoder_dim"],
        action_dim=cfg["action_dim"],
        action_horizon=cfg["horizon"],
        flow_sampling_steps=(
            cfg["sampling_steps"] if sampling_steps is None else sampling_steps
        ),
        flow_sampling_method=cast(
            "FlowSamplingMethod", sampling_method or cfg.get("sampling_method", "euler")
        ),
        time_embedding_scale=cfg.get("time_embedding_scale", 1000.0),
        time_logit_scale=cfg.get("time_logit_scale", 0.25),
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()

    return oracle, decoder, cfg


@torch.inference_mode()
def generate_snapshot(
    *,
    oracle: SyntheticOracle,
    decoder: FlowActionDecoder,
    cfg: dict,
    device: torch.device,
    sample_idx: int,
) -> Snapshot:
    condition = torch.randn(
        1, cfg["condition_tokens"], cfg["condition_dim"], device=device
    )
    noise = torch.randn(1, cfg["horizon"], cfg["action_dim"], device=device)
    target = oracle(condition)
    path = trace_sample_path(decoder=decoder, condition=condition, noise=noise)
    sample_l1 = float(F.l1_loss(path[-1], target).item())
    return Snapshot(
        step=sample_idx,
        metrics=Metrics(flow_mse=float("nan"), sample_l1=sample_l1),
        path=path.cpu(),
        target=target.cpu(),
        sampling_method=decoder.flow_sampling_method,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate denoising GIFs from a saved flow oracle checkpoint."
    )
    parser.add_argument(
        "checkpoint", type=Path, help="Path to checkpoint.pt saved by flow_oracle.py"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of GIFs to generate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for GIF output. Defaults to <checkpoint_dir>/samples/.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help="ODE steps for denoising. Defaults to the value saved in the checkpoint.",
    )
    parser.add_argument(
        "--sampling-method",
        choices=FLOW_SAMPLING_METHODS,
        default=None,
        help="ODE solver for denoising. Defaults to the checkpoint value.",
    )
    parser.add_argument("--gif-fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        msg = f"checkpoint not found: {args.checkpoint}"
        raise SystemExit(msg)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    oracle, decoder, cfg = load_checkpoint(
        args.checkpoint,
        device,
        sampling_steps=args.sampling_steps,
        sampling_method=args.sampling_method,
    )

    output_dir = args.output_dir or args.checkpoint.parent / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(args.num_samples), desc="generating", unit="gif"):
        snapshot = generate_snapshot(
            oracle=oracle, decoder=decoder, cfg=cfg, device=device, sample_idx=i
        )
        gif_path = output_dir / f"sample_{i:04d}.gif"
        save_snapshot_gif(path=gif_path, snapshot=snapshot, fps=args.gif_fps)
        tqdm.write(f"sample {i:04d}: l1={snapshot.metrics.sample_l1:.4f} -> {gif_path}")


if __name__ == "__main__":
    main()
