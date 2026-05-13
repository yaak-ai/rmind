import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from rmind.components.transformer import FlowActionDecoder
from rmind.components.transformer.decoder import FlowSamplingMethod

VISUAL_ACTION_DIM = 2
TARGET_PATH_COLOR = "black"
PREDICTION_PATH_COLOR = "0.45"
PLOT_DPI = 160
FLOW_SAMPLING_METHODS: tuple[FlowSamplingMethod, ...] = ("euler", "midpoint", "heun")
TIME_SAMPLING_METHODS = ("uniform", "logit-normal")
LOGIT_NORMAL_MEAN = 0.0
LOGIT_NORMAL_STD = 1.0
TIMESTEP_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#6A3D9A",
    "#B15928",
    "#33A02C",
    "#FB9A99",
    "#A6CEE3",
)


@dataclass(frozen=True, kw_only=True)
class Metrics:
    flow_mse: float
    sample_l1: float


@dataclass(frozen=True, kw_only=True)
class Snapshot:
    step: int
    metrics: Metrics
    path: Tensor
    target: Tensor
    sampling_method: str


@dataclass(frozen=True, kw_only=True)
class TrainingState:
    oracle: "SyntheticOracle"
    decoder: FlowActionDecoder
    optimizer: torch.optim.Optimizer
    eval_condition: Tensor
    eval_sample_noise: Tensor
    eval_flow_noise: Tensor
    eval_flow_time: Tensor
    time_sampling: str


@dataclass(frozen=True, kw_only=True)
class OutputPaths:
    run_dir: Path
    plot_path: Path | None
    gif_dir: Path | None
    checkpoint_path: Path


class SyntheticOracle(nn.Module):
    def __init__(
        self,
        *,
        condition_tokens: int,
        condition_dim: int,
        horizon: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.proj = nn.Linear(condition_tokens * condition_dim, horizon * action_dim)
        self.requires_grad_(requires_grad=False)

    def forward(self, condition: Tensor) -> Tensor:
        batch_size = condition.shape[0]
        actions = self.proj(condition.flatten(start_dim=1))
        actions = actions.view(batch_size, self.horizon, self.action_dim)
        actions = actions.cumsum(dim=1)
        actions -= actions.mean(dim=1, keepdim=True)
        actions /= actions.std(dim=1, keepdim=True).clamp_min(1e-3)
        return actions * 0.5


def sample_flow_time(
    batch_size: int, *, dtype: torch.dtype, device: torch.device, time_sampling: str
) -> Tensor:
    match time_sampling:
        case "uniform":
            return torch.rand(batch_size, dtype=dtype, device=device)
        case "logit-normal":
            z = torch.randn(batch_size, dtype=dtype, device=device)
            return torch.sigmoid(z * LOGIT_NORMAL_STD + LOGIT_NORMAL_MEAN)
        case _:
            msg = f"unsupported time sampling method: {time_sampling!r}"
            raise ValueError(msg)


def flow_loss(  # noqa: PLR0913
    *,
    decoder: FlowActionDecoder,
    condition: Tensor,
    target: Tensor,
    time_sampling: str,
    noise: Tensor | None = None,
    flow_time: Tensor | None = None,
) -> Tensor:
    noise = torch.randn_like(target) if noise is None else noise
    flow_time = (
        sample_flow_time(
            target.shape[0],
            dtype=target.dtype,
            device=target.device,
            time_sampling=time_sampling,
        )
        if flow_time is None
        else flow_time.to(dtype=target.dtype, device=target.device)
    )
    t = flow_time.view(-1, 1, 1)
    noised = torch.lerp(noise, target, t)
    velocity = decoder(
        condition_tokens=condition, noised_actions=noised, flow_time=flow_time
    )
    return F.mse_loss(velocity, target - noise)


@torch.inference_mode()
def evaluate(  # noqa: PLR0913
    *,
    decoder: FlowActionDecoder,
    oracle: SyntheticOracle,
    condition: Tensor,
    sample_noise: Tensor,
    flow_noise: Tensor,
    flow_time: Tensor,
    time_sampling: str,
) -> Metrics:
    target = oracle(condition)
    loss = flow_loss(
        decoder=decoder,
        condition=condition,
        target=target,
        time_sampling=time_sampling,
        noise=flow_noise,
        flow_time=flow_time,
    )
    sample = decoder.sample(condition_tokens=condition, noise=sample_noise)
    return Metrics(
        flow_mse=float(loss.item()), sample_l1=float(F.l1_loss(sample, target).item())
    )


@torch.inference_mode()
def trace_sample_path(  # noqa: PLR0914
    *, decoder: FlowActionDecoder, condition: Tensor, noise: Tensor
) -> Tensor:
    x = noise.clone()
    path = [x.clone()]
    steps = decoder.flow_sampling_steps
    dt = 1.0 / steps
    batch_size = condition.shape[0]
    method = decoder.flow_sampling_method

    for step in range(steps):
        t = step * dt
        flow_time = torch.full((batch_size,), t, dtype=x.dtype, device=x.device)
        velocity = decoder(
            condition_tokens=condition, noised_actions=x, flow_time=flow_time
        )

        match method:
            case "euler":
                x = torch.add(x, velocity, alpha=dt)
            case "midpoint":
                midpoint = torch.add(x, velocity, alpha=0.5 * dt)
                midpoint_flow_time = torch.full(
                    (batch_size,), t + 0.5 * dt, dtype=x.dtype, device=x.device
                )
                midpoint_velocity = decoder(
                    condition_tokens=condition,
                    noised_actions=midpoint,
                    flow_time=midpoint_flow_time,
                )
                x = torch.add(x, midpoint_velocity, alpha=dt)
            case "heun":
                predictor = torch.add(x, velocity, alpha=dt)
                corrected_flow_time = torch.full(
                    (batch_size,), t + dt, dtype=x.dtype, device=x.device
                )
                corrected_velocity = decoder(
                    condition_tokens=condition,
                    noised_actions=predictor,
                    flow_time=corrected_flow_time,
                )
                x += 0.5 * dt * (velocity + corrected_velocity)
            case _:
                msg = f"unsupported flow sampling method: {method!r}"
                raise ValueError(msg)

        path.append(x.clone())

    return torch.stack(path)


@torch.inference_mode()
def record_snapshot(  # noqa: PLR0913
    *,
    step: int,
    decoder: FlowActionDecoder,
    oracle: SyntheticOracle,
    condition: Tensor,
    sample_noise: Tensor,
    flow_noise: Tensor,
    flow_time: Tensor,
    time_sampling: str,
) -> Snapshot:
    metrics = evaluate(
        decoder=decoder,
        oracle=oracle,
        condition=condition,
        sample_noise=sample_noise,
        flow_noise=flow_noise,
        flow_time=flow_time,
        time_sampling=time_sampling,
    )
    path = trace_sample_path(
        decoder=decoder, condition=condition[:1], noise=sample_noise[:1]
    )
    target = oracle(condition[:1])
    return Snapshot(
        step=step,
        metrics=metrics,
        path=path.detach().cpu(),
        target=target.detach().cpu(),
        sampling_method=decoder.flow_sampling_method,
    )


def save_plot(
    *, path: Path, snapshots: list[Snapshot], train_losses: list[float]
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    _require_2d_actions(snapshots=snapshots)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [snapshot.step for snapshot in snapshots]
    flow_mse = [snapshot.metrics.flow_mse for snapshot in snapshots]
    sample_l1 = [snapshot.metrics.sample_l1 for snapshot in snapshots]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax_loss, ax_path, ax_traj, ax_error = axes.flatten()

    train_steps = list(range(1, len(train_losses) + 1))
    ax_loss.plot(
        train_steps,
        train_losses,
        color="tab:blue",
        alpha=0.25,
        linewidth=1,
        label="train loss (per step)",
    )
    ax_loss.plot(steps, flow_mse, marker="o", color="tab:blue", label="flow MSE (eval)")
    ax_loss.plot(
        steps, sample_l1, marker="o", color="tab:orange", label="sample L1 (eval)"
    )
    ax_loss.set_title("training progress")
    ax_loss.set_xlabel("train step")
    ax_loss.grid(visible=True, alpha=0.3)
    ax_loss.legend()

    for snapshot in snapshots:
        point_path = snapshot.path[:, 0, 0, :2]
        ax_path.plot(
            point_path[:, 0],
            point_path[:, 1],
            marker="o",
            label=f"step {snapshot.step}",
            alpha=0.8,
        )
    target_point = snapshots[-1].target[0, 0, :2]
    ax_path.scatter(
        target_point[0],
        target_point[1],
        marker="x",
        s=100,
        color="black",
        label="target",
    )
    ax_path.set_title(f"{snapshots[-1].sampling_method} path for horizon step 0")
    ax_path.set_xlabel("action dim 0")
    ax_path.set_ylabel("action dim 1")
    ax_path.grid(visible=True, alpha=0.3)
    ax_path.legend()

    horizon = torch.arange(snapshots[-1].target.shape[1])
    for dim in range(snapshots[-1].target.shape[2]):
        ax_traj.plot(
            horizon,
            snapshots[-1].target[0, :, dim],
            linestyle="--",
            label=f"target dim {dim}",
        )
        ax_traj.plot(
            horizon, snapshots[-1].path[-1, 0, :, dim], label=f"sample dim {dim}"
        )
    ax_traj.set_title(f"final sampled trajectory, step {snapshots[-1].step}")
    ax_traj.set_xlabel("horizon index")
    ax_traj.grid(visible=True, alpha=0.3)
    ax_traj.legend(ncols=2, fontsize="small")

    for snapshot in snapshots:
        per_horizon_l1 = (snapshot.path[-1, 0] - snapshot.target[0]).abs().mean(dim=-1)
        ax_error.plot(
            horizon, per_horizon_l1, marker="o", label=f"step {snapshot.step}"
        )
    ax_error.set_title("sample L1 by horizon")
    ax_error.set_xlabel("horizon index")
    ax_error.grid(visible=True, alpha=0.3)
    ax_error.legend()

    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _require_2d_actions(*, snapshots: list[Snapshot]) -> None:
    action_dim = snapshots[-1].target.shape[-1]
    if action_dim != VISUAL_ACTION_DIM:
        msg = (
            "flow oracle visualizations require 2D actions; "
            f"received action_dim={action_dim}"
        )
        raise ValueError(msg)


def _plot_bounds(
    *, target: Tensor, denoising_path: Tensor
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_points = torch.cat(
        [
            target.reshape(-1, VISUAL_ACTION_DIM),
            denoising_path.reshape(-1, VISUAL_ACTION_DIM),
        ],
        dim=0,
    )
    low = all_points.min(dim=0).values
    high = all_points.max(dim=0).values
    span = (high - low).clamp_min(1e-3)
    padding = 0.15 * span
    low -= padding
    high += padding
    return (float(low[0]), float(high[0])), (float(low[1]), float(high[1]))


def _snapshot_gif_path(*, directory: Path, snapshot: Snapshot) -> Path:
    return directory / f"flow_oracle_step_{snapshot.step:06d}.gif"


def _horizon_colors(*, plt: Any, horizon_steps: int) -> list[Any]:
    if horizon_steps <= len(TIMESTEP_COLORS):
        return list(TIMESTEP_COLORS[:horizon_steps])

    color_map = plt.get_cmap("turbo")
    return [
        color_map(horizon_idx / horizon_steps) for horizon_idx in range(horizon_steps)
    ]


def save_snapshot_gif(*, path: Path, snapshot: Snapshot, fps: int) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: PLC0415

    target = snapshot.target[0]
    denoising_path = snapshot.path[:, 0]
    xlim, ylim = _plot_bounds(target=target, denoising_path=denoising_path)
    timestep_colors = _horizon_colors(plt=plt, horizon_steps=target.shape[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("action dim 0")
    ax.set_ylabel("action dim 1")
    ax.grid(visible=True, alpha=0.3)

    initial = denoising_path[0]
    ax.plot(
        target[:, 0],
        target[:, 1],
        color=TARGET_PATH_COLOR,
        linewidth=2,
        label="target path",
    )
    ax.scatter(
        target[:, 0],
        target[:, 1],
        color=timestep_colors,
        marker="x",
        s=90,
        linewidths=2,
        label="target timesteps",
    )
    ax.plot(
        initial[:, 0], initial[:, 1], color="0.6", linestyle=":", label="initial noise"
    )
    ax.scatter(
        initial[:, 0],
        initial[:, 1],
        color=timestep_colors,
        marker=".",
        s=45,
        alpha=0.4,
        label="initial timesteps",
    )
    (current_line,) = ax.plot(
        [],
        [],
        color=PREDICTION_PATH_COLOR,
        linestyle=":",
        linewidth=2,
        label="denoised path",
    )
    current_points = ax.scatter(
        initial[:, 0],
        initial[:, 1],
        color=timestep_colors,
        edgecolors="white",
        linewidths=0.5,
        s=65,
        label="denoised timesteps",
    )
    trail_lines = [
        ax.plot([], [], color=color, alpha=0.3, linewidth=1.5)[0]
        for color in timestep_colors
    ]
    ax.legend(loc="best")

    title = ax.set_title("")
    max_frame = denoising_path.shape[0] - 1

    def update(frame: int) -> list:
        current = denoising_path[frame]
        current_line.set_data(current[:, 0], current[:, 1])
        current_points.set_offsets(current[:, :VISUAL_ACTION_DIM])
        for horizon_idx, trail_line in enumerate(trail_lines):
            trail = denoising_path[: frame + 1, horizon_idx]
            trail_line.set_data(trail[:, 0], trail[:, 1])
        title.set_text(
            f"train step {snapshot.step} | {snapshot.sampling_method} denoise {frame}/{max_frame}"
        )
        return [current_line, current_points, *trail_lines, title]

    animation = FuncAnimation(
        fig,
        update,
        frames=denoising_path.shape[0],
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_snapshot_gif_if_enabled(
    *, snapshot: Snapshot, gif_dir: Path | None, gif_fps: int
) -> None:
    if gif_dir is None:
        return
    path = _snapshot_gif_path(directory=gif_dir, snapshot=snapshot)
    save_snapshot_gif(path=path, snapshot=snapshot, fps=gif_fps)


def save_checkpoint(
    *, path: Path, state: TrainingState, args: argparse.Namespace
) -> None:
    torch.save(
        {
            "decoder_state_dict": state.decoder.state_dict(),
            "oracle_state_dict": state.oracle.state_dict(),
            "config": {
                "condition_tokens": args.condition_tokens,
                "condition_dim": args.condition_dim,
                "decoder_dim": args.decoder_dim,
                "horizon": args.horizon,
                "action_dim": args.action_dim,
                "sampling_steps": args.sampling_steps,
                "sampling_method": args.sampling_method,
                "time_embedding_scale": args.time_embedding_scale,
                "time_logit_scale": args.time_logit_scale,
                "time_sampling": args.time_sampling,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
            },
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit FlowActionDecoder on a synthetic condition-to-action oracle."
    )
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--condition-tokens", type=int, default=2)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--decoder-dim", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--action-dim",
        type=int,
        default=VISUAL_ACTION_DIM,
        help="Synthetic action dimension. Keep at 2 for path visualizations.",
    )
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument(
        "--sampling-method",
        choices=FLOW_SAMPLING_METHODS,
        default="heun",
        help="ODE solver used by decoder.sample and oracle path traces.",
    )
    parser.add_argument(
        "--time-embedding-scale",
        type=float,
        default=1000.0,
        help="Scale applied before sinusoidal flow-time embedding.",
    )
    parser.add_argument(
        "--time-logit-scale",
        type=float,
        default=0.25,
        help="Scale applied to the auxiliary logit(flow_time) embedding channel.",
    )
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--time-sampling",
        choices=TIME_SAMPLING_METHODS,
        default="logit-normal",
        help="Distribution for sampling flow time during training.",
    )
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--max-final-l1", type=float, default=0.5)
    parser.add_argument("--min-l1-improvement", type=float, default=0.25)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs"),
        help="Root directory for timestamped run outputs.",
    )
    parser.add_argument(
        "--plot-filename",
        default="plot-oracle-flow.png",
        help="PNG filename inside the timestamped output directory.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable PNG summary output."
    )
    parser.add_argument(
        "--gif-dir-name",
        default="gifs",
        help="GIF subdirectory name inside the timestamped output directory.",
    )
    parser.add_argument(
        "--no-gifs",
        action="store_true",
        help="Disable per-log-step denoising GIF output.",
    )
    parser.add_argument("--gif-fps", type=int, default=2)
    parser.add_argument(
        "--fresh-batches",
        action="store_true",
        help="Train on fresh synthetic conditions instead of overfitting one batch.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_int_args = {
        "--batch-size": args.batch_size,
        "--condition-tokens": args.condition_tokens,
        "--condition-dim": args.condition_dim,
        "--decoder-dim": args.decoder_dim,
        "--gif-fps": args.gif_fps,
        "--horizon": args.horizon,
        "--log-every": args.log_every,
        "--num-heads": args.num_heads,
        "--num-layers": args.num_layers,
        "--sampling-steps": args.sampling_steps,
        "--train-steps": args.train_steps,
    }
    for option, value in positive_int_args.items():
        if value <= 0:
            msg = f"{option} must be positive, received {value}"
            raise SystemExit(msg)

    if args.time_embedding_scale <= 0.0:
        msg = (
            "--time-embedding-scale must be positive, "
            f"received {args.time_embedding_scale}"
        )
        raise SystemExit(msg)

    if args.time_logit_scale < 0.0:
        msg = (
            f"--time-logit-scale must be non-negative, received {args.time_logit_scale}"
        )
        raise SystemExit(msg)

    if args.action_dim != VISUAL_ACTION_DIM and (not args.no_plot or not args.no_gifs):
        msg = (
            "flow oracle GIF/PNG visualizations expect --action-dim 2; "
            f"received {args.action_dim}. Use --no-plot --no-gifs for non-2D checks."
        )
        raise SystemExit(msg)
    validate_output_name(name=args.plot_filename, option="--plot-filename")
    validate_output_name(name=args.gif_dir_name, option="--gif-dir-name")


def validate_output_name(*, name: str, option: str) -> None:
    if not name or Path(name).name != name:
        msg = f"{option} must be a plain filename or directory name, received {name!r}"
        raise SystemExit(msg)


def create_output_paths(args: argparse.Namespace) -> OutputPaths:
    run_datetime = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d-%H%M%S-%f%z")
    run_dir = args.output_root / run_datetime
    run_dir.mkdir(parents=True, exist_ok=False)
    gif_dir = None if args.no_gifs else run_dir / args.gif_dir_name
    if gif_dir is not None:
        gif_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        run_dir=run_dir,
        plot_path=None if args.no_plot else run_dir / args.plot_filename,
        gif_dir=gif_dir,
        checkpoint_path=run_dir / "checkpoint.pt",
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_training_state(
    *, args: argparse.Namespace, device: torch.device
) -> TrainingState:
    oracle = SyntheticOracle(
        condition_tokens=args.condition_tokens,
        condition_dim=args.condition_dim,
        horizon=args.horizon,
        action_dim=args.action_dim,
    ).to(device)
    decoder = FlowActionDecoder(
        condition_dim=args.condition_dim,
        dim_model=args.decoder_dim,
        action_dim=args.action_dim,
        action_horizon=args.horizon,
        flow_sampling_steps=args.sampling_steps,
        flow_sampling_method=cast("FlowSamplingMethod", args.sampling_method),
        time_embedding_scale=args.time_embedding_scale,
        time_logit_scale=args.time_logit_scale,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)

    eval_condition = torch.randn(
        args.batch_size, args.condition_tokens, args.condition_dim, device=device
    )
    eval_sample_noise = torch.randn(
        args.batch_size, args.horizon, args.action_dim, device=device
    )
    eval_flow_noise = torch.randn(
        args.batch_size, args.horizon, args.action_dim, device=device
    )
    eval_flow_time = sample_flow_time(
        args.batch_size,
        dtype=eval_flow_noise.dtype,
        device=device,
        time_sampling=args.time_sampling,
    )
    return TrainingState(
        oracle=oracle,
        decoder=decoder,
        optimizer=optimizer,
        eval_condition=eval_condition,
        eval_sample_noise=eval_sample_noise,
        eval_flow_noise=eval_flow_noise,
        eval_flow_time=eval_flow_time,
        time_sampling=args.time_sampling,
    )


def train_step(
    *, args: argparse.Namespace, state: TrainingState, device: torch.device
) -> float:
    condition = (
        torch.randn(
            args.batch_size, args.condition_tokens, args.condition_dim, device=device
        )
        if args.fresh_batches
        else state.eval_condition
    )
    loss = flow_loss(
        decoder=state.decoder,
        condition=condition,
        target=state.oracle(condition),
        time_sampling=state.time_sampling,
    )

    state.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(state.decoder.parameters(), max_norm=1.0)
    state.optimizer.step()
    return float(loss.item())


def record_eval_snapshot(
    *, step: int, state: TrainingState, gif_dir: Path | None, gif_fps: int
) -> Snapshot:
    snapshot = record_snapshot(
        step=step,
        decoder=state.decoder,
        oracle=state.oracle,
        condition=state.eval_condition,
        sample_noise=state.eval_sample_noise,
        flow_noise=state.eval_flow_noise,
        flow_time=state.eval_flow_time,
        time_sampling=state.time_sampling,
    )
    save_snapshot_gif_if_enabled(snapshot=snapshot, gif_dir=gif_dir, gif_fps=gif_fps)
    return snapshot


def should_record_snapshot(*, step: int, train_steps: int, log_every: int) -> bool:
    return step % log_every == 0 or step == train_steps


def progress_postfix(snapshot: Snapshot) -> dict[str, str]:
    return {
        "flow_mse": f"{snapshot.metrics.flow_mse:.4f}",
        "sample_l1": f"{snapshot.metrics.sample_l1:.4f}",
    }


def main() -> None:
    args = parse_args()
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    validate_args(args)
    output_paths = create_output_paths(args)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    state = build_training_state(args=args, device=device)
    snapshots = [
        record_eval_snapshot(
            step=0, state=state, gif_dir=output_paths.gif_dir, gif_fps=args.gif_fps
        )
    ]
    initial = snapshots[0].metrics

    progress = tqdm(
        range(1, args.train_steps + 1),
        total=args.train_steps,
        desc="training",
        unit="step",
    )
    train_losses: list[float] = []
    with progress:
        progress.set_postfix(progress_postfix(snapshots[0]))
        for step in progress:
            train_losses.append(train_step(args=args, state=state, device=device))

            if should_record_snapshot(
                step=step, train_steps=args.train_steps, log_every=args.log_every
            ):
                snapshot = record_eval_snapshot(
                    step=step,
                    state=state,
                    gif_dir=output_paths.gif_dir,
                    gif_fps=args.gif_fps,
                )
                snapshots.append(snapshot)
                progress.set_postfix(progress_postfix(snapshot))

    final = snapshots[-1].metrics
    if output_paths.plot_path is not None:
        save_plot(
            path=output_paths.plot_path, snapshots=snapshots, train_losses=train_losses
        )
    save_checkpoint(path=output_paths.checkpoint_path, state=state, args=args)

    improvement = initial.sample_l1 - final.sample_l1
    if final.sample_l1 > args.max_final_l1 or improvement < args.min_l1_improvement:
        msg = (
            "synthetic oracle check failed: "
            f"initial_l1={initial.sample_l1:.6f}, "
            f"final_l1={final.sample_l1:.6f}, "
            f"improvement={improvement:.6f}"
            f", outputs={output_paths.run_dir}"
        )
        raise SystemExit(msg)

    tqdm.write(
        "synthetic oracle check passed: "
        f"initial_l1={initial.sample_l1:.6f} "
        f"final_l1={final.sample_l1:.6f} "
        f"improvement={improvement:.6f}"
    )
    tqdm.write(f"outputs: {output_paths.run_dir}")


if __name__ == "__main__":
    main()
