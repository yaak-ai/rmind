"""Report paired D12 expert errors and descriptive asynchronous command replays."""

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from rmind.scripts.eval_d12_fm import action_metrics
from rmind.scripts.evaluate_d12_rtc import (
    HEIGHT_VALID_THRESHOLD,
    SEEDS,
    SEGMENTS,
    write_json,
)
from rmind.scripts.replay_d12_rtc import command_metrics


def video_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        msg = "Video rendering requires ffmpeg on PATH"
        raise FileNotFoundError(msg)
    return str(Path(executable).resolve())


def experts(root: Path, destination: Path | None = None) -> dict[str, Any]:
    destination = root if destination is None else destination
    report = {}
    all_arrays = {}
    for model in ("baseline", "rtc"):
        arrays = [
            dict(np.load(path))
            for path in sorted((root / "expert" / model).glob("*.npz"))
        ]
        merged = {
            key: np.concatenate([item[key] for item in arrays])
            for key in ("target", "height", "prediction_prefix0")
        }
        all_arrays[model] = merged
        report[model] = {}
        for stratum, mask in (
            ("all", np.ones(len(merged["target"]), dtype=bool)),
            ("height_valid", merged["height"][:, 2] > HEIGHT_VALID_THRESHOLD),
            ("height_missing", merged["height"][:, 2] <= HEIGHT_VALID_THRESHOLD),
        ):
            report[model][stratum] = {
                "windows": int(mask.sum()),
                "near": action_metrics(
                    merged["prediction_prefix0"][mask, :, :6],
                    merged["target"][mask, :6],
                ),
                "full": action_metrics(
                    merged["prediction_prefix0"][mask], merged["target"][mask]
                ),
            }
    np.testing.assert_array_equal(
        all_arrays["baseline"]["target"], all_arrays["rtc"]["target"]
    )
    np.testing.assert_array_equal(
        all_arrays["baseline"]["height"], all_arrays["rtc"]["height"]
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 3))
    for axis, action in zip(axes, ("traction", "steering", "fork1"), strict=True):
        values = [
            report[model]["all"]["near"]["per_action"][action]["mae"]
            for model in ("baseline", "rtc")
        ]
        axis.bar(["baseline", "RTC native"], values, color=["tab:blue", "tab:green"])
        axis.set_title(action)
        axis.set_ylabel("Expert MAE, first6 actions")
    figure.tight_layout()
    figure.savefig(destination / "expert-comparison.png", dpi=140)
    plt.close(figure)
    return report


def async_report(
    root: Path, video: bool, folder: str = "async", destination: Path | None = None
) -> dict[str, Any]:
    destination = root if destination is None else destination
    report = {}
    for segment in SEGMENTS:
        for profile in ("zero", "measured", "overrun"):
            figure, axes = plt.subplots(3, 1, figsize=(12.8, 6.4), sharex=True)
            curves = {}
            for arm, color in (
                ("baseline", "tab:blue"),
                ("rtc_unconditioned", "tab:orange"),
                ("rtc", "tab:green"),
            ):
                model = "baseline" if arm == "baseline" else "rtc"
                arrays = [
                    dict(
                        np.load(
                            root
                            / folder
                            / model
                            / f"{segment[0]}-{arm}-{profile}-{seed}.npz"
                        )
                    )
                    for seed in SEEDS
                ]
                metrics = [command_metrics(value) for value in arrays]
                report[f"{segment[0]}/{profile}/{arm}"] = {
                    "seeds": SEEDS,
                    "per_seed": metrics,
                }
                selection = arrays[0]["scored"]
                walls = arrays[0]["wall_ns"][selection]
                times = (walls - walls[0]) / 1e9
                values = np.stack([value["commands"][selection] for value in arrays])
                curves[arm] = values
                for index, axis in enumerate(axes):
                    for sample in values:
                        axis.plot(
                            times,
                            sample[:, index],
                            color=color,
                            alpha=0.22,
                            linewidth=0.5,
                        )
                    valid = np.isfinite(values[:, :, index]).all(axis=0)
                    median = np.full(len(times), np.nan)
                    median[valid] = np.median(values[:, valid, index], axis=0)
                    axis.plot(times, median, color=color, linewidth=1, label=arm)
                    axis.set_ylabel(("traction", "steering", "fork1")[index])
                    axis.set_ylim(-1.05, 1.05)
            axes[0].set_title(
                f"{segment[0]} / {profile}: recorded-state commands, not physical outcomes"
            )
            axes[0].legend(loc="upper right", ncol=3, fontsize=8)
            axes[-1].set_xlabel("Seconds within recorded segment")
            figure.tight_layout()
            path = destination / f"{segment[0]}-{profile}.png"
            figure.savefig(path, dpi=100)
            if video and profile == "measured":
                figure.canvas.draw()
                graph = Image.open(path).convert("RGB")
                bounds = [axis.get_window_extent().bounds for axis in axes]
                metadata = json.loads(
                    (root / "tokens/baseline" / f"{segment[1]}.json").read_text()
                )
                images = np.load(
                    root / "tokens/baseline" / f"{segment[1]}-images.npy", mmap_mode="r"
                )
                observed = np.asarray([row["wall_ns"] for row in metadata["rows"]])
                command = [
                    video_executable(),
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "1280x960",
                    "-r",
                    "5",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    str(destination / f"{segment[0]}-comparison.mp4"),
                ]
                with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                    for position in range(0, len(walls), 6):
                        frame = int(
                            np.searchsorted(observed, walls[position], side="right") - 1
                        )
                        canvas = Image.new("RGB", (1280, 960), "white")
                        canvas.paste(
                            Image.fromarray(images[frame]).resize((512, 288)), (0, 0)
                        )
                        canvas.paste(graph, (0, 320))
                        draw = ImageDraw.Draw(canvas)
                        draw.text(
                            (530, 20),
                            f"{segment[0]} | recorded cam_fork 144x256\nTimestamp ns: {int(walls[position])}\nBlue: baseline; orange: RTC without prefix; green: RTC\nThree explicit noise seeds; thin traces show variants.\nThe video/state is recorded, NOT generated by these commands.\nBlank curves are unavailable commands after fail-closed expiry.",
                            fill="black",
                            spacing=10,
                        )
                        for left, bottom, width, height in bounds:
                            cursor = left + width * (
                                times[position] - axes[-1].get_xlim()[0]
                            ) / (axes[-1].get_xlim()[1] - axes[-1].get_xlim()[0])
                            draw.line(
                                (
                                    cursor,
                                    320 + 640 - bottom - height,
                                    cursor,
                                    320 + 640 - bottom,
                                ),
                                fill="red",
                                width=2,
                            )
                        encoder.stdin.write(np.asarray(canvas).tobytes())
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        msg = "video encoder failed"
                        raise RuntimeError(msg)
            plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--expert-only", action="store_true")
    parser.add_argument(
        "--async-folder", choices=("async", "async-paced"), default="async"
    )
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    destination_root = args.report_dir or args.output
    destination_root.mkdir(parents=True, exist_ok=True)
    result = {"expert": experts(args.output, destination_root)}
    if not args.expert_only:
        result["async"] = async_report(
            args.output, args.video, args.async_folder, destination_root
        )
    destination = destination_root / (
        "expert-summary.json" if args.expert_only else "comparison.json"
    )
    write_json(destination, result)
    images = sorted(destination_root.glob("*.png"))
    videos = sorted(destination_root.glob("*-comparison.mp4"))
    document = "<h1>D12 recorded-state RTC comparison</h1><p>No counterfactual closed-loop success claim. Failed historical commands are not ground truth. Deadzone thresholds are command proxies, not calibrated actuator limits. Target-conditioned expert suffixes are separate oracle diagnostics.</p>"
    document += "".join(
        f'<h2>{html.escape(path.stem)}</h2><img width="100%" src="{path.name}">'
        for path in images
    )
    document += "".join(
        f'<video controls width="100%" src="{path.name}"></video>' for path in videos
    )
    (destination_root / "report.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
