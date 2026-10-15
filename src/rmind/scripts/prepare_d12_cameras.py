"""Build an additive D12 dataset overlay with backward-camera JPEG streams."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def prepare_camera(job, camera):
    destination = job / camera
    destination.mkdir(exist_ok=True)
    prefix = (job / "job-prefix.txt").read_text().strip()
    for filename in ("video.mp4", f"frame_info_{camera}.txt"):
        output = destination / filename
        if not output.exists():
            temporary = output.with_suffix(output.suffix + ".partial")
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    f"{prefix}/videos/{camera}/{filename}",
                    str(temporary),
                    "--only-show-errors",
                ],
                check=True,
            )
            temporary.replace(output)
    expected = len((destination / f"frame_info_{camera}.txt").read_text().split()) - 1
    frames = destination / "frames/256x144"
    marker = destination / "frames/complete.json"
    if not marker.exists():
        frames.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-threads",
                "2",
                "-i",
                str(destination / "video.mp4"),
                "-vf",
                "scale=256:144,setpts=N/(30*TB)",
                "-threads",
                "2",
                "-fps_mode",
                "passthrough",
                "-q:v",
                "2",
                "-start_number",
                "0",
                str(frames / "%06d.jpg"),
            ],
            check=True,
        )
        actual = len(list(frames.glob("*.jpg")))
        if actual != expected:
            raise ValueError(
                f"{job.name}/{camera}: {actual} JPEGs != {expected} encoded frames"
            )
        marker.write_text(json.dumps({"frames": actual, "camera": camera}) + "\n")
    print(
        json.dumps({"job": job.name, "camera": camera, "frames": expected}), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(arguments.audit.read_text())["jobs"]
    tasks = []
    for entry in jobs:
        source = arguments.source / entry["job"]
        destination = arguments.output / entry["job"]
        destination.mkdir(exist_ok=True)
        for item in source.iterdir():
            link = destination / item.name
            if not link.exists():
                link.symlink_to(item.resolve(), target_is_directory=item.is_dir())
        tasks.extend(
            (destination, camera)
            for camera in ("cam_left_backward", "cam_right_backward")
        )
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        list(executor.map(lambda task: prepare_camera(*task), tasks))


if __name__ == "__main__":
    main()
