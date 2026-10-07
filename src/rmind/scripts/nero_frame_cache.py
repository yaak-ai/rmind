"""Build (or verify) the preprocessed nero frame cache (`rmind.data.nero_frame_cache`).

    python -m rmind.scripts.nero_frame_cache --root EPISODES --out CACHE [--workers 8]
    python -m rmind.scripts.nero_frame_cache --root EPISODES --out CACHE --verify 16

Caches every take of the bimanual split (`config/splits/*.json`, train + val) or
the takes named with `--take`, for the three cameras, at the experiment grid
(`--input-hw`, default 140 224). Up-to-date entries (a manifest matching the
current preprocessing, grid and mp4) are skipped unless `--force`. `--verify N`
decodes N random frames per (take, camera) through the rbyte decode path
(`TransformedSource(TorchCodecVideoSource, NeroImagePreprocess)`) and requires
them to be byte-identical to the cache; it exits 1 on any mismatch.

Put the cache on LOCAL disk (~8 GB for the 85-take 2026-10-07 corpus at 140x224)
and point the experiments at it with NERO_FRAME_CACHE.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rmind.scripts.nero_split_lib import SPLIT_JSON, load_split

CAMERAS = ("base", "side_left", "side_right")


def _build_one(  # noqa: PLR0913, PLR0917
    root: str,
    out: str,
    take: str,
    camera: str,
    hw: tuple[int, int],
    threads: int,
    force: bool,  # noqa: FBT001
) -> dict[str, Any]:
    from rmind.data.nero_frame_cache import (  # noqa: PLC0415
        build_camera_cache,
        cache_paths,
        manifest_problems,
    )

    video = Path(root) / take / f"{camera}.mp4"
    npy, _ = cache_paths(out, take, camera)
    if not force and not manifest_problems(npy, input_hw=hw, video=video):
        return {"take": take, "camera": camera, "status": "up-to-date"}
    t0 = time.monotonic()
    manifest = build_camera_cache(video, npy, input_hw=hw, num_threads=threads)
    return {
        "take": take,
        "camera": camera,
        "status": "built",
        "frames": manifest["num_frames"],
        "seconds": round(time.monotonic() - t0, 1),
    }


def _verify_one(  # noqa: PLR0913, PLR0917
    root: str, out: str, take: str, camera: str, hw: tuple[int, int], n: int
) -> str | None:
    import torch  # noqa: PLC0415
    from rbyte.streams.transformed import TransformedSource  # noqa: PLC0415
    from rbyte.streams.video import TorchCodecVideoSource  # noqa: PLC0415

    from rmind.data.nero_frame_cache import (  # noqa: PLC0415
        NeroFrameCacheSource,
        cache_paths,
    )
    from rmind.data.nero_image import NeroImagePreprocess  # noqa: PLC0415

    video = Path(root) / take / f"{camera}.mp4"
    npy, _ = cache_paths(out, take, camera)
    cached = NeroFrameCacheSource(path=npy, input_hw=hw, video=video)
    decoded = TransformedSource(
        source=TorchCodecVideoSource(source=video, num_ffmpeg_threads=2),
        transform=NeroImagePreprocess(hw),
    )
    total = len(cached)
    rng = random.Random(f"{take}/{camera}")  # noqa: S311  (sampling, not crypto)
    idx = sorted({0, total - 1, *rng.sample(range(total), min(n, total))})
    if not torch.equal(cached[idx], decoded[idx]):
        return f"{take}/{camera}: cached frames differ from the decode path"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="episode directory root"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="cache root (local disk)"
    )
    parser.add_argument("--split", type=Path, default=SPLIT_JSON)
    parser.add_argument(
        "--take", action="append", default=None, help="only these takes"
    )
    parser.add_argument("--input-hw", type=int, nargs=2, default=(140, 224))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--threads", type=int, default=4, help="ffmpeg threads per worker"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", type=int, default=0, metavar="N")
    args = parser.parse_args()
    takes = args.take or [
        *load_split(args.split)["train"],
        *load_split(args.split)["val"],
    ]
    hw = (int(args.input_hw[0]), int(args.input_hw[1]))
    jobs = [(t, c) for t in takes for c in CAMERAS]
    root, out = str(args.root), str(args.out)
    failures: list[str] = []
    time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_build_one, root, out, t, c, hw, args.threads, args.force): (
                t,
                c,
            )
            for t, c in jobs
        }
        built = 0
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001  (report every failure, then fail)
                failures.append(f"{futures[future]}: {exc}")
                continue
            built += result["status"] == "built"
            if result["status"] == "built" or done % 50 == 0:
                print(f"[{done}/{len(jobs)}] {json.dumps(result)}", flush=True)  # noqa: T201
        if args.verify and not failures:
            futures = {
                pool.submit(_verify_one, root, out, t, c, hw, args.verify): (t, c)
                for t, c in jobs
            }
            for future in as_completed(futures):
                try:
                    problem = future.result()
                except Exception as exc:  # noqa: BLE001
                    problem = f"{futures[future]}: {exc}"
                if problem:
                    failures.append(problem)
            if not failures:
                print(f"verify: {len(jobs)} streams byte-identical to the decode path")  # noqa: T201
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)  # noqa: T201
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
