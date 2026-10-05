"""Fill a nutron-cli patch replay bundle with rmind's offline chunks (+ input parity).

The offline half of the serving parity loop (nutron-cli `runtime/jetson/patch_replay.py`
documents the bundle):

    # 1. serving side: a raw episode through the PRODUCER's observation builder
    policy_check.py --patch-bundle EPISODE --contract ART/policy_contract.json --out b.npz
    # 2. this script: rmind's WINDOWED forward over the bundle's own inputs
    python -m rmind.scripts.nero_replay_expected --ckpt model.ckpt --artifact ART \\
        --bundle b.npz --out b_expected.npz [--episode EPISODE]
    # 3. serving side: stream the bundle through patch_runtime and compare
    policy_check.py --patch-replay b_expected.npz --contract ART/policy_contract.json

Step 2 runs the TRAINED model (not the export) exactly as training does: one
windowed forward per stream (the bundle's `reset` flags split the streams; a reset
restarts the frame counter, so each stream is its own forward), fp32 with TF32 off,
argmax codes. The standardized chunk is unstandardized with the tokenizer's action
standardizer and the relative mode is undone with each frame's RAW state as the
anchor, giving `expected_actions (N, 100, 26)` ABSOLUTE and side-major, plus
`expected_codes (N, Q)` of the first valid side (the graph's `codes` output).

`--episode` additionally checks INPUT parity (what step 3 cannot see): for every
bundle tick it finds rbyte's `NeroRobotReader` row with the same base
`t_ns` and compares the raw state, the composed hand token (the policy's hand
groups, `rmind.data.nero_robot.compose_hand_token`), and the three images decoded
by rbyte's `TorchCodecVideoSource` at the row's `frame_index.<camera>` and mapped
by `rmind.data.nero_image.preprocess` -- the training data path. Non-zero exit when
any input differs (state/hand: exactly; images: `--image-atol` uint8 levels).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from structlog import get_logger

from rmind.data.nero_robot import compose_hand_token, to_absolute
from rmind.models.nero_patch_policy import NeroPatchPolicy

logger = get_logger(__name__)


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def streams(reset: np.ndarray) -> list[tuple[int, int]]:
    starts = [int(i) for i in np.flatnonzero(reset)]
    if not starts or starts[0] != 0:
        msg = "bundle reset[0] must be True"
        raise ValueError(msg)
    return list(zip(starts, [*starts[1:], len(reset)], strict=True))


@torch.no_grad()
def expected(  # noqa: PLR0914
    policy: NeroPatchPolicy,
    bundle: dict[str, np.ndarray],
    manifest: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, chunk, S*A) ABSOLUTE chunks and (N, Q) first-valid-side codes."""
    n_sides = len(manifest["sides"])
    side_valid = torch.tensor(manifest["side_valid"], dtype=torch.bool, device=device)
    cond = torch.tensor(
        manifest["camera_cond"]["values"], dtype=torch.float32, device=device
    )
    first_valid = int(side_valid.to(torch.int64).argmax())
    images = bundle["images_u8"]  # (N, n_cams, 3, H, W) uint8
    state = torch.from_numpy(bundle["state"]).float()
    out_actions, out_codes = [], []
    for start, stop in streams(bundle["reset"]):
        t = stop - start
        raw = state[start:stop].reshape(t, n_sides, -1)
        batch: dict[str, Any] = {
            policy.state[0]: raw.unsqueeze(0).to(device),
            policy.side_valid[0]: side_valid.unsqueeze(0),
            policy.camera_cond[0]: cond.unsqueeze(0),
        }
        for index, camera in enumerate(policy.cameras):
            # uint8, as the training data path delivers it
            batch[policy.image_key.format(camera=camera)] = (
                torch.from_numpy(images[start:stop, index]).unsqueeze(0).to(device)
            )
        if policy.use_hand:
            batch[policy.hand_token_key] = (
                torch
                .from_numpy(bundle["hand_token"][start:stop])
                .float()
                .unsqueeze(0)
                .to(device)
            )
        features = policy._features(batch)  # noqa: SLF001  (1, t, d)
        chunk, codes = policy._predict_chunk_and_codes(features[0])  # noqa: SLF001
        # (t, S, H, A) standardized -> raw -> absolute (anchor = this frame's raw state)
        unstd = policy.tokenizer.standardizer.unstandardize(
            chunk.permute(0, 2, 1, 3)
        )  # (t, H, S, A)
        absolute = to_absolute(unstd, raw.to(device), policy.relative_mode)
        out_actions.append(
            absolute.reshape(t, absolute.shape[1], -1).float().cpu().numpy()
        )
        out_codes.append(codes[:, first_valid].cpu().numpy())
    return np.concatenate(out_actions), np.concatenate(out_codes)


def input_parity(  # noqa: PLR0914
    policy: NeroPatchPolicy,
    bundle: dict[str, np.ndarray],
    episode: Path,
    *,
    image_atol: int,
) -> dict[str, Any]:
    """Bundle inputs (serving's producer) vs rbyte's training rows for the same frames."""
    from rbyte.samples.nero import NeroRobotReader  # noqa: PLC0415
    from rbyte.streams.video import TorchCodecVideoSource  # noqa: PLC0415

    from rmind.data.nero_image import preprocess  # noqa: PLC0415

    df = NeroRobotReader(chunk_size=100)(episode / "data.mcap")
    t_rows = df["t_ns"].to_numpy()
    where = {int(t): i for i, t in enumerate(t_rows)}
    ticks = bundle["t_ns"].astype(np.int64)
    rows = [where.get(int(t)) for t in ticks]
    missing = [int(t) for t, r in zip(ticks, rows, strict=True) if r is None]
    keep = np.array([r is not None for r in rows])
    idx = np.array([r for r in rows if r is not None], dtype=np.int64)
    report: dict[str, Any] = {
        "ticks": len(ticks),
        "matched_rows": int(keep.sum()),
        "ticks_without_rbyte_row": len(missing),
    }

    state_rb = (
        np.stack(df["state"].to_numpy()[idx]).astype(np.float32).reshape(len(idx), -1)
    )
    state_b = bundle["state"][keep]
    report["state_max_abs"] = (
        float(np.abs(state_rb - state_b).max()) if len(idx) else None
    )
    report["state_rows_differing"] = int(
        (np.abs(state_rb - state_b).max(axis=1) > 0).sum()
    )

    if policy.use_hand:
        blocks = {}
        for col in df.columns:
            if col.startswith(policy.hand_prefix):
                blocks[col] = torch.from_numpy(np.stack(df[col].to_numpy()[idx]))
        token_rb = compose_hand_token(
            blocks, policy.hand_groups, prefix=policy.hand_prefix
        ).numpy()
        token_b = bundle["hand_token"][keep]
        diff = np.abs(token_rb - token_b)
        report["hand_max_abs"] = float(diff.max()) if len(idx) else None
        report["hand_rows_differing"] = int((diff.max(axis=1) > 0).sum())
        report["hand_valid_rbyte"] = float(token_rb[:, -1].mean())
        report["hand_valid_bundle"] = float(token_b[:, -1].mean())
        bad = np.flatnonzero(diff.max(axis=1) > 0)
        report["hand_first_diffs"] = [
            {
                "t_ns": int(ticks[keep][i]),
                "rbyte": token_rb[i].round(4).tolist(),
                "bundle": token_b[i].round(4).tolist(),
            }
            for i in bad[:3]
        ]

    hw = tuple(int(v) for v in bundle["images_u8"].shape[-2:])
    images = {}
    for index, camera in enumerate(policy.cameras):
        source = TorchCodecVideoSource(source=str(episode / f"{camera}.mp4"))
        ordinals = [int(v) for v in df[f"frame_index.{camera}"].to_numpy()[idx]]
        frames = source[ordinals]  # (n, H, W, 3) or (n, 3, H, W) uint8
        if frames.shape[-1] == 3:  # noqa: PLR2004
            frames = frames.permute(0, 3, 1, 2)
        grid = torch.stack([preprocess(f, hw) for f in frames]).numpy()
        d = np.abs(
            grid.astype(np.int16) - bundle["images_u8"][keep, index].astype(np.int16)
        )
        images[camera] = {"max_levels": int(d.max()), "mean_levels": float(d.mean())}
    report["images"] = images
    # rbyte drops the rows whose chunk is more than half hold-padding (the last
    # ~1.7 s of a run) -- serving still ticks there. A missing row anywhere ELSE
    # is a disagreement about which frames exist.
    last = int(ticks[keep].max()) if keep.any() else -1
    report["ticks_without_rbyte_row_mid_run"] = sum(t < last for t in missing)
    report["ok"] = bool(
        keep.any()
        and report["ticks_without_rbyte_row_mid_run"] == 0
        and report["state_rows_differing"] == 0
        and (not policy.use_hand or report["hand_rows_differing"] == 0)
        and all(v["max_levels"] <= image_atol for v in images.values())
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--artifact", type=Path, required=True, help="nero_export --out dir"
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--episode", type=Path, help="raw episode dir: also check input parity vs rbyte"
    )
    parser.add_argument("--image-atol", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    policy = NeroPatchPolicy.load_from_checkpoint(
        args.ckpt, map_location="cpu", weights_only=False
    )
    policy = policy.to(device).eval()
    policy.sample_codes = False
    manifest = json.loads((args.artifact / "policy_manifest.json").read_text())
    if policy.tokenizer.standardizer.digest != _artifact_digest(
        args.artifact, manifest
    ):
        msg = "the checkpoint's action standardizer is not the artifact's"
        raise SystemExit(msg)

    bundle = load_bundle(args.bundle)
    actions, codes = expected(policy, bundle, manifest, device=device)
    bundle["expected_actions"] = actions.astype(np.float32)
    bundle["expected_codes"] = codes.astype(np.int64)
    np.savez(args.out, **bundle)
    summary: dict[str, Any] = {
        "frames": len(actions),
        "streams": len(streams(bundle["reset"])),
        "finite": bool(np.isfinite(actions).all()),
        "out": args.out.as_posix(),
    }
    failed = not summary["finite"]
    if args.episode is not None:
        parity = input_parity(
            policy.cpu(), bundle, args.episode, image_atol=args.image_atol
        )
        summary["input_parity"] = parity
        failed |= not parity["ok"]
    if failed:
        sys.exit(1)


def _artifact_digest(artifact: Path, manifest: dict[str, Any]) -> str:
    from rmind.data.nero_robot import AxisStandardizer  # noqa: PLC0415

    return AxisStandardizer.load(
        artifact / manifest["standardizers"]["action"]["file"]
    ).digest


if __name__ == "__main__":
    main()
