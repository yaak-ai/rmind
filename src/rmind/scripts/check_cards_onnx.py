"""Run an exported palletjack ONNX policy on command-card images and report the actions.

This is the acceptance check for the overfit smoke test, and doubles as the
reference preprocessing for the kit: the model's exported graph starts *after*
crop/resize/normalize, so whatever runs on the vehicle has to reproduce exactly
what `_preprocess` does here.

    uv run --extra export python -m rmind.scripts.check_cards_onnx \
        --onnx outputs/.../model.onnx

By default it runs the generated card frames and compares against the labels in
`cards.json`. Pass `--images` to run photographs of the printed cards instead —
that is the test that actually exercises the print -> camera -> model path:

    uv run --extra export python -m rmind.scripts.check_cards_onnx \
        --onnx model.onnx --images photos/*.jpg
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.transforms import v2

ACTIONS: Final = ("traction", "steering", "fork1")

# the kit's camera frame, which `generate_cards` also renders at
FRAME_HEIGHT: Final = 324
FRAME_WIDTH: Final = 576

# must mirror `image_transform` in the experiment config, minus augmentation
_TRANSFORM: Final = v2.Compose([
    v2.CenterCrop(size=[320, 576]),
    v2.Resize(size=[256, 256], antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _to_camera_frame(image: torch.Tensor) -> torch.Tensor:
    """Scale to cover, then crop to the camera frame — a no-op for card frames.

    Photographs of printed cards come in whatever resolution the phone shot them
    at, so they have to be reduced to what the camera would have seen before the
    kit's own preprocessing applies. Cover-then-crop rather than a plain resize:
    no aspect distortion and no padded bars, neither of which the model ever saw.
    """
    _, height, width = image.shape
    scale = max(FRAME_WIDTH / width, FRAME_HEIGHT / height)
    resized = v2.functional.resize(
        image, size=[round(height * scale), round(width * scale)], antialias=True
    )

    return v2.functional.center_crop(resized, output_size=[FRAME_HEIGHT, FRAME_WIDTH])


def _preprocess(path: Path, *, episode_length: int) -> np.ndarray:
    """Image file -> [1, episode_length, 3, 256, 256] float32, as the graph expects."""
    image = torch.from_numpy(np.array(Image.open(path).convert("RGB")))
    frame = _TRANSFORM(_to_camera_frame(image.permute(2, 0, 1)))

    return frame.expand(1, episode_length, *frame.shape).numpy()


def _outputs(session: ort.InferenceSession) -> dict[str, str]:
    """Map action name -> ONNX output name (`policy.continuous.<action>`).

    Raises:
        ValueError: if an action does not resolve to exactly one graph output.
    """
    names = {output.name for output in session.get_outputs()}
    resolved = {}
    for action in ACTIONS:
        matches = sorted(name for name in names if name.endswith(action))
        if len(matches) != 1:
            msg = f"expected exactly one output ending in {action!r}, found {matches}"
            raise ValueError(msg)

        resolved[action] = matches[0]

    return resolved


def _feed(session: ort.InferenceSession, frames: np.ndarray) -> dict[str, np.ndarray]:
    """Bind `frames` to the camera input and zeros to any auxiliary input.

    The control-transformer graph takes the camera alone; the patch-policy trunk
    also takes a `speed` token, which the card rig has no source for and trained
    at constant zero. Binding by name keeps one checker valid for both.
    """
    feed: dict[str, np.ndarray] = {}
    for spec in session.get_inputs():
        if spec.name.startswith("cam"):
            feed[spec.name] = frames
        else:
            shape = [d if isinstance(d, int) else 1 for d in spec.shape]
            feed[spec.name] = np.zeros(shape, dtype=np.float32)

    return feed


def run(
    session: ort.InferenceSession, path: Path, *, episode_length: int
) -> dict[str, float]:
    outputs = _outputs(session)
    results = session.run(
        list(outputs.values()),
        _feed(session, _preprocess(path, episode_length=episode_length)),
    )

    return {
        action: float(np.asarray(value).reshape(-1)[-1])
        for action, value in zip(outputs, results, strict=True)
    }


def _report(
    rows: Sequence[tuple[str, dict[str, float], dict[str, float] | None]],
    *,
    tolerance: float,
) -> int:
    header = f"{'card':<12}" + "".join(f"{a:>27}" for a in ACTIONS)
    print(header)  # ruff: ignore[print]
    print("-" * len(header))  # ruff: ignore[print]

    worst = 0.0
    for name, predicted, expected in rows:
        cells = ""
        for action in ACTIONS:
            if expected is None:
                cells += f"{predicted[action]:>27.3f}"
            else:
                error = abs(predicted[action] - expected[action])
                worst = max(worst, error)
                cells += f"{predicted[action]:>+13.3f} (exp {expected[action]:+.2f})"

        print(f"{name:<12}{cells}")  # ruff: ignore[print]

    if rows[0][2] is None:
        return 0

    print(f"\nworst absolute error: {worst:.4f} (tolerance {tolerance})")  # ruff: ignore[print]

    return 0 if worst <= tolerance else 1


def main(  # ruff: ignore[too-many-arguments]
    onnx: Path,
    *,
    cards_dir: Path,
    images: Sequence[Path] | None,
    episode_length: int,
    tolerance: float,
    expect_zero: bool,
) -> int:
    session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])

    if images:
        # `--expect-zero` asserts the negative class: anything that is not a card
        # must command a full stop
        zero = dict.fromkeys(ACTIONS, 0.0)
        rows = [
            (
                path.stem,
                run(session, path, episode_length=episode_length),
                zero if expect_zero else None,
            )
            for path in images
        ]

        return _report(rows, tolerance=tolerance)

    cards = json.loads((cards_dir / "cards.json").read_text())
    rows = [
        (
            card["name"],
            run(
                session,
                cards_dir / "frames" / f"{card['index']:06d}.jpg",
                episode_length=episode_length,
            ),
            {action: card[action] for action in ACTIONS},
        )
        for card in cards
    ]

    return _report(rows, tolerance=tolerance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--cards-dir", type=Path, default=Path("data/palletjack/cards"))
    parser.add_argument(
        "--images", type=Path, nargs="*", help="run these images instead of the cards"
    )
    parser.add_argument("--episode-length", type=int, default=6)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="max absolute error before exiting non-zero; commands are capped at 0.3",
    )
    parser.add_argument(
        "--expect-zero",
        action="store_true",
        help="with --images: require every output to be ~0 (the negative class)",
    )
    args = parser.parse_args()

    raise SystemExit(
        main(
            args.onnx,
            cards_dir=args.cards_dir,
            images=args.images,
            episode_length=args.episode_length,
            tolerance=args.tolerance,
            expect_zero=args.expect_zero,
        )
    )
