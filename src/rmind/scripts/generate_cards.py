"""Generate the synthetic "command card" dataset used to smoke-test the palletjack pipeline.

Each card is one printable sheet that commands one actuation combination. The
model is overfit on these cards so that holding a printed card in front of the
kit's camera produces the corresponding action — an end-to-end check of
preprocessing, inference and actuation plumbing without any real recording.

Emits, under `--output-dir`:

    frames/{:06d}.jpg   camera-resolution training frames (one per card)
    print/{name}.png    high-resolution sheets to print
    samples.parquet     rbyte samples: one row per episode
    cards.json          card index -> action values (for verification scripts)

Each card differs from every other in background colour, glyph, and - for the
turn pair - which side of the card the white mass sits on. The redundancy is
deliberate: probing a trained model showed it keys on the whole template rather
than on colour alone, so any single cue being degraded by the camera must not be
able to change the answer. LEFT and RIGHT matter most, since confusing them
yields a *wrong-direction* steering command, so they are separated by a
low-frequency asymmetry (offset glyph + edge bar) that survives blur.

Commands are capped at `--max-magnitude` (default 0.3): this rig exists to prove
the pipeline moves the right actuator in the right direction, not to move fast.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from PIL import Image, ImageDraw, ImageFont
from structlog import get_logger

logger = get_logger(__name__)

# camera-resolution frames, matching the yaak drive frames the kit pipeline
# already center-crops to 320x576 and resizes to 256x256
FRAME_SIZE: Final = (576, 324)
# A4 landscape at 300 dpi
PRINT_SIZE: Final = (3508, 2480)


class Card(NamedTuple):
    name: str
    color: tuple[int, int, int]
    glyph: str
    # fractions of `--max-magnitude`, not absolute commands - see `scaled`
    traction: float
    steering: float
    fork1: float
    # Horizontal glyph centre as a fraction of width, plus a full-height edge bar
    # on the same side. LEFT and RIGHT would otherwise differ only by a mirrored
    # glyph, whose fine detail blur destroys - and a model that loses the
    # distinction emits a *wrong-direction* steering command. Displacing a large
    # mass to one side is low-frequency, so it survives blur and downscaling.
    glyph_x: float = 0.5
    edge_bar: int = 0  # -1 left, 0 none, +1 right

    def scaled(self, magnitude: float) -> "Card":
        return self._replace(
            traction=self.traction * magnitude,
            steering=self.steering * magnitude,
            fork1=self.fork1 * magnitude,
        )

    @property
    def label(self) -> str:
        return (
            f"{self.name}   "
            f"traction {self.traction:+.2f}  "
            f"steering {self.steering:+.2f}  "
            f"fork1 {self.fork1:+.2f}"
        )


# Relative magnitudes, scaled by `--max-magnitude` so the safety cap is one flag.
# NOTE: turns command a non-zero traction too, so that a correct prediction has
# to get two outputs right at once.
CARDS: Final[tuple[Card, ...]] = (
    Card("FORWARD", (0, 158, 78), "up", +1.0, 0.0, 0.0),
    Card("REVERSE", (208, 32, 32), "down", -1.0, 0.0, 0.0),
    Card("LEFT", (16, 80, 208), "left", +0.5, -1.0, 0.0, glyph_x=0.30, edge_bar=-1),
    Card("RIGHT", (228, 176, 0), "right", +0.5, +1.0, 0.0, glyph_x=0.70, edge_bar=+1),
    Card("FORK1_UP", (176, 0, 152), "double_up", 0.0, 0.0, +1.0),
    Card("FORK1_DOWN", (0, 160, 192), "double_down", 0.0, 0.0, -1.0),
    Card("STOP", (16, 16, 16), "square", 0.0, 0.0, 0.0),
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        from matplotlib import font_manager  # ruff: ignore[import-outside-top-level]

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=size)
    except Exception:  # ruff: ignore[blind-except] - label text is cosmetic
        logger.warning("no scalable font found, falling back to bitmap font")

        return ImageFont.load_default()


def _arrow(
    draw: ImageDraw.ImageDraw, *, box: tuple[float, float, float, float], direction: str
) -> None:
    """Draw a solid arrow pointing `direction`, inscribed in `box`.

    Raises:
        ValueError: on an unsupported `direction`.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # head spans the full width of the box, the shaft a third of it
    tail = (x1 - x0) / 6

    match direction:
        case "up":
            points = ((x0, cy), (cx, y0), (x1, cy), (cx + tail, cy), (cx + tail, y1), (cx - tail, y1), (cx - tail, cy))  # fmt: skip
        case "down":
            points = ((x0, cy), (cx, y1), (x1, cy), (cx + tail, cy), (cx + tail, y0), (cx - tail, y0), (cx - tail, cy))  # fmt: skip
        case "left":
            points = ((cx, y0), (x0, cy), (cx, y1), (cx, cy + tail), (x1, cy + tail), (x1, cy - tail), (cx, cy - tail))  # fmt: skip
        case "right":
            points = ((cx, y0), (x1, cy), (cx, y1), (cx, cy + tail), (x0, cy + tail), (x0, cy - tail), (cx, cy - tail))  # fmt: skip
        case _:
            msg = f"unsupported direction: {direction}"
            raise ValueError(msg)

    draw.polygon(points, fill="white")


def _glyph(
    draw: ImageDraw.ImageDraw, *, glyph: str, box: tuple[float, float, float, float]
) -> None:
    x0, y0, x1, y1 = box

    match glyph:
        case "up" | "down" | "left" | "right":
            _arrow(draw, box=box, direction=glyph)

        case "double_up" | "double_down":
            direction = "up" if glyph == "double_up" else "down"
            gap = (y1 - y0) * 0.08
            half = (y1 - y0 - gap) / 2
            _arrow(draw, box=(x0, y0, x1, y0 + half), direction=direction)
            _arrow(draw, box=(x0, y1 - half, x1, y1), direction=direction)

        case "square":
            inset = (x1 - x0) * 0.12
            draw.rectangle(
                (x0 + inset, y0 + inset, x1 - inset, y1 - inset), fill="white"
            )

        case _:
            msg = f"unsupported glyph: {glyph}"
            raise ValueError(msg)


def render(card: Card, size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, color=card.color)
    draw = ImageDraw.Draw(image)

    # generous white border so the card reads as a card even when the camera
    # frames it loosely, and the glyph is never clipped by a center crop
    margin = min(width, height) * 0.06
    draw.rectangle(
        (margin, margin, width - margin, height - margin),
        outline="white",
        width=max(2, int(margin / 6)),
    )

    if card.edge_bar:
        bar_width = width * 0.11
        inner = margin * 1.9
        left = inner if card.edge_bar < 0 else width - inner - bar_width
        # stops above the label row so the text stays legible
        draw.rectangle((left, inner, left + bar_width, height * 0.78), fill="white")

    glyph_size = height * 0.5
    cx, cy = width * card.glyph_x, height * 0.46
    _glyph(
        draw,
        glyph=card.glyph,
        box=(
            cx - glyph_size / 2,
            cy - glyph_size / 2,
            cx + glyph_size / 2,
            cy + glyph_size / 2,
        ),
    )

    font = _font(int(height * 0.055))
    # centred on the card, not on the (possibly offset) glyph
    draw.text(
        (width / 2, height * 0.87), card.label, fill="white", font=font, anchor="mm"
    )

    return image


def build_samples(
    cards: Sequence[Card], *, episode_length: int, repeats: int
) -> pl.DataFrame:
    """One row per episode: `episode_length` copies of a single card's frame and action."""
    return pl.DataFrame(
        {
            "input_id": ["cards"] * (len(cards) * repeats),
            # the patch-policy trunk prepends a speed token per frame; a held-up
            # card is a standing-still command, and the rig has no speed source
            "speed": [[0.0] * episode_length for _ in cards for _ in range(repeats)],
            "frame_idx": [
                [i] * episode_length for i in range(len(cards)) for _ in range(repeats)
            ],
            "traction": [
                [c.traction] * episode_length for c in cards for _ in range(repeats)
            ],
            "steering": [
                [c.steering] * episode_length for c in cards for _ in range(repeats)
            ],
            "fork1": [
                [c.fork1] * episode_length for c in cards for _ in range(repeats)
            ],
        },
        schema={
            "input_id": pl.String,
            "speed": pl.Array(pl.Float32, episode_length),
            "frame_idx": pl.Array(pl.Int32, episode_length),
            "traction": pl.Array(pl.Float32, episode_length),
            "steering": pl.Array(pl.Float32, episode_length),
            "fork1": pl.Array(pl.Float32, episode_length),
        },
    )


def main(
    output_dir: Path, *, episode_length: int, repeats: int, max_magnitude: float
) -> None:
    frames_dir = output_dir / "frames"
    print_dir = output_dir / "print"
    for directory in (frames_dir, print_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cards = tuple(card.scaled(max_magnitude) for card in CARDS)

    for index, card in enumerate(cards):
        render(card, FRAME_SIZE).save(frames_dir / f"{index:06d}.jpg", quality=95)
        render(card, PRINT_SIZE).save(print_dir / f"{index:02d}_{card.name}.png")

    samples = build_samples(cards, episode_length=episode_length, repeats=repeats)
    samples.write_parquet(output_dir / "samples.parquet")

    (output_dir / "cards.json").write_text(
        json.dumps(
            [
                {
                    "index": i,
                    "name": c.name,
                    "traction": c.traction,
                    "steering": c.steering,
                    "fork1": c.fork1,
                }
                for i, c in enumerate(cards)
            ],
            indent=2,
        )
    )

    logger.info(
        "generated cards",
        output_dir=output_dir.resolve().as_posix(),
        cards=len(cards),
        episodes=len(samples),
        max_magnitude=max_magnitude,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/palletjack/cards")
    )
    parser.add_argument(
        "--episode-length", type=int, default=6, help="timesteps per episode"
    )
    parser.add_argument("--repeats", type=int, default=64, help="episodes per card")
    parser.add_argument(
        "--max-magnitude",
        type=float,
        default=0.3,
        help="largest commanded value; caps every actuation for safety",
    )
    args = parser.parse_args()

    main(
        args.output_dir,
        episode_length=args.episode_length,
        repeats=args.repeats,
        max_magnitude=args.max_magnitude,
    )
