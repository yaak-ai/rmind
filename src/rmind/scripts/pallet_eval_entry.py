"""Score trained keypoint nets (`pallet_train_entry` checkpoints) on a labelled pallet set.

    uv run python -m rmind.scripts.pallet_eval_entry --labels .../entry_test.json \
        --boxes .../entry_test.boxes.json --checkpoints .../best.pt ...

Crops come from `--boxes` (the label set's `.boxes.json` by default; a null box falls back to
the labelled points' bounds). Same metrics as training: raw-pixel error on labelled points
(median, p90, PCK at 5 / 10 px). Writes `--out` JSON when given.
"""

import argparse
import json
from pathlib import Path

import torch
from structlog import get_logger
from torch.utils.data import DataLoader

from rmind.pallet.correspondence import NAMES
from rmind.pallet.entry import EntryKeypoints, load_labels
from rmind.pallet.keypoints import PalletKeypointNet
from rmind.scripts.pallet_train_entry import evaluate

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, default=Path("/"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    labels = load_labels(args.labels)
    boxes = json.loads(
        (args.boxes or args.labels.with_suffix(".boxes.json")).read_text()
    )
    loader = DataLoader(
        EntryKeypoints(args.images, labels, boxes, train=False),
        batch_size=16,
        num_workers=4,
    )
    results = {}
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = PalletKeypointNet(
            ck["backbone"], num_keypoints=len(NAMES), pretrained=False
        )
        net.load_state_dict(ck["state_dict"])
        key = f"{path.parent.parent.name}/{path.parent.name}/{path.stem}"
        results[key] = evaluate(net.to(args.device), loader, args.device)
        logger.info(
            "result",
            model=key,
            **{
                a: round(b, 3) if isinstance(b, float) else b
                for a, b in results[key].items()
            },
        )
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
