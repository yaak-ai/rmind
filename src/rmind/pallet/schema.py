"""Label format v5: one record per image, every pallet in it, real and synthetic alike.

    {
      "image": "/abs/path.jpg",            # full-resolution frame
      "source": "real" | "synthetic",
      "dataset": "multi_v3" | "niantic_v2" | ...,
      "job": str | null, "frame": int | null, "camera": "cam_fork",
      "width": 1920, "height": 1080,
      "split": "train" | "val" | "test",
      "pallets": [
        {
          "id": str,                       # stable within the dataset (instance / track id)
          "box": [x0, y0, x1, y1] | null,  # visible pallet base + entry face, full-res px
          "loaded": true | false | null,   # something rests on the deck
          "type": "eur" | "other" | null,
          "on_floor": true | false | null, # false: in a rack or on top of another pallet
          "truncated": bool, "occluded": bool, "unsure": bool,
          "carried": bool,                 # already on the forks
          "keypoints": {name: [u, v, "visible" | "hidden" | "unknown"]},
          "pose": {                        # metric, camera frame (x forward, y left, z up)
            "entry_centre_m": [x, y, z], "yaw_deg": float, "source": "gt" | "model"
          } | null,
        }
      ]
    }

Keypoints use the entry-face names of `correspondence.NAMES` (four face corners, four
corners of each fork pocket); they describe any pallet with two fork openings, not only
EUR ones. Missing names are not labelled. No floor contact is assumed anywhere, so pallets
in racks and on stacks are first-class.
"""

from typing import Final, Literal

from rmind.pallet.correspondence import NAMES

KEYPOINTS: Final = tuple(NAMES)
Visibility = Literal["visible", "hidden", "unknown"]


def pocket_centres(kp: dict[str, list]) -> dict[str, tuple[float, float]]:
    """Centre of each fork opening, from its four corners when all are labelled."""
    out = {}
    for side in ("l", "r"):
        corners = [
            kp.get(f"pocket_{side}_{c}")
            for c in ("top_left", "top_right", "bottom_left", "bottom_right")
        ]
        if all(c is not None for c in corners):
            out[f"pocket_{side}_centre"] = (
                sum(c[0] for c in corners) / 4,
                sum(c[1] for c in corners) / 4,
            )
    return out


def box_of(points: list[list[float]], width: int, height: int) -> list[float] | None:
    """Bounding box of image points, clipped to the frame; None if nothing is inside."""
    pts = [(u, v) for u, v, *_ in points if 0 <= u < width and 0 <= v < height]
    if len(pts) < 2:  # ruff: ignore[magic-value-comparison]
        return None
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    return [min(us), min(vs), max(us), max(vs)]
