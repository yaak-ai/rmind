import json
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw
from tensordict import TensorDict
from torch.nn import Module


class ApplyMask(Module):
    def __init__(self, mask_path: str, img_w: int = 576, img_h: int = 324):
        super().__init__()
        with open(mask_path) as f:
            mask_json = json.load(f)

        self.mask = self._create_binary_mask(img_w, img_h, mask_json["mask_polygon"])

    def forward(self, tensor):
        return tensor * self.mask.expand_as(tensor).to(tensor.device)

    def _create_binary_mask(self, img_w, img_h, polygon_points) -> torch.Tensor:
        mask = Image.new("L", (img_w, img_h), 0)  # 'L' mode for grayscale
        draw = ImageDraw.Draw(mask)

        # Convert polygon points to a list of tuples
        polygon = [(point["x"], point["y"]) for point in polygon_points]

        # Draw the polygon on the mask
        draw.polygon(polygon, fill=1)  # Fill the polygon with white (255)

        return 1 - torch.Tensor(np.array(mask)).type(
            torch.uint8
        )  # Convert to numpy array for further processing
