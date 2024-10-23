import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import Module


class ApplyMask(Module):
    def __init__(
        self,
        mask_path: str,
        img_w: int = 576,
        img_h: int = 324,
        transforms: Module | None = None,
    ):
        super().__init__()
        with Path.open(mask_path, encoding="utf-8") as f:
            mask_json = json.load(f)

        self.transforms = transforms
        self.mask = self._create_binary_mask(img_w, img_h, mask_json["mask_polygon"])

    def forward(self, tensor):
        return tensor * self.mask.expand_as(tensor).to(tensor.device)

    def _create_binary_mask(self, img_w, img_h, polygon_points) -> torch.Tensor:
        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)

        polygon = [(point["x"], point["y"]) for point in polygon_points]

        draw.polygon(polygon, fill=1)

        mask = 1 - torch.Tensor(np.array(mask)).type(torch.uint8)
        return self.transforms(mask) if self.transforms else mask
