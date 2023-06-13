from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import cv2
import more_itertools as mit
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from hydra.utils import instantiate
from jaxtyping import Float
from omegaconf import DictConfig
from pytorch_grad_cam import ActivationsAndGradients
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor
from torchvision.transforms import Normalize


class Unnormalize(Normalize):
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        **kwargs: Any,
    ) -> None:
        _mean: Float[Tensor, "c"] = torch.tensor(mean)
        _std: Float[Tensor, "c"] = torch.tensor(std)

        super().__init__(
            mean=(-_mean / _std).tolist(),
            std=(1.0 / _std).tolist(),
            **kwargs,
        )


class CILppWrapper(pl.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        images_transform: DictConfig,
        visualize_target: DictConfig,
        visualize_layers: List[str],
        labels: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        # This is a workaround as the grad-cam library expects .forward(x: Tensor)
        self._camera = None
        self.cilpp = instantiate(model)
        self.target_layers = [
            eval(layer, {"model": self.cilpp}) for layer in visualize_layers
        ]
        self.target = instantiate(visualize_target)
        self.images_transform = instantiate(images_transform)
        self.cam_cls = NegativeAndPositiveGradCAM

        self.font = labels or DictConfig({})
        self.add_labels = bool(labels)

    def prepare_input_tensor(self, batch) -> Float[Tensor, "i b t c h w"]:
        frames, speed, turn_signal, camera = self.cilpp.unpack_batch_for_predictions(
            batch
        )
        # keep the camera for the next forward call
        self._camera = camera

        # the speed and frames are tensors, so it's possible to put them into
        # a single tensor and unpack on self.forward to match CILpp forward
        b, t, c, h, w = frames.shape
        speed = repeat(speed, "b t -> b t c h w", c=c, h=h, w=w)
        turn_signal = repeat(turn_signal, "b t -> b t c h w", c=c, h=h, w=w)
        input_tensor = torch.stack([frames, speed, turn_signal], dim=0)

        return input_tensor

    def forward(
        self, input_tensor: Float[Tensor, "i b t c h w"]
    ) -> Float[Tensor, "b c"]:
        frames, speed, turn_signal = input_tensor.unbind(dim=0)
        speed = speed[..., 0, 0, 0]
        turn_signal = turn_signal[..., 0, 0, 0].to(torch.int64)

        pred = self.cilpp(
            frames=frames,
            speed=speed,
            turn_signal=turn_signal,
            camera=self._camera,
        )
        pred = rearrange(pred, "b 1 c -> b c")
        return pred

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> List[np.ndarray]:
        images = rearrange(
            get_images(batch, self.images_transform), "b c h w -> b h w c"
        )
        batch_size = images.shape[0]
        targets = [self.target] * batch_size

        visualizations = []
        with torch.inference_mode(False), torch.enable_grad():
            self.requires_grad_(True)
            input_tensor = self.prepare_input_tensor(batch)
            with self.cam_cls(
                model=self,
                target_layers=self.target_layers,
                reshape_transform=partial(reshape_transform, batch_size=batch_size),
                use_cuda=True,
            ) as cam:
                heatmaps = cam(input_tensor=input_tensor, targets=targets)
                predictions = cam.scalars_outputs

        for img, heatmap in zip(images, heatmaps):
            visualization = show_cam_on_image(img.cpu().numpy(), heatmap, use_rgb=True)
            visualizations.append(visualization)

        if self.add_labels:
            labels = self.cilpp._compute_labels(batch)
            put_labels(  # type: ignore
                visualizations,
                labels,
                predictions,
                target=self.target,
                **self.font,  # type: ignore
            )
        return visualizations


class ActivationsAndGradientsWithMemory(ActivationsAndGradients):
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Union[Callable, None] = None,
    ) -> None:
        super().__init__(model, target_layers, reshape_transform)
        self.outputs = None

    def __call__(self, *args, **kwargs) -> Float[Tensor, "..."]:
        self.outputs = super().__call__(*args, **kwargs)
        return self.outputs  # type: ignore[return-value]

    def release(self) -> None:
        self.outputs = None
        super().release()


class NegativeAndPositiveGradCAM(BaseCAM):
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable,
        use_cuda: bool = False,
    ) -> None:
        super().__init__(model, target_layers, use_cuda, reshape_transform)
        # Replace activations with gradients with a subclass that keeps outputs
        # thus saves forward pass output
        self.activations_and_grads.release()  # type: ignore
        self.activations_and_grads = ActivationsAndGradientsWithMemory(
            self.model, target_layers, reshape_transform
        )
        self.scalars_outputs = np.array([])

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        with torch.no_grad():
            self.scalars_outputs = np.array(
                [
                    target(output).detach().cpu().numpy()
                    for target, output in zip(
                        targets, self.activations_and_grads.outputs  # type: ignore
                    )
                ]
            )
        _, c, h, w = grads.shape
        out = repeat(self.scalars_outputs, "b -> b c h w", c=c, h=h, w=w)
        axes = (-2, -1)

        negative_grads = np.where(np.logical_and(out < 0, grads < 0), -grads, 0)
        negative_elems = np.sum(grads < 0, axis=axes)
        negative_mean = np.sum(negative_grads, axis=axes) / (negative_elems + 1e-8)

        positive_grads = np.where(np.logical_and(out >= 0, grads >= 0), grads, 0)
        positive_elems = np.sum(grads >= 0, axis=axes)
        positive_mean = np.sum(positive_grads, axis=axes) / (positive_elems + 1e-8)

        return negative_mean + positive_mean


class RawAccelerationTarget(ClassifierOutputTarget):
    def __init__(self) -> None:
        super().__init__(category=0)


class RawSteeringTarget(ClassifierOutputTarget):
    def __init__(self) -> None:
        super().__init__(category=1)


def reshape_transform(
    tensor: Float[Tensor, "B ..."], batch_size: int
) -> Float[Tensor, "b ..."]:
    tensor = rearrange(tensor, "(b t) ... -> b t ...", b=batch_size)
    # pick only the last frame from t
    tensor = tensor[:, -1, ...]
    return tensor


def get_images(batch, transform: Callable) -> Float[Tensor, "..."]:
    clips = mit.one(batch["clips"].values())
    frames = clips["frames"][:, -1, ...]
    imgs = transform(frames)
    return imgs.clamp(min=0.0, max=1.0)


def put_labels(
    visualizations: List[np.ndarray],
    labels: Dict[str, Float[Tensor, "b 1"]],
    predictions: np.ndarray,
    target: Union[RawSteeringTarget, RawAccelerationTarget],
    cv2_font: str = "FONT_HERSHEY_PLAIN",
    font_scale: int = 1,
    font_color: Sequence[int] = (255, 255, 255),
    font_thickness: int = 2,
) -> None:
    if isinstance(target, RawAccelerationTarget):
        gt_labels = labels["acceleration"]
    elif isinstance(target, RawSteeringTarget):
        gt_labels = labels["steering_angle"]
    else:
        raise RuntimeError(f"Unknown label for the target {target.__class__.__name__}")

    font = getattr(cv2, cv2_font)
    for image, gt_label, predicted in zip(visualizations, gt_labels, predictions):
        gt_text = f"gt: {gt_label.item():.4f}"
        pred_text = f"pred: {predicted:.4f}"
        (gt_width, gt_height), _ = cv2.getTextSize(
            gt_text, font, font_scale, font_thickness
        )

        gt_position = (10, gt_height + 10)
        pred_position = (10 + gt_width + 20, gt_height + 10)

        cv2.putText(
            image,
            gt_text,
            gt_position,
            font,
            font_scale,
            font_color,  # type: ignore
            font_thickness,
        )
        cv2.putText(
            image,
            pred_text,
            pred_position,
            font,
            font_scale,
            font_color,  # type: ignore
            font_thickness,
        )
