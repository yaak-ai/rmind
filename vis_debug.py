from typing import List

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import bertviz
import pytorch_grad_cam
from hydra.utils import instantiate
import torch

import more_itertools as mit
from deephouse.tools.camera import Camera
from pytorch_grad_cam.base_cam import BaseCAM
from torchvision.transforms import Normalize

from cargpt.models.cilpp import CILpp

import pytorch_lightning as pl
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Shaped
from torch import Tensor

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet34, resnet50


class Unnormalize(Normalize):
    def __init__(self, mean, std, **kwargs):
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        super().__init__(
            mean=(-mean / std).tolist(),
            std=(1.0 / std).tolist(),
            **kwargs,
        )

def get_unnorm_frames(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    clips = mit.one(batch["clips"].values())
    frames = rearrange(clips["frames"], "b 1 c h w -> b c h w")

    unorm = Unnormalize(mean, std)
    imgs = unorm(frames)
    return imgs.clamp(min=0.0, max=1.0)


class WrapperModel(pl.LightningModule):
    def __init__(self, cilpp):
        super().__init__()
        self.cilpp = cilpp
        self._camera = None

        cilpp.requires_grad_(True)
        cilpp.eval()

        # self.cuda()
        self.requires_grad_(True)
        self.eval()

    def prepare_input_tensor(self, batch):
        clips = mit.one(batch["clips"].values())
        frames = rearrange(clips["frames"], "b 1 c h w -> b c h w")

        meta = clips["meta"]
        speed: Float[Tensor, "b 1"] = meta["VehicleMotion_speed"].to(torch.float32)

        if any(camera_params := clips.get("camera_params", {}).copy()):
            camera_model = mit.one(set(camera_params.pop("model")))
            camera = Camera.from_params(model=camera_model, params=camera_params)
            camera = camera.to(frames)
        else:
            camera = None

        self._camera = camera

        b, c, h, w = frames.shape
        speed = repeat(speed, 'b sc -> b c h w sc', c=c, h=h, w=w)
        frames = rearrange(frames, "b c h (w fc) -> b c h w fc", fc=1)
        input_tensor = torch.concat([frames, speed], dim=-1)
        input_tensor = rearrange(input_tensor, "b c h w A -> b A c h w")
        return input_tensor

    def forward(self, input_tensor):
        input_tensor = rearrange(input_tensor, "b A c h w -> b c h w A")
        frames = input_tensor[..., 0]
        speed = input_tensor[:, 0, 0, 0, 1:]
        camera = self._camera
        pred = self.cilpp(frames=frames, speed=speed, camera=camera)
        pred = rearrange(pred, "b 1 c -> b c")
        return pred


class NegPosGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super().__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(input_tensor)  # [B, 2]
            if outputs.shape[0] > 1: raise NotImplementedError("Only batch size == 1 for now")
            scalars_outputs = [target(output)
                       for target, output in zip(targets, outputs)]

        out = scalars_outputs[0]
        if out < 0:
            grads = -1.0 * np.where(grads < 0, grads, 0)
        else:
            grads = np.where(grads >= 0, grads, 0)

        return np.mean(grads, axis=(2, 3))


class RawAccelerationTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return model_output[0]


class RawSteeringTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return model_output[1]


wandb_model = "yaak/cargpt/model-uv8xv088:v0"
cilpp = CILpp.load_from_wandb_artifact(name=wandb_model)
# cilpp = CILpp.load_from_checkpoint("artifacts/model-rc93mcrx:v7/model.ckpt")
# cilpp.eval()

cfg = OmegaConf.load("config/experiment/cilpp.yaml")
datamodule = instantiate(cfg.datamodule)

data = datamodule.val_dataloader()

batch = next(iter(data))
imgs = rearrange(get_unnorm_frames(batch), 'b c h w -> b h w c')

wrapper = WrapperModel(cilpp)
target_layers = [next(cilpp.state_embedding.modules())['frame']['backbone'].resnet.layer4[-1]]

idx=2
input_tensor = wrapper.prepare_input_tensor(batch)[idx: idx+1]

with NegPosGradCAM(model=wrapper, target_layers=target_layers, use_cuda=True) as cam:
    targets = [RawAccelerationTarget()]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

visualization = show_cam_on_image(imgs[idx].numpy(), grayscale_cam[0], use_rgb=True)
plt.imshow(visualization);
plt.show()

print(grayscale_cam.shape)
print()
