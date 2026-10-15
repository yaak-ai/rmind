"""Camera-layout configuration shared by audit and training launchers."""

from omegaconf import OmegaConf

CAMERA_LAYOUTS = {
    "forward": ("cam_fork", "cam_left_forward", "cam_right_forward"),
    "backward": ("cam_fork", "cam_left_backward", "cam_right_backward"),
    "fork": ("cam_fork",),
}


def configure_cameras(config, layout):
    cameras = CAMERA_LAYOUTS[layout]
    config.cameras = list(cameras)
    config.num_cameras = len(cameras)
    config.image_remap = OmegaConf.create({
        camera: ["data", camera] for camera in cameras
    })
    config.wandb.tags = [tag for tag in config.wandb.tags if tag != "three_cameras"] + [
        "single_camera" if len(cameras) == 1 else "three_cameras",
        f"camera_layout:{layout}",
    ]
    config.wandb.notes += f" Camera order: {list(cameras)}."
    if layout == "fork":
        config.wandb.notes = config.wandb.notes.replace(
            "three-camera", "fork-camera-only"
        )


def dataset_overrides(layout):
    return ["datamodule=palletjack/d12_fm_backward"] if layout == "backward" else []
