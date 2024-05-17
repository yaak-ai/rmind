import multiprocessing as mp
import sys

import hydra
import rerun as rr
import rerun.blueprint as rrb
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from cargpt.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="config", config_name="dataviz.yaml")
def run(cfg: DictConfig):
    dataloader = instantiate(cfg.dataloader)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="frames/"),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="meta", contents=["$origin/VehicleMotion_speed"]
                ),
                rrb.TimeSeriesView(
                    origin="meta",
                    contents=["+ $origin/**", "- $origin/VehicleMotion_speed"],
                ),
            ),
        )
    )

    for batch in tqdm(dataloader):
        for clip in batch.select("frames", "meta", strict=False):
            drive_id = (
                clip.meta.pop("drive_id")
                .to(torch.uint8)
                .numpy()
                .tobytes()
                .decode("ascii")
            )

            if rr.get_application_id(rr.get_global_data_recording()) != drive_id:
                recording = rr.new_recording(
                    application_id=drive_id,
                    make_default=True,
                    make_thread_default=False,
                    spawn=False,
                    default_enabled=True,
                )

                rr.connect(
                    recording=recording,
                    default_blueprint=blueprint,
                    flush_timeout_sec=None,
                )

            for elem in clip.auto_batch_size_():
                rr.set_time_sequence(
                    "/".join(k := ("meta", "cam_front_left/ImageMetadata_frame_idx")),
                    elem.pop(k).item(),
                )

                rr.set_time_nanos(
                    "/".join(k := ("meta", "cam_front_left/ImageMetadata_time_stamp")),
                    elem.pop(k).item() * 1000,
                )

                for k, v in elem.items(True, True):
                    if not v.isnan().all().item():
                        path = "/".join(k)
                        match k:
                            case ("frames", *_):
                                rr.log(path, rr.Image(v.permute(1, 2, 0)))

                            case ("meta", *_):
                                rr.log(path, rr.Scalar(v))

                            case _:
                                raise NotImplementedError


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("spawn", force=True)
    mp.set_forkserver_preload(["torch"])
    setup_logging()

    run()


if __name__ == "__main__":
    main()
