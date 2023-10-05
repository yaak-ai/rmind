#!/usr/bin/env python
import multiprocessing as mp
import sys

import fiftyone as fo
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, DictConfig

from cargpt.utils.logging import setup_logging

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="config", config_name="dataviz.yaml")
def dataviz(cfg: DictConfig):
    exporter = instantiate(cfg.exporter)

    logger.debug("instantiating datasets", names=list(cfg.datasets.keys()))
    datasets_yaak = {name: instantiate(ds) for name, ds in cfg.datasets.items()}

    logger.debug("exporting datasets", names=list(datasets_yaak.keys()))
    datasets_51 = {name: exporter.export(ds) for name, ds in datasets_yaak.items()}

    logger.debug("merging datasets", names=list(datasets_51.keys()))
    dataset_51_merged = fo.Dataset("merged", overwrite=True)

    for name, ds in datasets_51.items():
        ds.name = name
        ds.tag_samples(name)
        dataset_51_merged.merge_samples(ds.iter_samples())

    logger.debug("merged", dataset=dataset_51_merged)

    session = fo.launch_app(dataset_51_merged, remote=True)
    session.wait()


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("spawn", force=True)
    setup_logging()

    dataviz()


if __name__ == "__main__":
    main()
