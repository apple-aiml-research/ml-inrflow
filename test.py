#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, Trainer

import hydra
from utils.modelrunner import ModelRunner
from utils.utils import (
    extras,
    create_folders,
    task_wrapper,
)
from utils.instantiators import instantiate_callbacks
from utils.logging_utils import log_hyperparameters
from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def test(cfg):
    log.info(f"Loading checkpoint for {cfg.ckpt_file}...")
    runner = ModelRunner(cfg)
    model = runner.get_ckpt()
    model.viz_dir = cfg.paths.viz_dir

    pl.seed_everything(42, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=[], plugins=None
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": [],
        "trainer": trainer,
    }

    if log:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting evaluation!")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_file)


@hydra.main(version_base="1.3", config_path="configs", config_name="base_eval.yaml")
def submit_run(cfg):
    extras(cfg)
    create_folders(cfg)
    test(cfg)


if __name__ == "__main__":
    submit_run()
