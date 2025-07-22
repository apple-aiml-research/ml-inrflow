#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import os
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, Trainer, LightningModule
import hydra
from omegaconf import OmegaConf, open_dict
from utils.utils import (
    extras,
    create_folders,
    task_wrapper,
)
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_trainer,
)
from utils.logging_utils import log_hyperparameters
from utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg):
    plugins = None

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    else:
        pl.seed_everything(42, workers=True)

    # train from scratch
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    # loggers = []  # instantiate_loggers(cfg.get("logger"))
    OmegaConf.set_struct(cfg.logger, True)
    loggers = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate_trainer(
        cfg.trainer, callbacks=callbacks, logger=loggers, plugins=plugins
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if log:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="base_train.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    extras(cfg)
    create_folders(cfg)
    train(cfg)
    return


if __name__ == "__main__":
    submit_run()
