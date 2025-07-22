#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import os
import shlex
import subprocess
import torch


from utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)
dtype_lookup = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


def instantiate_trainer(trainer_cfg: DictConfig, callbacks, logger, plugins):

    if "mixed_precision" in trainer_cfg.strategy.keys():

        # mp_config = trainer_cfg.st
        mp = hydra.utils.instantiate(trainer_cfg.strategy.mixed_precision)
        mp = mp(
            param_dtype=dtype_lookup[trainer_cfg.strategy.mixed_precision.param_dtype],
            reduce_dtype=dtype_lookup[
                trainer_cfg.strategy.mixed_precision.reduce_dtype
            ],
            buffer_dtype=dtype_lookup[
                trainer_cfg.strategy.mixed_precision.buffer_dtype
            ],
        )
        strategy = hydra.utils.instantiate(trainer_cfg.strategy)
        strategy.mixed_precision = mp
        trainer = hydra.utils.instantiate(
            trainer_cfg,
            strategy=strategy,
            callbacks=callbacks,
            logger=logger,
            plugins=plugins,
        )
    else:
        trainer = hydra.utils.instantiate(
            trainer_cfg, callbacks=callbacks, logger=logger, plugins=plugins
        )

    return trainer


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))
            if "TensorBoard" in lg_conf._target_:
                build_tensorboard(lg_conf.save_dir)
    return logger


def build_tensorboard(summary_name):
    tbp = os.environ.get("TENSORBOARD_PORT")
    command = "tensorboard --logdir {} --port {} --bind_all".format(summary_name, tbp)

    subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
