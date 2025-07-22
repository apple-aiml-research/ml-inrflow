#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import os
import torch
from einops import rearrange
import math

import math
from inspect import isfunction
from omegaconf import OmegaConf, open_dict

import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
from omegaconf import DictConfig
from utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def noise_like(shape, noise_fn, device, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])

        return noise_fn(*shape_one, device=device).repeat(shape[0], *resid)

    else:
        return noise_fn(*shape, device=device)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def identity(t, *args, **kwargs):
    return t


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def clamp_log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def center_of_mass_norm(x, mask):
    x[~mask.bool()] = 0
    denom = torch.sum(mask, -1, keepdim=True)
    denom = denom.unsqueeze(-1)
    y_mean = torch.sum(x, dim=1, keepdim=True) / denom
    x -= y_mean
    return x


def convert_to_coord_format(h, w, device="cpu"):
    x_channel = torch.linspace(0, 1, w, device=device).view(1, 1, -1).repeat(1, w, 1)
    y_channel = torch.linspace(0, 1, h, device=device).view(1, -1, 1).repeat(1, 1, h)
    return torch.cat((x_channel, y_channel), dim=0)


def get_random_coordinates(resolution=64, npoints=1024, ndim=2):
    coord = convert_to_coord_format(resolution, resolution)
    coord = rearrange(coord, "c h w -> (h w) c")
    sampled_indices = torch.randperm(coord.shape[0])[:npoints]
    coord = coord[sampled_indices, :]
    return coord, sampled_indices


def create_folders(cfg: DictConfig) -> None:
    """Creates paths for folders"""

    if not cfg.get("paths"):
        log.warning("Paths config not found! <cfg.extras=null>")
        return

    # Create folders
    if not os.path.exists(cfg.paths.viz_dir):
        os.makedirs(cfg.paths.viz_dir)

    if not os.path.exists(cfg.paths.data_dir):
        os.makedirs(cfg.paths.data_dir)

    if not os.path.exists(cfg.paths.tmp_dir):
        os.makedirs(cfg.paths.tmp_dir)

    if not os.path.exists(cfg.paths.eval_dir):
        os.makedirs(cfg.paths.eval_dir)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=False, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
