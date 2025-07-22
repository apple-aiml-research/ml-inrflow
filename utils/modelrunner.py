#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
import hydra
from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ModelRunner:
    def __init__(self, cfg) -> None:
        cfg.model._target_ = "model.inrflow.INRFlow"
        self.model = hydra.utils.instantiate(cfg.model)
        self.cfg = cfg

    def get_ckpt(self):
        if not hasattr(self.cfg, "ckpt_file") or self.cfg.ckpt_file is None:
            print("No checkpoint file specified, using initial model.")
            return self.model
        self.model.load_state_dict(
            torch.load(self.cfg.ckpt_file, map_location="cpu")["state_dict"],
            strict=False,
        )
        self.model.eval()
        return self.model