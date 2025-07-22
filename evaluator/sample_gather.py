#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from torchmetrics import Metric
import torch

from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class SampleGather(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("samples", default=[], dist_reduce_fx="cat")

    def update(self, inputs) -> None:
        self.samples.append(inputs)

    def compute(self):

        samples_out = dim_zero_cat(self.samples)

        return samples_out
