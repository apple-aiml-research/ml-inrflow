#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch


class DefaultProcessor:
    def __init__(self, npoints_context, npoints_query, sampling, device):
        self.npoints_context = npoints_context
        self.npoints_query = npoints_query
        self.device = device
        self.sampling = sampling

    def preprocess_training(self, batch):
        label = batch.get("label")
        x = batch.get("coord")
        y = batch.get("signal")
        noise = torch.randn_like(y)

        # use all points as context
        # we do sample in encoder
        context_x = x
        context_y = y
        context_noise = noise

        # subsample query points if needed
        if self.npoints_query[0] < x.shape[1]:
            if self.sampling == "uniform":
                # random sample queries
                npoints_query = self.npoints_query[0]
                query_indices = torch.randperm(x.shape[1])[:npoints_query]
        else:
            query_indices = torch.arange(x.shape[1])

        query_x = x[:, query_indices]
        query_y = y[:, query_indices]
        query_noise = noise[:, query_indices]

        return (
            context_x,
            context_y,
            query_x,
            query_y,
            context_noise,
            query_noise,
            label,
        )

    def preprocess_inference(self, batch):
        label = batch.get("label")
        x = batch.get("coord")
        y = batch.get("signal")
        return x, y, label

    def postprocess(self, x, y, y_sampled):

        return x, y, y_sampled
