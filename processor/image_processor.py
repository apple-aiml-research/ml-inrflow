#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
from einops import rearrange


class ImageProcessor:
    def __init__(self, npoints_context, npoints_query, sampling, compressor, device):

        self.npoints_context = npoints_context
        self.npoints_query = npoints_query
        self.sampling = sampling
        self.device = device
        self.compressor = compressor().to(self.device)
        self.compressor.requires_grad_(False)

    def preprocess_training(self, batch):
        label = batch.get("label")
        x = batch.get("coord")
        y = batch.get("signal")

        y = rearrange(
            y,
            "b (h w) c -> b c h w",
            h=batch.get("resolution")[0],
            w=batch.get("resolution")[0],
        )
        x = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=batch.get("resolution")[0],
            w=batch.get("resolution")[0],
        )

        y = self.compressor.encode(y.to(self.device)).latent_dist.sample().mul_(0.18215)
        self.downsampled_resolution = y.shape[2]
        y = rearrange(y, "b c h w -> b (h w) c")
        self.downsample_factor = int(
            batch.get("resolution")[0] / self.downsampled_resolution
        )
        x = x[:, :, :: self.downsample_factor, :: self.downsample_factor]
        x = rearrange(x, "b c h w -> b (h w) c")

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

        y = rearrange(
            y,
            "b (h w) c -> b c h w",
            h=batch.get("resolution")[0],
            w=batch.get("resolution")[0],
        )

        x = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=batch.get("resolution")[0],
            w=batch.get("resolution")[0],
        )

        y = self.compressor.encode(y.to(self.device)).latent_dist.sample().mul_(0.18215)
        x = x[:, :, :: self.downsample_factor, :: self.downsample_factor]

        y = rearrange(y, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")

        return x, y, label

    def postprocess(self, x, y, y_sampled):

        y = rearrange(
            y,
            "b (h w) c -> b c h w",
            h=self.downsampled_resolution,
            w=self.downsampled_resolution,
        )
        y_sampled = rearrange(
            y_sampled,
            "b (h w) c -> b c h w",
            h=self.downsampled_resolution,
            w=self.downsampled_resolution,
        )

        y = self.compressor.decode(y / 0.18215).sample
        y_sampled = self.compressor.decode(y_sampled / 0.18215).sample

        y = rearrange(y, "b c h w -> b (h w) c").clamp(-1, 1)
        y_sampled = rearrange(y_sampled, "b c h w -> b (h w) c").clamp(-1, 1)

        return None, y, y_sampled
