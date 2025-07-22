#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from torch import nn
import functools


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PerceiverDecoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(
        self,
        block,
        depth=1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([block() for _ in range(depth)])

    def forward(
        self,
        queries=None,
        latents=None,
        c=None,
        **kwargs,
    ):

        assert len(self.blocks) == len(latents)
        for i, block in enumerate(self.blocks):
            queries = block(queries, latents[i], c, **kwargs)
        return queries
