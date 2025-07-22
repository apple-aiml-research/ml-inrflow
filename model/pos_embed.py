#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange


class AbsoluteEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim, resolution, include_input=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = embed_dim
        self.resolution = resolution
        self.include_input = include_input
        assert embed_dim % in_dim == 0, "embed_dim must be divisible by in_dim"
        self.embed_dim = embed_dim + in_dim if include_input else embed_dim

        grid = self.get_grid()
        pos_emb = self.get_pos_embed(grid)
        self.register_buffer('pos_emb', pos_emb)

    def reset_resolution(self, resolution):
        self.resolution = resolution
        grid = self.get_grid()
        pos_emb = self.get_pos_embed(grid)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, pos=None, resolution=None):
        if resolution is not None and resolution != self.resolution:
            self.resolution = resolution
            grid = self.get_grid()
            pos_emb = self.get_pos_embed(grid).to(self.device)
            self.register_buffer('pos_emb', pos_emb)

        if pos is None:
            return self.pos_emb
        else:
            return self.get_pos_embed(pos)

    def apply_pos_embed(self, x, pos=None, resolution=None):
        return x + self(pos, resolution).to(x)

    def get_grid(self):
        grids = []
        for _ in range(self.in_dim):
            grid = torch.arange(self.resolution).float()
            grids.append(grid)
        grids = torch.meshgrid(*grids)
        grids = torch.stack(grids, dim=0)
        grids = grids.reshape(self.in_dim, -1).transpose(0, 1)
        grids /= (self.resolution - 1)
        return grids

    def get_pos_embed(self, pos):
        pos_embs = []
        for i in range(self.in_dim):
            pe = self.get_1d_pos_embed(pos[..., i])
            pos_embs.append(pe)
        if self.include_input:
            pos_embs.append(pos)
        pos_embs = torch.cat(pos_embs, dim=-1)
        return pos_embs

    def get_1d_pos_embed(self, pos):
        """
        https://github.com/facebookresearch/DiT/blob/main/models.py#L303
        """

        embed_dim = self.hidden_dim // (self.in_dim * 2)
        omega = 2 ** torch.linspace(0, math.log(224, 2) - 1, embed_dim).to(pos.device)
        omega *= torch.pi

        if len(pos.shape) == 1:
            out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
        elif len(pos.shape) == 2:
            out = torch.einsum('nm,d->nmd', pos, omega)

        emb_sin = torch.sin(out) # (*, M, D/2)
        emb_cos = torch.cos(out) # (*, M, D/2)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (*, M, D)
        return emb


class FastFourierEmbed(torch.nn.Module):
    """Module for positional encodings."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)  # (nf,)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)  # (nf,)

        self.register_buffer('freq_bands', freq_bands)  # (nf,)
        self.embed_dim = out_dim + d * self.freq_bands.numel() * 2

    def forward(self, pos):
        # x: (*, d)
        x = pos

        out = []
        if self.kwargs["include_input"]:
            out = [x]  # (*b, d)

        x = x.unsqueeze(-1) * self.freq_bands  # (*b, d, nf)
        out += [
            torch.sin(x).flatten(start_dim=-2),  # (*b, d*nf)
            torch.cos(x).flatten(start_dim=-2),  # (*b, d*nf)
        ]
        out = torch.cat(out, dim=-1)  # (*b, 2*d*nf)
        return out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


# interpolate Fourier positional embedding 
def interpolate_2d_sincos_pos_embed(embed_dim, grid_size=127, default_grid_size=64):
    """
    embed_dim: output dimension for each position
    grid_size: size of the grid
    out: (grid_size, grid_size, D)
    """
    assert embed_dim % 2 == 0
    
    x_channel = torch.linspace(0, 1, default_grid_size).view(1, 1, -1).repeat(1, default_grid_size, 1)
    y_channel = torch.linspace(0, 1, default_grid_size).view(1, -1, 1).repeat(1, 1, default_grid_size)
    grid = torch.cat((x_channel, y_channel), dim=0)
    grid = rearrange(grid, "c h w -> (h w) c")
    pos_embed = get_2d_sincos_pos_embed_from_coord(embed_dim, grid)
    pos_embed = rearrange(pos_embed, "(h w) c -> c h w", h=default_grid_size)
    pos_embed = pos_embed.unsqueeze(0)
    
    pos_embed = F.interpolate(pos_embed, size=(grid_size, grid_size), mode='bicubic')
    pos_embed = pos_embed.squeeze(0)
    pos_embed = rearrange(pos_embed, "c h w -> (h w) c")

    return pos_embed


def get_2d_sincos_pos_embed_from_coord(embed_dim, coord):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_coord(embed_dim // 2, coord[..., 0])  # (M, D/2)
    emb_w = get_1d_sincos_pos_embed_from_coord(embed_dim // 2, coord[..., 1])  # (M, D/2)

    emb = torch.cat([emb_h, emb_w, coord], dim=-1) # (*, M, D)

    return emb

def get_3d_sincos_pos_embed_from_coord(embed_dim, coord):
    assert embed_dim % 3 == 0

    emb_f = get_1d_sincos_pos_embed_from_coord(embed_dim // 3, coord[..., 0])  # (M, D/3)
    emb_h = get_1d_sincos_pos_embed_from_coord(embed_dim // 3, coord[..., 1])  # (M, D/3)
    emb_w = get_1d_sincos_pos_embed_from_coord(embed_dim // 3, coord[..., 2])  # (M, D/3)

    emb = torch.cat([emb_f, emb_h, emb_w, coord], dim=-1) # (*, M, D)

    return emb

def get_1d_sincos_pos_embed_from_coord(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = 2 ** torch.linspace(0, math.log(224, 2) - 1, embed_dim // 2).to(pos.device)
    omega *= torch.pi

    if len(pos.shape) == 1:
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    elif len(pos.shape) == 2:
        out = torch.einsum('nm,d->nmd', pos, omega)

    emb_sin = torch.sin(out) # (*, M, D/2)
    emb_cos = torch.cos(out) # (*, M, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (*, M, D)
    return emb

def get_1d_sincos_temp_embed(embed_dim, length):
    """get 1d temporal positional embedding"""
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
