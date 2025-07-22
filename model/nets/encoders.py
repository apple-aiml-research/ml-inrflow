#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from model.nets.layers import RMSNorm

from typing import Optional
from einops import rearrange, repeat, reduce
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchCAEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        out_channels,
        coord_num_channels=3,
        img_size=128,
        patch_size=8,
        num_heads=8,
        qkv_bias=False,
        use_flash=True,
        use_rmsnorm=False,
        **kwargs,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "dim should be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.use_flash = use_flash

        self.context_y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)

        self.q_norm = RMSNorm(self.head_dim) if use_rmsnorm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if use_rmsnorm else nn.Identity()

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=out_channels,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=0,
        )

        self.img_size = img_size
        self.patch_size = patch_size
        num_patch_h = int(img_size // patch_size)
        self.num_latents = num_patch_h**2
        N = img_size**2
        indices = torch.arange(N)
        indices = rearrange(indices, "(h w) -> h w", h=img_size)
        indices = indices.unfold(0, patch_size, patch_size).unfold(
            1, patch_size, patch_size
        )
        indices = indices.reshape(-1)
        self.register_buffer("permute_indices", indices)

        x_channel = (
            torch.linspace(0, 1, patch_size).view(1, 1, -1).repeat(1, patch_size, 1)
        )
        y_channel = (
            torch.linspace(0, 1, patch_size).view(1, -1, 1).repeat(1, 1, patch_size)
        )
        patch_pos = torch.cat((x_channel, y_channel), dim=0)
        patch_pos = rearrange(patch_pos, "c h w -> (h w) c")
        patch_pos = repeat(patch_pos, "p c -> (n p) c", n=self.num_latents)
        patch_pos_embed = self.get_2d_pos_embed(hidden_size, patch_pos)
        self.register_buffer("patch_pos_embed", patch_pos_embed)

        self.center_latent = nn.Parameter(
            torch.zeros(self.num_latents, hidden_size), requires_grad=True
        )
        center_coord = torch.ones(1, 2) * 0.5
        center_coord_embed = self.get_2d_pos_embed(hidden_size, center_coord)
        center_coord = center_coord.squeeze(0)
        center_coord_embed = center_coord_embed.squeeze(0)
        self.register_buffer("center_coord", center_coord)
        self.register_buffer("center_coord_embed", center_coord_embed)

    def get_2d_pos_embed(self, embed_dim, coord):
        assert embed_dim % 2 == 0
        emb_h = self.get_1d_sincos_pos_embed_from_coord(
            embed_dim // 2, coord[..., 0]
        )  # (M, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_coord(
            embed_dim // 2, coord[..., 1]
        )  # (M, D/2)
        emb = torch.cat([emb_h, emb_w], dim=-1)  # (*, M, D)
        return emb

    def get_1d_sincos_pos_embed_from_coord(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = 2 ** torch.linspace(0, math.log(224, 2) - 1, embed_dim // 2).to(
            pos.device
        )
        omega *= torch.pi

        if len(pos.shape) == 1:
            out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        elif len(pos.shape) == 2:
            out = torch.einsum("nm,d->nmd", pos, omega)

        emb_sin = torch.sin(out)  # (*, M, D/2)
        emb_cos = torch.cos(out)  # (*, M, D/2)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (*, M, D)
        return emb

    def forward(self, context_y, context_x=None):
        """
        x: (B, N, C)
        coords: (B, N, 2)
        """
        latent = self.center_latent + self.center_coord_embed[None, :]
        q = self.q(latent)
        q = repeat(
            q, "m (h d) -> (b m) h n d", b=context_y.shape[0], h=self.num_heads, n=1
        )

        # group contexts by patches
        context_y = context_y[:, self.permute_indices, :]
        pos_embed = self.patch_pos_embed.unsqueeze(0).to(context_y.dtype)
        x = self.context_y_proj(context_y) + pos_embed
        x = rearrange(x, "b (t n) c -> (b t) n c", t=self.num_latents)
        kv = self.kv(x)
        kv = rearrange(kv, "b n (s h d) -> s b h n d", s=2, h=self.num_heads)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_flash:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = reduce(x, "(b t) h n d -> b t (h d)", "sum", t=self.num_latents)

        x = self.norm(x)
        x = self.mlp(x)

        return x


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(
        self,
        latents,
        block,
        sampling_points=False,
    ):
        super().__init__()

        # Construct the cross attention layer.
        self.sampling_points = sampling_points
        self.latents = latents
        self.block = block

    def forward(
        self,
        inputs=None,
        c=None,
        context_coords=None,
    ):

        n_points = inputs.shape[1]
        latents = self.latents(batch_size=inputs.shape[0])

        if self.sampling_points:
            sample_idx = torch.randperm(n_points)[: self.sampling_points]
            inputs_block = inputs[:, sample_idx]
            if context_coords is not None:
                context_coords_block = context_coords[:, sample_idx]
        else:
            inputs_block = inputs
            context_coords_block = context_coords

        latents = self.block(latents=latents, c=c, inputs=inputs_block)

        return latents


class PatchLinearEncoder(nn.Module):

    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        out_channels,
        mixing_block,
        depth=1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_patch_h = int(img_size // patch_size)
        self.num_latents = num_patch_h**2

        self.z_proj = nn.Linear(in_channels * patch_size**2, out_channels, bias=True)
        self.z_proj_ln = nn.LayerNorm(out_channels, eps=1e-6)

        x_channel = (
            torch.linspace(0, 1, num_patch_h).view(1, 1, -1).repeat(1, num_patch_h, 1)
        )
        y_channel = (
            torch.linspace(0, 1, num_patch_h).view(1, -1, 1).repeat(1, 1, num_patch_h)
        )
        patch_pos = torch.cat((x_channel, y_channel), dim=0)
        patch_pos = rearrange(patch_pos, "c h w -> (h w) c")

        self.patch_pos_embed = self.get_2d_pos_embed(out_channels, patch_pos)
        self.mixing_blocks = nn.ModuleList([mixing_block() for _ in range(depth)])

    def patchify(self, x):
        x = rearrange(x, "b (h w) c -> b c h w", h=self.img_size)

        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x  # [n, l, d]

    def forward(self, x, c):
        x = self.patchify(x)
        x = self.z_proj(x)
        x = self.z_proj_ln(x)

        x = x + self.patch_pos_embed.to(x.device)

        for i, mixing_block in enumerate(self.mixing_blocks):
            x = mixing_block(latents=x, c=c)

        return x

    def get_2d_pos_embed(self, embed_dim, coord):
        assert embed_dim % 2 == 0
        emb_h = self.get_1d_sincos_pos_embed_from_coord(
            embed_dim // 2, coord[..., 0]
        )  # (M, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_coord(
            embed_dim // 2, coord[..., 1]
        )  # (M, D/2)
        emb = torch.cat([emb_h, emb_w], dim=-1)  # (*, M, D)
        return emb

    def get_1d_sincos_pos_embed_from_coord(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = 2 ** torch.linspace(0, math.log(224, 2) - 1, embed_dim // 2).to(
            pos.device
        )
        omega *= torch.pi

        if len(pos.shape) == 1:
            out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        elif len(pos.shape) == 2:
            out = torch.einsum("nm,d->nmd", pos, omega)

        emb_sin = torch.sin(out)  # (*, M, D/2)
        emb_cos = torch.cos(out)  # (*, M, D/2)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (*, M, D)
        return emb
