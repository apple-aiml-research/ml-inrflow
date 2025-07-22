#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
from torch import nn
from timm.models.vision_transformer import Mlp
from einops import rearrange, repeat

from model.nets.layers import CrossAttentionLayer, EfficientCrossAttentionLayer
from model.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_temp_embed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        cross_attention_layer=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        if cross_attention_layer is None:
            self.cross_attn = None
        else:
            self.cross_attn = cross_attention_layer()

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        latents,
        c,
        y=None,
        **kwargs,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        latents = latents + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa)
        )

        if y is not None and self.cross_attn is not None:
            latents = latents + self.cross_attn(
                inputs=latents,
                hidden_states=y,
            )[0]

        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class PerceiverEncoderBlock(nn.Module):

    def __init__(
        self,
        cross_attention_layer,
        self_attention_layer,
        num_self_attends=2,
        hidden_size=512,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.ca_adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.ca_ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ca_ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.cross_attention_layer = cross_attention_layer()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ca_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        # Construct a single block of self-attention layers.
        # We get deeper architectures by applying this block more than once.
        _self_attention_layers = []
        _mlp_layers = []
        _adaln_layers = []
        _ln1_layers = []
        _ln2_layers = []
        for _ in range(num_self_attends):

            ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            _ln1_layers.append(ln1)
            _ln2_layers.append(ln2)

            adnln_zero_layer = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
            _adaln_layers.append(adnln_zero_layer)

            sa_layer = self_attention_layer()

            _self_attention_layers.append(sa_layer)

            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            mlp_layer = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
            _mlp_layers.append(mlp_layer)

        self.ln1_layers = nn.ModuleList(_ln1_layers)
        self.ln2_layers = nn.ModuleList(_ln2_layers)
        self.adaln_layers = nn.ModuleList(_adaln_layers)
        self.sa_layers = nn.ModuleList(_self_attention_layers)
        self.mlp_layers = nn.ModuleList(_mlp_layers)

    def forward(self, latents=None, c=None, **kwargs):

        inputs = kwargs.get("inputs")

        # 1. Cross attend inputs to latents
        shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.ca_adaln(
            c
        ).chunk(6, dim=1)

        latents = latents + gate_mca.unsqueeze(1) * self.cross_attention_layer(
            inputs=modulate(self.ca_ln1(latents), shift_mca, scale_mca),
            hidden_states=inputs,
        )

        latents = latents + gate_mlp.unsqueeze(1) * self.ca_mlp(
            modulate(self.ca_ln2(latents), shift_mlp, scale_mlp)
        )

        # 2. Self-attention on latents with AdaLN-Zero

        for adaln_layer, ln1, sa_layer, ln2, mlp_layer in zip(
            self.adaln_layers,
            self.ln1_layers,
            self.sa_layers,
            self.ln2_layers,
            self.mlp_layers,
        ):

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                adaln_layer(c).chunk(6, dim=1)
            )

            latents = latents + (
                gate_msa.unsqueeze(1)
                * sa_layer(modulate(ln1(latents), shift_msa, scale_msa))
            )

            latents = latents + gate_mlp.unsqueeze(1) * mlp_layer(
                modulate(ln2(latents), shift_mlp, scale_mlp)
            )

        return latents


class PerceiverDecoderBlock(nn.Module):

    def __init__(
        self,
        cross_attention_layer,
        hidden_size=512,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.ca_adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.ca_ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ca_ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.ca_layer = cross_attention_layer()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ca_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.constant_(self.ca_adaln[-1].weight, 0)
        nn.init.constant_(self.ca_adaln[-1].bias, 0)

    def forward(
        self,
        queries,
        latents,
        c,
        **kwargs,
    ):

        # 1. Cross attend latents to queries
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ca_adaln(
            c
        ).chunk(6, dim=1)

        queries = queries + gate_msa.unsqueeze(1) * self.ca_layer(
            inputs=modulate(self.ca_ln1(queries), shift_msa, scale_msa),
            hidden_states=latents,
            **kwargs,
        )

        queries = queries + gate_mlp.unsqueeze(1) * self.ca_mlp(
            modulate(self.ca_ln2(queries), shift_mlp, scale_mlp)
        )

        return queries
