#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
from torch import nn
from torch.nn import functional as F
from model.nets.layers import FinalLayer
from einops import rearrange, repeat


class PixelDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        encoder,
        trunk,
        decoder,
        time_embedder,
        pos_embedder,
        label_embedder=None,
        hidden_size=1152,
        depth=12,
        coord_num_channels=2,
        signal_num_channels=3,
        **kwargs,
    ):
        super().__init__()

        self.pos_embedder = pos_embedder
        self.time_embedder = time_embedder
        self.label_embedder = label_embedder

        self.encoder = encoder
        self.trunk = trunk
        self.decoder = decoder

        self.query_y_proj = nn.Sequential(
            nn.Linear(signal_num_channels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.final_layer = FinalLayer(
            hidden_size * 2, signal_num_channels, c_dim=hidden_size
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

        # Initialize label embedding table:
        try:
            nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)
        except:
            print("Not using label embedding")

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.trunk.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.decoder.blocks:
            nn.init.constant_(block.ca_adaln[-1].weight, 0)
            nn.init.constant_(block.ca_adaln[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y=None,
        label=None,
    ):
        """
        Forward pass of DiT.
        x: (N, L, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        query_coords = query_x

        t_emb = self.time_embedder(t)  # (N, D)
        if label is not None and self.label_embedder is not None:
            c_emb = self.label_embedder(label, self.training)  # (N, D)
            c_emb = t_emb + c_emb  # (N, D)
        else:
            c_emb = t_emb

        latents = self.encoder(context_y)
        if type(latents) == tuple:
            latents = latents[0]

        latents = self.trunk(latents=latents, c=c_emb)

        query_y = self.query_y_proj(query_y)
        query_cat = self.pos_embedder.apply_pos_embed(query_y, pos=query_x)

        query_pred = self.decoder(
            queries=query_cat,
            latents=latents,
            c=c_emb,
            query_coords=query_coords,
        )

        query_pred = torch.cat([query_pred, query_y], dim=2)
        query_pred = self.final_layer(query_pred, c=c_emb)

        return query_pred

    def forward_with_cfg(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y=None,
        label=None,
        cfg_scale=1.0,
        **kwargs
    ):
        half = query_y[: len(query_y) // 2]
        query_y_combined = torch.cat([half, half], dim=0)
        half = context_y[: len(context_y) // 2]
        context_y_combined = torch.cat([half, half], dim=0)
        model_out = self.forward(
            context_x=context_x,
            context_y=context_y_combined,
            t=t,
            query_x=query_x,
            query_y=query_y_combined,
            label=label,
        )
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps


class PointDiT(nn.Module):

    def __init__(
        self,
        encoder,
        trunk,
        decoder,
        time_embedder,
        pos_embedder,
        condition_embedder=None,
        hidden_size=1152,
        depth=12,
        coord_num_channels=2,
        signal_num_channels=3,
        **kwargs,
    ):
        super().__init__()
        self.pos_embedder = pos_embedder
        query_pos_embed_channels = pos_embedder.embed_dim

        self.time_embedder = time_embedder
        self.condition_embedder = condition_embedder

        self.encoder = encoder
        self.trunk = trunk
        self.decoder = decoder

        self.context_x_proj = nn.Sequential(
            nn.Linear(query_pos_embed_channels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.context_y_proj = nn.Sequential(
            nn.Linear(signal_num_channels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.context_cat_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.query_x_proj = nn.Sequential(
            nn.Linear(query_pos_embed_channels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.query_y_proj = nn.Sequential(
            nn.Linear(signal_num_channels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.query_cat_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.final_layer = FinalLayer(
            hidden_size * 2, signal_num_channels, c_dim=hidden_size
        )

        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize label embedding table:
        if hasattr(self, "class_embed"):
            nn.init.normal_(self.class_embed.embedding_table.weight, std=0.02)

        # # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.encoder.blocks:
            nn.init.constant_(block.ca_adaln[-1].weight, 0)
            nn.init.constant_(block.ca_adaln[-1].bias, 0)

            for ada_ln in block.adaln_layers:
                nn.init.constant_(ada_ln[-1].weight, 0)
                nn.init.constant_(ada_ln[-1].bias, 0)

        for block in self.decoder.blocks:
            nn.init.constant_(block.ca_adaln[-1].weight, 0)
            nn.init.constant_(block.ca_adaln[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y=None,
        label=None,
        cond_drop_ids=None,
    ):
        """
        Forward pass of DiT.
        x: (N, L, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        query_coords = query_x

        c_emb = self.time_embedder(t)  # (N, D)
        cond = None
        if label is not None and self.condition_embedder is not None:
            cond = self.condition_embedder(label, self.training, force_drop_ids=cond_drop_ids)

        context_x = self.pos_embedder(pos=context_y)
        context_x = self.query_x_proj(context_x)
        context_y = self.query_y_proj(context_y)
        context_cat = torch.cat([context_x, context_y], dim=2)
        context_cat = self.context_cat_proj(context_cat)

        latents = self.encoder(inputs=context_cat, c=c_emb)

        latents = self.trunk(
            latents=latents,
            c=c_emb,
            inputs=context_cat,
            y=cond,
        )

        query_x = self.pos_embedder(pos=query_y)
        query_x = self.query_x_proj(query_x)
        query_y = self.query_y_proj(query_y)
        query_cat = torch.cat([query_x, query_y], dim=2)
        query_cat = self.query_cat_proj(query_cat)

        query_pred = self.decoder(
            queries=query_cat,
            latents=latents,
            c=c_emb,
            query_coords=query_coords,
        )

        query_pred = torch.cat([query_pred, query_y], dim=2)
        query_pred = self.final_layer(query_pred, c=c_emb)

        return query_pred

    def forward_with_cfg(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y=None,
        label=None,
        cfg_scale=1.0,
        cond_drop_ids=None,
    ):
        half = query_y[: len(query_y) // 2]
        query_y_combined = torch.cat([half, half], dim=0)
        half = context_y[: len(context_y) // 2]
        context_y_combined = torch.cat([half, half], dim=0)
        model_out = self.forward(
            context_x=context_x,
            context_y=context_y_combined,
            t=t,
            query_x=query_x,
            query_y=query_y_combined,
            label=label,
            cond_drop_ids=cond_drop_ids,
        )
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps
