#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import math
from operator import __add__
import torch
from torch import nn
from einops import rearrange
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                            Attention Layers                                  #
#################################################################################


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_bias=True,
        qk_norm=True,
        resolution=None,
        patch_size=None,
        pos_embedder=None,
        linear_target: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = linear_target(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_target(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.pos_embedder = pos_embedder
        try:
            latent_res = resolution // patch_size
            self.pos_embedder.reset_resolution(latent_res)
            print(f"Resetting resolution in SelfAttention to {latent_res}")
        except:
            pass

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.pos_embedder:
            q = self.pos_embedder.apply_pos_embed(q, pos=None, head_first=True)
            k = self.pos_embedder.apply_pos_embed(k, pos=None, head_first=True)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientSelfAttentionLayer(SelfAttentionLayer):
    """Adapted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x, **kwargs):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")

        q, k, v = qkv.unbind(0)

        if self.pos_embedder:
            q = self.pos_embedder.apply_pos_embed(q, pos=None, head_first=True)
            k = self.pos_embedder.apply_pos_embed(k, pos=None, head_first=True)

        q, k = self.q_norm(q), self.k_norm(k)
        x = nn.functional.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_bias=True,
        qk_norm=True,
        # use_rope=False,
        resolution=None,
        patch_size=None,
        latents_pos_embedder=None,
        query_pos_embedder=None,
        linear_target: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = linear_target(hidden_size, hidden_size, bias=qkv_bias)
        self.kv = linear_target(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_target(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.latents_pos_embedder = latents_pos_embedder
        self.query_pos_embedder = query_pos_embedder
        try:
            latent_res = resolution // patch_size
            self.latents_pos_embedder.reset_resolution(latent_res)
            print(f"Resetting latent resolution in CrossAttention to {latent_res}")
        except:
            pass

    def forward(self, inputs, hidden_states, **kwargs):
        B, M, C_i = inputs.shape
        B, N, C_h = hidden_states.shape

        q = self.q(inputs).reshape(B, M, self.num_heads, C_i // self.num_heads)
        q = rearrange(q, "b n h c -> b h n c")

        kv = self.kv(hidden_states).reshape(
            B, N, 2, self.num_heads, C_h // self.num_heads
        )
        kv = rearrange(kv, "b n t h c -> t b h n c")

        k, v = (
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.query_pos_embedder:
            q = self.query_pos_embedder.apply_pos_embed(
                q, pos=kwargs.get("query_coords"), head_first=True
            )
        if self.latents_pos_embedder:
            latent_res = int(N ** 0.5)
            k = self.latents_pos_embedder.apply_pos_embed(
                k, pos=None, resolution=latent_res, head_first=True
            )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        outputs = (attn @ v).transpose(1, 2).reshape(B, N, C_i)
        outputs = self.proj(outputs)
        outputs = self.proj_drop(outputs)
        return outputs


class EfficientCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_bias=True,
        qk_norm=True,
        resolution=None,
        patch_size=None,
        latents_pos_embedder=None,
        query_pos_embedder=None,
        linear_target: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = linear_target(hidden_size, hidden_size, bias=qkv_bias)
        self.kv = linear_target(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_target(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.latents_pos_embedder = latents_pos_embedder
        self.query_pos_embedder = query_pos_embedder
        try:
            latent_res = resolution // patch_size
            self.latents_pos_embedder.reset_resolution(latent_res)
            print(f"Resetting latent resolution in CrossAttention to {latent_res}")
        except:
            pass

    def forward(self, inputs, hidden_states, **kwargs):
        B, M, C_i = inputs.shape
        B, N, C_h = hidden_states.shape

        q = self.q(inputs).reshape(B, M, self.num_heads, C_i // self.num_heads)
        q = rearrange(q, "b n h c -> b h n c")

        kv = self.kv(hidden_states).reshape(
            B, N, 2, self.num_heads, C_h // self.num_heads
        )
        kv = rearrange(kv, "b n t h c -> t b h n c")

        k, v = (
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.query_pos_embedder:
            q = self.query_pos_embedder.apply_pos_embed(
                q, pos=kwargs.get("query_coords"), head_first=True
            )
        if self.latents_pos_embedder:
            latent_res = int(N ** 0.5)
            k = self.latents_pos_embedder.apply_pos_embed(
                k, pos=None, resolution=latent_res, head_first=True
            )

        outputs = nn.functional.scaled_dot_product_attention(q, k, v)
        outputs = outputs.transpose(1, 2).reshape(B, M, C_i)
        outputs = self.proj(outputs)
        outputs = self.proj_drop(outputs)

        return outputs


#################################################################################
#                            Embedder Layers                                    #
#################################################################################


class PerceiverEmbeddings(nn.Module):
    """Construct the latent embeddings."""

    def __init__(self, num_latents, d_latents):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_latents))

    def set_latents(self, latents):
        self.latents = latents

    def forward(self, batch_size: int):
        return self.latents.expand(batch_size, -1, -1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        print("NUMBER OF CLASSES:", num_classes)
        self.dropout_prob = dropout_prob
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ConditionEmbedder(nn.Module):
    """
    Embeds condition tokens into vector representations. Also handles dropout for classifier-free guidance.
    """

    def __init__(
        self, 
        dim_cond_token, 
        dim_hidden, 
        cond_drop_prob, 
    ):
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_proj = Mlp(
            in_features=dim_cond_token, 
            hidden_features=dim_hidden, 
            out_features=dim_hidden, 
            act_layer=approx_gelu, 
            drop=0.0,
        )
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(dim_cond_token) / dim_cond_token ** 0.5))
        self.cond_drop_prob = cond_drop_prob

    def token_drop(self, cond, force_drop_ids=None):
        """
        Drops cond to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0]).cuda() < self.cond_drop_prob
        else:
            drop_ids = force_drop_ids == 1
        cond = torch.where(drop_ids[:, None, None], self.y_embedding, cond)
        return cond

    def forward(self, cond, train, force_drop_ids=None):
        if train:
            assert cond.shape[-1] == self.y_embedding.shape[-1]
        use_dropout = self.cond_drop_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        cond = self.y_proj(cond)
        return cond


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, c_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 2 * hidden_size, bias=True)
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

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
