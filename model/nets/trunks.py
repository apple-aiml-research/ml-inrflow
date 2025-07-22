#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from torch import nn


# Homogen trunk, same block applied iteratively
class HomogenTrunk(nn.Module):
    def __init__(
        self, 
        block, 
        depth, 
        num_dec_levels=1,
        pos_embedder=None
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([block() for _ in range(depth)])
        self.pos_embedder = pos_embedder

        interval = depth // num_dec_levels
        self.decode_levels = [
            depth - i * interval - 1 for i in range(num_dec_levels)
        ]

    def forward(self, latents, c, **kwargs):
        latents_out = []
        if self.pos_embedder:
            latents = self.pos_embedder.apply_pos_embed(latents)
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)

            # (N, T, D)
            if i in self.decode_levels:
                latents_out.append(latents)
        return latents_out


class TrunkWithEncoder(nn.Module):
    def __init__(
        self, 
        block, 
        depth, 
        encoder, 
        num_enc_levels=1, 
        num_dec_levels=1
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([block() for _ in range(depth)])
        self.hidden_encoder = nn.ModuleList(encoder() for _ in range(num_enc_levels))

        interval = depth // num_enc_levels
        self.encode_levels = [
            i * interval for i in range(num_enc_levels)
        ]

        interval = depth // num_dec_levels
        self.decode_levels = [
            depth - i * interval - 1 for i in range(num_dec_levels)
        ]

    def forward(self, latents, c, inputs, y, **kwargs):
        latents_out = []
        for i, block in enumerate(self.blocks):
            if i in self.encode_levels:
                latents = self.hidden_encoder[self.encode_levels.index(i)](
                    inputs=inputs,
                    latents=latents,
                    c=c,
                )[0]

            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, y=y, **kwargs)

            # (N, T, D)
            if i in self.decode_levels:
                latents_out.append(latents)
        return latents_out
