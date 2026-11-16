# --------------------------------------------------------
# References:
# JiT: https://github.com/LTH14/JiT
# --------------------------------------------------------
import torch
from torch import nn, Tensor

from typing import Optional


from .modules import (
    get_3d_sincos_pos_embed,
)
from .jvt_c2v import (
    RoPE3D,
    BottleneckPatchEmbed,
    TimestepEmbedder,
    JVTBlock,
    FinalLayer,
)


class JVTUnCond(nn.Module):
    """
    Just Video Transformer (unconditional) with uniform temporal grouping.
    """

    def __init__(
        self,
        x_shape: torch.Size,
        max_tokens: int,
        depth: int = 28,
        hidden_size: int = 1152,
        num_heads: int = 16,
        bottleneck_dim: int = 32,
        patch_size: int = 16,
        temporal_downsample: int = 4,
        in_context_len: int = 0,
        in_context_start: int = 28,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        in_channels = 3
        self.out_channels = 3
        self.temporal_tokens = max_tokens
        self.temporal_downsample = temporal_downsample
        if self.temporal_downsample < 1:
            raise ValueError(
                f"temporal_downsample must be >= 1, got {self.temporal_downsample}."
            )
        self.temporal_length = max_tokens * self.temporal_downsample
        self.patch_size = patch_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start

        input_size = x_shape[-1]

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.x_embedder = BottleneckPatchEmbed(
            img_size=input_size,
            patch_size=self.patch_size,
            in_chans=in_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
            temporal_length=self.temporal_length,
            temporal_downsample=self.temporal_downsample,
            bias=True,
        )

        num_patches_1dim = input_size // self.patch_size
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.temporal_tokens,
                num_patches_1dim,
                num_patches_1dim,
                hidden_size,
            ),
            requires_grad=False,
        )

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size),
                requires_grad=True,
            )
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size),
                requires_grad=True,
            )
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        self.head_dim = hidden_size // num_heads
        self.rope3d = RoPE3D(
            dim=self.head_dim,
            sizes=(self.temporal_tokens, num_patches_1dim, num_patches_1dim),
            num_cls_token=0,
        )
        self.rope3d_inctxt = RoPE3D(
            dim=self.head_dim,
            sizes=(self.temporal_tokens, num_patches_1dim, num_patches_1dim),
            num_cls_token=self.in_context_len,
        )

        # transformer
        self.blocks = nn.ModuleList([
            JVTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(
            hidden_size,
            self.patch_size,
            self.out_channels,
            frames_per_token=self.temporal_downsample,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=self.pos_embed.shape[-2],
            t_size=self.temporal_tokens,
        )  # (T*num_patches, hidden_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().reshape(self.pos_embed.shape))

        def _init_conv_weight(conv: nn.Module):
            weight = conv.weight.data
            nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
            if getattr(conv, "bias", None) is not None:
                nn.init.constant_(conv.bias, 0)

        _init_conv_weight(self.x_embedder.proj1)
        _init_conv_weight(self.x_embedder.proj2)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @staticmethod
    def pixel_shuffel(x: Tensor) -> Tensor:
        # x: (B, G, h, w, s, p, p, c)
        x = torch.einsum('bghwspqc->bgschpwq', x)  # (B, G, s, c, h, p, w, p)
        x = x.flatten(-4, -3).flatten(-2, -1).flatten(1, 2)  # (B, T=G*s, c, H=h*p, W=w*p)
        return x

    def unpatchify(self, x: Tensor):
        """
        x: (N, G*h*w, temporal_downsample * patch_size**2 * C)
        imgs: (N, T, C, H, W)
        """
        c = self.out_channels
        bsz = x.shape[0]
        psz = self.patch_size
        h = w = int((x.shape[1] // self.temporal_tokens) ** 0.5)
        assert h * w * self.temporal_tokens == x.shape[1]

        x = x.reshape(
            bsz,
            self.temporal_tokens, h, w,  # G*h*w
            self.temporal_downsample,
            psz, psz, c,  # s*h*w*c
        )  # （B, G, h, w, s, p, p, c)

        return self.pixel_shuffel(x)

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
        use_register: bool = True,
    ):
        """
        x: (B, T, C, H, W)
        noise_levels: (B,)
        external_cond: unused
        """
        del external_cond

        num_frames = x.shape[1]
        assert num_frames == self.temporal_length, (
            f"Temporal length of JVTUnCond is set to {self.temporal_length}, "
            f"but input has temporal length {num_frames}."
        )

        x = self.x_embedder(x)  # (B, G, h, w, hidden_size)

        c = self.t_embedder(noise_levels)   # (B, hidden_size)

        x = x + self.pos_embed
        x = x.flatten(1, 3)

        for i, block in enumerate(self.blocks):
            if use_register and self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = self.in_context_posemb.expand(c.shape[0], -1, -1)
                x = torch.cat([in_context_tokens, x], dim=1)
            rope = (
                self.rope3d_inctxt
                if use_register and i >= self.in_context_start
                else self.rope3d
            )
            x = block(x, c, rope)

        if use_register:
            x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x)

        return output
