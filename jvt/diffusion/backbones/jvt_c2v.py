# --------------------------------------------------------
# References:
# JiT: https://github.com/LTH14/JiT
# --------------------------------------------------------
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from .modules.embeddings import RandomDropoutPatchEmbed
from .modules import (
    rotate_half,
    get_3d_sincos_pos_embed,
    RMSNorm
)

from typing import Optional, Tuple


class RoPE3D(nn.Module):
    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int, int],
        theta: float = 10000.0,
        num_cls_token = 0
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE3D requires even dim"
        t, h, w = sizes

        # Split dim into dt, dh, dw
        # dim%3 == 0 -> (d,   d,   d  )
        #    %3 == 1 -> (d+1, d,   d  )
        #    %3 == 2 -> (d,   d+1, d+1)
        base_dim = (dim // 2) // 3
        rem = (dim // 2) % 3
        dim_t = base_dim + (1 if rem == 1 else 0)
        dim_h = base_dim + (1 if rem == 2 else 0)
        dim_w = base_dim + (1 if rem == 2 else 0)

        # dims = (dim_t * 2, dim_h * 2, dim_w * 2), x2 because each rotary pair has 2 dimensions
        def get_axis_freq(dim: int, seq_len: int) -> torch.Tensor:
            base_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # 1 / (theta ** (2i / d))
            pos = torch.arange(seq_len, dtype=torch.float)  # position sequence [0, 1, ..., size-1]

            axis_freq = torch.outer(pos, base_freq)  # Kronecker product, (size, d/2)
            return torch.repeat_interleave(axis_freq, repeats=2, dim=-1)  # (size, d)

        ft = get_axis_freq(dim_t * 2, t).view(t, 1, 1, -1).expand(t, h, w, -1)
        fh = get_axis_freq(dim_h * 2, h).view(1, h, 1, -1).expand(t, h, w, -1)
        fw = get_axis_freq(dim_w * 2, w).view(1, 1, w, -1).expand(t, h, w, -1)

        freqs = torch.cat((ft, fh, fw), dim=-1)  # (t, h, w, dim)
        freqs = freqs.flatten(0, 2)              # (t*h*w, dim)

        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        if num_cls_token > 0:
            cos_pad = torch.ones((num_cls_token, dim), device=freqs.device, dtype=freqs.dtype)
            sin_pad = torch.zeros((num_cls_token, dim), device=freqs.device, dtype=freqs.dtype)
            freqs_cos = torch.cat((cos_pad, freqs_cos), dim=0)
            freqs_sin = torch.cat((sin_pad, freqs_sin), dim=0)

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, t):
        # t: (B, num_heads, seq_len, head_dim)
        seq_len = t.shape[-2]
        cos_seq = self.freqs_cos[..., :seq_len, :]
        sin_seq = self.freqs_sin[..., :seq_len, :]
        return t * cos_seq + rotate_half(t) * sin_seq

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Video to Patch Embedding
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int, in_chans: int,
        pca_dim: int, embed_dim: int,
        temporal_length: int,
        temporal_downsample: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = num_patches

        self.temporal_length = temporal_length
        self.temporal_downsample = temporal_downsample
        if temporal_downsample < 1:
            raise ValueError(
                f"temporal_downsample must be >= 1, got {temporal_downsample}."
            )
        elif temporal_length % temporal_downsample != 0:
            raise ValueError(
                f"temporal_length must be divisible by temporal_downsample. Got temporal_length={temporal_length} and temporal_downsample={temporal_downsample}."
            )

        # for tubelet embedding with bottleneck, you can use 
        # self.temporal_proj1 = nn.Conv3d(
        #     in_chans,
        #     pca_dim * temporal_downsample,
        #     kernel_size=(temporal_downsample, patch_size, patch_size),
        #     stride=(temporal_downsample, patch_size, patch_size),
        #     bias=False,
        # )
        # self.temporal_proj2 = nn.Conv3d(
        #     pca_dim * temporal_downsample,
        #     embed_dim,
        #     kernel_size=1,
        #     stride=1,
        #     bias=bias,
        # )

        # we use patch-first embedding here
        self.proj1 = nn.Conv2d(
            in_chans,
            pca_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.proj2 = nn.Conv1d(
            pca_dim,
            embed_dim,
            kernel_size=temporal_downsample,
            stride=temporal_downsample,
            bias=bias
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        B, T, C, H, W = x.shape
        assert T == self.temporal_length

        # 1. spatial projection (B*T, C, H, W) -> (B*T, pca_dim, h, w)
        x = x.view(B * T, C, H, W)
        x = self.proj1(x)
        _, C_pca, h, w = x.shape
        
        # 2. temporal projection (B, h, w, C_pca, T) -> (B*h*w, C_pca, T_down)
        x = x.view(B, T, C_pca, h, w)
        x = x.permute(0, 3, 4, 2, 1).reshape(-1, C_pca, T)
        x = self.proj2(x)
        T_down = x.shape[-1]
        
        # 3. (B, h, w, T_down, embed_dim) -> (B, T_down, h, w, embed_dim)
        x = x.view(B, h, w, -1, T_down)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        dropout_p = self.attn_drop.p if self.training else 0.0
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False
        )

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, frames_per_token: int = 1):
        super().__init__()
        if frames_per_token < 1:
            raise ValueError(
                f"frames_per_token must be >= 1, got {frames_per_token}."
            )
        self.frames_per_token = frames_per_token
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels * frames_per_token,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JVTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)  # (B, hidden_size)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JVTC2V(nn.Module):
    """
    Just Video Transformer (class-conditioned video generation).
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
        num_classes: int = 101,
        temporal_downsample: int = 4,
        in_context_len: int = 0,
        in_context_start: int = 28,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        in_channels = 3   # in_channels
        self.out_channels = 3  # in_channels
        self.temporal_tokens = max_tokens
        self.temporal_downsample = temporal_downsample
        self.temporal_length = max_tokens * self.temporal_downsample
        self.patch_size = patch_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start

        input_size = x_shape[-1]  # 256

        # time and label embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # data embed
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

        # use fixed sin-cos embedding
        # num_patches = self.x_embedder.num_patches
        num_patches_1dim = input_size // self.patch_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.temporal_tokens, num_patches_1dim, num_patches_1dim, hidden_size),
            requires_grad=False
        )

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size), 
                requires_grad=True
            )
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        self.head_dim = hidden_size // num_heads
        self.rope3d = RoPE3D(
            dim = self.head_dim,
            sizes = (self.temporal_tokens, num_patches_1dim, num_patches_1dim),
            num_cls_token=0,
        )
        self.rope3d_inctxt = RoPE3D(
            dim = self.head_dim,
            sizes = (self.temporal_tokens, num_patches_1dim, num_patches_1dim),
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
        # Initialize transformer layers:
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
            # Initialize patch embed like nn.Linear (instead of nn.Conv):
            weight = conv.weight.data
            nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
            if getattr(conv, "bias", None) is not None:
                nn.init.constant_(conv.bias, 0)

        _init_conv_weight(self.x_embedder.proj1)
        _init_conv_weight(self.x_embedder.proj2)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
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
        noise_levels: (B, )
        external_cond: (B, )
        """
        num_frames = x.shape[1]
        assert num_frames == self.temporal_length, (
            f"Temporal length of JVTC2V is set to {self.temporal_length}, "
            f"but input has temporal length {num_frames}."
        )

        # forward JiT
        x = self.x_embedder(x)  # (B, G, h, w, hidden_size)

        t_emb = self.t_embedder(noise_levels)   # (B, hidden_size)
        y_emb = self.y_embedder(external_cond)  # (B, hidden_size)
        c = t_emb + y_emb

        x = x + self.pos_embed  # self.pos_embed: (1, G, h, w, hidden_size)
        x = x.flatten(1, 3)     # (B, num_patches, hidden_size)

        for i, block in enumerate(self.blocks):
            # in-context
            if use_register and self.in_context_len > 0 and i == self.in_context_start:
                # just use base register tokens, without any conditioning from y_emb or t_emb
                in_context_tokens = self.in_context_posemb.expand(y_emb.shape[0], -1, -1)
                x = torch.cat([in_context_tokens, x], dim=1)
            rope = (
                self.rope3d_inctxt
                if use_register and i >= self.in_context_start
                else self.rope3d
            )
            x = block(x, c, rope)
        
        if use_register:
            x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)  # -> (B, num_patches, temporal_downsample * patch_size**2 * out_channels)
        output = self.unpatchify(x)  # -> (B, T, C, H, W)

        return output

