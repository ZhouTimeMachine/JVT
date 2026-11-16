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
    # RoPE3D,
    rotate_half,
    get_3d_sincos_pos_embed,
    RMSNorm
)

from typing import Optional, Tuple


def get_rope3d_base_freqs(
    dim: int,
    sizes: Tuple[int, int, int],
    theta: float = 10000.0,
):
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

    return freqs


def apply_prope(x: Tensor, freq_cos: Tensor, freq_sin: Tensor, matrix: Tensor, inverse: bool = False) -> Tensor:
    # inverse: False for qkv rope, True for output inverse rope -> QKVO-RoPE
    bsz, num_heads, num_patches, head_dim = x.shape
    rope_dim = freq_cos.shape[-1]
    proj_dim = head_dim - rope_dim

    rope_part = x[..., :rope_dim]
    if inverse:
        rope_out = rope_part * freq_cos - rotate_half(rope_part) * freq_sin
    else:
        rope_out = rope_part * freq_cos + rotate_half(rope_part) * freq_sin

    cameras, camera_dim = matrix.shape[1], matrix.shape[-1]  # (bsz, cameras, camera_dim, camera_dim)
    proj_part = x[..., rope_dim:]
    proj_part = proj_part.reshape(bsz, num_heads, cameras, -1, proj_dim // camera_dim, camera_dim)
    proj_out = torch.einsum("bcij,bncpkj->bncpki", matrix, proj_part)
    proj_out = proj_out.reshape(bsz, num_heads, -1, proj_dim)

    return torch.cat([rope_out, proj_out], dim=-1)

# def modulate(x, shift, scale):
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
def modulate(x, shift, scale):
    # x: (B, T*H*W, C), shift/scale: (B, T, C)
    B, N, C = x.shape
    T = shift.shape[1]
    assert N % T == 0, "Number of tokens in x must be divisible by temporal length T"
    HW = N // T
    
    x = x.view(B, T, HW, C)
    
    # shift and scale are (B, T, C). Unsqueeze to (B, T, 1, C) for broadcasting
    shift = shift.unsqueeze(2)
    scale = scale.unsqueeze(2)
    
    x = x * (1 + scale) + shift
    
    return x.view(B, N, C)


class BottleneckPatchEmbed(nn.Module):
    """ Video to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = x.view(B * T, C, H, W)

        x = self.proj1(x)  # [B*T, pca_dim, H', W']
        x = self.proj2(x)  # [B*T, embed_dim, H', W']

        x = x.view(B, T, -1, x.shape[-2], x.shape[-1])  # [B, T, embed_dim, H', W']
        x = x.permute(0, 1, 3, 4, 2)                    # [B, T, H', W', embed_dim]

        # x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
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
        orig_shape = t.shape
        t = t.reshape(-1)  # flatten
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.reshape(*orig_shape, self.frequency_embedding_size)  # restore shape
        t_emb = self.mlp(t_freq)
        return t_emb


# def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1))
#     attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

#     with torch.cuda.amp.autocast(enabled=False):
#         attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#     return attn_weight @ value


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

    def forward(self, x, qkv_transform, out_transform):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = qkv_transform(q, k, v)

        # x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        dropout_p = self.attn_drop.p if self.training else 0.0
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False
        )

        x = out_transform(x)

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
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
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
    def forward(self, x,  c, qkv_transform, out_transform):
        B, N, C = x.shape
        T = c.shape[1]
        assert N % T == 0, "Number of tokens in x must be divisible by temporal length T"
        HW = N // T
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)  # (B, T, hidden_size)

        gate_msa = gate_msa.repeat_interleave(HW, dim=1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            qkv_transform, out_transform,
        )

        gate_mlp = gate_mlp.repeat_interleave(HW, dim=1)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def build_prope_camera_transforms(viewmats: Tensor, Ks: Optional[Tensor], image_height, image_width: int) -> tuple:
    """Build camera projection transforms."""
    batch, cameras = viewmats.shape[:2]
    
    if Ks is not None:
        # Ks_norm = torch.zeros_like(Ks)
        # Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        # Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        # Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        # Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        # Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks.clone()
        Ks_norm[..., 0, 2] -= 0.5
        Ks_norm[..., 1, 2] -= 0.5
        
        # Lift K to 4x4
        K_4x4 = torch.zeros(batch, cameras, 4, 4, device=Ks.device, dtype=Ks.dtype)
        K_4x4[..., :3, :3] = Ks_norm
        K_4x4[..., 3, 3] = 1.0

        P = torch.einsum("...ij,...jk->...ik", K_4x4, viewmats)
        P_T = P.transpose(-1, -2)

        # Inverse
        Rinv = viewmats[..., :3, :3].transpose(-1, -2)
        tinv = -torch.einsum("bcij,bcj->bci", Rinv, viewmats[..., :3, 3])
        viewmats_inv = torch.zeros_like(viewmats)
        viewmats_inv[..., :3, :3] = Rinv
        viewmats_inv[..., :3, 3] = tinv
        viewmats_inv[..., 3, 3] = 1.0
        
        K_inv = torch.zeros_like(Ks_norm)
        K_inv[..., 0, 0] = 1.0 / Ks_norm[..., 0, 0]
        K_inv[..., 1, 1] = 1.0 / Ks_norm[..., 1, 1]
        K_inv[..., 0, 2] = -Ks_norm[..., 0, 2] / Ks_norm[..., 0, 0]
        K_inv[..., 1, 2] = -Ks_norm[..., 1, 2] / Ks_norm[..., 1, 1]
        K_inv[..., 2, 2] = 1.0
        
        K_inv_4x4 = torch.zeros(batch, cameras, 4, 4, device=Ks.device, dtype=Ks.dtype)
        K_inv_4x4[..., :3, :3] = K_inv
        K_inv_4x4[..., 3, 3] = 1.0

        P_inv = torch.einsum("...ij,...jk->...ik", viewmats_inv, K_inv_4x4)
    else:
        # GTA-style: P = viewmats
        P = viewmats
        P_T = P.transpose(-1, -2)
        Rinv = viewmats[..., :3, :3].transpose(-1, -2)
        tinv = -torch.einsum("bcij,bcj->bci", Rinv, viewmats[..., :3, 3])
        P_inv = torch.zeros_like(viewmats)
        P_inv[..., :3, :3] = Rinv
        P_inv[..., :3, 3] = tinv
        P_inv[..., 3, 3] = 1.0

    return P_T, P_inv, P


class JVTCamCtrl(nn.Module):
    """
    Just Video Transformer.
    """
    def __init__(
        self,
        x_shape: torch.Size,
        max_tokens: int,
        depth: int = 24,
        hidden_size: int = 1024,
        num_heads: int = 16,
        bottleneck_dim: int = 128,
        patch_size: int = 16,
        camera_mode: str = "full",
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        in_channels = 3   # in_channels
        self.out_channels = 3  # in_channels

        self.temporal_length = max_tokens
        self.patch_size = patch_size
        self.camera_mode = camera_mode
        assert self.camera_mode in ["full", "extrinsics_only"], "Camera mode must be either 'full' or 'extrinsics_only'"
        input_size = x_shape[-1]  # 256


        # time and camera pose embed
        self.t_embedder = TimestepEmbedder(hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(
            img_size=input_size,
            patch_size=self.patch_size,
            in_chans=in_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size, bias=True)

        # use fixed sin-cos embedding
        # num_patches = self.x_embedder.num_patches
        num_patches_1dim = input_size // self.patch_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_tokens, num_patches_1dim, num_patches_1dim, hidden_size), 
            requires_grad=False
        )

        # rope
        self.head_dim = hidden_size // num_heads
        self.rope3d_base_freqs = get_rope3d_base_freqs(
            dim = self.head_dim // 2,
            sizes = (max_tokens, num_patches_1dim, num_patches_1dim),
        )

        # transformer
        self.blocks = nn.ModuleList([
            JVTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)

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
            t_size=self.temporal_length,
        )  # (T*num_patches, hidden_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().reshape(self.pos_embed.shape))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

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

    def _build_transforms(
            self, noise_levels: Tensor, 
            P_T: Tensor, P_inv: Tensor, P: Tensor
        ) -> tuple:
        """Build fused RoPE + projection transforms for a level."""

        # Build RoPE transforms
        bsz = noise_levels.shape[0]
        freqs = self.rope3d_base_freqs.to(noise_levels.device).unsqueeze(0).repeat(bsz, 1, 1, 1, 1)  # (B, t, h, w, head_dim // 2)
    
        rope_dim = self.rope3d_base_freqs.shape[-1]  # 1/2 of head_dim
        # freqs[..., :rope_dim//4] *= noise_levels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # for noise_level part
        freqs = freqs.flatten(1, 3).unsqueeze(1)

        freq_cos = freqs.cos()
        freq_sin = freqs.sin()

        # Projection checks
        assert P_T is not None and P_inv is not None and P is not None, "Camera projection matrices must be provided for PRoPE"
        assert P_T.shape == P_inv.shape == P.shape, "Projection matrices must have the same shape"

        proj_dim = self.head_dim - rope_dim
        assert proj_dim % P.shape[-1] == 0
        # assert num_patches % cameras == 0

        def qkv_transform(q, k, v):
            """Apply RoPE then projection to Q, K, V."""
            # assert q.shape[-2] == k.shape[-2] == v.shape[-2], "Q, K, V must have same sequence length"
            seq_len = q.shape[-2]
            cos_seq = freq_cos[..., :seq_len, :]
            sin_seq = freq_sin[..., :seq_len, :]
            q = apply_prope(q, cos_seq, sin_seq, P_T, inverse=False)
            k = apply_prope(k, cos_seq, sin_seq, P_inv, inverse=False)
            v = apply_prope(v, cos_seq, sin_seq, P_inv, inverse=False)
            return q, k, v
        
        def out_transform(x):
            """Apply inverse projection then inverse RoPE to output."""
            seq_len = x.shape[-2]
            cos_seq = freq_cos[..., :seq_len, :]
            sin_seq = freq_sin[..., :seq_len, :]
            return apply_prope(x, cos_seq, sin_seq, P, inverse=True)
        
        return qkv_transform, out_transform

    def unpatchify(self, x: Tensor, num_frames: int):
        """
        x: (N, T*h*w, patch_size**2 * C)
        imgs: (N, T, C, H, W)
        """
        c = self.out_channels
        bsz = x.shape[0]
        psz = self.patch_size
        h = w = int((x.shape[1] // num_frames) ** 0.5)
        assert h * w * num_frames == x.shape[1]

        x = x.reshape(bsz, num_frames, h, w, psz, psz, c)  # (B, T, h, w, p, p, C)
        x = torch.einsum('nthwpqc->ntchpwq', x)  # (B, T, C, h, p, w, p)
        imgs = x.flatten(-4, -3).flatten(-2, -1)     # (B, T, C, H, W)
        return imgs

    def forward(
        self, 
        x: Tensor, 
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
    ):
        """
        x: (B, T, C, H, W)
        noise_levels: (B, T)
        external_cond: (B, T, 16) for full, (B, T, 12) for extrinsics_only
        """
        bsz, num_frames, _, h, w = x.shape
        assert x.shape[1] == noise_levels.shape[1], "Temporal length of noise_levels must match that of x."

        # Parse camera parameters
        Ks = None
        if external_cond is not None:
            if self.camera_mode == "full":
                K_flat, RT_flat = external_cond.split([4, 12], dim=-1)
                fx, fy, cx, cy = K_flat.unbind(dim=-1)
                Ks = torch.zeros(bsz, num_frames, 3, 3, device=x.device)
                Ks[..., 0, 0], Ks[..., 1, 1] = fx, fy
                Ks[..., 0, 2], Ks[..., 1, 2] = cx, cy
                Ks[..., 2, 2] = 1.0
            else:  # extrinsics_only
                RT_flat = external_cond
            viewmats = torch.zeros(bsz, num_frames, 4, 4, device=x.device)
            viewmats[..., :3, :] = RT_flat.reshape(bsz, num_frames, 3, 4)
            viewmats[..., 3, 3] = 1.0
        else:
            print("Warning: No external conditioning provided, using identity view matrices and no intrinsics.")
            viewmats = torch.eye(4, device=x.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_frames, 4, 4)

        # Build camera transforms
        P_T, P_inv, P = build_prope_camera_transforms(viewmats, Ks, image_height=h, image_width=w)
        qkv_transform, out_transform = self._build_transforms(noise_levels, P_T, P_inv, P)

        # time embeddings
        t_emb = self.t_embedder(noise_levels)  # (B, T) -> (B, T, hidden_size)
        c = t_emb

        # forward JiT
        x = self.x_embedder(x)  # (B, T, h, w, hidden_size)
        x = x + self.pos_embed  # self.pos_embed: (1, T, h, w, hidden_size)
        x = x.flatten(1, 3)     # (B, num_patches, hidden_size)

        for i, block in enumerate(self.blocks):
            x = block(x, c, qkv_transform, out_transform)

        x = self.final_layer(x, c)  # -> (B, num_patches, patch_size**2 * out_channels)
        output = self.unpatchify(x, num_frames)  # -> (B, T, C, H, W)

        return output

