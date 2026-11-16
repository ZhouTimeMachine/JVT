# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

from math import pi

import torch
from torch import nn
import numpy as np

from typing import Tuple
from einops import rearrange, repeat


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class RoPE3D(nn.Module):
    def __init__(
        self,
        dim: int,
        hw_seq_len: int,
        t_seq_len: int,
        theta: int = 10000,
        num_cls_token: int = 0
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE3D requires even dim"

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

        t, h, w = t_seq_len, hw_seq_len, hw_seq_len
        ft = get_axis_freq(dim_t * 2, t).view(t, 1, 1, -1).expand(t, h, w, -1)
        fh = get_axis_freq(dim_h * 2, h).view(1, h, 1, -1).expand(t, h, w, -1)
        fw = get_axis_freq(dim_w * 2, w).view(1, 1, w, -1).expand(t, h, w, -1)
        # (t, h, w, dim), dim = 2*dim_t + 2*dim_h + 2*dim_w
        freqs = torch.cat((ft, fh, fw), dim=-1)
        freqs = freqs.flatten(0, 2)  # (t*h*w, dim)

        if num_cls_token > 0:
            raise NotImplementedError('RoPE3D with cls token is not implemented yet')
            # freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            # cos_img = freqs_flat.cos()
            # sin_img = freqs_flat.sin()

            # # prepend in-context cls token
            # N_img, D = cos_img.shape
            # cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            # sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            # self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()  # [N_cls+N_img, D]
            # self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
        else:
            self.freqs_cos = freqs.cos().cuda()
            self.freqs_sin = freqs.sin().cuda()

    def forward(self, t): return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: int, t_size: int, cls_token: bool = False, extra_tokens: int = 0):
    """
    grid_size: int, the grid height and width, H = W = grid_size
    t_size: int, temporal length, T
    return: [T * grid_size * grid_size, embed_dim]
    """

    grid_t = np.arange(t_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)

    assert embed_dim % 2 == 0
    base_dim = (embed_dim // 2) // 3
    rem = (embed_dim // 2) % 3
    dim_t = base_dim + (1 if rem == 1 else 0)
    dim_h = base_dim + (1 if rem == 2 else 0)
    dim_w = base_dim + (1 if rem == 2 else 0)

    emb_t = get_1d_sincos_pos_embed_from_grid(dim_t * 2, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h * 2, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w * 2, grid[2])  # (T*H*W, D/3)

    pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1) # (T*H*W, D)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0):
    """
    grid_size: int, the grid height and width, H = W = grid_size
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, theta=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / theta ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb