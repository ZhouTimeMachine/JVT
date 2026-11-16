# References:
#   https://github.com/Lightning-AI/torchmetrics/blob/main/src/torchmetrics/image/fid.py
#   https://github.com/kwsong0113/diffusion-forcing-transformer/blob/main/algorithms/common/metrics/video/video_metric.py
#   https://github.com/JunyaoHu/common_metrics_on_video_quality/blob/main/fvd/styleganv/fvd.py

from pathlib import Path
from urllib.request import urlretrieve
from typing import Optional

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor


I3D_DEFAULT_URL = "https://github.com/JunyaoHu/common_metrics_on_video_quality/raw/main/fvd/styleganv/i3d_torchscript.pt"
I3D_DEFAULT_PATH = Path(__file__).with_name("checkpoints") / "i3d_torchscript.pt"


@torch.no_grad()
def compute_statistics(features: Tensor) -> tuple[Tensor, Tensor]:
    """Return float64 mean and unbiased (N-1) covariance for (N, D) features."""
    # Following torchmetrics, we use unabised covariance (N-1)
    # https://github.com/Lightning-AI/torchmetrics/blob/6c551c6469dc04fe92291c000ce2607860df92d2/src/torchmetrics/image/fid.py#L409
    if features.ndim != 2 or not features.shape[1]:
        raise ValueError(f"FVD requires 2D features with a nonzero feature dimension, but receives feature.shape {features.shape}")
    if len(features) < 2:
        raise ValueError(f"FVD requires at least two samples in each distribution, but receives only {len(features)} sample")
    features = features.double()
    mu = features.mean(dim=0)
    centered = features - mu
    sigma = centered.T @ centered / (len(features) - 1)
    return mu, sigma


@torch.no_grad()
def compute_fvd(mu_fake: Tensor, sigma_fake: Tensor, mu_real: Tensor, sigma_real: Tensor) -> Tensor:
    """Fréchet distance from means (D,) and covariances (D, D)."""
    if mu_fake.ndim != 1 or mu_fake.shape != mu_real.shape or not mu_fake.numel():
        raise ValueError(f"FVD means must be nonempty vectors of the same shape, got {mu_fake.shape} and {mu_real.shape}")
    shape = (mu_fake.numel(), mu_fake.numel())
    if sigma_fake.shape != shape or sigma_real.shape != shape:
        raise ValueError(f"FVD covariances must match the mean feature dimension, got {sigma_fake} and {sigma_real.shape} with mu feature shape {mu_fake.shape}")
    mu_fake, sigma_fake, mu_real, sigma_real = (
        x.double() for x in (mu_fake, sigma_fake, mu_real, sigma_real)
    )
    # Calculate Fréchet distance based on torch 
    # https://github.com/Lightning-AI/torchmetrics/blob/6c551c6469dc04fe92291c000ce2607860df92d2/src/torchmetrics/image/fid.py#L174
    # Another implementation based on numpy and scipy:
    # https://github.com/ZhouTimeMachine/pytorch-fid-metrics/blob/b8846d2a37b37f048d141da6491246c034c2fd55/metrics.py#L114
    mean_distance = (mu_real - mu_fake).square().sum()
    cross_trace = torch.linalg.eigvals(sigma_real @ sigma_fake).sqrt().real.sum()
    fvd_value = mean_distance + sigma_real.trace() + sigma_fake.trace() - 2 * cross_trace
    if not math.isfinite(fvd_value):
        raise ValueError(f"The calculated FVD is not finite, got {fvd_value}")
    return fvd_value


class I3D(nn.Module):
    """Extract (B, 400) features from videos shaped (B, T, C, H, W) in [0, 1]."""

    def __init__(self, I3D_path: Optional[str] = None, I3D_url: Optional[str] = None):
        super().__init__()
        model_path = Path(I3D_path) if I3D_path is not None else I3D_DEFAULT_PATH
        model_url = I3D_url if I3D_url is not None else I3D_DEFAULT_URL
        if not model_path.is_file():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading I3D model to {model_path}...")
            urlretrieve(model_url, model_path)
        self.model = torch.jit.load(model_path, map_location="cpu").eval()

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # b t c h w -> b c t h w
        video = video.permute(0, 2, 1, 3, 4)
        # Symmetric padding to at least 9 frames, following DFoT
        if video.shape[2] < 9:
            pad = (10 - video.shape[2]) // 2
            video = F.pad(
                video, 
                pad=(
                    0, 0,     # w: no padding
                    0, 0,     # h: no padding
                    pad, pad  # t: symmetric padding to left and right
                ), 
                mode="replicate"
            )
        # Normalize from [0, 1] to [-1, 1] and clip
        video = (2.0 * video - 1.0).clamp(-1.0, 1.0).contiguous()
        # Extract features with resize=True (resizes to 224x224 internally, returns (B, 400))
        return self.model(video, rescale=False, resize=True, return_features=True)
