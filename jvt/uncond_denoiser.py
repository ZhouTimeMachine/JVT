import random
import torch

from torch import Tensor
from einops import repeat
from typing import Optional, Dict
from tqdm import tqdm
from omegaconf import DictConfig

from .diffusion import RectifiedFlowUncond


class UnCondDenoiser(torch.nn.Module):
    """
    An algorithm for training and evaluating
    Diffusion Transformer (DiT) for unconditional video generation.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # 0. Infra
        self.generator = None
        self.device = torch.device(cfg.device)
        self.global_rank = cfg.global_rank
        self.world_size = cfg.world_size

        # 1. Data
        self.x_shape = list(cfg.x_shape)
        self.max_frames = cfg.max_frames
        self.temporal_downsampling_factor = cfg.temporal_downsampling_factor
        if self.max_frames % self.temporal_downsampling_factor:
            raise ValueError(f"max_frames should be divisible by temporal_downsampling_factor, but got {self.max_frames} and {self.temporal_downsampling_factor}")
        self.max_tokens = self.max_frames // self.temporal_downsampling_factor
        self.horizontal_flip_prob = cfg.horizontal_flip_prob
        self.data_mean = torch.tensor(cfg.data_mean, device=self.device)
        self.data_std = torch.tensor(cfg.data_std, device=self.device)

        # 2. Guidance
        self.rg_scale = cfg.get("rg_scale", None)

        # 3. Diffusion
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.noise_shift = cfg.noise_sampling.shift
        self.clip_noise = cfg.diffusion.clip_noise
        self.diffusion_model = RectifiedFlowUncond(
            cfg=cfg.diffusion,
            backbone_cfg=cfg.backbone,
            x_shape=cfg.x_shape,
            max_tokens=self.max_tokens,
            rg_scale=self.rg_scale,
        ).to(self.device)

        # self.cfg = cfg  # for debug

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------
    def preprocess_video(self, xs: Tensor, augment: bool = False) -> Tensor:
        """
        Preprocess videos for training or sampling.

        Args:
            xs (Tensor, "B T C H W"): Input videos in [0, 1].
            augment (bool): Whether to apply training-time augmentation.
        Returns:
            Processed xs.
        """
        xs = xs.to(self.device)
        xs = self._normalize_x(xs)
        if augment and random.random() < self.horizontal_flip_prob:  # horizontal flip
            xs = xs.flip(-1)

        return xs

    def forward(self, batch):
        """Training step"""
        xs = self.preprocess_video(batch["videos"], augment=True)
        batch_size, n_tokens, *_ = xs.shape

        t = torch.sigmoid(
            torch.randn(
                batch_size, 
                device=xs.device,
                generator=self.generator
            )
        )
        noise_levels = self.noise_shift * t / (1 + (self.noise_shift - 1) * t)

        xs_pred, loss = self.diffusion_model(
            xs,
            external_cond=None,
            k=noise_levels,
        )
        loss = loss.mean()

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample_videos(
        self,
        xs: Tensor,
        conditions: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[Tensor] = None,
    ) -> Optional[Dict[str, Tensor]]:

        if conditions is not None:
            raise ValueError("Unconditional video sampling does not support conditions.")

        xs = self.preprocess_video(xs, augment=False)

        all_videos: Dict[str, Tensor] = {"gt": self._unnormalize_x(xs)}

        self.diffusion_model.eval()

        if xs.shape[1:] != (self.max_frames, *self.x_shape):
            raise ValueError(f"Expected xs shape {(xs.shape[0], self.max_frames, *self.x_shape)}, got {xs.shape}")

        all_videos["prediction"] = self.sample_sequence(
            batch_size=xs.shape[0],
            generator=generator,
            initial_noise=initial_noise,
        )

        return all_videos

    # ---------------------------------------------------------------------
    # Sampling Utils
    # ---------------------------------------------------------------------

    @property
    def scheduling_list(self) -> Tensor:
        scheduling_list = torch.arange(self.sampling_timesteps, -1, -1).long()

        scheduling_list = self.diffusion_model.ddim_idx_to_noise_level(
            scheduling_list
        )

        return scheduling_list

    def sample_sequence(
        self,
        batch_size: int,
        progress: bool = True,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        The unified sampling method for unconditional video generation.

        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, max_frames, *self.x_shape)
        """

        # create initial xs_pred with noise
        if initial_noise is not None:
            xs_pred = initial_noise.to(device=self.device)
        else:
            xs_pred = torch.randn(
                (batch_size, self.max_frames, *(self.x_shape)),
                device=self.device,
                generator=generator,
            )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        # generate scheduling matrix
        scheduling_list = self.scheduling_list.to(self.device)
        scheduling_list = repeat(scheduling_list, "m -> m b", b=batch_size)

        if progress:
            iterable = tqdm(
                range(scheduling_list.shape[0] - 1),
                desc=f"[Rank {self.global_rank}] Sampling",
                leave=False,
            )
        else:
            iterable = range(scheduling_list.shape[0] - 1)

        for m in iterable:
            from_noise_level = scheduling_list[m]
            to_noise_level = scheduling_list[m + 1]

            # update xs_pred by DDIM or DDPM sampling
            with torch.no_grad():
                xs_pred = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_level,
                    to_noise_level,
                    external_cond=None,
                )

        xs_pred = self._unnormalize_x(xs_pred)
        return xs_pred

    # ---------------------------------------------------------------------
    # Normalization Utils
    # ---------------------------------------------------------------------

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean
