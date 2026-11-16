import random
import torch

from torch import Tensor
from einops import rearrange, repeat
from typing import Optional, Dict, Tuple
from tqdm import tqdm
from omegaconf import DictConfig

from .diffusion import RectifiedFlowC2V


class C2VDenoiser(torch.nn.Module):
    """
    An algorithm for training and evaluating
    Diffusion Transformer (DiT) for class-conditioned video generation.
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
        self.num_classes = cfg.diffusion.num_classes
        self.label_drop_prob = cfg.label_drop_prob
        self.horizontal_flip_prob = cfg.horizontal_flip_prob
        self.data_mean = torch.tensor(cfg.data_mean, device=self.device)
        self.data_std = torch.tensor(cfg.data_std, device=self.device)

        # 2. Guidance
        self.rg_scale = cfg.get("rg_scale", None)
        self.cfg_scale = cfg.cfg_scale
        self.cfg_interval_begin = cfg.cfg_interval_begin
        self.cfg_interval_end = cfg.cfg_interval_end

        # 3. Diffusion
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.noise_shift = cfg.noise_sampling.shift
        self.clip_noise = cfg.diffusion.clip_noise
        self.diffusion_model = RectifiedFlowC2V(
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
    
    def preprocess_video_cond(self, xs: Tensor, conditions: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Preprocess videos and conditions for sampling.
        
        Args:
            xs (Tensor, "B T C H W"): Input videos in [0, 1].
            conditions (Tensor, "B"): Class labels for the videos.
        Returns:
            Processed xs and conditions.
        """
        xs = xs.to(self.device)
        xs = self._normalize_x(xs)
        if self.training and random.random() < self.horizontal_flip_prob:  # horizontal flip
            xs = xs.flip(-1)

        assert conditions is not None
        conditions = conditions.to(self.device)
        if self.training:  # drop label for unconditional training
            drop = torch.rand(conditions.shape[0], device=conditions.device) < self.label_drop_prob
            conditions = torch.where(drop, torch.full_like(conditions, self.num_classes), conditions)
        
        return xs, conditions

    def forward(self, batch):
        xs, conditions = self.preprocess_video_cond(batch["videos"], batch["conds"])
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
            conditions,
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

        xs, conditions = self.preprocess_video_cond(xs, conditions)

        all_videos: Dict[str, Tensor] = {"gt": self._unnormalize_x(xs)}

        self.diffusion_model.eval()

        if xs.shape[1:] != (self.max_frames, *self.x_shape):
            raise ValueError(f"Expected xs shape {(conditions.shape[0], self.max_frames, *self.x_shape)}, got {xs.shape}")
        if conditions is not None:
            if conditions.shape[0] != xs.shape[0]:
                raise ValueError(f"Batch size of conditions should match that of xs, expected {xs.shape[0]}, got {conditions.shape[0]}")

        all_videos["prediction"] = self.sample_sequence(
            batch_size=xs.shape[0],
            conditions=conditions,
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
        conditions: Tensor,
        progress: bool = True,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        conditions: Optional[torch.Tensor], Shape (batch_size, self.max_frames, ...)
            Unprocessed external conditions for sampling
        
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
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
                if self.cfg_scale > 1.0 + 1e-5:
                    begin = self.diffusion_model.shift_time(self.cfg_interval_begin)
                    end = self.diffusion_model.shift_time(self.cfg_interval_end)
                    if from_noise_level.device != self.diffusion_model.timestep2sigma.device:
                        self.diffusion_model.timestep2sigma = self.diffusion_model.timestep2sigma.to(from_noise_level.device)
                    t = self.diffusion_model.timestep2sigma[from_noise_level.long() + 1]
                    interval_mask = (t > end) & ((abs(begin - 1.0) < 1e-5) | (t < begin))
                    cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
                    cfg_scale_interval = rearrange(
                        cfg_scale_interval,
                        f"... -> ...{' 1' * (len(self.x_shape) + 1)}"
                    )
                else:
                    cfg_scale_interval = 1.0

                pred_cond = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_level,
                    to_noise_level,
                    conditions,
                    uncond_external_cond=(
                        torch.full_like(conditions, self.num_classes)
                        if self.cfg_scale > 1.0 + 1e-5
                        else None
                    ),
                    cfg_scale=cfg_scale_interval,
                )

                xs_pred = pred_cond

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
