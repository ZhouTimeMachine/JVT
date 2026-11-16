"""Rectified-flow training and sliding-window sampling for RGB/RGBD video."""

import torch
import torch.nn.functional as F
from einops import repeat
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm
from typing import Dict, Tuple

from .camera_utils import CameraPose
from .diffusion import RectifiedFlow


class CamCtrlDenoiser(torch.nn.Module):
    """Train and sample camera-controlled RGB/RGBD videos with rectified flow."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # 0. Infra
        self.generator = None
        self.device = torch.device(cfg.device)
        self.global_rank = cfg.get("global_rank", 0)
        self.world_size = cfg.get("world_size", 1)

        # 1. Data
        self.x_shape = list(cfg.x_shape)
        self.max_frames = cfg.max_frames
        self.n_context_frames = cfg.context_frames
        self.data_mean = torch.tensor(cfg.data_mean, device=self.device)
        self.data_std = torch.tensor(cfg.data_std, device=self.device)
        if not 0 <= self.n_context_frames < self.max_frames:
            raise ValueError("context_frames must be in [0, max_frames)")

        self.camera_pose_conditioning = cfg.camera_pose_conditioning
        self.camera_mode = self.camera_pose_conditioning.type
        if self.camera_mode != cfg.backbone.camera_mode:
            raise ValueError("camera_pose_conditioning.type must match backbone.camera_mode")
        if self.camera_mode not in {"full", "extrinsics_only"}:
            raise ValueError(f"Unknown camera pose conditioning type: {self.camera_mode}")
        disparity_cfg = cfg.get("online_disparity")
        self.use_online_disparity = bool(disparity_cfg and disparity_cfg.enable)
        self.online_disparity_helper = None
        if self.use_online_disparity:
            if self.x_shape[0] != 4:
                raise ValueError("Online Pi3X disparity requires RGBD x_shape with 4 channels")
            from .online_disparity import Pi3XOnlineDisparityHelper

            self.online_disparity_helper = Pi3XOnlineDisparityHelper(
                device=self.device,
                ckpt_path=disparity_cfg.ckpt,
                crop_size=int(disparity_cfg.get("crop_size", 252)),
            )

        # 2. Guidance
        self.history_guidance_scale = cfg.history_guidance_scale
        self.sliding_context_len = cfg.sliding_context_len

        # 3. Diffusion
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.noise_shift = cfg.noise_sampling.shift
        self.clip_noise = cfg.diffusion.clip_noise
        self.diffusion_model = RectifiedFlow(
            cfg=cfg.diffusion,
            backbone_cfg=cfg.backbone,
            x_shape=self.x_shape,
            max_tokens=self.max_frames,
        ).to(self.device)

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------

    def preprocess_video_cond(self, xs: Tensor, conditions: Tensor) -> Tuple[Tensor, Tensor]:
        """Normalize videos and prepare raw camera vectors for training or sampling."""
        xs = xs.to(self.device)
        conditions = conditions.to(self.device)
        if self.use_online_disparity and xs.shape[2] == 3:
            disparity = self.online_disparity_helper.infer(xs, conditions)
            xs = torch.cat([xs, disparity], dim=2)
        if xs.shape[2] != self.x_shape[0]:
            raise ValueError(f"Expected {self.x_shape[0]} video channels, got {xs.shape[2]}")
        return self._normalize_x(xs), conditions

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def _process_conditions(self, conditions):
        """Normalize raw (B, T, 16) camera vectors in FP32."""
        poses = CameraPose.from_vectors(conditions)
        normalize_by = self.camera_pose_conditioning.normalize_by
        if normalize_by == "first":
            poses.normalize_by_first()
        elif normalize_by == "mean":
            poses.normalize_by_mean()
        else:
            raise ValueError(f"Unknown camera pose normalization: {normalize_by}")
        if self.camera_pose_conditioning.bound is not None:
            poses.scale_within_bounds(self.camera_pose_conditioning.bound)
        extrinsics = poses.extrinsics(flatten=True)
        if self.camera_mode == "extrinsics_only":
            return extrinsics
        return torch.cat([poses.intrinsics(flatten=True), extrinsics], dim=-1)

    def forward(self, batch):
        """Training step with independent lognormal noise levels for each frame."""
        if "masks" in batch:
            raise NotImplementedError("Token-level training masks are not supported")
        xs, conditions = self.preprocess_video_cond(batch["videos"], batch["conds"])
        batch_size, n_tokens, *_ = xs.shape
        t = torch.sigmoid(
            torch.randn(batch_size, n_tokens, device=xs.device, generator=self.generator)
        )
        noise_levels = self.noise_shift * t / (1 + (self.noise_shift - 1) * t)
        xs_pred, raw_loss = self.diffusion_model(
            xs, self._process_conditions(conditions), k=noise_levels,
        )
        loss = raw_loss.mean()
        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        output_dict = {
            "loss": loss,
            "base_loss": loss.detach(),
            "rgbd_mse_loss": loss.detach(),
            # The old all-ones loss mask materialized contiguous channel slices.
            "rgb_mse_loss": raw_loss[:, :, :3].contiguous().mean().detach(),
            "depth_mse_loss": (
                raw_loss[:, :, 3:4].contiguous().mean().detach()
                if self.x_shape[0] == 4 else loss.new_zeros(())
            ),
            "xs": xs,
            "xs_pred": xs_pred,
        }
        return output_dict

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def sample_videos(self, xs: Tensor, conditions: Tensor) -> Dict[str, Tensor]:
        xs, conditions = self.preprocess_video_cond(xs, conditions)
        self.diffusion_model.eval()
        prediction = self._predict_sequence(
            xs[:, :self.n_context_frames], conditions, self.history_guidance_scale,
        )
        gt = self._unnormalize_x(xs).detach()
        prediction = self._unnormalize_x(prediction).detach()
        prediction[:, :self.n_context_frames] = gt[:, :self.n_context_frames]
        return {"gt": gt, "prediction": prediction}

    # ---------------------------------------------------------------------
    # Sampling Utils
    # ---------------------------------------------------------------------

    def _predict_sequence(self, context, conditions, guidance_scale):
        """Predict every frame, retaining the most recent history between windows."""
        length = conditions.shape[1]
        context_len = self.sliding_context_len or self.n_context_frames
        if context_len == -1:
            context_len = self.max_frames - 1
        if not context.shape[1] <= context_len < self.max_frames:
            raise ValueError("sliding_context_len must cover the initial context and be < max_frames")

        xs_pred = context
        batch_size = context.shape[0]
        pbar = tqdm(
            total=self.sampling_timesteps * (
                1 + (length - context_len - 1) // (self.max_frames - context_len)
            ),
            desc=f"[Rank {self.global_rank}] Predicting video",
            leave=False,
        )
        try:
            while xs_pred.shape[1] < length:
                current = xs_pred.shape[1]
                history_len = min(context_len, current)
                generated_len = min(length - current, self.max_frames - history_len)
                start = current - history_len
                window = torch.cat([
                    xs_pred[:, start:],
                    xs_pred.new_zeros(batch_size, generated_len, *self.x_shape),
                ], dim=1)
                # Both ground-truth and previously generated frames are clean history.
                mask = torch.zeros(window.shape[:2], dtype=torch.long, device=self.device)
                mask[:, :history_len] = 1
                cameras = conditions[:, start:start + self.max_frames]
                cameras = torch.cat([
                    cameras,
                    cameras[:, -1:].expand(-1, self.max_frames - cameras.shape[1], -1),
                ], dim=1)
                prediction = self._sample_sequence(window, mask, cameras, guidance_scale, pbar)
                xs_pred = torch.cat([xs_pred, prediction[:, -generated_len:]], dim=1)
        finally:
            pbar.close()
        return xs_pred

    def _sample_sequence(self, context, context_mask, conditions, guidance_scale, pbar):
        """Denoise one window, keeping its context and padding unchanged."""
        batch_size, length = context.shape[:2]
        padding = self.max_frames - length
        xs_pred = torch.randn((batch_size, self.max_frames, *self.x_shape), device=self.device)
        xs_pred = xs_pred.clamp(-self.clip_noise, self.clip_noise)
        if padding:
            context = torch.cat([context, context.new_zeros(batch_size, padding, *self.x_shape)], dim=1)
            context_mask = F.pad(context_mask, (0, padding), value=-1)
        history = context_mask[..., None, None, None] >= 1
        generated = context_mask[..., None, None, None] == 0
        xs_pred = torch.where(history, context, xs_pred)

        indices = torch.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(1, length)
        schedule = self.diffusion_model.ddim_idx_to_noise_level(indices)
        schedule = F.pad(schedule, (0, padding), value=self.timesteps - 1).to(self.device)
        schedule = repeat(schedule, "m t -> m b t", b=batch_size)
        schedule = torch.where(context_mask[None] >= 1, -1, schedule)

        # Camera normalization is independent of the denoising step. Keep the same
        # interleaved batch shape as the original sampler to preserve numerics.
        nfe = 1 if guidance_scale == 1 else 2
        conditions = self._process_conditions(
            repeat(conditions, "b ... -> (b nfe) ...", nfe=nfe).clone(),
        )
        pbar.set_postfix(NFE=nfe)
        for from_levels, to_levels in zip(schedule[:-1], schedule[1:]):
            inputs = xs_pred
            if nfe == 2:
                # Each sample pairs noisy history with clean history; cameras stay identical.
                inputs = repeat(inputs, "b ... -> b h ...", h=2).clone()
                from_levels, to_levels = (
                    repeat(levels, "b t -> b h t", h=2).clone()
                    for levels in (from_levels, to_levels)
                )
                for levels in (from_levels, to_levels):
                    levels[:, 0] = torch.where(context_mask >= 1, self.timesteps - 1, levels[:, 0])
                # Draw noise for the full window, including non-history frames, to preserve RNG.
                replacement = self.diffusion_model.q_sample(inputs[:, 0], from_levels[:, 0])
                inputs[:, 0] = torch.where(history, replacement, inputs[:, 0])
                inputs, from_levels, to_levels = (
                    value.flatten(0, 1) for value in (inputs, from_levels, to_levels)
                )
            prediction = self.diffusion_model.sample_step(inputs, from_levels, to_levels, conditions)
            if nfe == 2:
                prediction = prediction.unflatten(0, (batch_size, 2))
                # Preserve multiplication/subtraction order for checkpoint reproducibility.
                prediction = prediction[:, 1] * guidance_scale - prediction[:, 0] * (guidance_scale - 1)
            xs_pred = torch.where(generated, prediction, xs_pred)
            pbar.update(1)
        return xs_pred[:, :length]

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
