import torch
import numpy as np

from typing import Optional, Dict
from collections import namedtuple
from omegaconf import DictConfig, OmegaConf

from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .backbones import (
    JVTCamCtrl,
    JVTCamCtrlJoint,
    JVTUnCond,
    JVTC2V,
)


def extract(a, t, x_shape):
    shape = t.shape
    out = a[t]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))

ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_v", "pred_x0", "model_out"]
)

class RectifiedFlow(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
    ):
        super().__init__()
        self.x_shape = x_shape
        self.max_tokens = max_tokens
        self.shift = cfg.shift
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.sampling_method = cfg.get("sampling_method", "euler")
        if self.sampling_method not in {"euler", "heun"}:
            raise ValueError(f"unknown sampling method {self.sampling_method}")
        self.clip_noise = cfg.clip_noise

        self.backbone_cfg = backbone_cfg
        self.t_eps = cfg.t_eps
        self._build_model()

        # self.cfg = cfg  # for debug

    def _build_model(self):
        if self.backbone_cfg.name.startswith("jvt-camctrl"):
            model_cls = JVTCamCtrlJoint if self.backbone_cfg.name.startswith("jvt-camctrl-joint") else JVTCamCtrl
        elif self.backbone_cfg.name.startswith("jvt-uncond"):
            model_cls = JVTUnCond
        elif self.backbone_cfg.name.startswith("jvt-c2v"):
            model_cls = JVTC2V
        else:
            raise ValueError(f"unknown model type {self.backbone_cfg.name}")
        kwargs = OmegaConf.to_container(self.backbone_cfg, resolve=True)
        kwargs.pop("name", None)
        self.model = model_cls(
            x_shape=self.x_shape,
            max_tokens=self.max_tokens,
            **kwargs,
        )

    def load_pretrained_state_dict(self, state_dict, strict: bool = True):
        if not state_dict:
            return [], []

        if all(key.startswith("model.") for key in state_dict):
            model_state = {key[len("model.") :]: value for key, value in state_dict.items()}
        else:
            model_state = state_dict

        if hasattr(self.model, "load_pretrained_state_dict"):
            return self.model.load_pretrained_state_dict(model_state, strict=strict)

        incompatible = self.model.load_state_dict(model_state, strict=strict)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    def adjust_t_shape(self, t: torch.Tensor) -> torch.Tensor:
        return rearrange(t, f"... -> ...{' 1' * len(self.x_shape)}")

    def _model_predictions(self, x, t, external_cond=None, model_kwargs: Optional[Dict] = None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model_output = self.model(x, t, external_cond, **model_kwargs)

        t = self.adjust_t_shape(t)
        # x0-predition
        pred_x0 = model_output
        # pred_noise = (x - (1 - t) * pred_x0) / t.clamp_min(self.t_eps)  # noise <- x0
        pred_v = (pred_x0 - x) / t.clamp_min(self.t_eps)  # v <- x0

        return ModelPrediction(pred_v, pred_x0, model_output)

    def _guided_model_predictions(
        self,
        x,
        t,
        external_cond=None,
        uncond_external_cond=None,
        cfg_scale=1.0,
    ):
        if uncond_external_cond is None:
            return self._model_predictions(x, t, external_cond)
        if external_cond is None:
            raise ValueError("external_cond must be provided when using classifier-free guidance")

        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        external_cond_in = torch.cat([uncond_external_cond, external_cond], dim=0)

        model_pred = self._model_predictions(
            x=x_in,
            t=t_in,
            external_cond=external_cond_in,
        )

        if torch.is_tensor(cfg_scale):
            cfg_scale = cfg_scale.to(device=model_pred.pred_v.device, dtype=model_pred.pred_v.dtype)

        pred_v_uncond, pred_v_cond = model_pred.pred_v.chunk(2)
        pred_x0_uncond, pred_x0_cond = model_pred.pred_x0.chunk(2)
        model_out_uncond, model_out_cond = model_pred.model_out.chunk(2)

        pred_v = pred_v_uncond + cfg_scale * (pred_v_cond - pred_v_uncond)
        pred_x0 = pred_x0_uncond + cfg_scale * (pred_x0_cond - pred_x0_uncond)
        model_out = model_out_uncond + cfg_scale * (model_out_cond - model_out_uncond)

        return ModelPrediction(pred_v, pred_x0, model_out)

    def q_sample(self, x_start, k, noise=None):
        assert hasattr(self, "timestep2sigma"), "call ddim_idx_to_noise_level first to set timestep2sigma"
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        # NOTE: in this outer function, k in [-1, self.timesteps - 1]
        # t = self.adjust_t_shape((k + 1) / self.timesteps)
        if k.device != self.timestep2sigma.device:
            self.timestep2sigma = self.timestep2sigma.to(k.device)
        t = self.adjust_t_shape(self.timestep2sigma[k.long() + 1])
        return (1 - t) * x_start + t * noise
    
    def _q_sample(self, x_start, t, noise):
        t = self.adjust_t_shape(t)
        return (1 - t) * x_start + t * noise

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
    ):
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self._q_sample(x_start=x, t=k, noise=noise)
        model_pred = self._model_predictions(x=noised_x, t=k, external_cond=external_cond)

        # v-loss
        t = self.adjust_t_shape(k).clamp_min(self.t_eps)
        target = (x - noised_x) / t
        prediction = model_pred.pred_v
        loss = F.mse_loss(prediction, target.detach(), reduction="none")

        return model_pred.pred_x0, loss

    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
        shape = indices.shape
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        self.real_steps = real_steps.long().to(indices.device)
        self._build_denoise_schedule()
        k = self.real_steps[indices.flatten()]
        return k.view(shape)
    
    def shift_time(self, t):
        return self.shift * t / (1 + (self.shift - 1) * t)
    
    def _build_denoise_schedule(self):
        assert hasattr(self, "real_steps"), "call ddim_idx_to_noise_level first to set real_steps"
        device = self.real_steps.device

        # build for sampling timesteps
        timesteps = np.linspace(1, self.timesteps, self.timesteps, dtype=np.float32)[::-1]
        sigmas_train = timesteps / self.timesteps
        sigmas_train = self.shift_time(sigmas_train)
        sigma_min = sigmas_train[-1]
        sigma_max = sigmas_train[0]

        sigmas = np.linspace(sigma_max, sigma_min, self.sampling_timesteps, dtype=np.float32)
        sigmas = self.shift_time(sigmas)
        sigmas = np.concatenate([sigmas, np.array([0.0], dtype=np.float32)], axis=0)
        sigmas = torch.from_numpy(sigmas).to(device=device, dtype=torch.float32)

        self.timestep2sigma = torch.full((self.timesteps + 1, ), -1, device=device, dtype=torch.float32)
        self.timestep2sigma[self.real_steps + 1] = sigmas.flip(0)

    def sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        uncond_external_cond: Optional[torch.Tensor] = None,
        cfg_scale=1.0,
    ):
        assert hasattr(self, "timestep2sigma"), "call ddim_idx_to_noise_level first to set timestep2sigma"
        if curr_noise_level.device != self.timestep2sigma.device:
            self.timestep2sigma = self.timestep2sigma.to(curr_noise_level.device)
        curr_sigma = self.timestep2sigma[curr_noise_level.long() + 1]
        next_sigma = self.timestep2sigma[next_noise_level.long() + 1]
        model_pred = self._guided_model_predictions(
            x=x,
            t=curr_sigma,
            external_cond=external_cond,
            uncond_external_cond=uncond_external_cond,
            cfg_scale=cfg_scale,
        )
        pred_v = model_pred.pred_v

        delta_t = curr_sigma - next_sigma
        mask = curr_noise_level == next_noise_level
        delta_t = self.adjust_t_shape(delta_t)
        masked = self.adjust_t_shape(mask)

        x_euler = torch.where(
            masked,
            x,
            x + delta_t * pred_v,
        )
        if self.sampling_method == "euler":
            return x_euler

        pred_v_next = self._guided_model_predictions(
            x=x_euler,
            t=next_sigma,
            external_cond=external_cond,
            uncond_external_cond=uncond_external_cond,
            cfg_scale=cfg_scale,
        ).pred_v

        heun_mask = self.adjust_t_shape(next_sigma > self.t_eps)
        x_heun = x + 0.5 * delta_t * (pred_v + pred_v_next)

        return torch.where(
            masked,
            x,
            torch.where(heun_mask, x_heun, x_euler),
        )
