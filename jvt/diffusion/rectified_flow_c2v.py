import math

import torch
from typing import Optional
from omegaconf import DictConfig
from einops import rearrange

from .backbones import JVTC2V
from .rectified_flow import ModelPrediction, RectifiedFlow

class RectifiedFlowC2V(RectifiedFlow):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        rg_scale=None,
    ):
        super().__init__(cfg, backbone_cfg, x_shape, max_tokens)

        if not backbone_cfg.name.startswith("jvt-c2v"):
            raise ValueError(
                f"RectifiedFlowC2V requires a jvt-c2v backbone, got {backbone_cfg.name}."
            )

        self.rg_scale = self._validate_register_guidance(rg_scale, backbone_cfg)

        self.temporal_downsample = int(backbone_cfg.get("temporal_downsample", 1))
        if self.temporal_downsample < 1:
            raise ValueError(
                f"temporal_downsample must be >= 1, got {self.temporal_downsample}."
            )

    def _validate_register_guidance(self, scale, backbone_cfg) -> Optional[float]:
        if scale is None:
            return None

        if isinstance(scale, bool):
            raise ValueError("rg_scale must be a non-negative finite number.")
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("rg_scale must be a non-negative finite number.") from exc
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("rg_scale must be a non-negative finite number.")
        if not isinstance(self.model, JVTC2V):
            raise ValueError("Register guidance only supports the standard JVTC2V backbone.")

        in_context_len = int(backbone_cfg.get("in_context_len", 0))
        in_context_start = int(backbone_cfg.get("in_context_start", -1))
        depth = int(backbone_cfg.depth)
        if in_context_len <= 0 or not 0 <= in_context_start < depth:
            raise ValueError(
                "Register guidance requires an active in-context register: "
                "in_context_len must be positive and in_context_start must be within "
                f"[0, {depth}), got in_context_len={in_context_len} and "
                f"in_context_start={in_context_start}."
            )
        return scale

    def _guided_model_predictions(
        self,
        x,
        t,
        external_cond=None,
        uncond_external_cond=None,
        cfg_scale=1.0,
    ):
        if self.rg_scale is None or self.rg_scale == 0.0:
            return super()._guided_model_predictions(
                x=x,
                t=t,
                external_cond=external_cond,
                uncond_external_cond=uncond_external_cond,
                cfg_scale=cfg_scale,
            )
        if external_cond is None:
            raise ValueError("Register guidance requires class conditions.")

        def model_predictions_with_register(use_register, cond):
            return self._model_predictions(
                x=x,
                t=t,
                external_cond=cond,
                model_kwargs={"use_register": use_register},
            )

        pred_nr = model_predictions_with_register(False, external_cond)
        if uncond_external_cond is None:
            pred_r = model_predictions_with_register(True, external_cond)

            def apply_rg(value_r, value_nr):
                return value_nr + self.rg_scale * (value_r - value_nr)

            return ModelPrediction(
                pred_v=apply_rg(pred_r.pred_v, pred_nr.pred_v),
                pred_x0=apply_rg(pred_r.pred_x0, pred_nr.pred_x0),
                model_out=apply_rg(pred_r.model_out, pred_nr.model_out),
            )

        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        external_cond_in = torch.cat([uncond_external_cond, external_cond], dim=0)
        pred_reg = self._model_predictions(
            x=x_in,
            t=t_in,
            external_cond=external_cond_in,
            model_kwargs={"use_register": True},
        )

        if torch.is_tensor(cfg_scale):
            cfg_scale = cfg_scale.to(device=pred_reg.pred_v.device, dtype=pred_reg.pred_v.dtype)

        def apply_guidance(value_reg, value_nr):
            value_u, value_r = value_reg.chunk(2)
            return (
                value_u
                + cfg_scale * (value_r - value_u)
                + self.rg_scale * (value_r - value_nr)
            )

        return ModelPrediction(
            pred_v=apply_guidance(pred_reg.pred_v, pred_nr.pred_v),
            pred_x0=apply_guidance(pred_reg.pred_x0, pred_nr.pred_x0),
            model_out=apply_guidance(pred_reg.model_out, pred_nr.model_out),
        )
        
    def adjust_t_shape(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = rearrange(t, f"b -> b 1{' 1' * (len(self.x_shape))}")
        elif t.ndim == 2:
            if self.temporal_downsample > 1:
                t = t.repeat_interleave(self.temporal_downsample, dim=-1)
            t = rearrange(t, f"b t -> b t{' 1' * (len(self.x_shape))}")
        else:
            raise ValueError(f"Unsupported t shape: {t.shape}")
        return t
