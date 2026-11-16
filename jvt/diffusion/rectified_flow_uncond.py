import math

import torch
from typing import Optional
from einops import rearrange
from omegaconf import DictConfig

from .rectified_flow import ModelPrediction, RectifiedFlow


class RectifiedFlowUncond(RectifiedFlow):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        rg_scale=None,
    ):
        super().__init__(cfg, backbone_cfg, x_shape, max_tokens)

        if not backbone_cfg.name.startswith("jvt-uncond"):
            raise ValueError(
                f"RectifiedFlowUncond requires a jvt-uncond backbone, got {backbone_cfg.name}."
            )

        self.rg_scale = self._validate_register_guidance(rg_scale, backbone_cfg)

        self.temporal_tokens = max_tokens
        self.temporal_downsample = int(backbone_cfg.get("temporal_downsample", 1))
        if self.temporal_downsample < 1:
            raise ValueError(
                f"temporal_downsample must be >= 1, got {self.temporal_downsample}."
            )
        self.temporal_length = self.temporal_tokens * self.temporal_downsample

    def _validate_register_guidance(self, scale, backbone_cfg) -> Optional[float]:
        # 1. disable register guidance if scale is None
        if scale is None: return None

        # 2. rg_scale must be a finite positive value
        if isinstance(scale, bool):
            raise ValueError("register_guidance_scale must be a non-negative finite number.")
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "register_guidance_scale must be a non-negative finite number."
            ) from exc
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("register_guidance_scale must be a non-negative finite number.")

        # 3. validate that the backbone has an active in-context register
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
        scale = self.rg_scale
        if scale is None:
            return super()._guided_model_predictions(
                x=x,
                t=t,
                external_cond=external_cond,
                uncond_external_cond=uncond_external_cond,
                cfg_scale=cfg_scale,
            )
        if external_cond is not None or uncond_external_cond is not None:
            raise ValueError("Register guidance does not support external conditions.")

        def model_predictions_with_register(use_register: bool) -> ModelPrediction:
            return self._model_predictions(
                x=x,
                t=t,
                external_cond=None,
                model_kwargs={"use_register": use_register},
            )

        if scale == 0.0: return model_predictions_with_register(False)
        if scale == 1.0: return model_predictions_with_register(True)

        pred_r = model_predictions_with_register(True)
        pred_nr = model_predictions_with_register(False)

        def apply_guidance(value_r, value_nr):
            return value_r + scale * (value_r - value_nr)

        return ModelPrediction(
            pred_v=apply_guidance(pred_r.pred_v, pred_nr.pred_v),
            pred_x0=apply_guidance(pred_r.pred_x0, pred_nr.pred_x0),
            model_out=apply_guidance(pred_r.model_out, pred_nr.model_out),
        )

    def expand_t(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] != self.temporal_tokens:
            raise ValueError(
                f"Expected token-level t with {self.temporal_tokens} entries, got {t.shape[-1]}."
            )
        return t.repeat_interleave(self.temporal_downsample, dim=-1)

    def adjust_t_shape(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = rearrange(t, f"b -> b 1{' 1' * (len(self.x_shape))}")
        elif t.ndim == 2:
            if t.shape[-1] == self.temporal_tokens:
                t = self.expand_t(t)
            elif t.shape[-1] != self.temporal_length:
                raise ValueError(
                    "Unsupported token shape for unconditional rectified flow: "
                    f"expected {self.temporal_tokens} tokens or {self.temporal_length} frames, got {t.shape[-1]}."
                )
            t = rearrange(t, f"b t -> b t{' 1' * (len(self.x_shape))}")
        else:
            raise ValueError(f"Unsupported t shape: {t.shape}")
        return t
