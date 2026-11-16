import torch
from torch import nn, Tensor
from typing import Dict, Tuple

from .jvt_camctrl import (
    JVTCamCtrl,
    BottleneckPatchEmbed,
    FinalLayer,
)


class JVTCamCtrlJoint(JVTCamCtrl):
    """JVTCamCtrl variant that models RGB + disparity jointly."""

    joint_in_channels = 4
    joint_out_channels = 4

    def __init__(
        self,
        x_shape: torch.Size,
        max_tokens: int,
        hidden_size: int = 1024,
        bottleneck_dim: int = 128,
        **kwargs,
    ):
        super().__init__(
            x_shape,
            max_tokens,
            hidden_size=hidden_size,
            bottleneck_dim=bottleneck_dim,
            **kwargs,
        )

        input_size = x_shape[-1]

        self.in_channels = self.joint_in_channels
        self.out_channels = self.joint_out_channels

        self.x_embedder = BottleneckPatchEmbed(
            img_size=input_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
            bias=True,
        )
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)
        self._initialize_joint_io()

    def _initialize_joint_io(self) -> None:
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _adapt_patch_embed_weight(
        self,
        weight: Tensor,
        target: Tensor,
    ) -> Tensor:
        adapted = target.clone()
        adapted.zero_()

        in_channels_old = weight.shape[1]
        in_channels_new = adapted.shape[1]
        copy_channels = min(in_channels_old, in_channels_new)
        adapted[:, :copy_channels] = weight[:, :copy_channels].to(dtype=adapted.dtype)

        return adapted

    def _adapt_final_projection(
        self,
        tensor: Tensor,
        target: Tensor,
    ) -> Tensor:
        patch_area = self.patch_size * self.patch_size
        old_channels = tensor.shape[0] // patch_area
        new_channels = target.shape[0] // patch_area
        copy_channels = min(old_channels, new_channels)

        adapted = target.clone()
        adapted.zero_()

        if tensor.ndim == 2:
            hidden = tensor.shape[1]
            tensor_reshaped = tensor.reshape(self.patch_size, self.patch_size, old_channels, hidden)
            adapted_reshaped = adapted.reshape(self.patch_size, self.patch_size, new_channels, hidden)
            adapted_reshaped[..., :copy_channels, :] = tensor_reshaped[..., :copy_channels, :].to(
                dtype=adapted.dtype
            )
        else:
            tensor_reshaped = tensor.reshape(self.patch_size, self.patch_size, old_channels)
            adapted_reshaped = adapted.reshape(self.patch_size, self.patch_size, new_channels)
            adapted_reshaped[..., :copy_channels] = tensor_reshaped[..., :copy_channels].to(
                dtype=adapted.dtype
            )

        return adapted

    def load_pretrained_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        strict: bool = True,
    ) -> Tuple[list[str], list[str]]:
        current_state = self.state_dict()
        load_state: Dict[str, Tensor] = {}
        unexpected_keys: list[str] = []
        adapted_keys: list[str] = []

        for key, value in state_dict.items():
            if key not in current_state:
                unexpected_keys.append(key)
                continue

            target_value = current_state[key]
            if target_value.shape == value.shape:
                load_state[key] = value.to(dtype=target_value.dtype)
                continue

            if key == "x_embedder.proj1.weight":
                load_state[key] = self._adapt_patch_embed_weight(value, target_value)
                adapted_keys.append(key)
                continue

            if key in {"final_layer.linear.weight", "final_layer.linear.bias"}:
                load_state[key] = self._adapt_final_projection(value, target_value)
                adapted_keys.append(key)
                continue

            unexpected_keys.append(key)

        incompatible = self.load_state_dict(load_state, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys.extend(incompatible.unexpected_keys)

        if adapted_keys:
            print(
                "Adapted pretrained JVTPRoPE weights for joint RGB+disparity IO:",
                ", ".join(adapted_keys),
            )

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                "Failed to strictly load JVTPRoPEJoint pretrained weights. "
                f"Missing: {missing_keys}; Unexpected: {unexpected_keys}"
            )

        return missing_keys, unexpected_keys
