from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from evaluation.metrics.pi3.utils.geometry import se3_inverse

def recover_intrinsic_from_rays_d(
    rays_d: torch.Tensor, 
    ndc_coords: bool = False,
    force_center_principal_point: bool = False
):
    device = rays_d.device
    dtype = rays_d.dtype
    
    *batch_dims, H, W, _ = rays_d.shape
    num_points = H * W

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=dtype),
        indexing="ij" 
    )

    if ndc_coords:
        grid_x = grid_x * (2.0 / (W - 1)) - 1.0
        grid_y = grid_y * (2.0 / (H - 1)) - 1.0

    u_targets = grid_x.reshape(1, num_points).expand(*batch_dims, num_points) # (..., N)
    v_targets = grid_y.reshape(1, num_points).expand(*batch_dims, num_points) # (..., N)

    z_abs = rays_d[..., 2].abs() + 1e-10 
    x_projs = (rays_d[..., 0] / z_abs).reshape(*batch_dims, num_points) # (..., N)
    y_projs = (rays_d[..., 1] / z_abs).reshape(*batch_dims, num_points) # (..., N)

    if force_center_principal_point:
        if ndc_coords:
            cx_val, cy_val = 0.0, 0.0
        else:
            cx_val, cy_val = (W - 1) / 2.0, (H - 1) / 2.0
            
        u_centered = u_targets - cx_val
        v_centered = v_targets - cy_val
        
        fx = (x_projs * u_centered).sum(dim=-1) / (x_projs * x_projs).sum(dim=-1)
        fy = (y_projs * v_centered).sum(dim=-1) / (y_projs * y_projs).sum(dim=-1)
        
        cx = torch.full_like(fx, cx_val)
        cy = torch.full_like(fy, cy_val)
        
    else:
        def solve_linear_least_squares(X, Y):
            mean_X = X.mean(dim=-1, keepdim=True)
            mean_Y = Y.mean(dim=-1, keepdim=True)
            X_centered = X - mean_X
            Y_centered = Y - mean_Y
            
            a = (X_centered * Y_centered).sum(dim=-1) / (X_centered * X_centered).sum(dim=-1)
            b = mean_Y.squeeze(-1) - a * mean_X.squeeze(-1)
            return a, b

        fx, cx = solve_linear_least_squares(x_projs, u_targets)
        fy, cy = solve_linear_least_squares(y_projs, v_targets)

    K = torch.zeros((*batch_dims, 3, 3), device=device, dtype=dtype)
    K[..., 0, 0] = fx
    K[..., 0, 2] = cx
    K[..., 1, 1] = fy
    K[..., 1, 2] = cy
    K[..., 2, 2] = 1.0

    return K


class Pi3XOnlineDisparityHelper:
    def __init__(
        self,
        device: torch.device,
        ckpt_path: str,
        crop_size: int = 252,
    ):
        self.device = device
        self.crop_size = crop_size

        from evaluation.metrics.pi3.models.pi3x import Pi3X
        model = Pi3X().to(device).eval()
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_safetensors(ckpt_path, device="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Pi3X checkpoint mismatch. Missing: {missing_keys}; Unexpected: {unexpected_keys}"
            )
        model.requires_grad_(False)
        self.model = model

        self.autocast_dtype: Optional[torch.dtype] = None
        assert self.device.type == "cuda", "Pi3X online disparity helper currently only supports CUDA devices."
        major, _ = torch.cuda.get_device_capability(self.device)
        self.autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16

    def _build_camera_inputs(
        self, raw_camera_poses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intrinsics_vec, extrinsics_vec = raw_camera_poses.float().split([4, 12], dim=-1)
        fx, fy, cx, cy = intrinsics_vec.unbind(dim=-1)
        focal = (fx + fy) / 2

        intrinsics = torch.zeros(
            *intrinsics_vec.shape[:-1], 3, 3,
            device=raw_camera_poses.device,
            dtype=torch.float32,
        )
        intrinsics[..., 0, 0] = focal * self.crop_size
        intrinsics[..., 1, 1] = focal * self.crop_size
        intrinsics[..., 0, 2] = cx * self.crop_size
        intrinsics[..., 1, 2] = cy * self.crop_size
        intrinsics[..., 2, 2] = 1.0

        extrinsics = torch.zeros(
            *extrinsics_vec.shape[:-1], 4, 4,
            device=raw_camera_poses.device,
            dtype=torch.float32,
        )
        extrinsics[..., :3, :] = extrinsics_vec.reshape(*extrinsics_vec.shape[:-1], 3, 4)
        extrinsics[..., 3, 3] = 1.0
        poses = se3_inverse(extrinsics)

        return intrinsics, poses, extrinsics

    @staticmethod
    def solve_sim3_camera(pred_poses, gt_poses):
        """
        求解最优的正数 scale (s) 和 translation (t)
        Args:
            pred_poses: [bsz, 8, 4, 4]
            gt_poses:   [bsz, 8, 4, 4]
        """
        # 1. 提取位置 [bsz, 8, 3]
        x = pred_poses[:, :, :3, 3]
        y = gt_poses[:, :, :3, 3]
        
        # 2. 计算质心 [bsz, 1, 3]
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)
        
        # 3. 去中心化
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # 4. 计算最优正数 Scale s
        # 使用方差比例: s = sqrt( sum(||y'||^2) / sum(||x'||^2) )
        # sum(||y'||^2) 等价于 (y_centered**2).sum()
        dist_y = (y_centered ** 2).sum(dim=(1, 2)) # [bsz]
        dist_x = (x_centered ** 2).sum(dim=(1, 2)) # [bsz]
        
        s = torch.sqrt(dist_y / (dist_x + 1e-8))
        s = s.unsqueeze(-1) # [bsz, 1]
        
        # 5. 计算最优 Translation t
        # t = y_mean - s * x_mean
        t = y_mean.squeeze(1) - s * x_mean.squeeze(1) # [bsz, 3]
        
        return s, t

    @torch.inference_mode()
    def infer_disparity_and_points(
        self,
        videos: torch.Tensor,
        raw_camera_poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if videos.ndim != 5 or videos.shape[2] != 3:
            raise ValueError(
                f"Pi3X online disparity expects RGB videos of shape [B, T, 3, H, W], got {tuple(videos.shape)}"
            )
        if raw_camera_poses is None or raw_camera_poses.shape[-1] != 16:
            raise ValueError(
                "Pi3X online disparity expects raw camera poses with shape [B, T, 16]."
            )

        bsz, temporal_length, _, height, width = videos.shape
        resized_videos = F.interpolate(
            videos.reshape(-1, 3, height, width),
            size=(self.crop_size, self.crop_size),
            mode="area"
        ).reshape(bsz, temporal_length, 3, self.crop_size, self.crop_size)
        intrinsics, poses, _ = self._build_camera_inputs(raw_camera_poses)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            outputs = self.model(
                imgs=resized_videos,
                intrinsics=intrinsics,
                poses=poses,
                with_prior=True,
            )

        # rays_d = torch.nn.functional.normalize(outputs['local_points'], dim=-1)
        # intrinsics_pred = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)

        points = outputs["points"].float()
        points = F.interpolate(
            points.reshape(-1, self.crop_size, self.crop_size, 3).permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(bsz, temporal_length, height, width, 3)

        depth = outputs["local_points"][..., 2].float()
        depth = F.interpolate(
            depth.reshape(-1, 1, self.crop_size, self.crop_size),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(bsz, temporal_length, 1, height, width)

        scale, _ = self.solve_sim3_camera(outputs["camera_poses"].float(), poses)
        depth = depth * scale.view(bsz, 1, 1, 1, 1)

        # Always clamp to avoid a per-step GPU-to-CPU synchronization from
        # checking whether clamping is needed.
        depth = depth.clamp_min(1e-5)
        depth = depth + 1.0
        depth = torch.sqrt(depth)
        disparity = 1.0 / depth

        return disparity, points

    @torch.inference_mode()
    def infer(
        self,
        videos: torch.Tensor,
        raw_camera_poses: torch.Tensor,
    ) -> torch.Tensor:
        disparity, _ = self.infer_disparity_and_points(videos, raw_camera_poses)
        return disparity

# pts1 = unproject_depth_map_to_point_map(depth[1].unsqueeze(-1), se3_inverse(outputs["camera_poses"])[1], intrinsics[1])

# from evaluation.metrics.pi3.utils.basic import write_ply
# unproject_depth_map_to_point_map from vggt

# pts0 = unproject_depth_map_to_point_map(depth[0, :, 0].unsqueeze(-1), se3_inverse(poses)[0], intrinsics[0])
# write_ply(pts0, videos[0].permute(0, 2, 3, 1), "tmp0.ply")

# pts1 = unproject_depth_map_to_point_map(depth[1, :, 0].unsqueeze(-1), se3_inverse(poses)[1], intrinsics[1])
# write_ply(pts1, videos[1].permute(0, 2, 3, 1), "tmp1.ply")
