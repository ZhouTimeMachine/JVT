import datetime
import json
import numpy as np
import os
import time
import math
import sys
import imageio

import torch
import torch.backends.cudnn as cudnn

import hydra
import torch

from omegaconf import DictConfig, OmegaConf
from typing import Union, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from jvt.camctrl_denoiser import CamCtrlDenoiser
from evaluation.video_utils import save_video_local, save_concat_video
from evaluation.logging_utils import print_cfg
from utils.distributed_utils import (
    init_distributed_mode, all_reduce_mean, 
    is_main_process, is_dist_avail_and_initialized,
    save_on_master
)
from utils.checkpoint_utils import (
    load_model_safetensors,
)


def build_camera_inputs(
    raw_cameras: torch.Tensor,
    resolution: Union[Tuple[int, int], List[int]], 
) -> Tuple[torch.Tensor, torch.Tensor]:
    if resolution[0] != resolution[1]:
        raise ValueError(f"Only support square images, found resolution: {resolution}")
    intrinsics_vec, extrinsics_vec = raw_cameras.float().split([4, 12], dim=-1)
    fx, fy, cx, cy = intrinsics_vec.unbind(dim=-1)
    focal = (fx + fy) / 2

    intrinsics = torch.zeros(
        *intrinsics_vec.shape[:-1], 3, 3,
        device=raw_cameras.device,
        dtype=torch.float32,
    )
    intrinsics[..., 0, 0] = focal * resolution[0]
    intrinsics[..., 1, 1] = focal * resolution[1]
    intrinsics[..., 0, 2] = cx * resolution[0]
    intrinsics[..., 1, 2] = cy * resolution[1]
    intrinsics[..., 2, 2] = 1.0

    extrinsics = torch.zeros(
        *extrinsics_vec.shape[:-1], 4, 4,
        device=raw_cameras.device,
        dtype=torch.float32,
    )
    extrinsics[..., :3, :] = extrinsics_vec.reshape(*extrinsics_vec.shape[:-1], 3, 4)
    extrinsics[..., 3, 3] = 1.0

    return intrinsics, extrinsics

def make_depth_visualization_video(depth_video: torch.Tensor, mode: str = "grayscale") -> torch.Tensor:
    if depth_video.ndim != 4:
        raise ValueError(f"Expected depth video with 4 dims [B, T, H, W], got {depth_video.ndim}")

    if mode == "color":
        import matplotlib
        # Aether's demo passes inverse depth to colorize_depth, despite its name.
        # https://github.com/InternRobotics/Aether/blob/871c6f7ebcd66f3571ed27db5d65264a51d624a3/scripts/demo.py#L475
        # JVT predicts 1/sqrt(depth+1), so use the recovered depth's reciprocal,
        # not the raw model channel. One range per video keeps frames consistent.
        clips = depth_video.detach().float().cpu().numpy()
        colored = []
        for clip in clips:
            valid = np.isfinite(clip) & (clip > 0)
            disparity = np.zeros_like(clip)
            with np.errstate(over="ignore", divide="ignore"):
                np.divide(1.0, clip, out=disparity, where=valid)
            valid &= np.isfinite(disparity) & (disparity > 0)
            positive = disparity[valid]
            rgb = np.zeros((*clip.shape, 3), dtype=np.float32)
            if positive.size and positive.max() > positive.min():
                normalized = (positive.max() - disparity) / (positive.max() - positive.min())
                rgb = matplotlib.colormaps["Spectral"](normalized.clip(0, 1))[..., :3].astype(np.float32)
                rgb[~valid] = 0
            colored.append(rgb)
        return torch.from_numpy(np.stack(colored)).permute(0, 1, 4, 2, 3).to(depth_video.device)
    elif mode == "grayscale":
        depth_min = depth_video.amin(dim=(1, 2, 3), keepdim=True)
        depth_max = depth_video.amax(dim=(1, 2, 3), keepdim=True)
        depth_range = (depth_max - depth_min).clamp_min(1e-8)
        depth_video = ((depth_video - depth_min) / depth_range).clamp(0, 1)
        return depth_video.unsqueeze(2).repeat(1, 1, 3, 1, 1)
    else:
        raise ValueError(f"Unknown depth visualization mode: {mode}")

def point_cloud_filter_marker_path(ply_path: str) -> str:
    return f"{ply_path}.depth_edge_filter.json"

def write_depth_point_cloud(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    path: str,
    pointcloud_cfg: DictConfig,
) -> None:
    from utils.geometry import unproject_depth_map_to_point_map
    from evaluation.metrics.pi3.utils.basic import write_ply

    points = unproject_depth_map_to_point_map(depth.unsqueeze(-1), extrinsics, intrinsics)
    if torch.is_tensor(points):
        points = points.detach().cpu()
    else:
        points = torch.from_numpy(points)
    depth_cpu = depth.detach().cpu().float()
    rgb = rgb.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu()
    marker_path = point_cloud_filter_marker_path(path)

    if pointcloud_cfg.filter_depth_edges or pointcloud_cfg.filter_depth_normal_edges:
        from evaluation.metrics.pi3.utils.geometry import depth_edge, depth_normal_edge

        valid = torch.isfinite(depth_cpu) & (depth_cpu > 0)

        if pointcloud_cfg.filter_depth_normal_edges:
            edge = depth_normal_edge(
                points.float(),
                depth_cpu,
                rtol=pointcloud_cfg.depth_edge_rtol,
                normal_tol_deg=pointcloud_cfg.normal_edge_tol_deg,
                kernel_size=pointcloud_cfg.depth_edge_kernel_size,
                mask=valid,
            )
        else:
            edge = depth_edge(
                depth_cpu,
                atol=pointcloud_cfg.depth_edge_atol,
                rtol=pointcloud_cfg.depth_edge_rtol,
                kernel_size=pointcloud_cfg.depth_edge_kernel_size,
                mask=valid,
            )

        keep = valid & ~edge
        total_points = int(depth_cpu.numel())
        kept_points = int(keep.sum().item())
        removed_edge_points = int((valid & edge).sum().item())
        invalid_points = total_points - int(valid.sum().item())

        keep = keep.detach().cpu()
        write_ply(points[keep], rgb[keep], path)
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "filter_depth_edges": pointcloud_cfg.filter_depth_edges,
                    "filter_depth_normal_edges": pointcloud_cfg.filter_depth_normal_edges,
                    "depth_edge_rtol": pointcloud_cfg.depth_edge_rtol,
                    "depth_edge_atol": pointcloud_cfg.depth_edge_atol,
                    "depth_edge_kernel_size": pointcloud_cfg.depth_edge_kernel_size,
                    "normal_edge_tol_deg": pointcloud_cfg.normal_edge_tol_deg,
                    "total_points": total_points,
                    "kept_points": kept_points,
                    "removed_edge_points": removed_edge_points,
                    "invalid_points": invalid_points,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        return

    write_ply(points, rgb, path)
    if os.path.exists(marker_path):
        os.remove(marker_path)

def has_required_outputs(
    seq_id: str,
    save_dir_fake: str,
    save_dir_real: str,
    save_dir_concat: str,
    save_real_video: bool,
    save_dir_fake_vis: str = None,
    pointcloud_filter_enabled: bool = False,
    expected_frames: int | None = None,
    remove_invalid_outputs: bool = True,
) -> bool:
    required_paths = [
        os.path.join(save_dir_fake, f"{seq_id}.mp4"),
        os.path.join(save_dir_concat, f"{seq_id}.mp4"),
    ]
    if save_real_video:
        required_paths.append(os.path.join(save_dir_real, f"{seq_id}.mp4"))
    if save_dir_fake_vis is not None:
        required_paths.extend([
            os.path.join(save_dir_fake_vis, f"{seq_id}.ply"),
            os.path.join(save_dir_fake_vis, f"{seq_id}.mp4"),
        ])
        if pointcloud_filter_enabled:
            required_paths.append(point_cloud_filter_marker_path(os.path.join(save_dir_fake_vis, f"{seq_id}.ply")))

    for path in required_paths:
        reason = inspect_required_output(path, expected_frames)
        if reason is None:
            continue
        if remove_invalid_outputs and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                print(f"Warning: failed to remove invalid output {path}: {exc}")
        print(f"Regenerating {seq_id}: invalid output {path} ({reason})")
        return False
    return True

def inspect_required_output(path: str, expected_frames: int | None = None) -> str | None:
    if path.endswith(".mp4"):
        return inspect_video_output(path, expected_frames)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return "empty_or_missing_file"
    return None

def inspect_video_output(path: str, expected_frames: int | None = None) -> str | None:
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return "empty_or_missing_file"
    if expected_frames is None:
        return None
    try:
        # Decode frames instead of relying on container metadata, matching the
        # resume checks used by uncond and UCF C2V inference.
        frames = imageio.mimread(path, memtest=False)
    except Exception:
        return "corrupted_or_unreadable"
    frame_count = len(frames)
    if frame_count != expected_frames:
        return f"unexpected_frame_count={frame_count}"
    return None

def main(cfg: DictConfig, overrides: List[str]):
    # runtime infra config
    OmegaConf.set_struct(cfg, False)
    init_distributed_mode(cfg.infra)
    print_cfg(cfg, overrides)

    dataset_cfg: DictConfig = cfg.dataset
    infra_cfg: DictConfig = cfg.infra
    pointcloud_cfg: DictConfig = cfg.pointcloud
    cfg.algorithm.global_rank = infra_cfg.rank
    cfg.algorithm.world_size = infra_cfg.world_size
    cfg.algorithm.device = infra_cfg.device

    # Set seeds for reproducibility
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(cfg.output_dir, exist_ok=True)

    from evaluation.dataset.realestate10k_eval import RealEstate10KEvalVideoDataset
    dataset_test = RealEstate10KEvalVideoDataset(**dataset_cfg)
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=infra_cfg.world_size,
        rank=infra_cfg.rank, shuffle=False, seed=cfg.seed,
    )
    print("Sampler_test =", sampler_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        shuffle=False, drop_last=False, persistent_workers=False,
        **cfg.dataloader
    )
    print(f"Dataset contains {len(dataset_test):,} videos")

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    if cfg.algorithm.name != "camctrl_denoiser":
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")
    model = CamCtrlDenoiser(cfg.algorithm).eval().requires_grad_(False)

    checkpoint_path = cfg.resume
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model_state, meta = load_model_safetensors(checkpoint_path)
    model.load_state_dict(model_state, strict=True)
    epoch = int(meta["epoch"])
    print(f"Loaded checkpoint {checkpoint_path}: epoch={epoch}, step={meta.get('global_step')}")
    del model_state

    eff_batch_size = cfg.dataloader.batch_size * infra_cfg.world_size
    print("Effective batch size: %d" % eff_batch_size)

    # Infering loop
    print(f"Start testing for {len(dataset_test)} -> {infra_cfg.world_size}*{cfg.dataloader.batch_size}*{len(data_loader_test)} videos...")
    start_time = time.time()
    model.eval()
    eval_video_tag = cfg.get("eval_video_tag", None)
    eval_video_suffix = f"_{eval_video_tag}" if eval_video_tag else ""
    for idx, batch in tqdm(enumerate(data_loader_test), total=len(data_loader_test)):
        video_real = batch['video']              # [N, C, H, W], in [0, 1]
        conditions = batch['raw_cameras']  # [N, 16], in original scale

        save_dir_fake = os.path.join(cfg.output_dir, "fake")
        save_dir_real = os.path.join(cfg.output_dir, "real")
        save_dir_concat = os.path.join(cfg.output_dir, "concat")
        use_online_disparity = getattr(model, "use_online_disparity", False)
        save_real_video = cfg.save_real_video
        save_dir_fake_vis = os.path.join(cfg.output_dir, "fake_vis") if use_online_disparity else None
        
        all_exist = True
        for seq in batch['seq_id']:
            if not has_required_outputs(
                seq_id=seq,
                save_dir_fake=save_dir_fake,
                save_dir_real=save_dir_real,
                save_dir_concat=save_dir_concat,
                save_real_video=save_real_video,
                save_dir_fake_vis=save_dir_fake_vis,
                pointcloud_filter_enabled=pointcloud_cfg.filter_depth_edges or pointcloud_cfg.filter_depth_normal_edges,
                expected_frames=dataset_cfg.num_frames,
            ):
                all_exist = False
                break
        if all_exist:
            continue

        # The reference sampler runs in FP32. Pi3X manages its own autocast;
        # wrapping the entire sampler changes both RGBD conditioning and denoising.
        all_videos = model.sample_videos(video_real, conditions)

        video_fake = all_videos['prediction']  # [N, T, C, H, W], in [0, 1]
        video_fake_rgb = video_fake[:, :, :3]
        # Save fake video
        if save_dir_fake_vis is not None:
            if conditions is None:
                raise ValueError("Online disparity visualization requires camera conditions.")
            os.makedirs(save_dir_fake_vis, exist_ok=True)

            disparity_video = video_fake[:, :, 3]
            disparity_video = disparity_video.clamp(1e-5, 1)
            depth_video = (1.0 / disparity_video) ** 2 - 1.0
            depth_video_vis = make_depth_visualization_video(depth_video, mode=cfg.depth_vis_mode)

            intrinsics, extrinsics = build_camera_inputs(conditions, dataset_cfg.resolution)

            for b in range(video_fake_rgb.shape[0]):
                write_depth_point_cloud(
                    depth=depth_video[b],
                    rgb=video_fake_rgb[b],
                    extrinsics=extrinsics[b],
                    intrinsics=intrinsics[b],
                    path=os.path.join(save_dir_fake_vis, f"{batch['seq_id'][b]}.ply"),
                    pointcloud_cfg=pointcloud_cfg,
                )

            save_concat_video(
                keys=batch['seq_id'],
                video_fake=video_fake_rgb,
                video_real=depth_video_vis,
                save_dir=save_dir_fake_vis,
                fps=10,
            )

        save_video_local(
            keys=batch['seq_id'],
            video=video_fake_rgb,
            save_dir=save_dir_fake,
            fps=10,
        )


        # Save real video
        if save_real_video:
            save_video_local(
                keys=batch['seq_id'],
                video=video_real,
                save_dir=save_dir_real,
                fps=10,
            )

        # Save concatenated video (fake + real side by side)
        save_concat_video(
            keys=batch['seq_id'],
            video_fake=video_fake_rgb,
            video_real=video_real,
            save_dir=save_dir_concat,
            fps=10,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Inference time:', total_time_str)


if __name__ == '__main__':
    overrides = sys.argv[1:]

    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="eval", overrides=overrides)

    try:
        main(cfg, overrides)
    finally:
        if is_dist_avail_and_initialized():
            torch.distributed.destroy_process_group()
