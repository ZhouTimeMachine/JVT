"""
Unconditional Video Inference for FVD Evaluation.

Generates N videos using the unconditional JVT model, then saves them as
sequentially numbered .mp4 files.

Usage:
    python evaluation/infer_uncond.py \
        algorithm=dit_video_uncond \
        resume=path/to/checkpoint.safetensors

Configuration: configs/eval.yaml with evaluation=infer (selected automatically).
"""

import datetime
import math
import os
import sys
import time

import imageio
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import hydra
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.distributed_utils import (
    init_distributed_mode,
    is_dist_avail_and_initialized,
)
from utils.checkpoint_utils import load_model_safetensors
from evaluation.logging_utils import print_cfg
from evaluation.video_utils import (
    save_video_local,
    build_initial_noise,
    get_missing_indices
)


def main(cfg: DictConfig, overrides):
    OmegaConf.set_struct(cfg, False)
    init_distributed_mode(cfg.infra)

    print_cfg(cfg, overrides)

    infra_cfg = cfg.infra

    algorithm_cfg: Optional[DictConfig] = cfg.algorithm
    if algorithm_cfg is None:
        raise ValueError("cfg.algorithm must be given for inference")
    algorithm_cfg.global_rank = infra_cfg.rank
    algorithm_cfg.world_size = infra_cfg.world_size
    algorithm_cfg.device = infra_cfg.device

    device = torch.device(infra_cfg.device)

    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Model
    from jvt.uncond_denoiser import UnCondDenoiser
    model = UnCondDenoiser(algorithm_cfg)

    checkpoint_path = cfg.resume
    if checkpoint_path is None:
        raise ValueError("No checkpoint specified.")
    elif not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print("Loading checkpoint from", checkpoint_path)

    model_state, meta = load_model_safetensors(checkpoint_path)
    model.load_state_dict(model_state, strict=True)
    cfg.epoch = int(meta.get("epoch", -1))
    print(f"Loaded model of epoch {cfg.epoch}")
    del model_state

    # Generate N videos with random class labels (Latte-style)
    num_samples = cfg.num_samples
    max_frames = algorithm_cfg.max_frames
    x_shape = list(algorithm_cfg.x_shape)
    batch_size = cfg.batch_size
    world_size = infra_cfg.world_size
    rank = infra_cfg.rank

    # total = math.ceil(num_samples / world_size) * world_size
    total = num_samples
    per_gpu = total // world_size

    assert cfg.output_dir is not None
    os.makedirs(cfg.output_dir, exist_ok=True)

    if is_dist_avail_and_initialized():
        torch.distributed.barrier()

    if rank == 0:
        missing_indices = get_missing_indices(cfg.output_dir, total, expected_frames=max_frames)
        missing_indices_tensor = torch.tensor(missing_indices, dtype=torch.long, device=device)
        print(
            f"Found {total - len(missing_indices)} existing videos, "
            f"need to generate {len(missing_indices)} more."
        )
    else:
        missing_indices_tensor = torch.empty(0, dtype=torch.long, device=device)

    if is_dist_avail_and_initialized():
        missing_count = torch.tensor([missing_indices_tensor.numel()], dtype=torch.long, device=device)
        torch.distributed.broadcast(missing_count, src=0)
        if rank != 0:
            missing_indices_tensor = torch.empty(int(missing_count.item()), dtype=torch.long, device=device)
        if missing_count.item() > 0:
            torch.distributed.broadcast(missing_indices_tensor, src=0)

    if missing_indices_tensor.numel() == 0:
        print("All requested videos already exist. Nothing to do.")
        return

    local_missing = missing_indices_tensor[rank::world_size]

    print(
        f"Generating {local_missing.numel()} missing videos on rank {rank} "
        f"(global missing: {missing_indices_tensor.numel()})"
    )

    model.eval()
    start_time = time.time()

    for i in tqdm(
        range(0, local_missing.numel(), batch_size),
        desc=f"[Rank {rank}] Generating",
    ):
        batch_missing = local_missing[i : i + batch_size]
        cur = batch_missing.numel()

        xs = torch.zeros(cur, max_frames, *x_shape, device=device)
        initial_noise = build_initial_noise(batch_missing, cfg.seed, max_frames, x_shape, device)

        all_videos = model.sample_videos(xs, initial_noise=initial_noise)
        video_fake = all_videos["prediction"]

        keys = [f"{int(idx):04d}" for idx in batch_missing.tolist()]
        save_video_local(keys=keys, video=video_fake, save_dir=cfg.output_dir, fps=8)

    elapsed = time.time() - start_time
    print(
        f"Done. Generated {local_missing.numel()} videos on rank {rank} in "
        f"{datetime.timedelta(seconds=int(elapsed))}"
    )


if __name__ == "__main__":
    overrides = sys.argv[1:]
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="eval", overrides=["evaluation=infer_uncond", *overrides])
    try:
        main(cfg, overrides)
    finally:
        if is_dist_avail_and_initialized():
            torch.distributed.destroy_process_group()
