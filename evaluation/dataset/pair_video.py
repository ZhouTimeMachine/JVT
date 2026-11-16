import os
import imageio
import numpy as np
import torch

from typing import Tuple, Optional, List
from torch.utils.data import Dataset


class PairVideoDataset(Dataset):
    def __init__(
        self,
        seq_ids: List[str],
        fake_video_dir: str,
        real_video_dir: str,
        resolution_resized_to: Optional[Tuple[int, int]] = None,
    ):
        self.seq_ids = seq_ids
        self.fake_video_dir = fake_video_dir
        self.real_video_dir = real_video_dir
        self.resolution_resized_to = resolution_resized_to

    def __len__(self):
        return len(self.seq_ids)

    def load_video_and_preprocess(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        frames = imageio.mimread(video_path)
        video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0
        if (
            self.resolution_resized_to is not None
            and video_tensor.shape[-2:] != self.resolution_resized_to[::-1]
        ):
            video_tensor = torch.nn.functional.interpolate(
                video_tensor,
                size=self.resolution_resized_to[::-1],
                mode="bilinear",
                align_corners=False,
            )
        video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
        return video_tensor

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        fake_path = os.path.join(self.fake_video_dir, f"{seq_id}.mp4")
        real_path = os.path.join(self.real_video_dir, f"{seq_id}.mp4")

        fake_video = self.load_video_and_preprocess(fake_path)
        real_video = self.load_video_and_preprocess(real_path)

        return seq_id, fake_video, real_video


def collate_fn(batch):
    seq_ids = [item[0] for item in batch]
    fake_videos = torch.stack([item[1] for item in batch])
    real_videos = torch.stack([item[2] for item in batch])
    return seq_ids, fake_videos, real_videos


def get_paired_video_dirs(base_video_dir: str, real_video_dir: Optional[str] = None) -> Tuple[list, str, str]:
    fake_video_dir = os.path.join(base_video_dir, "fake")
    if real_video_dir is None:
        real_video_dir = os.path.join(base_video_dir, "real")
    if not os.path.exists(fake_video_dir):
        raise ValueError(f"Fake video directory not found: {fake_video_dir}")
    if not os.path.exists(real_video_dir):
        raise ValueError(f"Real video directory not found: {real_video_dir}")
    
    fake_files = {f for f in os.listdir(fake_video_dir) if f.endswith(".mp4")}
    real_files = {f for f in os.listdir(real_video_dir) if f.endswith(".mp4")}

    missing_files = fake_files - real_files
    if missing_files:
        sample_missing = list(missing_files)[:3]
        raise ValueError(
            f"Found {len(missing_files)} fake videos without matching real videos. "
            f"Examples: {sample_missing}"
        )

    seq_ids = sorted([os.path.splitext(f)[0] for f in fake_files])
    if len(seq_ids) == 0:
        raise ValueError(f"No mp4 files found in {fake_video_dir}")

    return seq_ids, fake_video_dir, real_video_dir