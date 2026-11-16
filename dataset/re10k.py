import random
from pathlib import Path

import torch
from omegaconf import DictConfig

from .base import BaseVideoDataset


def random_bool(p: float) -> bool:
    return abs(p) > 1e-8 and random.random() < p


class RealEstate10KVideoDataset(BaseVideoDataset):
    """256x256 MP4 training clips with camera poses and joint augmentations."""

    video_pattern = "train_256/*.mp4"

    def __init__(self, cfg: DictConfig, current_epoch: int | None = None):
        if cfg.resolution != 256:
            raise ValueError(f"Re10K only supports resolution=256, got {cfg.resolution}")
        self.maximize_training_data = cfg.maximize_training_data
        self.augmentation = cfg.augmentation
        super().__init__(cfg, current_epoch=current_epoch)

    def resolve_video_path(self, cached_path: Path) -> Path:
        return self.save_dir / "train_256" / cached_path.name

    def exclude_short_videos(self, metadata: list[dict], min_frames: int) -> list[dict]:
        if self.maximize_training_data:
            min_frames = min(min_frames, self.output_frames)
        return super().exclude_short_videos(metadata, min_frames)

    def _augment(self, video: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # https://github.com/kwsong0113/diffusion-forcing-transformer/blob/530f8bf4db91a21964993f214fa4050dbd529591/datasets/video/realestate10k.py#L286-L308
        if random_bool(self.augmentation.horizontal_flip_prob):
            video = video.flip(-1)
            # Extrinsics: E' = I' @ E @ I', with I' = diag([-1, 1, 1, 1]).
            cond[:, [5, 6, 7, 8, 12]] *= -1
        if random_bool(self.augmentation.back_and_forth_prob):
            video = torch.cat([video[::2], video[1::2].flip(0)]).contiguous()
            cond = torch.cat([cond[::2], cond[1::2].flip(0)]).contiguous()
        if random_bool(self.augmentation.reverse_prob):
            video, cond = video.flip(0).contiguous(), cond.flip(0).contiguous()
        return video, cond

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        video_idx, start = self.get_clip_location(idx)
        meta = self.metadata[video_idx]
        frame_skip = self.frame_skip
        if self.augmentation.frame_skip_increase:
            assert self.current_subepoch is not None, "Frame skip scheduling requires current_epoch"
            frame_skip += int(self.current_subepoch * self.augmentation.frame_skip_increase)
        frame_skip = min(frame_skip, (self.video_length(meta) - start - 1) // (self.output_frames - 1))
        assert frame_skip > 0, f"Frame skip must be positive, got {frame_skip}"
        end = start + (self.output_frames - 1) * frame_skip + 1
        video = self.load_video(meta, start, end, frame_skip)
        pose_path = self.save_dir / "train_poses" / f"{meta['video_paths'].stem}.pt"
        poses = torch.load(pose_path, weights_only=False)[start:end:frame_skip]
        cond = torch.cat([poses[:, :4], poses[:, 6:]], dim=-1).to(torch.float32)
        assert len(video) == len(cond), "Video and cond have different lengths"
        video, cond = self._augment(video, cond)
        return {"videos": video, "conds": cond, "nonterminal": torch.ones(self.output_frames, dtype=torch.bool)}
