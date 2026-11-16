import numpy as np
from omegaconf import DictConfig

from .base import BaseVideoDataset


class FaceForensicsVideoDataset(BaseVideoDataset):
    """Unconditional clips from preprocessed FaceForensics videos."""

    video_pattern = "*.mkv"
    center_crop = True


class SkyTimelapseVideoDataset(BaseVideoDataset):
    """Sample one random clip per video and repetition, refreshed each epoch."""

    def __init__(self, cfg: DictConfig, current_epoch: int | None = None):
        self.duplicates = cfg.duplicates
        super().__init__(cfg, current_epoch=current_epoch)

    def set_epoch(self, current_epoch: int | None) -> None:
        self.current_subepoch = current_epoch
        rng = np.random.default_rng(42 + (current_epoch or 0))
        offsets = [self.video_length(meta) - self.n_frames + 1 for meta in self.metadata]
        self.idx_remap = rng.integers(0, np.tile(offsets, self.duplicates)).tolist()

    def get_clip_location(self, idx: int) -> tuple[int, int]:
        return idx % len(self.metadata), self.idx_remap[idx]


class TaichiHDVideoDataset(BaseVideoDataset):
    """Unconditional clips from preprocessed Taichi-HD videos."""
