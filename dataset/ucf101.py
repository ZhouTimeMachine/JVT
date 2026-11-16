from pathlib import Path

import torch
from omegaconf import DictConfig

from .base import BaseVideoDataset


class UCF101VideoDataset(BaseVideoDataset):
    """UCF101 clips with optional class labels (num_classes=0 disables them)."""

    video_pattern = "**/*.avi"
    center_crop = True

    def __init__(self, cfg: DictConfig, current_epoch: int | None = None):
        classes = sorted(p.name for p in Path(cfg.save_dir).iterdir() if p.is_dir() and p.name != "metadata")
        if not classes:
            raise FileNotFoundError(f"No class folders in {cfg.save_dir}")
        self.num_classes = cfg.num_classes
        if self.num_classes not in (0, len(classes)):
            raise ValueError(f"Expected {self.num_classes} classes, found {len(classes)} in {cfg.save_dir}")
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        super().__init__(cfg, current_epoch=current_epoch)

    def resolve_video_path(self, cached_path: Path) -> Path:
        return self.save_dir / cached_path.parent.name / cached_path.name

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        if self.num_classes:
            video_idx, _ = self.get_clip_location(idx)
            class_name = Path(self.metadata[video_idx]["video_paths"]).parent.name
            sample["conds"] = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        return sample
