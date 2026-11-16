# Reference: https://github.com/kwsong0113/diffusion-forcing-transformer/blob/530f8bf4db91a21964993f214fa4050dbd529591/datasets/video/base_video.py

import bisect
import random
from itertools import accumulate
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as torchvisionF
from omegaconf import DictConfig
from torchcodec.decoders import VideoDecoder
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
from tqdm import tqdm


class BaseVideoDataset(torch.utils.data.Dataset):
    """Training video metadata, deterministic clip sampling, and frame decoding."""

    video_pattern = "*.mp4"
    center_crop = False

    def __init__(self, cfg: DictConfig, current_epoch: int | None = None):
        self.resolution = cfg.resolution
        self.save_dir = Path(cfg.save_dir)
        self.metadata_file = Path(cfg.metadata_file)
        self.frame_skip = cfg.frame_skip
        self.output_frames = cfg.max_frames
        self.n_frames = 1 + (self.output_frames - 1) * self.frame_skip
        self.subdataset_size = cfg.subdataset_size

        if not self.metadata_file.exists():
            self._build_metadata()
        metadata = torch.load(self.metadata_file, weights_only=False)
        # Caches may have been created before the dataset directory was moved.
        metadata["video_paths"] = [
            self.resolve_video_path(Path(path)) for path in metadata["video_paths"]
        ]
        self.metadata = self.exclude_short_videos(
            [dict(zip(metadata, values)) for values in zip(*metadata.values())],
            self.n_frames,
        )
        if not self.metadata:
            raise ValueError(f"No usable training videos in {self.save_dir}")

        self.cumulative_sizes = list(accumulate(
            max(self.video_length(meta) - self.n_frames + 1, 1) for meta in self.metadata
        ))
        self.idx_remap = []
        self.set_epoch(current_epoch)

    def _build_metadata(self) -> None:
        paths = sorted(self.save_dir.glob(self.video_pattern), key=str)
        if not paths:
            raise FileNotFoundError(f"No videos matching {self.video_pattern} in {self.save_dir}")
        loader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(paths), batch_size=16,
            num_workers=min(16, len(paths)), collate_fn=_collate_fn,
        )
        metadata = {
            "video_paths": [path.relative_to(self.save_dir) for path in paths],
            "video_pts": [], "video_fps": [],
        }
        for batch in tqdm(loader, desc="Building train metadata"):
            for pts, fps in batch:
                metadata["video_pts"].append(torch.as_tensor(pts, dtype=torch.long))
                metadata["video_fps"].append(fps)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(metadata, self.metadata_file)

    def resolve_video_path(self, cached_path: Path) -> Path:
        """Flat video datasets are identified by filename under the current root."""
        return self.save_dir / cached_path.name

    def video_length(self, video_metadata: dict) -> int:
        return len(video_metadata["video_pts"])

    def exclude_short_videos(self, metadata: list[dict], min_frames: int) -> list[dict]:
        filtered = [meta for meta in metadata if self.video_length(meta) >= min_frames]
        print(f"train: {len(filtered)} / {len(metadata)} videos have at least {min_frames} frames")
        return filtered

    def set_epoch(self, current_epoch: int | None) -> None:
        """Take consecutive subdatasets from reproducible full-dataset shuffles."""
        self.current_subepoch = current_epoch
        total = self.cumulative_sizes[-1]
        use_subset = (
            current_epoch is not None and self.subdataset_size is not None
            and self.subdataset_size < total
        )
        if self.idx_remap and not use_subset:
            return
        size = self.subdataset_size if use_subset else total
        epoch, start = divmod(size * current_epoch if use_subset else 0, total)
        self.idx_remap = []
        while len(self.idx_remap) < size:
            indices = list(range(total))
            # Preserve the RNG state used by the existing training augmentations.
            random.seed(epoch)
            random.shuffle(indices)
            self.idx_remap.extend(indices[start:start + size - len(self.idx_remap)])
            epoch, start = epoch + 1, 0

    def __len__(self) -> int:
        return len(self.idx_remap)

    def get_clip_location(self, idx: int) -> tuple[int, int]:
        idx = self.idx_remap[idx]
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        start = idx - (self.cumulative_sizes[video_idx - 1] if video_idx else 0)
        return video_idx, start

    def load_video(
        self, video_metadata: dict, start_frame: int, end_frame: int, frame_step: int = 1,
    ) -> torch.Tensor:
        video = VideoDecoder(
            str(video_metadata["video_paths"]), dimension_order="NCHW",
        ).get_frames_in_range(start_frame, end_frame, frame_step).data
        video = video.to(torch.float32) / 255.0
        if self.center_crop:
            return torchvisionF.center_crop(video, [self.resolution, self.resolution])
        if video.shape[-2:] != (self.resolution, self.resolution):
            raise RuntimeError(
                f"{self.__class__.__name__} expects preprocessed {self.resolution}x{self.resolution} "
                f"frames, got {tuple(video.shape[-2:])}"
            )
        return video

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        video_idx, start = self.get_clip_location(idx)
        meta = self.metadata[video_idx]
        end = min(start + self.n_frames, self.video_length(meta))
        video = self.load_video(meta, start, end, self.frame_skip)
        if len(video) != self.output_frames:
            raise ValueError(f"Expected {self.output_frames} frames, got {len(video)} from {meta['video_paths']}")
        return {"videos": video, "nonterminal": torch.ones(self.output_frames, dtype=torch.bool)}
