"""Video datasets and preprocessing shared by FVD evaluations."""

import json
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


def resolve_mapped_video_path(metadata: dict, real_video_dir: str | None = None) -> str:
    if "video_path" not in metadata:
        seq = metadata.get("seq")
        if seq is None or real_video_dir is None:
            raise FileNotFoundError(
                "Could not resolve real video path from metadata without "
                "'video_path'; expected 'seq' plus real_video_dir"
            )
        video_path = Path(real_video_dir) / f"{seq}.mp4"
        if video_path.is_file():
            return str(video_path)
        raise FileNotFoundError(f"Could not resolve real video path {video_path}")

    video_path = Path(metadata["video_path"])
    if video_path.is_file():
        return str(video_path)

    candidates = []
    if real_video_dir is not None:
        real_root = Path(real_video_dir)
        # Flat preprocessed datasets and class/video.avi UCF101 caches/maps.
        candidates.extend((
            real_root / video_path.parent.name / video_path.name,
            real_root / video_path.name,
        ))
        parts = video_path.parts
        for marker in ("UCF-101", "UCF101", "ucf101"):
            if marker in parts:
                suffix = Path(*parts[parts.index(marker) + 1 :])
                if suffix.parts:
                    candidates.append(real_root / suffix)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    tried = ", ".join(str(path) for path in candidates) or "no fallback candidates"
    raise FileNotFoundError(f"Could not resolve real video path {video_path}; tried {tried}")


def preprocess_video(video: torch.Tensor, resolution: int) -> torch.Tensor:
    if video.dtype != torch.float32:
        video = video.to(torch.float32)
    if video.max() > 1.0:
        video = video / 255.0

    _, _, height, width = video.shape
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    video = video[:, :, top : top + crop_size, left : left + crop_size]

    if video.shape[-2:] != (resolution, resolution):
        video = F.interpolate(
            video,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )
    return torch.clamp(video, 0.0, 1.0)


class FakeVideoDataset(Dataset):
    def __init__(self, video_dir: str, num_frames: int = 16, resolution: int = 256):
        self.num_frames = num_frames
        self.resolution = resolution
        self.video_files = sorted(Path(video_dir).glob("*.mp4"))
        if not self.video_files:
            raise ValueError(f"No .mp4 files found in {video_dir}")
        print(f"[FakeVideoDataset] Found {len(self.video_files)} videos")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        frames = imageio.mimread(str(self.video_files[index]))
        if len(frames) < self.num_frames:
            raise ValueError(
                f"{self.video_files[index]} has {len(frames)} frames; need {self.num_frames}"
            )
        frames = frames[: self.num_frames]
        video = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)
        return preprocess_video(video, self.resolution)


class RealVideoFVDDataset(Dataset):
    def __init__(
        self,
        seq_id_map: str,
        real_video_dir: str | None = None,
        resolution: int = 256,
        num_frames: int = 16,
    ):
        self.real_video_dir = real_video_dir
        self.resolution = resolution
        self.num_frames = num_frames
        with open(seq_id_map, "r", encoding="utf-8") as f:
            self.seq_id_map = json.load(f)

    def __len__(self):
        return len(self.seq_id_map)

    def _resolve_frame_range(self, metadata: dict, decoder: VideoDecoder) -> tuple[int, int, int]:
        start_frame = int(metadata["start_frame"])
        frame_skip = int(metadata["frame_skip"])
        if frame_skip < 1:
            raise ValueError(f"Invalid frame_skip={frame_skip} for {metadata}")

        num_video_frames = decoder.metadata.num_frames
        last_frame = start_frame + (self.num_frames - 1) * frame_skip
        if last_frame >= num_video_frames:
            max_start = num_video_frames - 1 - (self.num_frames - 1) * frame_skip
            if max_start < 0:
                raise ValueError(
                    f"{resolve_mapped_video_path(metadata, self.real_video_dir)} has "
                    f"{num_video_frames} frames; cannot sample {self.num_frames} frames "
                    f"with frame_skip={frame_skip}."
                )
            start_frame = max_start
            last_frame = start_frame + (self.num_frames - 1) * frame_skip

        # torchcodec treats stop as an exclusive bound and rejects stop > num_frames,
        # even when the stepped indices would all be valid.
        stop_frame = last_frame + 1
        return start_frame, stop_frame, frame_skip

    def __getitem__(self, index):
        metadata = self.seq_id_map[index]
        decoder = VideoDecoder(
            resolve_mapped_video_path(metadata, self.real_video_dir),
            dimension_order="NCHW",
        )
        start_frame, stop_frame, frame_skip = self._resolve_frame_range(metadata, decoder)
        video = decoder.get_frames_in_range(start_frame, stop_frame, frame_skip).data
        if video.shape[0] != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames from "
                f"{resolve_mapped_video_path(metadata, self.real_video_dir)}, got {video.shape[0]}"
            )
        return preprocess_video(video, self.resolution)
