import os
import imageio
import numpy as np
import torch

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import List, Union


def _save_single_video(path: str, video_data: np.ndarray, fps: int):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    imageio.mimsave(path, video_data, fps=fps, quality=8, macro_block_size=None)

def _preprocess_video(video: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    if video.dtype != np.uint8:
        video = np.clip(video, a_min=0, a_max=1) * 255
        video = video.astype(np.uint8)
    return video

def normalize_seq_id(seq_id: str) -> str:
    return seq_id.replace("\\", "__").replace("/", "__")


def normalize_seq_ids(seq_ids: List[str]) -> List[str]:
    return [normalize_seq_id(seq_id) for seq_id in seq_ids]


def save_video_local(
    keys: List[str],
    video: Union[np.ndarray, torch.Tensor],
    save_dir: str,
    fps: int = 10,
    max_workers: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)

    video = _preprocess_video(video)

    if video.ndim != 5:
        raise ValueError(f"Expected 5 dims, got {video.ndim}")

    tasks = []
    for i in range(video.shape[0]):
        vid = video[i].transpose(0, 2, 3, 1)
        save_path = os.path.join(save_dir, f"{keys[i]}.mp4")
        tasks.append((save_path, vid, fps))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for path, vid, f in tasks:
            executor.submit(_save_single_video, path, vid, f)


def save_concat_video(
    keys: List[str],
    video_fake: Union[np.ndarray, torch.Tensor],
    video_real: Union[np.ndarray, torch.Tensor],
    save_dir: str,
    fps: int = 10,
    max_workers: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)

    video_fake = _preprocess_video(video_fake)
    video_real = _preprocess_video(video_real)

    if video_fake.ndim != 5 or video_real.ndim != 5:
        raise ValueError(f"Expected 5 dims, got fake={video_fake.ndim}, real={video_real.ndim}")
    if video_fake.shape[0] != len(keys):
        raise ValueError(f"Batch size mismatch: fake video {video_fake.shape[0]} vs keys {len(keys)}")
    if video_fake.shape != video_real.shape:
        raise ValueError(f"Shape mismatch: fake {video_fake.shape} vs real {video_real.shape}")

    tasks = []
    for i in range(video_fake.shape[0]):
        fake_vid = video_fake[i].transpose(0, 2, 3, 1)
        real_vid = video_real[i].transpose(0, 2, 3, 1)
        concat_vid = np.concatenate([fake_vid, real_vid], axis=2)
        save_path = os.path.join(save_dir, f"{keys[i]}.mp4")
        tasks.append((save_path, concat_vid, fps))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for path, vid, f in tasks:
            executor.submit(_save_single_video, path, vid, f)


def inspect_existing_video(task: tuple[str, str, int, int]) -> tuple[str, int | None, str | None]:
    save_dir, fname, total, expected_frames = task
    stem, ext = os.path.splitext(fname)
    if ext != ".mp4" or not stem.isdigit():
        return fname, None, "ignored"

    idx = int(stem)
    if idx < 0 or idx >= total:
        return fname, None, "ignored"

    video_path = os.path.join(save_dir, fname)
    try:
        if not os.path.isfile(video_path) or os.path.getsize(video_path) <= 0:
            return fname, idx, "empty_or_missing_file"

        # imageio.mimread is more reliable than metadata-only frame counting for mp4s.
        frames = imageio.mimread(video_path, memtest=False)
    except Exception:
        return fname, idx, "corrupted_or_unreadable"

    frame_count = len(frames)
    if frame_count != expected_frames:
        return fname, idx, f"unexpected_frame_count={frame_count}"

    return fname, idx, None


def get_missing_indices(
    save_dir: str,
    total: int,
    expected_frames: int,
    max_workers: int | None = None,
) -> list[int]:
    video_files = [fname for fname in os.listdir(save_dir) if fname.endswith(".mp4")]
    if not video_files:
        return list(range(total))

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(video_files))
    max_workers = max(1, min(max_workers, len(video_files)))

    tasks = [(save_dir, fname, total, expected_frames) for fname in sorted(video_files)]
    valid_existing = set()
    invalid_videos = []

    print(
        f"Checking {len(tasks)} existing videos with {max_workers} workers "
        f"(expected_frames={expected_frames})..."
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(inspect_existing_video, tasks)
        for fname, idx, reason in tqdm(results, total=len(tasks), desc="Checking existing videos"):
            if reason is None:
                valid_existing.add(idx)
            elif idx is not None:
                invalid_videos.append((fname, idx, reason))

    if invalid_videos:
        invalid_videos.sort(key=lambda item: item[1])
        print(f"Deleting {len(invalid_videos)} invalid videos before resume:")
        for fname, idx, reason in invalid_videos:
            print(f"  {fname} (index={idx}): {reason}")
            os.remove(os.path.join(save_dir, fname))

    return [idx for idx in range(total) if idx not in valid_existing]


def build_initial_noise(indices: torch.Tensor, seed: int, max_frames: int, x_shape: list[int], device: torch.device) -> torch.Tensor:
    noises = []
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    for idx in indices.tolist():
        generator = torch.Generator(device=generator_device).manual_seed(seed + idx)
        noises.append(
            torch.randn(
                (max_frames, *x_shape),
                device=device,
                generator=generator,
            )
        )
    return torch.stack(noises, dim=0)