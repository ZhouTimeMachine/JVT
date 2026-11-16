# Reference: [DFoT](https://github.com/kwsong0113/diffusion-forcing-transformer)

import av
import io
import subprocess
import torch
import numpy as np

from typing import Optional, Dict
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
# from torchvision.io import write_video

# --- Configs ---
ROOT_DIR: Path        = Path("data/re10k")
RAW_VIDEO_DIR: Path   = ROOT_DIR / "raw_videos"
RAW_TXT_DIR: Path     = ROOT_DIR / "meta_txts"
OUTPUT_ROOT_DIR: Path = ROOT_DIR

PREPROCESS_FPS: int = 10
NUM_WORKERS: int    = 80

SUPPORTED_RESOLUTIONS: Dict[int, str] = {
    # 64: "npz",
    256: "mp4",
}

def _timestamp_to_str(timestamp_ms: int) -> str:
    total_ms = int(timestamp_ms / 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    seconds = (total_ms % 60000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def _read_frame_ffmpeg(video_path: Path, timestamp: str) -> Optional[torch.Tensor]:
    command = [
        "ffmpeg", "-ss", timestamp, "-i", str(video_path),
        "-frames:v", "1", "-q:v", "1", "-f", "image2pipe",
        "-vcodec", "bmp", "-"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image_bytes, _ = process.communicate()
    if not image_bytes:
        print(f"Warning: No frame extracted at {timestamp} from {video_path}")
        return None
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return torch.from_numpy(np.array(image))

def rescale_and_crop(
    video: torch.Tensor,
    resolution: int,
) -> np.ndarray:
    """
    Rescale and crop the video to the specified resolution. Used for preprocessing.
    Args:
        video (torch.Tensor): video tensor of shape (t, h, w, c). uint8.
        resolution (int): target resolution.
    Returns:
        np.ndarray: rescaled and cropped video tensor of shape (t, resolution, resolution, c). uint8.
    """
    *_, h, w, _ = video.shape
    scale_factor = max(resolution / h, resolution / w)
    h_scaled, w_scaled = round(h * scale_factor), round(w * scale_factor)
    assert h_scaled == resolution or w_scaled == resolution
    row = (h_scaled - resolution) // 2
    col = (w_scaled - resolution) // 2

    def _rescale_and_crop(image: torch.Tensor) -> torch.Tensor:
        image = Image.fromarray(image.numpy())
        image = image.resize((w_scaled, h_scaled), Image.Resampling.LANCZOS)
        return np.array(image)[row : row + resolution, col : col + resolution]

    video = np.stack([_rescale_and_crop(frame) for frame in video])
    return video

def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
) -> None:
    # from torchvision.io.video.write_video
    # av== 14.4.0
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    .. warning::

        In the near future, we intend to centralize PyTorch's video decoding
        capabilities within the `torchcodec
        <https://github.com/pytorch/torchcodec>`_ project. We encourage you to
        try it out and share your feedback, as the torchvision video decoders
        will eventually be deprecated.

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
    """
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy(force=True)

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = np.round(fps)

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = {}

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            # frame.pict_type = None
            frame.pict_type = 0  # PictureType.None
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

def process_single_clip(txt_file: Path, split: str):
    # 1. get url and timestamps from txt
    timestamps = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        youtube_url = lines[0].strip()
        video_id = youtube_url.split("=")[-1]
        for line in lines[1:]:
            ts = line.split(" ")[0]
            timestamps.append(_timestamp_to_str(int(ts)))

    # 2. find raw video
    video_path = RAW_VIDEO_DIR / split / f"{video_id}.mp4"
    if not video_path.exists():
        return f"Skip: {video_id} (mp4 not found)"

    # 3. extract frames
    frames = []
    for ts in timestamps:
        frame = _read_frame_ffmpeg(video_path, ts)
        if frame is not None:
            frames.append(frame)
        else:
            return f"Error: {video_id} (frame not found at {ts})"

    if not frames:
        return f"Error: {video_id} (no frames extracted)"
    
    video_tensor = torch.stack(frames) # (T, H, W, C)

    for resolution, preprocessing_type in SUPPORTED_RESOLUTIONS.items():
        
        processed_video = rescale_and_crop(video_tensor, resolution)

        output_dir = OUTPUT_ROOT_DIR / f"{split}_{resolution}"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{txt_file.stem}.{preprocessing_type}"

        if preprocessing_type == "npz":
            np.savez_compressed(str(save_path), video=processed_video.transpose(0, 3, 1, 2).copy())  # (T, C, H, W)
        elif preprocessing_type == "mp4":
            write_video(str(save_path), torch.from_numpy(processed_video).clone(), fps=PREPROCESS_FPS)
        else:
            return f"Error: Unsupported preprocessing type {preprocessing_type}"

    return f"Success: {txt_file.stem}"

def run_preprocessing(split: str, videoid_file: Optional[str] = None):
    split_path = RAW_TXT_DIR / split
    if not split_path.exists():
        print(f"[Error] Split path {split_path} does not exist. Skipping.")
        return
    
    txt_files = list((split_path).glob("*.txt"))
    print(f"Found {len(txt_files)} txt files for {split}")

    if videoid_file is not None:
        with open(videoid_file, 'r') as f:
            video_ids = set(line.strip() for line in f.readlines())
        new_txt_files = []
        for txt_file in tqdm(txt_files, desc=f"Filtering txt files for {split}"):
            youtube_url = ""
            with open(txt_file, 'r') as f:
                youtube_url = f.readline().strip()
            video_id = youtube_url.split("=")[-1]
            if video_id in video_ids:
                new_txt_files.append(txt_file)
        txt_files = new_txt_files
        print(f"After filtering, {len(txt_files)} txt files remain for {split}")

    worker_fn = partial(process_single_clip, split=split)

    results = []
    with Pool(NUM_WORKERS) as pool:
        # pbar = tqdm(pool.imap(worker_fn, txt_files), total=len(txt_files), desc=f"Processing {split}")
        results = list(tqdm(pool.imap(worker_fn, txt_files), total=len(txt_files)))

    success_count = sum(1 for r in results if r.startswith("Success"))
    skip_count = sum(1 for r in results if r.startswith("Skip"))
    error_count = len(txt_files) - success_count - skip_count

    print(f"\n--- {split} Finish ---")
    print(f"Total: {len(txt_files)} | Success: {success_count} | Skip: {skip_count} | Errors: {error_count}")
    
    if error_count > 0:
        print("All Errors:")
        errors = [r for r in results if r.startswith("Error")]
        for e in errors:
            print(f"  {e}")

def read_cameras(file_path: Path) -> torch.Tensor:
    cameras = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0, f"Empty file {file_path}"
        for idx, line in enumerate(lines):
            if idx > 0:
                timestamp, *camera = line.split(" ")
                # timestamps.append(_timestamp_to_str(int(timestamp)))
                cameras.append(np.fromstring(",".join(camera), sep=","))
        cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
    return cameras

def save_poses(split: str):
    split_path = RAW_TXT_DIR / split
    if not split_path.exists():
        print(f"[Error] Split path {split_path} does not exist. Skipping.")
        return
    pose_output_dir = OUTPUT_ROOT_DIR / f"{split}_poses"
    pose_output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list((split_path).glob("*.txt"))
    print(f"Found {len(txt_files)} txt files for {split}")

    for txt_file in tqdm(txt_files, desc=f"Saving poses for {split}"):
        cameras = read_cameras(txt_file)

        pose_save_path = pose_output_dir / f"{txt_file.stem}.pt"
        torch.save(cameras, pose_save_path)

if __name__ == "__main__":
    for split in ["training", "test"]:
        run_preprocessing(split)
    save_poses("train")