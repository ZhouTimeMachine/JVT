import os.path as osp
import os 
import numpy as np
import torch
import torchvision.transforms.v2.functional as torchvisionF
import random
import json

from torchcodec.decoders import VideoDecoder
from typing import Optional, Union, List, Tuple
from PIL import Image, ImageFile
from tqdm import tqdm
from einops import rearrange

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RealEstate10KEvalVideoDataset(torch.utils.data.Dataset):

    video_suffix = "mp4"

    def __init__(
        self,
        Re10K_EVAL_VIDEO_DIR: str,
        Re10K_EVAL_POSE_DIR: str,
        seq_id_map: str,
        resolution: Union[Tuple[int, int], List[int]] = (256, 256),  # (width, height)
        num_frames: int = 8,
        cache_file: Optional[str] = None,
        seq_file: Optional[str] = None,
        video_is_sampled_clip: bool = False,
    ):
        assert len(resolution) == 2 and resolution[0] == resolution[1], "Only support square resolution for now."
        self.Re10K_EVAL_VIDEO_DIR = Re10K_EVAL_VIDEO_DIR
        mapped_sequences = None

        with open(seq_id_map) as f:
            self.seq_id_map = json.load(f)
        if not self.seq_id_map:
            raise ValueError("seq_id_map must contain at least one sequence")
        mapped_sequences = [item["seq"] for item in self.seq_id_map]
        if len(set(mapped_sequences)) != len(mapped_sequences):
            raise ValueError("seq_id_map must contain unique sequence IDs")

        print(f"[Re10K-eval] Re10K_EVAL_VIDEO_DIR is {Re10K_EVAL_VIDEO_DIR}")
        print(f"[Re10K-eval] Re10K_EVAL_POSE_DIR is {Re10K_EVAL_POSE_DIR}")

        if cache_file is not None and osp.exists(cache_file):
            print(f"[Re10K-eval] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = mapped_sequences or sorted(self.metadata)
        else:
            if seq_file is not None:
                with open(seq_file, "r") as f:
                    seq_list = f.readlines()
                target_sequence_list = set(x.strip() for x in seq_list)
                avail_sequence_list = set([videofile[:-4] for videofile in os.listdir(Re10K_EVAL_VIDEO_DIR) if videofile.endswith('.' + self.video_suffix)])
                self.sequence_list = sorted(list(target_sequence_list & avail_sequence_list))
                print(f"[Re10K-eval] Loaded {len(self.sequence_list)} available sequences from {len(target_sequence_list)} target sequences in {seq_file}")
            else:
                self.sequence_list = sorted([videofile[:-4] for videofile in os.listdir(Re10K_EVAL_VIDEO_DIR) if videofile.endswith('.' + self.video_suffix)])
                print(f"[Re10K-eval] Found {len(self.sequence_list)} sequences in {Re10K_EVAL_VIDEO_DIR}")
            
            if mapped_sequences is not None:
                missing = set(mapped_sequences) - set(self.sequence_list)
                if missing:
                    raise FileNotFoundError(f"Mapped sequences are unavailable: {sorted(missing)[:10]}")
                self.sequence_list = mapped_sequences
            self.metadata = {}
            for seq in tqdm(self.sequence_list, desc=f"[Re10K-eval] Creating metadata..."):
                anno_path = osp.join(Re10K_EVAL_POSE_DIR, f"{seq}.pt")
                cameras = torch.load(anno_path, map_location="cpu", weights_only=False)

                if len(cameras) < num_frames:
                    if mapped_sequences is not None:
                        raise ValueError(f"Mapped sequence {seq} has only {len(cameras)} frames; need {num_frames}")
                    continue

                intrinsics = torch.eye(3, dtype=cameras.dtype).repeat(len(cameras), 1, 1)
                intrinsics[:, 0, 0] = resolution[0] * cameras[:, 0]  # fx
                intrinsics[:, 1, 1] = resolution[1] * cameras[:, 1]  # fy
                intrinsics[:, 0, 2] = resolution[0] * cameras[:, 2]  # cx
                intrinsics[:, 1, 2] = resolution[1] * cameras[:, 3]  # cy

                extrinsics = torch.eye(4, dtype=cameras.dtype).repeat(len(cameras), 1, 1)
                extrinsics[:, :3, :] = rearrange(cameras[:, 6:], "f (h w) -> f h w", h=3, w=4)
                
                seq_info = {
                    "seq_length": len(cameras),
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "raw_cameras": torch.cat([cameras[:, :4], cameras[:, 6:]], dim=-1),  # (N, 16)
                }

                self.metadata[seq] = seq_info

            self.sequence_list = mapped_sequences or sorted(self.metadata)
            if cache_file is not None:
                os.makedirs(osp.dirname(cache_file), exist_ok=True)
                np.save(cache_file, self.metadata)

        self.num_frames = num_frames
        self.resolution = resolution
        self.video_is_sampled_clip = video_is_sampled_clip
        
        missing_metadata = set(self.sequence_list) - self.metadata.keys()
        if missing_metadata:
            raise ValueError(f"Cached metadata is missing sequences: {sorted(missing_metadata)[:10]}")

        print(f"[Re10K-eval] Data size: {len(self)}, support num_frames: {self.num_frames}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_length(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return self.metadata[sequence_name]["seq_length"]

    def __getitem__(self, index: int):
        seq_info = self.seq_id_map[index]
        start_frame = seq_info['start_frame']
        frame_skip = seq_info['frame_skip']
        return self.get_data(index=index, start_frame=start_frame, frame_skip=frame_skip)

    def get_data(
            self,
            index: Optional[int] = None,
            sequence_name: Optional[str] = None,
            start_frame: int = 0,
            frame_skip: int = 1,
        ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]

        end_frame = start_frame + (self.num_frames - 1) * frame_skip
        if start_frame < 0 or frame_skip < 1 or end_frame >= metadata["seq_length"]:
            raise ValueError(f"Invalid start->end frame ({start_frame}->{end_frame}) with frame_skip {frame_skip}, in sequence {sequence_name} with {metadata['seq_length']} frames.")

        ids = torch.arange(self.num_frames) * frame_skip + start_frame
        extrinsics  = metadata["extrinsics"][ids]
        intrinsics  = metadata["intrinsics"][ids]
        raw_cameras = metadata["raw_cameras"][ids]

        if self.video_is_sampled_clip:
            video_start_frame = 0
            video_frame_skip = 1
            video_end_frame = self.num_frames - 1
        else:
            video_start_frame = start_frame
            video_frame_skip = frame_skip
            video_end_frame = end_frame

        video_path = osp.join(self.Re10K_EVAL_VIDEO_DIR, f"{sequence_name}.{self.video_suffix}")
        video_decoder = VideoDecoder(
            video_path,
            dimension_order="NCHW",
            num_ffmpeg_threads=1,
        )
        video = video_decoder.get_frames_in_range(
            video_start_frame,
            video_end_frame + 1,  # [start, stop)
            video_frame_skip
        )  # FrameBatch
        video = video.data
        video = torchvisionF.center_crop(video, output_size=self.resolution[::-1])  # (H, W)

        return {
            "seq_id": sequence_name, 
            "seq_length": metadata["seq_length"],
            "ids": ids,                             # Tensor [N], int64
            "video": video / 255,                   # Tensor [N, C, H, W], float32 in [0, 1]
            "extrinsics": extrinsics,               # Tensor [N, 4, 4], float32
            "intrinsics": intrinsics,               # Tensor [N, 3, 3], float32
            "raw_cameras": raw_cameras,             # Tensor [N, 16], float32
        }
