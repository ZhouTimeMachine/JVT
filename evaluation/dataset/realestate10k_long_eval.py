import json
import os
import os.path as osp
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.v2.functional as torchvisionF
from einops import rearrange
from PIL import Image, ImageFile
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RealEstate10KLongEvalVideoDataset(torch.utils.data.Dataset):

    video_suffix = "mp4"

    def __init__(
        self,
        Re10K_EVAL_VIDEO_DIR: str,
        Re10K_EVAL_POSE_DIR: str,
        seq_id_map: str,
        resolution: Union[Tuple[int, int], List[int]] = (256, 256),
        num_frames: int = 16,
        cache_file: Optional[str] = None,
        seq_file: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        assert len(resolution) == 2 and resolution[0] == resolution[1], "Only square resolution is supported."
        if resolution[0] not in [256, 252]:
            raise ValueError(f"Unsupported resolution {resolution}, expected 256 or 252.")

        self.Re10K_EVAL_VIDEO_DIR = Re10K_EVAL_VIDEO_DIR
        self.Re10K_EVAL_POSE_DIR = Re10K_EVAL_POSE_DIR
        self.num_frames = num_frames
        self.resolution = tuple(resolution)

        print(f"[Re10K-long-eval] Re10K_EVAL_VIDEO_DIR is {Re10K_EVAL_VIDEO_DIR}")
        print(f"[Re10K-long-eval] Re10K_EVAL_POSE_DIR is {Re10K_EVAL_POSE_DIR}")

        if cache_file is not None and osp.exists(cache_file):
            print(f"[Re10K-long-eval] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            cached_sequences = sorted(list(self.metadata.keys()))
        else:
            cached_sequences = self._discover_sequences(seq_file)
            self.metadata = self._build_metadata(cached_sequences)
            if cache_file is not None:
                os.makedirs(osp.dirname(cache_file), exist_ok=True)
                np.save(cache_file, self.metadata)

        with open(seq_id_map, "r", encoding="utf-8") as handle:
            self.seq_id_map = json.load(handle)
        if not isinstance(self.seq_id_map, list):
            raise ValueError(f"Expected list seq-id-map records in {seq_id_map}")
        self.seq_id_map = self._filter_seq_id_map(self.seq_id_map)
        self.sequence_list = [record["seq"] for record in self.seq_id_map]

        if max_samples is not None and max_samples > 0:
            self.seq_id_map = self.seq_id_map[:max_samples]
            self.sequence_list = [record["seq"] for record in self.seq_id_map]

        print(f"[Re10K-long-eval] Data size: {len(self)}, num_frames: {self.num_frames}")

    def _discover_sequences(self, seq_file: Optional[str]) -> List[str]:
        available = {
            videofile[: -(len(self.video_suffix) + 1)]
            for videofile in os.listdir(self.Re10K_EVAL_VIDEO_DIR)
            if videofile.endswith("." + self.video_suffix)
        }
        if seq_file is None:
            sequences = sorted(available)
            print(f"[Re10K-long-eval] Found {len(sequences)} sequences in {self.Re10K_EVAL_VIDEO_DIR}")
            return sequences

        with open(seq_file, "r", encoding="utf-8") as handle:
            requested = [line.strip() for line in handle if line.strip()]
        sequences = sorted(set(requested) & available)
        print(
            f"[Re10K-long-eval] Loaded {len(sequences)} available sequences "
            f"from {len(requested)} target sequences in {seq_file}"
        )
        return sequences

    def _build_metadata(self, sequence_list: List[str]):
        metadata = {}
        for seq in tqdm(sequence_list, desc="[Re10K-long-eval] Creating metadata"):
            anno_path = osp.join(self.Re10K_EVAL_POSE_DIR, f"{seq}.pt")
            if not osp.exists(anno_path):
                print(f"[Re10K-long-eval] pose file missing for {seq}, filtered.")
                continue
            cameras = torch.load(anno_path, map_location="cpu", weights_only=False)
            if len(cameras) < self.num_frames:
                continue

            intrinsics = torch.eye(3, dtype=cameras.dtype).repeat(len(cameras), 1, 1)
            intrinsics[:, 0, 0] = self.resolution[0] * cameras[:, 0]
            intrinsics[:, 1, 1] = self.resolution[1] * cameras[:, 1]
            intrinsics[:, 0, 2] = self.resolution[0] * cameras[:, 2]
            intrinsics[:, 1, 2] = self.resolution[1] * cameras[:, 3]

            extrinsics = torch.eye(4, dtype=cameras.dtype).repeat(len(cameras), 1, 1)
            extrinsics[:, :3, :] = rearrange(cameras[:, 6:], "f (h w) -> f h w", h=3, w=4)

            metadata[seq] = {
                "seq_length": len(cameras),
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "raw_cameras": torch.cat([cameras[:, :4], cameras[:, 6:]], dim=-1),
            }
        return metadata

    def _record_indices(self, record) -> torch.Tensor:
        if "frame_indices" in record:
            ids = torch.as_tensor(record["frame_indices"], dtype=torch.long)
        else:
            start_frame = int(record.get("start_frame", 0))
            frame_skip = int(record.get("frame_skip", 1))
            ids = torch.arange(self.num_frames, dtype=torch.long) * frame_skip + start_frame

        if ids.numel() != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frame indices, got {ids.numel()} for {record}")
        if ids.numel() > 1 and not bool((ids[1:] > ids[:-1]).all()):
            raise ValueError(f"frame_indices must be strictly increasing for {record.get('seq')}")
        return ids

    def _filter_seq_id_map(self, records):
        filtered = []
        for record in records:
            seq = record.get("seq")
            if seq not in self.metadata:
                continue
            ids = self._record_indices(record)
            seq_length = self.metadata[seq]["seq_length"]
            if int(ids[0]) < 0 or int(ids[-1]) >= seq_length:
                continue
            normalized = dict(record)
            normalized["seq"] = seq
            normalized["frame_indices"] = [int(i) for i in ids.tolist()]
            filtered.append(normalized)
        return filtered

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, index: int):
        record = self.seq_id_map[index]
        sequence_name = record["seq"]
        frame_indices = self._record_indices(record)
        return self.get_data(sequence_name=sequence_name, frame_indices=frame_indices)

    def get_data(self, sequence_name: str, frame_indices: torch.Tensor):
        metadata = self.metadata[sequence_name]
        if int(frame_indices[0]) < 0 or int(frame_indices[-1]) >= metadata["seq_length"]:
            raise ValueError(
                f"Invalid frame indices {frame_indices[0].item()}..{frame_indices[-1].item()} "
                f"for {sequence_name} with {metadata['seq_length']} frames."
            )

        extrinsics = metadata["extrinsics"][frame_indices]
        intrinsics = metadata["intrinsics"][frame_indices]
        raw_cameras = metadata["raw_cameras"][frame_indices]

        video_path = osp.join(self.Re10K_EVAL_VIDEO_DIR, f"{sequence_name}.{self.video_suffix}")
        video = self._load_mp4_frames(video_path, frame_indices)
        video = torchvisionF.center_crop(video, output_size=self.resolution[::-1])

        return {
            "seq_id": sequence_name,
            "seq_length": metadata["seq_length"],
            "ids": frame_indices,
            "video": video.to(torch.float32) / 255.0,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "raw_cameras": raw_cameras,
        }

    def _load_mp4_frames(self, video_path: str, frame_indices: torch.Tensor) -> torch.Tensor:
        decoder = VideoDecoder(video_path, dimension_order="NCHW")
        ids = frame_indices.cpu()
        if ids.numel() == 1:
            return decoder.get_frames_in_range(int(ids[0]), int(ids[0]) + 1, 1).data

        diffs = ids[1:] - ids[:-1]
        if bool((diffs == diffs[0]).all()):
            return decoder.get_frames_in_range(int(ids[0]), int(ids[-1]) + 1, int(diffs[0])).data

        clip = decoder.get_frames_in_range(int(ids[0]), int(ids[-1]) + 1, 1).data
        return clip[ids - ids[0]]
