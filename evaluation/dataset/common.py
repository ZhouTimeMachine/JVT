import json
import os.path as osp
import torch

from typing import Optional, Set, Tuple, List, Dict, Any, Union


def load_candidate_sequences(seq_file: Optional[str]) -> Optional[Set[str]]:
    if seq_file is None or not osp.exists(seq_file):
        return None
    with open(seq_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


def load_seq_id_map(seq_id_map: Optional[str]):
    if seq_id_map is None or not osp.exists(seq_id_map):
        return None
    with open(seq_id_map, "r") as f:
        return json.load(f)


def resolve_sampling(seq_id_map, index: int) -> Tuple[int, int]:
    if seq_id_map is None:
        return 0, 1
    seq_info = seq_id_map[index]
    return seq_info["start_frame"], seq_info["frame_skip"]


def build_frame_ids(num_frames: int, start_frame: int, frame_skip: int) -> torch.Tensor:
    return torch.arange(num_frames, dtype=torch.long) * frame_skip + start_frame


def validate_sampling(
    seq_length: int,
    num_frames: int,
    start_frame: int,
    frame_skip: int,
    sequence_name: str,
) -> int:
    end_frame = start_frame + (num_frames - 1) * frame_skip
    if start_frame < 0 or frame_skip < 1 or end_frame >= seq_length:
        raise ValueError(
            f"Invalid start->end frame ({start_frame}->{end_frame}) with frame_skip {frame_skip}, "
            f"in sequence {sequence_name} with {seq_length} frames."
        )
    return end_frame


def filter_sequences_with_min_length(sequence_lengths: dict, min_length: int) -> List[str]:
    return sorted([
        name for name, length in sequence_lengths.items()
        if length >= min_length
    ])


class BaseEvalVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Union[Tuple[int, int], List[int]],
        num_frames: int,
        seq_file: Optional[str] = None,
        seq_id_map: Optional[str] = None,
    ):
        assert len(resolution) == 2 and resolution[0] == resolution[1], "Only support square resolution for now."
        self.resolution = resolution
        self.num_frames = num_frames
        self.metadata = self._scan_metadata()
        self.sequence_list = self._build_sequence_list(seq_file)
        self.seq_id_map = load_seq_id_map(seq_id_map)

    def _scan_metadata(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def _candidate_sequences(self) -> Set[str]:
        return set(self.metadata.keys())

    def _build_sequence_list(self, seq_file: Optional[str]) -> List[str]:
        candidate_sequences = self._candidate_sequences()
        seq_candidates_from_file = load_candidate_sequences(seq_file)
        if seq_candidates_from_file is not None:
            candidate_sequences &= seq_candidates_from_file

        sequence_lengths = {
            sequence_name: self.metadata[sequence_name]["seq_length"]
            for sequence_name in candidate_sequences
        }
        return filter_sequences_with_min_length(sequence_lengths, self.num_frames)

    def __len__(self) -> int:
        return len(self.sequence_list)

    def get_seq_length(self, index: Optional[int] = None, sequence_name: Optional[str] = None) -> int:
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return self.metadata[sequence_name]["seq_length"]

    def _load_video(
        self,
        seq_info: Dict[str, Any],
        start_frame: int,
        end_frame: int,
        frame_skip: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _extra_output_fields(self, seq_info: Dict[str, Any]) -> Dict[str, Any]:
        return {}

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

        seq_info = self.metadata[sequence_name]
        end_frame = validate_sampling(
            seq_length=seq_info["seq_length"],
            num_frames=self.num_frames,
            start_frame=start_frame,
            frame_skip=frame_skip,
            sequence_name=sequence_name,
        )
        ids = build_frame_ids(self.num_frames, start_frame, frame_skip)
        video = self._load_video(seq_info, start_frame, end_frame, frame_skip)

        return {
            "seq_id": sequence_name,
            "ids": ids,
            "video": video,
            **self._extra_output_fields(seq_info),
        }

    def __getitem__(self, index: int):
        start_frame, frame_skip = resolve_sampling(self.seq_id_map, index)
        return self.get_data(index=index, start_frame=start_frame, frame_skip=frame_skip)
