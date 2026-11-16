import json
import argparse
import random
from pathlib import Path

from typing import Optional, Any
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def set_default(value: Optional[Any], default_value: Any):
    return value if value is not None else default_value

def load_dataset_config(config_name: str) -> DictConfig:
    default_cfg = OmegaConf.load(root / "configs/dataset/default.yaml")
    extra_cfg = OmegaConf.load(root / config_name)
    extra_cfg.pop("defaults")
    # Keep the dataset namespace used by ${dataset.name/resolution} interpolations.
    return OmegaConf.create({"dataset": OmegaConf.merge(default_cfg, extra_cfg)}).dataset

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("--num_seqs", "--num_samples", type=int, default=None)
    parser.add_argument("--frame_skip", "--frame-skip", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--real_video_dir", type=Path, default=None, help="UCF101 video directory")
    parser.add_argument("--num_frames", type=int, default=None, help="UCF101 sequence length")
    args = parser.parse_args(argv)

    match args.dataset_name:
        case "ffs":
            config_name: str      = "configs/dataset/faceforensics.yaml"
            data_cfg = load_dataset_config(config_name)
            if args.frame_skip is not None:
                data_cfg.frame_skip = args.frame_skip

            from dataset.uncond import FaceForensicsVideoDataset
            dataset_train = FaceForensicsVideoDataset(cfg=data_cfg, current_epoch=None)
            num_seqs      = set_default(args.num_seqs, 2048)
        case "sky":
            config_name: str      = "configs/dataset/skytimelapse.yaml"
            data_cfg = load_dataset_config(config_name)
            data_cfg.duplicates = 1
            if args.frame_skip is not None:
                data_cfg.frame_skip = args.frame_skip

            from dataset.uncond import SkyTimelapseVideoDataset
            dataset_train = SkyTimelapseVideoDataset(cfg=data_cfg, current_epoch=None)
            # The reference map used Python's RNG at epoch 1,000,000, not NumPy's.
            clip_rng = random.Random(42 + 1000000)
            dataset_train.idx_remap = [
                clip_rng.randrange(dataset_train.video_length(meta) - dataset_train.n_frames + 1)
                for meta in dataset_train.metadata
            ]
            num_seqs      = set_default(args.num_seqs, 2048)
        case "taichi":
            config_name: str      = "configs/dataset/taichi_hd.yaml"
            data_cfg = load_dataset_config(config_name)
            if args.frame_skip is not None:
                data_cfg.frame_skip = args.frame_skip

            from dataset.uncond import TaichiHDVideoDataset
            dataset_train = TaichiHDVideoDataset(cfg=data_cfg, current_epoch=None)
            num_seqs      = set_default(args.num_seqs, 2048)
        case "ucf101":
            config_name: str      = "configs/dataset/ucf101.yaml"
            data_cfg = load_dataset_config(config_name)
            if args.frame_skip is not None:
                data_cfg.frame_skip = args.frame_skip
            if args.num_frames is not None:
                data_cfg.max_frames = args.num_frames

            from torchcodec.decoders import VideoDecoder
            # Keep the reference UCF101 map's 48-frame span (16 * 3).
            n_frames = data_cfg.max_frames * data_cfg.frame_skip
            video_dir = set_default(args.real_video_dir, Path(data_cfg.save_dir))
            dataset_train = []
            for video_path in tqdm(sorted(video_dir.glob("**/*.avi")), desc="Scanning UCF101 videos"):
                try:
                    video_length = VideoDecoder(str(video_path)).metadata.num_frames
                except Exception:
                    continue
                if video_length >= n_frames:
                    dataset_train.append((str(video_path), video_length))
            num_seqs = set_default(args.num_seqs, 2048)
        case _:
            raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    rng = random.Random(args.seed)
    if len(dataset_train) < num_seqs:
        raise ValueError(f"Not enough videos ({len(dataset_train)}) to sample {num_seqs} sequences.")
    else:
        print(f"Sampling {num_seqs} sequences from {len(dataset_train)} available clips with seed {args.seed}...")
    clip_indices = rng.sample(range(len(dataset_train)), num_seqs)

    seq_id_map = []
    for idx, clip_idx in tqdm(enumerate(clip_indices), total=len(clip_indices), desc=f"Generating eval sequences for {args.dataset_name}"):
        if args.dataset_name == "ucf101":
            video_path, video_length = dataset_train[clip_idx]
            start_frame = rng.randint(0, video_length - n_frames)
        else:
            video_idx, start_frame = dataset_train.get_clip_location(clip_idx)
            video_metadata = dataset_train.metadata[video_idx]
            video_path = str(video_metadata["video_paths"])
            video_length = dataset_train.video_length(video_metadata)
            n_frames = dataset_train.n_frames

        seq_id_map.append({
            "idx": idx,
            "video_path": video_path,
            "start_frame": start_frame,
            "end_frame": min(start_frame + n_frames, video_length),
            "frame_skip": data_cfg.frame_skip,
        })

    stride_suffix = f"_stride{data_cfg.frame_skip}" if args.frame_skip is not None or args.dataset_name == "ucf101" else ""
    split = "" if args.dataset_name == "ucf101" else "train"
    output_path = args.output_path or Path(
        f"evaluation/dataset/seq-id-maps/{args.dataset_name}_{split}{num_seqs}_"
        f"{data_cfg.resolution}x{data_cfg.resolution}_seq{data_cfg.max_frames}"
        f"{stride_suffix}_seed{args.seed}_id_map.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(seq_id_map, f, indent=4)
    print(f"Saved seq-id map to {output_path}")


if __name__ == "__main__":
    main()
