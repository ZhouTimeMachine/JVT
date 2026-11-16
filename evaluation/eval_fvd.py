"""
Unified FVD evaluation for flat generated-video directories.

This script evaluates generated `.mp4` videos against real reference clips
specified by a deterministic `seq-id-map` JSON file.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

import hydra
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from evaluation.dataset.fvd import FakeVideoDataset, RealVideoFVDDataset
from evaluation.logging_utils import get_filelogger, print_cfg, save_metric_to_csv
from evaluation.metrics.fvd import I3D, compute_fvd, compute_statistics


CACHE_DIR = Path("outputs/.cache/real_fvd_stats")


@torch.no_grad()
def extract_statistics(model, dataset, cfg: DictConfig, device: torch.device, desc: str) -> tuple[Tensor, Tensor]:
    """Extract I3D features in dataset order and return their statistics."""
    if len(dataset) < 2:
        raise ValueError("FVD requires at least two samples in each dataset")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    features = [model(batch.to(device)) for batch in tqdm(loader, desc=desc)]
    return compute_statistics(torch.cat(features))


def _real_cache_path(cfg: DictConfig) -> Path:
    map_path = Path(cfg.real_seq_id_map)
    identity = [
        str(map_path.resolve()),
        str(Path(cfg.real_video_dir).resolve()) if cfg.real_video_dir else None,
        cfg.num_frames,
        cfg.resolution,
    ]
    digest = hashlib.sha256(json.dumps(identity).encode() + b"\0" + map_path.read_bytes()).hexdigest()[:16]
    return CACHE_DIR / f"{map_path.stem}_{digest}.pt"


def load_or_compute_real_statistics(model, dataset, cfg: DictConfig, device: torch.device, logger) -> tuple[Tensor, Tensor]:
    """Load cached real statistics, or extract them and cache only CPU mu/sigma."""
    cache_path = _real_cache_path(cfg)
    if cache_path.is_file():
        logger.info(f"Loading cached real statistics from {cache_path}")
        stats = torch.load(cache_path, map_location=device, weights_only=True)
        return stats["mu"], stats["sigma"]

    logger.info(f"Extracting I3D statistics from {len(dataset)} real videos ...")
    mu, sigma = extract_statistics(model, dataset, cfg, device, "Real")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mu": mu.cpu(), "sigma": sigma.cpu()}, cache_path)
    logger.info(f"Cached real statistics to {cache_path}")
    return mu, sigma


def main(cfg: DictConfig, overrides):
    print_cfg(cfg, overrides)

    if not cfg.real_seq_id_map or not Path(cfg.real_seq_id_map).is_file():
        raise ValueError(f"real_seq_id_map {cfg.real_seq_id_map} doesn't exists")
    if not cfg.fake_video_dir or not Path(cfg.fake_video_dir).is_dir():
        raise ValueError(f"fake_video_dir {cfg.fake_video_dir} doesn't exists")

    fake_ds = FakeVideoDataset(cfg.fake_video_dir, num_frames=cfg.num_frames, resolution=cfg.resolution)
    real_ds = RealVideoFVDDataset(
        cfg.real_seq_id_map, real_video_dir=cfg.real_video_dir,
        num_frames=cfg.num_frames, resolution=cfg.resolution,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, logger = get_filelogger(cfg)

    logger.info("Loading I3D model ...")
    i3d_model = I3D(I3D_path=cfg.I3D_path, I3D_url=cfg.I3D_url).to(device).eval()

    logger.info(f"Extracting I3D statistics from {len(fake_ds)} fake videos ...")
    mu_fake, sigma_fake = extract_statistics(i3d_model, fake_ds, cfg, device, "Fake")
    mu_real, sigma_real = load_or_compute_real_statistics(i3d_model, real_ds, cfg, device, logger)

    num_fake_videos = len(fake_ds)
    num_real_videos = len(real_ds)
    logger.info(f"Computing FVD with {num_fake_videos} fake samples and {num_real_videos} real samples...")

    fvd_value = compute_fvd(mu_fake, sigma_fake, mu_real, sigma_real).item()
    logger.info(f"FVD: {fvd_value:.4f}")
    print(f"FVD: {fvd_value:.4f}")

    results = {
        "fvd": fvd_value,
        "num_fake_videos": num_fake_videos,
        "num_real_videos": num_real_videos,
        "real_seq_id_map": cfg.real_seq_id_map,
    }
    save_metric_to_csv(cfg, results)
    logger.info(f"Results saved: {results}")


if __name__ == "__main__":
    overrides = sys.argv[1:]
    with hydra.initialize(version_base="1.2", config_path="../configs"):
        cfg = hydra.compose(config_name="eval", overrides=["evaluation=fvd"] + overrides)
    main(cfg, overrides)
