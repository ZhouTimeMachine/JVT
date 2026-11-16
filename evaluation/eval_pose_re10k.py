import sys
import os
import numpy as np
import torch
import torch.distributed as dist
import hydra

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import List
from safetensors.torch import load_file as load_safetensors

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from evaluation.dataset.realestate10k_eval import RealEstate10KEvalVideoDataset
from evaluation.metrics.pi3.models.pi3x import Pi3X
from evaluation.metrics.pi3.utils.geometry import se3_inverse
from evaluation.metrics.relpose import se3_to_relative_pose_error, calculate_auc_np
from evaluation.logging_utils import get_filelogger, save_metric_to_csv, gather_logs, print_cfg
from utils.distributed_utils import init_distributed_mode, is_main_process


def main(cfg: DictConfig, overrides: List[str]):
    OmegaConf.set_struct(cfg, False)
    init_distributed_mode(cfg.infra)
    if is_main_process():
        print_cfg(cfg, overrides)

    infra_cfg: DictConfig = cfg.infra
    torch.manual_seed(cfg.seed + infra_cfg.rank)
    np.random.seed(cfg.seed + infra_cfg.rank)

    timestamp, logger = get_filelogger(cfg)  # makedirs inside it
    logger.info(f"Start loading Pi3X model...")
    model = Pi3X().to(infra_cfg.device).eval()
    weight = load_safetensors(cfg.pretrained_model_name_or_path, device="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(weight, strict=True)
    del weight
    model.requires_grad_(False)
    logger.info(
        f"Loaded Pi3X from {cfg.pretrained_model_name_or_path}. "
        f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
    )

    dataset = RealEstate10KEvalVideoDataset(**cfg.dataset)
    # Gather dictionaries after inference; shards need not have equal lengths.
    sampler = range(infra_cfg.rank, len(dataset), infra_cfg.world_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        shuffle=False, drop_last=False, persistent_workers=False,
        **cfg.dataloader
    )

    # 2. ready to read, and look up sampled ids from sequence name
    model.eval()

    # 3. prepare for metrics
    rError = {}
    tError = {}
    logger.info(f"Start evaluating re10k-test with Pi3X...")

    tbar = tqdm(dataloader, desc=f"[{cfg.name}]")
    for batch in tbar:
        with torch.no_grad():
            with torch.inference_mode():
                video = batch['video'].to(infra_cfg.device)  # [B, N, C, H, W], in [0, 1]

                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.amp.autocast(device_type=infra_cfg.device, dtype=dtype):
                    pred = model(video, with_prior=False)

                for seq, gt_extrs, pred_poses in zip(
                    batch['seq_id'],
                    batch["extrinsics"],
                    pred['camera_poses'].float().cpu()
                ):
                    # 7. compute metrics
                    pred_extrs = se3_inverse(pred_poses)  # c2w -> w2c
                    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                        pred_se3   = pred_extrs,
                        gt_se3     = gt_extrs,
                        num_frames = cfg.dataset.num_frames,
                    )

                    # 8. update metric for a sequence
                    message_str = (
                        "Seq=%s | rErr=%5.2f | tErr=%5.2f" % (
                            seq,
                            rel_rangle_deg.mean(),
                            rel_tangle_deg.mean()
                        )
                    )
                    tbar.set_postfix_str(message_str)
                    logger.info(message_str)

                    rError[seq] = rel_rangle_deg.cpu().numpy()
                    tError[seq] = rel_tangle_deg.cpu().numpy()

    if infra_cfg.world_size > 1:
        gather_r = [None] * infra_cfg.world_size
        gather_t = [None] * infra_cfg.world_size
        
        logger.info("Gathering results via torch.distributed...")
        dist.all_gather_object(gather_r, rError)
        dist.all_gather_object(gather_t, tError)

        rError = {}
        tError = {}
        for r_dict, t_dict in zip(gather_r, gather_t):
            if r_dict is None or t_dict is None:
                raise ValueError("Received None in gathered results, which indicates an issue in distributed communication.")
            rError.update(r_dict)
            tError.update(t_dict)

    if is_main_process():
        if set(rError) != set(dataset.sequence_list) or set(tError) != set(dataset.sequence_list):
            raise RuntimeError("Pose evaluation did not cover exactly the requested sequence IDs")
        rErrors = np.concatenate(list(rError.values()))
        tErrors = np.concatenate(list(tError.values()))
        if not np.isfinite(rErrors).all() or not np.isfinite(tErrors).all():
            raise ValueError("Pose metrics contain non-finite values")

        # 9. arrange all intermediate results to metrics
        metric_dict: dict = {"num_sequences": len(rError), "num_pairs": len(rErrors)}
        for threshold in cfg.metric_thresholds:
            metric_dict[f"Racc_{threshold}"] = np.mean(rErrors < threshold).item() * 100
            metric_dict[f"Tacc_{threshold}"] = np.mean(tErrors < threshold).item() * 100
            Auc, _ = calculate_auc_np(rErrors, tErrors, max_threshold=threshold)
            metric_dict[f"Auc_{threshold}"]  = Auc.item() * 100
        print(f"Average pose estimation metrics: {metric_dict}")

        # 9. save evaluation metrics to csv
        save_metric_to_csv(cfg, metric_dict)

        if infra_cfg.world_size > 1: gather_logs(cfg, timestamp)

    del model
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    overrides = sys.argv[1:]

    with hydra.initialize(version_base="1.2", config_path="../configs"):
        cfg = hydra.compose(config_name="eval", overrides=["evaluation=pose_re10k"] + overrides)

    main(cfg, overrides)
