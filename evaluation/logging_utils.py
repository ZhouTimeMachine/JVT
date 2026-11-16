import os
import logging
import time
import pandas as pd

from typing import List, Tuple
from omegaconf import DictConfig, OmegaConf


def print_cfg(cfg: DictConfig, overrides: List[str]):
    print("cfg overrides:", overrides)
    print("-" * 20 + " Configuration " + "-" * 20)
    print(OmegaConf.to_yaml(cfg))
    print("-" * 55)


def get_filelogger(cfg: DictConfig) -> Tuple[str, logging.Logger]:
    logger = logging.getLogger(cfg.name)
    # log to local directory
    logger.setLevel(logging.INFO)

    os.makedirs(cfg.output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    if cfg.infra.world_size > 1:
        log_file = os.path.join(cfg.output_dir, f"{timestamp}-{cfg.name}-rank{cfg.infra.rank}.log")
    else:
        log_file = os.path.join(cfg.output_dir, f"{timestamp}-{cfg.name}.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    
    formatter = logging.Formatter(
        fmt='[%(asctime)s %(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return timestamp, logger


def gather_logs(cfg: DictConfig, timestamp: str):
    gathered_log_file = os.path.join(cfg.output_dir, f"{timestamp}-{cfg.name}.log")
    gathered_lines = []
    for rk in range(cfg.infra.world_size):
        log_file = os.path.join(cfg.output_dir, f"{timestamp}-{cfg.name}-rank{rk}.log")
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found: {log_file}")
            continue

        with open(log_file, 'r', encoding='utf-8') as f:
            gathered_lines.extend(line for line in f if "Seq=" in line)

    # remove duplicate lines (according to sequence name), then sort
    get_seqname = lambda line: line.split("Seq=")[1].split(" ")[0]
    unique_lines = {get_seqname(line): line for line in gathered_lines}
    sorted_lines = sorted(unique_lines.values(), key=get_seqname)

    with open(gathered_log_file, 'w', encoding='utf-8') as f:
        f.writelines(sorted_lines)
    
    # optionally remove individual rank logs
    for rk in range(cfg.infra.world_size):
        log_file = os.path.join(cfg.output_dir, f"{timestamp}-{cfg.name}-rank{rk}.log")
        if os.path.exists(log_file):
            os.remove(log_file)


def write_csv(file_path: str, data_dict: dict):
    # transform data of one row to DataFrame
    new_row = pd.DataFrame([data_dict])
    
    # directly save when the file does not exist; else we just append
    if not os.path.isfile(file_path):
        new_row.to_csv(file_path, index=False)
    else:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        updated_data.to_csv(file_path, index=False)


def save_metric_to_csv(cfg: DictConfig, metric_dict: dict):
    os.makedirs(cfg.output_dir, exist_ok=True)
    statistics_file = os.path.join(cfg.output_dir, f"metric")  # + ".csv"
    if getattr(cfg, "save_suffix", None) is not None:
        statistics_file += f"-{cfg.save_suffix}"
    statistics_file += ".csv"

    addi_keys_cfg: DictConfig = cfg.addi_keys
    addi_keys = {
        key: value
        for key, value in OmegaConf.to_container(addi_keys_cfg, resolve=True).items()
        if value is not None
    }
    metric_dict = {**addi_keys, **metric_dict}
    write_csv(statistics_file, metric_dict)
