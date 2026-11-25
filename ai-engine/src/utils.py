import os
import sys
import random
import logging
from typing import Optional, Dict, Any
import numpy as np
import torch
import yaml


def setup_logger(name: str = "defectx", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures structured logger for DEFECTX AI."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True, verbose: bool = True) -> torch.device:
    """
    Selects computation device (CUDA on RTX 4050 or CPU fallback).
    Never assumes a specific VRAM capacity, inspects dynamically.
    """
    logger = logging.getLogger("defectx")
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            if verbose:
                logger.info(f"Using GPU: {device_name} ({total_memory:.1f} MiB VRAM)")
        except Exception:
            if verbose:
                logger.info(f"Using GPU: {device_name}")
        return device
    else:
        if verbose:
            logger.info("Using device: CPU fallback")
        return torch.device("cpu")


def get_vram_info() -> Dict[str, float]:
    """Returns allocated and cached VRAM in MiB if CUDA is available."""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated(0) / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved(0) / (1024 ** 2),
            "max_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024 ** 2),
        }
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration file with error handling."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
