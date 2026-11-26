import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.utils import setup_logger, set_seed, get_device, load_config, get_vram_info
from src.dataset import load_dataset_splits
from src.model import DefectXEngine


def train(args):
    logger = setup_logger("defectx")
    logger.info("=== DEFECTX AI - Starting Model Training ===")

    # 1. Load configuration
    config = load_config(args.config)
    seed = config.get("dataset", {}).get("seed", 42)
    set_seed(seed)

    if getattr(args, "dataset_format", None):
        config.setdefault("dataset", {})["adapter"] = args.dataset_format

    # Device selection (CUDA on RTX 4050 with automatic fallback)
    prefer_cuda = (args.device.lower() != "cpu")
    device = get_device(prefer_cuda=prefer_cuda)

    # 2. Dataset loaders
    dataset_root = args.dataset_root or config.get("dataset", {}).get("root_dir", "data/mvtec")
    category = args.category or config.get("dataset", {}).get("category", "bottle")
    image_size = tuple(config.get("dataset", {}).get("image_size", [224, 224]))
    val_ratio = config.get("dataset", {}).get("val_ratio", 0.2)
    batch_size = config.get("dataset", {}).get("batch_size", 16)
    num_workers = config.get("dataset", {}).get("num_workers", 0)

    logger.info(f"Loading dataset splits for category '{category}' from '{dataset_root}'...")
    train_normal_ds, val_normal_ds, _ = load_dataset_splits(
        config=config,
        dataset_root=dataset_root,
        category=category,
        val_ratio=val_ratio,
        seed=seed,
        image_size=image_size,
    )
    logger.info(
        f"Split sizes: Train Normal = {len(train_normal_ds)} samples, "
        f"Val Normal = {len(val_normal_ds)} samples (held-out for thresholding)"
    )

    train_loader = DataLoader(
        train_normal_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_normal_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Initialize and fit DefectX Engine
    engine = DefectXEngine(config=config, device=device)
    fit_metrics = engine.fit(train_loader, val_loader)

    # 4. Save checkpoint
    checkpoint_dir = args.checkpoint_dir or config.get("output", {}).get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"defectx_{category}.pt")
    engine.save_checkpoint(checkpoint_path)

    vram = get_vram_info()
    logger.info(f"Training completed successfully! VRAM Peak: {vram['max_allocated_mb']:.1f} MiB")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")

    return checkpoint_path, fit_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DefectX Industrial Anomaly Detection Engine.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root directory of dataset")
    parser.add_argument("--category", type=str, default=None, help="Target category (default: bottle)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Computation device")
    parser.add_argument("--dataset-format", type=str, default=None, choices=["auto", "mvtec", "visa", "custom", "folder"], help="Dataset format/adapter")
    args = parser.parse_args()
    train(args)
