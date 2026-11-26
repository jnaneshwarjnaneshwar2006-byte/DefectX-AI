import os
import sys
import json
import argparse
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import setup_logger, load_config, get_device
from src.dataset import load_dataset_splits
from src.model import DefectXEngine
from src.thresholding import compute_threshold_sensitivity


def analyze_thresholds_cli(args):
    logger = setup_logger("defectx")
    logger.info("=== DEFECTX AI - Validation-Only Threshold Sensitivity Analysis ===")

    # 1. Verification of inputs and paths
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: '{checkpoint_path}'.")

    # Guard against accidental overwrite of main evaluation metrics
    output_file = os.path.abspath(args.output_file)
    blocked_filenames = ["evaluation_metrics.json", "test_predictions.csv", "roc_curves.png"]
    if os.path.basename(output_file) in blocked_filenames:
        raise ValueError(
            f"Output file '{output_file}' collides with official evaluation artifacts. "
            f"Please choose a distinct filename such as 'results/candle_threshold_sensitivity.json'."
        )

    config = load_config(args.config)
    if args.dataset_format:
        config.setdefault("dataset", {})["adapter"] = args.dataset_format

    prefer_cuda = (args.device.lower() != "cpu")
    device = get_device(prefer_cuda=prefer_cuda)

    dataset_root = args.dataset_root or config.get("dataset", {}).get("root_dir", "data/visa")
    category = args.category or config.get("dataset", {}).get("category", "candle")
    image_size = tuple(config.get("dataset", {}).get("image_size", [224, 224]))
    batch_size = config.get("dataset", {}).get("batch_size", 16)
    margin_multiplier = args.margin_multiplier

    # Parse percentiles
    if isinstance(args.percentiles, str):
        percentiles = [float(p.strip()) for p in args.percentiles.split(",")]
    else:
        percentiles = [float(p) for p in args.percentiles]

    # 2. Load held-out NORMAL validation dataset (ZERO test defect leakage)
    logger.info(f"Loading normal validation split for category '{category}' from '{dataset_root}'...")
    _, val_normal_ds, _ = load_dataset_splits(
        config=config,
        dataset_root=dataset_root,
        category=category,
        image_size=image_size,
    )
    val_count = len(val_normal_ds)
    logger.info(f"Loaded {val_count} normal validation images. Test dataset is strictly excluded.")

    val_loader = DataLoader(
        val_normal_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Initialize engine and load trained checkpoint
    engine = DefectXEngine(config=config, device=device)
    engine.load_checkpoint(checkpoint_path)

    # 4. Extract anomaly scores on normal validation samples
    logger.info("Computing anomaly scores on normal validation set...")
    val_img_scores: List[float] = []
    with torch.inference_mode():
        for batch_imgs, _ in val_loader:
            batch_imgs = batch_imgs.to(device)
            patches, _ = engine.feature_extractor(batch_imgs)
            _, img_scores = engine.memory_bank.compute_distances(patches)
            val_img_scores.extend(img_scores.cpu().tolist())

    # 5. Run sensitivity analysis
    analysis_report = compute_threshold_sensitivity(
        val_scores=val_img_scores,
        percentiles=percentiles,
        margin_multiplier=margin_multiplier,
    )

    # Add checkpoint and dataset metadata
    analysis_report["metadata"]["checkpoint_path"] = os.path.abspath(checkpoint_path)
    analysis_report["metadata"]["category"] = category
    analysis_report["metadata"]["dataset_root"] = os.path.abspath(dataset_root)
    analysis_report["metadata"]["device"] = str(device)

    # 6. Print human-readable report
    meta = analysis_report["metadata"]
    stats = meta["score_statistics"]

    print("\n" + "=" * 78)
    print("      DEFECTX AI - NORMAL-ONLY VALIDATION THRESHOLD SENSITIVITY REPORT")
    print("=" * 78)
    print(f" Target Checkpoint:      {meta['checkpoint_path']}")
    print(f" Category:               {category}")
    print(f" Validation Samples (N): {val_count} normal images (Zero test leakage)")
    print(f" Margin Multiplier:      {margin_multiplier:.2f}x")
    print(f" Score Statistics:       Mean = {stats['mean']:.4f} | Std = {stats['std']:.4f} | Min = {stats['min']:.4f} | Max = {stats['max']:.4f}")
    print("-" * 78)
    print(f" {'Pct':<6} {'Raw Thresh':<12} {'Raw FP':<9} {'Raw FPR':<10} {'Raw 95% CI':<14} {'Margin Thresh':<14} {'Margin FP':<10} {'Margin FPR':<10}")
    print("-" * 78)

    for item in analysis_report["percentile_analysis"]:
        p_str = f"{item['percentile']:.1f}%"
        raw_t = f"{item['raw_threshold']:.4f}"
        raw_fp = f"{item['raw_validation_false_positives']}/{val_count}"
        raw_fpr = f"{item['raw_empirical_fpr_pct']:.2f}%"
        raw_ci = f"[{item['raw_fpr_95_ci_pct'][0]:.1f}%, {item['raw_fpr_95_ci_pct'][1]:.1f}%]"

        m_t = f"{item['margin_threshold']:.4f}"
        m_fp = f"{item['margin_validation_false_positives']}/{val_count}"
        m_fpr = f"{item['margin_empirical_fpr_pct']:.2f}%"

        print(f" {p_str:<6} {raw_t:<12} {raw_fp:<9} {raw_fpr:<10} {raw_ci:<14} {m_t:<14} {m_fp:<10} {m_fpr:<10}")

    print("-" * 78)
    print(" STATISTICAL SAMPLE-SIZE CONFIDENCE LIMITATIONS:")
    lims = analysis_report["sample_size_confidence_limitations"]
    for i, lim in enumerate(lims["statistical_limitations"], 1):
        print(f"   [{i}] {lim}")
    print("=" * 78 + "\n")

    # 7. Save structured JSON artifact
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, indent=2)

    logger.info(f"Threshold sensitivity analysis successfully saved to: {output_file}")
    return analysis_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validation-only threshold sensitivity analysis for DEFECTX AI Engine."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/defectx_candle.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config")
    parser.add_argument("--category", type=str, default="candle", help="Target category (default: candle)")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root path of dataset")
    parser.add_argument("--dataset-format", type=str, default="visa", choices=["auto", "mvtec", "visa", "custom"], help="Dataset format")
    parser.add_argument("--percentiles", type=str, default="95.0,97.0,99.0", help="Comma-separated candidate percentiles")
    parser.add_argument("--margin-multiplier", type=float, default=1.05, help="Safety margin factor (default: 1.05)")
    parser.add_argument("--output-file", type=str, default="results/candle_threshold_sensitivity.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    args = parser.parse_args()

    analyze_thresholds_cli(args)
