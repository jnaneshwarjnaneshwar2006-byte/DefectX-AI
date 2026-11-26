import os
import argparse
import json
import torch
from torch.utils.data import DataLoader

from src.utils import setup_logger, load_config, get_device
from src.dataset import load_dataset_splits
from src.model import DefectXEngine
from src.evaluator import DefectXEvaluator


def evaluate(args):
    logger = setup_logger("defectx")
    logger.info("=== DEFECTX AI - Starting Model Evaluation ===")

    config = load_config(args.config)
    if getattr(args, "dataset_format", None):
        config.setdefault("dataset", {})["adapter"] = args.dataset_format
    prefer_cuda = (args.device.lower() != "cpu")
    device = get_device(prefer_cuda=prefer_cuda)

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: '{checkpoint_path}'. Train model first.")

    dataset_root = args.dataset_root or config.get("dataset", {}).get("root_dir", "data/mvtec")
    category = args.category or config.get("dataset", {}).get("category", "bottle")
    image_size = tuple(config.get("dataset", {}).get("image_size", [224, 224]))
    batch_size = config.get("dataset", {}).get("batch_size", 16)

    # Load test split (never leaked into training or thresholding)
    _, _, test_ds = load_dataset_splits(
        config=config,
        dataset_root=dataset_root,
        category=category,
        image_size=image_size,
    )
    logger.info(f"Loaded {len(test_ds)} test images (good + held-out defect categories).")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize engine and load trained checkpoint
    engine = DefectXEngine(config=config, device=device)
    engine.load_checkpoint(checkpoint_path)

    # Run evaluation
    output_dir = args.output_dir or config.get("output", {}).get("results_dir", "results")
    evaluator = DefectXEvaluator(engine)
    metrics = evaluator.evaluate(test_loader, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("           DEFECTX AI - EVALUATION REPORT")
    print("=" * 60)
    print(f" Category:                   {category}")
    print(f" Total Test Samples:         {metrics['total_test_samples']}")
    print(f" Image AUROC:                {metrics['image_auroc']:.4f}")
    print(f" Pixel AUROC:                {metrics['pixel_auroc']:.4f}")
    print(f" Image Threshold (Normal-Fit): {metrics['image_threshold']:.4f}")
    print(f" Pixel Threshold:            {metrics['pixel_threshold']:.4f}")
    print(f" Accuracy:                   {metrics['accuracy'] * 100:.2f}%")
    print(f" Precision:                  {metrics['precision']:.4f}")
    print(f" Recall:                     {metrics['recall']:.4f}")
    print(f" F1 Score:                   {metrics['f1_score']:.4f}")
    print(f" Normal False Positive Rate: {metrics['false_positive_rate_normal'] * 100:.2f}%")
    print("-" * 60)
    print(" Confusion Matrix:")
    print(f"   TN (Good as Good):        {metrics['confusion_matrix']['true_negatives']}")
    print(f"   FP (Good as Defect):      {metrics['confusion_matrix']['false_positives']}")
    print(f"   FN (Defect as Good):      {metrics['confusion_matrix']['false_negatives']}")
    print(f"   TP (Defect as Defect):    {metrics['confusion_matrix']['true_positives']}")
    print("-" * 60)
    print(" Per-Defect Detection Rates:")
    for d_name, d_stats in metrics["per_defect_breakdown"].items():
        print(f"   - {d_name:<16}: {d_stats['detected_count']}/{d_stats['count']} "
              f"({d_stats['detection_rate'] * 100:.1f}%) | Mean Score: {d_stats['mean_anomaly_score']:.4f}")
    print("=" * 60)
    print(f" Reports & Plots saved to:   {os.path.abspath(output_dir)}")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DefectX AI Engine on held-out test defects.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/defectx_bottle.pt", help="Path to checkpoint")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root directory of dataset")
    parser.add_argument("--category", type=str, default=None, help="Category name")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for reports and plots")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--dataset-format", type=str, default=None, choices=["auto", "mvtec", "visa", "custom", "folder"], help="Dataset format/adapter")
    args = parser.parse_args()
    evaluate(args)
