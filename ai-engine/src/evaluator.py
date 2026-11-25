import os
import json
import csv
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import DataLoader

from .model import DefectXEngine
from .utils import setup_logger


class DefectXEvaluator:
    """
    Evaluator for open-set industrial anomaly detection.
    Computes:
    - Image-level AUROC
    - Pixel-level AUROC (localization accuracy against ground-truth masks)
    - Precision, Recall, F1
    - False Positive Rate (FPR) on normal test images
    - Per-defect category breakdown
    - CSV/JSON reports and ROC plots
    """
    def __init__(self, engine: DefectXEngine):
        self.engine = engine
        self.logger = setup_logger("defectx")

    def evaluate(
        self,
        test_loader: DataLoader,
        output_dir: str = "results",
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info("Evaluating DefectX AI Engine on test set...")

        gt_image_labels: List[int] = []
        pred_image_scores: List[float] = []
        gt_pixel_masks_list: List[np.ndarray] = []
        pred_pixel_maps_list: List[np.ndarray] = []
        defect_types: List[str] = []
        image_paths: List[str] = []

        with torch.inference_mode():
            for batch in test_loader:
                images = batch["image"].to(self.engine.device)
                labels = batch["label"].tolist()
                masks = batch["mask"].numpy()  # (B, 1, H, W)
                d_types = batch["defect_type"]
                paths = batch["image_path"]

                patches, grid_shape = self.engine.feature_extractor(images)
                patch_scores, img_scores = self.engine.memory_bank.compute_distances(patches)
                pixel_maps = self.engine.heatmap_generator.generate_pixel_maps(
                    patch_scores, grid_shape, target_size=self.engine.image_size
                )

                gt_image_labels.extend(labels)
                pred_image_scores.extend(img_scores.cpu().tolist())
                defect_types.extend(d_types)
                image_paths.extend(paths)

                for i in range(len(labels)):
                    # Downsample masks to 112x112 or keep 224x224 for pixel AUROC to control memory
                    gt_pixel_masks_list.append(masks[i, 0].ravel())
                    pred_pixel_maps_list.append(pixel_maps[i].ravel())

        gt_image_np = np.array(gt_image_labels, dtype=int)
        pred_image_np = np.array(pred_image_scores, dtype=float)

        # 1. Image-level AUROC
        try:
            if len(np.unique(gt_image_np)) > 1:
                image_auroc = float(roc_auc_score(gt_image_np, pred_image_np))
            else:
                image_auroc = 1.0
        except Exception:
            image_auroc = float("nan")

        # 2. Pixel-level AUROC
        all_gt_pixels = np.concatenate(gt_pixel_masks_list)
        all_pred_pixels = np.concatenate(pred_pixel_maps_list)

        # Subsample pixels if count is huge (>500,000) for fast AUROC calculation
        if len(all_gt_pixels) > 500_000:
            rng = np.random.default_rng(42)
            sub_indices = rng.choice(len(all_gt_pixels), size=500_000, replace=False)
            eval_gt_pix = all_gt_pixels[sub_indices]
            eval_pred_pix = all_pred_pixels[sub_indices]
        else:
            eval_gt_pix = all_gt_pixels
            eval_pred_pix = all_pred_pixels

        try:
            if len(np.unique(eval_gt_pix)) > 1:
                pixel_auroc = float(roc_auc_score(eval_gt_pix, eval_pred_pix))
            else:
                pixel_auroc = 1.0
        except Exception:
            pixel_auroc = float("nan")

        # 3. Binary classification metrics at fitted threshold
        thresh = self.engine.threshold_selector.image_threshold
        pred_binary = (pred_image_np >= thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(gt_image_np, pred_binary, labels=[0, 1]).ravel()
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr_normal = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        accuracy = float((tp + tn) / len(gt_image_np)) if len(gt_image_np) > 0 else 0.0

        # 4. Per-defect breakdown
        defect_breakdown = {}
        unique_defects = sorted(list(set(defect_types)))
        for d in unique_defects:
            mask_d = np.array([t == d for t in defect_types])
            d_scores = pred_image_np[mask_d]
            d_binary = pred_binary[mask_d]
            defect_breakdown[d] = {
                "count": int(np.sum(mask_d)),
                "mean_anomaly_score": float(np.mean(d_scores)),
                "detected_count": int(np.sum(d_binary)),
                "detection_rate": float(np.mean(d_binary)),
            }

        metrics = {
            "image_auroc": round(image_auroc, 4),
            "pixel_auroc": round(pixel_auroc, 4),
            "image_threshold": round(thresh, 4),
            "pixel_threshold": round(self.engine.threshold_selector.pixel_threshold, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate_normal": round(fpr_normal, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "per_defect_breakdown": defect_breakdown,
            "total_test_samples": int(len(gt_image_labels)),
        }

        # Save JSON Report
        json_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Saved evaluation metrics to: {json_path}")

        # Save CSV Detailed Predictions
        csv_path = os.path.join(output_dir, "test_predictions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_path", "defect_type", "ground_truth", "anomaly_score",
                "predicted_label", "prediction"
            ])
            for i in range(len(image_paths)):
                writer.writerow([
                    image_paths[i],
                    defect_types[i],
                    gt_image_labels[i],
                    round(pred_image_scores[i], 4),
                    int(pred_binary[i]),
                    "anomaly" if pred_binary[i] == 1 else "normal"
                ])
        self.logger.info(f"Saved detailed predictions to: {csv_path}")

        # Generate and save ROC Curves Plot
        roc_plot_path = os.path.join(output_dir, "roc_curves.png")
        self._plot_roc_curves(
            gt_image_np, pred_image_np, image_auroc,
            eval_gt_pix, eval_pred_pix, pixel_auroc,
            roc_plot_path
        )

        return metrics

    def _plot_roc_curves(
        self,
        gt_img, pred_img, img_auc,
        gt_pix, pred_pix, pix_auc,
        save_path: str
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Image ROC
        if len(np.unique(gt_img)) > 1:
            fpr_img, tpr_img, _ = roc_curve(gt_img, pred_img)
            ax1.plot(fpr_img, tpr_img, color="blue", lw=2, label=f"Image ROC (AUC = {img_auc:.4f})")
        ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("Image-Level Anomaly Detection ROC")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Pixel ROC
        if len(np.unique(gt_pix)) > 1:
            fpr_pix, tpr_pix, _ = roc_curve(gt_pix, pred_pix)
            ax2.plot(fpr_pix, tpr_pix, color="red", lw=2, label=f"Pixel ROC (AUC = {pix_auc:.4f})")
        ax2.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("Pixel-Level Localization ROC")
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"Saved ROC curves plot to: {save_path}")
