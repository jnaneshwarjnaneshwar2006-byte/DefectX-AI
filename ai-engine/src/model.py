import os
import time
import uuid
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from .feature_extractor import PatchFeatureExtractor
from .memory_bank import MemoryBank
from .anomaly_map import AnomalyMapGenerator
from .thresholding import NormalOnlyThresholdSelector
from .severity import HeuristicSeverityEstimator
from .confidence import ConfidenceEvaluator
from .utils import get_device, setup_logger


class DefectXEngine:
    """
    Main DefectX Anomaly Detection Engine.
    Combines feature extraction, memory bank nearest-neighbor baseline,
    normal-only thresholding, heatmap generation, severity estimation, and confidence evaluation.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.logger = setup_logger("defectx")
        self.device = device or get_device(prefer_cuda=True)
        self.version = config.get("project", {}).get("version", "1.0.0")

        # Initialize sub-modules
        model_cfg = config.get("model", {})
        self.feature_extractor = PatchFeatureExtractor(
            backbone_name=model_cfg.get("backbone", "resnet18"),
            layers=model_cfg.get("layers", ["layer2", "layer3"]),
            patch_pooling_kernel=model_cfg.get("patch_pooling_kernel", 3),
            patch_pooling_stride=model_cfg.get("patch_pooling_stride", 1),
            patch_pooling_padding=model_cfg.get("patch_pooling_padding", 1),
            device=self.device,
        )

        mb_cfg = config.get("memory_bank", {})
        self.memory_bank = MemoryBank(
            coreset_sampling_ratio=mb_cfg.get("coreset_sampling_ratio", 0.1),
            subsample_method=mb_cfg.get("subsample_method", "approximate_greedy"),
            device=self.device,
            chunk_size=mb_cfg.get("chunk_size", 512),
            coreset_block_size=mb_cfg.get("coreset_block_size", 64),
            proj_dim=mb_cfg.get("proj_dim", 64),
        )

        hm_cfg = config.get("heatmap", {})
        self.heatmap_generator = AnomalyMapGenerator(
            gaussian_sigma=hm_cfg.get("gaussian_sigma", 4.0),
            overlay_alpha=hm_cfg.get("overlay_alpha", 0.45),
        )

        thresh_cfg = config.get("thresholding", {})
        self.threshold_selector = NormalOnlyThresholdSelector(
            method=thresh_cfg.get("method", "percentile"),
            percentile=thresh_cfg.get("percentile", 99.0),
            margin_multiplier=thresh_cfg.get("margin_multiplier", 1.05),
        )

        sev_cfg = config.get("severity", {})
        self.severity_estimator = HeuristicSeverityEstimator(
            area_threshold_low=sev_cfg.get("area_threshold_low", 0.005),
            area_threshold_medium=sev_cfg.get("area_threshold_medium", 0.02),
            area_threshold_high=sev_cfg.get("area_threshold_high", 0.06),
            intensity_weight=sev_cfg.get("intensity_weight", 0.5),
        )

        conf_cfg = config.get("confidence", {})
        self.confidence_evaluator = ConfidenceEvaluator(
            is_calibrated=conf_cfg.get("calibrated", False),
            temperature=conf_cfg.get("temperature", 1.0),
        )

        self.image_size = tuple(config.get("dataset", {}).get("image_size", [224, 224]))
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Fits the engine:
        1. Extracts normal patch embeddings from train_loader (train/good).
        2. Fits the memory bank (with coreset subsampling).
        3. Evaluates normal validation loader (val/good) to fit image and pixel thresholds.
        """
        self.logger.info("Extracting normal patch embeddings from training set...")
        all_train_patches = []
        grid_shape = None

        with torch.inference_mode():
            for batch_imgs, _ in train_loader:
                batch_imgs = batch_imgs.to(self.device)
                patches, grid_shape = self.feature_extractor(batch_imgs)
                # patches shape: (B, N_patches, C)
                B, N, C = patches.shape
                all_train_patches.append(patches.reshape(B * N, C).cpu())

        train_features = torch.cat(all_train_patches, dim=0)
        self.logger.info(
            f"Extracted {train_features.shape[0]} total patch embeddings of dim {train_features.shape[1]}. "
            f"Fitting memory bank with coreset ratio {self.memory_bank.coreset_sampling_ratio}..."
        )
        self.memory_bank.fit(train_features)
        self.logger.info(
            f"Memory bank fitted successfully with {self.memory_bank.features.shape[0]} coreset patches."
        )

        # Threshold fitting strictly on normal validation set
        self.logger.info("Fitting thresholds on held-out normal validation set...")
        val_img_scores: List[float] = []
        val_pix_maps: List[np.ndarray] = []

        with torch.inference_mode():
            for batch_imgs, _ in val_loader:
                batch_imgs = batch_imgs.to(self.device)
                patches, grid_shape = self.feature_extractor(batch_imgs)
                patch_scores, img_scores = self.memory_bank.compute_distances(patches)
                pix_maps = self.heatmap_generator.generate_pixel_maps(
                    patch_scores, grid_shape, target_size=self.image_size
                )
                val_img_scores.extend(img_scores.cpu().tolist())
                for m in pix_maps:
                    val_pix_maps.append(m)

        self.threshold_selector.fit(val_img_scores, val_pix_maps)
        self.logger.info(
            f"Thresholds fitted: Image Threshold = {self.threshold_selector.image_threshold:.4f}, "
            f"Pixel Threshold = {self.threshold_selector.pixel_threshold:.4f}"
        )

        return {
            "total_normal_patches": int(train_features.shape[0]),
            "coreset_patches": int(self.memory_bank.features.shape[0]),
            "image_threshold": self.threshold_selector.image_threshold,
            "pixel_threshold": self.threshold_selector.pixel_threshold,
            "val_stats": self.threshold_selector.normal_val_stats,
        }

    def predict(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_overlay: bool = True,
        inspection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stable prediction interface for the PHP backend and CLI:
        Validates input, extracts patch features, computes anomaly score and heatmap,
        estimates heuristic severity, evaluates confidence, and saves artifacts with
        unique inspection identifiers (guaranteeing no file overwrites).

        Returns JSON-compatible dictionary.
        """
        start_time = time.perf_counter()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: '{image_path}'")

        from .dataset_adapter import validate_image_file
        is_valid, err_msg, meta = validate_image_file(image_path)
        if not is_valid:
            raise ValueError(err_msg or f"Invalid image file: '{image_path}'")

        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Unable to decode image at '{image_path}': {str(e)}")

        orig_w, orig_h = pil_img.size

        # Preprocess
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            patches, grid_shape = self.feature_extractor(tensor)
            patch_scores, img_scores = self.memory_bank.compute_distances(patches)

            # Generate pixel anomaly map resized to original image dimensions
            pixel_map = self.heatmap_generator.generate_pixel_maps(
                patch_scores, grid_shape, target_size=(orig_h, orig_w)
            )[0]

        anomaly_score = float(img_scores[0].item())
        prediction = self.threshold_selector.classify_image(anomaly_score)
        is_anomaly = (prediction == "anomaly")

        # Severity estimation
        severity_info = self.severity_estimator.estimate(
            anomaly_map=pixel_map,
            pixel_threshold=self.threshold_selector.pixel_threshold,
            is_anomaly=is_anomaly,
        )

        # Confidence (returns None if uncalibrated)
        confidence = self.confidence_evaluator.get_confidence(
            anomaly_score=anomaly_score,
            threshold=self.threshold_selector.image_threshold,
        )

        # Output paths management: generate unique non-overwriting inspection ID and filenames
        if output_dir is None:
            output_dir = self.config.get("output", {}).get("heatmaps_dir", "results/heatmaps")
        os.makedirs(output_dir, exist_ok=True)

        if not inspection_id:
            inspection_id = f"insp_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Sanitize base_name for file safety
        safe_base_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)
        heatmap_filename = f"{inspection_id}_{safe_base_name}_heatmap.png"
        overlay_filename = f"{inspection_id}_{safe_base_name}_overlay.png"

        # Double check collision avoidance
        while os.path.exists(os.path.join(output_dir, heatmap_filename)):
            inspection_id = f"insp_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            heatmap_filename = f"{inspection_id}_{safe_base_name}_heatmap.png"
            overlay_filename = f"{inspection_id}_{safe_base_name}_overlay.png"

        heatmap_file = os.path.join(output_dir, heatmap_filename)
        overlay_file = os.path.join(output_dir, overlay_filename) if save_overlay else None

        self.heatmap_generator.save_heatmap_and_overlay(
            original_image_path=image_path,
            anomaly_map=pixel_map,
            heatmap_save_path=heatmap_file,
            overlay_save_path=overlay_file,
        )

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "inspection_id": inspection_id,
            "prediction": prediction,
            "anomaly_score": round(anomaly_score, 4),
            "severity": severity_info["level"],
            "confidence": confidence,
            "heatmap_path": os.path.abspath(heatmap_file),
            "overlay_path": os.path.abspath(overlay_file) if overlay_file else None,
            "model_version": self.version,
            "processing_time_ms": processing_time_ms,
            "shift_detected": None,  # Reserved for Member 4's distribution shift detector
            "details": {
                "image_path": os.path.abspath(image_path),
                "image_threshold": round(self.threshold_selector.image_threshold, 4),
                "pixel_threshold": round(self.threshold_selector.pixel_threshold, 4),
                "anomalous_area_ratio": severity_info["anomalous_area_ratio"],
                "severity_index": severity_info["severity_index"],
                "severity_disclaimer": severity_info["disclaimer"],
                "device": str(self.device),
            },
        }

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Saves memory bank features, thresholds, and configurations to a checkpoint file."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {
            "version": self.version,
            "config": self.config,
            "memory_bank": self.memory_bank.state_dict(),
            "threshold_selector": self.threshold_selector.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Model checkpoint saved to: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads and restores engine state from a checkpoint file."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.version = checkpoint.get("version", "1.0.0")
        self.memory_bank.load_state_dict(checkpoint["memory_bank"])
        self.threshold_selector.load_state_dict(checkpoint["threshold_selector"])
        self.logger.info(f"Loaded checkpoint from: {checkpoint_path} (Model version: {self.version})")
