from typing import Dict, Any, List
import numpy as np


class NormalOnlyThresholdSelector:
    """
    Selects operational anomaly thresholds strictly using held-out NORMAL validation images.
    ZERO leakage: Never sees or fits on test defect samples.

    Documented limitations:
    1. Operational FPR Trade-off: Setting a 99th percentile threshold on normal data
       targets a theoretical ~1% False Positive Rate on normal production images.
    2. Distribution Shift: If lighting, camera angle, or production batches drift,
       normal images may trigger false anomalies unless recalibrated.
    3. Open-Set Defect Magnitude: Because no defect samples are used during fitting,
       subtle micro-defects with anomaly scores below the normal variance cannot be guaranteed detectable.
    """
    def __init__(
        self,
        method: str = "percentile",
        percentile: float = 99.0,
        margin_multiplier: float = 1.05,
    ):
        self.method = method
        self.percentile = percentile
        self.margin_multiplier = margin_multiplier
        self.image_threshold: float = 0.0
        self.pixel_threshold: float = 0.0
        self.normal_val_stats: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, val_normal_image_scores: List[float], val_normal_pixel_maps: List[np.ndarray]) -> None:
        """
        Fits image and pixel thresholds using only normal validation samples.
        """
        img_scores = np.array(val_normal_image_scores, dtype=np.float32)
        if len(img_scores) == 0:
            raise ValueError("Cannot fit thresholds with empty validation scores.")

        # Aggregate pixel scores across validation maps (subsample if large)
        all_pixels = np.concatenate([m.ravel() for m in val_normal_pixel_maps])
        if len(all_pixels) > 100_000:
            rng = np.random.default_rng(42)
            all_pixels = rng.choice(all_pixels, size=100_000, replace=False)

        # Image threshold calculation
        if self.method == "percentile":
            raw_img_thresh = float(np.percentile(img_scores, self.percentile))
            raw_pix_thresh = float(np.percentile(all_pixels, self.percentile))
        elif self.method == "iqr":
            q75, q25 = np.percentile(img_scores, [75, 25])
            iqr = q75 - q25
            raw_img_thresh = float(q75 + 1.5 * iqr)

            pq75, pq25 = np.percentile(all_pixels, [75, 25])
            piqr = pq75 - pq25
            raw_pix_thresh = float(pq75 + 1.5 * piqr)
        elif self.method == "max_margin":
            raw_img_thresh = float(np.max(img_scores))
            raw_pix_thresh = float(np.max(all_pixels))
        else:
            raise ValueError(f"Unknown threshold method: {self.method}")

        self.image_threshold = raw_img_thresh * self.margin_multiplier
        self.pixel_threshold = raw_pix_thresh * self.margin_multiplier

        self.normal_val_stats = {
            "num_val_samples": int(len(img_scores)),
            "mean_image_score": float(np.mean(img_scores)),
            "std_image_score": float(np.std(img_scores)),
            "max_image_score": float(np.max(img_scores)),
            "min_image_score": float(np.min(img_scores)),
            "raw_image_threshold": raw_img_thresh,
            "fitted_image_threshold": self.image_threshold,
            "fitted_pixel_threshold": self.pixel_threshold,
            "method": self.method,
            "percentile": self.percentile,
            "margin_multiplier": self.margin_multiplier,
        }
        self.is_fitted = True

    def classify_image(self, anomaly_score: float) -> str:
        """Returns 'normal' or 'anomaly' based on fitted image threshold."""
        if not self.is_fitted:
            raise RuntimeError("ThresholdSelector has not been fitted yet.")
        return "anomaly" if anomaly_score >= self.image_threshold else "normal"

    def state_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "percentile": self.percentile,
            "margin_multiplier": self.margin_multiplier,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "normal_val_stats": self.normal_val_stats,
            "is_fitted": self.is_fitted,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.method = state["method"]
        self.percentile = state["percentile"]
        self.margin_multiplier = state["margin_multiplier"]
        self.image_threshold = float(state["image_threshold"])
        self.pixel_threshold = float(state["pixel_threshold"])
        self.normal_val_stats = state.get("normal_val_stats", {})
        self.is_fitted = state.get("is_fitted", True)
