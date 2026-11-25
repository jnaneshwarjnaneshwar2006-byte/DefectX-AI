from typing import Dict, Any
import numpy as np


class HeuristicSeverityEstimator:
    """
    Estimates anomaly severity based on anomalous area ratio and anomaly intensity.

    DISCLAIMER: This is an explicitly heuristic prototype based solely on visual
    feature deviation. It is NOT a verified physical damage, structural integrity,
    or metallurgical measurement.
    """
    DISCLAIMER = (
        "Heuristic visual estimate prototype only. "
        "Not a verified physical or structural damage measurement."
    )

    def __init__(
        self,
        area_threshold_low: float = 0.005,    # 0.5% of total pixels
        area_threshold_medium: float = 0.02,  # 2.0% of total pixels
        area_threshold_high: float = 0.06,    # 6.0% of total pixels
        intensity_weight: float = 0.5,
    ):
        self.area_threshold_low = area_threshold_low
        self.area_threshold_medium = area_threshold_medium
        self.area_threshold_high = area_threshold_high
        self.intensity_weight = intensity_weight

    def estimate(
        self,
        anomaly_map: np.ndarray,
        pixel_threshold: float,
        is_anomaly: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimates severity level from the 2D anomaly map and pixel threshold.
        """
        if not is_anomaly:
            return {
                "level": "NONE",
                "anomalous_pixel_count": 0,
                "anomalous_area_ratio": 0.0,
                "severity_index": 0.0,
                "disclaimer": self.DISCLAIMER,
            }

        total_pixels = anomaly_map.size
        anomalous_mask = (anomaly_map >= pixel_threshold)
        anomalous_pixel_count = int(np.sum(anomalous_mask))
        anomalous_area_ratio = float(anomalous_pixel_count / max(1, total_pixels))

        if anomalous_pixel_count > 0:
            peak_val = float(np.max(anomaly_map[anomalous_mask]))
            intensity_ratio = max(0.0, (peak_val - pixel_threshold) / max(1e-6, pixel_threshold))
        else:
            intensity_ratio = 0.0

        # Blended heuristic score combining area and peak intensity
        severity_index = float(
            anomalous_area_ratio * (1.0 + self.intensity_weight * min(2.0, intensity_ratio))
        )

        if severity_index < self.area_threshold_low:
            level = "LOW"
        elif severity_index < self.area_threshold_medium:
            level = "MEDIUM"
        elif severity_index < self.area_threshold_high:
            level = "HIGH"
        else:
            level = "CRITICAL"

        return {
            "level": level,
            "anomalous_pixel_count": anomalous_pixel_count,
            "anomalous_area_ratio": round(anomalous_area_ratio, 6),
            "severity_index": round(severity_index, 6),
            "disclaimer": self.DISCLAIMER,
        }
