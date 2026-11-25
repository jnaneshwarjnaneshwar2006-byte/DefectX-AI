from typing import Optional, Dict, Any, List
import numpy as np


class ConfidenceEvaluator:
    """
    Defensible Confidence Calculator and Evaluation-Ready Interface for Member 4.

    GROUND RULE: Nearest-neighbor anomaly distance is NOT a calibrated probability.
    Equating raw Euclidean distance or an uncalibrated sigmoid to confidence is statistically invalid.

    Default Behavior:
    - If calibration has not been formally verified on validation data: returns None (null in JSON).
    - Exposes a plug-in interface for Member 4 to integrate Platt scaling, isotonic regression,
      or conformal prediction intervals.
    """
    def __init__(self, is_calibrated: bool = False, temperature: float = 1.0):
        self.is_calibrated = is_calibrated
        self.temperature = temperature
        self.calibrator_model: Optional[Any] = None
        self.calibration_metadata: Dict[str, Any] = {
            "status": "uncalibrated",
            "reason": (
                "Anomaly distance is not a probability. Calibration validation is pending Member 4 integration."
            ),
        }

    def compute_uncalibrated_margin(self, anomaly_score: float, threshold: float) -> float:
        """
        Returns normalized decision margin relative to threshold:
        margin > 0 indicates anomaly region; margin < 0 indicates normal region.
        """
        if threshold <= 0:
            return 0.0
        return float((anomaly_score - threshold) / threshold)

    def get_confidence(
        self,
        anomaly_score: float,
        threshold: float,
        return_uncalibrated_margin: bool = False,
    ) -> Optional[float]:
        """
        Returns confidence score:
        - None (null) if calibration is unvalidated (default production behavior).
        - Calibrated probability if is_calibrated is True.
        - Uncalibrated margin if explicitly requested via parameter.
        """
        if not self.is_calibrated:
            if return_uncalibrated_margin:
                return round(self.compute_uncalibrated_margin(anomaly_score, threshold), 4)
            return None

        # If Member 4 has calibrated the model
        if self.calibrator_model is not None:
            try:
                # Sklearn isotonic / logistic calibrator
                prob = self.calibrator_model.predict_proba([[anomaly_score]])[0, 1]
                return float(round(prob, 4))
            except Exception:
                pass

        return None

    def register_calibrator(self, calibrator_model: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Interface for Member 4 to register a validated calibration model
        (e.g., Platt Scaler, Isotonic Regression, Conformal Predictor).
        """
        self.calibrator_model = calibrator_model
        self.is_calibrated = True
        self.calibration_metadata = metadata or {
            "status": "calibrated",
            "method": str(type(calibrator_model)),
        }

    def evaluate_calibration_quality(
        self,
        scores: List[float],
        true_labels: List[int],
        num_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluation utility for Member 4 to compute Expected Calibration Error (ECE) and Brier Score.
        """
        if not self.is_calibrated or self.calibrator_model is None:
            return {
                "ece": float("nan"),
                "brier_score": float("nan"),
                "status": "uncalibrated",
            }

        scores_arr = np.array(scores).reshape(-1, 1)
        probs = self.calibrator_model.predict_proba(scores_arr)[:, 1]
        labels = np.array(true_labels)

        # Brier score
        brier = float(np.mean((probs - labels) ** 2))

        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        for i in range(num_bins):
            in_bin = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if np.any(in_bin):
                bin_acc = np.mean(labels[in_bin])
                bin_conf = np.mean(probs[in_bin])
                bin_weight = np.sum(in_bin) / len(probs)
                ece += bin_weight * np.abs(bin_acc - bin_conf)

        return {
            "ece": round(float(ece), 4),
            "brier_score": round(brier, 4),
            "status": "calibrated",
        }
