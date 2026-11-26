from typing import Dict, Any, List, Tuple
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


def compute_wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score binomial confidence interval for empirical FPR.
    Args:
        successes: Number of false positives.
        total: Total normal validation samples.
        confidence: Confidence level (default: 0.95).
    Returns:
        (lower_bound, upper_bound)
    """
    if total <= 0:
        return 0.0, 1.0

    p_hat = successes / total
    # z-score for two-tailed normal distribution (1.95996 for 95%)
    z = 1.95996 if abs(confidence - 0.95) < 1e-4 else 2.57583  # 95% or 99%
    z_sq = z * z

    denom = 1.0 + z_sq / total
    center = (p_hat + z_sq / (2.0 * total)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) / total) + (z_sq / (4.0 * total * total)))) / denom

    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return lower, upper


def compute_threshold_sensitivity(
    val_scores: List[float] | np.ndarray,
    percentiles: List[float] = [95.0, 97.0, 99.0],
    margin_multiplier: float = 1.05,
) -> Dict[str, Any]:
    """
    Performs validation-only threshold sensitivity analysis across candidate percentiles.
    Strictly uses held-out NORMAL validation scores. Zero test defect samples or test labels used.

    Args:
        val_scores: List or numpy array of normal validation anomaly scores.
        percentiles: Percentiles to evaluate (default: [95.0, 97.0, 99.0]).
        margin_multiplier: Safety factor applied to raw threshold (default: 1.05).
    Returns:
        Structured dictionary containing sensitivity metrics, FPR confidence intervals,
        and statistical sample-size limitations.
    """
    scores = np.array(val_scores, dtype=np.float32)
    N = len(scores)
    if N == 0:
        raise ValueError("Cannot perform sensitivity analysis on empty validation scores.")

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    percentile_results = []
    for p in sorted(percentiles):
        raw_thresh = float(np.percentile(scores, p))
        margin_thresh = float(raw_thresh * margin_multiplier)

        fp_raw = int(np.sum(scores >= raw_thresh))
        fp_margin = int(np.sum(scores >= margin_thresh))

        fpr_raw = float(fp_raw / N)
        fpr_margin = float(fp_margin / N)

        ci_raw_low, ci_raw_high = compute_wilson_score_interval(fp_raw, N, confidence=0.95)
        ci_margin_low, ci_margin_high = compute_wilson_score_interval(fp_margin, N, confidence=0.95)

        percentile_results.append({
            "percentile": float(p),
            "raw_threshold": raw_thresh,
            "raw_validation_false_positives": fp_raw,
            "raw_empirical_fpr": fpr_raw,
            "raw_empirical_fpr_pct": round(fpr_raw * 100, 2),
            "raw_fpr_95_ci": [round(ci_raw_low, 4), round(ci_raw_high, 4)],
            "raw_fpr_95_ci_pct": [round(ci_raw_low * 100, 2), round(ci_raw_high * 100, 2)],
            "margin_threshold": margin_thresh,
            "margin_validation_false_positives": fp_margin,
            "margin_empirical_fpr": fpr_margin,
            "margin_empirical_fpr_pct": round(fpr_margin * 100, 2),
            "margin_fpr_95_ci": [round(ci_margin_low, 4), round(ci_margin_high, 4)],
            "margin_fpr_95_ci_pct": [round(ci_margin_low * 100, 2), round(ci_margin_high * 100, 2)],
        })

    # Statistical confidence bounds for small validation sets
    resolution_pct = round((1.0 / N) * 100, 4)
    rule_of_three_upper_pct = round((3.0 / N) * 100, 4)

    return {
        "metadata": {
            "analysis_type": "validation_only_threshold_sensitivity",
            "validation_sample_count": N,
            "margin_multiplier": margin_multiplier,
            "score_statistics": {
                "mean": mean_score,
                "std": std_score,
                "min": min_score,
                "max": max_score,
            },
            "integrity_guarantee": "Computed exclusively on held-out normal validation samples. Zero test defect samples or test labels used.",
        },
        "percentile_analysis": percentile_results,
        "sample_size_confidence_limitations": {
            "validation_sample_count": N,
            "single_sample_resolution_pct": resolution_pct,
            "rule_of_three_zero_fp_upper_bound_pct": rule_of_three_upper_pct,
            "statistical_limitations": [
                f"Small Sample Discrete Resolution: With N={N} validation samples, each observation represents {resolution_pct:.2f}% of the empirical distribution. Fractional FPRs between 0% and {resolution_pct:.2f}% cannot be resolved.",
                f"Upper Bound on Zero False Positives: Observing 0 false positives at 99% + margin only establishes with 95% confidence that the true normal FPR is below ~{rule_of_three_upper_pct:.2f}% (Rule of Three), not identically 0.0%.",
                "Distribution Shift Vulnerability: Normal variance under unobserved lighting, lens drift, or supplier batch variation can shift the empirical distribution and inflate operational FPR.",
                "Calibration vs Recall Disclaimer: This is a calibration analysis. Lowering thresholds increases theoretical sensitivity, but cannot be claimed to improve defect recall without an independent, separately documented test evaluation.",
            ],
        },
    }
