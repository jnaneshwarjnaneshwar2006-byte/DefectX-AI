import unittest
import numpy as np
from src.severity import HeuristicSeverityEstimator
from src.confidence import ConfidenceEvaluator


class TestSeverityAndConfidence(unittest.TestCase):
    def test_severity_normal_image(self):
        estimator = HeuristicSeverityEstimator()
        dummy_map = np.zeros((100, 100), dtype=float)
        sev = estimator.estimate(dummy_map, pixel_threshold=1.5, is_anomaly=False)

        self.assertEqual(sev["level"], "NONE")
        self.assertEqual(sev["anomalous_pixel_count"], 0)
        self.assertIn("Heuristic visual estimate", sev["disclaimer"])

    def test_severity_levels(self):
        estimator = HeuristicSeverityEstimator(
            area_threshold_low=0.01,
            area_threshold_medium=0.05,
            area_threshold_high=0.15,
        )
        # 1. Low severity (<1% anomalous area, low intensity)
        map_low = np.zeros((100, 100), dtype=float)
        map_low[:5, :10] = 2.0  # 50 / 10000 = 0.5% (< 1.0%)
        sev_low = estimator.estimate(map_low, pixel_threshold=1.5, is_anomaly=True)
        self.assertEqual(sev_low["level"], "LOW")

        # 2. Critical severity (>15% anomalous area)
        map_crit = np.zeros((100, 100), dtype=float)
        map_crit[:45, :45] = 3.5  # 2025 / 10000 = 20.25%
        sev_crit = estimator.estimate(map_crit, pixel_threshold=1.5, is_anomaly=True)
        self.assertEqual(sev_crit["level"], "CRITICAL")

    def test_confidence_uncalibrated_returns_null(self):
        evaluator = ConfidenceEvaluator(is_calibrated=False)
        conf = evaluator.get_confidence(anomaly_score=2.8, threshold=2.0)
        # By contract: must return None (null in JSON) if uncalibrated
        self.assertIsNone(conf)

    def test_confidence_margin_fallback(self):
        evaluator = ConfidenceEvaluator(is_calibrated=False)
        margin = evaluator.get_confidence(
            anomaly_score=2.5,
            threshold=2.0,
            return_uncalibrated_margin=True
        )
        self.assertAlmostEqual(margin, 0.25, places=3)


if __name__ == "__main__":
    unittest.main()
