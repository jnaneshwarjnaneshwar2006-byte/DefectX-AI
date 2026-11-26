import unittest
import numpy as np

from src.thresholding import (
    NormalOnlyThresholdSelector,
    compute_threshold_sensitivity,
    compute_wilson_score_interval,
)


class TestThresholdSensitivity(unittest.TestCase):
    def test_exact_threshold_percentile_calculation(self):
        """Tests that percentiles 95, 97, and 99 match exact numpy percentiles."""
        scores = np.linspace(0.0, 100.0, 101)  # 0, 1, 2, ..., 100
        report = compute_threshold_sensitivity(scores, percentiles=[95.0, 97.0, 99.0], margin_multiplier=1.0)

        analysis = report["percentile_analysis"]
        self.assertEqual(len(analysis), 3)

        self.assertAlmostEqual(analysis[0]["raw_threshold"], 95.0, places=4)
        self.assertAlmostEqual(analysis[1]["raw_threshold"], 97.0, places=4)
        self.assertAlmostEqual(analysis[2]["raw_threshold"], 99.0, places=4)

    def test_false_positive_counting_logic(self):
        """Tests that validation false positive count correctly identifies points >= threshold."""
        # 100 samples from 1 to 100
        scores = np.arange(1, 101, dtype=np.float32)
        # 95th percentile of 1..100 is 95.05. Scores >= 95.05 are 96, 97, 98, 99, 100 (5 samples)
        report = compute_threshold_sensitivity(scores, percentiles=[95.0], margin_multiplier=1.0)
        p95 = report["percentile_analysis"][0]

        self.assertEqual(p95["raw_validation_false_positives"], 5)
        self.assertAlmostEqual(p95["raw_empirical_fpr"], 0.05, places=4)

        # With 1.05 margin multiplier: 95.05 * 1.05 = 99.8025. Score >= 99.8025 is 100 (1 sample)
        report_margin = compute_threshold_sensitivity(scores, percentiles=[95.0], margin_multiplier=1.05)
        p95_margin = report_margin["percentile_analysis"][0]
        self.assertEqual(p95_margin["margin_validation_false_positives"], 1)
        self.assertAlmostEqual(p95_margin["margin_empirical_fpr"], 0.01, places=4)

    def test_wilson_score_interval_bounds(self):
        """Tests Wilson score confidence interval calculation for small sample sizes."""
        # 0 false positives out of 180 normal validation images
        low, high = compute_wilson_score_interval(successes=0, total=180, confidence=0.95)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.015)
        self.assertLess(high, 0.025)  # Wilson 95% upper bound is ~2.07%

        # 9 false positives out of 180 (5.0%)
        low_5, high_5 = compute_wilson_score_interval(successes=9, total=180, confidence=0.95)
        self.assertGreater(low_5, 0.02)
        self.assertLess(high_5, 0.10)
        self.assertLess(low_5, 0.05)
        self.assertGreater(high_5, 0.05)

    def test_empty_scores_raises_error(self):
        """Tests that passing empty score list raises ValueError."""
        with self.assertRaises(ValueError):
            compute_threshold_sensitivity([])

    def test_default_threshold_selector_unchanged(self):
        """Ensures that default NormalOnlyThresholdSelector behavior remains 100% backward-compatible."""
        selector = NormalOnlyThresholdSelector(method="percentile", percentile=99.0, margin_multiplier=1.05)
        val_img_scores = [1.0, 1.2, 0.9, 1.1, 1.05]
        val_pixel_maps = [np.zeros((10, 10)), np.ones((10, 10)) * 0.5]

        selector.fit(val_img_scores, val_pixel_maps)
        self.assertTrue(selector.is_fitted)
        self.assertGreater(selector.image_threshold, 1.0)
        self.assertEqual(selector.classify_image(0.5), "normal")
        self.assertEqual(selector.classify_image(10.0), "anomaly")


if __name__ == "__main__":
    unittest.main()
