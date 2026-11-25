import os
import shutil
import tempfile
import unittest
import torch
from torch.utils.data import DataLoader

from src.dataset import load_mvtec_splits
from src.model import DefectXEngine
from src.evaluator import DefectXEvaluator
from scripts.generate_sample_data import generate_sample_dataset


class TestEndToEndSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        # Generate minimal synthetic dataset: 10 train normal, 4 test normal, 4 test defect
        generate_sample_dataset(output_dir=cls.temp_dir, num_train=12)

        cls.config = {
            "project": {"name": "DefectX-Test", "version": "1.0.0"},
            "dataset": {"root_dir": cls.temp_dir, "category": "bottle", "image_size": [112, 112], "val_ratio": 0.25},
            "model": {"backbone": "resnet18", "layers": ["layer2", "layer3"]},
            "memory_bank": {"coreset_sampling_ratio": 0.1, "subsample_method": "greedy_coreset", "chunk_size": 256},
            "thresholding": {"method": "percentile", "percentile": 95.0, "margin_multiplier": 1.0},
            "heatmap": {"gaussian_sigma": 2.0, "overlay_alpha": 0.5},
            "severity": {"area_threshold_low": 0.01, "area_threshold_medium": 0.05, "area_threshold_high": 0.15},
            "confidence": {"calibrated": False},
            "output": {"checkpoints_dir": os.path.join(cls.temp_dir, "checkpoints"), "heatmaps_dir": os.path.join(cls.temp_dir, "heatmaps")},
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_end_to_end_pipeline(self):
        # 1. Test dataset loading
        train_ds, val_ds, test_ds = load_mvtec_splits(
            dataset_root=self.temp_dir,
            category="bottle",
            val_ratio=0.25,
            seed=42,
            image_size=(112, 112),
        )
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)
        self.assertGreater(len(test_ds), 0)

        # 2. Fit Engine (test CPU fallback and CUDA if available)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        engine = DefectXEngine(config=self.config, device=device)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        fit_res = engine.fit(train_loader, val_loader)
        self.assertIn("image_threshold", fit_res)
        self.assertGreater(fit_res["image_threshold"], 0.0)

        # 3. Test checkpoint save & reload
        ckpt_path = os.path.join(self.temp_dir, "checkpoints", "test_model.pt")
        engine.save_checkpoint(ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))

        reloaded_engine = DefectXEngine(config=self.config, device=device)
        reloaded_engine.load_checkpoint(ckpt_path)
        self.assertTrue(reloaded_engine.threshold_selector.is_fitted)

        # 4. Test inference prediction contract on test samples
        test_good_img = [s["image_path"] for s in test_ds.samples if s["label"] == 0][0]
        test_defect_img = [s["image_path"] for s in test_ds.samples if s["label"] == 1][0]

        res_good = reloaded_engine.predict(test_good_img, output_dir=os.path.join(self.temp_dir, "heatmaps"))
        res_defect = reloaded_engine.predict(test_defect_img, output_dir=os.path.join(self.temp_dir, "heatmaps"))

        # Verify PHP backend contract JSON fields
        for res in [res_good, res_defect]:
            self.assertIn("prediction", res)
            self.assertIn(res["prediction"], ["normal", "anomaly"])
            self.assertIsInstance(res["anomaly_score"], float)
            self.assertIn(res["severity"], ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
            self.assertIsNone(res["confidence"])  # Nullable when uncalibrated
            self.assertIsNone(res["shift_detected"])  # Reserved for Member 4
            self.assertTrue(os.path.exists(res["heatmap_path"]))
            self.assertTrue(os.path.exists(res["overlay_path"]))
            self.assertEqual(res["model_version"], "1.0.0")
            self.assertGreater(res["processing_time_ms"], 0.0)

        # Defect anomaly score should typically be higher than normal score
        self.assertGreater(res_defect["anomaly_score"], res_good["anomaly_score"])

        # 5. Test Evaluation metrics
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
        evaluator = DefectXEvaluator(reloaded_engine)
        metrics = evaluator.evaluate(test_loader, output_dir=os.path.join(self.temp_dir, "results"))

        self.assertIn("image_auroc", metrics)
        self.assertIn("pixel_auroc", metrics)
        self.assertGreaterEqual(metrics["image_auroc"], 0.5)


if __name__ == "__main__":
    unittest.main()
