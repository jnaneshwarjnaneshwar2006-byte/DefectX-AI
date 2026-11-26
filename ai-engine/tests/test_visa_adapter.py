import os
import csv
import shutil
import tempfile
import unittest
import torch
import numpy as np
from PIL import Image

from src.dataset_adapter import (
    VisAAdapter,
    MVTecAdapter,
    GenericFolderAdapter,
    get_dataset_adapter,
    detect_dataset_format,
    DatasetInspectionReport,
)


class TestVisAAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.split_csv_dir = os.path.join(self.temp_dir, "split_csv")
        os.makedirs(self.split_csv_dir, exist_ok=True)
        self.csv_path = os.path.join(self.split_csv_dir, "1cls.csv")

        # Build mock VisA category folders: 'candle' and 'capsules'
        self.candle_dir = os.path.join(self.temp_dir, "candle", "Data")
        self.candle_norm_dir = os.path.join(self.candle_dir, "Images", "Normal")
        self.candle_anom_dir = os.path.join(self.candle_dir, "Images", "Anomaly")
        self.candle_mask_dir = os.path.join(self.candle_dir, "Masks", "Anomaly")
        for d in [self.candle_norm_dir, self.candle_anom_dir, self.candle_mask_dir]:
            os.makedirs(d, exist_ok=True)

        # Create dummy normal images (including uppercase .JPG)
        for i in range(10):
            ext = ".JPG" if i % 2 == 0 else ".jpg"
            img_path = os.path.join(self.candle_norm_dir, f"{i:04d}{ext}")
            Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

        # Create dummy anomaly images and masks (.png)
        for i in range(4):
            img_path = os.path.join(self.candle_anom_dir, f"{i:03d}.JPG")
            mask_path = os.path.join(self.candle_mask_dir, f"{i:03d}.png")
            Image.new("RGB", (64, 64), color=(255, 0, 0)).save(img_path)
            # Create a mask with a bright 16x16 square in the center
            mask_arr = np.zeros((64, 64), dtype=np.uint8)
            mask_arr[24:40, 24:40] = 255
            Image.fromarray(mask_arr, mode="L").save(mask_path)

        # Write mock 1cls.csv
        # candle: 8 train-normal, 2 test-normal, 4 test-anomaly
        self.mock_rows = [
            # Train normal
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/0000.JPG", "mask": ""},
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/0001.jpg", "mask": ""},
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/0002.JPG", "mask": ""},
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/0003.jpg", "mask": ""},
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/0004.JPG", "mask": ""},
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/0005.jpg", "mask": ""},
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/0006.JPG", "mask": ""},
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/0007.jpg", "mask": ""},
            # Test normal
            {"object": "candle", "split": "test", "label": "normal", "image": "candle/Data/Images/Normal/0008.JPG", "mask": ""},
            {"object": "candle", "split": "test", "label": "normal", "image": "candle/Data/Images/Normal/0009.jpg", "mask": ""},
            # Test anomaly
            {"object": "candle", "split": "test", "label": "anomaly", "image": "candle/Data/Images/Anomaly/000.JPG", "mask": "candle/Data/Masks/Anomaly/000.png"},
            {"object": "candle", "split": "test", "label": "anomaly", "image": "candle/Data/Images/Anomaly/001.JPG", "mask": "candle/Data/Masks/Anomaly/001.png"},
            {"object": "candle", "split": "test", "label": "anomaly", "image": "candle/Data/Images/Anomaly/002.JPG", "mask": "candle/Data/Masks/Anomaly/002.png"},
            {"object": "candle", "split": "test", "label": "anomaly", "image": "candle/Data/Images/Anomaly/003.JPG", "mask": "candle/Data/Masks/Anomaly/003.png"},
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["object", "split", "label", "image", "mask"])
            writer.writeheader()
            writer.writerows(self.mock_rows)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_12_category_names(self):
        """Test 3: Confirms all 12 official VisA category names are defined in VisAAdapter."""
        expected = [
            "candle", "capsules", "cashew", "chewinggum",
            "fryum", "macaroni1", "macaroni2",
            "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
        ]
        self.assertEqual(sorted(VisAAdapter.ALL_CATEGORIES), sorted(expected))

    def test_official_csv_parsing_and_inspection(self):
        """Test 1 & 2: CSV parsing and category inspection."""
        adapter = VisAAdapter()
        report = adapter.inspect(root_dir=self.temp_dir, category="candle")
        self.assertEqual(report.detected_format, "visa")
        self.assertEqual(report.total_images, 14)
        self.assertEqual(report.valid_images, 14)
        self.assertEqual(report.normal_image_count, 10)
        self.assertEqual(report.defective_image_count, 4)
        self.assertEqual(report.mask_count, 4)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(len(report.missing_images), 0)

    def test_deterministic_train_val_split_and_zero_leakage(self):
        """Test 5, 6, 7: Normal-only training samples, 80/20 split, zero leakage."""
        adapter = VisAAdapter()
        train_ds, val_ds, test_ds = adapter.load_splits(
            root_dir=self.temp_dir,
            category="candle",
            val_ratio=0.25,  # 8 train samples * 0.25 = 2 val samples, 6 train samples
            seed=42,
            image_size=(64, 64),
        )

        self.assertEqual(len(train_ds), 6)
        self.assertEqual(len(val_ds), 2)
        self.assertEqual(len(test_ds), 6)  # 2 test-normal + 4 test-anomaly

        # Verify zero leakage: train and val samples come exclusively from train split
        train_paths = set(train_ds.image_paths)
        val_paths = set(val_ds.image_paths)
        self.assertTrue(train_paths.isdisjoint(val_paths))

        test_paths = set(s["image_path"] for s in test_ds.samples)
        self.assertTrue(train_paths.isdisjoint(test_paths))
        self.assertTrue(val_paths.isdisjoint(test_paths))

        # Same seed produces exact same splits
        t_ds2, v_ds2, _ = adapter.load_splits(
            root_dir=self.temp_dir,
            category="candle",
            val_ratio=0.25,
            seed=42,
            image_size=(64, 64),
        )
        self.assertEqual(train_ds.image_paths, t_ds2.image_paths)
        self.assertEqual(val_ds.image_paths, v_ds2.image_paths)

    def test_labels_and_mask_handling(self):
        """Test 4, 8, 9, 12: Normal/anomaly labels, mask loading, zero masks for normal, uppercase .JPG."""
        adapter = VisAAdapter()
        _, _, test_ds = adapter.load_splits(
            root_dir=self.temp_dir,
            category="candle",
            image_size=(64, 64),
        )

        normal_samples = [s for s in test_ds if s["label"] == 0]
        anomaly_samples = [s for s in test_ds if s["label"] == 1]

        self.assertEqual(len(normal_samples), 2)
        self.assertEqual(len(anomaly_samples), 4)

        # Normal samples must have zero-valued mask
        for s in normal_samples:
            self.assertEqual(s["label"], 0)
            self.assertEqual(s["defect_type"], "good")
            self.assertEqual(s["mask"].sum().item(), 0.0)
            self.assertEqual(s["image"].shape, (3, 64, 64))

        # Anomaly samples must have positive mask pixels and correct binary values (0 or 1)
        for s in anomaly_samples:
            self.assertEqual(s["label"], 1)
            self.assertEqual(s["defect_type"], "anomaly")
            self.assertGreater(s["mask"].sum().item(), 0.0)
            unique_vals = torch.unique(s["mask"]).tolist()
            for v in unique_vals:
                self.assertIn(v, [0.0, 1.0])

    def test_visa_mask_pixel_values_0_and_1(self):
        """Tests that VisA masks with raw pixel values {0, 1} are cleanly binarized to {0.0, 1.0}."""
        from src.dataset import MVTecTestDataset

        mask_p = os.path.join(self.temp_dir, "visa_raw_0_1_mask.png")
        img_p = os.path.join(self.temp_dir, "visa_dummy_img.png")

        # Create raw mask with {0, 1} pixel values
        raw_mask_arr = np.zeros((100, 100), dtype=np.uint8)
        raw_mask_arr[30:70, 30:70] = 1  # Foreground value is 1
        Image.fromarray(raw_mask_arr, mode="L").save(mask_p)
        Image.new("RGB", (100, 100), color=(100, 100, 100)).save(img_p)

        sample = [{
            "image_path": img_p,
            "label": 1,
            "defect_type": "anomaly",
            "mask_path": mask_p
        }]
        ds = MVTecTestDataset(sample, image_size=(224, 224))
        loaded = ds[0]

        mask = loaded["mask"]
        self.assertEqual(mask.shape, (1, 224, 224))
        self.assertGreater(mask.sum().item(), 0.0)
        unique_vals = sorted(torch.unique(mask).tolist())
        self.assertEqual(unique_vals, [0.0, 1.0])
        # Verify foreground pixels count preserved proportionally after resize
        expected_approx_pixels = int(1600 * ((224 / 100) ** 2))
        self.assertAlmostEqual(mask.sum().item(), expected_approx_pixels, delta=200)

    def test_mvtec_mask_pixel_values_0_and_255(self):
        """Tests that MVTec masks with raw pixel values {0, 255} are cleanly binarized to {0.0, 1.0}."""
        from src.dataset import MVTecTestDataset

        mask_p = os.path.join(self.temp_dir, "mvtec_raw_0_255_mask.png")
        img_p = os.path.join(self.temp_dir, "mvtec_dummy_img.png")

        # Create raw mask with {0, 255} pixel values
        raw_mask_arr = np.zeros((100, 100), dtype=np.uint8)
        raw_mask_arr[30:70, 30:70] = 255  # Foreground value is 255
        Image.fromarray(raw_mask_arr, mode="L").save(mask_p)
        Image.new("RGB", (100, 100), color=(100, 100, 100)).save(img_p)

        sample = [{
            "image_path": img_p,
            "label": 1,
            "defect_type": "anomaly",
            "mask_path": mask_p
        }]
        ds = MVTecTestDataset(sample, image_size=(224, 224))
        loaded = ds[0]

        mask = loaded["mask"]
        self.assertEqual(mask.shape, (1, 224, 224))
        self.assertGreater(mask.sum().item(), 0.0)
        unique_vals = sorted(torch.unique(mask).tolist())
        self.assertEqual(unique_vals, [0.0, 1.0])

    def test_mask_spatial_alignment_and_normal_all_zeros(self):
        """Tests that masks maintain exact spatial dimensions and normal test samples are all zeros."""
        from src.dataset import MVTecTestDataset

        img_p = os.path.join(self.temp_dir, "norm_test_img.png")
        Image.new("RGB", (200, 150), color=(50, 50, 50)).save(img_p)

        samples = [{
            "image_path": img_p,
            "label": 0,
            "defect_type": "good",
            "mask_path": None,
        }]
        ds = MVTecTestDataset(samples, image_size=(224, 224))
        loaded = ds[0]

        self.assertEqual(loaded["image"].shape, (3, 224, 224))
        self.assertEqual(loaded["mask"].shape, (1, 224, 224))
        self.assertEqual(loaded["mask"].sum().item(), 0.0)

    def test_missing_image_error(self):
        """Test 10: Missing image error handling."""
        # Add a reference to non-existent image in train
        bad_rows = list(self.mock_rows) + [
            {"object": "candle", "split": "train", "label": "normal", "image": "candle/Data/Images/Normal/missing_9999.JPG", "mask": ""}
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["object", "split", "label", "image", "mask"])
            writer.writeheader()
            writer.writerows(bad_rows)

        adapter = VisAAdapter()
        with self.assertRaises(FileNotFoundError):
            adapter.load_splits(root_dir=self.temp_dir, category="candle")

    def test_missing_mask_error(self):
        """Test 11: Missing mask error handling on anomalous test sample."""
        bad_rows = list(self.mock_rows) + [
            {"object": "candle", "split": "test", "label": "anomaly", "image": "candle/Data/Images/Anomaly/000.JPG", "mask": "candle/Data/Masks/Anomaly/missing_mask.png"}
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["object", "split", "label", "image", "mask"])
            writer.writeheader()
            writer.writerows(bad_rows)

        adapter = VisAAdapter()
        with self.assertRaises(FileNotFoundError):
            adapter.load_splits(root_dir=self.temp_dir, category="candle")

    def test_training_split_with_anomaly_rejected(self):
        """Test 5 (negative case): Anomaly sample in training split raises ValueError."""
        bad_rows = list(self.mock_rows) + [
            {"object": "candle", "split": "train", "label": "anomaly", "image": "candle/Data/Images/Anomaly/000.JPG", "mask": ""}
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["object", "split", "label", "image", "mask"])
            writer.writeheader()
            writer.writerows(bad_rows)

        adapter = VisAAdapter()
        with self.assertRaises(ValueError) as ctx:
            adapter.load_splits(root_dir=self.temp_dir, category="candle")
        self.assertIn("non-normal", str(ctx.exception))

    def test_adapter_factory_and_auto_detection(self):
        """Test 13: Factory creates VisAAdapter for explicit 'visa' and auto-detected directories."""
        cfg_explicit = {"dataset": {"adapter": "visa", "root_dir": self.temp_dir}}
        adapter1 = get_dataset_adapter(cfg_explicit)
        self.assertIsInstance(adapter1, VisAAdapter)

        # Auto detection when split_csv/1cls.csv is present
        fmt = detect_dataset_format(self.temp_dir)
        self.assertEqual(fmt, "visa")

        cfg_auto = {"dataset": {"adapter": "auto", "root_dir": self.temp_dir}}
        adapter2 = get_dataset_adapter(cfg_auto)
        self.assertIsInstance(adapter2, VisAAdapter)

    def test_regression_compatibility_mvtec_and_custom(self):
        """Test 14: Preserves MVTecAdapter and GenericFolderAdapter functionality."""
        cfg_mvtec = {"dataset": {"adapter": "mvtec", "root_dir": "data/mvtec", "category": "bottle"}}
        adapter_mvtec = get_dataset_adapter(cfg_mvtec)
        self.assertIsInstance(adapter_mvtec, MVTecAdapter)

        cfg_custom = {"dataset": {"adapter": "custom", "root_dir": "data/custom"}}
        adapter_custom = get_dataset_adapter(cfg_custom)
        self.assertIsInstance(adapter_custom, GenericFolderAdapter)


class TestRealVisAManifest(unittest.TestCase):
    """Integration check directly inspecting the live on-disk VisA manifest (READ-ONLY)."""

    def test_live_visa_manifest_integrity(self):
        real_visa_root = "data/visa" if os.path.isdir("data/visa") else "ai-engine/data/visa"
        csv_path = os.path.join(real_visa_root, "split_csv", "1cls.csv")
        if not os.path.exists(csv_path):
            self.skipTest(f"VisA dataset not found at '{real_visa_root}'. Skipping live manifest test.")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 10821, "VisA 1cls.csv must contain exactly 10,821 data rows.")

        train_rows = [r for r in rows if r["split"] == "train"]
        for r in train_rows:
            self.assertEqual(r["label"], "normal", "All training rows in VisA must have label == 'normal'.")

        adapter = VisAAdapter()
        # Verify inspection on live candle category (without loading images to keep fast)
        report = adapter.inspect(root_dir=real_visa_root, category="candle")
        self.assertEqual(report.total_images, 1100)
        self.assertEqual(report.normal_image_count, 1000)
        self.assertEqual(report.defective_image_count, 100)
        self.assertEqual(len(report.missing_images), 0)
        self.assertEqual(len(report.corrupt_images), 0)

    def test_live_visa_mask_binarization(self):
        real_visa_root = "data/visa" if os.path.isdir("data/visa") else "ai-engine/data/visa"
        csv_path = os.path.join(real_visa_root, "split_csv", "1cls.csv")
        if not os.path.exists(csv_path):
            self.skipTest(f"VisA dataset not found at '{real_visa_root}'. Skipping live mask test.")

        adapter = VisAAdapter()
        _, _, test_ds = adapter.load_splits(
            root_dir=real_visa_root,
            category="candle",
            image_size=(224, 224),
        )

        # Check normal test sample
        first_norm = test_ds[0]
        self.assertEqual(first_norm["label"], 0)
        self.assertEqual(first_norm["mask"].sum().item(), 0.0)

        # Check anomaly test sample (which has {0, 1} on disk)
        anom_idx = [i for i, s in enumerate(test_ds.samples) if s["label"] == 1][0]
        first_anom = test_ds[anom_idx]
        self.assertEqual(first_anom["label"], 1)
        # MUST be positive and contain clean binary values [0.0, 1.0]
        self.assertGreater(first_anom["mask"].sum().item(), 0.0)
        unique_vals = sorted(torch.unique(first_anom["mask"]).tolist())
        self.assertEqual(unique_vals, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
