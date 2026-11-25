import os
import shutil
import tempfile
import unittest
from PIL import Image

from src.dataset_adapter import (
    validate_image_file,
    detect_dataset_format,
    get_dataset_adapter,
    MVTecAdapter,
    GenericFolderAdapter,
)


class TestDatasetAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_image_validation_valid(self):
        valid_img_path = os.path.join(self.temp_dir, "valid.png")
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        img.save(valid_img_path)

        is_valid, err, meta = validate_image_file(valid_img_path)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertEqual(meta["format"], "PNG")
        self.assertEqual(meta["dimensions"], "128x128")
        self.assertEqual(meta["mode"], "RGB")

    def test_image_validation_corrupt(self):
        corrupt_img_path = os.path.join(self.temp_dir, "corrupt.png")
        # Write invalid binary data pretending to be PNG
        with open(corrupt_img_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorrupt_random_data")

        is_valid, err, meta = validate_image_file(corrupt_img_path)
        self.assertFalse(is_valid)
        self.assertIsNotNone(err)
        self.assertIn("Corrupted", err)

    def test_image_validation_zero_byte(self):
        zero_img_path = os.path.join(self.temp_dir, "empty.png")
        with open(zero_img_path, "wb") as f:
            pass  # 0 bytes

        is_valid, err, meta = validate_image_file(zero_img_path)
        self.assertFalse(is_valid)
        self.assertIn("Zero-byte", err)

    def test_image_validation_missing(self):
        missing_path = os.path.join(self.temp_dir, "does_not_exist.png")
        is_valid, err, meta = validate_image_file(missing_path)
        self.assertFalse(is_valid)
        self.assertIn("does not exist", err)

    def test_mvtec_adapter_inspection(self):
        # Build minimal MVTec layout
        bottle_dir = os.path.join(self.temp_dir, "bottle")
        train_good = os.path.join(bottle_dir, "train", "good")
        test_good = os.path.join(bottle_dir, "test", "good")
        test_defect = os.path.join(bottle_dir, "test", "crack")
        gt_defect = os.path.join(bottle_dir, "ground_truth", "crack")

        for d in [train_good, test_good, test_defect, gt_defect]:
            os.makedirs(d, exist_ok=True)

        for i in range(5):
            Image.new("RGB", (64, 64), color=(200, 200, 200)).save(os.path.join(train_good, f"{i:03d}.png"))
        for i in range(2):
            Image.new("RGB", (64, 64), color=(200, 200, 200)).save(os.path.join(test_good, f"{i:03d}.png"))
        for i in range(3):
            Image.new("RGB", (64, 64), color=(50, 50, 50)).save(os.path.join(test_defect, f"{i:03d}.png"))
            Image.new("L", (64, 64), color=255).save(os.path.join(gt_defect, f"{i:03d}_mask.png"))

        adapter = MVTecAdapter()
        report = adapter.inspect(root_dir=self.temp_dir, category="bottle")

        self.assertEqual(report.detected_format, "mvtec")
        self.assertEqual(report.total_images, 10)
        self.assertEqual(report.valid_images, 10)
        self.assertEqual(len(report.corrupt_images), 0)
        self.assertEqual(report.normal_image_count, 7)  # 5 train + 2 test good
        self.assertEqual(report.defective_image_count, 3)
        self.assertIn("crack", report.defect_categories)
        self.assertEqual(report.mask_count, 3)
        self.assertEqual(report.ground_truth_mask_availability["crack"]["coverage_pct"], 100.0)

        # Test splits & zero-leakage
        train_ds, val_ds, test_ds = adapter.load_splits(
            root_dir=self.temp_dir,
            category="bottle",
            val_ratio=0.2,
            seed=42,
        )
        self.assertEqual(len(train_ds), 4)
        self.assertEqual(len(val_ds), 1)
        self.assertEqual(len(test_ds), 5)  # 2 good + 3 crack

        # Verify zero leakage
        train_set = set(train_ds.image_paths)
        val_set = set(val_ds.image_paths)
        test_set = set(s["image_path"] for s in test_ds.samples)
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)

    def test_custom_organiser_folder_adapter(self):
        # Simulate an organiser dataset with custom layout:
        # custom_ds/
        #   train/
        #     normal/ (6 images)
        #   test/
        #     normal/ (2 images)
        #     scratch/ (3 images)
        #     dent/ (2 images)
        custom_root = os.path.join(self.temp_dir, "organiser_dataset")
        train_normal = os.path.join(custom_root, "train", "normal")
        test_normal = os.path.join(custom_root, "test", "normal")
        test_scratch = os.path.join(custom_root, "test", "scratch")
        test_dent = os.path.join(custom_root, "test", "dent")

        for p in [train_normal, test_normal, test_scratch, test_dent]:
            os.makedirs(p, exist_ok=True)

        for i in range(6):
            Image.new("RGB", (64, 64), color=(220, 220, 220)).save(os.path.join(train_normal, f"norm_{i}.jpg"))
        for i in range(2):
            Image.new("RGB", (64, 64), color=(220, 220, 220)).save(os.path.join(test_normal, f"t_norm_{i}.jpg"))
        for i in range(3):
            Image.new("RGB", (64, 64), color=(80, 80, 80)).save(os.path.join(test_scratch, f"scratch_{i}.jpg"))
        for i in range(2):
            Image.new("RGB", (64, 64), color=(30, 30, 30)).save(os.path.join(test_dent, f"dent_{i}.jpg"))

        adapter = GenericFolderAdapter(normal_class_names=["normal"])
        report = adapter.inspect(root_dir=custom_root)

        self.assertEqual(report.total_images, 13)
        self.assertEqual(report.normal_image_count, 8)
        self.assertEqual(report.defective_image_count, 5)
        self.assertIn("scratch", report.defect_categories)
        self.assertIn("dent", report.defect_categories)

        # Test loading splits
        train_ds, val_ds, test_ds = adapter.load_splits(
            root_dir=custom_root,
            val_ratio=0.34,
            seed=42,
        )
        # 6 normal train -> 4 train normal, 2 val normal
        self.assertEqual(len(train_ds), 4)
        self.assertEqual(len(val_ds), 2)
        # Test dataset contains 2 normal + 3 scratch + 2 dent = 7
        self.assertEqual(len(test_ds), 7)

        # Zero leakage check
        train_paths = set(train_ds.image_paths)
        val_paths = set(val_ds.image_paths)
        test_paths = set(s["image_path"] for s in test_ds.samples)
        self.assertEqual(len(train_paths.intersection(val_paths)), 0)
        self.assertEqual(len(train_paths.intersection(test_paths)), 0)

    def test_auto_detection_format(self):
        # 1. MVTec format
        mvtec_path = os.path.join(self.temp_dir, "detect_mvtec", "prod", "train", "good")
        os.makedirs(mvtec_path, exist_ok=True)
        detected = detect_dataset_format(os.path.join(self.temp_dir, "detect_mvtec"), category="prod")
        self.assertEqual(detected, "mvtec")

        # 2. Generic format
        generic_path = os.path.join(self.temp_dir, "detect_generic", "normal")
        os.makedirs(generic_path, exist_ok=True)
        detected_gen = detect_dataset_format(os.path.join(self.temp_dir, "detect_generic"))
        self.assertEqual(detected_gen, "generic_folder")


if __name__ == "__main__":
    unittest.main()
