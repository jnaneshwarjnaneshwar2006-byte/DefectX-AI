import os
import shutil
import tempfile
import unittest
from PIL import Image
from src.dataset import load_mvtec_splits


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.category_dir = os.path.join(self.temp_dir, "bottle")
        self.train_good_dir = os.path.join(self.category_dir, "train", "good")
        self.test_good_dir = os.path.join(self.category_dir, "test", "good")
        self.test_broken_dir = os.path.join(self.category_dir, "test", "broken_large")
        self.gt_broken_dir = os.path.join(self.category_dir, "ground_truth", "broken_large")

        os.makedirs(self.train_good_dir)
        os.makedirs(self.test_good_dir)
        os.makedirs(self.test_broken_dir)
        os.makedirs(self.gt_broken_dir)

        # Create dummy images
        for i in range(10):
            img = Image.new("RGB", (64, 64), color=(200, 200, 200))
            img.save(os.path.join(self.train_good_dir, f"{i:03d}.png"))

        for i in range(4):
            img = Image.new("RGB", (64, 64), color=(200, 200, 200))
            img.save(os.path.join(self.test_good_dir, f"{i:03d}.png"))

        for i in range(4):
            img = Image.new("RGB", (64, 64), color=(50, 50, 50))
            img.save(os.path.join(self.test_broken_dir, f"{i:03d}.png"))
            mask = Image.new("L", (64, 64), color=255)
            mask.save(os.path.join(self.gt_broken_dir, f"{i:03d}_mask.png"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_zero_leakage_and_splits(self):
        train_ds, val_ds, test_ds = load_mvtec_splits(
            dataset_root=self.temp_dir,
            category="bottle",
            val_ratio=0.2,
            seed=42,
            image_size=(64, 64),
        )

        # 10 train/good -> 8 train normal, 2 val normal
        self.assertEqual(len(train_ds), 8)
        self.assertEqual(len(val_ds), 2)
        # Total test = 4 good + 4 broken = 8
        self.assertEqual(len(test_ds), 8)

        # Verify zero leakage (no intersection between train and val paths)
        train_paths = set(train_ds.image_paths)
        val_paths = set(val_ds.image_paths)
        self.assertEqual(len(train_paths.intersection(val_paths)), 0)

        # Check mask loading on defective sample
        defect_sample = [s for s in test_ds.samples if s["label"] == 1][0]
        self.assertIsNotNone(defect_sample["mask_path"])


if __name__ == "__main__":
    unittest.main()
