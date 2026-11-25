import os
import sys
import json
import shutil
import tempfile
import subprocess
import unittest
from PIL import Image

import predict


class TestPHPBackendIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = "checkpoints/defectx_bottle.pt"
        cls.config_path = "config/default_config.yaml"
        cls.normal_image = "data/mvtec/bottle/test/good/000.png"
        cls.defect_image = "data/mvtec/bottle/test/broken_large/000.png"

        # Verify prerequisite test images exist
        if not os.path.exists(cls.normal_image) or not os.path.exists(cls.defect_image):
            raise unittest.SkipTest("Sample bottle dataset not found. Run generate_sample_data.py first.")
        if not os.path.exists(cls.checkpoint_path):
            raise unittest.SkipTest("Checkpoint defectx_bottle.pt not found. Run train.py first.")

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_python_api_normal_inference(self):
        """Tests stable Python predict() interface on normal image."""
        result = predict.predict(
            image_path=self.normal_image,
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            output_dir=os.path.join(self.temp_dir, "heatmaps"),
        )
        self.assertIsInstance(result, dict)
        self.assertIn("inspection_id", result)
        self.assertEqual(result["prediction"], "normal")
        self.assertEqual(result["severity"], "NONE")
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["shift_detected"])
        self.assertTrue(os.path.exists(result["heatmap_path"]))
        self.assertTrue(os.path.exists(result["overlay_path"]))
        self.assertEqual(result["model_version"], "1.0.0")
        self.assertGreater(result["processing_time_ms"], 0.0)

    def test_python_api_defective_inference(self):
        """Tests stable Python predict() interface on defective image."""
        result = predict.predict(
            image_path=self.defect_image,
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            output_dir=os.path.join(self.temp_dir, "heatmaps"),
        )
        self.assertEqual(result["prediction"], "anomaly")
        self.assertIn(result["severity"], ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        self.assertTrue(os.path.exists(result["heatmap_path"]))

    def test_cli_clean_json_output(self):
        """Simulates PHP shell_exec / proc_open invoking predict.py CLI."""
        cmd = [
            sys.executable,
            "predict.py",
            "--image", self.normal_image,
            "--checkpoint", self.checkpoint_path,
            "--config", self.config_path,
            "--output-dir", os.path.join(self.temp_dir, "heatmaps"),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.assertEqual(proc.returncode, 0, f"CLI exited with non-zero code: {proc.stderr}")
        # Stdout must be strictly valid JSON
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            self.fail(f"predict.py stdout did not return valid JSON: {proc.stdout}\nError: {e}")

        self.assertIn("prediction", data)
        self.assertIn("inspection_id", data)
        self.assertEqual(data["prediction"], "normal")

    def test_no_overwrite_repeated_filenames(self):
        """
        Tests output management: multiple images with identical filename (000.png)
        from different directories must NOT overwrite each other's heatmaps/overlays.
        """
        output_dir = os.path.join(self.temp_dir, "heatmaps")

        # 1. Run on normal 000.png
        res1 = predict.predict(
            image_path=self.normal_image,
            checkpoint_path=self.checkpoint_path,
            output_dir=output_dir,
        )
        # 2. Run on defective 000.png
        res2 = predict.predict(
            image_path=self.defect_image,
            checkpoint_path=self.checkpoint_path,
            output_dir=output_dir,
        )

        self.assertNotEqual(res1["inspection_id"], res2["inspection_id"])
        self.assertNotEqual(res1["heatmap_path"], res2["heatmap_path"])
        self.assertNotEqual(res1["overlay_path"], res2["overlay_path"])

        # Both files must coexist on disk simultaneously
        self.assertTrue(os.path.exists(res1["heatmap_path"]))
        self.assertTrue(os.path.exists(res2["heatmap_path"]))

    def test_path_with_spaces_handling(self):
        """Tests that paths with spaces are handled safely without breaking CLI."""
        spaced_dir = os.path.join(self.temp_dir, "path with spaces", "sub folder")
        os.makedirs(spaced_dir, exist_ok=True)
        spaced_img = os.path.join(spaced_dir, "bottle sample image.png")
        shutil.copy(self.normal_image, spaced_img)

        cmd = [
            sys.executable,
            "predict.py",
            "--image", spaced_img,
            "--checkpoint", self.checkpoint_path,
            "--output-dir", os.path.join(self.temp_dir, "output with spaces"),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        self.assertEqual(data["prediction"], "normal")
        self.assertTrue(os.path.exists(data["heatmap_path"]))

    def test_health_check_cli(self):
        """Tests health checking via CLI and Python interface."""
        # 1. Healthy state
        cmd = [sys.executable, "predict.py", "--health", "--checkpoint", self.checkpoint_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 0)
        health = json.loads(proc.stdout)
        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["checkpoint_exists"])
        self.assertTrue(health["model_loadable"])
        self.assertEqual(health["model_version"], "1.0.0")

        # 2. Missing checkpoint state
        missing_ckpt = os.path.join(self.temp_dir, "nonexistent.pt")
        cmd_unhealthy = [sys.executable, "predict.py", "--health", "--checkpoint", missing_ckpt]
        proc_un = subprocess.run(cmd_unhealthy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc_un.returncode, 1)
        health_un = json.loads(proc_un.stdout)
        self.assertEqual(health_un["status"], "unhealthy")
        self.assertFalse(health_un["checkpoint_exists"])

    def test_error_missing_image(self):
        """Tests error response on non-existent image path."""
        non_existent = os.path.join(self.temp_dir, "does_not_exist.png")
        cmd = [sys.executable, "predict.py", "--image", non_existent, "--checkpoint", self.checkpoint_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.assertEqual(proc.returncode, 1)
        data = json.loads(proc.stdout)
        self.assertIn("error", data)
        self.assertEqual(data["error_code"], "MISSING_IMAGE")

    def test_error_unsupported_format(self):
        """Tests error response on unsupported image format (.txt)."""
        txt_file = os.path.join(self.temp_dir, "sample.txt")
        with open(txt_file, "w") as f:
            f.write("not an image")

        cmd = [sys.executable, "predict.py", "--image", txt_file, "--checkpoint", self.checkpoint_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.assertEqual(proc.returncode, 3)
        data = json.loads(proc.stdout)
        self.assertIn("error", data)
        self.assertEqual(data["error_code"], "INVALID_IMAGE")

    def test_error_corrupt_image(self):
        """Tests error response on corrupted image bytes."""
        corrupt_png = os.path.join(self.temp_dir, "broken.png")
        with open(corrupt_png, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00corrupt")

        cmd = [sys.executable, "predict.py", "--image", corrupt_png, "--checkpoint", self.checkpoint_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.assertEqual(proc.returncode, 3)
        data = json.loads(proc.stdout)
        self.assertIn("error", data)
        self.assertEqual(data["error_code"], "INVALID_IMAGE")

    def test_error_missing_checkpoint(self):
        """Tests error response on missing checkpoint."""
        non_existent_ckpt = os.path.join(self.temp_dir, "missing_model.pt")
        cmd = [sys.executable, "predict.py", "--image", self.normal_image, "--checkpoint", non_existent_ckpt]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.assertEqual(proc.returncode, 2)
        data = json.loads(proc.stdout)
        self.assertIn("error", data)
        self.assertEqual(data["error_code"], "MISSING_CHECKPOINT")


if __name__ == "__main__":
    unittest.main()
