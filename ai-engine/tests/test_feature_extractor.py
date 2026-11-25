import unittest
import torch
from src.feature_extractor import PatchFeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def test_output_shapes_cpu(self):
        device = torch.device("cpu")
        extractor = PatchFeatureExtractor(
            backbone_name="resnet18",
            layers=["layer2", "layer3"],
            device=device,
        )
        dummy_input = torch.randn(2, 3, 224, 224, device=device)
        patches, grid_shape = extractor(dummy_input)

        # Expected grid for 224x224 input through ResNet layer2 is (28, 28) = 784 patches
        self.assertEqual(grid_shape, (28, 28))
        # Total channels: layer2 (128) + layer3 (256) = 384
        self.assertEqual(patches.shape, (2, 28 * 28, 384))

    def test_output_shapes_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this machine")
        device = torch.device("cuda:0")
        extractor = PatchFeatureExtractor(
            backbone_name="resnet18",
            layers=["layer2", "layer3"],
            device=device,
        )
        dummy_input = torch.randn(2, 3, 224, 224, device=device)
        patches, grid_shape = extractor(dummy_input)

        self.assertEqual(grid_shape, (28, 28))
        self.assertEqual(patches.shape, (2, 784, 384))


if __name__ == "__main__":
    unittest.main()
