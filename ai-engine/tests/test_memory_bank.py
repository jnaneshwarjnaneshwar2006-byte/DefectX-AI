import unittest
import torch
from src.memory_bank import MemoryBank


class TestMemoryBank(unittest.TestCase):
    def test_random_coreset_subsampling(self):
        device = torch.device("cpu")
        mb = MemoryBank(
            coreset_sampling_ratio=0.1,
            subsample_method="random",
            device=device,
        )
        # 1000 patches of dimension 64
        dummy_features = torch.randn(1000, 64)
        mb.fit(dummy_features, seed=42)

        self.assertIsNotNone(mb.features)
        self.assertEqual(mb.features.shape, (100, 64))

    def test_greedy_coreset_subsampling(self):
        device = torch.device("cpu")
        mb = MemoryBank(
            coreset_sampling_ratio=0.05,
            subsample_method="greedy_coreset",
            device=device,
        )
        dummy_features = torch.randn(200, 32)
        mb.fit(dummy_features, seed=42)

        self.assertEqual(mb.features.shape, (10, 32))

    def test_distance_computation(self):
        device = torch.device("cpu")
        mb = MemoryBank(coreset_sampling_ratio=1.0, device=device)
        # Fixed memory bank with known point
        bank_features = torch.zeros(10, 8)
        mb.fit(bank_features)

        # Query point with distance 2.0
        query_patches = torch.zeros(1, 4, 8)
        query_patches[0, 0, 0] = 2.0  # L2 distance = 2.0

        patch_scores, img_scores = mb.compute_distances(query_patches)

        self.assertEqual(patch_scores.shape, (1, 4))
        self.assertAlmostEqual(patch_scores[0, 0].item(), 2.0, places=4)
        self.assertAlmostEqual(patch_scores[0, 1].item(), 0.0, places=4)
        # Image score is top patch score = 2.0
        self.assertAlmostEqual(img_scores[0].item(), 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
