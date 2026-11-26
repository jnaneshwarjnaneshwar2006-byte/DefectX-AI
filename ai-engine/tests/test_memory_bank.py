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

    def test_approximate_greedy_subsampling(self):
        device = torch.device("cpu")
        mb = MemoryBank(
            coreset_sampling_ratio=0.1,
            subsample_method="approximate_greedy",
            device=device,
            coreset_block_size=16,
        )
        dummy_features = torch.randn(500, 128)
        mb.fit(dummy_features, seed=42)

        self.assertEqual(mb.features.shape, (50, 128))
        self.assertEqual(mb.features.device.type, "cpu")

    def test_exact_greedy_subsampling(self):
        device = torch.device("cpu")
        mb = MemoryBank(
            coreset_sampling_ratio=0.05,
            subsample_method="exact_greedy",
            device=device,
        )
        dummy_features = torch.randn(200, 64)
        mb.fit(dummy_features, seed=42)

        self.assertEqual(mb.features.shape, (10, 64))

    def test_legacy_greedy_subsampling(self):
        device = torch.device("cpu")
        mb = MemoryBank(
            coreset_sampling_ratio=0.05,
            subsample_method="legacy_greedy",
            device=device,
        )
        dummy_features = torch.randn(100, 32)
        mb.fit(dummy_features, seed=42)

        self.assertEqual(mb.features.shape, (5, 32))

    def test_384_dimension_preservation(self):
        """Verifies original 384-d patch embeddings are preserved in the final coreset."""
        device = torch.device("cpu")
        mb = MemoryBank(
            coreset_sampling_ratio=0.10,
            subsample_method="approximate_greedy",
            device=device,
        )
        dummy_features = torch.randn(1000, 384)
        mb.fit(dummy_features, seed=42)

        self.assertEqual(mb.features.shape, (100, 384))

    def test_deterministic_reproducibility(self):
        """Verifies deterministic behavior with seed 42."""
        device = torch.device("cpu")
        dummy_features = torch.randn(500, 64)

        mb1 = MemoryBank(coreset_sampling_ratio=0.1, subsample_method="approximate_greedy", device=device)
        mb1.fit(dummy_features, seed=42)

        mb2 = MemoryBank(coreset_sampling_ratio=0.1, subsample_method="approximate_greedy", device=device)
        mb2.fit(dummy_features, seed=42)

        self.assertTrue(torch.equal(mb1.features, mb2.features))

    def test_cuda_execution_if_available(self):
        """Tests CUDA execution and device placement when CUDA is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this environment.")

        device = torch.device("cuda")
        mb = MemoryBank(
            coreset_sampling_ratio=0.1,
            subsample_method="approximate_greedy",
            device=device,
            coreset_block_size=32,
        )
        # Test with CPU input tensor (as supplied by train.py)
        cpu_features = torch.randn(1000, 384, device="cpu")
        mb.fit(cpu_features, seed=42)

        self.assertEqual(mb.features.shape, (100, 384))
        self.assertEqual(mb.features.device.type, "cuda")

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
