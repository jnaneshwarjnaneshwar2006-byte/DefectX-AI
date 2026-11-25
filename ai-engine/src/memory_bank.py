from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class MemoryBank:
    """
    Normal feature memory bank for PatchCore-inspired nearest-neighbor anomaly detection.
    Stores normal patch embeddings extracted from training data and performs coreset subsampling.
    """
    def __init__(
        self,
        coreset_sampling_ratio: float = 0.1,
        subsample_method: str = "greedy_coreset",
        device: torch.device = torch.device("cpu"),
        chunk_size: int = 512,
    ):
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.subsample_method = subsample_method
        self.device = device
        self.chunk_size = chunk_size
        self.features: Optional[torch.Tensor] = None

    def fit(self, features: torch.Tensor, seed: int = 42) -> None:
        """
        Stores normal features and applies coreset subsampling if configured.
        Args:
            features: Tensor of shape (Total_Patches, Feature_Dim)
            seed: random seed for reproducibility
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features tensor (N, D), got shape {features.shape}")

        num_samples, dim = features.shape
        target_size = max(1, int(num_samples * self.coreset_sampling_ratio))

        if target_size >= num_samples or self.coreset_sampling_ratio >= 1.0:
            self.features = features.to(self.device).float()
            return

        if self.subsample_method == "random":
            rng = torch.Generator().manual_seed(seed)
            perm = torch.randperm(num_samples, generator=rng)[:target_size]
            self.features = features[perm].to(self.device).float()
        elif self.subsample_method == "greedy_coreset":
            coreset_indices = self._greedy_k_center(features, target_size, seed=seed)
            self.features = features[coreset_indices].to(self.device).float()
        else:
            raise ValueError(f"Unknown subsample method '{self.subsample_method}'. Choose 'greedy_coreset' or 'random'.")

    def _greedy_k_center(self, features: torch.Tensor, target_size: int, seed: int = 42) -> torch.Tensor:
        """
        Greedy k-Center / Minimax coreset selection.
        Iteratively selects the point that maximizes the minimum distance to existing coreset.
        Uses random projection if dimension is high to accelerate selection.
        """
        num_samples = features.shape[0]
        device = features.device

        # If features are very large, randomly project for fast coreset distance computation
        if features.shape[1] > 64 and num_samples > 5000:
            rng = torch.Generator(device=device).manual_seed(seed)
            proj_dim = 64
            proj_matrix = torch.randn((features.shape[1], proj_dim), generator=rng, device=device)
            proj_matrix = F.normalize(proj_matrix, p=2, dim=0)
            work_features = torch.matmul(features, proj_matrix)
        else:
            work_features = features

        selected_indices = []
        rng = torch.Generator(device=device).manual_seed(seed)
        first_idx = torch.randint(0, num_samples, (1,), generator=rng, device=device).item()
        selected_indices.append(first_idx)

        # Min distances from each point to current coreset
        min_distances = torch.norm(work_features - work_features[first_idx], dim=1)

        # Greedy selection loop
        for _ in range(1, target_size):
            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)

            dist_to_new = torch.norm(work_features - work_features[next_idx], dim=1)
            min_distances = torch.minimum(min_distances, dist_to_new)

        return torch.tensor(selected_indices, dtype=torch.long, device=device)

    @torch.inference_mode()
    def compute_distances(self, query_patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes nearest-neighbor anomaly distances for query patch embeddings.
        Args:
            query_patches: Tensor of shape (B, N_patches, C)
        Returns:
            patch_scores: Tensor of shape (B, N_patches) - anomaly score per patch
            image_scores: Tensor of shape (B,) - image-level anomaly score
        """
        if self.features is None:
            raise RuntimeError("Memory bank is empty. Please call fit() before computing distances.")

        B, N, C = query_patches.shape
        flat_queries = query_patches.reshape(-1, C).to(self.device).float()
        total_queries = flat_queries.shape[0]

        min_distances_list = []

        # Process queries in conservative chunks to control peak VRAM on RTX 4050
        for start_idx in range(0, total_queries, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_queries)
            chunk = flat_queries[start_idx:end_idx]

            # Compute pairwise Euclidean distances between chunk and memory bank
            # shape: (chunk_size, M)
            dists = torch.cdist(chunk, self.features, p=2.0)
            min_dists, _ = torch.min(dists, dim=1)
            min_distances_list.append(min_dists)

        all_min_dists = torch.cat(min_distances_list, dim=0)
        patch_scores = all_min_dists.reshape(B, N)

        # Image-level anomaly score: top patch distance with PatchCore softmax re-weighting
        # s* = max_i s_i
        max_patch_scores, max_indices = torch.max(patch_scores, dim=1)

        image_scores = max_patch_scores

        return patch_scores, image_scores

    def to_device(self, device: torch.device):
        self.device = device
        if self.features is not None:
            self.features = self.features.to(device)
        return self

    def state_dict(self):
        return {
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "subsample_method": self.subsample_method,
            "features": self.features.cpu() if self.features is not None else None,
        }

    def load_state_dict(self, state: dict):
        self.coreset_sampling_ratio = state["coreset_sampling_ratio"]
        self.subsample_method = state["subsample_method"]
        if state["features"] is not None:
            self.features = state["features"].to(self.device)
        else:
            self.features = None
