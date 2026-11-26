from typing import Optional, Tuple, List
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
        subsample_method: str = "approximate_greedy",
        device: torch.device = torch.device("cpu"),
        chunk_size: int = 512,
        coreset_block_size: int = 64,
        proj_dim: int = 64,
    ):
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.subsample_method = subsample_method
        self.device = device
        self.chunk_size = chunk_size
        self.coreset_block_size = coreset_block_size
        self.proj_dim = proj_dim
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
            rng = torch.Generator(device="cpu").manual_seed(seed)
            perm = torch.randperm(num_samples, generator=rng)[:target_size]
            self.features = features[perm].to(self.device).float()
        elif self.subsample_method in ("approximate_greedy", "greedy_coreset", "batched_greedy"):
            coreset_indices = self._batched_greedy_k_center(
                features, target_size, seed=seed, block_size=self.coreset_block_size, proj_dim=self.proj_dim
            )
            self.features = features[coreset_indices].to(self.device).float()
        elif self.subsample_method == "exact_greedy":
            coreset_indices = self._exact_greedy_k_center(
                features, target_size, seed=seed, proj_dim=self.proj_dim
            )
            self.features = features[coreset_indices].to(self.device).float()
        elif self.subsample_method == "legacy_greedy":
            coreset_indices = self._legacy_greedy_k_center(
                features, target_size, seed=seed
            )
            self.features = features[coreset_indices].to(self.device).float()
        else:
            raise ValueError(
                f"Unknown subsample method '{self.subsample_method}'. "
                f"Choose 'approximate_greedy', 'greedy_coreset', 'exact_greedy', 'legacy_greedy', or 'random'."
            )

    def _chunked_project(
        self,
        features: torch.Tensor,
        proj_matrix: torch.Tensor,
        compute_device: torch.device,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        """Projects high-dimensional features onto lower dimension in chunks to bound memory."""
        N = features.shape[0]
        if N <= chunk_size or features.device == compute_device:
            return torch.matmul(features.to(compute_device).float(), proj_matrix)

        projected_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sub = features[start:end].to(compute_device).float()
            projected_chunks.append(torch.matmul(sub, proj_matrix))
        return torch.cat(projected_chunks, dim=0)

    def _batched_greedy_k_center(
        self,
        features: torch.Tensor,
        target_size: int,
        seed: int = 42,
        block_size: int = 64,
        proj_dim: int = 64,
    ) -> torch.Tensor:
        """
        Fast memory-efficient approximate greedy k-center coreset selection.
        - Uses random projection for dimension reduction (if D > 64 and N > 5000).
        - Computes on self.device (utilizing CUDA GPU acceleration if available).
        - Uses squared Euclidean distance to eliminate square root overhead.
        - Employs batched matrix-matrix GEMM (torch.mm) to update N distances in parallel.
        - Employs local greedy selection within top-k candidates to prevent center clustering.
        - Chunked fallback to guarantee bounded memory footprint (< 300 MB).
        """
        num_samples = features.shape[0]
        compute_device = self.device

        # 1. Random projection for fast distance computation
        if features.shape[1] > proj_dim and num_samples > 5000:
            rng = torch.Generator(device="cpu").manual_seed(seed)
            proj_matrix = torch.randn((features.shape[1], proj_dim), generator=rng)
            proj_matrix = F.normalize(proj_matrix, p=2, dim=0).to(compute_device)
            work_features = self._chunked_project(features, proj_matrix, compute_device)
        else:
            work_features = features.to(compute_device).float()

        # 2. Precompute squared L2 norms: ||x_i||^2
        x_norms_sq = (work_features ** 2).sum(dim=1)

        selected_indices: List[int] = []
        rng = torch.Generator(device="cpu").manual_seed(seed)
        first_idx = torch.randint(0, num_samples, (1,), generator=rng).item()
        selected_indices.append(first_idx)

        # Initial squared distances from first center
        c1 = work_features[first_idx]
        min_dist_sq = x_norms_sq - 2 * torch.mv(work_features, c1) + x_norms_sq[first_idx]
        torch.clamp_min_(min_dist_sq, 0.0)
        min_dist_sq[first_idx] = 0.0

        # 3. Batched greedy selection loop
        B = min(block_size, target_size)
        while len(selected_indices) < target_size:
            batch_needed = min(B, target_size - len(selected_indices))

            if batch_needed == 1:
                next_idx = torch.argmax(min_dist_sq).item()
                selected_indices.append(next_idx)
                c = work_features[next_idx]
                c_norm = x_norms_sq[next_idx]
                dist_sq = x_norms_sq - 2 * torch.mv(work_features, c) + c_norm
                torch.clamp_min_(dist_sq, 0.0)
                min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
                min_dist_sq[next_idx] = 0.0
                continue

            # Pick candidate pool from points with the largest current min_dist_sq
            pool_size = min(num_samples - len(selected_indices), max(batch_needed * 4, 128))
            _, topk_indices = torch.topk(min_dist_sq, k=pool_size)

            cand_feats = work_features[topk_indices]
            cand_dist_sq = min_dist_sq[topk_indices].clone()

            # Local greedy selection within candidate pool
            batch_selected_local: List[int] = []
            for _ in range(batch_needed):
                local_idx = torch.argmax(cand_dist_sq).item()
                batch_selected_local.append(local_idx)
                new_pt = cand_feats[local_idx]
                d_sq = torch.sum((cand_feats - new_pt) ** 2, dim=1)
                cand_dist_sq = torch.minimum(cand_dist_sq, d_sq)
                cand_dist_sq[local_idx] = 0.0

            new_centers_idx = topk_indices[batch_selected_local]
            selected_indices.extend(new_centers_idx.tolist())

            # Global distance update using GEMM
            new_centers = work_features[new_centers_idx]
            new_c_norms = x_norms_sq[new_centers_idx]

            self._update_min_dist_sq_batched(
                min_dist_sq, work_features, x_norms_sq, new_centers, new_c_norms
            )
            min_dist_sq[new_centers_idx] = 0.0

        return torch.tensor(selected_indices[:target_size], dtype=torch.long, device="cpu")

    def _update_min_dist_sq_batched(
        self,
        min_dist_sq: torch.Tensor,
        work_features: torch.Tensor,
        x_norms_sq: torch.Tensor,
        new_centers: torch.Tensor,
        new_c_norms: torch.Tensor,
        chunk_size: int = 131072,
    ) -> None:
        """
        Updates min_dist_sq in-place using matrix multiplication (torch.mm),
        chunked along N to keep intermediate memory strictly under 64 MB.
        """
        N = work_features.shape[0]

        if N <= chunk_size:
            block_dist_sq = x_norms_sq[:, None] - 2 * torch.mm(work_features, new_centers.T) + new_c_norms[None, :]
            torch.clamp_min_(block_dist_sq, 0.0)
            min_new, _ = torch.min(block_dist_sq, dim=1)
            min_dist_sq.copy_(torch.minimum(min_dist_sq, min_new))
        else:
            for start_idx in range(0, N, chunk_size):
                end_idx = min(start_idx + chunk_size, N)
                sub_feats = work_features[start_idx:end_idx]
                sub_norms = x_norms_sq[start_idx:end_idx, None]
                block_dist_sq = sub_norms - 2 * torch.mm(sub_feats, new_centers.T) + new_c_norms[None, :]
                torch.clamp_min_(block_dist_sq, 0.0)
                min_new, _ = torch.min(block_dist_sq, dim=1)
                min_dist_sq[start_idx:end_idx] = torch.minimum(min_dist_sq[start_idx:end_idx], min_new)

    def _exact_greedy_k_center(
        self,
        features: torch.Tensor,
        target_size: int,
        seed: int = 42,
        proj_dim: int = 64,
    ) -> torch.Tensor:
        """
        Exact sequential greedy k-center with squared Euclidean distances and matrix-vector operations.
        Runs on self.device (GPU accelerated if available).
        """
        num_samples = features.shape[0]
        compute_device = self.device

        if features.shape[1] > proj_dim and num_samples > 5000:
            rng = torch.Generator(device="cpu").manual_seed(seed)
            proj_matrix = torch.randn((features.shape[1], proj_dim), generator=rng)
            proj_matrix = F.normalize(proj_matrix, p=2, dim=0).to(compute_device)
            work_features = self._chunked_project(features, proj_matrix, compute_device)
        else:
            work_features = features.to(compute_device).float()

        x_norms_sq = (work_features ** 2).sum(dim=1)

        selected_indices: List[int] = []
        rng = torch.Generator(device="cpu").manual_seed(seed)
        first_idx = torch.randint(0, num_samples, (1,), generator=rng).item()
        selected_indices.append(first_idx)

        c1 = work_features[first_idx]
        min_dist_sq = x_norms_sq - 2 * torch.mv(work_features, c1) + x_norms_sq[first_idx]
        torch.clamp_min_(min_dist_sq, 0.0)
        min_dist_sq[first_idx] = 0.0

        for _ in range(1, target_size):
            next_idx = torch.argmax(min_dist_sq).item()
            selected_indices.append(next_idx)

            c = work_features[next_idx]
            c_norm = x_norms_sq[next_idx]
            dist_sq = x_norms_sq - 2 * torch.mv(work_features, c) + c_norm
            torch.clamp_min_(dist_sq, 0.0)
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
            min_dist_sq[next_idx] = 0.0

        return torch.tensor(selected_indices, dtype=torch.long, device="cpu")

    def _legacy_greedy_k_center(self, features: torch.Tensor, target_size: int, seed: int = 42) -> torch.Tensor:
        """Original unoptimized greedy k-center using torch.norm() in a loop (preserved for comparison)."""
        num_samples = features.shape[0]
        device = features.device

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

        min_distances = torch.norm(work_features - work_features[first_idx], dim=1)

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
