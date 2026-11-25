from typing import Tuple, Optional
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from PIL import Image


class AnomalyMapGenerator:
    """
    Transforms 1D patch distance scores into 2D continuous pixel anomaly maps,
    smooths them with Gaussian filtering, and creates colored heatmap overlays.
    """
    def __init__(
        self,
        gaussian_sigma: float = 4.0,
        colormap: int = cv2.COLORMAP_JET,
        overlay_alpha: float = 0.45,
    ):
        self.gaussian_sigma = gaussian_sigma
        self.colormap = colormap
        self.overlay_alpha = overlay_alpha

    def generate_pixel_maps(
        self,
        patch_scores: torch.Tensor,
        grid_shape: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Converts patch anomaly distances to smoothed full-resolution pixel maps.
        Args:
            patch_scores: Tensor of shape (B, H_p * W_p)
            grid_shape: (H_p, W_p) e.g. (28, 28)
            target_size: (H, W) e.g. (224, 224)
        Returns:
            pixel_maps: numpy array of shape (B, H, W)
        """
        B, _ = patch_scores.shape
        H_p, W_p = grid_shape
        H_t, W_t = target_size

        # Reshape to (B, 1, H_p, W_p)
        grid_scores = patch_scores.reshape(B, 1, H_p, W_p).float()

        # Bilinear upsampling to target pixel resolution
        upsampled = F.interpolate(
            grid_scores,
            size=(H_t, W_t),
            mode="bilinear",
            align_corners=False
        )  # (B, 1, H_t, W_t)

        maps_np = upsampled.squeeze(1).cpu().numpy()

        # Apply Gaussian smoothing to each map
        smoothed_maps = np.zeros_like(maps_np)
        for i in range(B):
            smoothed_maps[i] = gaussian_filter(maps_np[i], sigma=self.gaussian_sigma)

        return smoothed_maps

    def colorize_map(
        self,
        anomaly_map: np.ndarray,
        threshold: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> np.ndarray:
        """
        Maps a 2D float anomaly map (H, W) to an RGB heatmap (H, W, 3) using colormap.
        """
        map_norm = anomaly_map.copy()
        if min_val is None:
            min_val = map_norm.min()
        if max_val is None:
            max_val = map_norm.max()

        if max_val > min_val:
            map_norm = (map_norm - min_val) / (max_val - min_val)
        else:
            map_norm = np.zeros_like(map_norm)

        map_uint8 = np.clip(map_norm * 255.0, 0, 255).astype(np.uint8)
        color_bgr = cv2.applyColorMap(map_uint8, self.colormap)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        return color_rgb

    def create_overlay(
        self,
        original_image: np.ndarray,
        anomaly_map: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Blends colored heatmap over the original RGB image.
        Args:
            original_image: RGB uint8 array (H, W, 3)
            anomaly_map: 2D float array (H, W)
        """
        if alpha is None:
            alpha = self.overlay_alpha

        heatmap_rgb = self.colorize_map(anomaly_map)

        # Resize heatmap if shapes differ
        if heatmap_rgb.shape[:2] != original_image.shape[:2]:
            heatmap_rgb = cv2.resize(
                heatmap_rgb,
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        overlay = cv2.addWeighted(
            original_image.astype(np.uint8),
            1.0 - alpha,
            heatmap_rgb.astype(np.uint8),
            alpha,
            0
        )
        return overlay

    def save_heatmap_and_overlay(
        self,
        original_image_path: str,
        anomaly_map: np.ndarray,
        heatmap_save_path: str,
        overlay_save_path: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Saves heatmap image and blended overlay to disk."""
        import os
        os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)

        heatmap_rgb = self.colorize_map(anomaly_map)
        Image.fromarray(heatmap_rgb).save(heatmap_save_path)

        if overlay_save_path:
            os.makedirs(os.path.dirname(overlay_save_path), exist_ok=True)
            orig_img = np.array(Image.open(original_image_path).convert("RGB"))
            overlay = self.create_overlay(orig_img, anomaly_map)
            Image.fromarray(overlay).save(overlay_save_path)

        return heatmap_save_path, overlay_save_path
