from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PatchFeatureExtractor(nn.Module):
    """
    Extracts multi-scale patch embeddings from intermediate layers of a pretrained ResNet-18.
    Follows the PatchCore methodology:
    - Hooks intermediate layers (layer2, layer3).
    - Applies local neighborhood aggregation (AvgPool2d).
    - Bilinearly aligns features to match spatial dimensions.
    - Concatenates multi-scale representations into patch tokens.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: List[str] = None,
        patch_pooling_kernel: int = 3,
        patch_pooling_stride: int = 1,
        patch_pooling_padding: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.layers = layers or ["layer2", "layer3"]

        if backbone_name != "resnet18":
            raise ValueError(f"Currently supported backbone is 'resnet18', got '{backbone_name}'")

        # Load pretrained ResNet-18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze all parameters
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        self.backbone = backbone.to(self.device)

        # Local neighborhood pooling
        self.patch_pool = nn.AvgPool2d(
            kernel_size=patch_pooling_kernel,
            stride=patch_pooling_stride,
            padding=patch_pooling_padding
        )

        self._features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name: str):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        for name, module in self.backbone.named_children():
            if name in self.layers:
                self._hooks.append(module.register_forward_hook(get_hook(name)))

    def to_device(self, device: torch.device):
        self.device = device
        self.backbone = self.backbone.to(device)
        return self

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extracts patch embeddings from input batch.
        Args:
            x: Tensor of shape (B, 3, H, W)
        Returns:
            patch_embeddings: Tensor of shape (B, N_patches, C_total)
            patch_grid_shape: Tuple (H_patch, W_patch)
        """
        self._features.clear()
        x = x.to(self.device)
        _ = self.backbone(x)

        extracted_maps = []
        # Target spatial size is the largest feature map among chosen layers (usually layer2: 28x28)
        target_size = None

        for layer_name in self.layers:
            feat = self._features[layer_name]
            # Apply local neighborhood pooling
            feat = self.patch_pool(feat)
            if target_size is None:
                target_size = (feat.shape[2], feat.shape[3])
            else:
                # Bilinearly interpolate deeper layers (layer3: 14x14) to match target_size
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                )
            extracted_maps.append(feat)

        # Concatenate multi-scale representations along channel dimension
        # (B, C1 + C2, H_patch, W_patch) -> (B, 384, 28, 28) for 224x224 input
        concat_features = torch.cat(extracted_maps, dim=1)
        B, C, H_p, W_p = concat_features.shape

        # Reshape to (B, H_p * W_p, C)
        patch_embeddings = concat_features.permute(0, 2, 3, 1).reshape(B, H_p * W_p, C)

        return patch_embeddings, (H_p, W_p)
