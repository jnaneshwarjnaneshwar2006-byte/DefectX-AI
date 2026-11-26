import os
import glob
from typing import List, Tuple, Optional, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MVTecNormalDataset(Dataset):
    """
    Dataset of normal images for MVTec AD.
    Used for train and validation splits (train/good).
    """
    def __init__(self, image_paths: List[str], image_size: Tuple[int, int] = (224, 224)):
        self.image_paths = sorted(image_paths)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, path


class MVTecTestDataset(Dataset):
    """
    Dataset for MVTec AD evaluation containing both good and defective samples.
    Loads images along with their ground-truth pixel masks when available.
    """
    def __init__(self, samples: List[Dict[str, any]], image_size: Tuple[int, int] = (224, 224)):
        self.samples = samples
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        label = sample["label"]  # 0: normal, 1: anomaly
        defect_type = sample["defect_type"]
        mask_path = sample.get("mask_path")

        image = Image.open(image_path).convert("RGB")
        tensor = self.image_transform(image)

        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            # Robust binarization:
            # Handles VisA masks with {0, 1} (where ToTensor yields ~0.00392 for foreground)
            # as well as MVTec masks with {0, 255} (where ToTensor yields 1.0 for foreground).
            # Any non-zero pixel (> 0.0) is a defect pixel (1.0), background remains 0.0.
            mask_tensor = (mask_tensor > 0.0).float()
        else:
            mask_tensor = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        return {
            "image": tensor,
            "label": label,
            "mask": mask_tensor,
            "defect_type": defect_type,
            "image_path": image_path,
        }


def load_mvtec_splits(
    dataset_root: str,
    category: str = "bottle",
    val_ratio: float = 0.2,
    seed: int = 42,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[MVTecNormalDataset, MVTecNormalDataset, MVTecTestDataset]:
    """
    Splits MVTec AD into:
    1. train_normal: (1 - val_ratio) of train/good images.
    2. val_normal: val_ratio of train/good images (strictly for threshold fitting).
    3. test_dataset: all test images (good + unseen defect categories) with ground-truth masks.

    Ensures zero leakage: val_normal comes solely from train/good. No test images are used for fitting.
    """
    category_dir = os.path.join(dataset_root, category)
    if not os.path.isdir(category_dir):
        raise FileNotFoundError(
            f"MVTec category directory not found: '{category_dir}'. "
            f"Please verify dataset root or run scripts/download_mvtec_bottle.py / scripts/generate_sample_data.py"
        )

    train_good_dir = os.path.join(category_dir, "train", "good")
    if not os.path.isdir(train_good_dir):
        raise FileNotFoundError(f"Missing train/good directory at: '{train_good_dir}'")

    train_images = [
        os.path.join(train_good_dir, f)
        for f in os.listdir(train_good_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    if not train_images:
        raise ValueError(f"No valid images found in: '{train_good_dir}'")

    train_images.sort()
    # Deterministic shuffle for splitting train/good
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_images), generator=rng).tolist()
    num_val = max(1, int(len(train_images) * val_ratio))
    val_indices = set(indices[:num_val])

    train_paths = [train_images[i] for i in range(len(train_images)) if i not in val_indices]
    val_paths = [train_images[i] for i in range(len(train_images)) if i in val_indices]

    train_normal_ds = MVTecNormalDataset(train_paths, image_size=image_size)
    val_normal_ds = MVTecNormalDataset(val_paths, image_size=image_size)

    # Load test split
    test_dir = os.path.join(category_dir, "test")
    gt_dir = os.path.join(category_dir, "ground_truth")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Missing test directory at: '{test_dir}'")

    test_samples: List[Dict[str, any]] = []
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir):
            continue

        is_good = (defect_type.lower() == "good")
        defect_images = sorted(glob.glob(os.path.join(defect_dir, "*.*")))

        for img_path in defect_images:
            if not img_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            mask_path = None
            if not is_good and os.path.isdir(gt_dir):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                possible_masks = [
                    os.path.join(gt_dir, defect_type, f"{base_name}_mask.png"),
                    os.path.join(gt_dir, defect_type, f"{base_name}.png"),
                ]
                for p_mask in possible_masks:
                    if os.path.exists(p_mask):
                        mask_path = p_mask
                        break

            test_samples.append({
                "image_path": img_path,
                "label": 0 if is_good else 1,
                "defect_type": defect_type,
                "mask_path": mask_path,
            })

    test_ds = MVTecTestDataset(test_samples, image_size=image_size)
    return train_normal_ds, val_normal_ds, test_ds


def load_dataset_splits(
    config: Dict[str, any],
    dataset_root: Optional[str] = None,
    category: Optional[str] = None,
    val_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
    dataset_format: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Unified entry point for loading dataset splits using the configured dataset adapter.
    Supports MVTec AD format, VisA benchmark dataset, and custom organizer dataset layouts.
    Strict zero-leakage guarantee: test data is never included in train or threshold validation splits.
    """
    from .dataset_adapter import get_dataset_adapter

    ds_cfg = config.get("dataset", {})
    root = dataset_root or ds_cfg.get("root_dir", "data/mvtec")
    cat = category if category is not None else ds_cfg.get("category", "bottle")
    v_ratio = val_ratio if val_ratio is not None else ds_cfg.get("val_ratio", 0.2)
    s = seed if seed is not None else ds_cfg.get("seed", 42)
    img_size = image_size or tuple(ds_cfg.get("image_size", [224, 224]))

    if dataset_format:
        config.setdefault("dataset", {})["adapter"] = dataset_format

    adapter = get_dataset_adapter(config)
    return adapter.load_splits(
        root_dir=root,
        category=cat,
        val_ratio=v_ratio,
        seed=s,
        image_size=img_size,
    )
