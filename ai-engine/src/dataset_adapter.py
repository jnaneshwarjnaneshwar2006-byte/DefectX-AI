import os
import glob
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .dataset import MVTecNormalDataset, MVTecTestDataset


VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def validate_image_file(image_path: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates image file existence, integrity, format, dimensions, and channel mode.
    Returns:
        (is_valid, error_message, metadata_dict)
    """
    if not os.path.exists(image_path):
        return False, f"File does not exist: '{image_path}'", None

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        return False, f"Unsupported image extension '{ext}'. Allowed extensions: {', '.join(VALID_IMAGE_EXTENSIONS)}", None

    try:
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return False, f"Zero-byte empty file: '{image_path}'", None

        with Image.open(image_path) as img:
            img.verify()  # Verifies file header and integrity

        # Re-open to read dimensions and mode safely
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            fmt = img.format or os.path.splitext(image_path)[1].lstrip(".").upper()

        metadata = {
            "path": os.path.abspath(image_path),
            "format": fmt,
            "dimensions": f"{width}x{height}",
            "width": width,
            "height": height,
            "mode": mode,
            "size_bytes": file_size,
        }
        return True, None, metadata

    except Exception as e:
        return False, f"Corrupted image '{image_path}': {str(e)}", None


@dataclass
class DatasetInspectionReport:
    """Structured report of dataset inspection."""
    root_path: str
    detected_format: str
    total_images: int = 0
    valid_images: int = 0
    corrupt_images: List[Dict[str, str]] = field(default_factory=list)
    missing_images: List[str] = field(default_factory=list)
    image_formats: Dict[str, int] = field(default_factory=dict)
    image_dimensions: Dict[str, int] = field(default_factory=dict)
    color_modes: Dict[str, int] = field(default_factory=dict)
    folder_structure: Dict[str, Any] = field(default_factory=dict)
    defect_categories: List[str] = field(default_factory=list)
    normal_image_count: int = 0
    defective_image_count: int = 0
    unlabelled_image_count: int = 0
    mask_count: int = 0
    ground_truth_mask_availability: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseDatasetAdapter(ABC):
    """Abstract Base Class for Dataset Adapters."""

    @abstractmethod
    def inspect(self, root_dir: str, category: Optional[str] = None) -> DatasetInspectionReport:
        """Inspects folder structure, image formats, integrity, defect types and ground truth masks."""
        pass

    @abstractmethod
    def load_splits(
        self,
        root_dir: str,
        category: Optional[str] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Loads:
        1. train_normal: Dataset of normal training images.
        2. val_normal: Held-out normal validation images (strictly for threshold fitting).
        3. test_dataset: Test images with ground truth masks if available.

        Strict zero-leakage invariant: Test data is never touched for training or threshold fitting.
        """
        pass


class MVTecAdapter(BaseDatasetAdapter):
    """
    Adapter for standard MVTec AD format:
    <root>/<category>/train/good/
    <root>/<category>/test/<defect_type>/
    <root>/<category>/ground_truth/<defect_type>/
    """
    def __init__(self, normal_class_name: str = "good"):
        self.normal_class_name = normal_class_name

    def inspect(self, root_dir: str, category: Optional[str] = None) -> DatasetInspectionReport:
        category = category or "bottle"
        target_dir = os.path.join(root_dir, category) if category else root_dir
        report = DatasetInspectionReport(root_path=os.path.abspath(target_dir), detected_format="mvtec")

        if not os.path.exists(target_dir):
            report.errors.append(f"Directory not found: '{target_dir}'")
            return report

        # Walk through directory
        folder_summary = {}
        all_image_paths = []
        mask_paths = []

        for root, dirs, files in os.walk(target_dir):
            rel_path = os.path.relpath(root, target_dir)
            img_files = [f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
            if img_files:
                folder_summary[rel_path] = len(img_files)
                for f in img_files:
                    full_p = os.path.join(root, f)
                    if "ground_truth" in rel_path:
                        mask_paths.append(full_p)
                    else:
                        all_image_paths.append((full_p, rel_path))

        report.folder_structure = folder_summary
        report.total_images = len(all_image_paths)
        report.mask_count = len(mask_paths)

        # Inspect and validate images
        for img_path, rel_dir in all_image_paths:
            is_valid, err, meta = validate_image_file(img_path)
            if not is_valid:
                report.corrupt_images.append({"path": img_path, "error": err})
                continue

            report.valid_images += 1
            report.image_formats[meta["format"]] = report.image_formats.get(meta["format"], 0) + 1
            report.image_dimensions[meta["dimensions"]] = report.image_dimensions.get(meta["dimensions"], 0) + 1
            report.color_modes[meta["mode"]] = report.color_modes.get(meta["mode"], 0) + 1

            # Determine defect category and labels
            parts = rel_dir.replace("\\", "/").split("/")
            if "train" in parts:
                report.normal_image_count += 1
            elif "test" in parts:
                defect_name = parts[-1]
                if defect_name.lower() == self.normal_class_name:
                    report.normal_image_count += 1
                else:
                    report.defective_image_count += 1
                    if defect_name not in report.defect_categories:
                        report.defect_categories.append(defect_name)

        # Mask availability verification
        test_dir = os.path.join(target_dir, "test")
        gt_dir = os.path.join(target_dir, "ground_truth")
        mask_stats = {}
        for d in report.defect_categories:
            d_test_imgs = glob.glob(os.path.join(test_dir, d, "*.*"))
            matched_masks = 0
            for img in d_test_imgs:
                base = os.path.splitext(os.path.basename(img))[0]
                possible = [
                    os.path.join(gt_dir, d, f"{base}_mask.png"),
                    os.path.join(gt_dir, d, f"{base}.png"),
                ]
                if any(os.path.exists(p) for p in possible):
                    matched_masks += 1
            mask_stats[d] = {
                "defect_images": len(d_test_imgs),
                "masks_found": matched_masks,
                "coverage_pct": round(matched_masks / max(1, len(d_test_imgs)) * 100, 1)
            }
        report.ground_truth_mask_availability = mask_stats

        # Inconsistency warnings
        if report.corrupt_images:
            report.warnings.append(f"Found {len(report.corrupt_images)} corrupt/unreadable images.")
        if not report.defect_categories:
            report.warnings.append("No defective test categories found in test/ directory.")

        return report

    def load_splits(
        self,
        root_dir: str,
        category: Optional[str] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[Dataset, Dataset, Dataset]:
        from .dataset import load_mvtec_splits
        return load_mvtec_splits(
            dataset_root=root_dir,
            category=category or "bottle",
            val_ratio=val_ratio,
            seed=seed,
            image_size=image_size,
        )


class GenericFolderAdapter(BaseDatasetAdapter):
    """
    Flexible adapter for custom organiser datasets.
    Supports layouts such as:
    - Root containing 'train/' and 'test/' subdirectories
    - Subdirectories named 'normal', 'good', 'ok' vs 'defect', 'anomaly', 'ng'
    - Or category folders containing 'normal' and custom defect subfolders
    """
    def __init__(
        self,
        train_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
        normal_class_names: Optional[List[str]] = None,
    ):
        self.train_dir_name = train_dir
        self.test_dir_name = test_dir
        self.gt_dir_name = ground_truth_dir
        self.normal_names = [n.lower() for n in (normal_class_names or ["good", "normal", "ok", "pass", "non-defective"])]

    def _is_normal_name(self, name: str) -> bool:
        return name.lower() in self.normal_names

    def inspect(self, root_dir: str, category: Optional[str] = None) -> DatasetInspectionReport:
        target_dir = os.path.join(root_dir, category) if category else root_dir
        report = DatasetInspectionReport(root_path=os.path.abspath(target_dir), detected_format="generic_folder")

        if not os.path.exists(target_dir):
            report.errors.append(f"Directory not found: '{target_dir}'")
            return report

        folder_summary = {}
        all_image_paths = []
        mask_paths = []

        for root, dirs, files in os.walk(target_dir):
            rel_path = os.path.relpath(root, target_dir)
            img_files = [f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
            if img_files:
                folder_summary[rel_path] = len(img_files)
                for f in img_files:
                    full_p = os.path.join(root, f)
                    if any(m in rel_path.lower() for m in ["mask", "ground_truth", "gt"]):
                        mask_paths.append(full_p)
                    else:
                        all_image_paths.append((full_p, rel_path))

        report.folder_structure = folder_summary
        report.total_images = len(all_image_paths)
        report.mask_count = len(mask_paths)

        # Validate images
        for img_path, rel_dir in all_image_paths:
            is_valid, err, meta = validate_image_file(img_path)
            if not is_valid:
                report.corrupt_images.append({"path": img_path, "error": err})
                continue

            report.valid_images += 1
            report.image_formats[meta["format"]] = report.image_formats.get(meta["format"], 0) + 1
            report.image_dimensions[meta["dimensions"]] = report.image_dimensions.get(meta["dimensions"], 0) + 1
            report.color_modes[meta["mode"]] = report.color_modes.get(meta["mode"], 0) + 1

            # Determine defect class name from path
            parts = rel_dir.replace("\\", "/").split("/")
            last_part = parts[-1]
            if any(self._is_normal_name(p) for p in parts):
                report.normal_image_count += 1
            else:
                report.defective_image_count += 1
                if last_part not in report.defect_categories and last_part != ".":
                    report.defect_categories.append(last_part)

        report.ground_truth_mask_availability = {
            "total_masks": len(mask_paths),
            "status": "available" if mask_paths else "none_found"
        }

        if report.corrupt_images:
            report.warnings.append(f"Detected {len(report.corrupt_images)} corrupted image files.")
        if report.normal_image_count == 0:
            report.errors.append("No normal/good images identified. Ensure normal folders are named 'good', 'normal', or 'ok'.")

        return report

    def load_splits(
        self,
        root_dir: str,
        category: Optional[str] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[Dataset, Dataset, Dataset]:
        target_dir = os.path.join(root_dir, category) if category else root_dir
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Target dataset directory not found: '{target_dir}'")

        # Discover train normal paths and test samples
        train_normal_paths = []
        test_samples = []

        # Check if explicit train/ and test/ directories exist
        subdirs = [d.lower() for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]

        if "train" in subdirs:
            train_root = os.path.join(target_dir, "train")
            for root, _, files in os.walk(train_root):
                rel = os.path.relpath(root, train_root).lower()
                for f in files:
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        p = os.path.join(root, f)
                        # Train should only collect normal images
                        if any(self._is_normal_name(part) for part in rel.replace("\\", "/").split("/")) or rel == ".":
                            train_normal_paths.append(p)

            # Test split
            if "test" in subdirs:
                test_root = os.path.join(target_dir, "test")
                for root, _, files in os.walk(test_root):
                    rel = os.path.relpath(root, test_root)
                    parts = rel.replace("\\", "/").split("/")
                    d_name = parts[0] if parts[0] != "." else "test_sample"
                    is_good = any(self._is_normal_name(p) for p in parts)
                    for f in files:
                        if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                            p = os.path.join(root, f)
                            test_samples.append({
                                "image_path": p,
                                "label": 0 if is_good else 1,
                                "defect_type": "good" if is_good else d_name,
                                "mask_path": None,
                            })
        else:
            # Flat class-folder structure: e.g. target_dir/normal, target_dir/defect1, target_dir/defect2
            for d in os.listdir(target_dir):
                d_path = os.path.join(target_dir, d)
                if not os.path.isdir(d_path):
                    continue
                is_normal = self._is_normal_name(d)
                img_files = [os.path.join(d_path, f) for f in os.listdir(d_path) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
                if is_normal:
                    # Normal images pool: split into train and test good
                    train_normal_paths.extend(img_files)
                else:
                    # Defect images belong strictly to test
                    for img_p in img_files:
                        test_samples.append({
                            "image_path": img_p,
                            "label": 1,
                            "defect_type": d,
                            "mask_path": None,
                        })

        if not train_normal_paths:
            raise ValueError(f"No normal training images found in '{target_dir}'. Check folder names.")

        # Deterministic split of train_normal into train_normal and val_normal (held-out for threshold fitting)
        train_normal_paths.sort()
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_normal_paths), generator=rng).tolist()
        num_val = max(1, int(len(train_normal_paths) * val_ratio))
        val_indices = set(perm[:num_val])

        final_train_paths = [train_normal_paths[i] for i in range(len(train_normal_paths)) if i not in val_indices]
        final_val_paths = [train_normal_paths[i] for i in range(len(train_normal_paths)) if i in val_indices]

        # If test_samples didn't have normal images, add held-out test normal
        has_test_normals = any(s["label"] == 0 for s in test_samples)
        if not has_test_normals and len(final_val_paths) > 1:
            # Allocate a portion of val normals to test_samples for FPR evaluation
            test_good_path = final_val_paths.pop()
            test_samples.append({
                "image_path": test_good_path,
                "label": 0,
                "defect_type": "good",
                "mask_path": None,
            })

        train_ds = MVTecNormalDataset(final_train_paths, image_size=image_size)
        val_ds = MVTecNormalDataset(final_val_paths, image_size=image_size)
        test_ds = MVTecTestDataset(test_samples, image_size=image_size)

        return train_ds, val_ds, test_ds


class VisAAdapter(BaseDatasetAdapter):
    """
    Adapter for the official VisA (Visual Anomaly) benchmark dataset.
    Reads splits from split_csv/1cls.csv (or specified CSV manifest).
    Validates images in <root>/<object>/Data/Images/Normal/ and Anomaly/
    Validates masks in <root>/<object>/Data/Masks/Anomaly/
    """
    ALL_CATEGORIES = [
        "candle", "capsules", "cashew", "chewinggum",
        "fryum", "macaroni1", "macaroni2",
        "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]

    def __init__(self, split_csv: Optional[str] = None):
        self.split_csv_rel = split_csv

    def _locate_csv(self, root_dir: str) -> str:
        candidates = []
        if self.split_csv_rel:
            candidates.append(os.path.normpath(self.split_csv_rel))
            candidates.append(os.path.normpath(os.path.join(root_dir, self.split_csv_rel)))
        candidates.extend([
            os.path.normpath(os.path.join(root_dir, "split_csv", "1cls.csv")),
            os.path.normpath(os.path.join(root_dir, "1cls.csv")),
        ])
        for p in candidates:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            f"Could not locate VisA split CSV in '{root_dir}'. Checked paths: {candidates}"
        )

    def inspect(self, root_dir: str, category: Optional[str] = None) -> DatasetInspectionReport:
        report = DatasetInspectionReport(root_path=os.path.abspath(root_dir), detected_format="visa")
        if not os.path.exists(root_dir):
            report.errors.append(f"Directory not found: '{root_dir}'")
            return report

        try:
            csv_path = self._locate_csv(root_dir)
        except FileNotFoundError as e:
            report.errors.append(str(e))
            return report

        if category and category not in self.ALL_CATEGORIES:
            report.warnings.append(
                f"Specified category '{category}' is not among official VisA categories: {', '.join(self.ALL_CATEGORIES)}"
            )

        rows = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                required_cols = {"object", "split", "label", "image", "mask"}
                if not required_cols.issubset(set(reader.fieldnames or [])):
                    report.errors.append(
                        f"CSV '{csv_path}' missing required columns. Expected {required_cols}, got {reader.fieldnames}"
                    )
                    return report
                for r in reader:
                    if category is None or r["object"] == category:
                        rows.append(r)
        except Exception as e:
            report.errors.append(f"Failed to read CSV '{csv_path}': {str(e)}")
            return report

        report.total_images = len(rows)
        folder_summary: Dict[str, int] = {}
        defect_samples_count = 0
        masks_found = 0

        for r in rows:
            img_rel = r["image"]
            img_full = os.path.normpath(os.path.join(root_dir, img_rel))

            parent_rel = os.path.dirname(img_rel).replace("\\", "/")
            folder_summary[parent_rel] = folder_summary.get(parent_rel, 0) + 1

            if not os.path.exists(img_full):
                report.missing_images.append(img_full)
                continue

            is_valid, err, meta = validate_image_file(img_full)
            if not is_valid:
                report.corrupt_images.append({"path": img_full, "error": err or "Corrupted image"})
                continue

            report.valid_images += 1
            report.image_formats[meta["format"]] = report.image_formats.get(meta["format"], 0) + 1
            report.image_dimensions[meta["dimensions"]] = report.image_dimensions.get(meta["dimensions"], 0) + 1
            report.color_modes[meta["mode"]] = report.color_modes.get(meta["mode"], 0) + 1

            lbl = r["label"].lower()
            if lbl == "normal":
                report.normal_image_count += 1
            elif lbl == "anomaly":
                report.defective_image_count += 1
                defect_samples_count += 1
                if "anomaly" not in report.defect_categories:
                    report.defect_categories.append("anomaly")

                mask_rel = r.get("mask", "").strip()
                if mask_rel:
                    mask_full = os.path.normpath(os.path.join(root_dir, mask_rel))
                    if os.path.exists(mask_full):
                        report.mask_count += 1
                        masks_found += 1
                    else:
                        report.errors.append(f"Mask file does not exist: '{mask_full}'")
                else:
                    report.errors.append(f"Anomaly sample has empty mask field: '{img_rel}'")
            else:
                report.unlabelled_image_count += 1

        report.folder_structure = folder_summary
        if defect_samples_count > 0:
            report.ground_truth_mask_availability["anomaly"] = {
                "defect_images": defect_samples_count,
                "masks_found": masks_found,
                "coverage_pct": round(masks_found / defect_samples_count * 100, 1)
            }

        if report.missing_images:
            report.errors.append(f"{len(report.missing_images)} image files referenced in CSV do not exist on disk.")
        if report.corrupt_images:
            report.warnings.append(f"{len(report.corrupt_images)} images are corrupted or unreadable.")

        return report

    def load_splits(
        self,
        root_dir: str,
        category: Optional[str] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[Dataset, Dataset, Dataset]:
        target_category = category or "candle"
        if target_category not in self.ALL_CATEGORIES:
            raise ValueError(
                f"Unknown VisA category '{target_category}'. Available categories: {', '.join(self.ALL_CATEGORIES)}"
            )

        csv_path = self._locate_csv(root_dir)
        cat_rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["object"] == target_category:
                    cat_rows.append(r)

        if not cat_rows:
            raise ValueError(
                f"No samples found for category '{target_category}' in VisA manifest '{csv_path}'."
            )

        train_rows = [r for r in cat_rows if r["split"].lower() == "train"]
        test_rows = [r for r in cat_rows if r["split"].lower() == "test"]

        # Validate training samples: must be strictly normal (zero leakage, zero anomaly in train)
        for r in train_rows:
            if r["label"].lower() != "normal":
                raise ValueError(
                    f"VisA training split for '{target_category}' contains non-normal sample: {r}"
                )

        train_paths = []
        for r in train_rows:
            img_p = os.path.normpath(os.path.join(root_dir, r["image"]))
            if not os.path.exists(img_p):
                raise FileNotFoundError(f"VisA training image not found: '{img_p}'")
            train_paths.append(img_p)

        # Deterministic split of normal training samples into train_normal and val_normal
        train_paths.sort()
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_paths), generator=rng).tolist()
        num_val = max(1, int(len(train_paths) * val_ratio))
        val_indices = set(indices[:num_val])

        final_train_paths = [train_paths[i] for i in range(len(train_paths)) if i not in val_indices]
        final_val_paths = [train_paths[i] for i in range(len(train_paths)) if i in val_indices]

        train_normal_ds = MVTecNormalDataset(final_train_paths, image_size=image_size)
        val_normal_ds = MVTecNormalDataset(final_val_paths, image_size=image_size)

        # Test samples: both normal and anomaly
        test_samples: List[Dict[str, Any]] = []
        for r in test_rows:
            img_p = os.path.normpath(os.path.join(root_dir, r["image"]))
            if not os.path.exists(img_p):
                raise FileNotFoundError(f"VisA test image not found: '{img_p}'")

            label_str = r["label"].lower()
            if label_str not in ("normal", "anomaly"):
                raise ValueError(f"Invalid label '{r['label']}' in test row: {r}")

            is_anomaly = (label_str == "anomaly")
            label = 1 if is_anomaly else 0
            defect_type = "anomaly" if is_anomaly else "good"

            mask_path = None
            if is_anomaly:
                mask_rel = r.get("mask", "").strip()
                if not mask_rel:
                    raise ValueError(f"Anomalous test sample missing mask reference in CSV: {r}")
                m_p = os.path.normpath(os.path.join(root_dir, mask_rel))
                if not os.path.exists(m_p):
                    raise FileNotFoundError(f"VisA anomaly mask file not found: '{m_p}'")
                mask_path = m_p

            test_samples.append({
                "image_path": img_p,
                "label": label,
                "defect_type": defect_type,
                "mask_path": mask_path,
            })

        test_ds = MVTecTestDataset(test_samples, image_size=image_size)
        return train_normal_ds, val_normal_ds, test_ds


def detect_dataset_format(root_dir: str, category: Optional[str] = None) -> str:
    """
    Auto-detects dataset format from folder structure.
    Returns: 'visa', 'mvtec', or 'generic_folder'.
    """
    # 1. Check if VisA split CSV exists
    if os.path.isfile(os.path.join(root_dir, "split_csv", "1cls.csv")) or os.path.isfile(os.path.join(root_dir, "1cls.csv")):
        return "visa"

    target = os.path.join(root_dir, category) if category else root_dir
    if not os.path.isdir(target):
        # Check subdirectories
        if os.path.isdir(root_dir):
            subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            if category and category in subdirs:
                target = os.path.join(root_dir, category)
            elif "train" in subdirs:
                target = root_dir

    train_good = os.path.join(target, "train", "good")
    if os.path.isdir(train_good):
        return "mvtec"

    return "generic_folder"


def get_dataset_adapter(config: Dict[str, Any]) -> BaseDatasetAdapter:
    """
    Factory creating configured dataset adapter based on configuration.
    """
    ds_cfg = config.get("dataset", {})
    adapter_name = ds_cfg.get("adapter") or ds_cfg.get("dataset_format") or ds_cfg.get("format") or "auto"
    adapter_name = str(adapter_name).lower()
    root_dir = ds_cfg.get("root_dir", "data/mvtec")
    category = ds_cfg.get("category", "bottle")
    normal_name = ds_cfg.get("normal_class_name", "good")

    if adapter_name == "auto":
        detected = detect_dataset_format(root_dir, category)
        if detected == "visa":
            return VisAAdapter(split_csv=ds_cfg.get("split_csv"))
        elif detected == "mvtec":
            return MVTecAdapter(normal_class_name=normal_name)
        else:
            return GenericFolderAdapter(normal_class_names=[normal_name])
    elif adapter_name in ["visa", "vis_a"]:
        return VisAAdapter(split_csv=ds_cfg.get("split_csv"))
    elif adapter_name == "mvtec":
        return MVTecAdapter(normal_class_name=normal_name)
    elif adapter_name in ["folder", "generic_folder", "custom"]:
        return GenericFolderAdapter(
            train_dir=ds_cfg.get("train_dir"),
            test_dir=ds_cfg.get("test_dir"),
            ground_truth_dir=ds_cfg.get("ground_truth_dir"),
            normal_class_names=[normal_name],
        )
    else:
        raise ValueError(f"Unknown dataset adapter: '{adapter_name}'. Choose 'auto', 'mvtec', 'visa', or 'custom'.")
