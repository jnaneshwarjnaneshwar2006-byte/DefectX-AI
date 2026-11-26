import os
import sys
import json
import argparse
from typing import Dict, Any

from src.utils import load_config
from src.dataset_adapter import get_dataset_adapter, BaseDatasetAdapter, DatasetInspectionReport


def print_human_report(report: DatasetInspectionReport):
    print("\n" + "=" * 65)
    print("          DEFECTX AI - DATASET INSPECTION REPORT")
    print("=" * 65)
    print(f" Target Path:            {report.root_path}")
    print(f" Detected Format:        {report.detected_format.upper()}")
    print(f" Total Images Inspected: {report.total_images}")
    print(f" Valid Readable Images:  {report.valid_images}")
    print(f" Corrupted Files:        {len(report.corrupt_images)}")
    print(f" Missing References:     {len(report.missing_images)}")
    print(f" Ground-Truth Masks:     {report.mask_count}")
    print("-" * 65)

    print(" Image Formats:")
    if report.image_formats:
        for fmt, count in sorted(report.image_formats.items()):
            pct = (count / max(1, report.valid_images)) * 100
            print(f"   - {fmt:<10}: {count:>4} files ({pct:5.1f}%)")
    else:
        print("   (None found)")

    print("\n Image Resolutions (Dimensions):")
    if report.image_dimensions:
        for dim, count in sorted(report.image_dimensions.items(), key=lambda x: -x[1]):
            pct = (count / max(1, report.valid_images)) * 100
            print(f"   - {dim:<12}: {count:>4} images ({pct:5.1f}%)")
    else:
        print("   (None found)")

    print("\n Color Channel Modes:")
    if report.color_modes:
        for mode, count in sorted(report.color_modes.items()):
            print(f"   - {mode:<8}: {count} files")
    else:
        print("   (None found)")

    print("-" * 65)
    print(" Sample Distribution:")
    print(f"   - Normal / Good Samples:   {report.normal_image_count}")
    print(f"   - Defective Samples:       {report.defective_image_count}")
    print(f"   - Unlabelled Samples:      {report.unlabelled_image_count}")

    print("\n Available Defect Categories:")
    if report.defect_categories:
        for d in report.defect_categories:
            mask_info = report.ground_truth_mask_availability.get(d, {})
            if mask_info:
                print(f"   - {d:<18} (Samples: {mask_info.get('defect_images', '?')}, Masks: {mask_info.get('masks_found', 0)}/{mask_info.get('defect_images', '?')} - {mask_info.get('coverage_pct', 0)}% coverage)")
            else:
                print(f"   - {d}")
    else:
        print("   (No specific defect subcategories detected)")

    print("-" * 65)
    print(" Folder Structure Summary:")
    for folder, count in sorted(report.folder_structure.items()):
        print(f"   - {folder:<35}: {count:>4} files")

    print("-" * 65)
    print(" Integrity & Validation Status:")
    if report.corrupt_images:
        print(f"   [FAIL] {len(report.corrupt_images)} corrupt images detected:")
        for c in report.corrupt_images[:5]:
            print(f"          * {c['path']}: {c['error']}")
        if len(report.corrupt_images) > 5:
            print(f"          ... and {len(report.corrupt_images) - 5} more.")
    else:
        print("   [PASS] All inspected image files are valid and readable.")

    if report.warnings:
        print("\n Warnings:")
        for w in report.warnings:
            print(f"   [!] {w}")

    if report.errors:
        print("\n Errors:")
        for e in report.errors:
            print(f"   [X] {e}")

    print("=" * 65 + "\n")


def inspect_dataset_cli(args):
    # Load config if present, or use defaults
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {"dataset": {}}

    # CLI args override YAML config
    if args.dataset_root:
        config["dataset"]["root_dir"] = args.dataset_root
    if args.category is not None:
        config["dataset"]["category"] = args.category
    if args.adapter:
        config["dataset"]["adapter"] = args.adapter

    root_dir = config["dataset"].get("root_dir", "data/mvtec")
    category = config["dataset"].get("category", "bottle")

    adapter: BaseDatasetAdapter = get_dataset_adapter(config)
    report = adapter.inspect(root_dir=root_dir, category=category)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_human_report(report)

    if args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Detailed inspection JSON saved to: {args.output_report}")

    # Return exit code: 1 if errors, 0 otherwise
    if report.errors or report.corrupt_images:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and validate dataset for DEFECTX AI Engine.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root path of dataset (overrides YAML)")
    parser.add_argument("--category", type=str, default=None, help="Target category (or empty for flat datasets)")
    parser.add_argument("--adapter", type=str, default=None, choices=["auto", "mvtec", "visa", "custom", "folder"], help="Adapter format")
    parser.add_argument("--format", dest="adapter", type=str, choices=["auto", "mvtec", "visa", "custom", "folder"], help="Alias for --adapter")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--output-report", type=str, default=None, help="Path to write JSON inspection report")
    args = parser.parse_args()
    inspect_dataset_cli(args)
