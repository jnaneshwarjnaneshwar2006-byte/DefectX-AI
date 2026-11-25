import os
import shutil
import tarfile
import urllib.request
import argparse


MVTEC_BOTTLE_URL = "https://www.mydrive.ch/shares/38536/38307f403064434199b42235ec72e218/download/420937413-1629951468/bottle.tar.xz"


def check_disk_space(path: str, min_mb: int = 1000) -> bool:
    """Verifies at least min_mb MB are free on the target drive."""
    total, used, free = shutil.disk_usage(os.path.abspath(path))
    free_mb = free / (1024 * 1024)
    print(f"Available disk space on drive: {free_mb:.1f} MB (Required: {min_mb} MB)")
    return free_mb >= min_mb


def download_mvtec_bottle(dest_dir: str = "data/mvtec", force: bool = False):
    """
    Downloads and extracts official MVTec AD bottle dataset.
    """
    os.makedirs(dest_dir, exist_ok=True)
    target_category_dir = os.path.join(dest_dir, "bottle")

    if os.path.exists(target_category_dir) and not force:
        print(f"Directory '{target_category_dir}' already exists. Use --force to redownload.")
        return

    if not check_disk_space(dest_dir, min_mb=1000):
        raise RuntimeError("Insufficient disk space to download and extract MVTec bottle (~125MB archive, ~300MB uncompressed).")

    archive_path = os.path.join(dest_dir, "bottle.tar.xz")
    print(f"Downloading official MVTec bottle dataset to {archive_path}...")
    try:
        urllib.request.urlretrieve(MVTEC_BOTTLE_URL, archive_path)
        print("Download complete. Extracting archive...")
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=target_category_dir)
        print(f"Extracted MVTec bottle dataset to '{target_category_dir}'.")
        # Remove tar archive to conserve space
        if os.path.exists(archive_path):
            os.remove(archive_path)
    except Exception as e:
        print(f"Failed to download from primary mirror: {e}")
        print("Tip: If downloading from official server requires interactive web confirmation,")
        print("you can run `python scripts/generate_sample_data.py` to immediately test with synthetic industrial samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official MVTec AD bottle category dataset.")
    parser.add_argument("--dest-dir", type=str, default="data/mvtec", help="Destination root directory")
    parser.add_argument("--force", action="store_true", help="Force redownload even if exists")
    args = parser.parse_args()
    download_mvtec_bottle(dest_dir=args.dest_dir, force=args.force)
