import os
import argparse
import numpy as np
import cv2
from PIL import Image


def draw_bottle(
    defect_type: str = "good",
    size: int = 256,
    variation_seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders a synthetic industrial bottle image and corresponding ground truth mask.
    Defect types: 'good', 'broken_large', 'broken_small', 'contamination'.
    """
    rng = np.random.default_rng(variation_seed)

    # Base background (neutral industrial conveyor background)
    img = np.full((size, size, 3), fill_value=220, dtype=np.uint8)
    # Add subtle background texture noise
    noise = rng.normal(0, 3, (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Bottle body geometry
    center_x = size // 2 + int(rng.integers(-2, 3))
    cap_top = int(size * 0.12)
    neck_top = int(size * 0.22)
    neck_bottom = int(size * 0.38)
    body_bottom = int(size * 0.88)

    body_w = int(size * 0.28)
    neck_w = int(size * 0.12)
    cap_w = int(size * 0.14)

    # Draw Bottle Body (Glass color with subtle gradients)
    body_color = (195, 215, 205)
    cv2.rectangle(
        img,
        (center_x - body_w // 2, neck_bottom),
        (center_x + body_w // 2, body_bottom),
        body_color,
        -1
    )
    # Bottle Neck
    cv2.rectangle(
        img,
        (center_x - neck_w // 2, neck_top),
        (center_x + neck_w // 2, neck_bottom),
        (190, 210, 200),
        -1
    )
    # Bottle Cap
    cap_color = (160, 40, 40)
    cv2.rectangle(
        img,
        (center_x - cap_w // 2, cap_top),
        (center_x + cap_w // 2, neck_top),
        cap_color,
        -1
    )

    # Glass highlight reflection
    highlight_x = center_x - body_w // 3
    cv2.line(
        img,
        (highlight_x, neck_bottom + 10),
        (highlight_x, body_bottom - 10),
        (230, 240, 235),
        thickness=3
    )

    mask = np.zeros((size, size), dtype=np.uint8)

    # Introduce defects and generate ground truth masks
    if defect_type == "broken_large":
        # Large fracture on the side of the bottle
        pt1 = (center_x + body_w // 2 - 2, neck_bottom + 30)
        pt2 = (center_x + body_w // 2 - 22, neck_bottom + 55)
        pt3 = (center_x + body_w // 2, neck_bottom + 80)
        pts = np.array([pt1, pt2, pt3, (center_x + body_w // 2 + 10, neck_bottom + 55)], dtype=np.int32)
        cv2.fillPoly(img, [pts], (110, 110, 110))
        cv2.polylines(img, [pts], True, (60, 60, 60), 2)
        cv2.fillPoly(mask, [pts], 255)

    elif defect_type == "broken_small":
        # Small chip or localized crack
        cx = center_x - body_w // 4
        cy = neck_bottom + int(size * 0.25)
        cv2.line(img, (cx - 6, cy - 6), (cx + 6, cy + 6), (40, 40, 40), 2)
        cv2.line(img, (cx + 6, cy + 6), (cx + 10, cy + 2), (40, 40, 40), 2)
        cv2.circle(mask, (cx, cy), 8, 255, -1)

    elif defect_type == "contamination":
        # Foreign dark particulate or stain on bottle glass
        cx = center_x + int(rng.integers(-10, 15))
        cy = neck_bottom + int(rng.integers(20, 60))
        cv2.circle(img, (cx, cy), 9, (35, 30, 30), -1)
        cv2.circle(img, (cx + 4, cy - 3), 4, (50, 45, 45), -1)
        cv2.circle(mask, (cx, cy), 12, 255, -1)

    return img, mask


def generate_sample_dataset(output_dir: str = "data/mvtec", num_train: int = 25):
    """
    Generates standard MVTec AD folder structure and synthetic images for testing.
    """
    category_dir = os.path.join(output_dir, "bottle")
    train_dir = os.path.join(category_dir, "train", "good")
    test_dir = os.path.join(category_dir, "test")
    gt_dir = os.path.join(category_dir, "ground_truth")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "good"), exist_ok=True)

    print(f"Generating synthetic MVTec bottle dataset at: {category_dir}")

    # 1. Train good samples
    for i in range(num_train):
        img, _ = draw_bottle(defect_type="good", variation_seed=100 + i)
        save_path = os.path.join(train_dir, f"{i:03d}.png")
        Image.fromarray(img).save(save_path)
    print(f"Generated {num_train} train/good samples.")

    # 2. Test good samples
    num_test_good = 8
    for i in range(num_test_good):
        img, _ = draw_bottle(defect_type="good", variation_seed=500 + i)
        save_path = os.path.join(test_dir, "good", f"{i:03d}.png")
        Image.fromarray(img).save(save_path)
    print(f"Generated {num_test_good} test/good samples.")

    # 3. Test defect categories with ground truth masks
    defect_types = ["broken_large", "broken_small", "contamination"]
    samples_per_defect = 6

    for d_type in defect_types:
        d_test_dir = os.path.join(test_dir, d_type)
        d_gt_dir = os.path.join(gt_dir, d_type)
        os.makedirs(d_test_dir, exist_ok=True)
        os.makedirs(d_gt_dir, exist_ok=True)

        for i in range(samples_per_defect):
            img, mask = draw_bottle(defect_type=d_type, variation_seed=1000 + i * 37)
            img_path = os.path.join(d_test_dir, f"{i:03d}.png")
            mask_path = os.path.join(d_gt_dir, f"{i:03d}_mask.png")

            Image.fromarray(img).save(img_path)
            Image.fromarray(mask).save(mask_path)

        print(f"Generated {samples_per_defect} test samples for '{d_type}' with ground truth masks.")

    print(f"Sample dataset ready at '{category_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample MVTec bottle dataset for smoke tests and CI.")
    parser.add_argument("--output-dir", type=str, default="data/mvtec", help="Destination root directory")
    parser.add_argument("--num-train", type=int, default=25, help="Number of normal training images")
    args = parser.parse_args()
    generate_sample_dataset(output_dir=args.output_dir, num_train=args.num_train)
