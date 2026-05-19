"""
scripts/diagnose_orientation.py — Class Orientation Diagnostic
==============================================================
Project : We ♥ Digital Humanities — Emblem Gesture Recognition

Purpose:
    Display a grid of raw test images (no transforms, no augmentation)
    grouped by class. This lets you visually compare:
      - What orientation each class was annotated in during training
      - Whether upside_down_heart images match what your webcam produces
      - Whether the bounding box crop framing looks correct per class

    Run this BEFORE making any further changes to demo.py.
    Interpret the grid, then report back what you see.

Output:
    outputs/figures/orientation_diagnostic.png
    (also opens interactively if a display is available)

Usage:
    python scripts/diagnose_orientation.py

Layout:
    One row per class (4 rows total).
    Up to N_SAMPLES columns per row (default 5).
    Each image is cropped to its YOLO bounding box — exactly what the
    model sees during training — so the comparison to webcam crops is direct.
"""

# ── 0. Path setup ───────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 1. Imports ──────────────────────────────────────────────────────────────────
import random
import matplotlib
matplotlib.use("Agg")   # headless rendering — no display required on Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from src.utils import (
    load_config,
    get_project_paths,
    get_image_paths,
    get_label_path,
    read_annotations,
    yolo_to_corners,
    Colours as Col,
)

# ── 2. Config ───────────────────────────────────────────────────────────────────
N_SAMPLES   = 5      # images to show per class (columns)
RANDOM_SEED = 42     # reproducible sampling
IMG_SIZE    = 224    # display size per cell in pixels (does not affect saved file)


def load_and_crop(image_path: Path, label_path: Path, img_w: int, img_h: int):
    """
    Load one image and crop it to its YOLO bounding box annotation.

    This replicates exactly what data_loader.py does during training:
    the model never sees the full image — it sees the bounding box crop.
    Showing these crops (not the full images) makes the orientation
    comparison to webcam crops meaningful and direct.

    Args:
        image_path : Path to the .jpg/.png image file
        label_path : Path to the matching YOLO .txt annotation file
        img_w      : image width in pixels
        img_h      : image height in pixels

    Returns:
        crop : RGB numpy array of the bounding box region, or
               the full image as fallback if annotation is missing
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    img_h_actual, img_w_actual = img_bgr.shape[:2]

    # Read YOLO annotation — returns list of (class_id, cx, cy, w, h) tuples
    annotations = read_annotations(label_path)

    if annotations:
        # Use the first annotation (single-label images after cleaning)
        class_id, bbox = annotations[0]
        cx, cy, bw, bh = bbox

        # yolo_to_corners returns normalised floats; scale to pixel integers
        nx1, ny1, nx2, ny2 = yolo_to_corners(cx, cy, bw, bh)
        x1 = int(nx1 * img_w_actual)
        y1 = int(ny1 * img_h_actual)
        x2 = int(nx2 * img_w_actual)
        y2 = int(ny2 * img_h_actual)

        # Clamp to image boundaries (safety guard)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w_actual, x2)
        y2 = min(img_h_actual, y2)

        crop_bgr = img_bgr[y1:y2, x1:x2]
    else:
        # No annotation found — show full image as fallback
        crop_bgr = img_bgr

    # Convert BGR (OpenCV) → RGB (matplotlib)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def main():
    # ── Load config and paths ──────────────────────────────────────────────────
    cfg   = load_config()
    paths = get_project_paths(cfg)

    class_names = cfg["dataset"]["class_names"]   # ordered list of 4 classes
    test_dir    = Path(paths["split_dir"]) / "test"
    images_dir  = test_dir / "images"
    labels_dir  = test_dir / "labels"
    figures_dir = Path(paths["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"{Col.RED}Test images directory not found: {images_dir}{Col.RESET}")
        sys.exit(1)

    print(f"{Col.CYAN}Loading test images from: {images_dir}{Col.RESET}")

    # ── Group image paths by class ─────────────────────────────────────────────
    # Filenames are Roboflow hashes — only some classes have a name prefix.
    # The reliable source of class identity is the YOLO label file, which
    # contains the integer class index on each annotation line.
    # We read each image's label file and use the class index to group it.
    class_images: dict[str, list[Path]] = {name: [] for name in class_names}

    all_image_paths = get_image_paths(images_dir)

    for img_path in all_image_paths:
        label_path  = get_label_path(img_path, labels_dir)
        annotations = read_annotations(label_path)

        if not annotations:
            # No label found — skip this image
            continue

        # Use the class index from the first annotation line
        class_idx = annotations[0][0]   # tuple is (class_id, cx, cy, w, h)

        if 0 <= class_idx < len(class_names):
            class_name = class_names[class_idx]
            class_images[class_name].append(img_path)

    # ── Build the grid ─────────────────────────────────────────────────────────
    n_rows = len(class_names)
    n_cols = N_SAMPLES

    # figsize: each cell ~2.5 inches, plus left margin for row labels
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.8),
    )

    # Ensure axes is always 2D even with 1 row or 1 col
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    random.seed(RANDOM_SEED)

    for row_idx, class_name in enumerate(class_names):
        img_paths = class_images[class_name]

        if not img_paths:
            print(f"{Col.RED}  Warning: no test images found for {class_name}{Col.RESET}")

        # Sample up to N_SAMPLES images; pad with None if fewer exist
        sample = random.sample(img_paths, min(N_SAMPLES, len(img_paths)))
        sample += [None] * (N_SAMPLES - len(sample))   # pad if < N_SAMPLES

        for col_idx, img_path in enumerate(sample):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            if img_path is None:
                # Empty cell — class had fewer than N_SAMPLES test images
                ax.set_facecolor("#1a1a1a")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="grey", fontsize=9, transform=ax.transAxes)
                continue

            # Get matching label file for this image
            label_path = get_label_path(img_path, labels_dir)

            # Load and crop to bounding box (mirrors training pipeline exactly)
            crop_rgb = load_and_crop(img_path, label_path,
                                     img_w=IMG_SIZE, img_h=IMG_SIZE)

            if crop_rgb is None:
                ax.set_facecolor("#1a1a1a")
                ax.text(0.5, 0.5, "unreadable", ha="center", va="center",
                        color="red", fontsize=8, transform=ax.transAxes)
                continue

            ax.imshow(crop_rgb)

            # Show filename (shortened) as subtitle under each image
            ax.set_title(img_path.name[:22], fontsize=7, color="#cccccc", pad=2)

        # Row label on the left — class name in bold
        axes[row_idx][0].set_ylabel(
            class_name,
            fontsize=10,
            fontweight="bold",
            color="white",
            rotation=0,
            labelpad=120,
            va="center",
        )

    # ── Style and save ─────────────────────────────────────────────────────────
    fig.patch.set_facecolor("#111111")   # dark background — easier to inspect crops

    fig.suptitle(
        "Orientation Diagnostic — Bounding Box Crops by Class (test set)\n"
        "Compare each row to what your webcam produces for the same gesture",
        fontsize=11,
        color="white",
        y=1.01,
    )

    plt.tight_layout()

    save_path = figures_dir / "orientation_diagnostic.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"{Col.GREEN}Diagnostic grid saved to: {save_path}{Col.RESET}")
    print()
    print("Next steps:")
    print("  1. Open the saved PNG and inspect each row carefully")
    print("  2. For upside_down_heart: does the gesture point downward as expected?")
    print("  3. For traditional-heart / traditional-heart-2: do both hands appear")
    print("     in the crop, or just one hand?")
    print("  4. Compare each row's crop framing to what the webcam demo produces")
    print("  5. Report back what you see — the fix depends on this observation")


if __name__ == "__main__":
    main()