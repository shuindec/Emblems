"""
Visualise Merged Bounding Boxes
================================
Project : Emblems Heart Gesture Recognition
Author  : Joyce Nguyen
Purpose : Draw the merged bounding box on each of the 11 restored
          images so you can visually confirm the annotation covers
          the gesture correctly before using these for training.

          Outputs:
            outputs/figures/merged_boxes/verified_{filename}.jpg
              — one annotated image per restored file

            outputs/figures/merged_boxes_grid.png
              — all 11 images in a single grid (for your report)

Usage
-----
    python scripts/visualise_merged_boxes.py
"""

from pathlib import Path
from collections import defaultdict

import cv2
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\suzuk\emblems\data")
OUTPUT_DIR = Path(r"C:\Users\suzuk\emblems\outputs\figures")
# ─────────────────────────────────────────────

TRAIN_IMGS  = DATA_DIR / "train" / "images"
TRAIN_LBLS  = DATA_DIR / "train" / "labels"
YAML_PATH   = DATA_DIR / "data.yaml"

# The 11 filenames we restored — exactly as they appear in train/
RESTORED_STEMS = [
    "HEART71_png.rf.uUSrOun9jNaVJ5bHThIq",
    "HEART76_png.rf.73y07MF2z44QfHNsxrIJ",
    "HEART79_png.rf.vgoG6Shfq8v2zgzaVsvp",
    "HEART80_png.rf.XUpwl56H1wQ4sgAgKn83",
    "HEART81_png.rf.CrM66k3NxQG1Wfgg3Hk0",
    "HEART82_png.rf.Plqd0EGjQkJUqYbQP8Fk",
    "HEART83_png.rf.d9Nh12vAPpKKuJfgmHUC",
    "HEART91_png.rf.7lQ0whExiYAKyta5pMfQ",
    "images (2)_jpg.rf.QgEvQhaadMiemRoOtLjs",
    "images (3)_jpg.rf.zcP8WRyf9zS1Lbps8pIE",
    "photo12414579856_jpg.rf.ZwzVVdajYwzZ7z4eoZUf",
]

# Colour per class (BGR for OpenCV, RGB for matplotlib)
CLASS_COLOURS_BGR = {
    "2-finger-heart"     : (0, 255, 180),   # teal
    "traditional-heart"  : (100, 220, 50),  # green
    "traditional-heart-2": (0, 165, 255),   # orange
    "upside_down_heart"  : (60, 60, 230),   # red
}
CLASS_COLOURS_RGB = {k: (v[2], v[1], v[0]) for k, v in CLASS_COLOURS_BGR.items()}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_class_names(yaml_path: Path) -> list[str]:
    with open(yaml_path) as f:
        return yaml.safe_load(f)["names"]


def find_image_file(img_dir: Path, stem: str) -> Path | None:
    """
    Find the actual image file for a given stem, regardless of extension.
    Roboflow mixes .jpg and .png so we search by stem name.
    """
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = img_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def read_single_annotation(lbl_path: Path) -> tuple | None:
    """
    Read the single merged annotation from a label file.
    Returns (class_idx, cx, cy, w, h) or None if file is missing/empty.
    """
    if not lbl_path.exists():
        return None
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                return int(parts[0]), float(parts[1]), float(parts[2]), \
                       float(parts[3]), float(parts[4])
    return None


def yolo_to_pixels(cx: float, cy: float, w: float, h: float,
                   img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """
    Convert normalised YOLO box coordinates to absolute pixel coordinates.

    YOLO stores coords normalised (0-1 relative to image size).
    We multiply by pixel dimensions to get actual pixel positions.

    Returns: (x_min, y_min, x_max, y_max) in pixels
    """
    x_min = int((cx - w / 2) * img_w)
    y_min = int((cy - h / 2) * img_h)
    x_max = int((cx + w / 2) * img_w)
    y_max = int((cy + h / 2) * img_h)

    # Clamp to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    return x_min, y_min, x_max, y_max


# ══════════════════════════════════════════════════════════════════════════════
# DRAW SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def draw_annotation(img_path: Path, lbl_path: Path,
                    class_names: list[str], save_dir: Path) -> np.ndarray | None:
    """
    Load an image, draw the merged bounding box + class label on it,
    save the annotated version, and return it (for the grid).

    The box is drawn with:
      - Solid coloured rectangle border (colour = class-specific)
      - Filled label background for readability
      - Confidence text showing this is a MERGED annotation
    """
    # ── load image ──
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ERROR: Could not read {img_path.name}")
        return None

    img_h, img_w = img.shape[:2]

    # ── read annotation ──
    ann = read_single_annotation(lbl_path)
    if ann is None:
        print(f"  ERROR: No annotation found for {img_path.name}")
        return None

    cls_idx, cx, cy, w, h = ann
    class_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
    colour_bgr = CLASS_COLOURS_BGR.get(class_name, (255, 255, 255))

    # ── convert to pixels ──
    x_min, y_min, x_max, y_max = yolo_to_pixels(cx, cy, w, h, img_w, img_h)

    # ── draw bounding box ──
    thickness = max(2, img_w // 300)   # scale thickness to image size
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), colour_bgr, thickness)

    # ── draw label background + text ──
    label_text = f"{class_name} [MERGED]"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, img_w / 1200)
    text_thick = max(1, thickness - 1)

    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, text_thick)
    label_y = max(y_min - 10, text_h + 10)    # keep label inside image top

    # Filled rectangle behind text for readability
    cv2.rectangle(img,
                  (x_min, label_y - text_h - baseline - 4),
                  (x_min + text_w + 4, label_y + baseline),
                  colour_bgr, cv2.FILLED)

    # Text in black over the coloured background
    cv2.putText(img, label_text,
                (x_min + 2, label_y - baseline),
                font, font_scale, (0, 0, 0), text_thick, cv2.LINE_AA)

    # ── also annotate box dimensions in corner ──
    dim_text = f"box: {x_max-x_min}x{y_max-y_min}px  |  img: {img_w}x{img_h}px"
    cv2.putText(img, dim_text,
                (10, img_h - 10),
                font, max(0.4, img_w / 2000), (200, 200, 200), 1, cv2.LINE_AA)

    # ── save individual file ──
    save_path = save_dir / f"verified_{img_path.stem}.jpg"
    cv2.imwrite(str(save_path), img)

    # Return RGB version for matplotlib grid
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# GRID FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def save_grid(annotated_images: list[tuple[str, np.ndarray]],
              save_path: Path) -> None:
    """
    Arrange all annotated images in a grid and save as a single PNG.
    Layout: 4 columns, as many rows as needed.
    Each cell shows the image with its filename as a caption.
    This is useful for your report / poster as a single verification figure.
    """
    n_cols = 4
    n_rows = (len(annotated_images) + n_cols - 1) // n_cols  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4, n_rows * 3.5))
    fig.patch.set_facecolor("#0d0f14")
    fig.suptitle("Merged Bounding Box Verification — 11 Restored Images",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    # Flatten axes for easy indexing regardless of grid shape
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, (name, img_rgb) in enumerate(annotated_images):
        ax = axes_flat[i]
        ax.imshow(img_rgb)
        ax.set_title(name[:30] + ("…" if len(name) > 30 else ""),
                     color="#c8d0e8", fontsize=7, pad=3)
        ax.axis("off")
        ax.set_facecolor("#13161e")

    # Hide unused subplot cells
    for j in range(len(annotated_images), len(axes_flat)):
        axes_flat[j].axis("off")
        axes_flat[j].set_facecolor("#0d0f14")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    class_names  = load_class_names(YAML_PATH)
    save_dir     = OUTPUT_DIR / "merged_boxes"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisualising {len(RESTORED_STEMS)} restored images ...\n")
    print(f"{'─'*60}")

    annotated_images = []   # collect (name, rgb_array) for the grid

    for stem in RESTORED_STEMS:
        img_path = find_image_file(TRAIN_IMGS, stem)
        lbl_path = TRAIN_LBLS / (stem + ".txt")

        if img_path is None:
            print(f"  SKIP  {stem}  — image not found in train/")
            continue

        img_rgb = draw_annotation(img_path, lbl_path, class_names, save_dir)

        if img_rgb is not None:
            # Read the annotation for terminal confirmation
            ann = read_single_annotation(lbl_path)
            if ann:
                cls_idx, cx, cy, w, h = ann
                cls_name = class_names[cls_idx]
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  SKIP  {img_path.name}  — failed to read image for size check")
                    print()
                    continue
                ih, iw = img.shape[:2]
                x_min, y_min, x_max, y_max = yolo_to_pixels(cx, cy, w, h, iw, ih)
                box_w_px = x_max - x_min
                box_h_px = y_max - y_min
                coverage  = (box_w_px * box_h_px) / (iw * ih) * 100

                # Coverage % tells you what fraction of the image the merged box covers
                # Good range: 20-70%. Very high (>80%) = box nearly fills whole image.
                cov_flag = "✓ good" if 15 <= coverage <= 75 else "⚠ check manually"
                print(f"  ✓  {img_path.name}")
                print(f"     Class    : {cls_name}")
                print(f"     Box (px) : {box_w_px} × {box_h_px}  at  ({x_min}, {y_min})")
                print(f"     Coverage : {coverage:.1f}% of image area  [{cov_flag}]")
                print()

            annotated_images.append((img_path.name, img_rgb))

    # ── save grid ──
    print(f"{'─'*60}")
    print(f"\nSaving verification grid ...")
    save_grid(annotated_images, OUTPUT_DIR / "merged_boxes_grid.png")

    print(f"\nIndividual annotated images → {save_dir}")
    print(f"Grid overview              → {OUTPUT_DIR / 'merged_boxes_grid.png'}")
    print(f"\nOpen the grid image to visually confirm each box covers the gesture.")
    print(f"If any box looks wrong, note the filename and we can re-annotate manually.\n")


if __name__ == "__main__":
    main()