"""
Restore Multi-Label Images Script
===================================
Purpose : For images with multiple bounding boxes of the SAME class
    (e.g. two hands both making a 2-finger-heart gesture),
    merge all boxes into one single bounding box that covers
    both hands, then move the corrected image + label back
    into data/train/ for training.

    Bounding box merge logic (all coords are YOLO normalised 0-1):
    1. Convert each box: centre format → corner format
        x_min = cx - w/2,  x_max = cx + w/2
        y_min = cy - h/2,  y_max = cy + h/2
    2. Merged box = outermost corners across all boxes
        merged_x_min = min(all x_min)
        merged_x_max = max(all x_max)
    3. Convert back to YOLO centre format
        merged_cx = (x_min + x_max) / 2
        merged_w  =  x_max - x_min
    4. Clamp to [0.0, 1.0] to stay within image bounds
"""

import shutil
from pathlib import Path

import yaml

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\suzuk\emblems\data")
# ─────────────────────────────────────────────

MULTI_IMGS  = DATA_DIR / "excluded" / "multi_label" / "images"
MULTI_LBLS  = DATA_DIR / "excluded" / "multi_label" / "labels"
TRAIN_IMGS  = DATA_DIR / "train" / "images"
TRAIN_LBLS  = DATA_DIR / "train" / "labels"
YAML_PATH   = DATA_DIR / "data.yaml"

# Terminal colours
R  = "\033[0m"
B  = "\033[1m"
G  = "\033[92m"
Y  = "\033[93m"
C  = "\033[96m"


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDING BOX UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def yolo_to_corners(cx: float, cy: float, w: float, h: float) -> tuple:
    """
    Convert YOLO centre format to corner format.
    All values are normalised (0.0 to 1.0 relative to image size).

    YOLO format : cx cy w h  (centre x, centre y, width, height)
    Corner format: x_min, y_min, x_max, y_max
    """
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return x_min, y_min, x_max, y_max


def corners_to_yolo(x_min: float, y_min: float,
                    x_max: float, y_max: float) -> tuple:
    """
    Convert corner format back to YOLO centre format.
    Clamps all values to [0.0, 1.0] to stay within image bounds —
    merging boxes can slightly push coordinates outside the image.
    """
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  =  x_max - x_min
    h  =  y_max - y_min

    # Clamp everything to valid normalised range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return cx, cy, w, h


def merge_boxes(annotations: list[tuple]) -> tuple:
    """
    Given a list of (class_idx, [cx, cy, w, h]) annotations,
    merge all bounding boxes into one that covers all of them.

    Steps:
      1. Convert every box to corner format
      2. Take the outermost corners (min of mins, max of maxes)
      3. Convert merged corners back to YOLO centre format

    The class index is taken from the first annotation —
    caller should verify all annotations share the same class.

    Returns: (class_idx, merged_cx, merged_cy, merged_w, merged_h)
    """
    class_idx = annotations[0][0]   # all same class for our case

    # Convert all boxes to corners
    all_corners = [yolo_to_corners(*ann[1]) for ann in annotations]

    # Outermost bounding box
    merged_x_min = min(c[0] for c in all_corners)
    merged_y_min = min(c[1] for c in all_corners)
    merged_x_max = max(c[2] for c in all_corners)
    merged_y_max = max(c[3] for c in all_corners)

    # Back to YOLO format (with clamping)
    merged_cx, merged_cy, merged_w, merged_h = corners_to_yolo(
        merged_x_min, merged_y_min, merged_x_max, merged_y_max
    )

    return class_idx, merged_cx, merged_cy, merged_w, merged_h


# ══════════════════════════════════════════════════════════════════════════════
# FILE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_class_names(yaml_path: Path) -> list[str]:
    """Read class names from data.yaml."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)["names"]


def get_image_paths(img_dir: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def read_annotations(lbl_path: Path) -> list[tuple]:
    """
    Parse YOLO label file into list of (class_idx, [cx, cy, w, h]) tuples.
    """
    annotations = []
    if not lbl_path.exists():
        return annotations
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_idx = int(parts[0])
                box     = [float(x) for x in parts[1:]]
                annotations.append((cls_idx, box))
    return annotations


def write_single_annotation(lbl_path: Path,
                             cls_idx: int,
                             cx: float, cy: float,
                             w: float,  h: float) -> None:
    """Write a single YOLO annotation line to a label file."""
    with open(lbl_path, "w") as f:
        f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── validate paths ──
    for p, name in [
        (MULTI_IMGS, "excluded/multi_label/images"),
        (MULTI_LBLS, "excluded/multi_label/labels"),
        (YAML_PATH,  "data.yaml"),
    ]:
        if not p.exists():
            print(f"ERROR: Cannot find {name} at {p}")
            return

    class_names = load_class_names(YAML_PATH)
    img_paths   = get_image_paths(MULTI_IMGS)

    print(f"\n{B}Restoring {len(img_paths)} multi-label images → train/{R}")
    print(f"Class names: {class_names}\n")
    print(f"{'─'*65}")

    restored = 0
    skipped  = 0

    for img_path in img_paths:
        lbl_path    = MULTI_LBLS / (img_path.stem + ".txt")
        annotations = read_annotations(lbl_path)

        if not annotations:
            print(f"  {Y}SKIP{R}  {img_path.name}  — no annotations found")
            skipped += 1
            continue

        # ── log original boxes for transparency ──
        class_idx   = annotations[0][0]
        class_label = class_names[class_idx] if class_idx < len(class_names) else f"idx_{class_idx}"
        print(f"  {C}•{R} {img_path.name}")
        print(f"    Class      : {class_label} ({len(annotations)} boxes)")

        for i, (cls, box) in enumerate(annotations):
            print(f"    Box {i+1}      : cx={box[0]:.4f}  cy={box[1]:.4f}  "
                  f"w={box[2]:.4f}  h={box[3]:.4f}")

        # ── merge all boxes into one ──
        merged_cls, merged_cx, merged_cy, merged_w, merged_h = merge_boxes(annotations)

        print(f"    {G}Merged     : cx={merged_cx:.4f}  cy={merged_cy:.4f}  "
              f"w={merged_w:.4f}  h={merged_h:.4f}{R}")

        # ── write corrected label file back to excluded/multi_label/labels ──
        # (overwrite in place before moving)
        write_single_annotation(lbl_path, merged_cls,
                                 merged_cx, merged_cy,
                                 merged_w,  merged_h)

        # ── move image + corrected label → train/ ──
        shutil.move(str(img_path),  str(TRAIN_IMGS / img_path.name))
        shutil.move(str(lbl_path),  str(TRAIN_LBLS / lbl_path.name))

        print(f"    {G}→ moved to train/{R}\n")
        restored += 1

    # ── final summary ──
    print(f"{'─'*65}")
    print(f"\n{B}{C}  DONE{R}")
    print(f"  Restored : {G}{restored} images{R} (merged boxes → single annotation)")
    if skipped:
        print(f"  Skipped  : {Y}{skipped} images{R} (no annotations — check manually)")

    # Count final totals per class
    print(f"\n  {B}Updated train/ class counts:{R}")
    from collections import Counter
    counts = Counter()
    for img in get_image_paths(TRAIN_IMGS):
        lbl = TRAIN_LBLS / (img.stem + ".txt")
        anns = read_annotations(lbl)
        for cls_idx, _ in anns:
            if cls_idx < len(class_names):
                counts[class_names[cls_idx]] += 1

    total = sum(counts.values())
    for cls in class_names:
        count = counts.get(cls, 0)
        pct   = count / total * 100 if total else 0
        flag  = f"{G}✓ OK{R}" if count >= 50 else f"{Y}⚠ LOW{R}"
        print(f"  {cls:<28} {count:>4}  ({pct:5.1f}%)  {flag}")

    print(f"\n  Total clean training images : {B}{len(get_image_paths(TRAIN_IMGS))}{R}")
    print(f"\n  {G}Next step → python scripts/split_dataset.py{R}\n")


if __name__ == "__main__":
    main()