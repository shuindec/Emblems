"""
Dataset Inspect & Fix Script
=============================
Purpose : Fix two data quality issues found during audit BEFORE splitting:
            1. 32 images OpenCV cannot read
               → Try Pillow as a fallback (handles more formats)
               → If Pillow can read it: re-save as clean .jpg → keep
               → If neither can read it: move to excluded/corrupt/
            2. 11 images with multiple bounding box annotations
               → For a classification proof-of-concept, these are
                 ambiguous (which gesture is the "true" label?)
               → Move to excluded/multi_label/ for safety

          After this script, every remaining image in train/ should be:
            - Readable by OpenCV
            - Annotated with exactly one class

Usage
-----
    python scripts/dataset_inspect_and_fix.py
"""

import sys
import shutil
from pathlib import Path
from collections import Counter

import cv2
import yaml
from PIL import Image          # fallback reader — handles HEIC-like issues

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\suzuk\emblems\data")
OUTPUT_DIR = Path(r"C:\Users\suzuk\emblems\outputs\figures")
# ─────────────────────────────────────────────

TRAIN_IMGS  = DATA_DIR / "train" / "images"
TRAIN_LBLS  = DATA_DIR / "train" / "labels"
YAML_PATH   = DATA_DIR / "data.yaml"

EXCL_CORRUPT     = DATA_DIR / "excluded" / "corrupt"
EXCL_MULTI_LABEL = DATA_DIR / "excluded" / "multi_label"

# Terminal colours
R  = "\033[0m"
B  = "\033[1m"
G  = "\033[92m"
Y  = "\033[93m"
RD = "\033[91m"
C  = "\033[96m"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_class_names(yaml_path: Path) -> list[str]:
    """Read class name list from data.yaml."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)["names"]


def get_image_paths(img_dir: Path) -> list[Path]:
    """Return all image files in a directory, sorted."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def lbl_for(img_path: Path) -> Path:
    """Return the expected YOLO label .txt path for an image."""
    return TRAIN_LBLS / (img_path.stem + ".txt")


def read_annotations(lbl_path: Path) -> list[tuple[int, list[float]]]:
    """
    Parse a YOLO label file into a list of (class_idx, [cx, cy, w, h]) tuples.
    Returns empty list if file is missing or blank.
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


def try_resave_with_pillow(img_path: Path) -> bool:
    """
    Attempt to open an image with Pillow (which handles more formats than cv2).
    If successful, re-save it as a clean RGB JPEG in place.
    Returns True if the image was saved successfully, False otherwise.
    """
    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")   # drop alpha channel if present
            # Save back to same path as JPEG (overwrite in place)
            img_rgb.save(img_path.with_suffix(".jpg"), "JPEG", quality=95)
            # If original was not .jpg, remove the old file
            if img_path.suffix.lower() != ".jpg":
                img_path.unlink()
        return True
    except Exception:
        return False


def move_pair(img_path: Path, dest_img_dir: Path, dest_lbl_dir: Path) -> None:
    """Move an image and its matching label file to destination directories."""
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(str(img_path), str(dest_img_dir / img_path.name))

    lbl_path = lbl_for(img_path)
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(dest_lbl_dir / lbl_path.name))


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — UNREADABLE IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def fix_unreadable_images(img_paths: list[Path], class_names: list[str]) -> dict:
    """
    For every image in img_paths:
      - Try cv2.imread() first
      - If that fails, try Pillow re-save
      - If Pillow also fails, move to excluded/corrupt/

    Returns a summary with counts of fixed vs moved images.
    """
    unreadable = []
    for img_path in img_paths:
        if cv2.imread(str(img_path)) is None:
            unreadable.append(img_path)

    if not unreadable:
        print(f"  {G}✓ No unreadable images found.{R}")
        return {"unreadable": 0, "fixed": 0, "corrupted": 0}

    print(f"\n  Found {RD}{len(unreadable)}{R} unreadable images. Attempting Pillow rescue ...\n")

    fixed     = []
    corrupted = []

    for img_path in unreadable:
        rescued = try_resave_with_pillow(img_path)
        if rescued:
            fixed.append(img_path.stem)
            print(f"  {G}✓ rescued{R}  {img_path.name}")
        else:
            corrupted.append(img_path)
            print(f"  {RD}✗ corrupt{R}  {img_path.name}  → moving to excluded/corrupt/")
            move_pair(img_path, EXCL_CORRUPT / "images", EXCL_CORRUPT / "labels")

    print(f"\n  Summary: {len(fixed)} rescued by Pillow, {len(corrupted)} moved to excluded/corrupt/")
    return {
        "unreadable" : len(unreadable),
        "fixed"      : len(fixed),
        "corrupted"  : len(corrupted),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — MULTI-ANNOTATION IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def fix_multi_label_images(img_paths: list[Path], class_names: list[str]) -> dict:
    """
    For a classification proof-of-concept, images with more than one
    bounding box annotation are ambiguous — we cannot assign a single
    ground-truth class to them reliably.

    This function finds all such images, prints their annotation details
    for transparency, and moves them to excluded/multi_label/.

    Returns a summary with count of moved images.
    """
    multi_label = []

    for img_path in img_paths:
        lbl_path    = lbl_for(img_path)
        annotations = read_annotations(lbl_path)
        if len(annotations) > 1:
            multi_label.append((img_path, annotations))

    if not multi_label:
        print(f"  {G}✓ No multi-annotation images found.{R}")
        return {"multi_label": 0}

    print(f"\n  Found {Y}{len(multi_label)}{R} images with multiple annotations:\n")

    for img_path, annotations in multi_label:
        class_labels = [
            class_names[a[0]] if a[0] < len(class_names) else f"idx_{a[0]}"
            for a in annotations
        ]
        print(f"  {Y}•{R} {img_path.name:<45} annotations: {class_labels}")

    print(f"\n  These are ambiguous for classification training.")
    print(f"  Moving to excluded/multi_label/ ...")

    # Move each multi-label image and its label
    for img_path, _ in multi_label:
        move_pair(img_path, EXCL_MULTI_LABEL / "images", EXCL_MULTI_LABEL / "labels")
        print(f"  {Y}→ moved{R}  {img_path.name}")

    print(f"\n  Moved {len(multi_label)} image+label pairs → excluded/multi_label/")
    return {"multi_label": len(multi_label)}


# ══════════════════════════════════════════════════════════════════════════════
# FINAL CLASS COUNT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_final_counts(class_names: list[str]) -> int:
    """
    Re-scan train/ after all fixes and print clean per-class counts.
    Returns total remaining image count.
    """
    img_paths = get_image_paths(TRAIN_IMGS)
    counts    = Counter()

    for img_path in img_paths:
        lbl_path    = lbl_for(img_path)
        annotations = read_annotations(lbl_path)
        for cls_idx, _ in annotations:
            if cls_idx < len(class_names):
                counts[class_names[cls_idx]] += 1

    total = len(img_paths)
    print(f"\n  {B}Final Class Counts (after all fixes):{R}")
    for cls in class_names:
        count = counts.get(cls, 0)
        pct   = (count / total * 100) if total > 0 else 0
        flag  = f"{G}✓ OK{R}" if count >= 50 else f"{Y}⚠ LOW{R}"
        print(f"  {cls:<28} {count:>4}  ({pct:5.1f}%)  {flag}")

    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── validate paths ──
    for p, name in [(TRAIN_IMGS, "train/images"), (TRAIN_LBLS, "train/labels"), (YAML_PATH, "data.yaml")]:
        if not p.exists():
            print(f"{RD}ERROR: Cannot find {name} at {p}{R}")
            sys.exit(1)

    class_names = load_class_names(YAML_PATH)
    print(f"\nClasses: {class_names}")

    # Get current image list once (both fixes read from same directory)
    img_paths = get_image_paths(TRAIN_IMGS)
    print(f"Images found in train/ : {len(img_paths)}\n")

    # ══════════════════════════════
    # FIX 1 — UNREADABLE IMAGES
    # ══════════════════════════════
    print(f"{'═'*60}")
    print(f"{B}[ FIX 1 ] Unreadable Images{R}")
    print(f"{'═'*60}")
    summary_1 = fix_unreadable_images(img_paths, class_names)

    # Refresh image list after potential re-saves / moves
    img_paths = get_image_paths(TRAIN_IMGS)

    # ══════════════════════════════
    # FIX 2 — MULTI-LABEL IMAGES
    # ══════════════════════════════
    print(f"\n{'═'*60}")
    print(f"{B}[ FIX 2 ] Multi-Annotation Images{R}")
    print(f"{'═'*60}")
    summary_2 = fix_multi_label_images(img_paths, class_names)

    # ══════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════
    print(f"\n{'═'*60}")
    print(f"{B}{C}  FINAL SUMMARY{R}")
    print(f"{'═'*60}")
    print(f"  Unreadable → rescued by Pillow : {G}{summary_1.get('fixed', 0)}{R}")
    print(f"  Unreadable → moved to corrupt/ : {RD}{summary_1.get('corrupted', 0)}{R}")
    print(f"  Multi-label → moved to excl/   : {Y}{summary_2.get('multi_label', 0)}{R}")

    total = print_final_counts(class_names)

    print(f"\n  {B}Total clean images remaining : {total}{R}")
    print(f"  Excluded/corrupt saved to    : {EXCL_CORRUPT}")
    print(f"  Excluded/multi_label to      : {EXCL_MULTI_LABEL}")
    print(f"\n  {G}Next step → python scripts/split_dataset.py{R}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()