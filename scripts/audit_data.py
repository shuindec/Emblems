"""
Dataset Audit Script
====================
Project : Emblems Heart Gesture Recognition
Purpose : Single-pass script that:
            1. Audits the raw Roboflow export (class balance, missing
               labels, image sizes, imbalance ratio)
            2. Removes a specified class by MOVING affected files to data/excluded/ (not deleting)
            3. Remaps remaining label indices so no gap is left
            4. Rewrites data.yaml with updated class list
            5. Saves a before/after class distribution figure
 
"""
 
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict, Counter
 
import cv2
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#configuration — update these paths and class name as needed
DATA_DIR        = Path(r"C:\Users\suzuk\emblems\data")
OUTPUT_DIR      = Path(r"C:\Users\suzuk\emblems\outputs\figures")
CLASS_TO_REMOVE = "full-arm-heart"   # set to "" to skip removal
# ─────────────────────────────────────────────
 
TRAIN_IMGS = DATA_DIR / "train" / "images"
TRAIN_LBLS = DATA_DIR / "train" / "labels"
YAML_PATH  = DATA_DIR / "data.yaml"
EXCL_IMGS  = DATA_DIR / "excluded" / "images"
EXCL_LBLS  = DATA_DIR / "excluded" / "labels"
 
# Chart colours — one per class slot (up to 6)
PALETTE = ["#2ec4b6", "#8338ec", "#a3d977", "#ff9f1c", "#e63946", "#74b0ff"]

# SECTION 1 — FILE UTILITIES
def load_yaml(path: Path) -> dict:
    """Load data.yaml and return as a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
 
def save_yaml(path: Path, data: dict) -> None:
    """Write an updated data.yaml back to disk, preserving key order."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
 
def get_image_paths(img_dir: Path) -> list[Path]:
    """Return sorted list of image files (common extensions) in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
 
 
def lbl_path_for(img_path: Path, lbl_dir: Path) -> Path:
    """Return the expected label .txt path for a given image path."""
    return lbl_dir / (img_path.stem + ".txt")
 
 
def read_class_indices(lbl_path: Path) -> list[int]:
    """
    Parse a YOLO label file and return the list of class indices it contains.
    YOLO format per line: class_idx cx cy w h
    Only the first value (class index) is needed here.
    """
    indices = []
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    indices.append(int(line.split()[0]))
    return indices

# SECTION 2 — AUDIT
def run_audit(img_dir: Path, lbl_dir: Path, class_names: list[str]) -> dict:
    """
    Walk every image in img_dir, match its label file, and collect:
      - per-class annotation counts
      - missing / empty label files
      - images with multiple annotations
      - image dimensions (for size consistency check)
      - unreadable images
 
    Returns a results dict used both for printing and for the chart.
    """
    img_paths = get_image_paths(img_dir)
 
    results = {
        "total_images"      : len(img_paths),
        "class_names"       : class_names,
        "class_counts"      : Counter(),   # {class_name: int}
        "missing_labels"    : [],          # stems with no .txt file
        "empty_labels"      : [],          # stems with blank .txt
        "multi_label_images": [],          # stems with >1 annotation box
        "image_sizes"       : [],          # list of (width, height) tuples
        "unreadable_images" : [],          # stems OpenCV cannot open
    }
 
    for img_path in img_paths:
        lbl_path = lbl_path_for(img_path, lbl_dir)
 
        # ── label presence check ──
        if not lbl_path.exists():
            results["missing_labels"].append(img_path.stem)
            continue
 
        indices = read_class_indices(lbl_path)
 
        if len(indices) == 0:
            results["empty_labels"].append(img_path.stem)
            continue
 
        if len(indices) > 1:
            results["multi_label_images"].append(img_path.stem)
 
        # ── tally class counts ──
        for idx in indices:
            if idx < len(class_names):
                results["class_counts"][class_names[idx]] += 1
 
        # ── image readability + size ──
        img = cv2.imread(str(img_path))
        if img is None:
            results["unreadable_images"].append(img_path.stem)
        else:
            h, w = img.shape[:2]
            results["image_sizes"].append((w, h))
 
    return results

# SECTION 3 — TERMINAL REPORT
# Terminal colour codes
R = "\033[0m"   # reset
B = "\033[1m"   # bold
G = "\033[92m"  # green
Y = "\033[93m"  # yellow
RD= "\033[91m"  # red
C = "\033[96m"  # cyan
 
 
def print_audit_report(results: dict, label: str = "BEFORE CLEANING") -> None:
    """Print a human-readable audit report to the terminal."""
    class_names = results["class_names"]
    counts      = results["class_counts"]
    total       = results["total_images"]
 
    print(f"\n{'═'*60}")
    print(f"{B}{C}  DATASET AUDIT — {label}{R}")
    print(f"{'═'*60}")
    print(f"  Dataset path : {TRAIN_IMGS}")
    print(f"  Total images : {B}{total}{R}")
    print(f"  Classes      : {len(class_names)}")
    print(f"{'─'*60}")
 
    # ── class distribution table ──
    print(f"\n  {B}Class Distribution:{R}")
    for cls in class_names:
        count = counts.get(cls, 0)
        pct   = (count / total * 100) if total > 0 else 0
 
        if count < 20:
            flag = f"{RD}⚠ CRITICAL — too few to train{R}"
        elif count < 50:
            flag = f"{Y}⚠ LOW — collect more{R}"
        else:
            flag = f"{G}✓ OK{R}"
 
        print(f"  {cls:<28} {count:>4}  ({pct:5.1f}%)  {flag}")
 
    # ── imbalance ratio ──
    if counts:
        max_c = max(counts.values())
        min_c = min(counts.values())
        ratio = max_c / min_c if min_c > 0 else float("inf")
        col   = RD if ratio > 5 else Y if ratio > 3 else G
        print(f"\n  Imbalance ratio (max / min) : {col}{ratio:.1f}x{R}")
        if ratio > 5:
            print(f"  {RD}→ Severe imbalance detected.{R}")
            print(f"    Use weighted loss or oversample minority classes.")
 
    # ── data quality checks ──
    print(f"\n  {B}Data Quality Checks:{R}")
    ok   = lambda m: print(f"  {G}✓{R} {m}")
    warn = lambda m: print(f"  {Y}⚠{R} {m}")
    err  = lambda m: print(f"  {RD}✗{R} {m}")
 
    if results["missing_labels"]:
        err(f"{len(results['missing_labels'])} images have NO label file")
        for s in results["missing_labels"][:5]:
            print(f"      - {s}")
    else:
        ok("All images have a matching label file")
 
    if results["empty_labels"]:
        warn(f"{len(results['empty_labels'])} label files are empty")
    else:
        ok("No empty label files")
 
    if results["multi_label_images"]:
        warn(f"{len(results['multi_label_images'])} images have multiple annotations")
    else:
        ok("No multi-label images")
 
    if results["unreadable_images"]:
        err(f"{len(results['unreadable_images'])} images could not be read by OpenCV")
    else:
        ok("All images readable by OpenCV")
 
    # ── image size summary ──
    if results["image_sizes"]:
        sizes   = results["image_sizes"]
        widths  = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        unique  = set(sizes)
        print(f"\n  {B}Image Dimensions:{R}")
        print(f"    Unique resolutions : {len(unique)}")
        print(f"    Width  range       : {min(widths)} – {max(widths)} px")
        print(f"    Height range       : {min(heights)} – {max(heights)} px")
        if len(unique) > 1:
            warn("Mixed resolutions — DataLoader will resize to 224×224 (expected, fine)")
        else:
            ok(f"Uniform resolution: {sizes[0][0]}×{sizes[0][1]}")
 
    print(f"{'─'*60}")

# SECTION 4 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_before_after(before: dict, after: dict, save_path: Path) -> None:
    """
    Save a single figure with two side-by-side bar charts:
      Left  — class distribution BEFORE removing the unwanted class
      Right — class distribution AFTER removal
 
    The 50-image warning threshold line is drawn on both charts.
    Bars below 20 images are outlined in red as a visual warning.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0f14")
 
    datasets = [
        (before, "Before Cleaning"),
        (after,  "After Cleaning"),
    ]
 
    for ax, (results, title) in zip(axes, datasets):
        class_names = results["class_names"]
        counts      = [results["class_counts"].get(c, 0) for c in class_names]
        colors      = PALETTE[: len(class_names)]
        total       = results["total_images"]
 
        ax.set_facecolor("#13161e")
 
        bars = ax.bar(class_names, counts, color=colors) # Warning threshold line at 50 images
        ax.axhline(
            50, color="#ffd166", linewidth=1.2,
            linestyle="--", zorder=4, label="Min. recommended (50)"
        )
 
        # Value labels on top of each bar
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                str(count),
                ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold"
            )
            # Red outline for critically low bars
            if count < 20:
                bar.set_edgecolor("#e63946")
                bar.set_linewidth(2.5)
 
        ax.set_title(title, color="white", fontsize=12, pad=10)
        ax.set_ylabel("Image Count", color="#c8d0e8", fontsize=10)
        ax.tick_params(colors="#c8d0e8", labelsize=8)
        ax.spines[:].set_color("#1e2330")
        ax.yaxis.grid(True, color="#1e2330", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(
            facecolor="#13161e", edgecolor="#1e2330",
            labelcolor="#ffd166", fontsize=8
        )
 
        # Subtitle with total count
        ax.set_xlabel(f"Total images: {total}", color="#4a5270", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
 
    fig.suptitle(
        "Heart Gesture Dataset — Class Distribution",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")

# SECTION 5 — CLASS REMOVAL
# ══════════════════════════════════════════════════════════════════════════════
 
def remap_label_file(lbl_path: Path, removed_index: int) -> None:
    """
    After removing a class at removed_index, all class indices above it
    must shift down by 1 to close the gap.
 
    Example: removing index 1 (full-arm-heart):
      old index 0 → stays 0  (2-finger-heart)
      old index 1 → REMOVED
      old index 2 → new index 1  (traditional-heart)
      old index 3 → new index 2  (traditional-heart-2)
      old index 4 → new index 3  (upside_down_heart)
 
    This is done IN PLACE on the .txt file.
    """
    new_lines = []
    with open(lbl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts  = line.split()
            cls_id = int(parts[0])
            # Shift down any index that was above the removed class
            if cls_id > removed_index:
                parts[0] = str(cls_id - 1)
            new_lines.append(" ".join(parts))
 
    with open(lbl_path, "w") as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
 
 
def remove_class(class_names: list[str], remove_name: str) -> dict:
    """
    Move all images/labels for remove_name to excluded/.
    Remap remaining label indices.
    Return a summary dict of what was done.
    """
    if remove_name not in class_names:
        print(f"  {RD}ERROR: '{remove_name}' not in class list: {class_names}{R}")
        sys.exit(1)
 
    remove_idx = class_names.index(remove_name)
 
    # Categorise every image as: move | remap | leave alone
    img_paths  = get_image_paths(TRAIN_IMGS)
    to_move    = []   # contains the removed class → move to excluded/
    to_remap   = []   # contains classes above removed_idx → update indices
 
    for img_path in img_paths:
        lbl_path = lbl_path_for(img_path, TRAIN_LBLS)
        if not lbl_path.exists():
            continue
        indices = read_class_indices(lbl_path)
 
        if remove_idx in indices:
            to_move.append(img_path)
        elif any(i > remove_idx for i in indices):
            to_remap.append(img_path)
 
    # ── confirm with user ──
    print(f"\n  Class to remove : '{remove_name}' (index {remove_idx})")
    print(f"  Images to MOVE  : {len(to_move)}")
    print(f"  Labels to REMAP : {len(to_remap)}")
    for img in to_move:
        print(f"    - {img.name}")
 
    confirm = input(f"\n  Proceed? Files will be MOVED to excluded/ (not deleted). [y/N] ").strip().lower()
    if confirm != "y":
        print(f"  {Y}Aborted. No files were changed.{R}")
        sys.exit(0)
 
    # ── create excluded folders ──
    EXCL_IMGS.mkdir(parents=True, exist_ok=True)
    EXCL_LBLS.mkdir(parents=True, exist_ok=True)
 
    # ── move image + label pairs ──
    for img_path in to_move:
        shutil.move(str(img_path), str(EXCL_IMGS / img_path.name))
        lbl_path = lbl_path_for(img_path, TRAIN_LBLS)
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(EXCL_LBLS / lbl_path.name))
 
    print(f"\n  {G}Moved {len(to_move)} image+label pairs → {DATA_DIR / 'excluded'}{R}")
 
    # ── remap remaining label indices ──
    for img_path in to_remap:
        lbl_path = lbl_path_for(img_path, TRAIN_LBLS)
        if lbl_path.exists():
            remap_label_file(lbl_path, remove_idx)
 
    if to_remap:
        print(f"  {G}Remapped class indices in {len(to_remap)} label files{R}")
 
    # ── update data.yaml ──
    cfg = load_yaml(YAML_PATH)
    updated_names = [n for n in class_names if n != remove_name]
    cfg["nc"]     = len(updated_names)
    cfg["names"]  = updated_names
    save_yaml(YAML_PATH, cfg)
    print(f"  {G}Updated data.yaml → nc: {len(updated_names)}, names: {updated_names}{R}")
 
    return {
        "moved"  : len(to_move),
        "remapped": len(to_remap),
        "updated_names": updated_names,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    # ── validate required paths ──
    for p, name in [
        (TRAIN_IMGS, "train/images"),
        (TRAIN_LBLS, "train/labels"),
        (YAML_PATH,  "data.yaml"),
    ]:
        if not p.exists():
            print(f"{RD}ERROR: Cannot find {name} at {p}{R}")
            print("       Update DATA_DIR at the top of this script.")
            sys.exit(1)
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
    # ── load class names ──
    cfg              = load_yaml(YAML_PATH)
    class_names      = cfg["names"]
    print(f"\nLoaded {len(class_names)} classes from data.yaml: {class_names}")
 
    # ══════════════════════════════
    # STEP 1 — AUDIT (before)
    # ══════════════════════════════
    print(f"\n{B}[ STEP 1 ] Running dataset audit ...{R}")
    before = run_audit(TRAIN_IMGS, TRAIN_LBLS, class_names)
    print_audit_report(before, label="BEFORE CLEANING")
 
    # ══════════════════════════════
    # STEP 2 — REMOVE CLASS
    # ══════════════════════════════
    if CLASS_TO_REMOVE:
        print(f"\n{B}[ STEP 2 ] Removing class: '{CLASS_TO_REMOVE}' ...{R}")
        summary = remove_class(class_names, CLASS_TO_REMOVE)
        updated_names = summary["updated_names"]
    else:
        print(f"\n{Y}[ STEP 2 ] CLASS_TO_REMOVE is empty — skipping removal.{R}")
        updated_names = class_names
 
    # ══════════════════════════════
    # STEP 3 — AUDIT (after)
    # ══════════════════════════════
    print(f"\n{B}[ STEP 3 ] Re-auditing cleaned dataset ...{R}")
    after = run_audit(TRAIN_IMGS, TRAIN_LBLS, updated_names)
    print_audit_report(after, label="AFTER CLEANING")
 
    # ══════════════════════════════
    # STEP 4 — SAVE BEFORE/AFTER CHART
    # ══════════════════════════════
    print(f"\n{B}[ STEP 4 ] Saving before/after chart ...{R}")
    chart_path = OUTPUT_DIR / "class_distribution_before_after.png"
    plot_before_after(before, after, chart_path)
 
    # ══════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════
    remaining = len(get_image_paths(TRAIN_IMGS))
    print(f"\n{'═'*60}")
    print(f"{B}{C}  DONE{R}")
    print(f"{'═'*60}")
    print(f"  Remaining training images : {B}{remaining}{R}")
    print(f"  Active classes            : {updated_names}")
    print(f"  Excluded files saved to   : {DATA_DIR / 'excluded'}")
    print(f"  Chart saved to            : {chart_path}")
    print(f"\n  Next step → python scripts/split_dataset.py")
    print(f"{'═'*60}\n")
 
 
if __name__ == "__main__":
    main()