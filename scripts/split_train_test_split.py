"""
Stratified Train / Val / Test Split
=====================================
PURPOSE:
    Split cleaned images into train / val / test using STRATIFIED
    sampling so every split has the same class ratio as the full dataset.

    All config (paths, ratios, seed) is read from configs/config.yaml.
    All shared utilities are imported from src/utils.py.

WHY STRATIFIED:
    A plain random split on ~231 images can put most images of a
    minority class into one split by chance. Stratified sampling
    guarantees proportional class representation in every split.

OUTPUT:
    data/split/train/  images/ + labels/   (~60%)
    data/split/val/    images/ + labels/   (~20%)
    data/split/test/   images/ + labels/   (~20%)
    data/split/data.yaml                   updated paths + class names
    outputs/split_manifest.csv             full audit trail
    outputs/figures/split_distribution.png balance chart

USAGE:
    python scripts/split_dataset.py
"""
import shutil
import csv
import random
import sys
from pathlib import Path
from collections import defaultdict

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── import shared utilities (no more copy-pasting) ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_config, get_project_paths, setup_output_dirs,
    load_class_names, get_image_paths, get_label_path,
    get_class_for_image, count_classes,
    Colours as Col,
)

# ══════════════════════════════════════════════════════════════════════════════
# STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def stratified_split(img_paths: list[Path],
                     lbl_dir: Path,
                     class_names: list[str],
                     train_ratio: float,
                     val_ratio: float,
                     seed: int) -> dict[str, list[Path]]:
    """
    Split images into train / val / test preserving class proportions.

    Algorithm (per class):
      1. Group all images by their class label
      2. Shuffle each group with a fixed seed (reproducible)
      3. Take first train_ratio fraction   -> train
         Take next  val_ratio fraction     -> val
         Remaining                         -> test
      4. Merge groups back into combined split lists

    Args:
        img_paths   : all source image paths
        lbl_dir     : labels directory (for class lookup)
        class_names : class name list from data.yaml
        train_ratio : e.g. 0.60
        val_ratio   : e.g. 0.20
        seed        : random seed for reproducibility

    Returns:
        dict {"train": [...], "val": [...], "test": [...]}
    """
    # Group images by class
    class_buckets: dict[str, list[Path]] = defaultdict(list)
    skipped = []

    for img_path in img_paths:
        cls = get_class_for_image(img_path, lbl_dir, class_names)
        if cls is None:
            skipped.append(img_path.name)
        else:
            class_buckets[cls].append(img_path)

    if skipped:
        print(f"  {Col.Y}Warning: Skipped {len(skipped)} images "
              f"(missing/unreadable labels){Col.R}")

    rng = random.Random(seed)   # isolated RNG — won't affect other code
    result: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for cls_name, paths in sorted(class_buckets.items()):
        shuffled = paths.copy()
        rng.shuffle(shuffled)

        n       = len(shuffled)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))
        n_test  = n - n_train - n_val

        # Guard: if rounding squeezes test to 0, borrow one from val
        if n_test < 1:
            n_val -= 1
            n_test = 1

        result["train"].extend(shuffled[:n_train])
        result["val"].extend(shuffled[n_train: n_train + n_val])
        result["test"].extend(shuffled[n_train + n_val:])

        print(f"  {cls_name:<28}  total={n:>3}  "
              f"train={n_train:>3}  val={n_val:>3}  test={n_test:>3}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FILE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def copy_split_files(split_assignments: dict[str, list[Path]],
                     src_lbl_dir: Path,
                     split_dir: Path) -> None:
    """
    Copy images and labels into data/split/{train,val,test}/ folders.
    Uses COPY not move — original data/train/ stays intact as backup.
    """
    for split_name, img_paths in split_assignments.items():
        dest_img = split_dir / split_name / "images"
        dest_lbl = split_dir / split_name / "labels"
        setup_output_dirs(dest_img, dest_lbl)

        for img_path in img_paths:
            shutil.copy2(img_path, dest_img / img_path.name)
            lbl_path = get_label_path(img_path, src_lbl_dir)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dest_lbl / lbl_path.name)


def save_manifest(split_assignments: dict[str, list[Path]],
                  lbl_dir: Path,
                  class_names: list[str],
                  save_path: Path) -> None:
    """
    Save a CSV recording every image -> split assignment + class name.
    Columns: filename, class_name, split

    This CSV serves two purposes:
      1. Audit trail — proves which images are in which split
      2. DataLoader input — src/data_loader.py reads this CSV
         instead of scanning directories (faster and more reliable)

    encoding="utf-8" is explicit to avoid Windows cp1252 errors.
    """
    setup_output_dirs(save_path.parent)

    # utf-8 encoding + newline="" prevents double line breaks on Windows
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_name", "split"])

        for split_name, img_paths in split_assignments.items():
            for img_path in sorted(img_paths):
                cls = get_class_for_image(img_path, lbl_dir, class_names)
                writer.writerow([img_path.name, cls or "unknown", split_name])

    print(f"  [saved] {save_path}")


def write_split_yaml(class_names: list[str], split_dir: Path) -> None:
    """
    Write a data.yaml into data/split/ pointing to the new folders.
    Training scripts reference this file instead of the original
    Roboflow data.yaml.
    """
    cfg = {
        "train" : str(split_dir / "train" / "images"),
        "val"   : str(split_dir / "val"   / "images"),
        "test"  : str(split_dir / "test"  / "images"),
        "nc"    : len(class_names),
        "names" : class_names,
    }
    save_path = split_dir / "data.yaml"
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  [saved] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_split_distribution(split_assignments: dict[str, list[Path]],
                             lbl_dir: Path,
                             class_names: list[str],
                             save_path: Path) -> None:
    """
    Grouped bar chart: images per class per split.
    Bar heights proportional across splits = stratification worked.
    """
    counts = {
        split: count_classes(paths, lbl_dir, class_names)
        for split, paths in split_assignments.items()
    }

    x         = np.arange(len(class_names))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d0f14")
    ax.set_facecolor("#13161e")

    split_style = {
        "train": ("#2ec4b6", "Train (60%)"),
        "val"  : ("#ffd166", "Val (20%)"),
        "test" : ("#e63946", "Test (20%)"),
    }

    for i, (split_name, (colour, label)) in enumerate(split_style.items()):
        values = [counts[split_name].get(cls, 0) for cls in class_names]
        offset = (i - 1) * bar_width
        bars   = ax.bar(
            x + offset, values, bar_width,
            label=label, color=colour,
            alpha=0.85, edgecolor="#0d0f14",
            linewidth=0.8, zorder=3
        )
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4, str(v),
                    ha="center", va="bottom",
                    color="white", fontsize=9, fontweight="bold"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=12, ha="right",
                       color="#c8d0e8", fontsize=9)
    ax.set_ylabel("Image Count", color="#c8d0e8", fontsize=10)
    ax.set_title("Stratified Split Distribution — Per Class Per Split",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#c8d0e8")
    ax.spines[:].set_color("#1e2330")
    ax.yaxis.grid(True, color="#1e2330", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(facecolor="#13161e", edgecolor="#1e2330",
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [saved] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg   = load_config()
    paths = get_project_paths(cfg)

    train_ratio = cfg["split"]["train_ratio"]
    val_ratio   = cfg["split"]["val_ratio"]
    seed        = cfg["split"]["random_seed"]

    for key in ("source_imgs", "source_lbls", "yaml_path"):
        if not paths[key].exists():
            print(f"{Col.RD}ERROR: {key} not found at {paths[key]}{Col.R}")
            sys.exit(1)

    setup_output_dirs(paths["figures_dir"], paths["output_dir"])

    class_names = load_class_names(paths["yaml_path"])
    img_paths   = get_image_paths(paths["source_imgs"])

    print(f"\n{Col.B}Dataset  : {len(img_paths)} images · "
          f"{len(class_names)} classes{Col.R}")
    print(f"Seed     : {seed}  (fixed for reproducibility)")
    print(f"Ratios   : train={train_ratio:.0%}  val={val_ratio:.0%}  "
          f"test={1 - train_ratio - val_ratio:.0%}\n")

    print(f"{Col.B}[ STEP 1 ] Stratified split per class:{Col.R}")
    print(f"{'─' * 60}")
    assignments = stratified_split(
        img_paths, paths["source_lbls"], class_names,
        train_ratio, val_ratio, seed
    )
    print(f"{'─' * 60}")
    for split_name, ps in assignments.items():
        print(f"  {split_name:<6} total : {Col.B}{len(ps)}{Col.R} images")

    print(f"\n{Col.B}[ STEP 2 ] Copying files to data/split/ ...{Col.R}")
    copy_split_files(assignments, paths["source_lbls"], paths["split_dir"])
    print(f"  {Col.G}Done.{Col.R}")

    print(f"\n{Col.B}[ STEP 3 ] Saving manifest and split data.yaml ...{Col.R}")
    save_manifest(assignments, paths["source_lbls"], class_names,
                  paths["manifest"])
    write_split_yaml(class_names, paths["split_dir"])

    print(f"\n{Col.B}[ STEP 4 ] Saving split distribution chart ...{Col.R}")
    plot_split_distribution(
        assignments, paths["source_lbls"], class_names,
        paths["figures_dir"] / "split_distribution.png"
    )

    total = sum(len(p) for p in assignments.values())
    print(f"\n{'=' * 60}")
    print(f"{Col.B}{Col.C}  SPLIT COMPLETE{Col.R}")
    print(f"{'=' * 60}")
    print(f"  Total images split : {Col.B}{total}{Col.R}")
    print(f"  Train              : {Col.B}{len(assignments['train'])}{Col.R}"
          f"  ->  data/split/train/")
    print(f"  Val                : {Col.B}{len(assignments['val'])}{Col.R}"
          f"  ->  data/split/val/")
    print(f"  Test               : {Col.B}{len(assignments['test'])}{Col.R}"
          f"  ->  data/split/test/")
    print(f"  Manifest           : outputs/split_manifest.csv")
    print(f"  Split data.yaml    : data/split/data.yaml")
    print(f"\n  Original data/train/ is untouched — full backup preserved.")
    print(f"\n  {Col.G}Next step -> src/data_loader.py{Col.R}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()