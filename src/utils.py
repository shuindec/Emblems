"""
src/utils.py — Shared Project Utilities
=========================================
PURPOSE:
    Central module of reusable functions and constants imported by
    every script and training module in the project.

    Instead of copy-pasting path handling, config loading, and label
    parsing into each script, every file does:

        from src.utils import load_config, get_image_paths, read_annotations

    This means:
      - One place to fix bugs (fix here → fixed everywhere)
      - No repeated code across scripts
      - Clean, readable script files that focus on their own logic

CONTENTS:
    Config      → load_config()
    Paths       → get_project_paths(), setup_output_dirs()
    Images      → get_image_paths(), get_label_path()
    Annotations → read_annotations(), get_class_for_image()
    Terminal    → Colours class (ANSI colour constants)
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import yaml


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

# Resolve project root relative to this file's location
# src/utils.py → parent is src/ → parent is project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load the central config.yaml and return as a dictionary.

    Searches in this order:
      1. Explicit path if provided
      2. configs/config.yaml relative to project root
      3. config.yaml in current working directory

    Args:
        config_path : optional explicit path to config.yaml

    Returns:
        dict of all config settings

    Raises:
        FileNotFoundError if config.yaml cannot be found
    """
    candidates = []

    if config_path:
        candidates.append(Path(config_path))

    candidates += [
        PROJECT_ROOT / "configs" / "config.yaml",
        Path.cwd() / "configs" / "config.yaml",
        Path.cwd() / "config.yaml",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"config.yaml not found. Searched:\n  {searched}\n"
        f"Make sure configs/config.yaml exists in your project root."
    )


def get_project_paths(cfg: dict) -> dict[str, Path]:
    """
    Resolve all project paths from config into absolute Path objects.

    Centralising path resolution here means scripts never build
    paths manually — they just call get_project_paths(cfg)["data_dir"].

    Args:
        cfg : loaded config dict from load_config()

    Returns:
        dict of named Path objects:
          data_dir, output_dir, figures_dir,
          source_imgs, source_lbls, yaml_path,
          split_dir
    """
    data_dir    = Path(cfg["paths"]["data_dir"])
    output_dir  = Path(cfg["paths"]["output_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    split_subdir = cfg["split"]["output_dir"]

    return {
        "data_dir"    : data_dir,
        "output_dir"  : output_dir,
        "figures_dir" : figures_dir,
        "source_imgs" : data_dir / cfg["dataset"]["source_images"],
        "source_lbls" : data_dir / cfg["dataset"]["source_labels"],
        "yaml_path"   : data_dir / cfg["dataset"]["yaml_file"],
        "split_dir"   : data_dir / split_subdir,
        "manifest"    : output_dir / "split_manifest.csv",
    }


def setup_output_dirs(*dirs: Path) -> None:
    """
    Create output directories if they don't exist.
    Accepts any number of Path arguments.

    Usage:
        setup_output_dirs(figures_dir, output_dir / "checkpoints")
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS NAMES
# ══════════════════════════════════════════════════════════════════════════════

def load_class_names(yaml_path: Path) -> list[str]:
    """
    Read class names from a YOLO data.yaml file.

    Args:
        yaml_path : path to data.yaml

    Returns:
        list of class name strings in index order
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["names"]


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE & LABEL FILE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_paths(img_dir: Path) -> list[Path]:
    """
    Return a sorted list of all image files in a directory.
    Handles mixed extensions (.jpg, .jpeg, .png, .bmp, .webp).

    Args:
        img_dir : directory to scan

    Returns:
        sorted list of Path objects
    """
    return sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def get_label_path(img_path: Path, lbl_dir: Path) -> Path:
    """
    Return the expected YOLO label .txt path for a given image.
    YOLO convention: same stem as image, .txt extension, in labels/ folder.

    Args:
        img_path : path to the image file
        lbl_dir  : directory containing label files

    Returns:
        Path to the corresponding .txt file (may or may not exist)
    """
    return lbl_dir / (img_path.stem + ".txt")


def find_image_file(img_dir: Path, stem: str) -> Path | None:
    """
    Find an image file by stem name, regardless of extension.
    Useful when Roboflow exports mix .jpg and .png filenames.

    Args:
        img_dir : directory to search
        stem    : filename without extension

    Returns:
        Path to the found image, or None if not found
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = img_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATION PARSING
# ══════════════════════════════════════════════════════════════════════════════

def read_annotations(lbl_path: Path) -> list[tuple[int, list[float]]]:
    """
    Parse a YOLO label file into a list of (class_idx, [cx, cy, w, h]) tuples.

    YOLO format: each line is  class_idx  cx  cy  w  h
    All bbox coordinates are normalised to [0.0, 1.0].

    Args:
        lbl_path : path to the .txt label file

    Returns:
        list of (class_idx, [cx, cy, w, h]) — empty list if file missing/blank
    """
    annotations = []
    if not lbl_path.exists():
        return annotations

    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_idx = int(parts[0])
                box     = [float(x) for x in parts[1:]]
                annotations.append((cls_idx, box))

    return annotations


def get_class_for_image(img_path: Path, lbl_dir: Path,
                         class_names: list[str]) -> str | None:
    """
    Return the class name for an image from its label file.
    Uses the first annotation line (assumes single-annotation images
    after cleaning).

    Args:
        img_path    : path to the image
        lbl_dir     : directory containing label files
        class_names : ordered list of class names

    Returns:
        class name string, or None if label missing/unreadable
    """
    lbl_path    = get_label_path(img_path, lbl_dir)
    annotations = read_annotations(lbl_path)

    if annotations:
        cls_idx = annotations[0][0]
        if cls_idx < len(class_names):
            return class_names[cls_idx]

    return None


def count_classes(img_paths: list[Path], lbl_dir: Path,
                  class_names: list[str]) -> Counter[str]:
    """
    Count how many images belong to each class across a list of images.

    Args:
        img_paths   : list of image paths to count
        lbl_dir     : directory containing label files
        class_names : ordered list of class names

    Returns:
        Counter mapping class_name → count
    """
    counts: Counter[str] = Counter()
    for img_path in img_paths:
        cls = get_class_for_image(img_path, lbl_dir, class_names)
        if cls:
            counts[cls] += 1
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDING BOX CONVERSIONS
# ══════════════════════════════════════════════════════════════════════════════

def yolo_to_pixels(cx: float, cy: float, w: float, h: float,
                   img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """
    Convert normalised YOLO centre-format box to absolute pixel coordinates.

    YOLO stores coords normalised (0-1 relative to image dimensions).
    Multiply by pixel dimensions to get actual pixel positions.

    Args:
        cx, cy : centre x, centre y (normalised)
        w, h   : width, height (normalised)
        img_w  : image width in pixels
        img_h  : image height in pixels

    Returns:
        (x_min, y_min, x_max, y_max) clamped to image bounds
    """
    x_min = int((cx - w / 2) * img_w)
    y_min = int((cy - h / 2) * img_h)
    x_max = int((cx + w / 2) * img_w)
    y_max = int((cy + h / 2) * img_h)

    return (
        max(0, x_min), max(0, y_min),
        min(img_w, x_max), min(img_h, y_max)
    )


def yolo_to_corners(cx: float, cy: float,
                    w: float,  h: float) -> tuple[float, float, float, float]:
    """
    Convert YOLO centre format to corner format (all normalised).

    Returns: (x_min, y_min, x_max, y_max)
    """
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def corners_to_yolo(x_min: float, y_min: float,
                    x_max: float, y_max: float) -> tuple[float, float, float, float]:
    """
    Convert corner format back to YOLO centre format.
    Clamps output to [0.0, 1.0] to stay within image bounds.

    Returns: (cx, cy, w, h)
    """
    cx = max(0.0, min(1.0, (x_min + x_max) / 2))
    cy = max(0.0, min(1.0, (y_min + y_max) / 2))
    w  = max(0.0, min(1.0,  x_max - x_min))
    h  = max(0.0, min(1.0,  y_max - y_min))
    return cx, cy, w, h


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL COLOURS
# ══════════════════════════════════════════════════════════════════════════════

class Colours:
    """
    ANSI terminal colour codes as class constants.
    Import and use instead of redefining in every script.

    Usage:
        from src.utils import Colours as C
        print(f"{C.GREEN}✓ Done{C.RESET}")
    """
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"

    # Shortcuts matching the old single-letter style
    R  = RESET
    B  = BOLD
    G  = GREEN
    Y  = YELLOW
    RD = RED
    C  = CYAN