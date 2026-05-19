"""
src/data_loader.py — Heart Gesture Dataset & DataLoaders
==========================================================

PURPOSE:
Provides the PyTorch Dataset class and DataLoader factory for
the heart gesture classification task.

Pipeline per image (inside __getitem__):
    1. Load full image from disk
    2. Read bounding box from YOLO label file
    3. Crop the hand region using the bounding box
    4. Resize crop to 224×224
    5. Apply augmentation (train only) or just normalise (val/test)
    6. Return (image_tensor, class_index)

Why crop before classifying?
    With only ~139 training images, giving the model a clean
    hand crop (not the full cluttered background) significantly
    improves the signal-to-noise ratio for learning.

USAGE:
    from src.data_loader import get_dataloaders, verify_dataloader
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    verify_dataloader(train_loader, cfg)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── shared utilities ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_config, get_project_paths,
    get_label_path, read_annotations,
    yolo_to_pixels, Colours as Col,
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — TRANSFORMS
# Define what happens to an image before it enters the model.
# Train gets augmentation to artificially expand variety.
# Val/Test get ONLY normalisation — augmenting eval data corrupts metrics.
# ══════════════════════════════════════════════════════════════════════════════

# ImageNet mean/std — standard for models pretrained on ImageNet
# (HAGRIDv2, ViT-B16, ResNet all expect this normalisation)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Return the correct torchvision transform pipeline for a given split.

    Train transforms (augmentation):
      RandomHorizontalFlip  → mirror the gesture (doubles effective data)
      RandomRotation        → handle tilted hands (±20°)
      ColorJitter           → handle lighting variation across images
      Resize + CenterCrop   → standardise to model input size
      ToTensor + Normalise  → convert to float tensor, apply ImageNet stats

    Val/Test transforms (no augmentation):
      Resize + CenterCrop   → standardise size only
      ToTensor + Normalise  → same normalisation as train (required)

    Args:
        split      : "train", "val", or "test"
        image_size : target size (default 224 for most pretrained models)

    Returns:
        torchvision.transforms.Compose pipeline
    """
    # Shared final steps — always applied regardless of split
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if split == "train":
        # Augmentation only for training — placed BEFORE resize/normalise
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(aug + base)

    # Val and test: base pipeline only
    return transforms.Compose(base)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATASET CLASS
# PyTorch Dataset: tells the DataLoader how to load ONE item at index idx.
# DataLoader calls __getitem__ in parallel across workers to fill batches.
# ══════════════════════════════════════════════════════════════════════════════

class HeartGestureDataset(Dataset):
    """
    PyTorch Dataset for heart gesture classification.

    Reads from split_manifest.csv which maps:
        filename → class_name → split

    For each item (__getitem__):
      1. Load full image (PIL)
      2. Look up bounding box in label .txt
      3. Crop to hand region
      4. Apply transforms (augment if train, normalise if val/test)
      5. Return (tensor [3, H, W], class_index int)
    """

    def __init__(self,    
                 manifest_path: Path,
                 img_dir: Path,
                 lbl_dir: Path,
                 class_names: list[str],
                 split: str,
                 image_size: int = 224,
                 use_crop: bool = True) -> None:
        """
        Args:
            manifest_path : path to split_manifest.csv
            img_dir       : directory containing images for this split
            lbl_dir       : directory containing label .txt files for this split
            class_names   : ordered list of class names (index = class label)
            split         : "train", "val", or "test"
            image_size    : resize target (default 224)
            use_crop      : whether to crop images to hand region (default True)
        """
        self.img_dir     = img_dir
        self.lbl_dir     = lbl_dir
        self.split       = split
        self.image_size  = image_size
        self.use_crop    = use_crop
        self.transform   = get_transforms(split, image_size)

        # Map class name → integer index (used as training label)
        # e.g. {"2-finger-heart": 0, "traditional-heart": 1, ...}
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        # Load only rows for this split from the manifest
        df = pd.read_csv(manifest_path, encoding="utf-8")
        self.records = df[df["split"] == split].reset_index(drop=True)

        print(f"  [{split:>5}] {len(self.records):>4} images loaded  "
              f"| classes: {list(self.class_to_idx.keys())}")

    def __len__(self) -> int:
        """Return total number of images in this split."""
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]: # transform converts PIL → Tensor
        """
        Load, crop, transform and return one image + its label.

        Steps:
          1. Get filename + class from manifest row
          2. Load full image with PIL (handles mixed formats reliably)
          3. Look up bounding box from label .txt file
          4. Convert normalised YOLO box → pixel coordinates
          5. Crop image to bounding box region
          6. Apply transforms (augment/normalise)
          7. Return (tensor, class_index)

        Args:
            idx : integer index into this split's records

        Returns:
            tuple of (image_tensor [3, H, W], class_index int)
        """
        row        = self.records.iloc[idx]
        filename   = row["filename"]
        class_name = row["class_name"]
        label      = self.class_to_idx[class_name]

        # ── Step 1: Load full image ──
        img_path = self.img_dir / filename
        image    = Image.open(img_path).convert("RGB")   # PIL, always RGB
        img_w, img_h = image.size                        # PIL: (width, height)

        if self.use_crop: #checker for config option to disable cropping (for ablation)

            # ── Step 2: Read bounding box from label file ──
            lbl_path    = get_label_path(img_path, self.lbl_dir)
            annotations = read_annotations(lbl_path)

            if annotations:
                # Take the first (and after cleaning, only) annotation
                _, (cx, cy, w, h) = annotations[0]

                # ── Step 3: Convert YOLO normalised → pixel coordinates ──
                x_min, y_min, x_max, y_max = yolo_to_pixels(
                    cx, cy, w, h, img_w, img_h
                )

                # ── Step 4: Crop to hand region ──
                # Add a small padding (5% of box size) to avoid cutting edge pixels
                pad_x = int((x_max - x_min) * 0.05)
                pad_y = int((y_max - y_min) * 0.05)
                x_min = max(0,     x_min - pad_x)
                y_min = max(0,     y_min - pad_y)
                x_max = min(img_w, x_max + pad_x)
                y_max = min(img_h, y_max + pad_y)

                image = image.crop((x_min, y_min, x_max, y_max))
            else:
                # No bounding box found — fall back to full image
                # This should not happen after cleaning, but guards against it
                pass
        else: use_crop=False

        # ── Step 5: Apply transforms (augment or normalise) ──
        tensor = self.transform(image)   # shape: [3, image_size, image_size]

        return tensor, label # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATALOADER FACTORY
# Wraps all three Dataset instances in DataLoaders.
# DataLoader handles batching, shuffling, and parallel loading.
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return train, val, and test DataLoaders.

    Reads all paths and settings from config dict (no hardcoded values).
    Train DataLoader shuffles every epoch — val/test do not.

    Args:
        cfg : loaded config dict from load_config()

    Returns:
        (train_loader, val_loader, test_loader)
    """
    paths       = get_project_paths(cfg)
    class_names = cfg["dataset"]["class_names"]
    image_size  = cfg["preprocessing"]["image_size"]
    batch_size  = cfg["training"]["batch_size"]
    use_crop = cfg["training"].get("use_crop", True) # use_crop defaults to True — only HAGRIDv2 full-frame models set False. Controlled via config.yaml
    manifest    = paths["manifest"]
    split_dir   = paths["split_dir"]

    print(f"\n{Col.B}Building DataLoaders ...{Col.R}")
    print(f"  Manifest   : {manifest}")
    print(f"  Split dir  : {split_dir}")
    print(f"  Image size : {image_size}×{image_size}")
    print(f"  Batch size : {batch_size}\n")

    loaders = {}
    for split in ("train", "val", "test"):
        dataset = HeartGestureDataset(
            manifest_path = manifest,
            img_dir       = split_dir / split / "images",
            lbl_dir       = split_dir / split / "labels",
            class_names   = class_names,
            split         = split,
            image_size    = image_size,
            use_crop      = use_crop,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == "train"),  # shuffle train, not val/test
            num_workers = 0,    # 0 = load on main thread (safe on Windows)
            pin_memory  = torch.cuda.is_available(),  # faster GPU transfer
        )

    return loaders["train"], loaders["val"], loaders["test"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SANITY CHECK
# Quick verification that the DataLoader produces correct shapes and labels.
# Run this once after building loaders — catches transform/shape bugs early.
# ══════════════════════════════════════════════════════════════════════════════

def verify_dataloader(loader: DataLoader, cfg: dict) -> None:
    """
    Load one batch and print shapes + label distribution.
    Confirms the DataLoader pipeline is working end-to-end.

    Expected output:
      Batch images shape : torch.Size([32, 3, 224, 224])
      Batch labels shape : torch.Size([32])
      Labels in batch    : tensor([0, 1, 2, 3, ...])

    Args:
        loader : any DataLoader (train recommended)
        cfg    : loaded config dict
    """
    class_names = cfg["dataset"]["class_names"]

    images, labels = next(iter(loader))

    print(f"\n{Col.B}DataLoader Verification:{Col.R}")
    print(f"  Batch images shape : {images.shape}")
    print(f"  Batch labels shape : {labels.shape}")
    print(f"  Label indices      : {labels.tolist()}")
    print(f"  Label names        : "
          f"{[class_names[i] for i in labels.tolist()]}")
    print(f"  Image dtype        : {images.dtype}")
    print(f"  Image value range  : [{images.min():.3f}, {images.max():.3f}]")
    print(f"  {Col.G}DataLoader OK{Col.R}\n")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run this file directly to verify everything works
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    verify_dataloader(train_loader, cfg)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")