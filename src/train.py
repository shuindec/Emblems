"""
src/train.py — Training Loop
==============================
Project : Emblems Heart Gesture Recognition
Author  : Joyce Nguyen

PURPOSE:
    Runs the full training loop for the heart gesture classifier.
    Handles train/val phases per epoch, early stopping, learning
    rate scheduling, checkpoint saving, and metric logging.

    Gradient flow per training batch:
      optimiser.zero_grad()   ← clear previous gradients
      outputs = model(imgs)   ← forward pass
      loss = criterion(...)   ← compute loss
      loss.backward()         ← backpropagation (compute gradients)
      optimiser.step()        ← update weights using gradients

    Val phase uses torch.no_grad() — disables gradient tracking
    entirely, which is faster and uses less memory. No weight
    updates happen in the val phase.

OUTPUTS:
    outputs/checkpoints/best_model_{model_name}.pth  ← best val acc
    outputs/logs/training_log_{model_name}.csv        ← all epoch metrics

USAGE:
    python src/train.py
"""

from __future__ import annotations

import sys
import csv
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── shared utilities ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, get_project_paths, setup_output_dirs, Colours as Col
from src.data_loader import get_dataloaders
from src.models import get_model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LEARNING RATE SCHEDULER
# Cosine annealing smoothly reduces LR from initial value to near-zero
# over the training run. Prevents large oscillating updates near the
# optimum at the end of training.
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimiser: torch.optim.Optimizer,
                    cfg: dict) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Build a learning rate scheduler from config.

    "cosine" → CosineAnnealingLR: LR decays from lr → ~0 over T_max epochs
    "step"   → StepLR: LR multiplied by 0.1 every 10 epochs
    "none"   → no scheduler, constant LR throughout

    Args:
        optimiser : the Adam optimiser from get_model()
        cfg       : loaded config dict

    Returns:
        scheduler object, or None if scheduler = "none"
    """
    scheduler_name = cfg["training"].get("scheduler", "cosine")
    epochs         = cfg["training"]["epochs"]

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=10, gamma=0.1
        )
    return None   # constant LR


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SINGLE EPOCH (TRAIN or VAL)
# Both phases share the same structure — only difference is whether
# gradients are computed and weights updated (train) or not (val).
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model: nn.Module,
              loader: DataLoader,
              criterion: nn.Module,
              optimiser: torch.optim.Optimizer | None,
              device: torch.device,
              is_train: bool) -> tuple[float, float]:
    """
    Run one epoch of either training or validation.

    Train phase (is_train=True):
      - model.train() enables dropout + batch norm in training mode
      - gradients computed, weights updated each batch

    Val phase (is_train=False):
      - model.eval() disables dropout, uses running batch norm stats
      - torch.no_grad() disables gradient tracking (faster, less memory)
      - weights NOT updated

    Args:
        model     : the nn.Module being trained
        loader    : DataLoader for this phase
        criterion : loss function (CrossEntropyLoss with class weights)
        optimiser : Adam optimiser (None for val phase)
        device    : cuda or cpu
        is_train  : True for training, False for validation

    Returns:
        (avg_loss, accuracy) for this epoch
    """
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    # Context manager: enable gradients for train, disable for val
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in loader:
            # Move batch to GPU/CPU
            images = images.to(device)
            labels = labels.to(device)

            # ── Forward pass ──
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]             # YOLO returns (logits, ...) tuple
            loss    = criterion(outputs, labels)

            if is_train and optimiser is not None:
                # ── Backward pass + weight update ──
                optimiser.zero_grad()   # clear gradients from previous batch
                loss.backward()         # compute gradients via backprop
                optimiser.step()        # update weights

            # ── Track metrics ──
            total_loss    += loss.item() * images.size(0)   # weighted by batch size
            preds          = outputs.argmax(dim=1)           # predicted class index
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CHECKPOINT SAVING
# Save the model weights when val accuracy improves.
# Only the BEST model is kept — not every epoch — to save disk space.
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: nn.Module,
                    optimiser: torch.optim.Optimizer,
                    epoch: int,
                    val_acc: float,
                    cfg: dict,
                    save_path: Path) -> None:
    """
    Save model checkpoint when validation accuracy improves.

    Saves:
      - model state dict (weights)
      - optimiser state (for resuming training)
      - epoch number and val accuracy (for logging)
      - config snapshot (so you know exactly what produced this model)

    Args:
        model     : the trained nn.Module
        optimiser : Adam optimiser
        epoch     : current epoch number
        val_acc   : validation accuracy at this epoch
        cfg       : full config dict (saved alongside weights)
        save_path : where to write the .pth file
    """
    torch.save({
        "epoch"      : epoch,
        "val_acc"    : val_acc,
        "model_name" : cfg["training"]["model_name"],
        "num_classes": cfg["training"]["num_classes"],
        "class_names": cfg["dataset"]["class_names"],
        "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "config"     : cfg,
    }, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CSV LOGGER
# Write one row per epoch to a CSV for later analysis and plotting.
# This is your training history — used in evaluate.py to plot curves.
# ══════════════════════════════════════════════════════════════════════════════

def init_log(log_path: Path) -> None:
    """Create the training log CSV with header row."""
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr", "elapsed_sec"])


def append_log(log_path: Path, row: dict) -> None:
    """Append one epoch's metrics to the training log CSV."""
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["epoch"],
            f"{row['train_loss']:.4f}",
            f"{row['train_acc']:.4f}",
            f"{row['val_loss']:.4f}",
            f"{row['val_acc']:.4f}",
            f"{row['lr']:.6f}",
            f"{row['elapsed']:.1f}",
        ])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MAIN TRAINING FUNCTION
# Ties together: data → model → epoch loop → early stopping → checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: dict) -> dict:
    """
    Full training run: loads data and model, runs epoch loop,
    applies early stopping, saves best checkpoint and training log.

    Early stopping logic:
      - Tracks best_val_acc seen so far
      - If val_acc improves → save checkpoint, reset patience counter
      - If val_acc does NOT improve for `patience` epochs → stop training
      - Prevents overfitting on small datasets

    Args:
        cfg : loaded config dict from load_config()

    Returns:
        dict with "best_val_acc", "best_epoch", "checkpoint_path"
    """
    paths      = get_project_paths(cfg)
    model_name = cfg["training"]["model_name"]
    epochs     = cfg["training"]["epochs"]
    patience   = cfg["training"]["early_stopping"]

    # ── output directories ──
    ckpt_dir = paths["output_dir"] / "checkpoints"
    log_dir  = paths["output_dir"] / "logs"
    setup_output_dirs(ckpt_dir, log_dir)

    ckpt_path = ckpt_dir / f"best_model_{model_name}.pth"
    log_path  = log_dir  / f"training_log_{model_name}.csv"

    # ── build data loaders ──
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ── build model ──
    model, optimiser, criterion = get_model(cfg)
    device    = next(model.parameters()).device
    scheduler = build_scheduler(optimiser, cfg)

    # ── initialise log ──
    init_log(log_path)

    print(f"\n{Col.B}{'═'*60}{Col.R}")
    print(f"{Col.B}  Training: {model_name}  |  "
          f"epochs={epochs}  |  patience={patience}{Col.R}")
    print(f"{Col.B}{'═'*60}{Col.R}\n")

    best_val_acc   = 0.0
    best_epoch     = 0
    patience_count = 0   # counts epochs without improvement
    start_time     = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── train phase ──
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion,
            optimiser, device, is_train=True
        )

        # ── val phase ──
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion,
            None, device, is_train=False
        )

        # ── scheduler step ──
        current_lr = optimiser.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        elapsed = time.time() - epoch_start

        # ── print epoch summary ──
        improved = val_acc > best_val_acc
        marker   = f"  {Col.G}✓ best{Col.R}" if improved else ""
        print(
            f"  Epoch {epoch:>3}/{epochs}  |  "
            f"train loss={train_loss:.4f}  acc={train_acc:.3f}  |  "
            f"val loss={val_loss:.4f}  acc={val_acc:.3f}  |  "
            f"lr={current_lr:.5f}  [{elapsed:.1f}s]{marker}"
        )

        # ── log to CSV ──
        append_log(log_path, {
            "epoch"     : epoch,
            "train_loss": train_loss,
            "train_acc" : train_acc,
            "val_loss"  : val_loss,
            "val_acc"   : val_acc,
            "lr"        : current_lr,
            "elapsed"   : elapsed,
        })

        # ── checkpoint if improved ──
        if improved:
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0
            save_checkpoint(model, optimiser, epoch,
                            val_acc, cfg, ckpt_path)
        else:
            patience_count += 1

        # ── early stopping ──
        if patience_count >= patience:
            print(f"\n  {Col.Y}Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs){Col.R}")
            break

    total_time = time.time() - start_time
    print(f"\n{'─'*60}")
    print(f"  {Col.B}Training complete{Col.R}")
    print(f"  Best val accuracy : {Col.G}{best_val_acc:.3f}{Col.R} "
          f"at epoch {best_epoch}")
    print(f"  Checkpoint saved  : {ckpt_path}")
    print(f"  Training log      : {log_path}")
    print(f"  Total time        : {total_time/60:.1f} min")
    print(f"\n  {Col.G}Next step -> python src/evaluate.py{Col.R}")
    print(f"{'─'*60}\n")

    return {
        "best_val_acc"   : best_val_acc,
        "best_epoch"     : best_epoch,
        "checkpoint_path": str(ckpt_path),
        "log_path"       : str(log_path),
    }
def apply_model_overrides(cfg: dict) -> dict:
    """
    Merge per-model hyperparameter overrides into the training config.
 
    Reads the optional `model_overrides` block from config.yaml and,
    if the current model_name has an entry there, selectively replaces
    values in cfg["training"] with the model-specific ones.
 
    This lets each model carry its own ideal batch_size, lr, etc.
    without touching the shared training defaults that other models use.
 
    Example: hagridv2 needs batch_size=8 and lr=0.0003, while resnet18
    works at batch_size=32 and lr=0.001. Both live in config.yaml with
    no code changes required to switch between them.
 
    Args:
        cfg : dict — full config loaded from config.yaml (mutated in place)
 
    Returns:
        cfg : dict — same dict, with training values overridden if applicable
    """
 
    model_name = cfg["training"]["model_name"]
 
    # model_overrides block is optional — skip gracefully if absent
    overrides_block = cfg.get("model_overrides", {})
    model_overrides = overrides_block.get(model_name, {})
 
    if not model_overrides:
        # No override for this model — use training: defaults as-is
        return cfg
 
    # Apply each override key, printing what changed for transparency
    print(f"{Col.CYAN}[Config] Applying overrides for model '{model_name}':{Col.RESET}")
    for key, value in model_overrides.items():
        old_value = cfg["training"].get(key, "<not set>")
        cfg["training"][key] = value
        print(f"{Col.CYAN}  {key}: {old_value} → {value}{Col.RESET}")
 
    return cfg

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg     = load_config()
    cfg     = apply_model_overrides(cfg)
    results = train(cfg)
    print(f"  Final best val accuracy: {results['best_val_acc']:.3f}")