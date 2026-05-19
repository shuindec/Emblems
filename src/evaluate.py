"""
src/evaluate.py — Model Evaluation
=====================================
Project : Emblems Heart Gesture Recognition
Author  : Joyce Nguyen

PURPOSE:
    Loads the best saved checkpoint and evaluates it on the held-out
    test set (45 images, never seen during training).

    Produces:
      1. Terminal report — per-class precision, recall, F1, accuracy
      2. Confusion matrix — normalised heatmap (saved as PNG)
      3. Training curves  — loss + accuracy over epochs (saved as PNG)
      4. Low-confidence report — predictions where model was uncertain

    Why normalised confusion matrix?
      Raw counts are misleading when classes are imbalanced.
      Normalising by true class size shows recall per class as %,
      making it easy to spot which gestures the model confuses.

USAGE:
    python src/evaluate.py
"""

from __future__ import annotations

import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

# ── shared utilities ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, get_project_paths, setup_output_dirs, Colours as Col
from src.data_loader import get_dataloaders
from src.models import get_model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD CHECKPOINT
# Restore the best model weights saved during training.
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(cfg: dict,
                    model: nn.Module,
                    device: torch.device) -> dict:
    """
    Load the best checkpoint saved during training into the model.

    The checkpoint contains:
      model_state → weights saved at best val accuracy epoch
      val_acc     → the val accuracy when this checkpoint was saved
      epoch       → which epoch produced the best result
      config      → the config used during training

    Args:
        cfg    : current config dict
        model  : the nn.Module to load weights into
        device : cuda or cpu

    Returns:
        checkpoint dict (for accessing metadata like epoch, val_acc)
    """
    paths      = get_project_paths(cfg)
    model_name = cfg["training"]["model_name"]
    ckpt_path  = paths["output_dir"] / "checkpoints" / f"best_model_{model_name}.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}\n"
            f"Run src/train.py first to generate it."
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()   # set to eval mode — disables dropout, uses running BN stats

    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_acc = {checkpoint['val_acc']:.3f})")
    return checkpoint


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — RUN INFERENCE ON TEST SET
# Pass all test images through the model, collect predictions + confidences.
# torch.no_grad() disables gradient tracking — faster, less memory.
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model: nn.Module,
                  test_loader,
                  device: torch.device,
                  class_names: list[str],
                  confidence_threshold: float = 0.60) -> dict:
    """
    Run inference on the entire test set and collect results.

    For each image:
      - Compute softmax probabilities over all classes
      - Take argmax as the predicted class
      - Flag if max probability < confidence_threshold (uncertain prediction)

    Args:
        model                : trained nn.Module (in eval mode)
        test_loader          : DataLoader for test split
        device               : cuda or cpu
        class_names          : list of class name strings
        confidence_threshold : below this → flagged as low confidence

    Returns:
        dict with keys:
          all_preds       : list of predicted class indices
          all_labels      : list of true class indices
          all_probs       : list of max softmax probabilities
          low_confidence  : list of (true_label, pred_label, confidence)
    """
    all_preds      = []
    all_labels     = []
    all_probs      = []
    low_confidence = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)                          # raw logits [B, C]
            probs   = torch.softmax(outputs, dim=1)          # probabilities [B, C]
            confs, preds = probs.max(dim=1)                  # max prob + its index

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(confs.cpu().tolist())

            # Flag uncertain predictions for manual review
            for i in range(len(preds)):
                if confs[i].item() < confidence_threshold:
                    low_confidence.append({
                        "true_label" : class_names[int(labels[i])],
                        "pred_label" : class_names[int(preds[i].item())],
                        "confidence" : confs[i].item(),
                    })

    return {
        "all_preds"     : all_preds,
        "all_labels"    : all_labels,
        "all_probs"     : all_probs,
        "low_confidence": low_confidence,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TERMINAL METRICS REPORT
# Per-class precision, recall, F1 + overall accuracy.
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics_report(results: dict, class_names: list[str]) -> dict:
    """
    Print a formatted per-class metrics table to the terminal.

    Metrics explained:
      Precision : of all images predicted as class X, what % were correct?
                  High precision = few false positives
      Recall    : of all true class X images, what % did the model find?
                  High recall = few false negatives
      F1        : harmonic mean of precision and recall
                  Balances both — use this as your main metric
      Accuracy  : overall % correct across all classes

    Args:
        results     : output from run_inference()
        class_names : list of class name strings

    Returns:
        dict of per-class metrics for further use
    """
    preds  = results["all_preds"]
    labels = results["all_labels"]

    accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(class_names)))
    )
    precision = np.asarray(precision, dtype=np.float64)
    recall    = np.asarray(recall,    dtype=np.float64)
    f1        = np.asarray(f1,        dtype=np.float64)
    support   = np.asarray(support,   dtype=np.intp)

    print(f"\n{'═'*65}")
    print(f"{Col.B}{Col.C}  TEST SET EVALUATION REPORT{Col.R}")
    print(f"{'═'*65}")
    print(f"  Overall Accuracy : {Col.B}{accuracy:.3f}  ({accuracy*100:.1f}%){Col.R}")
    print(f"{'─'*65}")
    print(f"  {'Class':<25}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'N':>4}")
    print(f"{'─'*65}")

    metrics = {}
    for i, cls in enumerate(class_names):
        p, r, f, n = precision[i], recall[i], f1[i], int(support[i])
        flag = f"  {Col.G}✓{Col.R}" if f >= 0.80 else f"  {Col.Y}⚠{Col.R}"
        print(f"  {cls:<25}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {n:>4}{flag}")
        metrics[cls] = {"precision": p, "recall": r, "f1": f, "support": n}

    print(f"{'─'*65}")

    # Macro average (unweighted — treats all classes equally)
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f = f1.mean()
    print(f"  {'macro average':<25}  {macro_p:>6.3f}  {macro_r:>6.3f}  "
          f"{macro_f:>6.3f}")
    print(f"{'═'*65}\n")

    # Low confidence report
    if results["low_confidence"]:
        print(f"  {Col.Y}Low-confidence predictions "
              f"({len(results['low_confidence'])} images):{Col.R}")
        for item in results["low_confidence"]:
            print(f"    true={item['true_label']:<22}  "
                  f"pred={item['pred_label']:<22}  "
                  f"conf={item['confidence']:.3f}")
    else:
        print(f"  {Col.G}No low-confidence predictions — "
              f"model is confident on all test images.{Col.R}")

    return {"accuracy": accuracy, "per_class": metrics,
            "macro_f1": macro_f}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CONFUSION MATRIX FIGURE
# Normalised so each row sums to 1 — shows recall per class as %.
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(results: dict,
                          class_names: list[str],
                          save_path: Path) -> None:
    """
    Plot and save a normalised confusion matrix heatmap.

    Rows = true class, Columns = predicted class.
    Each cell = fraction of true-class images predicted as that column class.
    Diagonal = recall per class (correct predictions).
    Off-diagonal = confusions (e.g. traditional-heart predicted as traditional-heart-2).

    Normalised (not raw counts) so minority classes are fairly represented.

    Args:
        results     : output from run_inference()
        class_names : list of class name strings
        save_path   : output PNG path
    """
    cm = confusion_matrix(results["all_labels"], results["all_preds"])

    # Normalise each row by its true class total → values in [0, 1]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Short labels for the matrix (long names get crowded)
    short_names = [
        "2-finger",
        "trad-heart",
        "trad-heart-2",
        "upside-dn",
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#f0f0f0")
    ax.set_facecolor("#f0f0f0")

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (fraction)", color="#949aaf", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#949aaf")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#949aaf")

    # Tick labels
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=30, ha="right",
                       color="#949aaf", fontsize=9)
    ax.set_yticklabels(short_names, color="#949aaf", fontsize=9)

    # Annotate each cell with percentage + raw count
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            frac  = cm_norm[i, j]
            count = cm[i, j]
            colour = "white" if frac > 0.495 else "#b8bcc9"
            ax.text(j, i, f"{frac:.0%}\n({count})",
                    ha="center", va="center",
                    color=colour, fontsize=9, fontweight="bold")

    ax.set_xlabel("Predicted class", color="#949aaf", fontsize=10)
    ax.set_ylabel("True class",      color="#949aaf", fontsize=10)
    ax.set_title("Confusion Matrix — Test Set (normalised)",
                 color="#949aaf", fontsize=12, pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAINING CURVES
# Plot loss and accuracy over epochs from the training log CSV.
# Shows if the model overfit, underfit, or converged cleanly.
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(cfg: dict, save_path: Path) -> None:
    """
    Load training_log_{model_name}.csv and plot:
      Left panel  → train loss + val loss over epochs
      Right panel → train accuracy + val accuracy over epochs

    The gap between train and val curves reveals overfitting:
      Small gap → model generalises well
      Large gap → overfitting (val worse than train)

    Args:
        cfg       : loaded config dict
        save_path : output PNG path
    """
    paths      = get_project_paths(cfg)
    model_name = cfg["training"]["model_name"]
    log_path   = paths["output_dir"] / "logs" / f"training_log_{model_name}.csv"

    if not log_path.exists():
        print(f"  {Col.Y}Training log not found — skipping curve plot.{Col.R}")
        return

    df = pd.read_csv(log_path, encoding="utf-8")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d0f14")

    for ax in (ax_loss, ax_acc):
        ax.set_facecolor("#13161e")
        ax.spines[:].set_color("#1e2330")
        ax.tick_params(colors="#c8d0e8")
        ax.yaxis.grid(True, color="#1e2330", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("Epoch", color="#c8d0e8", fontsize=9)

    # ── loss curves ──
    ax_loss.plot(df["epoch"], df["train_loss"],
                 color="#2ec4b6", linewidth=2, label="Train loss")
    ax_loss.plot(df["epoch"], df["val_loss"],
                 color="#ffd166", linewidth=2, label="Val loss",
                 linestyle="--")
    ax_loss.set_title("Loss", color="white", fontsize=11)
    ax_loss.set_ylabel("Loss", color="#c8d0e8", fontsize=9)
    ax_loss.legend(facecolor="#13161e", edgecolor="#1e2330",
                   labelcolor="white", fontsize=8)

    # ── accuracy curves ──
    ax_acc.plot(df["epoch"], df["train_acc"],
                color="#2ec4b6", linewidth=2, label="Train acc")
    ax_acc.plot(df["epoch"], df["val_acc"],
                color="#ffd166", linewidth=2, label="Val acc",
                linestyle="--")

    # Mark best val accuracy epoch
    best_epoch = df["epoch"].iloc[int(df["val_acc"].argmax())].item()
    best_acc   = df["val_acc"].max().item()
    ax_acc.axvline(best_epoch, color="#e63946", linewidth=1,
                   linestyle=":", label=f"Best epoch ({int(best_epoch)})")
    ax_acc.annotate(f"{best_acc:.3f}",
                    xy=(best_epoch, best_acc),
                    xytext=(best_epoch + 0.5, best_acc - 0.05),
                    color="#e63946", fontsize=9)

    ax_acc.set_title("Accuracy", color="white", fontsize=11)
    ax_acc.set_ylabel("Accuracy", color="#c8d0e8", fontsize=9)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(facecolor="#13161e", edgecolor="#1e2330",
                  labelcolor="white", fontsize=8)

    fig.suptitle(f"Training Curves — {model_name}",
                 color="white", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg         = load_config()
    paths       = get_project_paths(cfg)
    class_names = cfg["dataset"]["class_names"]
    model_name  = cfg["training"]["model_name"]

    setup_output_dirs(paths["figures_dir"])

    print(f"\n{Col.B}Evaluating: {model_name}{Col.R}")

    # ── load model + checkpoint ──
    model, _, _ = get_model(cfg)
    device      = next(model.parameters()).device
    checkpoint  = load_checkpoint(cfg, model, device)

    # ── get test loader ──
    _, _, test_loader = get_dataloaders(cfg)

    # ── run inference ──
    print(f"\n  Running inference on test set ...")
    results = run_inference(model, test_loader, device, class_names)

    # ── print metrics ──
    metrics = print_metrics_report(results, class_names)

    # ── confusion matrix ──
    print(f"{Col.B}Saving figures ...{Col.R}")
    plot_confusion_matrix(
        results, class_names,
        paths["figures_dir"] / f"confusion_matrix_{model_name}.png"
    )

    # ── training curves ──
    plot_training_curves(
        cfg,
        paths["figures_dir"] / f"training_curves_{model_name}.png"
    )

    # ── final summary ──
    print(f"\n{'═'*55}")
    print(f"{Col.B}{Col.C}  EVALUATION COMPLETE{Col.R}")
    print(f"{'═'*55}")
    print(f"  Model          : {model_name}")
    print(f"  Test accuracy  : {Col.G}{metrics['accuracy']:.3f}"
          f"  ({metrics['accuracy']*100:.1f}%){Col.R}")
    print(f"  Macro F1       : {Col.G}{metrics['macro_f1']:.3f}{Col.R}")
    print(f"  Val accuracy   : {checkpoint['val_acc']:.3f} "
          f"(epoch {checkpoint['epoch']})")
    print(f"\n  Figures saved to : {paths['figures_dir']}")
    print(f"\n  {Col.G}Next step -> app/app.py  (Gradio demo){Col.R}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()