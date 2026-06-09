"""
calibrate_temperature.py — Post-hoc temperature scaling for the YOLO classifier
We ♥ Digital Humanities — Automating Emblem Gesture Recognition

Purpose:
    Find the single scalar temperature T that best calibrates the model's
    softmax confidence scores on the validation set, without retraining.

    Temperature scaling (Guo et al. 2017):
        calibrated_probs = softmax(logits / T)

    When T < 1: logits are sharpened  → higher peak confidence
    When T > 1: logits are flattened  → more uniform confidence
    When T = 1: no change (original softmax)

    T is found by minimising Negative Log-Likelihood (NLL) on the val set.
    NLL measures probability calibration — lower is better.
    Accuracy is NOT affected because argmax(logits / T) == argmax(logits).

Outputs:
    outputs/temperature.json  →  {"temperature": <optimal_T>}
    app.py reads this file at startup and applies T in run_inference().

Usage:
    cd C:/Users/suzuk/emblems
    python app/calibrate_temperature.py

Libraries used (all in emblem_env):
    torch, torchvision, scipy, numpy, pyyaml
"""

# ---------------------------------------------------------------------------
# Standard path setup — required by every script in this project
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from src.utils import load_config, get_project_paths, load_class_names
from src.models import get_model
from src.data_loader import HeartGestureDataset   # reuse the project DataLoader


# ===========================================================================
# 1. COLLECT RAW LOGITS FROM THE VALIDATION SET
# ===========================================================================

def collect_logits(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run one full forward pass over the validation set and collect raw logits.

    We collect logits (pre-softmax) so that temperature scaling can be
    applied analytically during optimisation without re-running the model.

    Args:
        model:      loaded nn.Module in eval mode
        val_loader: DataLoader for the validation split
        device:     torch.device matching the model's weights

    Returns:
        all_logits  — (N, 4) float32 tensor of raw logits
        all_labels  — (N,)   int64  tensor of ground-truth class indices
    """
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)

            # ClassificationModel returns (logits, aux) tuple — unpack safely
            output = model(images)
            logits: torch.Tensor = output[0] if isinstance(output, (tuple, list)) else output

            # Move back to CPU immediately to keep GPU memory free
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * val_loader.batch_size} images...")

    all_logits = torch.cat(logits_list, dim=0)   # (N, 4)
    all_labels = torch.cat(labels_list, dim=0)   # (N,)

    print(f"  Collected logits: {all_logits.shape}  labels: {all_labels.shape}")
    return all_logits, all_labels


# ===========================================================================
# 2. NLL OBJECTIVE FUNCTION
# ===========================================================================

def nll_with_temperature(T: float, logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute mean Negative Log-Likelihood after applying temperature T.

    This is the objective function minimised by scipy to find optimal T.
    Lower NLL = better-calibrated confidence scores.

    NLL formula:
        NLL = -mean( log( softmax(logits / T)[true_class] ) )
            = mean( CrossEntropyLoss(logits / T, labels) )

    Args:
        T:      temperature scalar (> 0)
        logits: (N, 4) raw logit tensor
        labels: (N,)   ground-truth class index tensor

    Returns:
        scalar float NLL value for scipy to minimise
    """
    # Guard against invalid T values during optimisation search
    if T <= 0:
        return float("inf")

    scaled_logits = logits / T
    # F.cross_entropy computes log-softmax + NLL in one numerically stable step
    nll = F.cross_entropy(scaled_logits, labels).item()
    return nll


# ===========================================================================
# 3. FIND OPTIMAL TEMPERATURE
# ===========================================================================

def find_optimal_T(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Use Brent's method (scipy.optimize.minimize_scalar) to find the T value
    that minimises NLL over the validation set.

    Search interval [0.05, 5.0] covers the practical range:
        T = 0.05 → near-deterministic (one class gets ~100%)
        T = 5.0  → near-uniform (all classes ~25%)
    The uncalibrated model sits near T = 1.0 by definition.

    Args:
        logits: (N, 4) raw logit tensor from collect_logits()
        labels: (N,)   ground-truth class index tensor

    Returns:
        Optimal temperature value as a float
    """
    result = minimize_scalar(
        fun=nll_with_temperature,
        args=(logits, labels),
        bounds=(0.05, 5.0),
        method="bounded",   # Brent's method with explicit bounds
    )

    if not result.success:
        print(f"  [WARN] Optimisation did not converge cleanly: {result.message}")
        print(f"  [WARN] Using best found T = {result.x:.4f} anyway.")

    return float(result.x)


# ===========================================================================
# 4. CALIBRATION REPORT
# ===========================================================================

def print_calibration_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    T_optimal: float,
) -> None:
    """
    Print a comparison of model behaviour before and after temperature scaling.

    Reports:
        - NLL before and after (lower is better calibration)
        - Mean confidence before and after (confidence = max softmax prob)
        - Accuracy before and after (must be identical — T does not change argmax)

    Args:
        logits:    (N, 4) raw logit tensor
        labels:    (N,)   ground-truth labels
        T_optimal: the optimal temperature found by find_optimal_T()
    """
    with torch.no_grad():
        # ── Before scaling (T = 1.0) ─────────────────────────────────────────
        probs_before   = torch.softmax(logits, dim=1)
        nll_before     = F.cross_entropy(logits, labels).item()
        conf_before    = probs_before.max(dim=1).values.mean().item()
        acc_before     = (probs_before.argmax(dim=1) == labels).float().mean().item()

        # ── After scaling (T = T_optimal) ────────────────────────────────────
        scaled_logits  = logits / T_optimal
        probs_after    = torch.softmax(scaled_logits, dim=1)
        nll_after      = F.cross_entropy(scaled_logits, labels).item()
        conf_after     = probs_after.max(dim=1).values.mean().item()
        acc_after      = (probs_after.argmax(dim=1) == labels).float().mean().item()

    print()
    print("=" * 55)
    print("CALIBRATION REPORT")
    print("=" * 55)
    print(f"  Optimal temperature T     : {T_optimal:.4f}")
    print()
    print(f"  {'Metric':<28} {'Before':>8}  {'After':>8}")
    print(f"  {'-'*44}")
    print(f"  {'NLL (lower = better cal.)':<28} {nll_before:>8.4f}  {nll_after:>8.4f}")
    print(f"  {'Mean confidence':<28} {conf_before:>8.1%}  {conf_after:>8.1%}")
    print(f"  {'Accuracy (must match)':<28} {acc_before:>8.1%}  {acc_after:>8.1%}")
    print("=" * 55)

    if abs(acc_before - acc_after) > 1e-4:
        print("  [WARN] Accuracy changed — this should never happen.")
        print("         Temperature scaling only rescales logits, not argmax.")


# ===========================================================================
# 5. SAVE TEMPERATURE TO JSON
# ===========================================================================

def save_temperature(T: float, output_dir: Path) -> Path:
    """
    Save the optimal temperature to a JSON file that app.py reads at startup.

    Format:  {"temperature": 0.4231}

    Args:
        T:          optimal temperature scalar
        output_dir: project outputs/ directory (from get_project_paths)

    Returns:
        Path to the saved JSON file
    """
    out_path = output_dir / "temperature.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"temperature": round(T, 6)}, f, indent=2)

    print(f"\n  [OK] Temperature saved to: {out_path}")
    print(f"       Restart app.py to apply it.")
    return out_path


# ===========================================================================
# 6. LOAD MODEL — same pattern as app.py
# ===========================================================================

def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> nn.Module:
    """
    Load the YOLO model using the project-standard pattern (get_model + state_dict).
    Identical to load_model_once() in app.py — kept here for script independence.

    Args:
        cfg:       parsed config.yaml dict
        ckpt_path: path to the .pth checkpoint file
        device:    target device (cpu or cuda)

    Returns:
        Loaded nn.Module in eval mode on the target device
    """
    cfg["training"]["model_name"] = "yolo"
    model, _, _ = get_model(cfg)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Support both key conventions used across project checkpoints
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "MODEL_STATE" in checkpoint:
        state_dict = checkpoint["MODEL_STATE"]
    else:
        raise KeyError(f"No weights key in checkpoint. Found: {list(checkpoint.keys())}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ===========================================================================
# 7. MAIN
# ===========================================================================

def main() -> None:
    """
    Full calibration pipeline:
        1. Load config and resolve paths
        2. Load the YOLO model checkpoint
        3. Build the validation DataLoader
        4. Collect raw logits over the val set
        5. Optimise temperature T via NLL minimisation
        6. Print calibration report (NLL, confidence, accuracy before/after)
        7. Save T to outputs/temperature.json
    """
    print("=" * 55)
    print("TEMPERATURE CALIBRATION — YOLO gesture classifier")
    print("=" * 55)

    # ── Config and paths ─────────────────────────────────────────────────────
    cfg   = load_config()
    paths = get_project_paths(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Locate checkpoint — same priority as app.py ──────────────────────────
    for ckpt_name in ("best_model_yolo.pth", "best_model_yolo_fullframe.pth"):
        ckpt_path = Path(paths["output_dir"]) / "checkpoints" / ckpt_name
        if ckpt_path.exists():
            break
    else:
        print("[ERROR] No YOLO checkpoint found. Run train.py first.")
        return

    print(f"  Checkpoint: {ckpt_path.name}")

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = load_model(cfg, ckpt_path, device)
        print(f"  Model loaded on {device}")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        traceback.print_exc()
        return

    # ── Build validation DataLoader ───────────────────────────────────────────
    # use_crop mirrors the model condition: yolo.pth = crop, yolo_fullframe = full
    use_crop = "fullframe" not in ckpt_path.name
    print(f"  Input mode: {'cropped' if use_crop else 'full frame'}")

    class_names = cfg["dataset"]["class_names"]
    image_size  = cfg["preprocessing"]["image_size"]
    split_dir   = Path(paths["split_dir"])

    val_dataset = HeartGestureDataset(
        manifest_path = Path(paths["manifest"]),
        img_dir       = split_dir / "val" / "images",
        lbl_dir       = split_dir / "val" / "labels",
        class_names   = class_names,
        split         = "val",
        image_size    = image_size,
        use_crop      = use_crop,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,    # order does not matter; deterministic pass
        num_workers=0,    # 0 = main process, avoids Windows multiprocessing issues
    )
    print(f"  Validation images: {len(val_dataset)}")

    # ── Collect raw logits ────────────────────────────────────────────────────
    print("\nCollecting logits over validation set ...")
    all_logits, all_labels = collect_logits(model, val_loader, device)

    # ── Find optimal T ────────────────────────────────────────────────────────
    print("\nOptimising temperature ...")
    T_optimal = find_optimal_T(all_logits, all_labels)
    print(f"  Optimal T = {T_optimal:.4f}")

    # ── Report ────────────────────────────────────────────────────────────────
    print_calibration_report(all_logits, all_labels, T_optimal)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_temperature(T_optimal, Path(paths["output_dir"]))


if __name__ == "__main__":
    main()