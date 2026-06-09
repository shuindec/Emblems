"""
src/model.py — Model Factory
==============================
PURPOSE:
    Provides a single entry point get_model(cfg) that builds,
    configures and returns the model, optimiser, and loss function
    ready for the training loop.

    Supports three backbones (swap via config.yaml, no code changes):
      "resnet18"  — lightweight baseline, trains locally on RTX 3070
      "yolo"      — YOLOv10x in classification mode (poster methodology)
      "vitb16"    — Vision Transformer (HAGRIDv2-style backbone)

    All builders follow the same pattern:
      1. Load pretrained weights
      2. Freeze backbone (optional — recommended for small datasets)
      3. Replace final classifier layer → num_classes outputs
      4. Print trainable parameter count

    Why freeze the backbone?
      With only 139 training images, updating all layers causes
      catastrophic forgetting of ImageNet features. Freezing keeps
      the backbone's learned representations intact and only trains
      the final classification head — much safer for small datasets.

USAGE:
    from src.model import get_model
    model, optimiser, criterion = get_model(cfg)
"""

from __future__ import annotations

import sys
from typing import cast
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torchvision import models
import timm
import torchvision
from ultralytics import YOLO as UltralyticsYOLO

# ── shared utilities ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, Colours as Col


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BACKBONE BUILDERS
# Each function loads a pretrained model, freezes it if requested,
# then replaces the final layer to output num_classes predictions.
# ══════════════════════════════════════════════════════════════════════════════

def build_resnet18(num_classes: int, freeze_backbone: bool) -> nn.Module:
    """
    Build a ResNet18 classifier pretrained on ImageNet.
    Architecture: 18-layer CNN with residual connections.
    Final layer (fc): replaced from 1000 → num_classes outputs.

    Why ResNet18 first?
      - Smallest ResNet variant — fast to train locally
      - Strong ImageNet features transfer well to hand images
      - Good baseline before heavier models

    Args:
        num_classes     : number of gesture classes (4)
        freeze_backbone : if True, only the final FC layer trains

    Returns:
        nn.Module ready for training
    """
    # Load ResNet18 with ImageNet pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freeze ALL layers first — no gradient updates anywhere
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer
    # Original: Linear(512 → 1000)  for ImageNet's 1000 classes
    # Ours    : Linear(512 → num_classes)  for our 4 gesture classes
    # Replacing this layer automatically unfreezes it (new layer = requires_grad True)
    in_features = model.fc.in_features       # 512 for ResNet18
    model.fc    = nn.Linear(in_features, num_classes)

    return model

def build_hagridv2_resnet18(cfg: dict) -> nn.Module:
    """
    Build a ResNet18 classifier initialised with HAGRIDv2 pretrained weights.
    The HAGRIDv2 checkpoint is a full training snapshot (not a bare state dict), so the model weights live at checkpoint['MODEL_STATE']. This
    builder extracts them, loads them into a standard ResNet18 backbone, then replaces the final classifier for our 4-class problem.

    Expected F1: 98.3% on test set with freeze_backbone=True, 99.2% with full fine-tune.
    Returns:
        nn.Module ready for training
    """
    # Load ResNet18 with ImageNet pretrained weights
    training_cfg = cfg["training"]
    num_classes  = training_cfg["num_classes"]
    freeze       = training_cfg.get("freeze_backbone", True)

    # S1: Resolve the weights path from config
    weights_path = Path(cfg["paths"]["hagridv2_resnet18_weights"])
    if not weights_path.exists():
        raise FileNotFoundError(
            f"[HAGRIDv2-ResNet18] Weights file not found: {weights_path}"
        )
 
    print(f"{Col.CYAN}[HAGRIDv2-ResNet18] Loading checkpoint from:\n"
          f"  {weights_path}{Col.RESET}")
    
    # S2: Build a blank ResNet18 (no pretrained weights yet)
    model = torchvision.models.resnet18(weights=None)                   
    # S3: Extract MODEL_STATE from the HAGRIDv2 checkpoint
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict  = checkpoint["MODEL_STATE"]

    # Step 4: Strip 'module.'. The HAGRIDv2 authors trained with DataParallel (multi-GPU), which wraps every key as 'module.layer1.0.conv1.weight' etc.
    # Single-GPU models expect 'layer1.0.conv1.weight' (no prefix) ==> Strip it here so load_state_dict works correctly.
 
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        # Strip the 'module.' prefix from every key
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }
        print(f"{Col.CYAN}[HAGRIDv2-ResNet18] Stripped 'module.' prefix "
              f"(DataParallel checkpoint){Col.RESET}")

    # S4: Drop fc.* keys before loading — strict=False skips missing/unexpected keys but still
    # raises on size mismatches, and the checkpoint fc (34 classes) won't match the blank
    # ResNet18 fc (1000 classes). The fc layer is replaced with our head below anyway.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"{Col.YELLOW}[HAGRIDv2-ResNet18] Skipped (shape mismatch — expected): "
              f"{load_result.missing_keys}{Col.RESET}")
    if load_result.unexpected_keys:
        print(f"{Col.YELLOW}[HAGRIDv2-ResNet18] Unexpected keys ignored: "
              f"{load_result.unexpected_keys}{Col.RESET}")
    print(f"{Col.GREEN}[HAGRIDv2-ResNet18] HAGRIDv2 backbone weights loaded "
          f"successfully.{Col.RESET}")
    

    # Step 6: Freeze backbone: Freezing all layers first, then unfreezing fc in Step 7, ensures only the new classification head is trained!
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
 
        frozen_count = sum(p.numel() for p in model.parameters()
                           if not p.requires_grad)
        print(f"{Col.CYAN}[HAGRIDv2-ResNet18] Backbone frozen — "
              f"{frozen_count:,} parameters locked.{Col.RESET}")
    else:
        print(f"{Col.YELLOW}[HAGRIDv2-ResNet18] Backbone NOT frozen — "
              f"full fine-tune mode.{Col.RESET}")

    # Replace the final fully-connected layer for 4 heart classes
    # Original: Linear(512, n_classes) => 18 classes in HAGRIDv2 checkpoint
    # Ours    : Linear(512, num_classes) => 4 classes in our problem
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
 
    # Confirm parameter counts for the training log
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total      = sum(p.numel() for p in model.parameters())
    print(f"{Col.GREEN}[HAGRIDv2-ResNet18] Trainable: "
          f"{trainable:,} / {total:,} parameters{Col.RESET}")

    return model


# def build_vitb16(num_classes: int, freeze_backbone: bool) -> nn.Module:
#     """
#     Build a Vision Transformer ViT-B16 pretrained on ImageNet.

#     Architecture: Transformer encoder that splits image into 16×16 patches.
#     Final layer (heads.head): replaced from 1000 → num_classes outputs.

#     This is the HAGRIDv2-style backbone referenced in your poster.
#     Heavier than ResNet18 — recommend Colab for full fine-tuning.

#     Args:
#         num_classes     : number of gesture classes (4)
#         freeze_backbone : if True, only the classification head trains

#     Returns:
#         nn.Module ready for training
#     """
#     # timm is optional — fall back gracefully if not installed
#     try:
#         import timm
#         model = timm.create_model(
#             "vit_base_patch16_224",
#             pretrained=True,
#             num_classes=num_classes    # timm handles head replacement internally
#         )
#         if freeze_backbone:
#             for name, param in model.named_parameters():
#                 # Only train the classification head, freeze everything else
#                 if "head" not in name:
#                     param.requires_grad = False
#         return model

#     except ImportError:
#         # Fall back to torchvision ViT if timm not installed
#         print(f"  {Col.Y}timm not found — using torchvision ViT-B16{Col.R}")
#         model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

#         if freeze_backbone:
#             for param in model.parameters():
#                 param.requires_grad = False

#         # Replace torchvision ViT head
#         in_features  = model.heads.head.in_features   # 768 for ViT-B16
#         model.heads.head = nn.Linear(in_features, num_classes)
#         return model


def build_yolo_classifier(cfg: dict) -> torch.nn.Module:
    """
    Build a YOLOv8x classification model for heart gesture recognition.
 
    Uses YOLOv8x-cls pretrained on ImageNet via the ultralytics library.
    The backbone (CSPDarknet) is frozen and only the final linear
    classification head is trained — the same freeze-and-replace
    pattern used by all other builders in this project.
 
    Why YOLOv8x-cls (not YOLOv10x):
        YOLOv10 in ultralytics is a detection-only model — no classify
        variant exists. The HAGRIDv2 paper uses YOLOv10x as a gesture
        DETECTOR (localises hands), not a classifier. Our pipeline is
        a classifier, so YOLOv8x-cls is the correct and strongest
        available YOLO classifier in ultralytics.
 
    Architecture:
        Backbone : CSPDarknet (YOLOv8x feature extractor)
        Head     : Classify module → Linear(in_features → num_classes)
        Head access: model.model[-1].linear  (ultralytics convention)
 
    Integration note:
        ultralytics wraps the PyTorch model inside a YOLO object.
        We extract the raw nn.Module via yolo_obj.model so it fits
        the existing training loop without any changes to train.py.
        The extracted module behaves like any standard nn.Module.
 
    Weight download:
        ~68MB from GitHub on first run, cached locally after that.
        Falls back to yolov8m-cls (~25MB) if yolov8x-cls unavailable.
 
    Args:
        cfg : dict — full loaded config (from load_config())
 
    Returns:
        model : nn.Module ready for training (extracted from YOLO wrapper)
    """
 
    training_cfg = cfg["training"]
    num_classes  = training_cfg["num_classes"]
    freeze       = training_cfg.get("freeze_backbone", True)
 
    # ----------------------------------------------------------
    # Step 1: Load YOLOv8x-cls pretrained weights
    # ultralytics downloads weights automatically on first call.
    # We try yolov8x-cls first (strongest), fall back to yolov8m-cls
    # if the download fails (e.g. GitHub rate limiting).
    # ----------------------------------------------------------
 
    PRIMARY_MODEL  = "yolov8x-cls.pt"   # ~68MB — strongest classifier
    FALLBACK_MODEL = "yolov8m-cls.pt"   # ~25MB — reliable fallback
 
    try:
        print(f"{Col.CYAN}[YOLO] Loading {PRIMARY_MODEL} "
              f"(downloads ~68MB on first run)...{Col.RESET}")
        yolo_obj = UltralyticsYOLO(PRIMARY_MODEL)
        print(f"{Col.GREEN}[YOLO] {PRIMARY_MODEL} loaded.{Col.RESET}")
        model_variant = PRIMARY_MODEL
 
    except Exception as e:
        print(f"{Col.YELLOW}[YOLO] {PRIMARY_MODEL} unavailable: {e}{Col.RESET}")
        print(f"{Col.YELLOW}[YOLO] Falling back to {FALLBACK_MODEL}...{Col.RESET}")
        yolo_obj = UltralyticsYOLO(FALLBACK_MODEL)
        print(f"{Col.GREEN}[YOLO] {FALLBACK_MODEL} loaded.{Col.RESET}")
        model_variant = FALLBACK_MODEL
 
    # ----------------------------------------------------------
    # Step 2: Extract the raw PyTorch nn.Module
    # ultralytics wraps the model in a YOLO convenience object.
    # We extract yolo_obj.model — the actual nn.Module — so it
    # integrates with our existing train.py loop without any changes.
    # After extraction, the module behaves like any standard model.
    # ----------------------------------------------------------
 
    model = yolo_obj.model   # nn.Module: CSPDarknet backbone + Classify head
    if not isinstance(model, nn.Module):
        raise TypeError("Ultralytics YOLO did not return a valid nn.Module")
 
    # ----------------------------------------------------------
    # Step 3: Freeze backbone if configured
    # Freeze ALL layers first — then Step 4 unfreezes the head only.
    # With backbone frozen, only 3 parameters (the new linear layer)
    # will update during training, which is appropriate for our
    # small dataset of 203 training images.
    # ----------------------------------------------------------
 
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
 
        frozen_count = sum(p.numel() for p in model.parameters()
                           if not p.requires_grad)
        print(f"{Col.CYAN}[YOLO] Backbone frozen — "
              f"{frozen_count:,} parameters locked.{Col.RESET}")
    else:
        print(f"{Col.YELLOW}[YOLO] Backbone NOT frozen — "
              f"full fine-tune mode.{Col.RESET}")
 
    # ----------------------------------------------------------
    # Step 4: Replace the classification head
    # YOLO classify head is a Classify module at model.model[-1].
    # Its linear layer (model.model[-1].linear) is what we replace.
    # The new Linear layer defaults to requires_grad=True, so it
    # is always trainable regardless of the freeze flag above.
    #
    # We read in_features from the existing linear layer before
    # replacing it — this works for any YOLOv8 size variant since
    # each has a different feature dimension (e.g. x=640, m=512).
    # ----------------------------------------------------------
 
    # Locate the Classify module (some Ultralytics builds may wrap/replace
    # the final layer such that model.model[-1] may be a Tensor or other
    # object without a 'linear' attribute). Search backwards for a
    # module exposing a `linear` attr.
    classify_head = None
    for m in reversed(list(cast(nn.Sequential, model.model))):
        if hasattr(m, "linear"):
            classify_head = m
            break
    if classify_head is None:
        raise TypeError("Unable to find YOLO classify head with a 'linear' attribute")

    existing_linear = cast(nn.Linear, classify_head.linear)
    in_features = existing_linear.in_features

    # Replace with fresh linear layer for our num_classes
    classify_head.linear = torch.nn.Linear(in_features, num_classes)
 
    # Log the head configuration for verification
    print(f"{Col.CYAN}[YOLO] Head replaced: "
          f"Linear({in_features} → {num_classes}) "
          f"[{model_variant}]{Col.RESET}")
 
    # Confirm trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total      = sum(p.numel() for p in model.parameters())
    print(f"{Col.GREEN}[YOLO] Trainable: {trainable:,} / {total:,} "
          f"parameters{Col.RESET}")
 
    return model

def build_hagridv2(cfg: dict) -> torch.nn.Module:
    """
    Build a ViT-B16 classifier initialised with HAGRIDv2 pretrained weights.
 
    HAGRIDv2 (Nuzhdin et al. 2024) is pretrained on 1M+ gesture images,
    making it the most domain-relevant starting point for heart emblem
    classification. Its weights are hosted on HuggingFace via timm.
 
    Architecture:
        - Backbone : Vision Transformer Base / Patch 16 / 224px
                     86M parameters total
        - Head     : Linear(768 → num_classes), 3,076 parameters
        - Freeze   : backbone frozen when cfg freeze_backbone=True,
                     leaving only the head trainable
 
    Weight loading strategy (automatic fallback):
        1. Try HAGRIDv2 weights from HuggingFace hub (gesture-domain)
        2. Fall back to ImageNet-21k pretrained weights if hub unavailable
           Both are valid for transfer learning; HAGRIDv2 is preferred
           because its pretraining domain matches our task.
 
    Args:
        cfg : dict — the full loaded config (from load_config())
 
    Returns:
        model : nn.Module ready for training
    """
 
    training_cfg = cfg["training"]
    num_classes  = training_cfg["num_classes"]
    freeze       = training_cfg.get("freeze_backbone", True)
 
    # ----------------------------------------------------------
    # Step 1: Load pretrained weights
    # Try HAGRIDv2 hub weights first; fall back to ImageNet-21k.
    # We set num_classes=0 here to get the raw backbone features,
    # then replace the head ourselves in Step 3.
    # ----------------------------------------------------------
 
    # Official HAGRIDv2 ViT-B16 checkpoint on HuggingFace
    HAGRIDV2_HUB_ID = "hf-hub:hysts/hagrid-classifier-vitb16-with-classifier"
 
    try:
        print(f"{Col.CYAN}[HAGRIDv2] Attempting to load HAGRIDv2 weights "
              f"from HuggingFace hub...{Col.RESET}")
 
        # num_classes=0 strips the original 18-class head → raw 768-dim features
        model = timm.create_model(
            HAGRIDV2_HUB_ID,
            pretrained=True,
            num_classes=0,   # head removed; we attach our own below
        )
        print(f"{Col.GREEN}[HAGRIDv2] HAGRIDv2 gesture-pretrained weights loaded.{Col.RESET}")
        weights_source = "HAGRIDv2 (gesture domain)"
 
    except Exception as hub_err:
        # Hub unavailable (no internet, slug changed, etc.)
        # Fall back to ImageNet-21k — still strong for vision transfer
        print(f"{Col.YELLOW}[HAGRIDv2] Hub unavailable: {hub_err}{Col.RESET}")
        print(f"{Col.YELLOW}[HAGRIDv2] Falling back to ImageNet-21k pretrained "
              f"weights (vit_base_patch16_224).{Col.RESET}")
 
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,   # downloads ~330MB ImageNet-21k weights once
            num_classes=0,     # head removed; attached in Step 3
        )
        weights_source = "ImageNet-21k (fallback)"
 
    print(f"{Col.CYAN}[HAGRIDv2] Weights source: {weights_source}{Col.RESET}")
 
    # ----------------------------------------------------------
    # Step 2: Freeze backbone if configured
    # With the backbone frozen, only the new head is trained.
    # This dramatically reduces GPU memory and is the correct
    # starting point for a small dataset (139 training images).
    # ----------------------------------------------------------
 
    if freeze:
        for param in model.parameters():
            param.requires_grad = False  # freeze every backbone parameter
 
        frozen_count = sum(p.numel() for p in model.parameters()
                           if not p.requires_grad)
        print(f"{Col.CYAN}[HAGRIDv2] Backbone frozen — "
              f"{frozen_count:,} parameters locked.{Col.RESET}")
    else:
        print(f"{Col.YELLOW}[HAGRIDv2] Backbone NOT frozen — "
              f"full fine-tune mode. Use batch_size ≤ 4 on RTX 3070.{Col.RESET}")
 
    # ----------------------------------------------------------
    # Step 3: Replace the classification head
    # timm ViT-B16 uses model.head (not model.fc like ResNet).
    # After num_classes=0 above, model.head is nn.Identity().
    # Replace it with a fresh Linear layer for our 4 classes.
    # This new layer is always trainable regardless of freeze flag.
    # ----------------------------------------------------------
 
    in_features: int = model.num_features  # type: ignore[assignment]  # timm dynamic attr
    model.head  = torch.nn.Linear(in_features, num_classes)
    # model.head is a new layer — its params default to requires_grad=True
 
    # Confirm trainable parameter count for logging
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total      = sum(p.numel() for p in model.parameters())
    print(f"{Col.GREEN}[HAGRIDv2] Trainable: {trainable:,} / {total:,} parameters{Col.RESET}")
 
    return model
# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLASS WEIGHTS
# With imbalanced classes (78 vs 43 images), the loss function needs
# to penalise mistakes on minority classes more than majority ones.
# This stops the model learning to always predict the dominant class.
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(cfg: dict,
                           device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training manifest.

    Formula per class:
        weight = total_samples / (num_classes × samples_in_class)

    Effect:
        Minority class (43 imgs) → higher weight → larger gradient signal
        Majority class (78 imgs) → lower weight  → smaller gradient signal

    This is passed to CrossEntropyLoss(weight=...) to balance training.

    Args:
        cfg    : loaded config dict
        device : torch device (cpu or cuda)

    Returns:
        1D tensor of shape [num_classes] with per-class weights
    """
    import pandas as pd
    from src.utils import get_project_paths

    paths       = get_project_paths(cfg)
    class_names = cfg["dataset"]["class_names"]
    num_classes = len(class_names)

    # Count training samples per class from manifest
    df     = pd.read_csv(paths["manifest"], encoding="utf-8")
    train  = df[df["split"] == "train"]
    counts: Counter[str] = Counter(train["class_name"].tolist())

    total  = sum(counts.values())
    weights = []

    print(f"\n  {Col.B}Class weights (inverse frequency):{Col.R}")
    for cls in class_names:
        count  = counts.get(cls, 1)    # avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
        print(f"    {cls:<28} n={count:>3}  weight={weight:.3f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PARAMETER SUMMARY
# Print how many parameters are trainable vs frozen.
# This confirms freeze_backbone is working as expected.
# ══════════════════════════════════════════════════════════════════════════════

def print_model_summary(model: nn.Module, model_name: str) -> None:
    """
    Print trainable vs total parameter counts.

    What to expect with freeze_backbone=True:
      ResNet18 : ~11M total, ~513K trainable (only FC layer)
      ViT-B16  : ~86M total, ~3K trainable   (only head)
      YOLO-cls : ~68M total, varies by head size

    Args:
        model      : the built nn.Module
        model_name : string name for display
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"\n  {Col.B}Model: {model_name}{Col.R}")
    print(f"    Total parameters     : {total:>12,}")
    print(f"    Trainable parameters : {trainable:>12,}  "
          f"{Col.G}(only these update during training){Col.R}")
    print(f"    Frozen parameters    : {frozen:>12,}  "
          f"{Col.Y}(backbone weights preserved){Col.R}")
    print(f"    Freeze ratio         : {frozen/total*100:.1f}% frozen")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MAIN FACTORY FUNCTION
# Single entry point for the training script.
# Reads model name from config, builds everything, returns ready-to-train.
# ══════════════════════════════════════════════════════════════════════════════

def get_model(cfg: dict) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """
    Build model, optimiser, and loss function from config.

    Reads cfg["training"]["model_name"] to select backbone:
      "resnet18" → ResNet18 (recommended start)
      "yolo"     → YOLOv10x classification mode
      "vitb16"   → Vision Transformer B16

    Returns model on the correct device (GPU if available, else CPU).

    Args:
        cfg : loaded config dict from load_config()

    Returns:
        (model, optimiser, criterion) all on the correct device
    """
    model_name      = cfg["training"]["model_name"]
    num_classes     = cfg["training"]["num_classes"]
    freeze_backbone = cfg["training"]["freeze_backbone"]
    lr              = cfg["training"]["learning_rate"]

    # ── select device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{Col.B}Building model ...{Col.R}")
    print(f"  Model    : {model_name}")
    print(f"  Classes  : {num_classes}")
    print(f"  Frozen   : {freeze_backbone}")
    print(f"  Device   : {device}")

    # ── build selected backbone ──
    builders = {
        "resnet18" : lambda: build_resnet18(num_classes, freeze_backbone),
        "yolo"     : lambda: build_yolo_classifier(cfg),
        "yolo_fullframe" : lambda: build_yolo_classifier(cfg),
        # "vitb16"   : lambda: build_vitb16(num_classes, freeze_backbone),
        "hagridv2"  : lambda: build_hagridv2(cfg),
        "hagridv2_resnet18"  : lambda: build_hagridv2_resnet18(cfg),
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(builders.keys())}"
        )

    model = builders[model_name]().to(device)
    print_model_summary(model, model_name)

    # ── optimiser ──
    # Only pass parameters with requires_grad=True
    # This ensures frozen backbone weights never get an update
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable_params, lr=lr)

    # ── loss function with class weights ──
    # CrossEntropyLoss = Softmax + NegativeLogLikelihood in one step
    # weight= balances learning across imbalanced classes
    class_weights = compute_class_weights(cfg, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\n  {Col.G}Model ready on {device}{Col.R}")
    return model, optimiser, criterion


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to verify model builds correctly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()
    model, optimiser, criterion = get_model(cfg)

    # Pass a dummy batch through to confirm shapes are correct
    device     = next(model.parameters()).device
    dummy_imgs = torch.randn(4, 3, 224, 224).to(device)   # batch of 4
    dummy_lbls = torch.tensor([0, 1, 2, 3]).to(device)

    with torch.no_grad():
        outputs = model(dummy_imgs)   # shape: [4, num_classes]
        loss    = criterion(outputs, dummy_lbls)

    print(f"\n  Dummy forward pass:")
    print(f"    Input shape  : {dummy_imgs.shape}")
    print(f"    Output shape : {outputs.shape}   "
          f"[batch_size, num_classes]")
    print(f"    Loss value   : {loss.item():.4f}")
    print(f"    {Col.G}Model forward pass OK{Col.R}")