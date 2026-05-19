"""
app/demo.py — Live Webcam Heart Gesture Recognition Demo  (v3)
==============================================================
Project : We ♥ Digital Humanities — Emblem Gesture Recognition
Phase   : 4 (Weeks 10–11) — Minimal Research Interface

Changes from v2:
  - Removed cv2.flip() — training images match natural webcam orientation
  - crop_hand_region() now uses a tighter expansion multiplier (1.2×) instead
    of flat padding so the hand fills the crop more like training images do
  - Added two-hand distance gate: when two hands are far apart (centre
    distance > MAX_HAND_SEPARATION × frame_width), fall back to cropping
    the larger single hand instead of producing a wide sparse merged crop
  - CROP_EXPANSION and MAX_HAND_SEPARATION added to constants block so
    they are easy to tune without touching any function logic

Root cause addressed:
  Training images (Roboflow/web) were close-up photography where the hand
  fills most of the frame. Webcam captures hands smaller within a wider
  scene. The previous flat-padding crop produced too much background context,
  causing the model to see a spatially different composition from training.
  The tighter expansion + distance gate produces crops closer to training
  image composition, improving confidence especially for upside_down_heart
  and the two-handed classes.

Pipeline per frame:
    cap.read()
        → MediaPipe Hands (1 or 2 hands)
        → crop_hand_region()   (tight square; distance gate for 2 hands)
        → classify_hand()      (ResNet18 / HAGRIDv2 forward pass)
        → draw_overlay()       (all skeletons + merged box + label)
        → cv2.imshow()

Usage:
    emblem_env\Scripts\python.exe app/demo.py --checkpoint outputs/checkpoints/best_model_resnet18.pth

Controls:
    Q — quit cleanly

Dependencies (all already in emblem_env):
    torch, torchvision, mediapipe, opencv-python, Pillow
"""

# ── 0. Path setup (must come before any src.* import) ──────────────────────────
import sys
from pathlib import Path

# Insert project root so Python can find src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 1. Standard imports ─────────────────────────────────────────────────────────
import argparse

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Project utilities — always import from src/utils, never redefine here
from src.utils import Colours as Col

# Model factory — reconstructs the architecture from model_name in checkpoint
from src.models import get_model
import torch.nn as nn
import numpy as np


# ── 2. Display constants ────────────────────────────────────────────────────────
# Colours in BGR (OpenCV convention)
COLOUR_GREEN  = ( 50, 205,  50)   # high-confidence prediction label
COLOUR_ORANGE = (  0, 165, 255)   # low-confidence prediction label
COLOUR_WHITE  = (255, 255, 255)   # neutral UI text
COLOUR_BOX    = (255, 200,  50)   # merged bounding box rectangle

CONFIDENCE_THRESHOLD  = 0.50   # below this → orange label, above → green
MIN_CROP_SIZE         = 20     # pixels; crops smaller than this are skipped

# Crop framing — tune these if confidence is still low after testing
CROP_EXPANSION        = 1.0   # half-side multiplier beyond the tight landmark box.
                               # 1.0 = exactly the landmark bounding box.
                               # 1.2 = 20% expansion each side (matches training crop density).
                               # Increase if fingertips are clipped; decrease if too much
                               # background context pulls confidence down.

# Two-hand distance gate
MAX_HAND_SEPARATION   = 0.35   # fraction of frame width. When two hands are detected,
                               # if their centres are further apart than this threshold
                               # the hands are not yet joined into a gesture — fall back
                               # to classifying the larger single hand instead of producing
                               # a wide sparse merged crop that confuses the model.

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICK = 2


# ── 3. Model loader ─────────────────────────────────────────────────────────────
def load_classifier(checkpoint_path: Path, device: torch.device):
    """
    Reconstruct the model architecture and load trained weights from a checkpoint.

    The checkpoint dict (saved by train.py) contains:
      - "model_name"  : string key for get_model() factory
      - "model_state" : state dict (weights only, not full model object)
      - "class_names" : list of class name strings
      - "config"      : full config snapshot used during training

    Steps:
      1. Load the checkpoint dict from disk
      2. Call get_model(cfg) to reconstruct the architecture (same factory as train.py)
      3. Load the saved weights with load_state_dict()
      4. Set to eval mode and move to device

    Args:
        checkpoint_path : Path to the .pth file
        device          : torch.device ("cuda" or "cpu")

    Returns:
        model       : nn.Module ready for inference
        class_names : list[str] read from checkpoint
        model_name  : str (e.g. "resnet18")
    """
    print(f"{Col.CYAN}Loading checkpoint: {checkpoint_path}{Col.RESET}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    model_name  = ckpt["model_name"]
    class_names = ckpt["class_names"]
    cfg         = ckpt["config"]

    print(f"{Col.CYAN}  Model  : {model_name}{Col.RESET}")
    print(f"{Col.CYAN}  Classes: {class_names}{Col.RESET}")

    # get_model() returns (model, optimiser, criterion) — only model needed here
    model, _, _ = get_model(cfg)
    model.load_state_dict(ckpt["model_state"])

    # eval() disables dropout and makes BatchNorm use its running statistics
    model.eval()
    model.to(device)

    print(f"{Col.GREEN}Checkpoint loaded successfully.{Col.RESET}")
    return model, class_names, model_name


# ── 4. Inference-time image transform ───────────────────────────────────────────
def build_transform() -> transforms.Compose:
    """
    Build the torchvision transform pipeline for inference.

    Must exactly match training preprocessing:
      - Resize to 224×224
      - ToTensor: uint8 [0,255] → float32 [0.0, 1.0]
      - Normalise with ImageNet mean and std (matches pretrained backbone)

    Returns:
        transforms.Compose callable: PIL Image → normalised tensor
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean (RGB channels)
            std =[0.229, 0.224, 0.225],   # ImageNet std  (RGB channels)
        ),
    ])


# ── 5. Hand region cropper ───────────────────────────────────────────────────────
def _landmarks_to_bbox(hand_lm, frame_w: int, frame_h: int) -> tuple:
    """
    Convert one hand's MediaPipe landmarks to a pixel bounding box.

    Returns (x_min, y_min, x_max, y_max, cx, cy) where cx/cy is the centre.
    Used both for per-hand bbox computation and for the distance gate.
    """
    xs = [int(lm.x * frame_w) for lm in hand_lm.landmark]
    ys = [int(lm.y * frame_h) for lm in hand_lm.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    return x_min, y_min, x_max, y_max, cx, cy


def _square_crop(frame, cx: int, cy: int, half_side: int) -> "tuple | None":
    """
    Cut a square crop centred on (cx, cy) with the given half-side length,
    expanded by CROP_EXPANSION and clamped to the frame boundaries.

    The CROP_EXPANSION multiplier scales the half-side beyond the tight
    landmark bounding box. At 1.2 the hand fills ~80% of the crop, which
    closely matches the density of training images (close-up web photography).

    Returns (crop_bgr, (x1, y1, x2, y2)) or None if the crop is too small.
    """
    h, w = frame.shape[:2]

    # Scale the half-side by the expansion factor
    expanded = int(half_side * CROP_EXPANSION)

    x1 = max(0,     cx - expanded)
    y1 = max(0,     cy - expanded)
    x2 = min(w - 1, cx + expanded)
    y2 = min(h - 1, cy + expanded)

    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
        return None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def crop_hand_region(
    frame:         "np.ndarray",
    all_landmarks: list,
) -> "tuple | None":
    """
    Produce a tight square crop around the detected hand(s) for classification.

    Two-hand logic — distance gate:
      When two hands are detected, their spatial relationship matters:
        - Hands CLOSE together (centre distance <= MAX_HAND_SEPARATION × frame_w):
            Merge both into a single crop. This is the joined-gesture state
            (traditional-heart, traditional-heart-2). The merged crop contains
            both hands and is what the model was trained to classify.
        - Hands FAR apart (centre distance > MAX_HAND_SEPARATION × frame_w):
            The hands are still approaching each other. A merged crop would be
            wide and sparse (mostly background), which looks nothing like the
            training images. Instead, crop around the larger hand (bigger bbox
            area) as a fallback, which at least gives the model a clean input.

    Single-hand logic:
      Compute a tight square from the landmark bounding box, expanded by
      CROP_EXPANSION so the hand fills the crop similarly to training images.

    Args:
        frame         : BGR frame from webcam
        all_landmarks : list of MediaPipe NormalizedLandmarkList (1 or 2 hands)

    Returns:
        (crop_bgr, (x1, y1, x2, y2)) or None if crop is too small
    """
    h, w = frame.shape[:2]

    if len(all_landmarks) == 1:
        # ── Single hand ───────────────────────────────────────────────────────
        x_min, y_min, x_max, y_max, cx, cy = _landmarks_to_bbox(
            all_landmarks[0], w, h
        )
        half_side = max(x_max - x_min, y_max - y_min) // 2
        return _square_crop(frame, cx, cy, half_side)

    # ── Two hands detected ────────────────────────────────────────────────────
    bbox0 = _landmarks_to_bbox(all_landmarks[0], w, h)
    bbox1 = _landmarks_to_bbox(all_landmarks[1], w, h)

    cx0, cy0 = bbox0[4], bbox0[5]
    cx1, cy1 = bbox1[4], bbox1[5]

    # Euclidean distance between hand centres, normalised by frame width
    centre_distance = ((cx0 - cx1) ** 2 + (cy0 - cy1) ** 2) ** 0.5
    separation      = centre_distance / w

    if separation <= MAX_HAND_SEPARATION:
        # Hands are close enough — merge both into one crop
        all_xs = [bbox0[0], bbox0[2], bbox1[0], bbox1[2]]
        all_ys = [bbox0[1], bbox0[3], bbox1[1], bbox1[3]]
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)
        cx       = (x_min + x_max) // 2
        cy       = (y_min + y_max) // 2
        half_side = max(x_max - x_min, y_max - y_min) // 2
        return _square_crop(frame, cx, cy, half_side)

    else:
        # Hands are far apart — fall back to the larger single hand
        # "Larger" = bigger bounding box area, more likely to be the primary hand
        area0 = (bbox0[2] - bbox0[0]) * (bbox0[3] - bbox0[1])
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        chosen = bbox0 if area0 >= area1 else bbox1

        x_min, y_min, x_max, y_max, cx, cy = chosen
        half_side = max(x_max - x_min, y_max - y_min) // 2
        return _square_crop(frame, cx, cy, half_side)


# ── 6. Classifier ───────────────────────────────────────────────────────────────
def classify_hand(
    crop_bgr:    "np.ndarray",
    model:       "nn.Module",
    transform:   transforms.Compose,
    device:      torch.device,
    class_names: list,
) -> tuple:
    """
    Run the trained classifier on a single hand crop image.

    Conversion chain (each step is required — do not skip):
        BGR numpy  →  RGB numpy  →  PIL Image  →  normalised tensor  →  model

    Why BGR→RGB: OpenCV reads frames as BGR. The ResNet18 backbone was
    pretrained on ImageNet images in RGB format. Feeding BGR would shift
    all colour statistics and degrade accuracy.

    The model outputs raw logits (one per class, unbounded).
    softmax converts them to a probability distribution summing to 1.
    argmax picks the index with the highest probability.

    Args:
        crop_bgr    : hand region as BGR numpy array
        model       : loaded nn.Module in eval mode
        transform   : inference-time transform (build_transform())
        device      : torch.device
        class_names : list of class name strings from checkpoint

    Returns:
        (class_name, confidence) e.g. ("2-finger-heart", 0.872)
    """
    # Step 1: BGR (OpenCV) → RGB (torchvision/PIL convention)
    crop_rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # Step 2: numpy array → PIL Image (required by torchvision transforms)
    pil_image = Image.fromarray(crop_rgb)

    # Step 3: resize + normalise + add batch dimension [C,H,W] → [1,C,H,W]
    transformed = transform(pil_image)
    if isinstance(transformed, Image.Image):
        tensor = transforms.ToTensor()(transformed).unsqueeze(0).to(device)
    else:
        tensor = transformed.unsqueeze(0).to(device)

    # Step 4: forward pass — torch.no_grad() disables gradient tracking (faster)
    with torch.no_grad():
        logits = model(tensor)               # shape: [1, num_classes]
        probs  = F.softmax(logits, dim=1)[0] # shape: [num_classes]

    confidence, class_idx = probs.max(dim=0)
    # Ensure native Python types for indexing and return values to satisfy type checkers
    idx = int(class_idx.item())
    conf = float(confidence.item())
    return class_names[idx], conf

# ── 7. Overlay renderer ──────────────────────────────────────────────────────────
def draw_overlay(
    frame:         "np.ndarray",
    label:         str,
    confidence:    float,
    bbox:          tuple,
    all_landmarks: list,
    mp_drawing,
    mp_hands,
    model_name:    str,
) -> None:
    """
    Draw all visual elements onto the frame in-place (no copy made).

    Elements drawn:
      1. MediaPipe hand skeleton for EACH detected hand (supports 1 or 2 hands)
      2. Single merged bounding box that covers all detected hands
      3. Prediction label + confidence above the merged bounding box
      4. Model name and quit hint in the top-left corner (always visible)

    Label colour coding:
      - Green  : confidence >= CONFIDENCE_THRESHOLD → reliable prediction
      - Orange : confidence <  CONFIDENCE_THRESHOLD → uncertain, review manually

    Args:
        frame         : BGR frame (modified in-place)
        label         : predicted class name
        confidence    : float in [0, 1]
        bbox          : (x1, y1, x2, y2) merged bounding box pixel coordinates
        all_landmarks : list of MediaPipe NormalizedLandmarkList (1 or 2 hands)
        mp_drawing    : mediapipe drawing_utils module
        mp_hands      : mediapipe hands module (for HAND_CONNECTIONS constant)
        model_name    : shown in top-left corner
    """
    # 1. Draw skeleton for every detected hand
    for hand_lm in all_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_lm,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=3),
            mp_drawing.DrawingSpec(color=(100, 100, 255), thickness=2),
        )

    # 2. Merged bounding box
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOUR_BOX, thickness=2)

    # 3. Label above the merged bounding box; clamp so text stays on screen
    label_colour = COLOUR_GREEN if confidence >= CONFIDENCE_THRESHOLD else COLOUR_ORANGE
    label_text   = f"{label}  {confidence * 100:.1f}%"
    text_y       = max(y1 - 10, 20)
    cv2.putText(frame, label_text, (x1, text_y),
                FONT, FONT_SCALE, label_colour, FONT_THICK, cv2.LINE_AA)

    # 4. Top-left HUD — always visible regardless of detection state
    cv2.putText(frame, f"Model: {model_name}", (10, 25),
                FONT, 0.55, COLOUR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit", (10, 50),
                FONT, 0.50, COLOUR_WHITE, 1, cv2.LINE_AA)


def draw_status(frame: "np.ndarray", message: str, model_name: str) -> None:
    """
    Draw a status message (no hand / crop too small) plus the persistent HUD.
    Keeps the display informative when no valid gesture is present.
    """
    cv2.putText(frame, f"Model: {model_name}", (10, 25),
                FONT, 0.55, COLOUR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit", (10, 50),
                FONT, 0.50, COLOUR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, message, (10, 80),
                FONT, FONT_SCALE, COLOUR_WHITE, FONT_THICK, cv2.LINE_AA)


# ── 8. Main loop ─────────────────────────────────────────────────────────────────
def main():
    """
    Entry point — wires all components together and runs the frame loop.

    Frame loop steps:
      a. Read raw frame from webcam
      b. Pass to MediaPipe for hand detection (up to 2 hands)
      c. If hands found: distance gate → tight square crop → classify → draw
      d. If no hand found: draw status message
      e. Display annotated frame
      f. Exit cleanly on Q keypress
    """

    # ── CLI argument ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Live webcam heart gesture demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pth checkpoint (relative to project root or absolute)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        project_root    = Path(__file__).resolve().parent.parent
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        print(f"{Col.RED}Checkpoint not found: {checkpoint_path}{Col.RESET}")
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{Col.CYAN}Device: {device}{Col.RESET}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, class_names, model_name = load_classifier(checkpoint_path, device)
    transform = build_transform()

    # ── MediaPipe Hands ───────────────────────────────────────────────────────
    # max_num_hands=2 so both hands of a two-handed heart gesture are tracked.
    # MediaPipe is the hand DETECTOR only — ResNet18 is the gesture CLASSIFIER.
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,       # video mode: uses temporal tracking
        max_num_hands=2,               # support two-handed gestures
        min_detection_confidence=0.70, # raised from 0.6 to reduce noisy crops
        min_tracking_confidence=0.50,
    )

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"{Col.RED}Error: Could not open webcam.{Col.RESET}")
        sys.exit(1)

    print(f"{Col.GREEN}Webcam opened. Press Q to quit.{Col.RESET}")

    # ── Frame loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"{Col.RED}Warning: Failed to read frame. Retrying...{Col.RESET}")
            continue

        # MediaPipe requires RGB input; frame is BGR from OpenCV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # all_landmarks is a list of 1 or 2 NormalizedLandmarkList objects
            all_landmarks = results.multi_hand_landmarks

            # Compute ONE merged square crop covering all detected hands
            crop_result = crop_hand_region(frame, all_landmarks)

            if crop_result is not None:
                crop_bgr, bbox = crop_result

                label, confidence = classify_hand(
                    crop_bgr, model, transform, device, class_names
                )

                draw_overlay(
                    frame, label, confidence, bbox,
                    all_landmarks, mp_drawing, mp_hands, model_name,
                )
            else:
                draw_status(frame, "Hand too small / too close", model_name)

        else:
            draw_status(frame, "No hand detected", model_name)

        cv2.imshow("We Heart Digital Humanities — Gesture Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Clean shutdown ────────────────────────────────────────────────────────
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"{Col.GREEN}Demo closed cleanly.{Col.RESET}")


# ── Entry point guard ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()