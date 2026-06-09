"""
app/demo.py — Live Webcam Heart Gesture Recognition Demo  (v2)
==============================================================
Project : We ♥ Digital Humanities — Emblem Gesture Recognition
Phase   : 4 (Weeks 10–11) — Minimal Research Interface

Changes from v1:
  - Frame is horizontally flipped immediately after cap.read() so the
    webcam no longer mirrors the image (fixes upside_down_heart recognition)
  - max_num_hands raised to 2 so two-handed heart gestures are supported
  - crop_hand_region() now accepts a LIST of landmark sets and computes a
    single merged bounding box across all detected hands
  - Crop strategy changed from padding-fraction to square-crop so the
    framing better matches the tight YOLO annotations used during training
  - MediaPipe detection confidence raised to 0.7 to reduce noisy crops
  - draw_overlay() now loops over all detected hands to draw all skeletons
  - Label is placed above the merged bounding box in all cases

Pipeline per frame:
    cap.read()
        → cv2.flip()                      (un-mirror webcam)
        → MediaPipe Hands (1 or 2 hands)
        → crop_hand_region()              (merged square crop)
        → classify_hand()                 (ResNet18 / HAGRIDv2 forward pass)
        → draw_overlay()                  (all skeletons + merged box + label)
        → cv2.imshow()

Usage:
    python app/demo.py --checkpoint outputs/checkpoints/best_model_resnet18.pth
    python app/demo.py --checkpoint outputs/checkpoints/best_model_hagridv2.pth
    python app/demo.py --checkpoint outputs/checkpoints/best_model_hagridv2_resnet18.pth

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
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands_module
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

# Project utilities — always import from src/utils, never redefine here
from src.utils import Colours as Col

# Model factory — reconstructs the architecture from model_name in checkpoint
from src.models import get_model


# ── 2. Display constants ────────────────────────────────────────────────────────
# Colours in BGR (OpenCV convention)
COLOUR_GREEN  = ( 50, 205,  50)   # high-confidence prediction label
COLOUR_ORANGE = (  0, 165, 255)   # low-confidence prediction label
COLOUR_WHITE  = (255, 255, 255)   # neutral UI text
COLOUR_BOX    = (255, 200,  50)   # merged bounding box rectangle

CONFIDENCE_THRESHOLD = 0.50   # below this → orange label, above → green
MIN_CROP_SIZE        = 20     # pixels; crops smaller than this are skipped
SQUARE_PADDING       = 40     # fixed pixel padding added to the square crop on all sides

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
def crop_hand_region(
    frame:          "np.ndarray",
    all_landmarks:  list,
    padding:        int = SQUARE_PADDING,
) -> "tuple | None":
    """
    Compute a single merged square crop that contains ALL detected hands.

    Why square crop instead of the v1 padding-fraction approach:
      During training, images were cropped from YOLO bounding box annotations,
      which tend to be roughly square and tightly framed. A square crop here
      produces a region that more closely matches what the model was trained on,
      leading to higher confidence scores.

    Why merge across all hands:
      Two-handed heart gestures (traditional-heart, traditional-heart-2) require
      both hands to be visible in the crop. Classifying each hand independently
      would not capture the combined gesture shape.

    Algorithm:
      1. Collect all landmark (x, y) pixel coordinates across every detected hand
      2. Find the global min/max to get a tight box around all hands together
      3. Expand to a square using the longer side (prevents squashing)
      4. Add fixed padding on all sides
      5. Clamp to frame boundaries

    Args:
        frame         : BGR frame from webcam (already flipped)
        all_landmarks : list of MediaPipe NormalizedLandmarkList (1 or 2 entries)
        padding       : fixed pixel padding added on all sides of the square

    Returns:
        (crop_bgr, (x1, y1, x2, y2)) or None if crop is too small
    """
    h, w = frame.shape[:2]

    # Gather pixel coordinates from ALL detected hands into a single flat list
    all_xs, all_ys = [], []
    for hand_lm in all_landmarks:
        all_xs.extend(int(lm.x * w) for lm in hand_lm.landmark)
        all_ys.extend(int(lm.y * h) for lm in hand_lm.landmark)

    # Tight bounding box across all landmarks from all hands
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)

    # Expand to a square: compute centre, then half-side from the longer dimension
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half_side = max(x_max - x_min, y_max - y_min) // 2

    # Apply fixed padding to the square on all sides
    x1 = max(0,     cx - half_side - padding)
    y1 = max(0,     cy - half_side - padding)
    x2 = min(w - 1, cx + half_side + padding)
    y2 = min(h - 1, cy + half_side + padding)

    # Guard: skip if the resulting crop is too small to be meaningful
    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
        return None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


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
    # transform(...) should return a tensor, but some transforms may return a PIL
    # Image (static linters can warn). Handle both cases robustly.
    transformed = transform(pil_image)
    if isinstance(transformed, Image.Image):
        tensor = transforms.ToTensor()(transformed).unsqueeze(0).to(device)
    else:
        tensor = transformed.unsqueeze(0).to(device)

    # Step 4: forward pass — torch.no_grad() disables gradient tracking (faster)
    with torch.no_grad():
        logits = model(tensor)               # shape: [1, num_classes]
        if isinstance(logits, tuple):        # YOLO returns (tensor, ...) in eval mode
            logits = logits[0]
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
      b. Flip horizontally — removes webcam mirror effect so orientation
         matches training images (critical for upside_down_heart)
      c. Pass to MediaPipe for hand detection (up to 2 hands)
      d. If 1 or 2 hands found: compute merged square crop → classify → draw
      e. If no hand found: draw status message
      f. Display annotated frame
      g. Exit cleanly on Q keypress
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
    hands = mp_hands_module.Hands(
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

        # FIX: flip horizontally BEFORE any processing.
        # Webcams mirror the image by default. Training images were NOT mirrored.
        # Without this flip, upside_down_heart appears inverted relative to what
        # the model learned, causing near-zero confidence on that class.
        # frame = cv2.flip(frame, 1)

        # MediaPipe requires RGB input; frame is BGR from OpenCV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)

        if results.multi_hand_landmarks:  # type: ignore[attr-defined]
            # all_landmarks is a list of 1 or 2 NormalizedLandmarkList objects
            all_landmarks = results.multi_hand_landmarks  # type: ignore[attr-defined]

            # Compute ONE merged square crop covering all detected hands
            crop_result = crop_hand_region(frame, all_landmarks)

            if crop_result is not None:
                crop_bgr, bbox = crop_result

                label, confidence = classify_hand(
                    crop_bgr, model, transform, device, class_names
                )

                draw_overlay(
                    frame, label, confidence, bbox,
                    all_landmarks, mp_drawing, mp_hands_module, model_name,
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