"""
app/app.py — Corpus Gesture Classification Demo
We ♥ Digital Humanities — Automating Emblem Gesture Recognition

Purpose:
    A Gradio interface for researchers to upload an image and receive a
    gesture class prediction from the best fine-tuned YOLO model.

Key design decision — checkpoint loading:
    build_yolo_classifier() in src/models.py already replaces the final
    linear head from 1000 → 4 classes before returning. So the correct
    loading sequence is:
        1. Call get_model(_cfg) which runs build_yolo_classifier() and
           returns an nn.Module with the correct [4, 1280] head.
        2. Load checkpoint["model_state"] into that nn.Module directly.
    Do NOT use UltralyticsYOLO("yolov8x-cls.pt") and then load_state_dict —
    that builds a fresh [1000, 1280] head and causes a size mismatch error.

Processing pipeline for each uploaded image:
    1. Researcher uploads an image via the Gradio UI
    2. Image is saved to a temp file
    3. A matching YOLO label (.txt) is searched for bounding-box crop info
    4. If found  → hand region is cropped (mirrors training data_loader.py)
       If missing → full image used as fallback
    5. Resize to 224x224 and normalise with ImageNet statistics
    6. Forward pass through the loaded nn.Module
    7. Softmax over logits → probability distribution over 4 classes
    8. Return: cropped input image, top class + confidence, bar chart

Libraries used (all in emblem_env — no new installs needed):
    gradio, torch, torchvision, ultralytics, Pillow, matplotlib, pyyaml
"""

# ---------------------------------------------------------------------------
# Standard path setup — every script in this project needs this so that
# src.utils and src.models can be imported from the project root
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
import io
import json
import tempfile
import traceback
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")          # headless — no display server needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Project utilities — always import from here, never redefine
from src.utils import load_config, get_project_paths, load_class_names
from src.models import get_model   # same factory used by train.py / evaluate.py

# ---------------------------------------------------------------------------
# Module-level globals — loaded once at startup, reused for every request
# ---------------------------------------------------------------------------
_model: Optional[nn.Module] = None   # the loaded torch nn.Module (4-class head)
_class_names: list[str]     = []     # ["2-finger-heart", ...]
_cfg: dict                  = {}     # full parsed config.yaml dict
_temperature: float         = 1.0   # calibration scalar; 1.0 = no scaling


# ---------------------------------------------------------------------------
# Preprocessing transform — must exactly match data_loader.py training setup
# (ImageNet mean/std, resize to 224x224)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ===========================================================================
# 1. MODEL LOADING
# ===========================================================================

def load_model_once() -> bool:
    """
    Rebuild the YOLO model architecture via get_model() — which already
    replaces the classifier head to num_classes=4 — then load the saved
    fine-tuned weights from the project-format checkpoint.

    Why get_model() and NOT UltralyticsYOLO("yolov8x-cls.pt"):
        UltralyticsYOLO builds a fresh 1000-class head (ImageNet default).
        build_yolo_classifier() inside get_model() replaces it with a
        4-class head before returning, matching the shape saved in the
        checkpoint. Loading state_dict into a mismatched head raises a
        size mismatch RuntimeError.

    Returns:
        True  — model and class names loaded successfully
        False — checkpoint missing or load error (full traceback printed)
    """
    global _model, _class_names, _cfg, _temperature

    if _model is not None:
        return True   # already loaded — skip

    try:
        # ── Resolve paths from config ────────────────────────────────────────
        _cfg   = load_config()
        paths  = get_project_paths(_cfg)

        # ── Find checkpoint — prefer crop-trained, fall back to fullframe ────
        # The crop-trained model gives sharper confidence on hand-region input.
        for ckpt_name in ("best_model_yolo.pth", "best_model_yolo_fullframe.pth"):
            ckpt_path = Path(paths["output_dir"]) / "checkpoints" / ckpt_name
            if ckpt_path.exists():
                break
        else:
            print("[ERROR] No YOLO checkpoint found in outputs/checkpoints/")
            print("        Expected: best_model_yolo.pth or best_model_yolo_fullframe.pth")
            return False

        # ── Build architecture with 4-class head via project model factory ───
        # get_model() reads cfg["training"]["model_name"] to select the builder.
        _cfg["training"]["model_name"] = "yolo"
        model, _, _ = get_model(_cfg)   # returns (nn.Module, optimiser, criterion)

        # ── Load fine-tuned weights from the project-format checkpoint ───────
        # weights_only=False required in PyTorch >= 2.0 for full dict loading.
        print(f"[INFO] Loading weights from {ckpt_path.name} ...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Detect key convention: train.py uses 'model_state' (lowercase)
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "MODEL_STATE" in checkpoint:
            state_dict = checkpoint["MODEL_STATE"]
        else:
            print(f"[ERROR] No weights key found in checkpoint.")
            print(f"        Available keys: {list(checkpoint.keys())}")
            return False

        # load_state_dict succeeds because model already has [4, 1280] head
        model.load_state_dict(state_dict)
        model.eval()   # inference mode — disables dropout / batchnorm train
        _model = model

        # ── Load class names from data.yaml ──────────────────────────────────
        data_yaml = Path(paths["data_dir"]) / _cfg["dataset"]["yaml_file"]
        _class_names = load_class_names(data_yaml)

        # ── Load temperature if calibration has been run ────────────────
        # calibrate_temperature.py writes outputs/temperature.json.
        # If the file is absent, T defaults to 1.0 (no scaling applied).
        temp_path = Path(paths["output_dir"]) / "temperature.json"
        if temp_path.exists():
            with open(temp_path, encoding="utf-8") as f:
                _temperature = float(json.load(f)["temperature"])
            print(f"[OK] Temperature   : {_temperature:.4f} (from temperature.json)")
        else:
            _temperature = 1.0
            print("[INFO] No temperature.json — using T=1.0 (uncalibrated)")

        print(f"[OK] Model loaded  : {ckpt_path.name}")
        print(f"[OK] Classes       : {_class_names}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()   # always show full stack trace
        return False


# ===========================================================================
# 2. IMAGE PRE-PROCESSING
# ===========================================================================

def find_label_file(image_path: Path) -> Optional[Path]:
    """
    Search for the YOLO label (.txt) that corresponds to this image.

    Checks two standard YOLO directory layouts:
        Layout A: .../images/<name>.jpg  ->  .../labels/<name>.txt
        Layout B: label sits alongside the image in the same folder

    Args:
        image_path: path to the uploaded image

    Returns:
        Path to the label file if it exists, otherwise None
    """
    # Layout A — sibling 'labels/' directory one level up from 'images/'
    sibling = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    if sibling.exists():
        return sibling

    # Layout B — label in the same directory as the image
    flat = image_path.with_suffix(".txt")
    if flat.exists():
        return flat

    return None


def crop_from_label(image: Image.Image, label_path: Path) -> Optional[Image.Image]:
    """
    Read the first bounding box from a YOLO label file and crop the image.

    YOLO format per line:  class_id  cx  cy  w  h   (all normalised 0-1)

    Args:
        image:      full-frame PIL Image
        label_path: path to the matching .txt label file

    Returns:
        Cropped PIL Image of the hand region, or None if parsing fails
    """
    try:
        with open(label_path, encoding="utf-8") as f:
            line = f.readline().strip()

        if not line or len(line.split()) < 5:
            return None

        parts = line.split()
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        img_w, img_h = image.size

        # Convert normalised centre+size -> absolute pixel corners, clamped to bounds
        x1 = max(0,     int((cx - bw / 2) * img_w))
        y1 = max(0,     int((cy - bh / 2) * img_h))
        x2 = min(img_w, int((cx + bw / 2) * img_w))
        y2 = min(img_h, int((cy + bh / 2) * img_h))

        if x2 <= x1 or y2 <= y1:
            return None

        return image.crop((x1, y1, x2, y2))

    except Exception as e:
        print(f"[WARN] Label parse failed ({label_path}): {e}")
        return None


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, Image.Image, str]:
    """
    Full preprocessing pipeline — mirrors HeartGestureDataset in data_loader.py.

    Steps:
        1. Open image as RGB (handles RGBA, palette, grayscale safely)
        2. Look for a YOLO bounding-box label -> crop hand region if found
        3. Apply resize (224) + ToTensor + ImageNet normalisation
        4. Add batch dimension: (3,224,224) -> (1,3,224,224)

    Args:
        image_path: path to the temp-saved uploaded image

    Returns:
        tensor      -- (1,3,224,224) float32 tensor ready for the model
        display_img -- PIL Image of the region fed to the model (for UI display)
        status_msg  -- human-readable string describing the crop decision
    """
    image = Image.open(image_path).convert("RGB")

    label_path = find_label_file(image_path)
    cropped: Optional[Image.Image] = None
    status_msg = ""

    if label_path is not None:
        cropped = crop_from_label(image, label_path)
        status_msg = (
            "Cropped from bounding box label"
            if cropped is not None
            else "Label found but crop failed — using full image"
        )
    else:
        status_msg = "No label file found — using full image"

    input_image: Image.Image = cropped if cropped is not None else image
    display_img = input_image.copy()   # preserve pre-normalisation copy for display

    # _transform returns torch.Tensor; type: ignore avoids Pylance Compose ambiguity
    tensor: torch.Tensor = _transform(input_image)  # type: ignore[assignment]
    tensor = tensor.unsqueeze(0)    # (3,224,224) -> (1,3,224,224)

    return tensor, display_img, status_msg


# ===========================================================================
# 3. INFERENCE
# ===========================================================================

def run_inference(tensor: torch.Tensor) -> dict[str, float]:
    """
    Run a forward pass and return per-class probabilities.

    Device handling:
        The tensor is always built on CPU. The model may be on CUDA.
        tensor.to(model_device) ensures they match before the forward pass.

    Tuple handling:
        The ultralytics ClassificationModel returns a tuple (logits, aux)
        rather than a bare tensor. We extract index [0] which is always the
        (1, num_classes) logit tensor. The isinstance guard makes this safe
        for any model that returns a plain tensor instead.

    Args:
        tensor: (1,3,224,224) normalised float32 tensor (on CPU)

    Returns:
        Dict of {class_name: probability} for all 4 gesture classes
    """
    assert _model is not None, "Model must be loaded before inference"

    # Move tensor to the same device as the model weights
    model_device = next(_model.parameters()).device
    tensor = tensor.to(model_device)

    _model.eval()

    with torch.no_grad():
        # ClassificationModel returns (logits, aux_tensor) as a tuple
        _output = _model(tensor)
        logits: torch.Tensor = _output[0] if isinstance(_output, (tuple, list)) else _output

        # Apply temperature scaling: dividing logits by T < 1 sharpens
        # confidence; T > 1 flattens it. T = 1.0 means no change.
        # Run calibrate_temperature.py to write the optimal T to
        # outputs/temperature.json, which load_model_once() reads.
        calibrated_logits = logits / _temperature

        probs_arr: np.ndarray = (
            torch.softmax(calibrated_logits, dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    return {name: float(p) for name, p in zip(_class_names, probs_arr)}


# ===========================================================================
# 4. VISUALISATION
# ===========================================================================

def build_bar_chart(probs: dict[str, float]) -> Image.Image:
    """
    Render a minimal horizontal bar chart of the 4 class confidence scores.

    The highest-confidence bar is highlighted in blue; the rest are grey.
    Percentage labels appear at the right end of each bar.

    Args:
        probs: {class_name: probability} from run_inference()

    Returns:
        PIL Image of the chart (rendered to BytesIO — no disk I/O)
    """
    classes = list(probs.keys())
    values  = [probs[c] for c in classes]
    top_idx = int(np.argmax(values))

    colours = ["#4A90D9" if i == top_idx else "#B0B8C1" for i in range(len(classes))]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(classes, values, color=colours, height=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence")
    ax.set_title("Class Probabilities")
    ax.invert_yaxis()       # highest-confidence class at the top
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ===========================================================================
# 5. MAIN PREDICTION FUNCTION — wired to Gradio
# ===========================================================================

def predict(
    uploaded_image: Optional[np.ndarray],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[Optional[Image.Image], str, Optional[Image.Image]]:
    """
    End-to-end prediction pipeline called by Gradio on every classify request.

    Args:
        uploaded_image: RGB numpy array from Gradio's gr.Image(type="numpy")
        progress:       Gradio progress tracker for visible step-by-step status

    Returns:
        (display_img, result_markdown, chart_img) for the three Gradio outputs
    """
    if _model is None:
        return None, "Model not loaded — check the terminal for the error and traceback.", None

    if uploaded_image is None:
        return None, "Please upload an image to classify.", None

    # ── Step 1: Save numpy upload to a temp JPEG ─────────────────────────────
    progress(0.10, desc="Saving uploaded image...")
    pil_img = Image.fromarray(uploaded_image.astype("uint8"), "RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        pil_img.save(tmp_path)

    # ── Step 2: Detect hand region and build normalised tensor ───────────────
    progress(0.35, desc="Detecting hand region...")
    try:
        tensor, display_img, status_msg = preprocess_image(tmp_path)
    except Exception as e:
        return None, f"Pre-processing failed: {e}", None

    # ── Step 3: Forward pass through the fine-tuned model ───────────────────
    progress(0.65, desc="Running gesture classifier...")
    try:
        probs = run_inference(tensor)
    except Exception as e:
        return None, f"Inference failed: {e}", None

    # ── Step 4: Format results ───────────────────────────────────────────────
    progress(0.85, desc="Building results...")

    top_class = max(probs, key=lambda k: probs[k])
    top_conf  = probs[top_class]

    result_md = (
        f"**Prediction:** {top_class}\n\n"
        f"**Confidence:** {top_conf:.1%}\n\n"
        f"_{status_msg}_"
    )

    chart = build_bar_chart(probs)
    progress(1.0, desc="Done!")

    try:
        tmp_path.unlink()
    except Exception:
        pass

    return display_img, result_md, chart


# ===========================================================================
# 6. GRADIO INTERFACE
# ===========================================================================

def build_interface() -> gr.Blocks:
    """
    Define the two-column Gradio Blocks layout.

    Left column  — image upload component + classify button
    Right column — model input preview, prediction text, confidence chart
    """
    with gr.Blocks(title="We Heart DH - Gesture Classifier") as demo:

        gr.Markdown(
            "## We Heart Digital Humanities - Heart Emblem Gesture Classifier\n"
            "Upload an image of a hand gesture and the model will identify "
            "which of the 4 heart emblem classes it belongs to."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Gesture Image",
                    type="numpy",       # Gradio delivers RGB numpy array
                    sources=["upload"],
                )
                classify_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                output_crop  = gr.Image(
                    label="Input fed to model (cropped or full)",
                    type="pil",
                )
                output_label = gr.Markdown(
                    value="Upload an image and click **Classify** to see results."
                )
                output_chart = gr.Image(
                    label="Class Confidence Chart",
                    type="pil",
                )

        classify_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[output_crop, output_label, output_chart],
        )

        gr.Markdown(
            "---\n"
            "**Classes:** 2-finger-heart | traditional-heart | "
            "traditional-heart-2 | upside_down_heart\n\n"
            "_Model: YOLOv8x-cls fine-tuned on 231 images | We Heart DH project, DH2026_"
        )

    return demo


# ===========================================================================
# 7. ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    model_ready = load_model_once()

    if not model_ready:
        print()
        print("[WARN] App launching without a loaded model.")
        print("       Any classify attempt will show an error message.")
        print("       Fix the error printed above, then restart the app.")

    demo = build_interface()
    demo.launch(share=False)