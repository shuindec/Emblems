"""
    app/app_spaces.py â€” Polished Corpus Demo for HuggingFace Spaces & Presentation
We Hert Digital Humanities - Automating Emblem Gesture Recognition

Visual style: Warm Vintage / Hand-crafted (DH2026 presentation edition)
    - Palette: 41431B (dark olive) Â· AEB784 (sage green) Â· E3DBBB (warm sand)
               Â· F8F3E1 (cream) Â· B5532A (terracotta accent â€” wax-seal red)
               Â· 5C3A1E (ink brown for secondary text)
    - Typography: DM Serif Display (headlines) Â· Lora (body) Â·
                  IM Fell English SC (letterpress small-caps labels)
    - Subtle paper-grain SVG noise background
    - Ornamental dividers (â¦ âœ¦ â¦) framing each section
    - Card-style panels with deckle-edge feel and warm shadows

Features (UNCHANGED):
    - Model switcher: YOLOv8x (crop), YOLOv8x (full frame), ResNet18, Hagridv2
    - Confidence badge: HIGH / MEDIUM / LOW colour-coded emoji
    - Reference panel with one image per class
    - All models loaded at startup and cached for instant switching
"""

# ---------------------------------------------------------------------------
# Standard path setup
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import io
import json
import tempfile
import traceback
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from src.utils import load_config, get_project_paths, load_class_names
from src.models import get_model

# ---------------------------------------------------------------------------
# Palette  (extended with terracotta accent + ink brown for the vintage feel)
# ---------------------------------------------------------------------------
PALETTE = {
    "dark_olive" : "#41431B",
    "sage_green" : "#AEB784",
    "warm_sand"  : "#E3DBBB",
    "cream"      : "#F8F3E1",
    "terracotta" : "#B5532A",   # NEW â€” wax-seal accent for buttons, badges, ornaments
    "ink_brown"  : "#5C3A1E",   # NEW â€” softer body text colour for letterpress feel
}

# ---------------------------------------------------------------------------
# Checkpoint map â€” each entry: display_name â†’ (model_name, filename)
# load_all_models() iterates this and skips any file that does not exist.
# ---------------------------------------------------------------------------
CHECKPOINT_MAP: dict[str, tuple[str, str]] = {
    "YOLOv8x (crop)"       : ("yolo",     "best_model_yolo.pth"),
    "YOLOv8x (full frame)" : ("yolo",     "best_model_yolo_fullframe.pth"),
    "ResNet18"             : ("resnet18", "best_model_resnet18.pth"),
    "Hagridv2"             : ("hagridv2_resnet18",   "best_model_hagridv2_resnet18.pth"),
}

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_models: dict[str, nn.Module] = {}
_temperatures: dict[str, float] = {}
_class_names: list[str] = []
_cfg: dict              = {}

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------
HIGH_THRESHOLD   = 0.70
MEDIUM_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Example images â€” one per class for the reference panel
# ---------------------------------------------------------------------------
EXAMPLE_IMAGES: dict[str, str] = {
   "2-finger-heart"      : r"C:\Users\suzuk\emblems\data\split\train\images\2-finger-heart_b77c1109_jpg.rf.eCUI8PZrMU9yJKaiG7pe.jpg",
    "traditional-heart"   : r"C:\Users\suzuk\emblems\data\split\train\images\traditional-heart_2cd9d598_jpg.rf.wpjt3mTgrRSLlsAgAm6M.jpg",
    "traditional-heart-2" : r"C:\Users\suzuk\emblems\data\split\train\images\traditional-heart-2_101e09a5_jpg.rf.FR3wrDIrioLU3TeIN8VT.jpg",
    "upside_down_heart"   : r"C:\Users\suzuk\emblems\web_image_scraper\outputs\dataset_raw\upside_down_heart\upside_down_heart_b5b710c1.jpg",
}

# ---------------------------------------------------------------------------
# ImageNet normalisation â€” must match data_loader.py training setup exactly
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Custom CSS â€” Warm Vintage / Hand-crafted edition
#
# Highlights:
#   - Google Fonts: DM Serif Display, Lora, IM Fell English SC
#   - SVG-noise grain overlay on the cream background (subtle paper texture)
#   - Ornamental dividers in terracotta
#   - Hero title with letterpress small-caps tagline
#   - Sage-green model radio rendered as warm pill chips
#   - Terracotta classify button with wax-seal hover effect
#   - Card panels with soft warm shadows + sage borders
#
# NOTE on f-string: literal CSS braces are escaped as {{ and }}.
# ---------------------------------------------------------------------------
CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=IM+Fell+English+SC&display=swap');

/* ============== Page canvas â€” cream paper with subtle grain ============== */
body, .gradio-container, .gradio-container > .main, gradio-app {{
    background-color: {PALETTE['cream']} !important;
    background-image:
        url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.25 0 0 0 0 0.23 0 0 0 0 0.11 0 0 0 0.06 0'/></filter><rect width='240' height='240' filter='url(%23n)'/></svg>");
    color: {PALETTE['ink_brown']} !important;
    font-family: 'Lora', Georgia, serif !important;
    font-size: 17px !important;
}}

.gradio-container {{
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 28px 36px !important;
}}

/* ============== Typography ============== */
h1, h2, h3, h4 {{
    font-family: 'DM Serif Display', Georgia, serif !important;
    color: {PALETTE['dark_olive']} !important;
    letter-spacing: 0.005em;
    font-weight: 400 !important;
    margin-top: 0.2em !important;
}}
h1 {{ font-size: 3.2em !important; line-height: 1.05 !important; margin-bottom: 0.1em !important; }}
h2 {{ font-size: 1.9em !important; line-height: 1.15 !important; }}
h3 {{ font-size: 1.35em !important; }}
h4 {{ font-size: 1.05em !important; text-transform: uppercase; letter-spacing: 0.14em; color: {PALETTE['terracotta']} !important; font-family: Georgia, 'Times New Roman', Times, serif !important; }}
p, li, label {{ color: {PALETTE['ink_brown']} !important; line-height: 1.65 !important; }}

/* ============== Hero banner ============== */
.hero-banner {{
    text-align: center;
    padding: 36px 24px 24px 24px;
    background: linear-gradient(180deg, rgba(227,219,187,0.55) 0%, rgba(248,243,225,0) 100%);
    border-bottom: 1px dashed {PALETTE['sage_green']};
    margin-bottom: 12px;
}}
.hero-banner h1 {{
    font-style: italic;
    color: {PALETTE['dark_olive']} !important;
}}
.hero-banner h1::first-letter {{
    color: {PALETTE['terracotta']};
}}
.hero-tagline {{
    font-family: Georgia, 'Times New Roman', Times, serif !important;
    font-size: 1.3em !important;
    letter-spacing: 0.22em;
    color: {PALETTE['terracotta']} !important;
    margin: 8px 0 14px 0 !important;
    text-transform: uppercase;
}}
.hero-byline {{
    font-style: italic;
    color: {PALETTE['ink_brown']} !important;
    font-size: 0.95em !important;
    max-width: 720px;
    margin: 14px auto 0 auto !important;
}}

/* ============== Ornamental divider ============== */
.ornament {{
    text-align: center;
    color: {PALETTE['terracotta']};
    font-size: 1.4em;
    letter-spacing: 0.6em;
    margin: 18px 0 22px 0;
    user-select: none;
}}

/* ============== Section title (small caps letterpress) ============== */
.section-title {{
    font-family: Georgia, 'Times New Roman', Times, serif !important;
    font-size: 1.5em !important;
    letter-spacing: 0.18em;
    color: {PALETTE['terracotta']} !important;
    text-transform: uppercase;
    text-align: center;
    margin: 4px 0 14px 0 !important;
}}

/* ============== Card / panel surfaces ============== */
.gr-panel, .gr-box, .gr-form, .gradio-container .block {{
    background-color: rgba(248,243,225,0.7) !important;
    border: 1px solid {PALETTE['sage_green']} !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 0 rgba(92,58,30,0.06), 0 8px 22px -12px rgba(65,67,27,0.18) !important;
}}

/* Pull-out card class we apply via elem_classes */
.vintage-card {{
    background-color: rgba(248,243,225,0.85) !important;
    border: 1px solid {PALETTE['sage_green']} !important;
    border-radius: 6px !important;
    padding: 18px !important;
    box-shadow:
        inset 0 0 0 1px rgba(174,183,132,0.25),
        0 1px 0 rgba(92,58,30,0.04),
        0 10px 28px -16px rgba(65,67,27,0.22) !important;
}}

/* ============== Radio buttons â†’ pill chips ============== */
.gr-radio {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
.gr-radio > label, .gr-radio fieldset > label {{
    background-color: {PALETTE['cream']} !important;
    border: 1.5px solid {PALETTE['sage_green']} !important;
    border-radius: 999px !important;
    padding: 8px 16px !important;
    margin: 4px 6px 4px 0 !important;
    cursor: pointer;
    transition: background-color 0.18s ease, color 0.18s ease, border-color 0.18s ease;
    color: {PALETTE['ink_brown']} !important;
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.95em !important;
}}
.gr-radio > label:hover, .gr-radio fieldset > label:hover {{
    background-color: {PALETTE['warm_sand']} !important;
    border-color: {PALETTE['dark_olive']} !important;
}}
.gr-radio > label.selected, .gr-radio input:checked + span, .gr-radio fieldset > label:has(input:checked) {{
    background-color: {PALETTE['dark_olive']} !important;
    color: {PALETTE['cream']} !important;
    border-color: {PALETTE['dark_olive']} !important;
}}

/* ============== Classify button â€” terracotta wax seal ============== */
#classify-btn {{
    background-color: {PALETTE['terracotta']} !important;
    background-image: linear-gradient(180deg, #C26239 0%, #A04826 100%) !important;
    border: 1.5px solid {PALETTE['dark_olive']} !important;
    color: {PALETTE['cream']} !important;
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 1.25em !important;
    letter-spacing: 0.04em;
    padding: 14px 28px !important;
    border-radius: 4px !important;
    box-shadow: 0 3px 0 {PALETTE['dark_olive']}, 0 8px 18px -8px rgba(181,83,42,0.55) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, background-image 0.12s ease !important;
}}
#classify-btn:hover {{
    background-image: linear-gradient(180deg, #A04826 0%, #7F3819 100%) !important;
    transform: translateY(1px);
    box-shadow: 0 2px 0 {PALETTE['dark_olive']}, 0 6px 14px -8px rgba(181,83,42,0.55) !important;
}}
#classify-btn:active {{
    transform: translateY(3px);
    box-shadow: 0 0 0 {PALETTE['dark_olive']} !important;
}}

/* ============== Result text panel ============== */
.result-text {{
    background-color: rgba(227,219,187,0.4) !important;
    border-left: 4px solid {PALETTE['terracotta']} !important;
    padding: 18px 22px !important;
    border-radius: 2px;
}}
.result-text h2 {{
    margin-top: 0 !important;
    color: {PALETTE['terracotta']} !important;
    font-size: 1.7em !important;
}}
.result-text p {{
    font-size: 1.05em !important;
    line-height: 1.7 !important;
    color: {PALETTE['ink_brown']} !important;
}}
.result-text strong {{
    color: {PALETTE['dark_olive']} !important;
}}

/* ============== Reference thumbnails ============== */
.ref-strip .gr-image, .ref-strip .image-container {{
    border: 1px solid {PALETTE['sage_green']} !important;
    background-color: {PALETTE['cream']} !important;
    box-shadow: 0 4px 14px -8px rgba(65,67,27,0.35) !important;
}}
.ref-strip label {{
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.9em !important;
    letter-spacing: 0.08em;
    color: {PALETTE['dark_olive']} !important;
    text-transform: uppercase;
    text-align: center;
}}

/* ============== Upload area ============== */
.gr-image, .image-container, .upload-container {{
    border: 1.5px dashed {PALETTE['sage_green']} !important;
    background-color: rgba(248,243,225,0.6) !important;
    border-radius: 4px !important;
}}
.gr-image:hover {{
    border-color: {PALETTE['terracotta']} !important;
}}

/* ============== Footer / colophon ============== */
.colophon {{
    text-align: center;
    font-style: italic;
    color: {PALETTE['ink_brown']} !important;
    font-size: 0.92em !important;
    padding: 18px 16px 8px 16px;
    border-top: 1px dashed {PALETTE['sage_green']};
    margin-top: 24px;
}}
.colophon strong {{
    font-family: 'Lora', Georgia, serif !important;
    color: {PALETTE['dark_olive']} !important;
    letter-spacing: 0.12em;
    font-weight: normal !important;
}}

/* ============== Misc Gradio cleanup ============== */
footer {{ display: none !important; }}
.gr-button-secondary {{ font-family: 'Lora', Georgia, serif !important; }}
"""


# ===========================================================================
# 1. HELPER â€” human-readable class labels
# ===========================================================================

def class_label(class_name: str) -> str:
    labels = {
        "2-finger-heart"      : "2-Finger Heart  (Korean/Celebrity)",
        "traditional-heart"   : "Traditional Hand Heart  (Millennial)",
        "traditional-heart-2" : "Gen Z Heart Variant",
        "upside_down_heart"   : "Upside-Down Heart",
    }
    return labels.get(class_name, class_name)


# ===========================================================================
# 2. MODEL LOADING â€” all available models cached at startup
# ===========================================================================

def _load_single_model(
    model_name: str,
    ckpt_path: Path,
    device: torch.device,
    cfg: dict,
) -> Optional[nn.Module]:
    try:
        cfg["training"]["model_name"] = model_name
        model, _, _ = get_model(cfg)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "MODEL_STATE" in checkpoint:
            state_dict = checkpoint["MODEL_STATE"]
        else:
            print(f"    [ERROR] No weights key. Found: {list(checkpoint.keys())}")
            return None

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"    [ERROR] {e}")
        traceback.print_exc()
        return None


def load_all_models() -> bool:
    global _models, _temperatures, _class_names, _cfg

    _cfg  = load_config()
    paths = get_project_paths(_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    temp_path = Path(paths["output_dir"]) / "temperature.json"
    if temp_path.exists():
        with open(temp_path, encoding="utf-8") as f:
            shared_T = float(json.load(f)["temperature"])
        print(f"[OK] Temperature   : {shared_T:.4f} (from temperature.json)")
    else:
        shared_T = 1.0
        print("[INFO] No temperature.json â€” using T=1.0")

    data_yaml    = Path(paths["data_dir"]) / _cfg["dataset"]["yaml_file"]
    _class_names = load_class_names(data_yaml)

    for display_name, (model_name, ckpt_filename) in CHECKPOINT_MAP.items():
        ckpt_path = Path(paths["output_dir"]) / "checkpoints" / ckpt_filename

        if not ckpt_path.exists():
            print(f"[SKIP] {display_name}: checkpoint not found ({ckpt_filename})")
            continue

        print(f"[INFO] Loading {display_name} ...")
        model = _load_single_model(model_name, ckpt_path, device, dict(_cfg))

        if model is not None:
            _models[display_name]      = model
            _temperatures[display_name] = shared_T
            print(f"[OK]  {display_name} ready")
        else:
            print(f"[FAIL] {display_name} could not be loaded")

    if not _models:
        print("[ERROR] No models loaded â€” check outputs/checkpoints/")
        return False

    print(f"[OK] Available models: {list(_models.keys())}")
    return True


# ===========================================================================
# 3. IMAGE PRE-PROCESSING
# ===========================================================================

def find_label_file(image_path: Path) -> Optional[Path]:
    sibling = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    if sibling.exists():
        return sibling
    flat = image_path.with_suffix(".txt")
    if flat.exists():
        return flat
    return None


def crop_from_label(image: Image.Image, label_path: Path) -> Optional[Image.Image]:
    try:
        with open(label_path, encoding="utf-8") as f:
            line = f.readline().strip()
        if not line or len(line.split()) < 5:
            return None
        parts = line.split()
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        img_w, img_h = image.size
        x1 = max(0,     int((cx - bw / 2) * img_w))
        y1 = max(0,     int((cy - bh / 2) * img_h))
        x2 = min(img_w, int((cx + bw / 2) * img_w))
        y2 = min(img_h, int((cy + bh / 2) * img_h))
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"[WARN] Label parse failed ({label_path.name}): {e}")
        return None


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, Image.Image, str]:
    image      = Image.open(image_path).convert("RGB")
    label_path = find_label_file(image_path)
    cropped: Optional[Image.Image] = None
    status_msg = ""

    if label_path is not None:
        cropped    = crop_from_label(image, label_path)
        status_msg = (
            "Cropped from bounding box label"
            if cropped is not None
            else "Label found but crop failed - using full image"
        )
    else:
        status_msg = "No label file — using full image"

    input_image: Image.Image = cropped if cropped is not None else image
    display_img  = input_image.copy()

    tensor: torch.Tensor = _transform(input_image)  # type: ignore[assignment]
    tensor = tensor.unsqueeze(0)

    return tensor, display_img, status_msg


# ===========================================================================
# 4. INFERENCE
# ===========================================================================

def run_inference(
    tensor: torch.Tensor,
    model: nn.Module,
    temperature: float,
) -> dict[str, float]:
    model_device = next(model.parameters()).device
    tensor = tensor.to(model_device)

    model.eval()
    with torch.no_grad():
        _output = model(tensor)
        logits: torch.Tensor = _output[0] if isinstance(_output, (tuple, list)) else _output
        calibrated = logits / temperature
        probs_arr: np.ndarray = (
            torch.softmax(calibrated, dim=1).squeeze(0).cpu().numpy()
        )

    return {name: float(p) for name, p in zip(_class_names, probs_arr)}


# ===========================================================================
# 5. CONFIDENCE BADGE
# ===========================================================================

def confidence_badge(confidence: float) -> str:
    if confidence >= HIGH_THRESHOLD:
        return "HIGH CONFIDENCE"
    elif confidence >= MEDIUM_THRESHOLD:
        return "MEDIUM CONFIDENCE"
    else:
        return "LOW CONFIDENCE"


# ===========================================================================
# 6. BAR CHART  (palette-matched, terracotta highlight on top class)
# ===========================================================================

def build_bar_chart(probs: dict[str, float]) -> Image.Image:
    classes = list(probs.keys())
    values  = [probs[c] for c in classes]
    labels  = [class_label(c) for c in classes]
    top_idx = int(np.argmax(values))

    colours = [
        PALETTE["terracotta"] if i == top_idx else PALETTE["sage_green"]
        for i in range(len(classes))
    ]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor(PALETTE["cream"])
    ax.set_facecolor(PALETTE["cream"])

    bars = ax.barh(labels, values, color=colours, height=0.55,
                   edgecolor=PALETTE["dark_olive"], linewidth=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center", fontsize=10,
            color=PALETTE["dark_olive"],
            fontweight="bold",
        )

    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Confidence", color=PALETTE["ink_brown"], fontsize=10, style="italic")
    ax.set_title("Class Probabilities", color=PALETTE["dark_olive"],
                 fontweight="bold", fontsize=12, family="serif")
    ax.tick_params(colors=PALETTE["ink_brown"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["sage_green"])
        spine.set_linewidth(0.8)
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=PALETTE["cream"])
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ===========================================================================
# 7. MAIN PREDICTION FUNCTION
# ===========================================================================

def predict(
    uploaded_image: Optional[np.ndarray],
    selected_model: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[Optional[Image.Image], str, Optional[Image.Image]]:
    if not _models:
        return None, "âš ï¸ No models loaded â€” check the terminal for details.", None

    if uploaded_image is None:
        return None, "Please upload an image to classify.", None

    if selected_model not in _models:
        selected_model = next(iter(_models))

    model       = _models[selected_model]
    temperature = _temperatures.get(selected_model, 1.0)

    progress(0.10, desc="Saving image...")
    pil_img = Image.fromarray(uploaded_image.astype("uint8"), "RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        pil_img.save(tmp_path)

    progress(0.35, desc="Detecting hand region...")
    try:
        tensor, display_img, status_msg = preprocess_image(tmp_path)
    except Exception as e:
        return None, f"Pre-processing failed: {e}", None

    progress(0.65, desc=f"Classifying with {selected_model}...")
    try:
        probs = run_inference(tensor, model, temperature)
    except Exception as e:
        return None, f"Inference failed: {e}", None

    progress(0.85, desc="Building results...")

    top_class    = max(probs, key=lambda k: probs[k])
    top_conf     = probs[top_class]
    badge        = confidence_badge(top_conf)
    display_name = class_label(top_class)

    result_md = (
        f"## {badge}\n\n"
        f"**Gesture Detected:** {display_name}\n\n"
        f"**Confidence Score:** {top_conf:.1%}\n\n"
        f"**Model used:** {selected_model}\n\n"
        f"*{status_msg}*"
    )

    chart = build_bar_chart(probs)
    progress(1.0, desc="Done!")

    try:
        tmp_path.unlink()
    except Exception:
        pass

    return display_img, result_md, chart


# ===========================================================================
# 8. GRADIO INTERFACE â€” Warm Vintage / Hand-crafted layout
# ===========================================================================

def load_example_image(path_str: str) -> Optional[Image.Image]:
    p = Path(path_str)
    if not p.exists():
        print(f"[WARN] Example image not found: {p.name}")
        return None
    return Image.open(p).convert("RGB")


# Ornamental divider as reusable HTML snippet
_ORNAMENT = '<div class="ornament">⟫     ⟫     ⟫</div>'


def build_interface() -> gr.Blocks:
    """
    Layout (top-to-bottom, reads like a museum catalogue page):

    1. HERO BANNER  â€” large italic title, terracotta drop-cap on â™¥,
                      letterpress tagline, byline & affiliation
    2. ORNAMENT
    3. CONTROLS ROW â€” model pill-chips (left, scale 3) + upload card + classify
                      button (right, scale 2), inside vintage-cards
    4. ORNAMENT
    5. GESTURE LIBRARY â€” horizontal strip of 4 reference thumbnails
    6. ORNAMENT
    7. RESULTS â€” input image (scale 1) | result text + badge (scale 1) | chart (scale 1)
    8. COLOPHON footer (italic credit line)
    """
    available_models = list(_models.keys())
    default_model    = available_models[0] if available_models else None

    with gr.Blocks(title="We Heart DH - Gesture Classifier",
                   css=CUSTOM_CSS) as demo:
       
        gr.HTML(
            """
            <div class="hero-banner">
                <h1> We Heart ❤️ Digital Humanities</h1>
                <div class="hero-tagline">Automating Emblem Gesture Recognition</div>
                <p class="hero-byline">
                    Heart emblems - gestures forming a heart shape - carry distinct cultural
                    meanings across generations and regions. This tool uses computer vision
                    to classify heart-gesture variants for large-scale corpus analysis in
                    gesture studies.
                </p>
                <p class="hero-byline" style="margin-top:18px; font-size:0.85em;">
                    <strong style="font-family:'Lora', serif; letter-spacing:0.16em; color:#41431B;">
                    DH2026 Conference
                    </strong><br/>
                    Dr Lauren Gawne &middot; Dr Yang Zhao &middot; Dr Judith Bishop &middot; (Thanh Anh Thu) Nguyen <br/>
                    <em>La Trobe University</em>
                </p>
            </div>
            """
        )

        # â”€â”€ 4. ORNAMENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gr.HTML(_ORNAMENT)

        # â”€â”€ 5. GESTURE LIBRARY â€” horizontal strip â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gr.HTML('<div class="section-title">I · Gesture reference library</div>')
        with gr.Row(elem_classes=["ref-strip"]):
            for class_name, img_path in EXAMPLE_IMAGES.items():
                with gr.Column(scale=1, min_width=160):
                    gr.Image(
                        value=load_example_image(img_path),
                        label=class_label(class_name),
                        interactive=False,
                        height=170,
                        container=True,
                    )

        # â”€â”€ 2. ORNAMENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gr.HTML(_ORNAMENT)

        # â”€â”€ 3. CONTROLS ROW â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with gr.Row(equal_height=True):
            # Left â€” model selector card
            with gr.Column(scale=3, elem_classes=["vintage-card"]):
                gr.HTML('<div class="section-title">II · Choose your model</div>')
                model_radio = gr.Radio(
                    choices=available_models,
                    value=default_model,
                    label="",
                    interactive=True,
                )
                gr.Markdown(
                    "<p style='font-style:italic; font-size:1.22em; color:#5C3A1E; margin:15px;'>"
                    "All available models load at startup &mdash; switching is instant."
                    "</p>"
                )

            # Right â€” upload + classify card
            with gr.Column(scale=2, elem_classes=["vintage-card"]):
                gr.HTML('<div class="section-title">III · Upload an image</div>')
                input_image = gr.Image(
                    label="",
                    type="numpy",
                    sources=["upload"],
                    height=240,
                )
                classify_btn = gr.Button(
                    "⭐ Classify Gesture ",
                    elem_id="classify-btn",
                    size="lg",
                )


        # â”€â”€ 6. ORNAMENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gr.HTML(_ORNAMENT)

        # â”€â”€ 7. RESULTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gr.HTML('<div class="section-title">IV · Classification result</div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["vintage-card"]):
                gr.HTML('<h4 style="text-align:center;">Input fed to model</h4>')
                output_crop = gr.Image(
                    label="",
                    type="pil",
                    height=260,
                )

            with gr.Column(scale=1):
                output_label = gr.Markdown(
                    value=(
                        "## Awaiting your image...\n\n"
                        "*Upload an image above and press* **✦ Classify Gesture ✦** *to reveal "
                        "the predicted heart-emblem class, confidence score, and full probability "
                        "breakdown.*"
                    ),
                    elem_classes=["result-text"],
                )

            with gr.Column(scale=1, elem_classes=["vintage-card"]):
                gr.HTML('<h4 style="text-align:center;">Confidence by class</h4>')
                output_chart = gr.Image(
                    label="",
                    type="pil",
                )

        # Wire the button â†’ predict
        classify_btn.click(
            fn=predict,
            inputs=[input_image, model_radio],
            outputs=[output_crop, output_label, output_chart],
        )

        # â”€â”€ 8. COLOPHON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gr.HTML(
            """
            <div class="colophon">
                <strong>Models</strong> &nbsp; YOLOv8x-cls (crop) &middot;
                YOLOv8x-cls (full frame) &middot; ResNet18 &middot; Hagridv2
                &mdash; fine-tuned on 350 labelled hand-gesture images, temperature-calibrated.<br/><br/>
                <strong>Classes</strong> &nbsp; 2-Finger Heart (Korean/Celebrity) &middot;
                Hand Heart (Millennial) &middot; Gen Z Heart Variant &middot;
                Upside-Down Heart<br/><br/>
                <em>We Heart Digital Humanities &mdash; DH2026 Conference &middot; La Trobe University</em>
            </div>
            """
        )

    return demo


# ===========================================================================
# 9. ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    success = load_all_models()

    if not success:
        print()
        print("[WARN] App launching with no models loaded.")
        print("       Ensure at least one checkpoint exists in outputs/checkpoints/")

    demo = build_interface()
    # CSS is now attached directly to the Blocks via theme; share=False for local.
    demo.launch(share=False)

