# We ♥ Digital Humanities — Heart Emblem Gesture Recognition

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2011.8-orange)
![License](https://img.shields.io/badge/License-CC--BY%204.0-green)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/shuindec/heart-gesture-classification-tool)

> Automating emblem gesture recognition to support data analysis at scale —
> presented at **DH2026**, La Trobe University.

---

## Live Demo & Poster

| Resource | Link |
|---|---|
| 🤗 Interactive classifier (upload any heart gesture image) | [HuggingFace Spaces](https://huggingface.co/spaces/shuindec/heart-gesture-classification-tool) |
| 🖼️ Conference poster | [DH2026 Poster](https://dh2026post-kuhqkunv.manus.space/) |
| 🧠 Pre-trained model weights | [shuindec/heart-emblem-models](https://huggingface.co/shuindec/heart-emblem-models) |

---

## Project Overview

**Emblems** are word-like gestures that carry dictionary-style meanings within a cultural group — think "thumbs up" or "peace sign." Heart emblems are a particularly interesting case: multiple hand configurations all express the idea of a heart, yet they are associated with different demographics and cultural contexts. The **Korean finger heart** (두 손가락 하트) spread globally through K-pop media, while the **traditional two-handed heart** is common among Millennials, and a newer variant is closely associated with Gen Z users.

Building a large-scale corpus of these gestures from images and video requires tools that can automatically classify which variant appears in each frame — a task that is tedious to do by hand at research scale. This project investigates whether **transfer learning from pretrained vision models** can reliably distinguish four heart emblem variants using fewer than 500 labeled examples, and deploys the best-performing model as a no-code web tool for corpus researchers.

**Research questions:**
1. Can transfer learning from pretrained models enable accurate heart emblem classification with limited labeled data (<500 examples)?
2. How does classification accuracy vary across heart emblem variants when training data is constrained?

**Supervisors:** Dr Lauren Gawne · Dr Judith Bishop · Dr Yang Zhao
**Student researcher:** Nguyen (Thanh Anh Thu)
**Institution:** La Trobe University · DH2026 Conference · June 2026

---

## The Four Gesture Classes

| Class | Description |
|---|---|
| `2-finger-heart` | Korean finger heart — thumb and index finger crossed to form a small heart |
| `traditional-heart` | Classic two-handed heart — both hands joined with fingers forming a heart shape |
| `traditional-heart-2` | Variant of the two-handed heart with a slightly different finger configuration |
| `upside-down-heart` | Inverted heart gesture, typically formed with both hands pointing downward |

---

## Repository Structure

```
emblems/
├── configs/
│   └── config.yaml              ← central config: all paths, hyperparameters, model selection
├── src/
│   ├── __init__.py
│   ├── utils.py                 ← shared utilities (paths, YOLO parsing, colour constants)
│   ├── data_loader.py           ← PyTorch Dataset class + DataLoader factory with bbox crop
│   ├── model.py                 ← model factory: get_model(cfg) for all architectures
│   ├── train.py                 ← training loop with early stopping and checkpoint saving
│   └── evaluate.py              ← metrics, confusion matrix, training curves
├── scripts/
│   ├── dataset_audit_and_clean.py     ← removes full-arm-heart class, audits labels
│   ├── dataset_inspect_and_fix.py     ← rescues unreadable images via Pillow
│   ├── restore_multi_label.py         ← merges overlapping bounding boxes
│   ├── split_dataset.py               ← stratified 60/20/20 split (seed=42, run once)
│   ├── visualise_merged_boxes.py      ← visual verification of bbox merges
│   └── diagnose_orientation.py        ← class orientation diagnostic grid
├── app/
│   ├── app.py                   ← local corpus tool (GPU, CSV export, YOLO crop model)
│   ├── app_spaces.py            ← conference demo (all models, HuggingFace-ready)
│   └── calibrate_temperature.py ← post-hoc temperature calibration via NLL minimisation
|   └── stream_demo.py           ← demo of Mediapipe + MLP for real-time recognising
├── data/
│   ├── data.yaml                ← class names and dataset paths (YOLO format)
│   └── examples/                ← 4 sample images, one per class (for interface testing)
│   # train/, split/, excluded/ are gitignored — raw images are large and version-controlled
│   # separately. See Dataset section below for access.
├── outputs/
│   ├── temperature.json         ← calibrated temperature scalar T = 0.1168 (YOLO)
│   └── figures/
│       ├── class_distribution_before_after.png
│       ├── split_distribution.png
│       ├── merged_boxes_grid.png
│       ├── confusion_matrix_resnet18.png
│       └── training_curves_resnet18.png
└── requirements.txt
```

> **Note:** `data/train/`, `data/split/`, `data/excluded/`, and `outputs/checkpoints/*.pth`
> are excluded from this repository. Model weights are hosted on
> [HuggingFace Hub](https://huggingface.co/shuindec/heart-emblem-models) and downloaded
> automatically by `app/app_spaces.py`. Raw image data was annotated via Roboflow.

---

## Setup and Installation

**Requirements:** Python 3.12 · CUDA 11.8 · NVIDIA RTX 3070 (8 GB VRAM) or equivalent

```powershell
# 1. Clone the repository
git clone https://github.com/shuindec/Emblems.git
cd emblems

# 2. Create and activate a virtual environment
python -m venv emblem_env
emblem_env\Scripts\Activate.ps1   # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

> All experiments were run on an RTX 3070 (8 GB VRAM). YOLO training uses
> `batch_size=16` to stay within memory limits. For larger backbones (ViT-B16),
> consider reducing batch size to 8 or using a cloud GPU (Kaggle / Colab).

---

## How to Run

### 1 — Verify the data loader

```powershell
python src/data_loader.py
```

Confirms images load, bounding box crops are applied, and class counts are correct.

### 2 — Train a model

Open `configs/config.yaml` and set `model_name` to one of:
`resnet18` · `resnet18_fullframe` · `hagridv2_resnet18` · `vit` · `yolo` · `yolo_fullframe`

```powershell
python src/train.py
```

Saves the best checkpoint to `outputs/checkpoints/best_model_{model_name}.pth`
and the training log to `outputs/logs/training_log_{model_name}.csv`.

### 3 — Evaluate and generate figures

```powershell
python src/evaluate.py
```

Produces a confusion matrix and training curves under `outputs/figures/`.

### 4 — Run the local corpus tool

```powershell
python app/app.py
```

Opens the Gradio interface at `http://127.0.0.1:7860`.
Upload a hand gesture image to get a class prediction with confidence scores.

---

## Results

All six model configurations were evaluated on a held-out test set of 45 images
(stratified 60/20/20 split, seed=42).

| Model | Pretraining | Input | Accuracy | Macro F1 |
|---|---|---|---|---|
| **YOLOv8x-cls** | ImageNet | Cropped | **92.5%** | **0.923** |
| **ResNet18** | ImageNet | Cropped | **91.0%** | **0.908** |
| ViT-B16 | ImageNet-21k | Cropped | 88.1% | 0.871 |
| ResNet18 | ImageNet | Full frame | 74.6% | 0.732 |
| ResNet18 | HAGRIDv2 | Cropped | 53.7% | 0.507 |
| ResNet18 | HAGRIDv2 | Full frame | 44.8% | 0.442 |

**Key findings:**

- **Bounding box cropping is the single most impactful design decision**, adding 18.7 percentage points over full-frame input for the same model and weights. Removing background noise matters more than architecture choice.

- **Architecture matters less than expected.** ViT-B16 (86M parameters, ImageNet-21k) underperforms ResNet18 (11M, ImageNet) by 5.2 points, because patch-based attention needs more training signal than 203 images can provide.

- **Gesture-specific pretraining hurts on within-family discrimination.** HAGRIDv2 weights, designed to distinguish 34 broadly different gesture classes, produce features too coarse to separate four variants of the same heart gesture. General ImageNet features transfer better.

- **YOLO was selected for the deployed interface** despite ResNet18 having slightly higher accuracy, because YOLO achieves more uniform recall across all four classes — minimising missed gestures is more important than raw accuracy for corpus annotation work.

---

## Dataset

The dataset was assembled from three sources: a Roboflow export in YOLOv8 format,
self-recorded video frames, and web-sourced images. After cleaning it contains
**231 images across 4 classes**, split as follows:

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| 2-finger-heart | 47 | 16 | 15 | 78 |
| traditional-heart | 28 | 9 | 10 | 47 |
| traditional-heart-2 | 26 | 9 | 8 | 43 |
| upside-down-heart | 38 | 13 | 12 | 63 |
| **Total** | **139** | **47** | **45** | **231** |

Raw images are not included in this repository. Annotation was performed via
[Roboflow](https://roboflow.com). To replicate data collection, refer to the
methodology section of the technical report.

---

## Pre-trained Models

Model weights are hosted on HuggingFace Hub and downloaded automatically by the
conference demo application. To use them manually:

```python
from huggingface_hub import hf_hub_download

# Download YOLO crop model (recommended — best recall)
path = hf_hub_download(repo_id="shuindec/heart-emblem-models",
                       filename="best_model_yolo.pth")
```

Available checkpoints: `best_model_yolo.pth` · `best_model_yolo_fullframe.pth` ·
`best_model_resnet18.pth` · `temperature.json`

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{nguyen2026weheart,
  title     = {We {\heartsuit} Digital Humanities: Automating Emblem Gesture
               Recognition to Support Data Analysis at Scale},
  author    = {Nguyen, Thanh Anh Thu and Gawne, Lauren and Bishop, Judith and Zhao, Yang},
  year      = {2026},
  note      = {Poster presented at DH2026, La Trobe University},
  url       = {https://huggingface.co/spaces/shuindec/heart-gesture-classification-tool}
}
```

---

## License

This project is licensed under the
[Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE) licence.
You are free to share and adapt the material for any purpose, provided appropriate
credit is given.

---

## Acknowledgements

This project was developed during a research placement at **La Trobe University**
under the supervision of Dr Lauren Gawne, Dr Judith Bishop, and Dr Yang Zhao.
It was presented as a poster at the **Digital Humanities 2026 (DH2026)** conference.
Model hosting is provided by [HuggingFace](https://huggingface.co). Image annotation
was conducted using [Roboflow](https://roboflow.com).