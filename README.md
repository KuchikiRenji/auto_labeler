# Auto Labeler — Automatic Computer Vision Dataset Labeling

**Automatically label computer vision datasets with zero or near-zero manual cost.** A Python library that generates high-quality pseudo-labels for image classification, object detection, instance segmentation, OCR, visual question answering, and feature matching using state-of-the-art models (CLIP, OWL-ViT, SAM, LoFTR, VLMs, TrOCR, and more).

---

## Author & Contact

| | |
|---|---|
| **Author** | KuchikiRenji |
| **Email** | [KuchikiRenji@outlook.com](mailto:KuchikiRenji@outlook.com) |
| **GitHub** | [github.com/KuchikiRenji](https://github.com/KuchikiRenji) |
| **Discord** | `kuchiki_renji` |

---

## Table of Contents

- [What is Auto Labeler?](#what-is-auto-labeler)
- [Features](#features)
- [Supported Tasks & Models](#supported-tasks--models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage by Task](#usage-by-task)
  - [Image Classification](#image-classification)
  - [Object Detection](#object-detection)
  - [Instance Segmentation](#instance-segmentation)
  - [Visual Question Answering](#visual-question-answering)
  - [Feature Matching](#feature-matching)
  - [OCR](#ocr)
- [Roadmap](#roadmap)
- [License & Contributing](#license--contributing)

---

## What is Auto Labeler?

**Auto Labeler** is a simple, modular Python framework for **automatic dataset labeling** and **pseudo-label generation** in computer vision. It wraps popular frameworks (Hugging Face, OpenCLIP, Kornia, etc.) and exposes a **uniform interface** so you can:

- Label images for **classification** (image-to-image or text-to-image retrieval)
- Generate **object detection** and **instance segmentation** labels (zero-shot or prompt-based)
- Run **visual question answering** and **OCR** on images
- Do **feature/keypoint matching** for retrieval and correspondence

Minimal configuration and a single `label.py` entry point per task keep manual effort low while leveraging SOTA architectures.

---

## Features

- **High abstraction** — One interface over Hugging Face, OpenCLIP, and other SOTA sources; less boilerplate for researchers and teams.
- **Modular design** — Separate modules per vision task; easy to add or swap algorithms.
- **Minimal touchpoints** — Set config (model, weights, hyperparameters) and run `label.py`; no deep integration work.
- **Multiple vision tasks** — Classification, detection, segmentation, VQA, OCR, and feature matching in one repo.

---

## Supported Tasks & Models

| Task | Models / Architectures |
|------|------------------------|
| **Image Classification** | [CLIP](https://github.com/mlfoundations/open_clip) (OpenCLIP) |
| **Object Detection** | [OWL-ViT-v2](https://huggingface.co/docs/transformers/model_doc/owlvit) |
| **Instance Segmentation** | [Segment Anything (SAM)](https://huggingface.co/docs/transformers/main/model_doc/sam) |
| **Visual Question Answering** | [LLaVA-NeXT](https://huggingface.co/docs/transformers/en/model_doc/llava_next), [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct), [PaliGemma2](https://huggingface.co/docs/transformers/en/model_doc/paligemma), [Qwen2-VL](https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl), [BLIP](https://huggingface.co/docs/transformers/en/model_doc/blip) |
| **Feature Matching** | [LoFTR](https://kornia.github.io/tutorials/nbs/image_matching.html) (Kornia) |
| **OCR** | [TrOCR](https://huggingface.co/docs/transformers/v4.48.0/en/model_doc/trocr), [docTR (Mindee)](https://github.com/mindee/doctr) |

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/KuchikiRenji/auto_labeler.git
   cd auto_labeler
   ```

2. **Create a virtual environment** (Python 3.8+ recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

Each task has a `label.py` in its folder. General pattern:

```bash
cd <task_folder>   # e.g. image_classification, object_detection, ocr
python label.py --unlabelled-dump <path_to_images> --result-path <output_path> [other options]
```

See [Usage by Task](#usage-by-task) for exact commands and options.

---

## Usage by Task

### Image Classification

```bash
cd image_classification
python label.py \
  --unlabelled-dump 'path/to/unlabelled/images' \
  --class2prompts 'path/to/class_prompts.json' \
  --result-path 'path/to/save/labels'
```

### Object Detection

```bash
cd object_detection
python label.py \
  --unlabelled-dump 'path/to/images' \
  --class-texts-path 'path/to/class_objects.json' \
  --prompt-images 'path/to/prompt_images' \
  --result-path 'path/to/detection_results.json' \
  --viz False \
  --viz-path 'path/to/bbox_viz'
```

### Instance Segmentation

```bash
cd instance_segmentation
python label.py \
  --unlabelled-dump 'path/to/images' \
  --class-texts-path 'path/to/class_objects.json' \
  --result-path 'path/to/segmentation_results.pkl' \
  --viz False \
  --viz-path 'path/to/mask_viz'
```

### Visual Question Answering

```bash
cd visual_question_answering
python label.py \
  --unlabelled-dump 'path/to/images' \
  --result-path 'path/to/vqa_results.json'
```

### Feature Matching

```bash
cd feature_matching
python label.py \
  --unlabelled-dump 'path/to/images' \
  --reference-images 'path/to/reference/index_images' \
  --result-path 'path/to/matching_results'
```

### OCR

```bash
cd ocr
python label.py \
  --unlabelled-dump 'path/to/document_images' \
  --result-path 'path/to/ocr_results.json'
```

---

## Roadmap

- [ ] Config-driven prompting for VLMs
- [ ] Visualization support for LoFTR
- [ ] Support for SuperGlue, SIFT, SURF and other classical feature matching methods

---

## License & Contributing

For issues, feature requests, or contributions, open an issue or PR on [GitHub](https://github.com/KuchikiRenji/auto_labeler) or reach out via the contact details above.

---

*Auto Labeler — automatic computer vision dataset labeling and pseudo-label generation. Author: KuchikiRenji | [GitHub](https://github.com/KuchikiRenji) | [Email](mailto:KuchikiRenji@outlook.com)*
