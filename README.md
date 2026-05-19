# 🧠 Real-Time Face Analysis System

A real-time computer vision system that uses a webcam feed to detect faces and classify them using deep learning. Built with **OpenCV** for face detection, **TensorFlow / Keras** for CNN-based classification, and a polished overlay UI rendered directly on the video stream.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-GPL--3.0-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Running Real-Time Inference](#running-real-time-inference)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This project implements an end-to-end pipeline for real-time face analysis:

1. **Face Detection** — Haar Cascade classifiers (OpenCV) locate faces in each webcam frame.
2. **Classification** — A custom CNN model predicts categories for each detected face (e.g., emotion, mask presence).
3. **Overlay UI** — Bounding boxes, labels, and confidence scores are drawn on the live video feed.

The system supports two classification tasks out of the box:

| Task | Dataset | Classes |
|------|---------|---------|
| **Emotion Recognition** | FER2013 | Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral |
| **Face Mask Detection** | Face Mask Dataset | With Mask, Without Mask |

> You can easily switch between tasks or add your own by changing the dataset and config.

---

## Features

- 🎥 Real-time webcam face detection using Haar Cascades
- 🧠 Custom CNN architecture built with TensorFlow / Keras
- 📊 Training pipeline with data augmentation, early stopping, and learning rate scheduling
- 📈 Automatic generation of training history plots (accuracy & loss curves)
- 🖼️ Confidence-scored bounding boxes with color-coded labels
- ⚙️ Centralized YAML configuration for easy experimentation
- 💾 Model checkpointing — automatically saves the best model during training

---

## Project Structure

```
SZI-project/
├── README.md                   # This file
├── LICENSE                     # GPL-3.0 license
├── requirements.txt            # Python dependencies
├── config.yaml                 # Central configuration file
│
├── data/                       # Datasets (not tracked by git)
│   ├── raw/                    # Original downloaded archives
│   └── processed/              # Organized train/val/test splits
│       ├── train/
│       │   ├── class_0/
│       │   └── class_1/
│       ├── val/
│       │   ├── class_0/
│       │   └── class_1/
│       └── test/
│           ├── class_0/
│           └── class_1/
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py               # Configuration loader (reads config.yaml)
│   ├── data_preprocessing.py   # Dataset loading, augmentation, generators
│   ├── model.py                # CNN model definition
│   ├── train.py                # Training loop with callbacks
│   ├── evaluate.py             # Model evaluation & metrics
│   ├── detect.py               # Haar Cascade face detection utilities
│   ├── camera.py               # Webcam capture & real-time inference
│   └── utils.py                # Helper functions (plotting, logging)
│
├── models/                     # Saved models (not tracked by git)
│   └── best_model.keras        # Best model checkpoint
│
├── outputs/                    # Training outputs (not tracked by git)
│   ├── training_history.png    # Accuracy & loss curves
│   └── confusion_matrix.png   # Confusion matrix visualization
│
├── notebooks/                  # Jupyter notebooks for exploration
│   └── exploration.ipynb       # Data exploration & prototyping
│
└── scripts/                    # Entry-point scripts
    ├── prepare_data.py         # Download & organize dataset
    ├── train_model.py          # Train the CNN
    ├── evaluate_model.py       # Evaluate on test set
    └── run_camera.py           # Launch real-time webcam inference
```

---

## Prerequisites

- **Python** 3.10 or higher
- **Webcam** (built-in or USB)
- **GPU** (optional but recommended for training) — NVIDIA GPU with CUDA support

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/SZI-project.git
   cd SZI-project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation

### Option A: Emotion Recognition (FER2013)

1. Go to [Kaggle — FER2013](https://www.kaggle.com/datasets/msambare/fer2013) and download the dataset.
2. Extract the archive into `data/processed/` so that the folder structure matches:
   ```
   data/processed/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── sad/
   │   ├── surprise/
   │   └── neutral/
   └── test/
       ├── angry/
       └── ...
   ```

### Option B: Face Mask Detection

1. Go to [Kaggle — Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) and download the dataset.
2. Organize images into `data/processed/train/` and `data/processed/test/` with subfolders `with_mask/` and `without_mask/`.

> **Tip:** You can use `scripts/prepare_data.py` to automate the split into train/val/test.

---

## Training the Model

```bash
python scripts/train_model.py
```

This will:
- Load images from `data/processed/train/` and `data/processed/val/`
- Apply data augmentation (rotation, flip, zoom, shift)
- Train the CNN for the configured number of epochs
- Save the best model to `models/best_model.keras`
- Generate training plots in `outputs/`

### Training Configuration

All hyperparameters are controlled via `config.yaml`:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
```

---

## Running Real-Time Inference

```bash
python scripts/run_camera.py
```

This will:
- Open your webcam
- Detect faces using Haar Cascade
- Classify each face using the trained model
- Display bounding boxes with predicted labels and confidence scores

**Controls:**
- Press `q` to quit
- Press `s` to save a screenshot

---

## Configuration

All settings live in `config.yaml`:

| Key | Description | Default |
|-----|-------------|---------|
| `task` | Classification task (`emotion` or `mask`) | `emotion` |
| `image_size` | Input image dimensions | `48` |
| `batch_size` | Training batch size | `32` |
| `epochs` | Maximum training epochs | `50` |
| `learning_rate` | Initial learning rate | `0.001` |
| `early_stopping_patience` | Epochs to wait before early stopping | `10` |
| `model_path` | Where to save/load the trained model | `models/best_model.keras` |
| `cascade_path` | Path to Haar Cascade XML | (OpenCV default) |

---

## Architecture

### CNN Model

```
Input (48×48×1 or 48×48×3)
  │
  ├── Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
  ├── Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
  ├── Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
  ├── Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
  │
  ├── Flatten
  ├── Dense(512) → BatchNorm → ReLU → Dropout(0.5)
  ├── Dense(256) → BatchNorm → ReLU → Dropout(0.5)
  └── Dense(num_classes) → Softmax
```

### Pipeline

```
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  Webcam   │───▶│ Haar Det. │───▶│ CNN Pred │───▶│ Overlay  │
│  (OpenCV) │    │ (Faces)   │    │ (Keras)  │    │ (UI)     │
└──────────┘    └───────────┘    └──────────┘    └──────────┘
```

---

## Results

After training, check `outputs/` for:
- **`training_history.png`** — Accuracy and loss curves over epochs
- **`confusion_matrix.png`** — Per-class performance on the test set

Expected performance (FER2013):
- Validation accuracy: **~62–68%** (state of the art for this dataset is ~75%)
- Real-time FPS: **15–30 FPS** depending on hardware

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `cv2.error: camera not found` | Check webcam connection, try changing camera index in `config.yaml` |
| `CUDA out of memory` | Reduce `batch_size` in `config.yaml` |
| Low accuracy | Increase epochs, try different augmentation, ensure dataset is balanced |
| Slow inference | Ensure model is loaded once (not per-frame), reduce image size |

---

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.