# ECE 253 Project: Material Texture Classification

A deep learning system for classifying material textures using transfer learning with MobileNetV2. This project implements a robust texture classifier that can identify five material types: drywall, grass, metal, stone, and wood.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Links](#project-links)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Dataset](#dataset)
- [Pretrained Models](#pretrained-models)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Running Inference](#running-inference)
  - [Robustness Testing](#robustness-testing)
- [Augmentation Experiments](#augmentation-experiments)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Project Overview

This project explores the effectiveness of different data augmentation strategies on material texture classification robustness. We use transfer learning with a pretrained MobileNetV2 model and compare three augmentation approaches:

- **None**: No augmentation (baseline)
- **Basic**: Minimal augmentation (random crop and horizontal flip)
- **Advanced**: Full augmentation suite (crop, flips, rotation, color jitter, blur)

The system includes comprehensive robustness testing with various distortion types (blur, noise, brightness, rotation, CLAHE, gamma correction) to evaluate model performance under challenging conditions.

## Project Links

| Item                | Relevant Link                                                                                                                                         |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Project proposal writeup           | [Link to proposal](https://www.overleaf.com/project/68dd9b4ce0ab37bcb4477e84)                                                              |
| Report Writeup             | [Link to report](https://www.overleaf.com/project/693609532e2fd40ca9d70189)                                                                |
| Update Presentation | [Link to update presentation](https://docs.google.com/presentation/d/1XJB_9bpHgP9oPlBVilFr73cW7soRsAsw4blUIN93gIU/edit?usp=sharing)         |
| Final presentation | [Link to final presentation](https://docs.google.com/presentation/d/1EEg7_dOLlnPiNl-PCSVgeXcvAr1EA7--YYcdI8q3LXo/edit?usp=sharing)         |

## Directory Structure

```
ece_253_project/
├── data/                          # Training/validation data (organized by class)
│   ├── drywall/
│   ├── grass/
│   ├── metal/
│   ├── stone/
│   └── wood/
├── models/                        # Saved model checkpoints
│   ├── mobilenet_none_aug.pth
│   ├── mobilenet_basic_aug.pth
│   └── mobilenet_advanced_aug.pth
├── plots/                         # Training curves and comparison plots
│   ├── training_curve_*.png
│   └── confidence_comparison_*.png
├── train.py                       # Training script
├── inference.py                   # Inference with robustness testing
├── model_utils.py                 # Model and transform utilities
├── experiments.py                 # Preprocessing experimentation tool
├── CLAUDE.md                      # Development documentation
└── README.md                      # This file
```

## Setup and Installation

### Requirements
- Python 3.8+
- PyTorch with MPS support (for Apple Silicon) or CPU
- torchvision
- OpenCV (cv2)
- PIL (Pillow)
- matplotlib
- numpy
- tqdm

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ece_253_project

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python pillow matplotlib numpy tqdm
```

## Dataset

The dataset consists of texture images organized into five material classes:

- **drywall**: 129 images
- **grass**: 112 images
- **metal**: 83 images
- **stone**: 120 images
- **wood**: 199 images

### Data Organization

Place your images in the `data/` directory with subdirectories for each class:

```
data/
├── drywall/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── grass/
│   └── ...
├── metal/
│   └── ...
├── stone/
│   └── ...
└── wood/
    └── ...
```

**Important**: Class names must be in alphabetical order as PyTorch's `ImageFolder` automatically assigns class indices alphabetically.

### Data Download Links

[Download Dataset](https://drive.google.com/drive/folders/1svCqcNgnudxJ4vbKvtEfXd0NcTlUKKxa?usp=share_link)

## Pretrained Models

Pretrained models are stored in the `models/` directory. Each model is named according to its augmentation strategy:

- `mobilenet_none_aug.pth` - Trained without augmentation
- `mobilenet_basic_aug.pth` - Trained with basic augmentation
- `mobilenet_advanced_aug.pth` - Trained with advanced augmentation

### Model Details

- **Base Architecture**: MobileNetV2 (pretrained on ImageNet)
- **Transfer Learning**: Feature layers frozen, only final classifier trained
- **Output Classes**: 5 (drywall, grass, metal, stone, wood)
- **Training**: 10 epochs, 80/20 train/validation split, Adam optimizer (lr=1e-3)

*[To be added: Download links or release tag for pretrained models]*

## Usage

### Training Models

To train a model, edit the `AUGMENTATION` variable in `train.py` and run:

```bash
# Edit train.py and set AUGMENTATION to "none", "basic", or "advanced"
python train.py
```

**Configuration options in train.py:**
- `AUGMENTATION`: "none", "basic", or "advanced"
- `BATCH_SIZE`: Default 32
- `EPOCHS`: Default 10
- `LR`: Learning rate, default 1e-3
- `VAL_SPLIT`: Validation split ratio, default 0.2

**Outputs:**
- Model checkpoint: `models/mobilenet_{augmentation}_aug.pth`
- Training curve: `plots/training_curve_{augmentation}_aug.png`

### Running Inference

Basic inference on a single image:

```bash
python inference.py path/to/image.jpg
```

Specify a different model:

```bash
python inference.py path/to/image.jpg --model mobilenet_none_aug.pth
```

### Robustness Testing

Test model robustness with various distortions:

```bash
# Blur distortion
python inference.py image.jpg --distortion blur --intensity 0.7 --save-comparison

# Noise distortion
python inference.py image.jpg --distortion noise --intensity 0.5 --save-comparison

# Brightness adjustment
python inference.py image.jpg --distortion brightness --intensity 0.8 --save-comparison

# Rotation
python inference.py image.jpg --distortion rotation --intensity 0.5 --save-comparison

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
python inference.py image.jpg --distortion clahe --clahe-clip-limit 3.0 --clahe-tile-size 8 --save-comparison

# Gamma correction (brightness via power law)
python inference.py image.jpg --distortion gamma --gamma 0.5 --save-comparison  # Brighter
python inference.py image.jpg --distortion gamma --gamma 2.0 --save-comparison  # Darker

# All distortions combined
python inference.py image.jpg --distortion all --intensity 0.5 --save-comparison
```

**Available distortion types:**
- `blur`: Gaussian blur
- `noise`: Additive Gaussian noise
- `brightness`: Color jitter (brightness and contrast)
- `rotation`: Random rotation
- `clahe`: Contrast enhancement
- `gamma`: Gamma correction
- `all`: Combines blur, noise, brightness, and rotation

**Command-line options:**
- `--model`: Model filename in `models/` directory (default: `mobilenet_advanced_aug.pth`)
- `--distortion`: Type of distortion (default: `none`)
- `--intensity`: Distortion intensity 0.0-1.0 (default: 0.5)
- `--save-comparison`: Save comparison plot with images and confidence bars
- `--clahe-clip-limit`: CLAHE clip limit (default: 2.0)
- `--clahe-tile-size`: CLAHE tile grid size (default: 8)
- `--gamma`: Gamma value (default: 1.0, <1.0=brighter, >1.0=darker)

**Outputs:**
- Console output with predictions and confidence scores
- Comparison plot (if `--save-comparison` used): `plots/confidence_comparison_{image}_{distortion}.png`

## Augmentation Experiments

To compare different augmentation strategies:

### 1. Train three models

```bash
# Edit train.py, set AUGMENTATION = "none"
python train.py

# Edit train.py, set AUGMENTATION = "basic"
python train.py

# Edit train.py, set AUGMENTATION = "advanced"
python train.py
```

### 2. Compare training curves

Check the `plots/` directory for training curves:
- `training_curve_none_aug.png`
- `training_curve_basic_aug.png`
- `training_curve_advanced_aug.png`

### 3. Test robustness

Test each model with the same distortion:

```bash
python inference.py test.jpg --model mobilenet_none_aug.pth --distortion all --intensity 0.5 --save-comparison
python inference.py test.jpg --model mobilenet_basic_aug.pth --distortion all --intensity 0.5 --save-comparison
python inference.py test.jpg --model mobilenet_advanced_aug.pth --distortion all --intensity 0.5 --save-comparison
```

### 4. Analyze results

Compare:
- Validation accuracy from training curves
- Confidence drops under distortions
- Visual quality of predictions in comparison plots

## Model Architecture

**Base Model**: MobileNetV2 (ImageNet pretrained)

**Transfer Learning Approach**:
- Freeze all feature extraction layers (`model.features`)
- Replace final classifier layer with 5-output linear layer
- Train only the classifier head (reduces parameters and training time)

**Data Augmentation Levels**:

1. **None**:
   - Resize to 256x256
   - Center crop to 224x224
   - Normalize with ImageNet statistics

2. **Basic**:
   - Random resized crop (224x224, scale 0.8-1.0)
   - Random horizontal flip (p=0.5)
   - Normalize

3. **Advanced**:
   - Random resized crop (224x224, scale 0.7-1.0, ratio 0.75-1.33)
   - Random horizontal and vertical flips (p=0.5 each)
   - Random rotation (±30 degrees)
   - Color jitter (brightness, contrast, saturation, hue)
   - Gaussian blur (p=0.3)
   - Normalize

**Training Details**:
- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropyLoss
- Batch size: 32
- Epochs: 10
- Train/Val split: 80/20
- Random seed: 42 (for reproducibility)

## Results

*[To be added: Validation accuracies, robustness metrics, confusion matrices, example predictions]*

### Training Curves

Training curves are saved in `plots/training_curve_{augmentation}_aug.png` showing:
- Training loss over epochs
- Validation accuracy over epochs

### Robustness Analysis

Confidence comparison plots show:
- Original and distorted images side-by-side
- Probability distributions for each class
- Confidence drop metric
- Visual differences due to distortions

---

**Note**: This README will be updated with dataset download links and a release tag for the final submission.
