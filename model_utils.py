# model_utils.py

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

# IMPORTANT: must match the folder names in data/, in alphabetical order
# ImageFolder will map: {'drywall': 0, 'grass': 1, 'metal': 2, 'stone': 3, 'wood': 4}
CLASS_NAMES = ["drywall", "grass", "metal", "stone", "wood"]
NUM_CLASSES = len(CLASS_NAMES)


def get_device():
    # Use MPS on Apple Silicon if available, otherwise CPU
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_train_transform(augmentation_level="advanced"):
    """
    Get training transform with specified augmentation level.

    Args:
        augmentation_level: "none", "basic", or "advanced"
    """
    if augmentation_level == "none":
        # No augmentation, just resize and normalize
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    elif augmentation_level == "basic":
        # Basic augmentation: flips and crop only
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    else:  # "advanced"
        # Advanced augmentation: all transformations
        return T.Compose([
            # Random crop & resize (simulates zoom/position)
            T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            # Flips – textures usually okay with this
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # Small rotation – don't go too crazy to avoid making them unnatural
            T.RandomRotation(degrees=30),
            # Lighting / color changes
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
            ),
            # Slight blur sometimes helps robustness
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def get_eval_transform():
    # For validation / test – no randomness
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_model(device=None, freeze_features=True):
    if device is None:
        device = get_device()

    # Pretrained MobileNetV2
    model = models.mobilenet_v2(weights="DEFAULT")

    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    model = model.to(device)
    return model
