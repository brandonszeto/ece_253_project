# inference.py

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    get_device,
    get_eval_transform,
    get_model,
)

models_dir = Path("models")
plots_dir = Path("plots")

# Ensure plots directory exists
plots_dir.mkdir(exist_ok=True)


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Args:
        img: PIL Image
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        PIL Image with CLAHE applied
    """
    # Convert PIL to numpy array
    img_np = np.array(img)

    # Convert RGB to LAB color space
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])

    # Convert back to RGB
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img_clahe)


def apply_gamma_correction(img, gamma=1.0):
    """
    Apply gamma correction to an image.

    Args:
        img: PIL Image
        gamma: Gamma value (< 1.0 = brighter, > 1.0 = darker, 1.0 = no change)
               Typical range: 0.5 (very bright) to 2.5 (very dark)

    Returns:
        PIL Image with gamma correction applied
    """
    # Convert PIL to numpy array and normalize to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0

    # Apply gamma correction: output = input^gamma
    img_gamma = np.power(img_np, gamma)

    # Convert back to [0, 255] range
    img_gamma = (img_gamma * 255).astype(np.uint8)

    return Image.fromarray(img_gamma)


def get_distortion_transform(distortion_type="none", intensity=0.5):
    """
    Create a transform that applies distortions to images.

    Args:
        distortion_type: Type of distortion - "none", "blur", "noise", "brightness", "rotation", "clahe", "gamma", "all"
        intensity: Distortion intensity (0.0 to 1.0)
    """
    if distortion_type == "none":
        return T.Compose([])

    transforms = []

    if distortion_type == "blur" or distortion_type == "all":
        kernel_size = int(5 + intensity * 10)  # 5 to 15
        if kernel_size % 2 == 0:
            kernel_size += 1
        transforms.append(T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)))

    if distortion_type == "noise" or distortion_type == "all":
        # Noise will be added separately in predict function
        pass

    if distortion_type == "brightness" or distortion_type == "all":
        # More aggressive brightness adjustment for visibility
        brightness_factor = 0.5 + intensity * 1.5  # Range: 0.5 to 2.0
        contrast_factor = 0.5 + intensity * 1.0    # Range: 0.5 to 1.5
        transforms.append(T.ColorJitter(brightness=(brightness_factor, brightness_factor),
                                       contrast=(contrast_factor, contrast_factor)))

    if distortion_type == "rotation" or distortion_type == "all":
        angle = int(intensity * 45)  # 0 to 45 degrees
        transforms.append(T.RandomRotation(degrees=(-angle, angle)))

    # CLAHE and gamma are handled separately in predict function since they need custom parameters

    return T.Compose(transforms) if transforms else T.Compose([])


def add_noise(img_tensor, intensity=0.5):
    """Add Gaussian noise to image tensor."""
    noise = torch.randn_like(img_tensor) * intensity * 0.3
    return torch.clamp(img_tensor + noise, 0, 1)


def predict(image_path: str, model_path: Path, distortion_type="none", distortion_intensity=0.5,
            save_comparison=False, clahe_clip_limit=2.0, clahe_tile_size=8, gamma_value=1.0):
    """
    Run inference on an image with optional distortion and comparison.

    Args:
        image_path: Path to input image
        model_path: Path to trained model
        distortion_type: Type of distortion to apply
        distortion_intensity: Intensity of distortion (0.0 to 1.0)
        save_comparison: Whether to save confidence comparison plot
        clahe_clip_limit: CLAHE clip limit parameter (default: 2.0)
        clahe_tile_size: CLAHE tile grid size (default: 8 for 8x8 grid)
        gamma_value: Gamma correction value (default: 1.0, < 1.0 = brighter, > 1.0 = darker)
    """
    device = get_device()
    transform = get_eval_transform()

    # Load model
    model = get_model(device=device, freeze_features=True)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load and preprocess original image
    img = Image.open(image_path).convert("RGB")
    img_name = Path(image_path).stem

    # Predict on original image
    x_orig = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_orig = model(x_orig)
        probs_orig = F.softmax(logits_orig, dim=1)[0]
        conf_orig, idx_orig = probs_orig.max(dim=0)

    label_orig = CLASS_NAMES[idx_orig.item()]
    print(f"Original - Predicted: {label_orig} ({conf_orig.item() * 100:.1f}%)")

    # If distortion requested, predict on distorted image
    if distortion_type != "none":
        # Handle CLAHE and gamma separately since they have custom parameters
        if distortion_type == "clahe":
            img_distorted_display = apply_clahe(img, clip_limit=clahe_clip_limit,
                                                tile_grid_size=(clahe_tile_size, clahe_tile_size))
        elif distortion_type == "gamma":
            img_distorted_display = apply_gamma_correction(img, gamma=gamma_value)
        else:
            distortion_transform = get_distortion_transform(distortion_type, distortion_intensity)
            # Apply distortion - keep a copy for visualization
            img_distorted_display = distortion_transform(img)

        # Apply eval transform for model input
        x_distorted = transform(img_distorted_display).unsqueeze(0).to(device)

        # Add noise if requested (after normalization)
        if distortion_type == "noise" or distortion_type == "all":
            x_distorted = add_noise(x_distorted, distortion_intensity)
            # For visualization, also apply noise to display image
            # Convert to tensor, add noise, convert back
            img_distorted_display_tensor = T.ToTensor()(img_distorted_display)
            img_distorted_display_tensor = add_noise(img_distorted_display_tensor.unsqueeze(0), distortion_intensity)[0]
            img_distorted_display = T.ToPILImage()(img_distorted_display_tensor)

        with torch.no_grad():
            logits_distorted = model(x_distorted)
            probs_distorted = F.softmax(logits_distorted, dim=1)[0]
            conf_distorted, idx_distorted = probs_distorted.max(dim=0)

        label_distorted = CLASS_NAMES[idx_distorted.item()]
        if distortion_type == "clahe":
            print(f"Distorted (CLAHE: clip_limit={clahe_clip_limit:.1f}, tile_size={clahe_tile_size}x{clahe_tile_size}) - Predicted: {label_distorted} ({conf_distorted.item() * 100:.1f}%)")
        elif distortion_type == "gamma":
            print(f"Distorted (Gamma: γ={gamma_value:.2f}) - Predicted: {label_distorted} ({conf_distorted.item() * 100:.1f}%)")
        else:
            print(f"Distorted ({distortion_type}, intensity={distortion_intensity:.2f}) - Predicted: {label_distorted} ({conf_distorted.item() * 100:.1f}%)")

        # Show confidence difference
        conf_diff = (conf_orig.item() - conf_distorted.item()) * 100
        print(f"Confidence drop: {conf_diff:.1f}%")

        # Save comparison plot if requested
        if save_comparison:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

            # Top left: Original image
            ax_img_orig = fig.add_subplot(gs[0, 0])
            ax_img_orig.imshow(img)
            ax_img_orig.axis('off')
            ax_img_orig.set_title(f"Original Image", fontsize=14, fontweight='bold')

            # Top right: Distorted image
            ax_img_dist = fig.add_subplot(gs[0, 1])
            ax_img_dist.imshow(img_distorted_display)
            ax_img_dist.axis('off')
            if distortion_type == "clahe":
                title_str = f"Distorted Image (CLAHE: clip={clahe_clip_limit:.1f}, tile={clahe_tile_size}x{clahe_tile_size})"
            elif distortion_type == "gamma":
                title_str = f"Distorted Image (Gamma: γ={gamma_value:.2f})"
            else:
                title_str = f"Distorted Image ({distortion_type}, intensity={distortion_intensity:.2f})"
            ax_img_dist.set_title(title_str, fontsize=14, fontweight='bold')

            # Bottom left: Original probabilities
            ax_prob_orig = fig.add_subplot(gs[1, 0])
            colors_orig = ['#2ecc71' if i == idx_orig.item() else '#3498db' for i in range(len(CLASS_NAMES))]
            ax_prob_orig.barh(CLASS_NAMES, probs_orig.cpu().numpy(), color=colors_orig)
            ax_prob_orig.set_xlabel("Probability", fontsize=12)
            ax_prob_orig.set_title(f"Original Prediction: {label_orig} ({conf_orig.item() * 100:.1f}%)",
                                  fontsize=12, fontweight='bold')
            ax_prob_orig.set_xlim(0, 1)
            ax_prob_orig.grid(axis='x', alpha=0.3)

            # Bottom right: Distorted probabilities
            ax_prob_dist = fig.add_subplot(gs[1, 1])
            colors_dist = ['#2ecc71' if i == idx_distorted.item() else '#3498db' for i in range(len(CLASS_NAMES))]
            ax_prob_dist.barh(CLASS_NAMES, probs_distorted.cpu().numpy(), color=colors_dist)
            ax_prob_dist.set_xlabel("Probability", fontsize=12)
            ax_prob_dist.set_title(f"Distorted Prediction: {label_distorted} ({conf_distorted.item() * 100:.1f}%)",
                                  fontsize=12, fontweight='bold')
            ax_prob_dist.set_xlim(0, 1)
            ax_prob_dist.grid(axis='x', alpha=0.3)

            # Add confidence drop annotation
            fig.suptitle(f"Robustness Test: Confidence Drop = {conf_diff:.1f}%",
                        fontsize=16, fontweight='bold', y=0.98)

            plot_filename = f"confidence_comparison_{img_name}_{distortion_type}.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved comparison plot to {plot_path}")
            plt.show()
    else:
        # Show all class probabilities for original
        print("\nClass probabilities:")
        for i, p in enumerate(probs_orig):
            print(f"  {CLASS_NAMES[i]:>8}: {p.item() * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image with optional distortion")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="mobilenet_advanced_aug.pth",
                        help="Model filename in models/ directory (default: mobilenet_advanced_aug.pth)")
    parser.add_argument("--distortion", type=str, default="none",
                        choices=["none", "blur", "noise", "brightness", "rotation", "clahe", "gamma", "all"],
                        help="Type of distortion to apply (default: none)")
    parser.add_argument("--intensity", type=float, default=0.5,
                        help="Distortion intensity 0.0-1.0 (default: 0.5)")
    parser.add_argument("--save-comparison", action="store_true",
                        help="Save confidence comparison plot to plots/ directory")

    # CLAHE-specific parameters
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0,
                        help="CLAHE clip limit (default: 2.0, higher = more contrast)")
    parser.add_argument("--clahe-tile-size", type=int, default=8,
                        help="CLAHE tile grid size (default: 8 for 8x8 grid)")

    # Gamma correction parameter
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction value (default: 1.0, < 1.0 = brighter, > 1.0 = darker)")

    args = parser.parse_args()

    model_path = models_dir / args.model
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        print(f"Available models in {models_dir}:")
        for model_file in models_dir.glob("*.pth"):
            print(f"  - {model_file.name}")
        sys.exit(1)

    predict(
        args.image_path,
        model_path,
        distortion_type=args.distortion,
        distortion_intensity=args.intensity,
        save_comparison=args.save_comparison,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_size=args.clahe_tile_size,
        gamma_value=args.gamma
    )
