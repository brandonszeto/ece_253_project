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
from tqdm import tqdm

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
        contrast_factor = 0.5 + intensity * 1.0  # Range: 0.5 to 1.5
        transforms.append(
            T.ColorJitter(
                brightness=(brightness_factor, brightness_factor),
                contrast=(contrast_factor, contrast_factor),
            )
        )

    if distortion_type == "rotation" or distortion_type == "all":
        angle = int(intensity * 45)  # 0 to 45 degrees
        transforms.append(T.RandomRotation(degrees=(-angle, angle)))

    # CLAHE and gamma are handled separately in predict function since they need custom parameters

    return T.Compose(transforms) if transforms else T.Compose([])


def add_noise(img_tensor, intensity=0.5):
    """Add Gaussian noise to image tensor."""
    noise = torch.randn_like(img_tensor) * intensity * 0.3
    return torch.clamp(img_tensor + noise, 0, 1)


def predict(
    image_path: str,
    model_path: Path,
    distortion_type="none",
    distortion_intensity=0.5,
    save_comparison=False,
    clahe_clip_limit=2.0,
    clahe_tile_size=8,
    gamma_value=1.0,
):
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
            img_distorted_display = apply_clahe(
                img,
                clip_limit=clahe_clip_limit,
                tile_grid_size=(clahe_tile_size, clahe_tile_size),
            )
        elif distortion_type == "gamma":
            img_distorted_display = apply_gamma_correction(img, gamma=gamma_value)
        else:
            distortion_transform = get_distortion_transform(
                distortion_type, distortion_intensity
            )
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
            img_distorted_display_tensor = add_noise(
                img_distorted_display_tensor.unsqueeze(0), distortion_intensity
            )[0]
            img_distorted_display = T.ToPILImage()(img_distorted_display_tensor)

        with torch.no_grad():
            logits_distorted = model(x_distorted)
            probs_distorted = F.softmax(logits_distorted, dim=1)[0]
            conf_distorted, idx_distorted = probs_distorted.max(dim=0)

        label_distorted = CLASS_NAMES[idx_distorted.item()]
        if distortion_type == "clahe":
            print(
                f"Distorted (CLAHE: clip_limit={clahe_clip_limit:.1f}, tile_size={clahe_tile_size}x{clahe_tile_size}) - Predicted: {label_distorted} ({conf_distorted.item() * 100:.1f}%)"
            )
        elif distortion_type == "gamma":
            print(
                f"Distorted (Gamma: γ={gamma_value:.2f}) - Predicted: {label_distorted} ({conf_distorted.item() * 100:.1f}%)"
            )
        else:
            print(
                f"Distorted ({distortion_type}, intensity={distortion_intensity:.2f}) - Predicted: {label_distorted} ({conf_distorted.item() * 100:.1f}%)"
            )

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
            ax_img_orig.axis("off")
            ax_img_orig.set_title(f"Original Image", fontsize=14, fontweight="bold")

            # Top right: Distorted image
            ax_img_dist = fig.add_subplot(gs[0, 1])
            ax_img_dist.imshow(img_distorted_display)
            ax_img_dist.axis("off")
            if distortion_type == "clahe":
                title_str = f"Distorted Image (CLAHE: clip={clahe_clip_limit:.1f}, tile={clahe_tile_size}x{clahe_tile_size})"
            elif distortion_type == "gamma":
                title_str = f"Distorted Image (Gamma: γ={gamma_value:.2f})"
            else:
                title_str = f"Distorted Image ({distortion_type}, intensity={distortion_intensity:.2f})"
            ax_img_dist.set_title(title_str, fontsize=14, fontweight="bold")

            # Bottom left: Original probabilities
            ax_prob_orig = fig.add_subplot(gs[1, 0])
            colors_orig = [
                "#2ecc71" if i == idx_orig.item() else "#3498db"
                for i in range(len(CLASS_NAMES))
            ]
            ax_prob_orig.barh(CLASS_NAMES, probs_orig.cpu().numpy(), color=colors_orig)
            ax_prob_orig.set_xlabel("Probability", fontsize=12)
            ax_prob_orig.set_title(
                f"Original Prediction: {label_orig} ({conf_orig.item() * 100:.1f}%)",
                fontsize=12,
                fontweight="bold",
            )
            ax_prob_orig.set_xlim(0, 1)
            ax_prob_orig.grid(axis="x", alpha=0.3)

            # Bottom right: Distorted probabilities
            ax_prob_dist = fig.add_subplot(gs[1, 1])
            colors_dist = [
                "#2ecc71" if i == idx_distorted.item() else "#3498db"
                for i in range(len(CLASS_NAMES))
            ]
            ax_prob_dist.barh(
                CLASS_NAMES, probs_distorted.cpu().numpy(), color=colors_dist
            )
            ax_prob_dist.set_xlabel("Probability", fontsize=12)
            ax_prob_dist.set_title(
                f"Distorted Prediction: {label_distorted} ({conf_distorted.item() * 100:.1f}%)",
                fontsize=12,
                fontweight="bold",
            )
            ax_prob_dist.set_xlim(0, 1)
            ax_prob_dist.grid(axis="x", alpha=0.3)

            # Add confidence drop annotation
            fig.suptitle(
                f"Robustness Test: Confidence Drop = {conf_diff:.1f}%",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            plot_filename = f"confidence_comparison_{img_name}_{distortion_type}.png"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved comparison plot to {plot_path}")
            plt.show()
    else:
        # Show all class probabilities for original
        print("\nClass probabilities:")
        for i, p in enumerate(probs_orig):
            print(f"  {CLASS_NAMES[i]:>8}: {p.item() * 100:.1f}%")


def compare_distorted_images(model_path: Path, data_dir: Path, distorted_dir: Path):
    """
    Compare first image from each class with its distorted variants (AWGN, Sim, SP).

    Args:
        model_path: Path to trained model
        data_dir: Path to data directory containing class subdirectories
        distorted_dir: Path to distorted directory containing class/distortion subdirectories
    """
    device = get_device()
    transform = get_eval_transform()

    # Load model
    model = get_model(device=device, freeze_features=True)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Distortion types and their suffixes
    distortion_mappings = {
        "AWGN": "_noise1",
        "Sim": "_sim",
        "SP": "_noise2"
    }

    # Process each class
    for class_name in CLASS_NAMES:
        print(f"\n{'='*70}")
        print(f"Processing class: {class_name.upper()}")
        print(f"{'='*70}")

        # Find first image in data/{class}/
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory {class_dir} does not exist, skipping")
            continue

        # Get first image
        image_files = sorted(
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpeg"))
        )

        if not image_files:
            print(f"Warning: No images found in {class_dir}, skipping")
            continue

        original_path = image_files[0]
        img_basename = original_path.stem  # e.g., "IMG_0283"
        img_ext = original_path.suffix  # e.g., ".jpeg"

        print(f"Original image: {original_path.name}")

        # Find corresponding distorted images
        distorted_paths = {}
        distorted_class_dir = distorted_dir / class_name

        for dist_type, suffix in distortion_mappings.items():
            dist_subdir = distorted_class_dir / dist_type
            if not dist_subdir.exists():
                print(f"Warning: Distortion directory {dist_subdir} does not exist")
                distorted_paths[dist_type] = None
                continue

            # Look for image with expected suffix
            expected_name = f"{img_basename}{suffix}{img_ext}"
            dist_path = dist_subdir / expected_name

            if dist_path.exists():
                distorted_paths[dist_type] = dist_path
                print(f"  {dist_type:4s}: {dist_path.name}")
            else:
                print(f"  {dist_type:4s}: NOT FOUND (expected {expected_name})")
                distorted_paths[dist_type] = None

        # Run inference on all versions
        results = {}

        # Original image
        img_orig = Image.open(original_path).convert("RGB")
        x_orig = transform(img_orig).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_orig = model(x_orig)
            probs_orig = F.softmax(logits_orig, dim=1)[0]
            conf_orig, idx_orig = probs_orig.max(dim=0)

        results["Original"] = {
            "image": img_orig,
            "probs": probs_orig.cpu().numpy(),
            "pred_idx": idx_orig.item(),
            "confidence": conf_orig.item()
        }

        print(f"\nPredictions:")
        print(f"  Original: {CLASS_NAMES[idx_orig.item()]} ({conf_orig.item() * 100:.1f}%)")

        # Distorted images
        for dist_type in ["AWGN", "Sim", "SP"]:
            if distorted_paths[dist_type] is not None:
                img_dist = Image.open(distorted_paths[dist_type]).convert("RGB")
                x_dist = transform(img_dist).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits_dist = model(x_dist)
                    probs_dist = F.softmax(logits_dist, dim=1)[0]
                    conf_dist, idx_dist = probs_dist.max(dim=0)

                results[dist_type] = {
                    "image": img_dist,
                    "probs": probs_dist.cpu().numpy(),
                    "pred_idx": idx_dist.item(),
                    "confidence": conf_dist.item()
                }

                conf_drop = (conf_orig.item() - conf_dist.item()) * 100
                print(f"  {dist_type:4s}:     {CLASS_NAMES[idx_dist.item()]} ({conf_dist.item() * 100:.1f}%) [Δ: {conf_drop:+.1f}%]")

        # Create comparison plot
        plot_distorted_comparison(results, class_name, img_basename, model_path.stem)


def plot_distorted_comparison(results: dict, class_name: str, img_basename: str, model_name: str):
    """
    Create a 4-panel comparison plot showing original and distorted predictions.

    Args:
        results: Dictionary with keys "Original", "AWGN", "Sim", "SP" containing prediction results
        class_name: Name of the material class
        img_basename: Base name of the image file
        model_name: Name of the model used
    """
    # Determine layout based on available results
    available_distortions = [k for k in ["AWGN", "Sim", "SP"] if k in results]
    n_plots = 1 + len(available_distortions)  # Original + distortions

    # Create figure with 2 rows: images on top, probability bars on bottom
    fig = plt.figure(figsize=(5 * n_plots, 10))
    gs = fig.add_gridspec(2, n_plots, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    # Plot original and each distortion
    plot_order = ["Original"] + available_distortions

    for i, result_type in enumerate(plot_order):
        if result_type not in results:
            continue

        result = results[result_type]

        # Top row: Image
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(result["image"])
        ax_img.axis("off")
        ax_img.set_title(f"{result_type}", fontsize=14, fontweight="bold")

        # Bottom row: Probability bars
        ax_prob = fig.add_subplot(gs[1, i])
        colors = [
            "#2ecc71" if j == result["pred_idx"] else "#3498db"
            for j in range(len(CLASS_NAMES))
        ]
        ax_prob.barh(CLASS_NAMES, result["probs"], color=colors)
        ax_prob.set_xlabel("Probability", fontsize=11)

        pred_label = CLASS_NAMES[result["pred_idx"]]
        ax_prob.set_title(
            f"{pred_label}\n({result['confidence'] * 100:.1f}%)",
            fontsize=12,
            fontweight="bold"
        )
        ax_prob.set_xlim(0, 1)
        ax_prob.grid(axis="x", alpha=0.3)

    # Overall title
    fig.suptitle(
        f"Distorted Image Comparison: {class_name.capitalize()}\nModel: {model_name} | Image: {img_basename}",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # Save plot
    plot_filename = f"distorted_comparison_{class_name}_{img_basename}_{model_name}.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {plot_path}")
    plt.close()


def evaluate_dark_directories(
    model_path: Path, data_dir: Path, brightening_methods: dict
):
    """
    Evaluate model accuracy on dark directories with various brightening techniques.

    Args:
        model_path: Path to trained model
        data_dir: Path to data directory containing *_dark subdirectories
        brightening_methods: Dict of {method_name: (distortion_type, params_dict)}

    Returns:
        Dictionary with results for each method
    """
    device = get_device()
    transform = get_eval_transform()

    # Load model
    model = get_model(device=device, freeze_features=True)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Find all *_dark directories
    dark_dirs = sorted(data_dir.glob("*_dark"))

    if not dark_dirs:
        print(f"No *_dark directories found in {data_dir}")
        return None

    print(f"Found {len(dark_dirs)} dark directories: {[d.name for d in dark_dirs]}")

    # Initialize results
    results = {
        method: {"correct": 0, "total": 0, "by_class": {}}
        for method in brightening_methods.keys()
    }
    results["original"] = {"correct": 0, "total": 0, "by_class": {}}

    # Initialize class-specific counters
    for method in results.keys():
        for class_name in CLASS_NAMES:
            results[method]["by_class"][class_name] = {"correct": 0, "total": 0}

    # Process each dark directory
    for dark_dir in dark_dirs:
        # Extract ground truth class from directory name (e.g., "wood_dark" -> "wood")
        true_class = dark_dir.name.replace("_dark", "")
        if true_class not in CLASS_NAMES:
            print(
                f"Warning: Unknown class '{true_class}' from directory '{dark_dir.name}', skipping"
            )
            continue

        true_idx = CLASS_NAMES.index(true_class)

        # Get all images in directory
        image_files = (
            list(dark_dir.glob("*.jpg"))
            + list(dark_dir.glob("*.png"))
            + list(dark_dir.glob("*.jpeg"))
        )

        print(
            f"\nProcessing {dark_dir.name}: {len(image_files)} images (ground truth: {true_class})"
        )

        for img_path in tqdm(image_files, desc=f"  Evaluating {dark_dir.name}"):
            img = Image.open(img_path).convert("RGB")

            # Evaluate original (no brightening)
            x_orig = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x_orig)
                pred_idx = logits.argmax(dim=1).item()

            results["original"]["total"] += 1
            results["original"]["by_class"][true_class]["total"] += 1
            if pred_idx == true_idx:
                results["original"]["correct"] += 1
                results["original"]["by_class"][true_class]["correct"] += 1

            # Evaluate with each brightening method
            for method_name, (distortion_type, params) in brightening_methods.items():
                # Apply brightening
                if distortion_type == "clahe":
                    img_brightened = apply_clahe(
                        img,
                        clip_limit=params.get("clip_limit", 2.0),
                        tile_grid_size=(
                            params.get("tile_size", 8),
                            params.get("tile_size", 8),
                        ),
                    )
                elif distortion_type == "gamma":
                    img_brightened = apply_gamma_correction(
                        img, gamma=params.get("gamma", 0.5)
                    )
                elif distortion_type == "brightness":
                    brightness_transform = T.ColorJitter(
                        brightness=(
                            params.get("brightness", 1.5),
                            params.get("brightness", 1.5),
                        ),
                        contrast=(
                            params.get("contrast", 1.3),
                            params.get("contrast", 1.3),
                        ),
                    )
                    img_brightened = brightness_transform(img)
                else:
                    img_brightened = img

                x_brightened = transform(img_brightened).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x_brightened)
                    pred_idx = logits.argmax(dim=1).item()

                results[method_name]["total"] += 1
                results[method_name]["by_class"][true_class]["total"] += 1
                if pred_idx == true_idx:
                    results[method_name]["correct"] += 1
                    results[method_name]["by_class"][true_class]["correct"] += 1

    return results


def plot_brightening_comparison(results: dict, model_name: str, plot_suffix: str = ""):
    """
    Create a comparison chart showing accuracy with different brightening methods.

    Args:
        results: Results dictionary from evaluate_dark_directories
        model_name: Name of the model being evaluated
        plot_suffix: Optional suffix for plot filename
    """
    if results is None:
        return

    # Calculate overall accuracies
    methods = list(results.keys())
    accuracies = [
        results[m]["correct"] / results[m]["total"] * 100
        if results[m]["total"] > 0
        else 0
        for m in methods
    ]

    # Calculate per-class accuracies
    class_accuracies = {}
    for class_name in CLASS_NAMES:
        class_accuracies[class_name] = []
        for method in methods:
            by_class = results[method]["by_class"][class_name]
            acc = (
                by_class["correct"] / by_class["total"] * 100
                if by_class["total"] > 0
                else 0
            )
            class_accuracies[class_name].append(acc)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Overall accuracy comparison
    colors = ["#e74c3c" if m == "original" else "#2ecc71" for m in methods]
    bars = ax1.bar(
        range(len(methods)), accuracies, color=colors, alpha=0.7, edgecolor="black"
    )
    ax1.set_xlabel("Brightening Method", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Dark Image Classification Accuracy\nModel: {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 2: Per-class accuracy comparison
    x = np.arange(len(methods))
    width = 0.15

    for i, class_name in enumerate(CLASS_NAMES):
        if any(results[m]["by_class"][class_name]["total"] > 0 for m in methods):
            offset = width * (i - len(CLASS_NAMES) / 2)
            ax2.bar(
                x + offset,
                class_accuracies[class_name],
                width,
                label=class_name,
                alpha=0.8,
            )

    ax2.set_xlabel("Brightening Method", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Per-Class Accuracy with Different Brightening Methods",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.set_ylim(0, 100)
    ax2.legend(title="Material Class", loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_filename = (
        f"dark_brightening_comparison_{model_name.replace('.pth', '')}{plot_suffix}.png"
    )
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {plot_path}")
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("BRIGHTENING COMPARISON SUMMARY")
    print("=" * 70)
    for method in methods:
        acc = results[method]["correct"] / results[method]["total"] * 100
        total = results[method]["total"]
        correct = results[method]["correct"]
        print(f"\n{method.upper():20s}: {correct:4d}/{total:4d} correct ({acc:.2f}%)")

        # Show per-class breakdown
        for class_name in CLASS_NAMES:
            by_class = results[method]["by_class"][class_name]
            if by_class["total"] > 0:
                class_acc = by_class["correct"] / by_class["total"] * 100
                print(
                    f"  {class_name:10s}: {by_class['correct']:3d}/{by_class['total']:3d} ({class_acc:.1f}%)"
                )


def plot_clahe_heatmap(
    results: dict, model_name: str, clip_limits: list, tile_sizes: list
):
    """
    Create a heatmap showing CLAHE parameter performance.

    Args:
        results: Results dictionary with CLAHE parameters as keys
        model_name: Name of the model being evaluated
        clip_limits: List of clip limit values tested
        tile_sizes: List of tile size values tested
    """
    # Extract accuracies into a grid
    accuracy_grid = np.zeros((len(clip_limits), len(tile_sizes)))

    for i, clip_limit in enumerate(clip_limits):
        for j, tile_size in enumerate(tile_sizes):
            method_key = f"clahe_c{clip_limit:.1f}_t{tile_size}"
            if method_key in results:
                acc = (
                    results[method_key]["correct"] / results[method_key]["total"] * 100
                )
                accuracy_grid[i, j] = acc

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(accuracy_grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(tile_sizes)))
    ax.set_yticks(np.arange(len(clip_limits)))
    ax.set_xticklabels(tile_sizes)
    ax.set_yticklabels([f"{cl:.1f}" for cl in clip_limits])

    # Labels
    ax.set_xlabel("Tile Grid Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Clip Limit", fontsize=14, fontweight="bold")
    ax.set_title(
        f"CLAHE Parameter Sweep: Accuracy Heatmap\nModel: {model_name}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add text annotations
    for i in range(len(clip_limits)):
        for j in range(len(tile_sizes)):
            text = ax.text(
                j,
                i,
                f"{accuracy_grid[i, j]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Save plot
    plot_filename = f"clahe_parameter_sweep_{model_name.replace('.pth', '')}.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved CLAHE parameter sweep heatmap to {plot_path}")
    plt.show()

    # Find best parameters
    best_idx = np.unravel_index(np.argmax(accuracy_grid), accuracy_grid.shape)
    best_clip = clip_limits[best_idx[0]]
    best_tile = tile_sizes[best_idx[1]]
    best_acc = accuracy_grid[best_idx[0], best_idx[1]]

    print("\n" + "=" * 70)
    print("CLAHE PARAMETER SWEEP RESULTS")
    print("=" * 70)
    print(f"\nBest Parameters:")
    print(f"  Clip Limit: {best_clip:.1f}")
    print(f"  Tile Size:  {best_tile}x{best_tile}")
    print(f"  Accuracy:   {best_acc:.2f}%")

    # Compare with original
    if "original" in results:
        original_acc = (
            results["original"]["correct"] / results["original"]["total"] * 100
        )
        improvement = best_acc - original_acc
        print(
            f"\nImprovement over original: {improvement:.2f}% ({original_acc:.2f}% → {best_acc:.2f}%)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on an image with optional distortion"
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to input image (not needed for --eval-dark mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_advanced_aug.pth",
        help="Model filename in models/ directory (default: mobilenet_advanced_aug.pth)",
    )
    parser.add_argument(
        "--distortion",
        type=str,
        default="none",
        choices=[
            "none",
            "blur",
            "noise",
            "brightness",
            "rotation",
            "clahe",
            "gamma",
            "all",
        ],
        help="Type of distortion to apply (default: none)",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.5,
        help="Distortion intensity 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save confidence comparison plot to plots/ directory",
    )

    # CLAHE-specific parameters
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (default: 2.0, higher = more contrast)",
    )
    parser.add_argument(
        "--clahe-tile-size",
        type=int,
        default=8,
        help="CLAHE tile grid size (default: 8 for 8x8 grid)",
    )

    # Gamma correction parameter
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction value (default: 1.0, < 1.0 = brighter, > 1.0 = darker)",
    )

    # Dark directory evaluation mode
    parser.add_argument(
        "--eval-dark",
        action="store_true",
        help="Evaluate all *_dark directories with brightening methods comparison",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing *_dark subdirectories (default: data)",
    )

    # CLAHE parameter sweep mode
    parser.add_argument(
        "--clahe-sweep",
        action="store_true",
        help="Perform CLAHE parameter sweep over clip_limit and tile_size ranges",
    )

    # Distorted directory comparison mode
    parser.add_argument(
        "--compare-distorted",
        action="store_true",
        help="Compare first image from each class with distorted variants (AWGN, Sim, SP)",
    )
    parser.add_argument(
        "--distorted-dir",
        type=str,
        default="distorted",
        help="Path to distorted directory (default: distorted)",
    )

    args = parser.parse_args()

    model_path = models_dir / args.model
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        print(f"Available models in {models_dir}:")
        for model_file in models_dir.glob("*.pth"):
            print(f"  - {model_file.name}")
        sys.exit(1)

    # CLAHE parameter sweep mode
    if args.clahe_sweep:
        print("=" * 70)
        print("CLAHE PARAMETER SWEEP MODE")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Data directory: {args.data_dir}")

        # Define parameter ranges
        clip_limits = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        tile_sizes = [4, 6, 8, 10, 12, 16]

        print(f"\nClip limits to test: {clip_limits}")
        print(f"Tile sizes to test: {tile_sizes}")
        print(f"Total combinations: {len(clip_limits) * len(tile_sizes)}")

        # Build CLAHE methods dictionary
        brightening_methods = {}
        for clip_limit in clip_limits:
            for tile_size in tile_sizes:
                method_name = f"clahe_c{clip_limit:.1f}_t{tile_size}"
                brightening_methods[method_name] = (
                    "clahe",
                    {"clip_limit": clip_limit, "tile_size": tile_size},
                )

        data_dir_path = Path(args.data_dir)
        results = evaluate_dark_directories(
            model_path, data_dir_path, brightening_methods
        )

        if results:
            plot_clahe_heatmap(results, args.model, clip_limits, tile_sizes)

    # Distorted directory comparison mode
    elif args.compare_distorted:
        print("=" * 70)
        print("DISTORTED DIRECTORY COMPARISON MODE")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Data directory: {args.data_dir}")
        print(f"Distorted directory: {args.distorted_dir}")

        data_dir_path = Path(args.data_dir)
        distorted_dir_path = Path(args.distorted_dir)

        if not distorted_dir_path.exists():
            print(f"Error: Distorted directory '{distorted_dir_path}' does not exist.")
            sys.exit(1)

        compare_distorted_images(model_path, data_dir_path, distorted_dir_path)

    # Dark directory evaluation mode
    elif args.eval_dark:
        print("=" * 70)
        print("DARK DIRECTORY EVALUATION MODE")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Data directory: {args.data_dir}")

        # Define brightening methods to test
        brightening_methods = {
            "gamma_1": ("gamma", {"gamma": 1.0}),
            "gamma_1.2": ("gamma", {"gamma": 1.2}),
            "gamma_1.4": ("gamma", {"gamma": 1.4}),
            "gamma_1.6": ("gamma", {"gamma": 1.6}),
            "gamma_1.8": ("gamma", {"gamma": 1.8}),
            "gamma_2.0": ("gamma", {"gamma": 2.0}),
            # "clahe_2.0": ("clahe", {"clip_limit": 2.0, "tile_size": 8}),
            # "clahe_3.0": ("clahe", {"clip_limit": 3.0, "tile_size": 8}),
            # "clahe_5": ("clahe", {"clip_limit": 5.0, "tile_size": 16}),
            # "brightness": ("brightness", {"brightness": 1.8, "contrast": 1.5}),
        }

        print("\nBrightening methods to test:")
        for method, (dtype, params) in brightening_methods.items():
            print(f"  - {method}: {dtype} with params {params}")

        data_dir_path = Path(args.data_dir)
        results = evaluate_dark_directories(
            model_path, data_dir_path, brightening_methods
        )

        if results:
            plot_brightening_comparison(results, args.model)

    # Single image inference mode
    else:
        if args.image_path is None:
            print("Error: image_path is required when not using --eval-dark mode")
            print("Usage: python inference.py <image_path> [options]")
            print("   or: python inference.py --eval-dark [options]")
            sys.exit(1)

        predict(
            args.image_path,
            model_path,
            distortion_type=args.distortion,
            distortion_intensity=args.intensity,
            save_comparison=args.save_comparison,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_size=args.clahe_tile_size,
            gamma_value=args.gamma,
        )
