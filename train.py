# train.py

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from tqdm.auto import tqdm

from model_utils import (
    CLASS_NAMES,
    get_device,
    get_eval_transform,
    get_model,
    get_train_transform,
)

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
VAL_SPLIT = 0.2  # 20% of data for validation
RANDOM_SEED = 42
AUGMENTATION = "advanced"  # Options: "none", "basic", "advanced"
INCLUDE_DARK = True  # Set to True to include dark variants (except drywall_dark)

data_dir = Path(
    "data"
)  # expects data/drywall, data/grass, data/metal, data/stone, data/wood
models_dir = Path("models")
plots_dir = Path("plots")

# Ensure output directories exist
models_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

# Model filename includes augmentation level and dark variant flag
model_suffix = "_dark" if INCLUDE_DARK else ""
MODEL_PATH = models_dir / f"mobilenet_{AUGMENTATION}_aug{model_suffix}.pth"

device = get_device()
print("Using device:", device)
print(f"Augmentation level: {AUGMENTATION}")
print(f"Include dark variants: {INCLUDE_DARK}")

train_transform = get_train_transform(augmentation_level=AUGMENTATION)
val_transform = get_eval_transform()


class TransformSubset(Dataset):
    """
    Wrap a Subset and apply a transform on the fly.
    Assumes the underlying dataset returns (PIL_image, label).
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class CombinedDataset(Dataset):
    """
    Combines regular and dark variant datasets.
    Maps both to the same class labels.
    """

    def __init__(self, datasets_list, transform=None):
        self.datasets = datasets_list
        self.transform = transform
        self.cumulative_sizes = [0]
        for ds in datasets_list:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(ds))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes[1:]):
            if idx < cumsum:
                dataset_idx = i
                break

        # Get item from appropriate dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        img, label = self.datasets[dataset_idx][local_idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# Load dataset(s)
if INCLUDE_DARK:
    print("\n" + "=" * 70)
    print("Loading regular and dark variant datasets")
    print("=" * 70)

    # Load regular dataset
    regular_ds = datasets.ImageFolder(str(data_dir), transform=None)
    print(f"Regular dataset classes: {regular_ds.classes}")
    print(f"Regular dataset size: {len(regular_ds)}")

    # Create a custom dataset that includes dark variants
    # We need to manually load dark directories and map them to the same class indices
    import os

    from PIL import Image

    class DarkVariantDataset(Dataset):
        def __init__(self, data_dir, class_names, transform=None):
            self.data_dir = Path(data_dir)
            self.class_names = class_names
            self.transform = transform
            self.samples = []

            # Load images from *_dark directories (except drywall_dark which doesn't exist)
            for class_name in class_names:
                if class_name == "drywall":
                    continue  # Skip drywall_dark as it doesn't exist

                dark_dir = self.data_dir / f"{class_name}_dark"
                if not dark_dir.exists():
                    print(f"Warning: {dark_dir} does not exist, skipping")
                    continue

                class_idx = class_names.index(class_name)
                image_files = (
                    list(dark_dir.glob("*.jpg"))
                    + list(dark_dir.glob("*.png"))
                    + list(dark_dir.glob("*.jpeg"))
                )

                for img_path in image_files:
                    self.samples.append((str(img_path), class_idx))

                print(
                    f"Loaded {len(image_files)} images from {dark_dir.name} (class: {class_name})"
                )

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
            return img, label

    dark_ds = DarkVariantDataset(data_dir, CLASS_NAMES, transform=None)
    print(f"\nDark variant dataset size: {len(dark_ds)}")
    print(f"Total combined size: {len(regular_ds) + len(dark_ds)}")

    # Combine datasets
    full_ds = CombinedDataset([regular_ds, dark_ds], transform=None)

else:
    # Load only regular dataset
    full_ds = datasets.ImageFolder(str(data_dir), transform=None)
    print(
        "ImageFolder classes:", full_ds.classes
    )  # should be ['drywall', 'grass', 'metal', 'stone', 'wood']

print("Model CLASS_NAMES   :", CLASS_NAMES)

num_samples = len(full_ds)
val_size = int(VAL_SPLIT * num_samples)
train_size = num_samples - val_size

generator = torch.Generator().manual_seed(RANDOM_SEED)
train_subset, val_subset = random_split(
    full_ds, [train_size, val_size], generator=generator
)

train_ds = TransformSubset(train_subset, transform=train_transform)
val_ds = TransformSubset(val_subset, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

model = get_model(device=device, freeze_features=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# Track metrics per epoch
train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    # ---- Training ----
    model.train()
    running_loss = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
        leave=False,
    )

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_ds)
    train_losses.append(epoch_loss)

    # ---- Validation ----
    model.eval()
    correct = 0
    total = 0

    val_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
        leave=False,
    )

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total if total > 0 else 0.0
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Train Loss: {epoch_loss:.4f}, Val Acc: {val_acc * 100:.2f}%"
    )

# ---- Save model ----
torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved fine-tuned model to {MODEL_PATH}")

# ---- Plot training curve ----
epochs_range = range(1, EPOCHS + 1)
fig, ax1 = plt.subplots()

ax1.plot(epochs_range, train_losses, label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Loss")

ax2 = ax1.twinx()
ax2.plot(epochs_range, val_accuracies, linestyle="--", label="Val Accuracy")
ax2.set_ylabel("Val Accuracy")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

title_suffix = " with dark variants" if INCLUDE_DARK else ""
plt.title(
    f"Training Loss and Validation Accuracy ({AUGMENTATION} augmentation{title_suffix})"
)
plt.tight_layout()
plot_path = plots_dir / f"training_curve_{AUGMENTATION}_aug{model_suffix}.png"
plt.savefig(plot_path)
print(f"Saved training curve to {plot_path}")
plt.show()
