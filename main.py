import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

# Load pretrained MobileNetV2
model = models.mobilenet_v2(weights="DEFAULT")
print(model)
model.eval()

# Define transform
transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load image
img = Image.open("crying_cat.jpg").convert("RGB")
x = transform(img).unsqueeze(0)

# Move to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
x = x.to(device)

# Forward pass
with torch.no_grad():
    y = model(x)

import torch.nn.functional as F

probs = F.softmax(y, dim=1)
top5 = torch.topk(probs, k=5)

from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

weights = MobileNet_V2_Weights.DEFAULT
categories = weights.meta["categories"]

idx = top5.indices[0][0].item()
label = categories[idx]
confidence = top5.values[0][0].item()

print(f"Predicted: {label} ({confidence * 100:.1f}%)")
