"""Model factory for EBHI-SEG image classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet50


class ImageClassifier(nn.Module):
    """Lightweight convolutional classifier for the UNet comparison slot."""

    def __init__(self, n_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_channels * 2), base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_channels * 4), base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_channels * 4, n_classes)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features.flatten(1))


def create_classification_model(
    model_type: str,
    n_classes: int,
    base_channels: int = 32,
) -> nn.Module:
    """Create a classifier matching the requested experiment model name."""
    model_type = model_type.lower().strip()
    if model_type == "unet":
        return ImageClassifier(n_classes=n_classes, base_channels=base_channels)

    if model_type in {"deeplab", "fcn"}:
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model

    raise ValueError(
        f"Unknown classification model type '{model_type}'. "
        "Supported types: unet, deeplab, fcn."
    )