"""Model factory for EBHI-SEG image classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


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


def _apply_freezing(model: nn.Module, model_type: str, freeze_layers: str | None) -> None:
    if not freeze_layers or freeze_layers.lower() in ("none", "false"):
        return

    freeze_strategy = freeze_layers.lower().strip()
    if model_type in {"deeplab", "fcn"}:
        if freeze_strategy in ("backbone", "head_only"):
            for name, param in model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False
        elif freeze_strategy == "early":
            for name, param in model.named_parameters():
                if any(name.startswith(prefix) for prefix in ("conv1.", "bn1.", "layer1.", "layer2.")):
                    param.requires_grad = False
        else:
            raise ValueError(
                f"Unknown freeze_layers '{freeze_layers}'. Choose 'none', 'early', or 'backbone'."
            )
    elif model_type == "unet":
        if freeze_strategy in ("backbone", "head_only"):
            for name, param in model.named_parameters():
                if not name.startswith("classifier."):
                    param.requires_grad = False
        elif freeze_strategy == "early":
            for name, param in model.named_parameters():
                if name.startswith("features.0.") or name.startswith("features.1."):
                    param.requires_grad = False


def create_classification_model(
    model_type: str,
    n_classes: int,
    base_channels: int = 32,
    pretrained: bool = True,
    freeze_layers: str | None = None,
) -> nn.Module:
    """Create a classifier matching the requested experiment model name."""
    model_type = model_type.lower().strip()
    if model_type == "unet":
        model = ImageClassifier(n_classes=n_classes, base_channels=base_channels)
    elif model_type in {"deeplab", "fcn"}:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError(
            f"Unknown classification model type '{model_type}'. "
            "Supported types: unet, deeplab, fcn."
        )

    _apply_freezing(model, model_type, freeze_layers)
    return model