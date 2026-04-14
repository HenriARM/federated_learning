"""
Model factory for creating segmentation models.

Supports:
  - UNet: Custom lightweight implementation
  - DeepLabV3: Pre-trained ResNet50 backbone
  - FCN: Classic fully-convolutional networks
"""

import torch
import torch.nn as nn

from src.models.unet import UNet


# ---------------------------------------------------------------------------
# Wrapper for models that return OrderedDict outputs
# ---------------------------------------------------------------------------

class _DeepLabOutputWrapper(nn.Module):
    """
    Wrapper to extract main output from DeepLab's OrderedDict.
    DeepLab returns {'out': tensor, 'aux': tensor}, but we only need 'out'.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, dict):
            return outputs['out']
        return outputs


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def create_model(
    model_type: str,
    in_channels: int,
    n_classes: int,
    base_channels: int = 32,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a segmentation model by name.

    Args:
        model_type: Type of model ('unet', 'deeplab', 'fcn').
        in_channels: Number of input channels (3 for RGB). Ignored for pre-trained models.
        n_classes: Number of output classes (1 for binary segmentation).
        base_channels: Base width multiplier for UNet only.
        **kwargs: Additional architecture-specific arguments.

    Returns:
        A torch.nn.Module instance ready for segmentation.

    Raises:
        ValueError: If model_type is not supported.
    """
    model_type = model_type.lower().strip()

    if model_type == "unet":
        return UNet(
            in_channels=in_channels,
            out_channels=n_classes,
            base_channels=base_channels,
        )
    
    elif model_type == "deeplab":
        # DeepLabV3 with ResNet50 backbone from torchvision
        from torchvision.models.segmentation import deeplabv3_resnet50
        
        model = deeplabv3_resnet50(
            weights=None,  # Start from scratch (no ImageNet pre-training)
            num_classes=n_classes,
        )
        
        # Ensure input compatibility (torchvision expects 3-channel RGB)
        if in_channels != 3:
            # Replace the first conv layer if needed
            model.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Wrap DeepLab to extract main output (it returns OrderedDict)
        return _DeepLabOutputWrapper(model)
    
    elif model_type == "fcn":
        # FCN (Fully Convolutional Network) with ResNet50 backbone
        from torchvision.models.segmentation import fcn_resnet50
        
        model = fcn_resnet50(
            weights=None,  # Start from scratch
            num_classes=n_classes,
        )
        
        if in_channels != 3:
            model.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Wrap FCN to extract main output (it returns OrderedDict)
        return _DeepLabOutputWrapper(model)
    
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Supported types: unet, deeplab, fcn."
        )
