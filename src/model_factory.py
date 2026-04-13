"""Factory for creating segmentation models."""

import torch.nn as nn

from src.model_registry import ModelRegistry


def create_model(
    model_name: str,
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
) -> nn.Module:
    """
    Create a segmentation model by name.

    Args:
        model_name: Name of model (e.g., 'unet', 'attention_unet')
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base channel multiplier

    Returns:
        Instantiated model ready for training
    """
    model_class = ModelRegistry.get(model_name)
    return model_class(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )


def get_model_info(model_name: str) -> dict:
    """Get information about a model (e.g., number of parameters)."""
    # Create model with default params
    model = create_model(model_name)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / (1024**2),  # rough estimate in MB (float32)
    }
