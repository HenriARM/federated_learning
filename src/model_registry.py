"""Model registry for dynamic model instantiation."""

from typing import Dict, Type

import torch.nn as nn


class ModelRegistry:
    """Registry for segmentation models. Enables dynamic model creation."""

    _models: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model."""

        def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """Get model class by name."""
        if name not in cls._models:
            raise ValueError(
                f"Unknown model: {name}. Available: {list(cls._models.keys())}"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names."""
        return sorted(cls._models.keys())


# Import and auto-register models
from src.models.unet import UNet

ModelRegistry.register("unet")(UNet)
