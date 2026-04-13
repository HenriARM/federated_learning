"""Base class for federated learning aggregation algorithms."""

from abc import ABC, abstractmethod
from typing import Sequence

import torch


class FLAlgorithm(ABC):
    """Abstract base class for federated learning aggregation algorithms."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        """
        Initialize FL algorithm.

        Args:
            model: Global model to be maintained by server
            **kwargs: Algorithm-specific hyperparameters
        """
        self.model = model
        self.hyperparams = kwargs

    @abstractmethod
    def aggregate(
        self,
        client_updates: Sequence[tuple[dict, int]],
    ) -> None:
        """
        Aggregate client updates into global model.

        Args:
            client_updates: List of (client_model_state_dict, num_samples) tuples
        """
        pass

    def get_global_model(self) -> torch.nn.Module:
        """Return reference to current global model."""
        return self.model

    def get_weights(self) -> dict:
        """Get current global model weights as state dict."""
        return self.model.state_dict()

    def set_weights(self, weights: dict) -> None:
        """Set global model weights from state dict."""
        self.model.load_state_dict(weights)
