"""Registry for federated learning algorithms."""

from typing import Dict, Type

import torch.nn as nn

from federated.algorithms.base import FLAlgorithm


class FLAlgorithmRegistry:
    """Registry for federated learning aggregation algorithms."""

    _algorithms: Dict[str, Type[FLAlgorithm]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an FL algorithm."""

        def decorator(algo_class: Type[FLAlgorithm]) -> Type[FLAlgorithm]:
            cls._algorithms[name] = algo_class
            return algo_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[FLAlgorithm]:
        """Get algorithm class by name."""
        if name not in cls._algorithms:
            raise ValueError(
                f"Unknown algorithm: {name}. Available: {list(cls._algorithms.keys())}"
            )
        return cls._algorithms[name]

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all registered algorithm names."""
        return sorted(cls._algorithms.keys())


# Import and auto-register algorithms
from federated.algorithms.fedavg import FedAvg
from federated.algorithms.fedprox import FedProx
from federated.algorithms.fedoptimizer import FedOptimizer

FLAlgorithmRegistry.register("fedavg")(FedAvg)
FLAlgorithmRegistry.register("fedprox")(FedProx)
FLAlgorithmRegistry.register("fedoptimizer")(FedOptimizer)
