"""Factory for creating federated learning aggregation algorithms."""

import torch.nn as nn

from federated.algorithm_registry import FLAlgorithmRegistry
from federated.algorithms.base import FLAlgorithm


def create_algorithm(algorithm_name: str, model: nn.Module, **kwargs) -> FLAlgorithm:
    """
    Create a federated learning algorithm by name.

    Args:
        algorithm_name: Name of algorithm (e.g., 'fedavg', 'fedprox', 'fedoptimizer')
        model: Global model to be maintained by server
        **kwargs: Algorithm-specific hyperparameters
                 (e.g., mu=0.01 for FedProx, optimizer='adam' for FedOptimizer)

    Returns:
        Instantiated algorithm ready to aggregate client updates
    """
    algo_class = FLAlgorithmRegistry.get(algorithm_name)
    return algo_class(model, **kwargs)


def get_algorithm_info(algorithm_name: str) -> dict:
    """Get information about an algorithm."""
    algo_class = FLAlgorithmRegistry.get(algorithm_name)

    info = {
        "name": algorithm_name,
        "class": algo_class.__name__,
    }

    # Add algorithm-specific hyperparameters
    if algorithm_name == "fedprox":
        info["hyperparameters"] = {
            "mu": "Proximal penalty weight (default: 0.01)",
        }
    elif algorithm_name == "fedoptimizer":
        info["hyperparameters"] = {
            "optimizer": "adam or yogi (default: adam)",
            "lr": "Learning rate (default: 0.001)",
            "beta1": "Momentum (default: 0.9)",
            "beta2": "Variance (default: 0.999)",
        }
    else:
        info["hyperparameters"] = {}

    return info
