"""Federated Learning algorithms."""

from federated.algorithms.fedavg import FedAvg
from federated.algorithms.fedprox import FedProx
from federated.algorithms.fedoptimizer import FedOptimizer

__all__ = ["FedAvg", "FedProx", "FedOptimizer"]
