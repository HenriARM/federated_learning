"""FedAvg: Federated Averaging (McMahan et al., 2017)."""

from typing import Sequence

import torch

from federated.algorithms.base import FLAlgorithm


class FedAvg(FLAlgorithm):
    """
    Federated Averaging algorithm.

    Aggregates client model updates via weighted average:
    w_global(t+1) = Σ (n_k / n_total) * w_k(t)

    where n_k is number of samples on client k.
    """

    def aggregate(
        self,
        client_updates: Sequence[tuple[dict, int]],
    ) -> None:
        """
        Aggregate client updates using weighted averaging.

        Args:
            client_updates: List of (state_dict, num_samples) tuples
        """
        if not client_updates:
            return

        # Calculate total number of samples
        total_samples = sum(num_samples for _, num_samples in client_updates)

        # Initialize aggregated weights
        global_state = self.model.state_dict()
        aggregated_state = {
            key: torch.zeros_like(param) for key, param in global_state.items()
        }

        # Weighted average
        for client_state, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in aggregated_state:
                aggregated_state[key] += weight * client_state[key].to(
                    aggregated_state[key].device
                )

        # Update global model
        self.model.load_state_dict(aggregated_state)
