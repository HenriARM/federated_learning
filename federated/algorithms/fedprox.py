"""FedProx: Federated Proximal (Li et al., 2020)."""

from typing import Sequence

import torch

from federated.algorithms.base import FLAlgorithm


class FedProx(FLAlgorithm):
    """
    Federated Proximal algorithm.

    Adds proximal term to prevent client drift in non-IID settings:
    L_k(w) = original_loss(w) + (μ/2)||w - w_global||^2

    This encourages clients to stay close to global model while
    still optimizing for local data.

    Args:
        model: Global model
        mu: Proximal term weight (default: 0.01)
             μ=0 reduces to FedAvg
             Higher μ = stronger penalty for deviating from global model
    """

    def __init__(self, model: torch.nn.Module, mu: float = 0.01, **kwargs):
        super().__init__(model, mu=mu, **kwargs)
        self.mu = mu

    def aggregate(
        self,
        client_updates: Sequence[tuple[dict, int]],
    ) -> None:
        """
        Aggregate using weighted averaging (same as FedAvg for server).

        The proximal term is applied client-side during local training.
        Here we just do standard FedAvg aggregation.

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

    def get_proximal_loss(self, local_model: torch.nn.Module) -> torch.Tensor:
        """
        Compute proximal penalty term: (μ/2)||w_local - w_global||^2

        Used by client to regularize local training.

        Args:
            local_model: Client's local model

        Returns:
            Proximal penalty as scalar tensor
        """
        proximal_penalty = torch.tensor(
            0.0, device=next(self.model.parameters()).device
        )

        for (name, global_param), (_, local_param) in zip(
            self.model.named_parameters(),
            local_model.named_parameters(),
        ):
            proximal_penalty += (self.mu / 2.0) * torch.sum(
                (local_param - global_param) ** 2
            )

        return proximal_penalty
