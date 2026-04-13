"""FedOptimizer: Federated Adaptive Optimization (Reddi et al., 2020)."""

from typing import Optional, Sequence

import torch

from federated.algorithms.base import FLAlgorithm


class FedOptimizer(FLAlgorithm):
    """
    Federated Optimizer using adaptive learning rates at server.

    Applies Adam/Yogi optimization at server level for aggregation,
    instead of simple averaging. This provides faster convergence
    in non-IID settings.

    Aggregation update:
    g_t = Σ_k (n_k/n_total) * [w_k(t) - w(t)]  [gradient direction]
    m_t = β₁*m_{t-1} + (1-β₁)*g_t              [exponential average]
    v_t = β₂*v_{t-1} + (1-β₂)*g_t²            [variance]
    w(t+1) = w(t) - α * m_t / (√v_t + ε)      [adaptive step]

    Args:
        model: Global model
        optimizer: 'adam' or 'yogi' (default: 'adam')
        lr: Learning rate (default: 0.001)
        beta1: Momentum parameter (default: 0.9)
        beta2: Variance parameter (default: 0.999)
        eps: Numerical stability constant (default: 1e-8)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: str = "adam",
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(
            model,
            optimizer=optimizer,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            **kwargs,
        )
        self.optimizer_type = optimizer.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Initialize momentum buffers
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0  # step counter

        # Initialize buffers for each parameter
        for name, param in model.named_parameters():
            self.m[name] = torch.zeros_like(param.data)
            self.v[name] = torch.zeros_like(param.data)

    def aggregate(
        self,
        client_updates: Sequence[tuple[dict, int]],
    ) -> None:
        """
        Aggregate using adaptive optimizer at server.

        Args:
            client_updates: List of (state_dict, num_samples) tuples
        """
        if not client_updates:
            return

        # Increment step counter
        self.t += 1

        # Calculate total number of samples
        total_samples = sum(num_samples for _, num_samples in client_updates)

        # Compute aggregated gradient direction
        global_state = self.model.state_dict()
        gradient_direction = {
            key: torch.zeros_like(param) for key, param in global_state.items()
        }

        for client_state, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in gradient_direction:
                # g_t = Σ (n_k/n_total) * (w_k - w_global)
                gradient_direction[key] += weight * (
                    client_state[key].to(gradient_direction[key].device)
                    - global_state[key].to(gradient_direction[key].device)
                )

        # Update with adaptive optimizer
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in gradient_direction:
                    continue

                g_t = gradient_direction[name]

                # Update biased first moment estimate
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g_t

                if self.optimizer_type == "adam":
                    # Update biased second raw moment estimate
                    self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (
                        g_t**2
                    )
                elif self.optimizer_type == "yogi":
                    # Yogi: v_t = v_{t-1} - (1-β₂)*sign(v_{t-1} - g_t²)*g_t²
                    self.v[name] = self.v[name] - (1 - self.beta2) * torch.sign(
                        self.v[name] - g_t**2
                    ) * (g_t**2)
                else:
                    raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

                # Bias correction
                m_hat = self.m[name] / (1 - self.beta1**self.t)
                v_hat = self.v[name] / (1 - self.beta2**self.t)

                # Update parameters
                param.data = param.data - self.lr * m_hat / (
                    torch.sqrt(v_hat) + self.eps
                )
