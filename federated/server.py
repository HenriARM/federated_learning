"""
Federated Learning server – implements FedAvg aggregation.

FedAvg (McMahan et al., 2017):
  1. Broadcast current global weights to selected clients.
  2. Each client trains locally and sends back updated weights.
  3. Aggregate via weighted average (weight ∝ local dataset size).
  4. Repeat for R communication rounds.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from src.metrics import dice_coefficient, iou_score


class FederatedServer:
    """
    Central aggregator that maintains the global model.

    Args:
        model: Initialised (but not yet trained) global model.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.global_model = copy.deepcopy(model)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def get_global_model(self) -> torch.nn.Module:
        """Return a reference to the current global model."""
        return self.global_model

    # ------------------------------------------------------------------
    # FedAvg aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        client_updates: Sequence[tuple[dict[str, torch.Tensor], int]],
    ) -> None:
        """
        FedAvg: update the global model as a weighted average of client
        models, where the weight of each client is proportional to its
        local dataset size.

        Args:
            client_updates: Sequence of (state_dict, n_samples) pairs –
                            one per participating client.
        """
        total_samples = sum(n for _, n in client_updates)

        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key in self.global_model.state_dict().keys():
            weighted_sum = torch.stack(
                [
                    state_dict[key].float() * (n_samples / total_samples)
                    for state_dict, n_samples in client_updates
                ]
            ).sum(dim=0)
            aggregated[key] = weighted_sum

        self.global_model.load_state_dict(aggregated)

    # ------------------------------------------------------------------
    # Evaluation of the global model
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
        """
        Evaluate the global model on a DataLoader.

        Returns:
            Dict with ``loss``, ``dice``, and ``iou`` (all batch-averaged).
        """
        model = self.global_model.to(device)
        model.eval()

        total_loss = total_dice = total_iou = 0.0
        n = 0

        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            total_dice += dice_coefficient(logits, masks)
            total_iou += iou_score(logits, masks)
            n += 1

        n = max(n, 1)
        return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}
