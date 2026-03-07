"""
Federated Learning client – simulates a single virtual hospital.

Each client:
  1. Receives the current global model weights from the server.
  2. Fine-tunes the model on its local dataset for ``local_epochs`` epochs.
  3. Returns the updated weights and the number of training samples.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class FederatedClient:
    """
    Virtual hospital node that performs local training.

    Args:
        client_id:  Human-readable identifier (e.g. the tissue class name).
        dataset:    Local training dataset for this client.
        criterion:  Loss function (shared with the server).
        config:     Training hyper-parameters dict – must contain at least
                    ``batch_size``, ``learning_rate``, and ``local_epochs``.
    """

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        criterion: torch.nn.Module,
        config: dict[str, Any],
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.criterion = criterion
        self.config = config
        # Model placeholder – filled via set_parameters() before training
        self._model: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # Parameter exchange
    # ------------------------------------------------------------------

    def set_parameters(self, model: torch.nn.Module) -> None:
        """Clone the global model into the client's local copy."""
        self._model = copy.deepcopy(model)

    def get_parameters(self) -> dict[str, torch.Tensor]:
        """Return a deep-copy of the local model's state dict."""
        assert self._model is not None, "Call set_parameters() before get_parameters()"
        return copy.deepcopy(self._model.state_dict())

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def local_train(self, device: torch.device) -> dict[str, Any]:
        """
        Run ``local_epochs`` epochs of SGD/Adam on the local dataset.

        Args:
            device: torch.device to move tensors to.

        Returns:
            Dict with ``loss`` (float) and ``n_samples`` (int).
        """
        assert self._model is not None, "Call set_parameters() before local_train()"
        model = self._model.to(device)
        model.train()

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=(device.type == "cuda"),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

        total_loss = 0.0
        n_batches = 0
        local_epochs = self.config.get("local_epochs", 3)

        for _ in range(local_epochs):
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        n_samples = len(self.dataset)
        print(
            f"    [{self.client_id}]  avg_loss={avg_loss:.4f}  "
            f"n_samples={n_samples}  local_epochs={local_epochs}"
        )
        return {"loss": avg_loss, "n_samples": n_samples}
