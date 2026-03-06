"""
Shared utility functions: device selection, checkpointing, seeding,
history serialisation, and visualisation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device(preferred: str = "auto") -> torch.device:
    """
    Return the best available torch.device.

    Priority:  CUDA → MPS (Apple Silicon) → CPU.
    If ``preferred`` is set to a specific device string (e.g. "cpu"),
    that device is returned directly.
    """
    if preferred and preferred != "auto":
        dev = torch.device(preferred)
        print(f"Using device: {dev}")
        return dev

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    print(f"Using device: {dev}")
    return dev


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    path: str | Path,
) -> None:
    """Save model (and optionally optimizer) state to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict]:
    """
    Load checkpoint from disk into *model* (and optionally *optimizer*).

    Returns:
        (epoch, metrics) from the checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------


def save_history(history: dict, path: str | Path) -> None:
    """Serialise training history dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_history(
    history: dict[str, list[dict]],
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot train / val loss, Dice and IoU curves.

    Args:
        history:     Dict with keys ``"train"`` and ``"val"``, each a list of
                     per-epoch metric dicts.
        output_path: If given, the figure is saved to this path.

    Returns:
        The matplotlib Figure object.
    """
    train_m = history.get("train", [])
    val_m = history.get("val", [])
    epochs = range(1, len(train_m) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("loss", "Loss"), ("dice", "Dice"), ("iou", "IoU")]

    for ax, (key, label) in zip(axes, metrics):
        ax.plot(epochs, [m[key] for m in train_m], label="Train", linewidth=1.5)
        ax.plot(epochs, [m[key] for m in val_m], label="Val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training History", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    n: int = 4,
    threshold: float = 0.5,
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """
    Show a grid of (image | ground-truth mask | predicted mask) triples.

    Args:
        images:      (B, 3, H, W) normalised image tensor.
        masks:       (B, 1, H, W) binary ground-truth tensor.
        logits:      (B, 1, H, W) raw model output tensor.
        n:           Number of samples to display.
        threshold:   Binarisation threshold for predictions.
        output_path: Optional path to save the figure.
    """
    n = min(n, images.size(0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    preds = (torch.sigmoid(logits) > threshold).cpu().numpy()

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Image", "Ground Truth", "Prediction"]
    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(masks[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)

        for j, title in enumerate(titles):
            axes[i, j].set_title(title if i == 0 else "")
            axes[i, j].axis("off")

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
