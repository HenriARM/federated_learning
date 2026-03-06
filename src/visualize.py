"""
Shared inference & visualisation helpers.

Used by both the training pipelines (called automatically at the end of
training with the held-out test_ds) and by the standalone inference.py CLI.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import EBHISEGDataset
from src.metrics import dice_coefficient, iou_score

matplotlib.use("Agg")  # headless-safe; override before calling plt if needed

# ImageNet normalisation constants – must match training transforms
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (C,H,W) tensor → (H,W,3) uint8 numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * _STD + _MEAN, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def overlay(image_uint8: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Colour overlay on the original image:
      Green  – true positive  (GT=1, pred=1)
      Red    – false negative (GT=1, pred=0)
      Yellow – false positive (GT=0, pred=1)
    """
    out = image_uint8.copy()
    tp = (gt_mask == 1) & (pred_mask == 1)
    fn = (gt_mask == 1) & (pred_mask == 0)
    fp = (gt_mask == 0) & (pred_mask == 1)

    alpha = 0.45
    colours = {
        "tp": np.array([0,   210,  90], dtype=np.uint8),
        "fn": np.array([220,  30,  30], dtype=np.uint8),
        "fp": np.array([255, 220,   0], dtype=np.uint8),
    }
    for mask, colour in [(tp, colours["tp"]), (fn, colours["fn"]), (fp, colours["fp"])]:
        out[mask] = (alpha * colour + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: EBHISEGDataset,
    device: torch.device,
    n_samples: int | None = None,
    seed: int = 42,
    threshold: float = 0.5,
) -> tuple[list, list, list, list, list]:
    """
    Run the model over *dataset* and collect per-sample results.

    Args:
        model:     Trained model (will be put in eval mode).
        dataset:   Dataset to evaluate – should be the *test* split.
        device:    torch.device.
        n_samples: If given, randomly subsample this many indices.
                   Pass ``None`` to use the entire dataset.
        seed:      Random seed used for subsampling.
        threshold: Sigmoid binarisation threshold.

    Returns:
        (images_raw, gt_masks, pred_masks, dice_scores, iou_scores)
        All items are lists of numpy arrays / floats, one per sample.
    """
    model.eval()

    indices = list(range(len(dataset)))
    if n_samples is not None and n_samples < len(indices):
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, n_samples))

    images_raw, gt_masks, pred_masks, dices, ious = [], [], [], [], []

    for idx in indices:
        image_t, mask_t = dataset[idx]                    # (3,H,W), (1,H,W)
        image_t = image_t.unsqueeze(0).to(device)         # (1,3,H,W)
        mask_t  = mask_t.unsqueeze(0).to(device)          # (1,1,H,W)

        logits = model(image_t)
        pred   = (torch.sigmoid(logits) > threshold).float()

        images_raw.append(denormalize(image_t.squeeze(0)))
        gt_masks.append(mask_t.squeeze().cpu().numpy())
        pred_masks.append(pred.squeeze().cpu().numpy())
        dices.append(dice_coefficient(logits, mask_t))
        ious.append(iou_score(logits, mask_t))

    return images_raw, gt_masks, pred_masks, dices, ious


# ---------------------------------------------------------------------------
# Plot builder
# ---------------------------------------------------------------------------

def build_plot(
    images_raw:  list,
    gt_masks:    list,
    pred_masks:  list,
    dices:       list,
    ious:        list,
    output_path: Path | str,
    cols_per_row: int = 4,
    title: str = "Inference results",
) -> None:
    """
    Save a grid figure where each sample column shows:
        Original image | Ground-truth mask | Prediction | Colour overlay

    Args:
        images_raw:   List of (H,W,3) uint8 arrays.
        gt_masks:     List of (H,W) float arrays with values in {0,1}.
        pred_masks:   List of (H,W) float arrays with values in {0,1}.
        dices:        Per-sample Dice scores.
        ious:         Per-sample IoU scores.
        output_path:  Path to save the PNG.
        cols_per_row: Number of samples to lay out per grid row.
        title:        Figure super-title.
    """
    output_path = Path(output_path)
    n = len(images_raw)
    n_cols = min(n, cols_per_row)
    n_rows_per_sample = 4                              # image / GT / pred / overlay
    n_grid_rows = int(np.ceil(n / n_cols)) * n_rows_per_sample

    fig, axes = plt.subplots(
        n_grid_rows, n_cols,
        figsize=(n_cols * 3.0, n_grid_rows * 3.0),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.axis("off")

    row_labels = ["Image", "Ground Truth", "Prediction", "Overlay"]

    for i, (img, gt, pred, dice, iou) in enumerate(
        zip(images_raw, gt_masks, pred_masks, dices, ious)
    ):
        grid_row_start = (i // n_cols) * n_rows_per_sample
        col = i % n_cols

        ov = overlay(img, pred.astype(bool), gt.astype(bool))
        panels = [img, gt, pred, ov]
        cmaps  = [None, "gray", "gray", None]

        for r_offset, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[grid_row_start + r_offset, col]
            ax.axis("on")
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1 if cmap == "gray" else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[r_offset], fontsize=9,
                              rotation=90, labelpad=4, va="center")

        axes[grid_row_start, col].set_title(
            f"#{i+1}  Dice={dice:.3f}  IoU={iou:.3f}", fontsize=8
        )

    # Legend for overlay colours
    tp_patch = mpatches.Patch(color=(0/255, 210/255, 90/255),  label="True Positive")
    fn_patch = mpatches.Patch(color=(220/255, 30/255, 30/255), label="False Negative")
    fp_patch = mpatches.Patch(color=(255/255, 220/255, 0/255), label="False Positive")

    mean_dice = float(np.mean(dices))
    mean_iou  = float(np.mean(ious))

    fig.suptitle(
        f"{title} — {n} samples   mean Dice={mean_dice:.4f}   mean IoU={mean_iou:.4f}",
        fontsize=12, y=1.002,
    )
    fig.legend(
        handles=[tp_patch, fn_patch, fp_patch],
        loc="lower center", ncol=3, fontsize=9,
        bbox_to_anchor=(0.5, -0.012),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved inference plot → {output_path}")
    print(f"  Mean Dice : {mean_dice:.4f}")
    print(f"  Mean IoU  : {mean_iou:.4f}")


# ---------------------------------------------------------------------------
# Convenience wrapper  (used by training pipelines)
# ---------------------------------------------------------------------------

def run_and_plot(
    model: torch.nn.Module,
    test_ds: EBHISEGDataset,
    device: torch.device,
    output_path: Path | str,
    n_samples: int | None = 32,
    seed: int = 42,
    threshold: float = 0.5,
    cols_per_row: int = 4,
    title: str = "Inference results",
) -> dict[str, float]:
    """
    One-call shortcut: run inference on *test_ds* and save the plot.

    Returns:
        {"dice": mean_dice, "iou": mean_iou}
    """
    print(f"\nRunning inference on {len(test_ds)} test samples …")
    images_raw, gt_masks, pred_masks, dices, ious = run_inference(
        model, test_ds, device,
        n_samples=n_samples, seed=seed, threshold=threshold,
    )
    build_plot(
        images_raw, gt_masks, pred_masks, dices, ious,
        output_path=output_path,
        cols_per_row=cols_per_row,
        title=title,
    )
    return {"dice": float(np.mean(dices)), "iou": float(np.mean(ious))}
