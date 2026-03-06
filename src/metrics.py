"""
Loss functions and evaluation metrics for binary segmentation.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Metrics (operate on raw logits; sigmoid applied internally)
# ---------------------------------------------------------------------------


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Sørensen–Dice coefficient (batch mean).

    Args:
        logits:    Raw model output  (B, 1, H, W).
        targets:   Binary ground-truth masks (B, 1, H, W) with values in {0, 1}.
        threshold: Binarisation threshold applied after sigmoid.
        smooth:    Numerical stability term.

    Returns:
        Scalar float – mean Dice over the batch.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Jaccard / IoU score (batch mean).

    Args:
        logits:    Raw model output  (B, 1, H, W).
        targets:   Binary ground-truth masks (B, 1, H, W) with values in {0, 1}.
        threshold: Binarisation threshold applied after sigmoid.
        smooth:    Numerical stability term.

    Returns:
        Scalar float – mean IoU over the batch.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou.mean().item()


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class BceDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross-Entropy and Dice losses.

    BCE is computed on raw logits (numerically stable via
    ``BCEWithLogitsLoss``); Dice is computed on sigmoid probabilities.

    Args:
        bce_weight: Weight assigned to the BCE term.
                    The Dice term receives weight ``(1 - bce_weight)``.
        smooth:     Smoothing constant for the Dice term.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
