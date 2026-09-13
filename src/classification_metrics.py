"""Metrics for multi-class image classification."""

import torch


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
) -> dict[str, float]:
    """Return accuracy and macro F1 for a batch of class logits."""
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    f1_scores = []
    for class_index in range(n_classes):
        true_positive = ((predictions == class_index) & (targets == class_index)).sum()
        false_positive = ((predictions == class_index) & (targets != class_index)).sum()
        false_negative = ((predictions != class_index) & (targets == class_index)).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(
            (2 * true_positive.float() / denominator.float()).item()
            if denominator.item() > 0
            else 0.0
        )
    return {"accuracy": accuracy, "f1_macro": sum(f1_scores) / n_classes}