"""Centralized training pipeline for EBHI-SEG classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification_dataset import build_classification_datasets
from src.classification_metrics import classification_metrics
from src.classification_model import create_classification_model
from src.dataset import CLASSES
from src.utils import get_device, load_checkpoint, save_checkpoint, save_history, set_seed


def _run_epoch(model, loader, criterion, device, n_classes, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0}

    for images, targets in tqdm(loader, desc="  train" if training else "  eval ", leave=False):
        images = images.to(device)
        targets = targets.to(device, dtype=torch.long)
        if training:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()
        batch_metrics = classification_metrics(logits.detach(), targets, n_classes)
        totals["loss"] += loss.item()
        totals["accuracy"] += batch_metrics["accuracy"]
        totals["f1_macro"] += batch_metrics["f1_macro"]

    batches = max(len(loader), 1)
    return {key: value / batches for key, value in totals.items()}


def run_centralized_classification(config: dict) -> tuple[dict, dict]:
    """Train and evaluate a centralized six-class classifier."""
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    output_dir = Path(config.get("output_dir", "outputs")) / "centralized_classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(CLASSES)

    train_ds, val_ds, test_ds = build_classification_datasets(
        data_root=config["data_root"],
        image_size=config.get("image_size", 224),
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        seed=config.get("seed", 42),
    )
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    model = create_classification_model(
        config.get("model_type", "unet"),
        n_classes,
        base_channels=config.get("base_channels", 32),
        pretrained=config.get("pretrained", True),
        freeze_layers=config.get("freeze_layers", None),
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    criterion = torch.nn.CrossEntropyLoss()
    epochs = config.get("epochs", 30)
    history = {"train": [], "val": []}
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_metrics = _run_epoch(model, train_loader, criterion, device, n_classes, optimizer)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, criterion, device, n_classes)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        print(f"  Train -> {train_metrics}")
        print(f"  Val   -> {val_metrics}")
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_checkpoint(model, optimizer, epoch, val_metrics, output_dir / "best_model.pt")

    save_history(history, output_dir / "history.json")
    load_checkpoint(output_dir / "best_model.pt", model)
    with torch.no_grad():
        test_metrics = _run_epoch(model, test_loader, criterion, device, n_classes)
    print(f"Test -> {test_metrics}")
    return history, test_metrics