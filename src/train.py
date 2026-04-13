"""
Training / evaluation loops and centralized training orchestration.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_datasets
from src.metrics import BceDiceLoss, dice_coefficient, iou_score
from src.model import create_model
from src.utils import (
    get_device,
    load_checkpoint,
    plot_history,
    save_checkpoint,
    save_history,
    set_seed,
)
from src.visualize import run_and_plot


# ---------------------------------------------------------------------------
# Single-epoch loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Run one training epoch and return averaged metrics."""
    model.train()
    total_loss = total_dice = total_iou = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        dice = dice_coefficient(logits.detach(), masks)
        iou = iou_score(logits.detach(), masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    n = len(loader)
    return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a DataLoader and return averaged metrics."""
    model.eval()
    total_loss = total_dice = total_iou = 0.0

    for images, masks in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_coefficient(logits, masks)
        total_iou += iou_score(logits, masks)

    n = len(loader)
    return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}


# ---------------------------------------------------------------------------
# Full centralized training pipeline
# ---------------------------------------------------------------------------


def run_centralized_training(config: dict) -> tuple[dict, dict]:
    """
    Train a UNet on the full EBHI-SEG dataset (all classes combined).

    Args:
        config: Dictionary of hyperparameters (see configs/config.yaml).

    Returns:
        (history, test_metrics) where history = {"train": [...], "val": [...]}.
    """
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    output_dir = Path(config.get("output_dir", "outputs")) / "centralized"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    print("Loading dataset …")
    train_ds, val_ds, test_ds = build_datasets(
        data_root=config["data_root"],
        image_size=config.get("image_size", 256),
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        seed=config.get("seed", 42),
    )
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("batch_size", 16),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.get("batch_size", 16),
        num_workers=num_workers,
    )

    # ---- Model ----
    model = create_model(
        model_type=config.get("model_type", "unet"),
        in_channels=3,
        n_classes=1,
        base_channels=config.get("base_channels", 32),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = config.get("model_type", "unet").upper()
    print(f"  {model_name} parameters: {n_params:,}")

    # ---- Optimizer / scheduler / loss ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    epochs = config.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = BceDiceLoss(bce_weight=config.get("bce_weight", 0.5))

    # Mixed-precision scaler (CUDA only)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    # ---- Training loop ----
    best_val_dice = 0.0
    history: dict[str, list] = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{epochs}  lr={lr:.2e}")

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"  Train → loss={train_m['loss']:.4f}  dice={train_m['dice']:.4f}  iou={train_m['iou']:.4f}"
        )
        print(
            f"  Val   → loss={val_m['loss']:.4f}  dice={val_m['dice']:.4f}  iou={val_m['iou']:.4f}"
        )

        history["train"].append(train_m)
        history["val"].append(val_m)

        if val_m["dice"] > best_val_dice:
            best_val_dice = val_m["dice"]
            save_checkpoint(
                model, optimizer, epoch, val_m, output_dir / "best_model.pt"
            )
            print(f"  ✓ Best model saved  (dice={best_val_dice:.4f})")

    # ---- Save artifacts ----
    save_history(history, output_dir / "history.json")
    plot_history(history, output_dir / "training_curves.png")

    # ---- Final test evaluation ----
    print("\nLoading best model for test evaluation …")
    load_checkpoint(output_dir / "best_model.pt", model)
    test_m = evaluate(model, test_loader, criterion, device)
    print(
        f"Test → loss={test_m['loss']:.4f}  dice={test_m['dice']:.4f}  iou={test_m['iou']:.4f}"
    )

    # ---- Inference visualisation on the held-out test set ----
    run_and_plot(
        model=model,
        test_ds=test_ds,
        device=device,
        output_path=output_dir / "inference.png",
        n_samples=config.get("inference_samples", 32),
        seed=config.get("seed", 42),
        title="Centralized – Test Set Inference",
    )

    return history, test_m
