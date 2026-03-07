"""
Federated Learning training orchestration.

Two data-partitioning strategies are supported:

  ``by_class``  (non-IID)
      Each hospital holds data from exactly one tissue-type class.
      Mirrors a realistic scenario where different clinics specialise in
      different types of pathology.  6 hospitals for 6 EBHI-SEG classes.

  ``random``  (IID)
      Training samples are randomly distributed across N hospitals.
      Useful as a baseline where data is identically distributed.
      N is controlled by ``config["n_clients"]``.

A shared validation set and a global test set are built from held-out
samples spanning all classes.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader

from federated.client import FederatedClient
from federated.server import FederatedServer
from src.dataset import CLASSES, EBHISEGDataset, get_transforms, split_samples
from src.metrics import BceDiceLoss
from src.model import UNet
from src.utils import get_device, plot_history, save_checkpoint, save_history, set_seed
from src.visualize import run_and_plot


# ---------------------------------------------------------------------------
# Data partitioning
# ---------------------------------------------------------------------------


def partition_data(
    data_root: str,
    config: dict,
) -> tuple[dict[str, EBHISEGDataset], EBHISEGDataset, EBHISEGDataset]:
    """
    Split the EBHI-SEG dataset into per-hospital training sets plus a shared
    validation set and a global test set.

    Args:
        data_root: Path to the EBHI-SEG root directory.
        config:    Configuration dict (see configs/config.yaml).

    Returns:
        (client_datasets, val_ds, test_ds)
          client_datasets: {hospital_id → EBHISEGDataset (train, augmented)}
          val_ds:          Shared validation set (all classes, no augment).
          test_ds:         Held-out test set (all classes, no augment).
    """
    seed = config.get("seed", 42)
    image_size = config.get("image_size", 256)
    val_ratio = config.get("val_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)

    train_tf = get_transforms(image_size, augment=True)
    eval_tf = get_transforms(image_size, augment=False)

    # Collect all samples (no transform; used only to enumerate paths)
    all_ds = EBHISEGDataset(data_root=data_root, transform=None)
    train_samples_global, val_samples_global, test_samples_global = split_samples(
        all_ds.samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    val_ds = EBHISEGDataset(transform=eval_tf, samples=val_samples_global)
    test_ds = EBHISEGDataset(transform=eval_tf, samples=test_samples_global)

    strategy = config.get("partition_strategy", "by_class")

    if strategy == "by_class":
        # Group training samples by class (= hospital)
        class_samples: dict[str, list] = defaultdict(list)
        for img_path, lbl_path in train_samples_global:
            # Path structure: .../EBHI-SEG/<Class>/image/<file>.png
            cls = Path(img_path).parent.parent.name
            class_samples[cls].append((img_path, lbl_path))

        client_datasets = {
            cls: EBHISEGDataset(transform=train_tf, samples=samples)
            for cls, samples in class_samples.items()
            if len(samples) > 0
        }

    elif strategy == "random":
        n_clients = config.get("n_clients", 6)
        random.seed(seed)
        shuffled = list(train_samples_global)
        random.shuffle(shuffled)

        # Split into approximately equal chunks
        chunk_size = max(1, len(shuffled) // n_clients)
        client_datasets = {}
        for i in range(n_clients):
            start = i * chunk_size
            end = start + chunk_size if i < n_clients - 1 else len(shuffled)
            chunk = shuffled[start:end]
            if chunk:
                client_datasets[f"client_{i}"] = EBHISEGDataset(
                    transform=train_tf, samples=chunk
                )

    else:
        raise ValueError(
            f"Unknown partition_strategy '{strategy}'. "
            "Choose 'by_class' or 'random'."
        )

    return client_datasets, val_ds, test_ds


# ---------------------------------------------------------------------------
# Federated training loop
# ---------------------------------------------------------------------------


def run_federated_training(config: dict) -> tuple[list[dict], dict]:
    """
    Simulate federated learning on a single machine.

    Args:
        config: Configuration dict (see configs/config.yaml).

    Returns:
        (round_history, test_metrics)
          round_history: list of per-round metric dicts.
          test_metrics:  Final evaluation on the held-out test set.
    """
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    output_dir = Path(config.get("output_dir", "outputs")) / "federated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    print("Partitioning data across virtual hospitals …")
    client_datasets, val_ds, test_ds = partition_data(config["data_root"], config)

    print(f"  Strategy : {config.get('partition_strategy', 'by_class')}")
    print(f"  Hospitals: {list(client_datasets.keys())}")
    for name, ds in client_datasets.items():
        print(f"    {name}: {len(ds)} train samples")
    print(f"  Validation : {len(val_ds)} samples")
    print(f"  Test       : {len(test_ds)} samples")

    num_workers = config.get("num_workers", 0)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("batch_size", 16),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.get("batch_size", 16),
        num_workers=num_workers,
    )

    # ---- Global model & server ----
    global_model = UNet(
        in_channels=3,
        base_channels=config.get("base_channels", 32),
        n_classes=1,
    )
    criterion = BceDiceLoss(bce_weight=config.get("bce_weight", 0.5))
    server = FederatedServer(global_model)

    # ---- Clients ----
    clients = [
        FederatedClient(
            client_id=hospital_id,
            dataset=ds,
            criterion=criterion,
            config=config,
        )
        for hospital_id, ds in client_datasets.items()
    ]

    fl_rounds = config.get("fl_rounds", 20)
    client_fraction = config.get("client_fraction", 1.0)
    print(
        f"\nStarting Federated Learning: {len(clients)} clients | "
        f"{fl_rounds} rounds | {config.get('local_epochs', 3)} local epochs/round"
    )

    best_val_dice = 0.0
    round_history: list[dict] = []

    for rnd in range(1, fl_rounds + 1):
        print(f"\n─── Round {rnd}/{fl_rounds} ───")

        # Select a fraction of clients
        n_selected = max(1, int(len(clients) * client_fraction))
        selected = random.sample(clients, n_selected)

        # Distribute current global model → local train → collect updates
        global_model_ref = server.get_global_model()
        client_updates: list[tuple[dict, int]] = []

        for client in selected:
            client.set_parameters(global_model_ref)
            stats = client.local_train(device)
            client_updates.append((client.get_parameters(), stats["n_samples"]))

        # FedAvg aggregation
        server.aggregate(client_updates)

        # Evaluate aggregated global model
        val_m = server.evaluate(val_loader, criterion, device)
        print(
            f"  Global model → "
            f"loss={val_m['loss']:.4f}  dice={val_m['dice']:.4f}  iou={val_m['iou']:.4f}"
        )

        round_history.append({"round": rnd, **val_m})

        if val_m["dice"] > best_val_dice:
            best_val_dice = val_m["dice"]
            save_checkpoint(
                server.global_model,
                None,
                rnd,
                val_m,
                output_dir / "best_model.pt",
            )
            print(f"  ✓ Best global model saved  (dice={best_val_dice:.4f})")

    # ---- Save history ----
    save_history({"rounds": round_history}, output_dir / "fl_history.json")
    _plot_fl_history(round_history, output_dir / "fl_curves.png")

    # ---- Final test evaluation ----
    print("\nLoading best global model for test evaluation …")
    from src.utils import load_checkpoint

    load_checkpoint(output_dir / "best_model.pt", server.global_model)
    test_m = server.evaluate(test_loader, criterion, device)
    print(
        f"Test → loss={test_m['loss']:.4f}  dice={test_m['dice']:.4f}  iou={test_m['iou']:.4f}"
    )

    # ---- Inference visualisation on the held-out test set ----
    run_and_plot(
        model=server.global_model,
        test_ds=test_ds,
        device=device,
        output_path=output_dir / "inference.png",
        n_samples=config.get("inference_samples", 32),
        seed=config.get("seed", 42),
        title=f"Federated ({config.get('partition_strategy', 'by_class')}) – Test Set Inference",
    )

    return round_history, test_m


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------


def _plot_fl_history(round_history: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rounds = [r["round"] for r in round_history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, label) in zip(
        axes, [("loss", "Loss"), ("dice", "Dice"), ("iou", "IoU")]
    ):
        ax.plot(
            rounds,
            [r[key] for r in round_history],
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.set_title(f"Global {label} (val)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Federated Learning – Global Model Validation", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
