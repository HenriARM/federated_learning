"""Federated learning pipeline for EBHI-SEG image classification."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.classification_dataset import (
    EBHISEGClassificationDataset,
    build_classification_datasets,
)
from src.classification_metrics import classification_metrics
from src.classification_model import create_classification_model
from src.dataset import CLASSES
from src.utils import get_device, load_checkpoint, save_checkpoint, save_history, set_seed


def _partition_classification_data(data_root: str, config: dict):
    full_train, val_ds, test_ds = build_classification_datasets(
        data_root=data_root,
        image_size=config.get("image_size", 224),
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        seed=config.get("seed", 42),
    )
    train_samples = full_train.samples
    strategy = config.get("partition_strategy", "random")
    if strategy == "by_class":
        grouped = {class_index: [] for class_index in range(len(CLASSES))}
        for sample in train_samples:
            grouped[sample[1]].append(sample)
        client_samples = {
            CLASSES[class_index]: samples
            for class_index, samples in grouped.items()
            if samples
        }
    elif strategy == "random":
        rng = random.Random(config.get("seed", 42))
        shuffled = list(train_samples)
        rng.shuffle(shuffled)
        n_clients = config.get("n_clients", 6)
        client_samples = {
            f"client_{index}": shuffled[index::n_clients]
            for index in range(n_clients)
        }
        client_samples = {name: samples for name, samples in client_samples.items() if samples}
    else:
        raise ValueError(f"Unknown partition_strategy '{strategy}'")

    train_transform = full_train.transform
    clients = {
        name: EBHISEGClassificationDataset(transform=train_transform, samples=samples)
        for name, samples in client_samples.items()
    }
    return clients, val_ds, test_ds


def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_total = 0.0
    metric_total = {"accuracy": 0.0, "f1_macro": 0.0}
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device, dtype=torch.long)
            logits = model(images)
            loss_total += criterion(logits, targets).item()
            metrics = classification_metrics(logits, targets, len(CLASSES))
            for key in metric_total:
                metric_total[key] += metrics[key]
    batches = max(len(loader), 1)
    return {
        "loss": loss_total / batches,
        **{key: value / batches for key, value in metric_total.items()},
    }


def _local_update(global_model, dataset, config, device):
    model = copy.deepcopy(global_model).to(device)
    model.train()
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(config.get("local_epochs", 3)):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device, dtype=torch.long)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
    return model.state_dict(), len(dataset)


def _fedavg(model, updates):
    total_samples = sum(sample_count for _, sample_count in updates)
    global_state = model.state_dict()
    aggregated = {}
    for key, value in global_state.items():
        if torch.is_floating_point(value):
            aggregated[key] = sum(
                client_state[key].to(value.device) * (sample_count / total_samples)
                for client_state, sample_count in updates
            )
        else:
            aggregated[key] = updates[0][0][key].to(value.device)
    model.load_state_dict(aggregated)


def run_federated_classification(config: dict) -> tuple[list[dict], dict]:
    """Run weighted FedAvg for the classification task."""
    algorithm = config.get("fl_algorithm", "fedavg")
    if algorithm != "fedavg":
        raise NotImplementedError(
            f"Classification currently supports only fedavg, got '{algorithm}'."
        )
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    output_dir = Path(config.get("output_dir", "outputs")) / "federated_classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    clients, val_ds, test_ds = _partition_classification_data(config["data_root"], config)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 16))
    test_loader = DataLoader(test_ds, batch_size=config.get("batch_size", 16))
    model = create_classification_model(
        config.get("model_type", "unet"), len(CLASSES), config.get("base_channels", 32)
    )
    criterion = torch.nn.CrossEntropyLoss()
    history: list[dict] = []
    best_f1 = -1.0

    for round_index in range(1, config.get("fl_rounds", 20) + 1):
        client_names = list(clients)
        selected_count = max(1, int(len(client_names) * config.get("client_fraction", 1.0)))
        selected = random.sample(client_names, selected_count)
        global_model = model.to(device)
        updates = [_local_update(global_model, clients[name], config, device) for name in selected]
        _fedavg(global_model, updates)
        metrics = _evaluate(global_model, val_loader, criterion, device)
        history.append({"round": round_index, **metrics})
        print(f"Round {round_index}: {metrics}")
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            save_checkpoint(global_model, None, round_index, metrics, output_dir / "best_model.pt")

    save_history({"rounds": history}, output_dir / "fl_history.json")
    load_checkpoint(output_dir / "best_model.pt", model)
    test_metrics = _evaluate(model, test_loader, criterion, device)
    print(f"Test: {test_metrics}")
    return history, test_metrics