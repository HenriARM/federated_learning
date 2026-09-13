"""Dataset utilities for EBHI-SEG image classification."""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.dataset import CLASSES


class EBHISEGClassificationDataset(Dataset):
    """Load EBHI-SEG images and return ``(image, class_index)`` pairs."""

    def __init__(
        self,
        data_root: str | None = None,
        transform: A.Compose | None = None,
        samples: list[tuple[str, int]] | None = None,
    ) -> None:
        self.transform = transform

        if samples is not None:
            self.samples = list(samples)
        else:
            if data_root is None:
                raise ValueError("data_root is required when samples is not provided")
            self.samples = self._load_samples(Path(data_root))

    @staticmethod
    def _load_samples(data_root: Path) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for class_index, class_name in enumerate(CLASSES):
            image_dir = data_root / class_name / "image"
            if not image_dir.exists():
                continue
            samples.extend(
                (str(image_path), class_index)
                for image_path in sorted(image_dir.glob("*.png"))
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, class_index = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, class_index


def split_classification_samples(
    samples: list[tuple[str, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Split samples per class so every sufficiently large class is represented."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[str, int]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    train: list[tuple[str, int]] = []
    val: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []
    for class_samples in by_class.values():
        shuffled = list(class_samples)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        n_val = max(1, int(len(shuffled) * val_ratio))
        if len(shuffled) <= n_test + n_val:
            train.extend(shuffled)
            continue
        test.extend(shuffled[:n_test])
        val.extend(shuffled[n_test : n_test + n_val])
        train.extend(shuffled[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_classification_datasets(
    data_root: str,
    image_size: int = 224,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    EBHISEGClassificationDataset,
    EBHISEGClassificationDataset,
    EBHISEGClassificationDataset,
]:
    """Build augmented train and deterministic validation/test datasets."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    eval_transform = A.Compose(
        [A.Resize(image_size, image_size), A.Normalize(mean=mean, std=std), ToTensorV2()]
    )

    all_samples = EBHISEGClassificationDataset(data_root=data_root).samples
    train_samples, val_samples, test_samples = split_classification_samples(
        all_samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return (
        EBHISEGClassificationDataset(transform=train_transform, samples=train_samples),
        EBHISEGClassificationDataset(transform=eval_transform, samples=val_samples),
        EBHISEGClassificationDataset(transform=eval_transform, samples=test_samples),
    )