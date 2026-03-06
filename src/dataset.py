"""
EBHI-SEG Dataset loader for binary histopathological image segmentation.

Dataset paper:
  https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2023.1114673/full
  https://figshare.com/articles/dataset/EBHI-SEG/21540159

Classes (6 cancer/tissue types):
  Adenocarcinoma, High-grade IN, Low-grade IN, Normal, Polyp, Serrated adenoma

Each class directory contains:
  image/  - RGB histopathological patch images
  label/  - Binary segmentation masks (white=tissue, black=background)
"""

from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

CLASSES = [
    "Adenocarcinoma",
    "High-grade IN",
    "Low-grade IN",
    "Normal",
    "Polyp",
    "Serrated adenoma",
]


def get_transforms(image_size: int = 256, augment: bool = True) -> A.Compose:
    """
    Build an albumentations transform pipeline.

    Args:
        image_size: Output spatial resolution (square).
        augment:    Whether to apply training-time augmentation.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augment:
        return A.Compose(
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
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class EBHISEGDataset(Dataset):
    """
    PyTorch dataset for EBHI-SEG binary segmentation.

    Can be instantiated either from a data root directory (loads all matching
    image/label pairs) or from a pre-built list of ``(image_path, label_path)``
    tuples – useful when creating train/val/test splits without duplicating
    files on disk.

    Args:
        data_root:  Path to the EBHI-SEG root directory.  Required when
                    ``samples`` is not provided.
        classes:    List of class sub-directories to include.
                    Defaults to all six EBHI-SEG classes.
        transform:  Albumentations ``Compose`` pipeline applied to every sample.
        samples:    Optional pre-built list of ``(image_path, label_path)``
                    strings.  When supplied, ``data_root`` and ``classes``
                    are ignored.
    """

    def __init__(
        self,
        data_root: str | None = None,
        classes: list[str] | None = None,
        transform: A.Compose | None = None,
        samples: list[tuple[str, str]] | None = None,
    ) -> None:
        self.transform = transform

        if samples is not None:
            self.samples = list(samples)
        else:
            assert (
                data_root is not None
            ), "data_root is required when samples is not provided"
            self.data_root = Path(data_root)
            self.classes = classes if classes is not None else CLASSES
            self.samples = self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self) -> list[tuple[str, str]]:
        samples: list[tuple[str, str]] = []
        for cls in self.classes:
            img_dir = self.data_root / cls / "image"
            lbl_dir = self.data_root / cls / "label"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                lbl_path = lbl_dir / img_path.name
                if lbl_path.exists():
                    samples.append((str(img_path), str(lbl_path)))
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(lbl_path).convert("L"))

        # Binarise: any pixel > 127 → foreground (1), otherwise background (0)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # (C, H, W) float32 tensor
            mask = augmented["mask"].unsqueeze(0)  # (1, H, W) float32 tensor

        return image, mask


# ---------------------------------------------------------------------------
# Dataset splitting utilities
# ---------------------------------------------------------------------------


def split_samples(
    samples: list[tuple[str, str]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Randomly split a list of (image_path, label_path) tuples into
    train / val / test subsets.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    import random

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    if n_train <= 0:
        # Edge case: very small class – use all for training
        return shuffled, [], []

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_datasets(
    data_root: str,
    image_size: int = 256,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[EBHISEGDataset, EBHISEGDataset, EBHISEGDataset]:
    """
    Convenience function that builds train / val / test EBHISEGDataset objects
    from the EBHI-SEG root directory (all classes combined).

    Returns:
        (train_ds, val_ds, test_ds)
    """
    train_tf = get_transforms(image_size, augment=True)
    eval_tf = get_transforms(image_size, augment=False)

    # Collect ALL samples (no transform – used only to enumerate paths)
    all_ds = EBHISEGDataset(data_root=data_root, transform=None)

    train_samples, val_samples, test_samples = split_samples(
        all_ds.samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    train_ds = EBHISEGDataset(transform=train_tf, samples=train_samples)
    val_ds = EBHISEGDataset(transform=eval_tf, samples=val_samples)
    test_ds = EBHISEGDataset(transform=eval_tf, samples=test_samples)

    return train_ds, val_ds, test_ds
