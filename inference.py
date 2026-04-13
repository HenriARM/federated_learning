"""
Inference & visualisation script for the trained binary segmentation model.

Loads the best checkpoint, runs it on the test split (or a custom image
directory), and saves one large figure showing every sample as a row of:

    Original image  |  Ground-truth mask  |  Predicted mask  |  Overlay

Usage examples
--------------
# Default: use centralized best model on the test split from the config
python inference.py

# Use the federated best model
python inference.py --checkpoint outputs/federated/best_model.pt

# Limit to N random samples (avoids a massive plot for large test sets)
python inference.py --n_samples 24

# Custom image/label directory (ignores dataset split, loads directly)
python inference.py --image_dir data/EBHI-SEG/Adenocarcinoma/image \
                    --label_dir data/EBHI-SEG/Adenocarcinoma/label

# Save to a specific path
python inference.py --output inference_results.png
e.x.
python inference.py --n_samples 32 --cols 4 2>&1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.dataset import EBHISEGDataset, build_datasets, get_transforms
from src.metrics import dice_coefficient, iou_score
from src.model import create_model
from src.utils import get_device, load_checkpoint, set_seed

matplotlib.use("Agg")   # headless-safe; switch to "TkAgg" / "MacOSX" for interactive

# ImageNet normalisation constants (must match training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (C,H,W) tensor back to a (H,W,3) uint8 array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
    img = np.clip(img * _STD + _MEAN, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def overlay(image_uint8: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Draw a colour overlay on top of the original image:
      - Green  : true positive  (both GT and pred = 1)
      - Red    : false negative (GT=1, pred=0)
      - Yellow : false positive (GT=0, pred=1)
    """
    out = image_uint8.copy()
    tp = (gt_mask == 1) & (pred_mask == 1)
    fn = (gt_mask == 1) & (pred_mask == 0)
    fp = (gt_mask == 0) & (pred_mask == 1)

    alpha = 0.45
    green  = np.array([0,   210, 90],  dtype=np.uint8)
    red    = np.array([220, 30,  30],  dtype=np.uint8)
    yellow = np.array([255, 220, 0],   dtype=np.uint8)

    for mask, colour in [(tp, green), (fn, red), (fp, yellow)]:
        out[mask] = (alpha * colour + (1 - alpha) * out[mask]).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: EBHISEGDataset,
    device: torch.device,
    n_samples: int | None,
    seed: int,
    threshold: float = 0.5,
) -> tuple[list, list, list, list, list]:
    """
    Run inference on *dataset* and return lists of:
        images_raw, gt_masks, pred_masks, dice_scores, iou_scores
    (all as numpy arrays / float scalars, already subsampled to n_samples).
    """
    model.eval()

    indices = list(range(len(dataset)))
    if n_samples is not None and n_samples < len(indices):
        random.seed(seed)
        indices = random.sample(indices, n_samples)
        indices.sort()   # keep original order in the dataset

    images_raw, gt_masks, pred_masks, dices, ious = [], [], [], [], []

    for idx in indices:
        image_t, mask_t = dataset[idx]          # (3,H,W), (1,H,W)
        image_t = image_t.unsqueeze(0).to(device)   # (1,3,H,W)
        mask_t  = mask_t.unsqueeze(0).to(device)    # (1,1,H,W)

        logits = model(image_t)
        pred   = (torch.sigmoid(logits) > threshold).float()

        dice = dice_coefficient(logits, mask_t)
        iou  = iou_score(logits, mask_t)

        images_raw.append(denormalize(image_t.squeeze(0)))
        gt_masks.append(mask_t.squeeze().cpu().numpy())
        pred_masks.append(pred.squeeze().cpu().numpy())
        dices.append(dice)
        ious.append(iou)

    return images_raw, gt_masks, pred_masks, dices, ious


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def build_plot(
    images_raw: list,
    gt_masks:   list,
    pred_masks: list,
    dices:      list,
    ious:       list,
    output_path: Path,
    cols_per_row: int = 4,   # number of samples per grid row
) -> None:
    """
    Lay out samples in a grid.  Every sample occupies one column of 4 subplots:
        image | GT mask | prediction | overlay
    """
    n = len(images_raw)
    n_cols = min(n, cols_per_row)          # samples per grid row
    n_rows_per_sample = 4                  # image / GT / pred / overlay
    n_grid_rows = int(np.ceil(n / n_cols)) * n_rows_per_sample

    fig_w = n_cols * 3.0
    fig_h = n_grid_rows * 3.0
    fig, axes = plt.subplots(n_grid_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    # Turn all axes off first; we enable only what we use
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

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(row_labels[r_offset], fontsize=9, rotation=90,
                              labelpad=4, va="center")

        # Per-sample title on the Image row
        axes[grid_row_start, col].set_title(
            f"#{i+1}  Dice={dice:.3f}  IoU={iou:.3f}", fontsize=8
        )

    # Overlay legend
    tp_patch = mpatches.Patch(color=(0/255, 210/255, 90/255),  label="True Positive")
    fn_patch = mpatches.Patch(color=(220/255, 30/255, 30/255), label="False Negative")
    fp_patch = mpatches.Patch(color=(255/255, 220/255, 0/255), label="False Positive")

    mean_dice = float(np.mean(dices))
    mean_iou  = float(np.mean(ious))
    fig.suptitle(
        f"Inference results — {n} test samples   "
        f"mean Dice={mean_dice:.4f}   mean IoU={mean_iou:.4f}",
        fontsize=12, y=1.002,
    )
    fig.legend(handles=[tp_patch, fn_patch, fp_patch],
               loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.012))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot → {output_path}  ({n} samples)")
    print(f"Mean Dice : {mean_dice:.4f}")
    print(f"Mean IoU  : {mean_iou:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference & visualisation for EBHI-SEG segmentation.")
    p.add_argument("--config",      default="configs/config.yaml",
                   help="Config file (default: configs/config.yaml).")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to .pt checkpoint. Defaults to outputs/<mode>/best_model.pt.")
    p.add_argument("--mode",        choices=["centralized", "federated"],
                   default="centralized",
                   help="Which training mode's checkpoint to use (default: centralized).")
    p.add_argument("--n_samples",   type=int, default=32,
                   help="Max test samples to visualise (default: 32). Use 0 for all.")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Sigmoid threshold for binarising predictions (default: 0.5).")
    p.add_argument("--cols",        type=int, default=4,
                   help="Samples per grid row (default: 4).")
    p.add_argument("--output",      default=None,
                   help="Output PNG path. Defaults to outputs/<mode>/inference.png.")
    # Custom directory mode (skips dataset split)
    p.add_argument("--image_dir",   default=None,
                   help="Folder of .png images to run inference on directly.")
    p.add_argument("--label_dir",   default=None,
                   help="Folder of corresponding .png label masks (optional).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device     = get_device(config.get("device", "auto"))
    image_size = config.get("image_size", 224)

    # ---- Checkpoint path ----
    checkpoint = args.checkpoint or (
        Path(config.get("output_dir", "outputs")) / args.mode / "best_model.pt"
    )
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # ---- Output path ----
    output_path = Path(args.output) if args.output else (
        Path(config.get("output_dir", "outputs")) / args.mode / "inference.png"
    )

    # ---- Model ----
    model = create_model(
        model_type=config.get("model_type", "unet"),
        in_channels=3,
        n_classes=1,
        base_channels=config.get("base_channels", 32),
    ).to(device)
    epoch, metrics = load_checkpoint(checkpoint, model)
    print(f"Loaded checkpoint from {checkpoint}  (epoch {epoch}, "
          f"val dice={metrics.get('dice', '?'):.4f})")

    # ---- Dataset ----
    if args.image_dir:
        # Custom directory mode — load all paired images directly
        img_dir = Path(args.image_dir)
        lbl_dir = Path(args.label_dir) if args.label_dir else None
        eval_tf = get_transforms(image_size, augment=False)
        samples = []
        for img_path in sorted(img_dir.glob("*.png")):
            lbl_path = (lbl_dir / img_path.name) if lbl_dir else img_path
            samples.append((str(img_path), str(lbl_path)))
        test_ds = EBHISEGDataset(transform=eval_tf, samples=samples)
        print(f"Custom directory: {len(test_ds)} images from {img_dir}")
    else:
        # Use the same test split that was held out during training
        _, _, test_ds = build_datasets(
            data_root=config["data_root"],
            image_size=image_size,
            val_ratio=config.get("val_ratio", 0.15),
            test_ratio=config.get("test_ratio", 0.15),
            seed=config.get("seed", 42),
        )
        print(f"Test split: {len(test_ds)} samples")

    n_samples = None if args.n_samples == 0 else args.n_samples

    # ---- Run ----
    images_raw, gt_masks, pred_masks, dices, ious = run_inference(
        model, test_ds, device,
        n_samples=n_samples,
        seed=config.get("seed", 42),
        threshold=args.threshold,
    )

    # ---- Plot ----
    build_plot(
        images_raw, gt_masks, pred_masks, dices, ious,
        output_path=output_path,
        cols_per_row=args.cols,
    )


if __name__ == "__main__":
    main()
