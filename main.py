"""
Entry point for centralized and federated segmentation training.

Dataset:
  EBHI-SEG – Enteroscope Biopsy Histopathological Image dataset for SEGmentation
  https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2023.1114673/full
  https://figshare.com/articles/dataset/EBHI-SEG/21540159/1?file=38179080

  4,456 H&E-stained colorectal tissue patch images across 6 classes:
    Normal, Polyp, Serrated adenoma, Low-grade IN, High-grade IN, Adenocarcinoma

Usage:
  # Centralised training (all hospitals combined)
  python main.py --mode centralized

  # Federated simulation – non-IID, one class per hospital (6 hospitals)
  python main.py --mode federated

  # Federated simulation – IID, random partition into 4 hospitals
  python main.py --mode federated --partition random --n_clients 4

  # Override any config value on the command line
  python main.py --mode centralized --epochs 30 --batch_size 8

  # Use a custom config file
  python main.py --config configs/my_config.yaml --mode federated

IID = Independent and Identically Distributed
Each hospital/client has data that looks statistically the same — all classes, similar proportions. Like randomly shuffling a deck and dealing equal hands.

Non-IID = the opposite — each client's data has a different distribution. One hospital only sees colon polyps, another only sees adenocarcinomas, etc. This is realistic in medicine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary segmentation on EBHI-SEG: centralized or federated training."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=["centralized", "federated"],
        default="centralized",
        help="Training mode (default: centralized).",
    )
    # Convenience overrides matching config.yaml keys
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps | auto")
    parser.add_argument("--seed", type=int, default=None)
    # Federated-specific
    parser.add_argument("--fl_rounds", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument(
        "--partition",
        choices=["by_class", "random"],
        default=None,
        dest="partition_strategy",
        help="Data partition strategy for federated mode (default: by_class).",
    )
    parser.add_argument("--n_clients", type=int, default=None)
    parser.add_argument("--client_fraction", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load base config
    config = load_config(args.config)

    # Apply CLI overrides (skip None values)
    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("config", "mode")}
    config.update(overrides)

    print(f"Mode  : {args.mode}")
    print(f"Config: {args.config}")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print()

    if args.mode == "centralized":
        from src.train import run_centralized_training
        history, test_metrics = run_centralized_training(config)
        print("\n=== Centralized Training Complete ===")
        print(f"Test Dice : {test_metrics['dice']:.4f}")
        print(f"Test IoU  : {test_metrics['iou']:.4f}")
        print(f"Test Loss : {test_metrics['loss']:.4f}")
        print(f"Results saved to: {Path(config['output_dir']) / 'centralized'}")

    elif args.mode == "federated":
        from federated.fl_train import run_federated_training
        round_history, test_metrics = run_federated_training(config)
        print("\n=== Federated Training Complete ===")
        print(f"Test Dice : {test_metrics['dice']:.4f}")
        print(f"Test IoU  : {test_metrics['iou']:.4f}")
        print(f"Test Loss : {test_metrics['loss']:.4f}")
        print(f"Results saved to: {Path(config['output_dir']) / 'federated'}")


if __name__ == "__main__":
    main()
