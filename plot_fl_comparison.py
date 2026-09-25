"""Generate comparison plots (Accuracy, F1-macro, Loss) between IID and Non-IID FL experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_history(history_path: str | Path) -> list[dict]:
    with open(history_path, "r") as f:
        data = json.load(f)
    return data.get("rounds", data)


def plot_comparison(iid_path: str, non_iid_path: str, output_path: str) -> None:
    iid_rounds = load_history(iid_path)
    non_iid_rounds = load_history(non_iid_path)

    rounds_iid = [r["round"] for r in iid_rounds]
    rounds_non_iid = [r["round"] for r in non_iid_rounds]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    metrics = [
        ("accuracy", "Accuracy", "Validation Accuracy"),
        ("f1_macro", "Macro F1-Score", "Validation Macro F1"),
        ("loss", "Loss", "Validation Loss"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        ax.plot(
            rounds_iid,
            [r[key] for r in iid_rounds],
            marker="o",
            label="IID (Random Partition)",
            linewidth=2,
            color="#1f77b4",
        )
        ax.plot(
            rounds_non_iid,
            [r[key] for r in non_iid_rounds],
            marker="s",
            label="Non-IID (By-Class Partition)",
            linewidth=2,
            color="#d62728",
        )
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Federated Learning on EBHI-SEG Classification: IID vs Non-IID (FedAvg)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot successfully saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot IID vs Non-IID FL Comparison")
    parser.add_argument(
        "--iid_history",
        default="outputs/classification_iid_fedavg/fl_history.json",
        help="Path to IID fl_history.json",
    )
    parser.add_argument(
        "--non_iid_history",
        default="outputs/classification_noniid_fedavg/fl_history.json",
        help="Path to Non-IID fl_history.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/iid_vs_non_iid_comparison.png",
        help="Path to save the generated comparison plot",
    )
    args = parser.parse_args()

    plot_comparison(args.iid_history, args.non_iid_history, args.output)
