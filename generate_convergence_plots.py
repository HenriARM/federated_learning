"""
Generate convergence plots from fl_history.json files for paper figures.
Creates comprehensive plots comparing algorithms and distributions.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
EXP_DIR = Path("exp_v2")
OUTPUT_DIR = Path(".")

# Color scheme for algorithms
ALGO_COLORS = {
    "fedavg": "#1f77b4",      # blue
    "fedprox": "#ff7f0e",     # orange
    "fedoptimizer": "#2ca02c"  # green
}

ALGO_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedoptimizer": "FedOptimizer"
}

DIST_LABELS = {
    "by_class": "Non-IID (by-class)",
    "random": "IID (random)"
}

def extract_experiment_info(dirname):
    """Extract model, algorithm, and distribution from directory name."""
    # Format: {model}_{algorithm}_{distribution}
    # Example: unet_fedavg_by_class
    parts = dirname.lower().split("_")
    if len(parts) >= 3:
        model = parts[0]
        algorithm = parts[1]
        distribution = "_".join(parts[2:])  # Handle "by_class"
        return model, algorithm, distribution
    return None, None, None

def load_history(json_path):
    """Load fl_history.json file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def plot_algorithm_comparison_by_distribution(experiments_data, distribution):
    """Plot all algorithms for a single distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Algorithm Comparison - {DIST_LABELS[distribution]}", fontsize=14, fontweight='bold')
    
    metrics = ["dice", "iou", "loss"]
    metric_titles = ["Dice Coefficient", "IoU", "Loss"]
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        
        algorithms_found = set()
        for (model, algo, dist), history in experiments_data.items():
            if dist == distribution and history:
                algo_lower = algo.lower()
                if algo_lower not in algorithms_found:
                    # Extract metric values from rounds structure
                    try:
                        rounds_data = history.get("rounds", [])
                        values = [round_data.get(metric) for round_data in rounds_data if metric in round_data]
                        
                        if values:
                            rounds = range(1, len(values) + 1)
                            ax.plot(rounds, values, 
                                   color=ALGO_COLORS.get(algo_lower, "#000000"),
                                   label=ALGO_LABELS.get(algo_lower, algo),
                                   linewidth=2, marker='o', markersize=5)
                            algorithms_found.add(algo_lower)
                    except Exception as e:
                        print(f"Error plotting {metric} for {algo}: {e}")
        
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 2:  # Legend on last subplot
            ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_all_algorithms_non_iid():
    """Generate main figure: all algorithms under Non-IID (most challenging)."""
    print("📊 Generating Non-IID algorithm comparison plot...")
    
    experiments_data = load_all_experiments()
    fig = plot_algorithm_comparison_by_distribution(experiments_data, "by_class")
    output_path = OUTPUT_DIR / "fl_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close(fig)

def plot_distribution_comparison():
    """Plot IID vs Non-IID comparison for each algorithm."""
    print("📊 Generating IID vs Non-IID comparison plots...")
    
    experiments_data = load_all_experiments()
    
    # Group by algorithm
    algo_data = defaultdict(lambda: {"by_class": {}, "random": {}})
    
    for (model, algo, dist), history in experiments_data.items():
        if history:
            algo_lower = algo.lower()
            if dist in algo_data[algo_lower]:
                # Store UNet results (for consistency)
                if model.lower() == "unet":
                    algo_data[algo_lower][dist] = history
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Distribution Impact on Convergence (All Algorithms - UNet)", 
                fontsize=14, fontweight='bold')
    
    metrics = ["dice", "iou", "loss"]
    metric_titles = ["Dice Coefficient", "IoU", "Loss"]
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        
        for algo in ["fedavg", "fedprox", "fedoptimizer"]:
            for dist, dist_label, linestyle in [("by_class", "Non-IID", "-"), 
                                                 ("random", "IID", "--")]:
                history = algo_data[algo][dist]
                if history:
                    try:
                        rounds_data = history.get("rounds", [])
                        values = [round_data.get(metric) for round_data in rounds_data if metric in round_data]
                        
                        if values:
                            rounds = range(1, len(values) + 1)
                            ax.plot(rounds, values, 
                                   color=ALGO_COLORS.get(algo, "#000000"),
                                   linestyle=linestyle,
                                   label=f"{ALGO_LABELS.get(algo, algo)} ({DIST_LABELS[dist]})",
                                   linewidth=2, marker='o', markersize=4)
                    except Exception as e:
                        print(f"Error plotting {metric} for {algo}: {e}")
        
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.legend(loc='best', fontsize=9, ncol=2)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "iid_vs_non_iid.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close(fig)

def plot_model_comparison(distribution, output_filename, dist_label):
    """Compare models for specified distribution (IID or Non-IID)."""
    print(f"📊 Generating model comparison plots ({dist_label})...")
    
    experiments_data = load_all_experiments()
    
    # Group by algorithm and model
    algo_model_data = defaultdict(lambda: defaultdict(dict))
    
    for (model, algo, dist), history in experiments_data.items():
        if history and dist == distribution:
            algo_lower = algo.lower()
            model_lower = model.lower()
            algo_model_data[algo_lower][model_lower] = history
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Model Comparison - {dist_label} Data Distribution", 
                fontsize=14, fontweight='bold')
    
    metrics = ["dice", "iou", "loss"]
    metric_titles = ["Dice Coefficient", "IoU", "Loss"]
    
    model_colors = {"unet": "#1f77b4", "deeplab": "#ff7f0e", "fcn": "#2ca02c"}
    model_labels = {"unet": "UNet", "deeplab": "DeepLabV3", "fcn": "FCN"}
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        
        for algo in ["fedavg", "fedprox", "fedoptimizer"]:
            for model in ["unet", "deeplab", "fcn"]:
                history = algo_model_data[algo][model]
                if history:
                    try:
                        rounds_data = history.get("rounds", [])
                        values = [round_data.get(metric) for round_data in rounds_data if metric in round_data]
                        
                        if values:
                            rounds = range(1, len(values) + 1)
                            ax.plot(rounds, values, 
                                   color=model_colors.get(model, "#000000"),
                                   label=f"{ALGO_LABELS.get(algo, algo)} - {model_labels[model]}",
                                   linewidth=2, marker='o', markersize=4)
                    except Exception as e:
                        print(f"Error plotting {metric}: {e}")
        
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close(fig)

def load_all_experiments():
    """Load all experiments from exp_v2 folder."""
    experiments_data = {}
    
    if not EXP_DIR.exists():
        print(f"❌ Directory not found: {EXP_DIR}")
        return experiments_data
    
    for exp_folder in sorted(EXP_DIR.iterdir()):
        if exp_folder.is_dir():
            model, algo, dist = extract_experiment_info(exp_folder.name)
            if model and algo and dist:
                history_path = exp_folder / "fl_history.json"
                if history_path.exists():
                    history = load_history(history_path)
                    if history:
                        experiments_data[(model, algo, dist)] = history
                        print(f"✅ Loaded: {exp_folder.name}")
    
    return experiments_data

if __name__ == "__main__":
    print("🚀 Generating convergence plots for paper figures...\n")
    
    # Non-IID and IID model comparison plots
    plot_model_comparison("by_class", "model_comparison_non_iid.png", "Non-IID")
    plot_model_comparison("random", "model_comparison_iid.png", "IID")
    
    print("\n✅ All plots generated successfully!")
    print(f"📁 Check the output folder for PNG files ready for Overleaf")
