"""
Generate publication-ready results table for federated learning experiments.
Organized by: Model × Algorithm × Distribution (Non-IID vs IID)
"""
import json
from pathlib import Path
from collections import defaultdict

exp_dir = Path('exp_v2')
results = []

# Parse all experiment results
for exp_folder in sorted(exp_dir.glob('*_*_*')):
    if not exp_folder.is_dir():
        continue
    
    history_file = exp_folder / 'fl_history.json'
    if not history_file.exists():
        continue
    
    with open(history_file) as f:
        history = json.load(f)
    
    if history.get('rounds'):
        final = history['rounds'][-1]
        
        # Parse folder name: model_algorithm_distribution
        name_parts = exp_folder.name.split('_')
        model = name_parts[0].upper()
        algo = name_parts[1].upper()
        dist_str = '_'.join(name_parts[2:])  # e.g., 'by_class' or 'random'
        dist_type = 'Non-IID' if 'by_class' in dist_str else 'IID'
        
        results.append({
            'model': model,
            'algorithm': algo,
            'distribution': dist_type,
            'dice': final['dice'],
            'iou': final['iou'],
            'loss': final['loss'],
            'rounds': len(history['rounds'])
        })

# ============================================================================
# TABLE 1: Complete Results (all 18 experiments)
# ============================================================================
print("\n" + "="*120)
print("TABLE 1: Federated Learning Results - Complete Summary")
print("="*120)
print(f"{'Model':<10} {'Algorithm':<12} {'Distribution':<12} {'Dice':<10} {'IoU':<10} {'Loss':<10}")
print("-"*120)

for r in sorted(results, key=lambda x: (x['model'], x['algorithm'], x['distribution'] == 'Non-IID')):
    print(f"{r['model']:<10} {r['algorithm']:<12} {r['distribution']:<12} {r['dice']:.6f}    {r['iou']:.6f}    {r['loss']:.6f}")

# ============================================================================
# TABLE 2: By Model (comparing all algorithms and distributions)
# ============================================================================
print("\n" + "="*120)
print("TABLE 2: Results by Model (ordered by Algorithm and Distribution)")
print("="*120)

for model in ['UNET', 'DEEPLAB', 'FCN']:
    model_results = [r for r in results if r['model'] == model]
    
    print(f"\n{model} (n=6 experiments):")
    print(f"  {'Algorithm':<12} {'Non-IID Dice':<14} {'Non-IID IoU':<14} {'IID Dice':<14} {'IID IoU':<14}")
    print("  " + "-"*110)
    
    # Group by algorithm
    by_algo = defaultdict(lambda: {'non_iid': None, 'iid': None})
    for r in model_results:
        key = 'non_iid' if r['distribution'] == 'Non-IID' else 'iid'
        by_algo[r['algorithm']][key] = r
    
    for algo in ['FEDAVG', 'FEDPROX', 'FEDOPTIMIZER']:
        if algo in by_algo:
            non_iid = by_algo[algo]['non_iid']
            iid = by_algo[algo]['iid']
            
            non_iid_dice = f"{non_iid['dice']:.6f}" if non_iid else "N/A"
            non_iid_iou = f"{non_iid['iou']:.6f}" if non_iid else "N/A"
            iid_dice = f"{iid['dice']:.6f}" if iid else "N/A"
            iid_iou = f"{iid['iou']:.6f}" if iid else "N/A"
            
            print(f"  {algo:<12} {non_iid_dice:<14} {non_iid_iou:<14} {iid_dice:<14} {iid_iou:<14}")

# ============================================================================
# TABLE 3: Aggregated Statistics
# ============================================================================
print("\n" + "="*120)
print("TABLE 3: Aggregated Performance Statistics")
print("="*120)

print("\nBy Model:")
print(f"  {'Model':<12} {'n':<3} {'Avg Dice':<12} {'Std Dice':<12} {'Avg IoU':<12} {'Std IoU':<12}")
print("  " + "-"*60)

for model in ['UNET', 'DEEPLAB', 'FCN']:
    model_data = [r for r in results if r['model'] == model]
    if model_data:
        dices = [r['dice'] for r in model_data]
        ious = [r['iou'] for r in model_data]
        avg_dice = sum(dices) / len(dices)
        avg_iou = sum(ious) / len(ious)
        std_dice = (sum((x - avg_dice)**2 for x in dices) / len(dices))**0.5
        std_iou = (sum((x - avg_iou)**2 for x in ious) / len(ious))**0.5
        print(f"  {model:<12} {len(model_data):<3} {avg_dice:.6f}      {std_dice:.6f}      {avg_iou:.6f}      {std_iou:.6f}")

print("\nBy Algorithm:")
print(f"  {'Algorithm':<15} {'n':<3} {'Avg Dice':<12} {'Std Dice':<12} {'Avg IoU':<12} {'Std IoU':<12}")
print("  " + "-"*70)

for algo in ['FEDAVG', 'FEDPROX', 'FEDOPTIMIZER']:
    algo_data = [r for r in results if r['algorithm'] == algo]
    if algo_data:
        dices = [r['dice'] for r in algo_data]
        ious = [r['iou'] for r in algo_data]
        avg_dice = sum(dices) / len(dices)
        avg_iou = sum(ious) / len(ious)
        std_dice = (sum((x - avg_dice)**2 for x in dices) / len(dices))**0.5
        std_iou = (sum((x - avg_iou)**2 for x in ious) / len(ious))**0.5
        print(f"  {algo:<15} {len(algo_data):<3} {avg_dice:.6f}      {std_dice:.6f}      {avg_iou:.6f}      {std_iou:.6f}")

print("\nBy Distribution:")
print(f"  {'Distribution':<15} {'n':<3} {'Avg Dice':<12} {'Std Dice':<12} {'Avg IoU':<12} {'Std IoU':<12}")
print("  " + "-"*70)

for dist in ['Non-IID', 'IID']:
    dist_data = [r for r in results if r['distribution'] == dist]
    if dist_data:
        dices = [r['dice'] for r in dist_data]
        ious = [r['iou'] for r in dist_data]
        avg_dice = sum(dices) / len(dices)
        avg_iou = sum(ious) / len(ious)
        std_dice = (sum((x - avg_dice)**2 for x in dices) / len(dices))**0.5
        std_iou = (sum((x - avg_iou)**2 for x in ious) / len(ious))**0.5
        print(f"  {dist:<15} {len(dist_data):<3} {avg_dice:.6f}      {std_dice:.6f}      {avg_iou:.6f}      {std_iou:.6f}")

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "="*120)
print("SUMMARY")
print("="*120)

all_dices = [r['dice'] for r in results]
all_ious = [r['iou'] for r in results]
avg_dice = sum(all_dices) / len(all_dices)
avg_iou = sum(all_ious) / len(all_ious)
std_dice = (sum((x - avg_dice)**2 for x in all_dices) / len(all_dices))**0.5
std_iou = (sum((x - avg_iou)**2 for x in all_ious) / len(all_ious))**0.5

print(f"\nTotal experiments: {len(results)}")
print(f"\nOverall Performance:")
print(f"  Dice: {avg_dice:.6f} ± {std_dice:.6f}")
print(f"  IoU:  {avg_iou:.6f} ± {std_iou:.6f}")

best_dice = max(results, key=lambda x: x['dice'])
best_iou = max(results, key=lambda x: x['iou'])
print(f"\nBest Performers:")
print(f"  Best Dice: {best_dice['model']} + {best_dice['algorithm']} + {best_dice['distribution']}: {best_dice['dice']:.6f}")
print(f"  Best IoU:  {best_iou['model']} + {best_iou['algorithm']} + {best_iou['distribution']}: {best_iou['iou']:.6f}")

print("\n" + "="*120 + "\n")
