"""
Analyze convergence smoothness across algorithms.
Smoothness = lower variance in consecutive round changes.
"""
import json
from pathlib import Path
from collections import defaultdict

exp_dir = Path('exp_v2')

# Parse all results with full history
results = defaultdict(list)

for exp_folder in sorted(exp_dir.glob('*_*_*')):
    if not exp_folder.is_dir():
        continue
    
    history_file = exp_folder / 'fl_history.json'
    if not history_file.exists():
        continue
    
    with open(history_file) as f:
        history = json.load(f)
    
    if not history.get('rounds'):
        continue
    
    # Parse folder name
    name_parts = exp_folder.name.split('_')
    model = name_parts[0].upper()
    algo = name_parts[1].upper()
    dist_str = '_'.join(name_parts[2:])
    dist_type = 'Non-IID' if 'by_class' in dist_str else 'IID'
    
    # Calculate smoothness metrics
    dices = [r['dice'] for r in history['rounds']]
    ious = [r['iou'] for r in history['rounds']]
    losses = [r['loss'] for r in history['rounds']]
    
    # Consecutive differences (variance in changes)
    dice_diffs = [abs(dices[i+1] - dices[i]) for i in range(len(dices)-1)]
    iou_diffs = [abs(ious[i+1] - ious[i]) for i in range(len(ious)-1)]
    loss_diffs = [abs(losses[i+1] - losses[i]) for i in range(len(losses)-1)]
    
    # Smoothness = lower variance in consecutive changes
    dice_smoothness = sum(dice_diffs) / len(dice_diffs) if dice_diffs else 0
    iou_smoothness = sum(iou_diffs) / len(iou_diffs) if iou_diffs else 0
    loss_smoothness = sum(loss_diffs) / len(loss_diffs) if loss_diffs else 0
    
    results[f'{model}_{dist_type}'].append({
        'algorithm': algo,
        'dice_smoothness': dice_smoothness,
        'iou_smoothness': iou_smoothness,
        'loss_smoothness': loss_smoothness,
        'max_dice_drop': max([-d for d in dice_diffs]) if dice_diffs else 0,  # biggest drop
        'final_dice': dices[-1]
    })

# Print convergence analysis
print("\n" + "="*120)
print("CONVERGENCE SMOOTHNESS ANALYSIS")
print("(Lower smoothness score = smoother convergence)")
print("="*120)

print("\nBy Model + Distribution:")
print(f"{'Model':<12} {'Distribution':<10} {'Algorithm':<12} {'Dice Smooth':<12} {'IoU Smooth':<12} {'Max Drop':<12} {'Final Dice':<12}")
print("-"*120)

for key in sorted(results.keys()):
    for r in sorted(results[key], key=lambda x: x['algorithm']):
        model_dist = key
        model, dist = model_dist.split('_')
        print(f"{model:<12} {dist:<10} {r['algorithm']:<12} {r['dice_smoothness']:.6f}      {r['iou_smoothness']:.6f}      {r['max_dice_drop']:.6f}      {r['final_dice']:.6f}")

# Group by algorithm for summary
print("\n" + "="*120)
print("ALGORITHM COMPARISON (Convergence Smoothness)")
print("="*120)

by_algo = defaultdict(lambda: {'smoothness': [], 'max_drop': [], 'final': []})

for key in results:
    for r in results[key]:
        by_algo[r['algorithm']]['smoothness'].append(r['dice_smoothness'])
        by_algo[r['algorithm']]['max_drop'].append(r['max_dice_drop'])
        by_algo[r['algorithm']]['final'].append(r['final_dice'])

print(f"\n{'Algorithm':<15} {'Avg Smoothness':<15} {'Max Drop':<15} {'Final Dice':<15}")
print("-"*60)

for algo in ['FEDAVG', 'FEDPROX', 'FEDOPTIMIZER']:
    if algo in by_algo:
        smooth = sum(by_algo[algo]['smoothness']) / len(by_algo[algo]['smoothness'])
        max_drop = sum(by_algo[algo]['max_drop']) / len(by_algo[algo]['max_drop'])
        final_dice = sum(by_algo[algo]['final']) / len(by_algo[algo]['final'])
        
        print(f"{algo:<15} {smooth:.6f}          {max_drop:.6f}          {final_dice:.6f}")

print("\n" + "="*120)
print("INTERPRETATION")
print("="*120)
print("""
- Dice Smoothness: Average absolute change between consecutive rounds
  (Lower = smoother trajectory, less oscillation)
  
- Max Drop: Largest single-round decrease in Dice
  (Lower = more stable, less convergence fluctuation)

- Final Dice: Convergence endpoint (higher is better)

Key Question: Does FedProx converge more smoothly than FedAvg/FedOptimizer?
""")

# Analysis
if 'FEDPROX' in by_algo:
    fedprox_smooth = sum(by_algo['FEDPROX']['smoothness']) / len(by_algo['FEDPROX']['smoothness'])
    fedavg_smooth = sum(by_algo['FEDAVG']['smoothness']) / len(by_algo['FEDAVG']['smoothness']) if 'FEDAVG' in by_algo else 0
    fedopt_smooth = sum(by_algo['FEDOPTIMIZER']['smoothness']) / len(by_algo['FEDOPTIMIZER']['smoothness']) if 'FEDOPTIMIZER' in by_algo else 0
    
    smoothness_rank = sorted({
        'FEDAVG': fedavg_smooth,
        'FEDPROX': fedprox_smooth,
        'FEDOPTIMIZER': fedopt_smooth
    }.items(), key=lambda x: x[1])
    
    print(f"\nSmoothness Ranking (lower is better):")
    for i, (algo, smooth) in enumerate(smoothness_rank, 1):
        print(f"  {i}. {algo}: {smooth:.6f}")
    
    if smoothness_rank[0][0] == 'FEDPROX':
        diff_to_worst = smoothness_rank[-1][1] - smoothness_rank[0][1]
        print(f"\n✓ FedProx IS smoother (by {diff_to_worst*100:.3f}% vs worst)")
    else:
        print(f"\n✗ {smoothness_rank[0][0]} is actually smoother than FedProx")

print("\n" + "="*120 + "\n")
