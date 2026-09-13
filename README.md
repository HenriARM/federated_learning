# Federated Learning for Medical Image Segmentation

Compare federated learning algorithms (FedAvg, FedProx, FedOptimizer) and segmentation models (UNet, DeepLabV3, FCN) on the EBHI-SEG dataset.

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate federated_learning

# Download dataset to ./data/EBHI-SEG/
# https://figshare.com/articles/dataset/EBHI-SEG/21540159
```

## Quick Start

### Baseline: Centralized Training
```bash
python main.py --mode centralized --model_type unet --epochs 30
```

### Federated Training: Compare Models
```bash
python main.py --mode federated --model_type unet --fl_rounds 20
python main.py --mode federated --model_type deeplab --fl_rounds 20
python main.py --mode federated --model_type fcn --fl_rounds 20
```

### Federated Training: Compare Algorithms
```bash
python main.py --mode federated --model_type unet --fl_algorithm fedavg --fl_rounds 20
python main.py --mode federated --model_type unet --fl_algorithm fedprox --fl_rounds 20
python main.py --mode federated --model_type unet --fl_algorithm fedoptimizer --fl_rounds 20
```

### Non-IID vs IID Data Distribution
```bash
# Non-IID: Each hospital specializes in one tissue class
python main.py --mode federated --partition by_class --fl_rounds 20

# IID: Random distribution across hospitals
python main.py --mode federated --partition random --n_clients 6 --fl_rounds 20
```

## Configuration

Edit `configs/config.yaml`:
```yaml
model_type: "unet"           # unet | deeplab | fcn
fl_algorithm: "fedavg"       # fedavg | fedprox | fedoptimizer
partition_strategy: "by_class"  # by_class (non-IID) | random (IID)
fl_rounds: 20
local_epochs: 3
batch_size: 16
learning_rate: 0.001
device: "auto"               # auto | cpu | cuda | mps
```

## Results

Results saved to `outputs/{centralized,federated}/`:
- `best_model.pt` — best checkpoint
- `history.json` or `fl_history.json` — metrics per epoch/round
- Training curves and inference visualizations

## Inference
```bash
python inference.py --checkpoint outputs/federated/best_model.pt --n_samples 32
```

## Models

| Model | Params | Use Case |
|-------|--------|----------|
| UNet | ~1.9M | Lightweight, medical images |
| DeepLabV3 | ~39M | SOTA semantic segmentation |
| FCN | ~35M | Classic baseline |

## Algorithms

| Algorithm | Method | Best For |
|-----------|--------|----------|
| FedAvg | Weighted averaging | Baseline |
| FedProx | Proximal regularization | Non-IID data |
| FedOptimizer | Server-side Adam | Fast convergence |

## References

- FedAvg: [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)
- FedProx: [Li et al., 2020](https://arxiv.org/abs/1907.02745)
- FedOptimizer: [Reddi et al., 2020](https://arxiv.org/abs/2003.00295)



python main.py --task classification --mode centralized \
  --model_type deeplab --device auto

  python main.py --task classification --mode federated \
  --model_type deeplab \
  --partition by_class \
  --fl_algorithm fedavg \
  --fl_rounds 20 \
  --device auto