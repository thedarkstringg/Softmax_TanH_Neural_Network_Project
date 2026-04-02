# Core Experiments

## Overview
`run_core_experiments.py` evaluates both **Softmax** and **Neural Network** models on three datasets: Linear Gaussian, Moons, and Digits. This is the primary baseline experiment for the project.

## Purpose
Establish baseline performance metrics across different problem complexities:
- **Linear Gaussian**: Simple linearly separable task
- **Moons**: Non-linear 2D classification task
- **Digits**: Real-world 10-class handwritten digit recognition

## Usage

### Basic Run
```bash
python run_core_experiments.py
```

### Output
- **Console Output**: Training progress and metrics printed every 10 epochs
- **Log File**: Saved to `results/logs/run_core_experiments_YYYYMMDD_HHMMSS.log`
- **Results**: Dictionary containing:
  - Train/Val/Test loss and accuracy for each model
  - Training history (metrics per epoch)
  - Best checkpoint epoch

## Architecture

### Class-Based Design
```python
from training_utils import DataUtils
from train_softmax import SoftmaxTrainer
from train_nn_runner import NNTrainer
from experiment_logger import ExperimentLogger

# Create trainers
softmax_trainer = SoftmaxTrainer(d, k, seed=seed)
nn_trainer = NNTrainer(d, h, k, seed=seed)

# Train
W, b, history, best_epoch = softmax_trainer.train(X_train, Y_train, ...)
W1, b1, W2, b2, history, best_epoch = nn_trainer.train(X_train, Y_train, ...)

# Log output
logger = ExperimentLogger(CURRENT_DIR, "run_core_experiments")
logger.run(main)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `softmax_epochs` | 100 (Linear/Moons), 200 (Digits) | Training epochs for Softmax |
| `nn_epochs` | 100 (Linear/Moons), 200 (Digits) | Training epochs for Neural Network |
| `lr` | 0.05 | Learning rate for both models |
| `batch_size` | 64 | Mini-batch size |
| `lam` | 1e-4 | L2 regularization coefficient |
| `seed` | 0 | Random seed for reproducibility |
| `checkpoint_on_val` | False (Linear/Moons), True (Digits) | Whether to use best validation checkpoint |
| `nn_hidden` | 32 | Hidden layer dimension for Neural Network |

## Expected Results

### Linear Gaussian
- Softmax: ~99% accuracy (linear model sufficient)
- NN: ~99% accuracy (extra capacity not needed)

### Moons
- Softmax: ~70% accuracy (linear model struggles)
- NN: ~95% accuracy (non-linear model required)

### Digits
- Softmax: ~95% accuracy
- NN: ~97% accuracy (better fits complex patterns)

## Data Dependencies
Requires pre-generated datasets in `data/`:
- `linear_gaussian.npz`
- `moons.npz`
- `digits_data.npz` (with `digits_split_indices.npz`)

Generate with: `scripts/generate_synthetic.py` and `scripts/make_digits_split.py`

## Reproducibility
- Fixed random seeds (seed=0) for all experiments
- Same train/val/test splits across runs
- Checkpoint on validation for Digits (best model selection)

## Integration
This script is the foundation for:
- **Ablation studies** (`run_digits_5seeds.py`, `run_digits_optimizers.py`, `run_moons_ablation.py`)
- **Analysis** (`failure_case_analysis.py`, `run_track_b.py`)
- **Visualization** (`plot_decision_boundaries.py`)
