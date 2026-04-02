# Ablation Studies

Three complementary ablations exploring model capacity, optimizer behavior, and failure cases.

## 1. Capacity Ablation on Moons (`run_moons_ablation.py`)

### Purpose
Investigate whether increasing hidden layer width improves performance on the Moons task.

### What It Does
- Trains Neural Networks with h ∈ {2, 8, 32}
- Computes metrics and generates decision boundary plots
- Creates comparison plot showing all three models side-by-side

### Usage
```bash
python run_moons_ablation.py
```

### Output
- **Individual plots**: `decision_boundary_moons_ablation_h_2.png`, `h_8.png`, `h_32.png`
- **Comparison plot**: `moons_ablation_comparison.png`
- **Console metrics**: Train/Val/Test loss and accuracy for each width
- **Log file**: `results/logs/run_moons_ablation_YYYYMMDD_HHMMSS.log`

### Key Findings
| Hidden Width | Test Accuracy | Test Loss | Overfitting |
|--------------|---------------|-----------|------------|
| h=2          | ~85%          | High      | Low        |
| h=8          | ~85%          | Lower     | Minimal    |
| h=32         | ~85%          | Higher    | **YES**    |

**Conclusion**: No capacity benefit beyond h=8. Moons is fundamentally simple; extra parameters cause overfitting with same or worse generalization.

### Class-Based Implementation
```python
from train_nn_runner import NNTrainer

for h in [2, 8, 32]:
    trainer = NNTrainer(d, h, k, seed=0)
    W1, b1, W2, b2, history, _ = trainer.train(
        X_train, Y_train, X_val, Y_val,
        epochs=500, lr=0.1, batch_size=64, lam=1e-4,
        checkpoint_on_val=False
    )
    metrics = trainer.evaluate(X_test, Y_test, lam)
```

---

## 2. Optimizer Study on Digits (`run_digits_optimizers.py`)

### Purpose
Compare three optimizers (SGD, Momentum, Adam) on the same fair setup with controlled learning rates.

### What It Does
- Trains with SGD (lr=0.05), Momentum (lr=0.05), Adam (lr=0.001)
- Tracks training/validation metrics every 10 epochs
- Selects best checkpoint based on validation loss
- Generates training curves

### Usage
```bash
python run_digits_optimizers.py
```

### Output
- **Training curves**: `decision_boundary_digits_optimizer_*.png` (if using plot functions)
- **Console metrics**: Final accuracy/loss for each optimizer
- **Log file**: `results/logs/run_digits_optimizers_YYYYMMDD_HHMMSS.log`

### Key Results
| Optimizer | Test Accuracy | Best Epoch | Stability | Note |
|-----------|---------------|-----------|-----------|------|
| SGD       | 95.11%        | Stable    | ✓ Consistent | Baseline |
| Momentum  | 94.57%        | Stable    | ✓ Smooth | Slight oscillation |
| Adam      | 96.20%        | Epoch 67  | ✗ **Unstable** | Loss spikes at epochs 30, 80, 130 |

**Key Finding**: Adam achieves highest accuracy but suffers from instability due to lr=0.05 being too aggressive for adaptive learning rates. See `failure_case_analysis.py` for root cause.

### Implementation
```python
from train_nn_runner import NNTrainer

trainer = NNTrainer(d, 32, k, seed=0)
W1, b1, W2, b2, history, best_epoch = trainer.train(
    X_train, Y_train, X_val, Y_val,
    epochs=200, lr=lr, batch_size=64, lam=1e-4,
    checkpoint_on_val=True
)
```

---

## 3. Five-Seed Validation (`run_digits_5seeds.py`)

### Purpose
Estimate mean and 95% CI for model performance across multiple random seeds (robustness evaluation).

### What It Does
- Trains both Softmax and NN models 5 times with seeds {0, 1, 2, 3, 4}
- Computes 95% confidence intervals using t-distribution (n=5)
- Reports per-seed and aggregate statistics

### Usage
```bash
python run_digits_5seeds.py
```

### Output
- **Console**: Per-seed accuracy/loss, summary statistics with 95% CI
- **Log file**: `results/logs/run_digits_5seeds_YYYYMMDD_HHMMSS.log`

### Example Output
```
Softmax Test Accs : [0.951, 0.954, 0.952, 0.950, 0.953]
Softmax Accuracy
Mean: 0.9520, 95% CI: [0.9495, 0.9545], Std: 0.0016

NN Test Accs      : [0.968, 0.971, 0.970, 0.967, 0.969]
NN Accuracy
Mean: 0.9690, 95% CI: [0.9656, 0.9724], Std: 0.0015
```

### Implementation
```python
from training_utils import MetricsCalculator

for seed in [0, 1, 2, 3, 4]:
    trainer = SoftmaxTrainer(d, k, seed=seed)
    W, b, _, _ = trainer.train(...)
    loss, acc = trainer.evaluate(X_test, Y_test, y_test, lam)
    results.append(acc)

mean, lower, upper, std = MetricsCalculator.compute_ci95_for_five(results)
```

**Why 5 seeds?** Standard in ML literature for resource-limited settings. Uses Welch's t-distribution quantile (t_0.025,4 = 2.776) for accurate CI.

---

## Running All Ablations

```bash
# Sequential
python run_moons_ablation.py
python run_digits_optimizers.py
python run_digits_5seeds.py

# Or use a loop script
for script in run_moons_ablation.py run_digits_optimizers.py run_digits_5seeds.py; do
    echo "Running $script..."
    python $script
done
```

## Data Dependencies
All three require:
- `data/moons.npz` (Moons, Optimizers, 5-Seeds)
- `data/digits_data.npz` & `data/digits_split_indices.npz` (Optimizers, 5-Seeds)

## Reproducibility Notes
- Fixed random seeds (0-4) for 5-seed study
- Fixed learning rates for fair optimizer comparison
- Checkpoint on validation for all ablations
- Batch size and regularization held constant

## Integration with Main Pipeline
- **Input**: Run `run_core_experiments.py` first to verify baseline
- **Output**: Feeds into analysis (`failure_case_analysis.py`) and visualization scripts
