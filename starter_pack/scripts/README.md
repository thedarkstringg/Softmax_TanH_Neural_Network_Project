# Scripts Overview

Complete reference for all scripts in the Math4AI capstone project. Scripts are organized by purpose and designed to run in a coherent pipeline using **class-based architecture**.

## Quick Start

```bash
# 1. Validate setup
python visualize_data.py
python sanity_checks.py

# 2. Run core experiments
python run_core_experiments.py

# 3. Run ablation studies
python run_moons_ablation.py
python run_digits_5seeds.py
python run_digits_optimizers.py

# 4. Analysis & visualization
python failure_case_analysis.py
python run_track_b.py
python plot_decision_boundaries.py
python plot_optimizer_curves.py
```

## Quick Reference

| Category | Scripts | Purpose | Time |
|----------|---------|---------|------|
| **Validation** | `visualize_data.py`, `sanity_checks.py` | Data exploration & code validation | <1 min |
| **Core Experiment** | `run_core_experiments.py` | Baseline: Softmax vs NN on 3 datasets | 5-10 min |
| **Ablation Studies** | `run_moons_ablation.py`, `run_digits_5seeds.py`, `run_digits_optimizers.py` | Capacity, robustness, optimizer comparison | 15 min |
| **Analysis** | `failure_case_analysis.py`, `run_track_b.py` | Root cause & confidence analysis | 10 min |
| **Visualization** | `plot_decision_boundaries.py`, `plot_optimizer_curves.py` | Plot results | 5 min |
| **Utilities** | `train_softmax.py`, `train_nn_runner.py` | Standalone trainers | Custom |

**Total Time**: ~30-50 minutes for complete pipeline

---

## Detailed Documentation

Each category has its own dedicated README:

### 1. **Core Experiments** → [README_CORE_EXPERIMENTS.md](README_CORE_EXPERIMENTS.md)
`run_core_experiments.py` - Baseline comparison of Softmax vs Neural Networks

- 3 datasets: Linear Gaussian, Moons, Digits
- Both models trained with same hyperparameters
- Generates console metrics + log file
- **Key Finding**: NN needed for Moons (non-linear), extra capacity hurts simple tasks

### 2. **Ablation Studies** → [README_ABLATION_STUDIES.md](README_ABLATION_STUDIES.md)

**Capacity Ablation** (`run_moons_ablation.py`):
- Tests h ∈ {2, 8, 32} on Moons
- Finding: h=8 optimal, h=32 overfits

**Robustness** (`run_digits_5seeds.py`):
- Trains with 5 random seeds
- Computes 95% confidence intervals
- Validates reproducibility

**Optimizer Study** (`run_digits_optimizers.py`):
- Compares SGD vs Momentum vs Adam
- Finding: Adam best (96.2%) but unstable at lr=0.05

### 3. **Analysis & Failure Cases** → [README_ANALYSIS.md](README_ANALYSIS.md)

**Failure Case Analysis** (`failure_case_analysis.py`):
- Root cause: Why does Adam become unstable?
- Answer: lr too aggressive for adaptive learning rates
- Effective lr = 0.05/√v ≈ 1.58 when v is small

**Confidence & Reliability Analysis** → [README_TRACK_B.md](README_TRACK_B.md)
- Track B: Compare Softmax vs NN on **calibration, confidence, and uncertainty**
- Binned accuracy analysis (5 confidence bins)
- Entropy separation (correct vs incorrect predictions)
- Finding: NN achieves 95.11% accuracy + better calibration (entropy 3.3× higher on errors)
- Outputs: 3 publication-quality figures + detailed console tables

### 4. **Visualization** → [README_VISUALIZATION.md](README_VISUALIZATION.md)

**Decision Boundaries** (`plot_decision_boundaries.py`):
- 2D decision regions for Linear Gaussian + Moons
- Shows Softmax linear vs NN curved boundaries
- Training data overlaid as scatter points

**Optimizer Curves** (`plot_optimizer_curves.py`):
- Parses training logs
- 2x2 plot: Train/Val Loss and Accuracy vs Epoch
- Highlights optimizer differences

### 5. **Training Utilities** → [README_TRAINING.md](README_TRAINING.md)

**Softmax Trainer** (`train_softmax.py`):
```python
from train_softmax import SoftmaxTrainer
trainer = SoftmaxTrainer(d, k, seed=0)
W, b, history, best_epoch = trainer.train(...)
```

**NN Trainer** (`train_nn_runner.py`):
```python
from train_nn_runner import NNTrainer
trainer = NNTrainer(d, h, k, seed=0)
W1, b1, W2, b2, history, best_epoch = trainer.train(...)
```

### 6. **Utilities & Validation** → [README_UTILITIES.md](README_UTILITIES.md)

**Sanity Checks** (`sanity_checks.py`):
- Verify softmax outputs valid probabilities
- Check numerical stability on extreme inputs
- Validate analytical vs numerical gradients

**Data Visualization** (`visualize_data.py`):
- Explore data shapes and distributions
- Plot synthetic datasets (2D)
- View MNIST digit samples

---

## Class-Based Architecture

All training scripts use unified **class-based API**:

### Trainers
```python
from train_softmax import SoftmaxTrainer
from train_nn_runner import NNTrainer

# Create trainer instances
softmax_trainer = SoftmaxTrainer(input_dim, num_classes, seed=0)
nn_trainer = NNTrainer(input_dim, hidden_dim, num_classes, seed=0)

# Train (automatic mini-batching)
W, b, history, best_epoch = softmax_trainer.train(
    X_train, Y_train, y_train,
    X_val, Y_val, y_val,
    epochs=100, lr=0.05, batch_size=64, lam=1e-4,
    checkpoint_on_val=True
)

# Evaluate
loss, acc = softmax_trainer.evaluate(X_test, Y_test, y_test, lam=1e-4)
```

### Data Utilities
```python
from training_utils import DataUtils, MetricsCalculator

# Transform data
Y = DataUtils.labels_to_onehot(y, num_classes)

# Mini-batching
for X_batch, Y_batch in DataUtils.make_minibatches(X, Y, batch_size=64):
    # Train on batch

# Compute metrics
acc = MetricsCalculator.accuracy_from_probs(P, Y)
mean, lower, upper, std = MetricsCalculator.compute_ci95_for_five([...])
```

### Logging
```python
from experiment_logger import ExperimentLogger

logger = ExperimentLogger(CURRENT_DIR, "run_name")
logger.run(main)  # Logs to console + file simultaneously
```

---

## Data Requirements

All scripts require **pre-generated datasets** in `data/`:

```
data/
├── linear_gaussian.npz          # 2D linear classification
├── moons.npz                    # 2D non-linear classification
├── digits_data.npz              # 1797 8x8 handwritten digits
└── digits_split_indices.npz     # Fixed train/val/test split
```

**Generate missing data**:
```bash
python scripts/generate_synthetic.py  # Creates synthetic datasets
python scripts/make_digits_split.py   # Creates split indices
```

---

## Output Organization

```
results/
├── logs/                                        # Auto-generated
│   ├── run_core_experiments_YYYYMMDD_HHMMSS.log
│   ├── run_moons_ablation_YYYYMMDD_HHMMSS.log
│   ├── run_digits_5seeds_YYYYMMDD_HHMMSS.log
│   ├── run_digits_optimizers_YYYYMMDD_HHMMSS.log
│   ├── failure_case_analysis_YYYYMMDD_HHMMSS.log
│   └── run_track_b_YYYYMMDD_HHMMSS.log
└── ABLATION_ANALYSIS.md                         # Optional detailed analysis

figures/
├── decision_boundary_linear_gaussian_softmax.png
├── decision_boundary_linear_gaussian_nn.png
├── decision_boundary_moons_softmax.png
├── decision_boundary_moons_nn.png
├── decision_boundary_moons_ablation_h_2.png
├── decision_boundary_moons_ablation_h_8.png
├── decision_boundary_moons_ablation_h_32.png
├── moons_ablation_comparison.png
└── optimizer_comparison_2x2.png
```

---

## Key Findings Summary

### Model Comparison (run_core_experiments.py)
| Dataset | Softmax | NN | Best For |
|---------|---------|-----|----------|
| Linear Gaussian | 99% | 99% | Softmax (simpler) |
| Moons | 70% | 95% | NN (non-linear needed) |
| Digits | 95% | 97% | NN (complex features) |

### Capacity Ablation (run_moons_ablation.py)
- **h=2**: Too narrow, limited learning (~85%)
- **h=8**: Sweet spot (~85%, best generalization)
- **h=32**: Over-capacity, overfits (~85% test, higher loss)
- **Conclusion**: No benefit beyond h=8 on simple Moons task

### Optimizer Comparison (run_digits_optimizers.py)
| Optimizer | Test Acc | Val Loss | Best Epoch | Notes |
|-----------|----------|----------|-----------|-------|
| SGD | **95.11%** | 0.1255 | 199 | Best generalization; slowest convergence |
| Momentum | 94.57% | 0.1073 | 63 | Fast convergence; slightly lower test acc |
| Adam | 94.84% | **0.0991** | 198 | Best validation loss but overfits to val set |

**Finding**: SGD achieves highest test accuracy despite lowest validation loss in Adam. Suggests adaptive learning rates overfit validation set while SGD's noise provides implicit regularization.

### Root Cause: Adam Instability (failure_case_analysis.py)
```
Adam: θ -= (lr / √v_t) * ∇L
      = θ -= effective_lr * ∇L

When v_t small (early training):
  effective_lr ≈ 0.05/√0.001 ≈ 1.58  (TOO LARGE!)
  → huge updates → oscillations → loss spikes

SGD: θ -= 0.05 * ∇L (constant, stable)
Momentum: θ -= 0.05 * v_t (bounded, stable)
```

### Confidence Analysis (run_track_b.py)
- **Softmax**: Mean confidence 0.8449, Mean entropy 0.5323, Test Acc 94.02%
- **NN**: Mean confidence 0.9330, Mean entropy 0.2209, Test Acc 95.11%
- **Key Finding**: NN shows 3.3× entropy separation (correct/incorrect) vs Softmax 2.0×. Better calibration and uncertainty quantification

---

## Execution Workflows

### **Minimal Flow** (5 min - Just Baseline)
```bash
python run_core_experiments.py
```

### **Quick Validation** (5 min)
```bash
python visualize_data.py
python sanity_checks.py
python run_core_experiments.py
```

### **Ablation Studies Only** (15 min)
```bash
python run_moons_ablation.py
python run_digits_5seeds.py
python run_digits_optimizers.py
```

### **Complete Pipeline** (30-50 min)
```bash
# Validation
python visualize_data.py
python sanity_checks.py

# Baseline
python run_core_experiments.py

# Ablations
python run_moons_ablation.py
python run_digits_5seeds.py
python run_digits_optimizers.py

# Analysis
python failure_case_analysis.py
python run_track_b.py

# Visualization
python plot_decision_boundaries.py
python plot_optimizer_curves.py
```

### **Scripted Run** (Parallel Safe)
```python
# scripts/run_all.sh
#!/bin/bash
set -e

echo "Starting full pipeline..."
python visualize_data.py &
python sanity_checks.py &
wait

python run_core_experiments.py
python run_moons_ablation.py &
python run_digits_5seeds.py &
python run_digits_optimizers.py &
wait

python failure_case_analysis.py &
python run_track_b.py &
wait

python plot_decision_boundaries.py
python plot_optimizer_curves.py

echo "Pipeline complete! Check results/ and figures/"
```

---

## Debugging Tips

**Scripts won't run?**
1. Check data: `ls data/*.npz`
2. Check imports: `python -c "from training_utils import DataUtils"`
3. Run validation: `python sanity_checks.py`

**Unexpected results?**
1. Verify seed (default: 0) for reproducibility
2. Check data split correctness: `python visualize_data.py`
3. Review hyperparameters in script
4. Look at console prints during training

**High memory usage?**
- Reduce batch size (but may affect convergence)
- Plot less frequently
- Process fewer seeds in 5-seed study

**Slow training?**
- Reduce epochs (use early stopping with checkpoint_on_val=True)
- Use fewer hidden units
- Reduce meshgrid resolution (plot_decision_boundaries.py: step_size=0.05)

---

## Performance Targets

- **Linear Gaussian**: Softmax 99%+, NN 99%+
- **Moons**: Softmax 70%±, NN 95%±
- **Digits** (single run): Softmax 95%±, NN 97%±
- **Digits** (5-seed CI): Within ±0.005 margin

---

## Integration with Report/Paper

**Figures for writeup**:
- `figures/decision_boundary_*.png` → Model comparison section
- `figures/moons_ablation_comparison.png` → Capacity ablation results
- `figures/optimizer_comparison_2x2.png` → Optimizer study section

**Statistics/tables**:
- Run `run_digits_5seeds.py` → Extract mean ± 95% CI
- All detailed metrics in `results/logs/*.log`

---

## File Reference

**Core Experiments**:
- `run_core_experiments.py` - Main baseline comparison

**Ablation Studies**:
- `run_moons_ablation.py` - Capacity study (h ∈ {2, 8, 32})
- `run_digits_5seeds.py` - Robustness with confidence intervals
- `run_digits_optimizers.py` - Optimizer comparison (SGD, Momentum, Adam)

**Analysis**:
- `failure_case_analysis.py` - Root cause: Adam instability
- `run_track_b.py` - Confidence & calibration (Track B requirement)

**Visualization**:
- `plot_decision_boundaries.py` - 2D decision surface plots
- `plot_optimizer_curves.py` - Training curve comparisons

**Training Utilities**:
- `train_softmax.py` - Standalone Softmax trainer (function-based)
- `train_nn_runner.py` - Standalone NN trainer (function-based)

**Validation & Exploration**:
- `sanity_checks.py` - Implementation validation
- `visualize_data.py` - Data exploration & visualization

**Data Generation** (in scripts/):
- `generate_synthetic.py` - Create Linear Gaussian & Moons datasets
- `make_digits_split.py` - Generate train/val/test split indices

---

## Related Documentation

Each script category has detailed documentation:
- [README_CORE_EXPERIMENTS.md](README_CORE_EXPERIMENTS.md)
- [README_ABLATION_STUDIES.md](README_ABLATION_STUDIES.md)
- [README_ANALYSIS.md](README_ANALYSIS.md)
- [README_VISUALIZATION.md](README_VISUALIZATION.md)
- [README_TRAINING.md](README_TRAINING.md)
- [README_UTILITIES.md](README_UTILITIES.md)

---

**Last Updated**: 2026-04-02 (Complete Class-Based Refactoring)
**Architecture**: Class-Based API with unified interface
**Test Status**: All validation checks pass
