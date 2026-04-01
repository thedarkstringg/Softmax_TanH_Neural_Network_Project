# Math4AI Capstone: ML Classifiers from Scratch

A ground-up NumPy implementation comparing multiclass Softmax Regression with a single-hidden-layer Tanh Neural Network. This study evaluates the practical trade-off between model complexity and performance to determine exactly when a hidden layer adds value versus when it becomes redundant."

---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-only-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Project Overview

This capstone implements:
- **Softmax Regression** - Multiclass linear classifier with cross-entropy loss
- **Neural Network (1 Hidden Layer)** - Nonlinear classifier with tanh activation
- **Full Backpropagation** - Gradient computation from first principles
- **Multiple Optimizers** - SGD, Momentum, ADAM
- **Comprehensive Validation** - Numerical gradient checking, sanity tests, ablation studies

**Key Achievement:** No high-level ML frameworks used for core logic (NumPy only)

---

## Quick Start

```bash
cd starter_pack

# 1. Generate synthetic datasets
python scripts/generate_synthetic.py

# 2. Run core experiments (Softmax + NN on 3 datasets)
python scripts/run_core_experiments.py

# 3. Run 5-seed robustness analysis (Digits dataset)
python scripts/run_digits_5seeds.py

# 4. Compare optimizers (SGD vs Momentum vs ADAM)
python scripts/run_digits_optimizers.py

# 5. Architecture ablation study (Hidden layer width)
python scripts/run_moons_ablation.py

# 6. Verify numerical correctness
python scripts/sanity_checks.py
```

---

## Repository Structure

```
Math4AI_Final_Project/
├── deliverables/
│   └── math4ai_capstone_assignment.tex    # Official assignment
├── starter_pack/                          # Project root (active work)
│   ├── src/                              # Core ML implementations
│   │   ├── softmax.py                   # Softmax regression (NumPy)
│   │   ├── neural_net.py                # 1-hidden-layer NN (NumPy)
│   │   ├── training_utils.py            # Helpers: onehot, accuracy, minibatches
│   │   ├── train_softmax.py             # Softmax training loop
│   │   ├── train_nn_runner.py           # NN training loop
│   │   ├── optimizers.py                # SGD, Momentum, ADAM
│   │   ├── plot_decision_boundaries.py  # 2D decision boundary plots
│   │   ├── plot_training_curves.py      # Training/val loss & accuracy
│   │   ├── confidence_analysis.py       # Confidence calibration (Track B)
│   │   ├── experiment_logger.py         # Logging utilities
│   │   └── README.md                    # Core module documentation
│   │
│   ├── scripts/                          # Experiment runners
│   │   ├── generate_synthetic.py        # Create synthetic datasets
│   │   ├── run_core_experiments.py      # Main: Softmax + NN on 3 datasets
│   │   ├── run_digits_5seeds.py         # 5-run robustness analysis
│   │   ├── run_digits_optimizers.py     # Optimizer comparison
│   │   ├── run_moons_ablation.py        # Architecture ablation study
│   │   └── sanity_checks.py             # Numerical validation tests
│   │
│   ├── data/                             # Datasets
│   │   ├── digits_data.npz              # MNIST digits (1797 samples, 8×8)
│   │   ├── digits_split_indices.npz     # Fixed train/val/test splits
│   │   ├── linear_gaussian.npz          # Synthetic: linearly separable
│   │   └── moons.npz                    # Synthetic: nonlinear (crescent)
│   │
│   ├── figures/                          # Outputs
│   │   ├── decision_boundary_*.png      # Clean decision boundaries
│   │   ├── confidence_reliability.png   # Confidence calibration plot
│   │   └── README.md                    # Figure documentation
│   │
│   └── README.md                         # Detailed project guide
│
└── README.md                              # This file (project overview)
```

---

## Core Implementations

### Foundation: Softmax Regression
**File:** `starter_pack/src/softmax.py`

```
Forward:  P = softmax(X @ W.T + b)
Loss:     L = -sum(y * log(P)) + λ||W||²
Gradient: dW = (P - y).T @ X + 2λW
```

**Key Features:**
- Numerically stable (max subtraction)
- Epsilon clipping for log stability
- Analytical gradient computation
- Parameters: W (k×d), b (k,)

### Advanced: Neural Network
**File:** `starter_pack/src/neural_net.py`

```
Forward:  H = tanh(X @ W1.T + b1)
          P = softmax(H @ W2.T + b2)
Loss:     L = -sum(y * log(P)) + λ(||W1||² + ||W2||²)
Backward: Analytical backprop with chain rule
```

**Architecture:**
- Input: d features
- Hidden: h units (tanh activation)
- Output: k classes (softmax)

---

## Experiments & Results

| Experiment | Purpose | File | Outputs |
|-----------|---------|------|---------|
| **Core Experiments** | Train Softmax & NN on 3 datasets | `run_core_experiments.py` | Training curves, convergence validation |
| **5-Seed Robustness** | Statistical significance on Digits | `run_digits_5seeds.py` | Mean ± 95% CI across random initializations |
| **Optimizer Comparison** | SGD vs Momentum vs ADAM | `run_digits_optimizers.py` | Convergence speed, final accuracy |
| **Architecture Ablation** | Hidden layer width impact | `run_moons_ablation.py` | Decision boundaries for h∈{2,8,32} |

### Results Summary
- **Softmax on Digits:** ~97% test accuracy
- **Neural Network on Digits:** ~98% test accuracy
- **Nonlinear datasets:** NN decisively beats Softmax
- **All experiments reproducible** with fixed seeds

---

## Validation & Verification

### Sanity Checks (5/5 Pass)
```
✓ Softmax probabilities sum to 1
✓ No NaN/Inf in extreme inputs
✓ Numerical gradients match analytical (error < 1e-10)
✓ Loss decreases during training
✓ Tiny subset overfitting (100% on n=10)
```

### NumPy-Only Verification
- **Core ML:** Only NumPy used
- **No autodiff:** Full backprop written manually
- **No frameworks:** TensorFlow/PyTorch/scikit-learn not used for learning

**Run:** `python scripts/sanity_checks.py`

---

## Team & Contributions

| Member | Role |
|--------|------|
| **Ziyad Muradov** | Data loading/preprocessing, Softmax foundation, cross-entropy loss, numerical gradient checking |
| **Emin Huseynli** | Forward pass, hidden layer tanh activation, full backpropagation derivation |
| **Ismayil Yusifli** | Training loops, mini-batching implementation, experiment runners, seed studies |
| **Kamal Soltanaliyev** | Decision boundary visualization, training curves, statistical analysis (CI), Track B (confidence calibration) |

---

## Documentation

- **`starter_pack/README.md`** - Detailed project guide with usage examples
- **`starter_pack/src/README.md`** - API reference for core modules
- **`starter_pack/figures/README.md`** - Output figures documentation
- **`deliverables/math4ai_capstone_assignment.tex`** - Official assignment statement

---

## How to Reproduce Results

```bash
cd starter_pack

# Step 1: Ensure data is available
python scripts/generate_synthetic.py

# Step 2: Run all experiments (takes ~5 min)
python scripts/run_core_experiments.py
python scripts/run_digits_5seeds.py
python scripts/run_digits_optimizers.py
python scripts/run_moons_ablation.py

# Step 3: Verify numerical correctness
python scripts/sanity_checks.py

# Results appear in figures/ directory
```

**Reproducibility:** All experiments use fixed seeds (seed=0 by default, except 5-seed runs)

---

## Key Design Decisions

| Decision | Rationale | Implementation |
|----------|-----------|-----------------|
| Max subtraction in softmax | Overflow prevention | `S_shifted = S - max(S)` before exp |
| Epsilon clipping | Prevent log(0) | `log(max(P, ε))` |
| Mini-batch SGD | True stochastic training | Shuffle + batch updates |
| Tanh activation | Smooth, centered, gradient < 1 | Prevents vanishing gradients |
| L2 regularization | Weight decay | λ·||W||² added to loss |
| Validation checkpointing | Best-model selection | Restore best epoch post-training |
| Numerical gradient checking | Verify backprop | Compare finite diff vs analytical |

---

## Dependencies

**Core (Required):**
- Python 3.8+
- NumPy (only numerical library)

**Utilities (For experiments):**
- Matplotlib (visualization only)
- scikit-learn.model_selection (standard data splitting)

**NOT used despite available:**
- TensorFlow/PyTorch/JAX (ML frameworks)
- scikit-learn (ML algorithms)

---

## Submission Checklist

- ✅ All core ML code in `starter_pack/src/` (NumPy only)
- ✅ Clean decision boundary plots in `starter_pack/figures/`
- ✅ All runner scripts reproducible (`starter_pack/scripts/`)
- ✅ Sanity checks validate numerical correctness
- ✅ Team contributions documented
- ✅ README documentation complete
- ✅ Timestamped experiment logs cleaned

---

## Capstone Learning Objectives

✅ Implement softmax regression from scratch
✅ Implement feedforward neural network from scratch
✅ Derive and implement backpropagation analytically
✅ Implement SGD with momentum and ADAM optimizers
✅ Handle numerical stability (overflow, underflow, gradient checking)
✅ Validate using finite-difference gradient checking
✅ Compare models across datasets and architectures
✅ Analyze confidence calibration and model behavior

---


**Last Updated:** 1 April 2026 | **Implementation:** Complete | **Status:** Ready ✅
