# Analysis Scripts

## 1. Failure Case Analysis (`failure_case_analysis.py`)

### Purpose
Investigate why Adam optimizer becomes unstable on Digits dataset despite achieving highest accuracy. Root cause analysis of loss spikes.

### What It Does
- Trains narrow (h=2) vs wide (h=32) networks to compare capacity effects
- Analyzes Adam's loss instability patterns
- Generates decision boundary visualizations
- Provides quantitative breakdown of failure modes

### Usage
```bash
python failure_case_analysis.py
```

### Output
- **Decision boundary plots**: `decision_boundary_moons_h_2.png`, `h_32.png`
- **Console output**: Detailed analysis with epoch-by-epoch metrics
- Prints spike detection and effective learning rate analysis

### Key Analysis: Why Adam Fails

**Observed Problem**:
Loss spikes at epochs 30, 80, 130 (7-20x degradation, then recovery)

**Root Cause**:
```
Adam Update Rule: θ_t = θ_{t-1} - lr * m̂_t / (√ŝ_t + ε)
                         = θ_{t-1} - effective_lr * ∇L

Where: effective_lr = lr / √ŝ_t

During early training with small ŝ_t:
    effective_lr ≈ 0.05 / √0.001 ≈ 1.58  (HUGE!)
    vs SGD's fixed: 0.05
```

**Why SGD/Momentum Don't Have This**:
- **SGD**: Direct update `θ -= lr * ∇L` (constant scaling)
- **Momentum**: Linear scaling with velocity `θ -= lr * v_t` (bounded)
- **Adam**: Divides by sqrt(second moment) → can explode early on

**Fix**: Use lower learning rate for Adam (e.g., lr=0.001)

### Implementation
```python
from train_nn_runner import NNTrainer

# Compare wide vs narrow
trainer_narrow = NNTrainer(d, 2, k, seed=0)
W1_n, b1_n, W2_n, b2_n, history_n, _ = trainer_narrow.train(
    X_train, Y_train, X_val, Y_val,
    epochs=500, lr=0.1, batch_size=64, lam=1e-4,
    checkpoint_on_val=False
)

trainer_wide = NNTrainer(d, 32, k, seed=0)
W1_w, b1_w, W2_w, b2_w, history_w, _ = trainer_wide.train(
    X_train, Y_train, X_val, Y_val,
    epochs=500, lr=0.1, batch_size=64, lam=1e-4,
    checkpoint_on_val=False
)
```

### Class-Based Advantages
- Trainers maintain model state throughout training
- `evaluate()` method provides consistent loss computation
- History tracking per epoch without manual accumulation

---

## 2. Confidence & Calibration Analysis (`run_track_b.py`)

### Purpose
"Track B" analysis: Evaluate model confidence and reliability (calibration) across prediction confidence levels.

### What It Does
1. **Train models** on Digits with fixed train/val/test split
2. **Compute confidence metrics**:
   - Maximum predicted class probability (confidence score)
   - Predictive entropy for uncertainty
   - Predicted class labels
3. **Binned analysis**: Partition predictions by confidence into 5 bins
4. **Generate visualizations**:
   - Reliability diagram (confidence vs accuracy)
   - Confidence-entropy comparison
   - Combined reliability curves
5. **Print interpretation**: Which model is better calibrated?

### Usage
```bash
python run_track_b.py
```

### Output
- **Tables**: Printed to console + log file
  - Confidence accuracy for each model
  - Summary statistics
  - Per-sample analysis for high-confidence mistakes
- **Plots**: Saved to `figures/`
  - `confidence_reliability.png` - reliability diagram
  - Confidence vs entropy scatter plots
  - Combined comparison plot
- **Log file**: `run_track_b_YYYYMMDD_HHMMSS.log`

### Key Metrics

**Confidence Calibration**:
```
For each confidence bin [0.0-0.2, 0.2-0.4, ..., 0.8-1.0]:
  - Plot: Confidence (x-axis) vs Accuracy (y-axis)
  - Perfect calibration: diagonal line (confidence = accuracy)
  - Above diagonal: under-confident (good!)
  - Below diagonal: over-confident (bad!)
```

**Predictive Entropy**:
```
H = -Σ p_i * log(p_i)
- High entropy: uncertain predictions
- Low entropy: confident predictions
- Compare: Does high confidence match low entropy?
```

### Implementation
```python
from training_utils import DataUtils
from train_softmax import SoftmaxTrainer
from train_nn_runner import NNTrainer

# Train
softmax_trainer = SoftmaxTrainer(d, k, seed=0)
W_sm, b_sm, _, _ = softmax_trainer.train(
    X_train, Y_train, y_train, X_val, Y_val, y_val,
    epochs=200, lr=0.05, batch_size=64, lam=1e-4,
    checkpoint_on_val=True
)

nn_trainer = NNTrainer(d, 32, k, seed=0)
W1, b1, W2, b2, _, _ = nn_trainer.train(
    X_train, Y_train, X_val, Y_val,
    epochs=200, lr=0.05, batch_size=64, lam=1e-4,
    checkpoint_on_val=True
)

# Get predictions and analyze calibration
P_sm = softmax_forward(X_test, W_sm, b_sm)
confidence = np.max(P_sm, axis=1)
entropy = -np.sum(P_sm * np.log(P_sm + 1e-10), axis=1)
```

### Typical Findings
- **Softmax**: Well-calibrated on simple patterns, poor on ambiguous cases
- **Neural Network**: Better calibrated overall, captures uncertainty in complex regions
- **Overconfidence**: Both models tend to be overconfident on hard examples

### Data Dependencies
- `data/digits_data.npz`
- `data/digits_split_indices.npz` (fixed split)

---

## Running Both Analysis Scripts

```bash
# Individual runs
python failure_case_analysis.py
python run_track_b.py

# Combined analysis workflow
python run_core_experiments.py      # Baseline
python run_digits_optimizers.py     # Optimizer comparison
python failure_case_analysis.py     # Root cause analysis
python run_track_b.py               # Confidence analysis
```

## Reproducibility
- Both use seed=0 for consistency
- Fixed data splits (indices-based)
- Checkpoint on validation for all models
- Batch size and regularization constants

## Integration
- **Input**: Trained models from main experiments
- **Output**: Analysis plots and insights inform final report
- **Feeds into**: Visualization and summary generation
