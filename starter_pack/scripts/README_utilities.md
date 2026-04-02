# Utility Scripts

Helper scripts for validation, testing, and data exploration. These don't train models but support the main pipeline.

## 1. Sanity Checks (`sanity_checks.py`)

### Purpose
Verify correctness of core mathematical implementations (Softmax, gradients, loss functions).

### What It Does
Runs 3 validation checks on low-level functions:

#### Check 1: Probability Normalization
```python
def check_probabilities_sum_to_one():
    """Verify softmax outputs are valid probability distributions."""
    S = np.random.randn(5, 4)  # random logits
    P = softmax(S)
    # Check: sum(P) == 1 for each sample (row)
    sums = np.sum(P, axis=1)
    assert np.allclose(sums, 1.0)
```
- **Pass**: Each sample's probabilities sum to 1.0
- **Fail**: Softmax implementation broken

#### Check 2: Numerical Stability on Extreme Inputs
```python
def check_no_nan_inf_extreme_inputs():
    """Test softmax with extreme values doesn't overflow."""
    S_large_pos = np.full((3, 4), 1000.0)  # huge positive
    S_large_neg = np.full((3, 4), -1000.0) # huge negative
    S_mixed = np.array([[1000, -1000, 500, -500], ...])

    P = softmax(S_large_pos)
    assert not np.any(np.isnan(P))
    assert not np.any(np.isinf(P))
```
- **Pass**: No NaN/Inf even with extreme inputs (use log-subtract trick)
- **Fail**: Need better numerical stability

#### Check 3: Gradient Correctness via Numerical Gradient
```python
def check_numerical_gradient():
    """Verify analytical gradients match numerical gradients."""
    # Analytical gradient: dL/dW = backward()
    # Numerical gradient: (L(W+ε) - L(W-ε)) / (2ε)

    analytical_grad = softmax_gradients(X, P, Y)
    numerical_grad = compute_numerical_gradient(X, Y)

    assert np.allclose(analytical_grad, numerical_grad, rtol=1e-5)
```
- **Pass**: Analytical ≈ Numerical (both correct)
- **Fail**: Gradient computation has bugs

### Usage
```bash
python sanity_checks.py
```

### Output
```
PASS: probabilities sum to 1
PASS: no NaN or Inf on extreme inputs
PASS: analytical gradient matches numerical gradient
```

### Implementation Details
```python
from src.softmax import softmax, cross_entropy_loss, softmax_gradients

# All checks use low-level functions directly
# No model training involved
# Pure mathematical validation
```

### When to Run
1. **After implementing softmax**: Verify it works
2. **After changing gradient code**: Ensure still correct
3. **Debugging numerical issues**: Identify where problem is
4. **Integration testing**: Validate whole pipeline

### Typical Issues Found
- **NaN gradients**: Often from division by zero or log(0)
- **Gradient mismatch**: Implementation bug in backward pass
- **Overflow/underflow**: Need numerical stability tricks
  - Use `max trick`: `log(sum(exp(x - max(x))))`
  - Keep log-probabilities, not raw probabilities

---

## 2. Data Visualization (`visualize_data.py`)

### Purpose
Explore and visualize the three datasets used in experiments.

### What It Does
1. **Loads all datasets** from `data/` directory
2. **Prints dataset info**: Shapes, data types, ranges
3. **Plots synthetic datasets**: 2D scatter plots (Linear Gaussian, Moons)
4. **Plots MNIST digits**: 2x5 grid of sample images

### Usage
```bash
python visualize_data.py
```

### Output

**Console Output**:
```
--- Loaded linear_gaussian.npz ---
Key: X_train, Shape: (700, 2)
Key: y_train, Shape: (700,)
Key: X_val, Shape: (150, 2)
Key: y_val, Shape: (150,)
Key: X_test, Shape: (150, 2)
Key: y_test, Shape: (150,)

--- Loaded moons.npz ---
...

--- Loaded digits_data.npz ---
Key: X, Shape: (1797, 64)  # 1797 images, 64 pixels (8x8)
Key: y, Shape: (1797,)     # Labels 0-9
```

**Plotted Figures**:
1. **Synthetic Datasets** (2 subplots):
   - Left: Linear Gaussian (two Gaussian blobs, separable)
   - Right: Moons (two crescents, non-linearly separable)
   - Color = class, edge = outline for visibility

2. **MNIST Digits** (2x5 grid):
   - 10 sample images (one of each digit 0-9)
   - Grayscale visualization
   - Reshaped from (64,) to (8, 8) images

### Data Characteristics

**Linear Gaussian**:
```
- 700 training, 150 val, 150 test samples
- 2D features with Gaussian distribution
- Classes: 2 (binary classification)
- Linearly separable ✓
```

**Moons**:
```
- 700 training, 150 val, 150 test samples
- 2D features in crescent shapes
- Classes: 2 (binary classification)
- Non-linearly separable ✗ (need NN)
```

**MNIST Digits**:
```
- 1797 samples total
- 64 features (8x8 pixel images)
- Classes: 10 (0-9)
- Custom split: 70% train, 15% val, 15% test
```

### Implementation
```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_npz(path):
    """Load and print info about .npz file."""
    data = np.load(path)
    print(f"--- Loaded {path.name} ---")
    for key in data.keys():
        print(f"Key: {key}, Shape: {data[key].shape}")
    return data

def plot_synthetic(data, title, ax):
    """Plot 2D synthetic dataset."""
    X = np.vstack([data['X_train'], data['X_val'], data['X_test']])
    y = np.concatenate([data['y_train'], data['y_val'], data['y_test']])

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y,
                        cmap='coolwarm', edgecolors='k', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

def plot_digits(data):
    """Plot sample MNIST digits."""
    X = data['X']
    y = data['y']

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Example Digits (8x8)", fontsize=16)

    for i in range(10):
        idx = np.where(y == i)[0][0]  # First occurrence of digit i
        img = X[idx].reshape(8, 8)
        ax = axes[i // 5, i % 5]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {i}")
        ax.axis('off')

    plt.tight_layout()
```

### When to Use
1. **First time exploring project**: Understand dataset structure
2. **Debugging data issues**: Check if data loaded correctly
3. **Presentation/report**: Screenshots for documentation
4. **Sanity check**: Verify data hasn't been corrupted

### File Locations
```
data/
├── linear_gaussian.npz       # Synthetic: linearly separable
├── moons.npz                 # Synthetic: non-linearly separable
├── digits_data.npz           # Real: MNIST digits
└── digits_split_indices.npz  # Train/val/test split
```

### Tips for Interpretation

**Linear Gaussian**:
- Clear separation → Softmax (linear) sufficient
- No overlap allows perfect classification

**Moons**:
- Crescent moons → Linear classifier fails perpendicular
- Softmax struggles (linear boundary can't fit crescents)
- NN required for good performance

**MNIST Digits**:
- Handwritten variations within each class
- Some ambiguity between similar digits (3↔8, 1↔7, etc.)
- 10-way classification challenge

---

## Integration in Workflow

```
visualize_data.py → Understand data structure
    ↓
sanity_checks.py → Validate implementations
    ↓
run_core_experiments.py → Train baseline models
    ↓
plot_decision_boundaries.py → Visualize learned boundaries
```

## Quick Validation Checklist

Before running experiments:
- [ ] Run `python sanity_checks.py` → All PASS
- [ ] Run `python visualize_data.py` → Data looks correct
- [ ] Check figures in `figures/` → Decision boundaries reasonable

## Troubleshooting

**sanity_checks.py fails**:
1. Check softmax implementation → `src/softmax.py`
2. Verify gradient computation → `backward()` method
3. Test with smaller inputs first

**visualize_data.py missing plots**:
1. Check if data files exist: `ls data/*.npz`
2. Run `python scripts/generate_synthetic.py` if missing
3. Run `python scripts/make_digits_split.py` if missing

**Data corruption detected**:
- Check file sizes: `ls -lh data/`
- Regenerate clean data if needed
- Don't train models on corrupted data
