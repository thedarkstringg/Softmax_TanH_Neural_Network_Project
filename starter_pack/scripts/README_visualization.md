# Visualization Scripts

## 1. Decision Boundary Plots (`plot_decision_boundaries.py`)

### Purpose
Visualize model decision boundaries on 2D toy datasets to understand learning behavior and model differences.

### What It Does
1. **Trains models** on Linear Gaussian and Moons datasets
2. **Creates meshgrid** for fine-grained decision surface
3. **Plots contours** showing decision regions
4. **Overlays training data** points colored by class
5. **Generates**: 1x3 comparison per dataset (Softmax + NN side-by-side)

### Usage
```bash
python plot_decision_boundaries.py
```

### Output
- **6 PNG files** in `figures/`:
  - `decision_boundary_linear_gaussian_softmax.png`
  - `decision_boundary_linear_gaussian_nn.png`
  - `decision_boundary_moons_softmax.png`
  - `decision_boundary_moons_nn.png`
  - Plus comparison plots from ablations
- **Console**: Progress messages during training and plotting

### Visualization Details

**Linear Gaussian**:
```
Softmax: Linear decision boundary (straight line separates classes)
NN:      Linear decision boundary (same, extra capacity unused)
```

**Moons**:
```
Softmax: Linear boundary (misses crescent shape, ~70% accuracy)
NN:      Curved boundary (fits crescents perfectly, ~95% accuracy)
```

### Implementation
```python
from train_softmax import SoftmaxTrainer
from train_nn_runner import NNTrainer
from training_utils import DataUtils

# Create meshgrid
x_min = X_train[:, 0].min() - 0.5
x_max = X_train[:, 0].max() + 0.5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)
X_mesh = np.c_[xx.ravel(), yy.ravel()]

# Get predictions
P = softmax_forward(X_mesh, W, b)
Z = np.argmax(P, axis=1).reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Spectral')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
           edgecolors='k', cmap='Spectral', s=40)
```

### Mesh Parameters
- **Step size**: 0.02 (fine detail, ~2500 points per dimension)
- **Padding**: 0.5 units beyond data extremes
- **Resolution**: ~100x100 grid per plot (balances detail vs speed)

### Class-Based Benefits
- Trainers encapsulate model state transparently
- Easy to extract trained weights: `trainer.model.W`, `trainer.model.b`
- Consistent interface across model types

### Typical Outputs
- **Softmax**: Piecewise linear decision regions
- **NN**: Smooth, curved boundaries (tanh non-linearity)
- **Over-capacity**: Overly complex boundaries on simple tasks

---

## 2. Optimizer Curves (`plot_optimizer_curves.py`)

### Purpose
Parse training logs and generate comparison curves for optimizer performance visualization.

### What It Does
1. **Parses log file** from `run_digits_optimizers.py`
2. **Extracts metrics** for each optimizer (SGD, Momentum, Adam)
3. **Generates comparison plots**:
   - Training loss vs epoch
   - Validation loss vs epoch
   - Training accuracy vs epoch
   - Validation accuracy vs epoch
4. **Saves 2x2 subplot figure** with all metrics

### Usage
```bash
# First, run the optimizer study
python run_digits_optimizers.py

# Then visualize the results
python plot_optimizer_curves.py
```

### Output
- **Saved plot**: `optimizer_comparison_2x2.png` in `figures/`
- **Shows**: SGD (blue), Momentum (green), Adam (red)
- **Highlights**: Stability differences, convergence speed, final accuracy

### Log Parsing
```python
def parse_optimizer_log(log_path):
    """Extract metrics from log lines like:
    [sgd] Epoch 0, Train Loss: 2.3045, Train Acc: 0.2315, ...
    """
    pattern = r'\[(\w+)\] Epoch (\d+), .*Loss: ([\d.]+), Acc: ([\d.]+)'
    # Extracts: optimizer_name, epoch, loss, accuracy
```

### Visualization Features
- **Colors**: SGD=blue, Momentum=green, Adam=red (consistent across plots)
- **Line styles**: Training=solid, Validation=dashed
- **Markers**: Small circles for readability
- **Legend**: Clearly labeled per optimizer
- **Grid**: Light grid for easy value reading

### Typical Patterns Observed

**Training Loss**:
- SGD: Steady, smooth decrease
- Momentum: Faster initial decrease, may oscillate slightly
- Adam: Fast initial, then **spikes** at epochs 30, 80, 130

**Validation Accuracy**:
- SGD: ~95%, stable
- Momentum: ~94%, stable
- Adam: ~96% (best), but **unstable trajectory**

### Data Dependencies
- Requires log file from `run_digits_optimizers.py`
- Default: `results/logs/run_digits_optimizers_*.log`
- Can specify custom log path in `parse_optimizer_log(log_path)`

### Matplotlib Configuration
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {'sgd': 'blue', 'momentum': 'green', 'adam': 'red'}

# Plot i: Training Loss
ax.plot(epochs, loss_values, 'o-', color=colors[opt], label=opt.upper())
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Training Loss', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
```

---

## Running Visualization Pipeline

```bash
# Generate decision boundaries for main experiments
python plot_decision_boundaries.py

# Generate decision boundaries for ablation (h=2, 8, 32)
python run_moons_ablation.py  # Includes plot generation

# Generate optimizer comparison curves
python run_digits_optimizers.py
python plot_optimizer_curves.py

# View all outputs
ls figures/
```

## Output Organization
```
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

## Customization

**Plot Resolution** (plot_decision_boundaries.py):
```python
step_size = 0.01  # Finer detail, slower rendering
step_size = 0.02  # Default, balanced
step_size = 0.05  # Coarser, faster
```

**Figure Size**:
```python
figsize=(10, 8)  # Default 2D plot
figsize=(15, 4)  # Wide 1x3 comparison
figsize=(14, 10) # 2x2 optimizer grid
```

## Reproducibility Notes
- All plots use seed=0 models
- Consistent color schemes across visualizations
- High DPI (300) for publication-quality output
- Saved in PNG format (compatible with reports)
