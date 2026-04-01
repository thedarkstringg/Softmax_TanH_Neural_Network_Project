# plot_decision_boundaries.py

A script for visualizing **decision boundaries** of trained Softmax and Neural Network models on two-dimensional synthetic datasets. For each dataset × model combination, a separate PNG file is saved to `figures/`.

---

## What the Script Does

1. Loads two synthetic datasets: `linear_gaussian.npz` and `moons.npz`.
2. For each dataset, trains a **Softmax classifier** and a **two-layer Neural Network**.
3. Builds a dense meshgrid over the 2D feature space.
4. Predicts the class for every grid point and fills regions with corresponding colors.
5. Overlays the training points on top of the colored regions.
6. Saves 4 plots (2 datasets × 2 models) to `figures/`.

---

## Dependencies

| Module | Source |
|--------|--------|
| `numpy`, `matplotlib` | pip |
| `softmax.softmax_forward`, `softmax.accuracy` | `src/` |
| `neural_net.nn_forward`, `neural_net.initialize_nn` | `src/` |
| `training_utils.labels_to_onehot` | `src/` |
| `train_softmax.train_softmax` | `src/` |
| `train_nn_runner.train_nn_runner`, `.evaluate_nn` | `src/` |

---

## Project Structure

```
project/
├── data/
│   ├── linear_gaussian.npz    # (X_train, y_train, X_val, y_val, X_test, y_test)
│   └── moons.npz              # same format
├── src/
│   ├── softmax.py
│   ├── neural_net.py
│   ├── training_utils.py
│   ├── train_softmax.py
│   └── train_nn_runner.py
├── figures/                   # created automatically
└── scripts/
    └── plot_decision_boundaries.py
```

---

## How to Run

```bash
python scripts/plot_decision_boundaries.py
```

Output files saved to `figures/`:

```
figures/
├── decision_boundary_linear_gaussian_softmax.png
├── decision_boundary_linear_gaussian_nn.png
├── decision_boundary_moons_softmax.png
└── decision_boundary_moons_nn.png
```

---

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs (Softmax) | 100 |
| Epochs (NN) | 100 |
| Learning rate | 0.05 |
| Batch size | 64 |
| Regularization (λ) | 1e-4 |
| Hidden units (NN) | 32 |
| Meshgrid resolution | 200 × 200 |
| Padding around data | 0.2 |
| Seed | 0 |

---

## Key Functions

### `predict_class(X_mesh, W, b, model_type='softmax')`
Predicts class labels for all points in the meshgrid.
- `model_type='softmax'`: calls `softmax_forward(X_mesh, W, b)`
- `model_type='nn'`: `W` is a tuple `(W1, b1, W2, b2)`, calls `nn_forward`

### `plot_decision_boundary(X_train, y_train, model_params, dataset_name, model_name, mesh_resolution=100, padding=0.1)`
Builds and returns a `matplotlib` figure containing:
- Filled decision regions (`contourf`) with a color per class
- Black contour lines along decision boundaries
- Scatter plot of training points overlaid on the regions
- A colorbar indicating class labels

`model_params` is a dictionary in one of two formats:
```python
{'softmax': (W, b)}          # for Softmax
{'nn': (W1, b1, W2, b2)}     # for Neural Network
```

### `load_synthetic_dataset(filename)`
Loads a `.npz` file from `../data/` and returns:
```python
(X_train, y_train, X_val, y_val, X_test, y_test)
```

---

## Output

Each of the 4 plots shows:
- **Background** — color indicates the predicted class for every point in feature space
- **Dots** — training samples colored by their true class
- **Black lines** — boundaries between decision regions

Comparing both models on the same dataset illustrates that the Neural Network can learn significantly more complex, non-linear decision boundaries than the linear Softmax classifier.
