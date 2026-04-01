# confidence_analysis.py

A script for analyzing the **confidence calibration** of two models — Softmax and a Neural Network — on the digits dataset (`digits_data.npz`). It produces a **Reliability Diagram** showing how well each model's predicted confidence aligns with its actual accuracy.

---

## What the Script Does

1. Loads `digits_data.npz` and splits it into train / val / test (70 / 15 / 15).
2. Trains a **Softmax classifier** and a **two-layer Neural Network**.
3. Runs both models on the test set to get probability predictions.
4. Computes **confidence** (max predicted probability) and **predicted class** for each sample.
5. Bins samples into 5 confidence intervals and computes real accuracy per bin.
6. Plots and saves a **Reliability Diagram** to `figures/confidence_reliability.png`.

---

## Dependencies

| Module | Source |
|--------|--------|
| `numpy`, `matplotlib` | pip |
| `sklearn.model_selection` | pip (`scikit-learn`) |
| `softmax.softmax_forward` | `src/` |
| `neural_net.nn_forward` | `src/` |
| `training_utils.labels_to_onehot` | `src/` |
| `train_softmax.train_softmax` | `src/` |
| `train_nn_runner.train_nn_runner` | `src/` |

---

## Project Structure

```
project/
├── data/
│   └── digits_data.npz        # dataset (X, y)
├── src/
│   ├── softmax.py
│   ├── neural_net.py
│   ├── training_utils.py
│   ├── train_softmax.py
│   └── train_nn_runner.py
├── figures/                   # created automatically
└── scripts/
    └── confidence_analysis.py
```

---

## How to Run

```bash
python scripts/confidence_analysis.py
```

Output saved to:

```
figures/confidence_reliability.png
```

---

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Learning rate | 0.05 |
| Batch size | 64 |
| Regularization (λ) | 1e-4 |
| Hidden units (NN) | 32 |
| Seed | 0 |

---

## Key Functions

### `compute_confidence_and_predictions(P)`
Takes a probability matrix `P` of shape `(n, k)`.  
Returns:
- `confidence` — max probability per row (the model's confidence score)
- `predictions` — predicted class label (`argmax`)

### `compute_binned_accuracy(confidence, predictions, y_true, num_bins=5)`
Splits samples into `num_bins` equal-width confidence bins.  
For each non-empty bin, computes the fraction of correct predictions.  
Returns arrays of bin centers and corresponding accuracies.

---

## Output

The saved plot `confidence_reliability.png` contains:
- **Softmax** calibration curve (marker `o`)
- **Neural Net** calibration curve (marker `s`)
- **Perfect Calibration** diagonal (dashed line)

The closer a model's curve is to the diagonal, the better calibrated it is — meaning its confidence scores are reliable estimates of true accuracy.
