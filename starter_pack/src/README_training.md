# Training Utilities

Direct training scripts for Softmax and Neural Network models. These are lower-level utilities for custom training workflows.

## 1. Softmax Training (`train_softmax.py`)

### Purpose
Simplified standalone script for training Softmax regression models. Useful for:
- Quick model training without full experiment pipeline
- Debugging training issues
- Custom data loading workflows

### What It Does
1. **Loads custom data** (you modify this part)
2. **Trains Softmax model** with specified hyperparameters
3. **Prints training progress** every 10 epochs
4. **Optionally checkpoints** on best validation loss
5. **Returns trained weights** (W, b)

### Key Code
```python
def train_softmax(
    X_train, Y_train_onehot, y_train,
    X_val, Y_val_onehot, y_val,
    d, k,  # input_dim, num_classes
    epochs=100,
    lr=0.05,
    batch_size=64,
    lam=1e-4,
    seed=0,
    checkpoint_on_val=False,
):
    """
    Train softmax regression model.

    Returns:
        W, b, history, best_epoch
    """
    trainer = SoftmaxTrainer(d, k, seed=seed)
    return trainer.train(
        X_train, Y_train_onehot, y_train,
        X_val, Y_val_onehot, y_val,
        epochs=epochs, lr=lr, batch_size=batch_size,
        lam=lam, checkpoint_on_val=checkpoint_on_val
    )

# Usage: Custom data loading example
X_train = load_features()  # shape (n, d)
Y_train = labels_to_onehot(y_train, k)
W, b, history, best_epoch = train_softmax(
    X_train, Y_train, y_train,
    X_val, Y_val, y_val,
    d, k,
    epochs=100, lr=0.05, batch_size=64, lam=1e-4, seed=0
)
```

### Class Architecture
```python
from train_softmax import SoftmaxTrainer
from training_utils import DataUtils

trainer = SoftmaxTrainer(d, k, seed=seed)
# trainer.model: SoftmaxLayer instance
# trainer.seed: random seed

# Train with automatic mini-batching
W, b, history, best_epoch = trainer.train(
    X_train, Y_train_onehot, y_train,
    X_val, Y_val_onehot, y_val,
    epochs=100, lr=0.05, batch_size=64, lam=1e-4,
    checkpoint_on_val=True
)

# Evaluate
loss, acc = trainer.evaluate(X_test, Y_test, y_test, lam=1e-4)
```

### Training History Tracking
```python
history = {
    "train_loss": [...],    # per epoch
    "train_acc": [...],     # per epoch
    "val_loss": [...],      # per epoch
    "val_acc": [...],       # per epoch
}
```

### Checkpoint Behavior
- **checkpoint_on_val=False**: Return weights from final epoch
- **checkpoint_on_val=True**: Return weights from best validation epoch
  - Useful for preventing overfitting
  - Returns `best_epoch` index

### Output Format
```python
W, b, history, best_epoch = train_softmax(...)

# W: shape (k, d) - weight matrix
# b: shape (k,)   - bias vector
# history: dict with training metrics lists
# best_epoch: int or None (if checkpoint_on_val=True)
```

---

## 2. Neural Network Training (`train_nn_runner.py`)

### Purpose
Simplified standalone script for training 1-hidden-layer Neural Networks.

### What It Does
1. **Loads custom data**
2. **Trains NN model** with specified architecture and hyperparameters
3. **Prints progress** with metrics per 10 epochs
4. **Optionally checkpoints** on validation performance
5. **Returns trained weights** (W1, b1, W2, b2)

### Key Code
```python
def train_nn_runner(
    X_train, Y_train_onehot,
    X_val, Y_val_onehot,
    d, h, k,  # input_dim, hidden_dim, num_classes
    epochs=100,
    lr=0.05,
    batch_size=64,
    lam=1e-4,
    seed=0,
    checkpoint_on_val=False,
):
    """
    Train 1-hidden-layer neural network.

    Returns:
        W1, b1, W2, b2, history, best_epoch
    """
    trainer = NNTrainer(d, h, k, seed=seed)
    return trainer.train(
        X_train, Y_train_onehot,
        X_val, Y_val_onehot,
        epochs=epochs, lr=lr, batch_size=batch_size,
        lam=lam, checkpoint_on_val=checkpoint_on_val
    )

# Usage
Y_train = labels_to_onehot(y_train, k)
W1, b1, W2, b2, history, best_epoch = train_nn_runner(
    X_train, Y_train, X_val, Y_val,
    d, 32, k,  # 32-dim hidden layer
    epochs=100, lr=0.05, batch_size=64, lam=1e-4, seed=0
)
```

### Class Architecture
```python
from train_nn_runner import NNTrainer
from training_utils import DataUtils

trainer = NNTrainer(d, h, k, seed=seed)
# trainer.model: NeuralNetworkModel instance
# trainer.seed: random seed

W1, b1, W2, b2, history, best_epoch = trainer.train(...)
loss, acc = trainer.evaluate(X_test, Y_test, lam=1e-4)
```

### Network Architecture
```
Input (d features)
    |
    | W1: (h, d), b1: (h,)
    V
Hidden Layer (h units)
    | tanh activation
    |
    | W2: (k, h), b2: (k,)
    V
Output (k classes)
    |
    | softmax
    V
Probabilities (k,)
```

### Output Format
```python
W1, b1, W2, b2, history, best_epoch = train_nn_runner(...)

# W1: shape (h, d) - first layer weights
# b1: shape (h,)   - first layer bias
# W2: shape (k, h) - second layer weights
# b2: shape (k,)   - second layer bias
# history: dict with training metrics
# best_epoch: int or None (if checkpoint_on_val=True)
```

### Making Predictions
```python
from neural_net import NeuralNetworkModel

# Create model and load weights
model = NeuralNetworkModel(d, h, k)
model.W1, model.b1 = W1, b1
model.W2, model.b2 = W2, b2

# Forward pass
H, P = model.forward(X_test)  # H: hidden activations, P: probabilities
```

---

## Quick Start Examples

### Example 1: Softmax on Custom Data
```python
from train_softmax import train_softmax
from training_utils import DataUtils
import numpy as np

# Your data
X_train = np.random.randn(100, 64)  # 100 samples, 64 features
y_train = np.random.randint(0, 10, 100)  # labels 0-9
k = 10

# Format for training
Y_train = DataUtils.labels_to_onehot(y_train, k)

# Train
W, b, history, _ = train_softmax(
    X_train, Y_train, y_train,
    X_val, Y_val_onehot, y_val,
    d=64, k=10,
    epochs=100, lr=0.05, batch_size=64, lam=1e-4, seed=0
)
```

### Example 2: NN Training with Validation
```python
from train_nn_runner import train_nn_runner
from training_utils import DataUtils

# Data
X_train = load_train_features()  # (n_train, d)
y_train = load_train_labels()    # (n_train,)
X_val = load_val_features()      # (n_val, d)
y_val = load_val_labels()        # (n_val,)

Y_train = DataUtils.labels_to_onehot(y_train, 10)
Y_val = DataUtils.labels_to_onehot(y_val, 10)

# Train with checkpoint on validation
W1, b1, W2, b2, history, best_epoch = train_nn_runner(
    X_train, Y_train, X_val, Y_val,
    d=X_train.shape[1], h=32, k=10,
    epochs=200, lr=0.05, batch_size=64, lam=1e-4, seed=0,
    checkpoint_on_val=True
)

print(f"Best checkpoint at epoch {best_epoch}")
# Use W1, b1, W2, b2 for inference (best validation model)
```

---

## Class-Based Advantages

### State Management
```python
trainer = SoftmaxTrainer(d, k)
trainer.train(...)  # Weights stored in trainer.model

# Access trained weights
W = trainer.model.W
b = trainer.model.b
```

### Consistency
- Both scripts use identical API:
  - `trainer.train()` → returns weights + history
  - `trainer.evaluate()` → returns loss + accuracy

### Mini-Batching
- Automatic via `DataUtils.make_minibatches()`
- Reproducible with seed management
- Shuffled per epoch for SGD robustness

### Checkpoint Logic
- Validation-based checkpointing built-in
- Tracks best epoch automatically
- No need for manual validation loops

---

## Integration with Pipeline

```
train_softmax.py / train_nn_runner.py
    ↓ (return weights)
plot_decision_boundaries.py
    ↓ (load weights)
run_track_b.py (confidence analysis)
    ↓
failure_case_analysis.py
```

## Hyperparameter Reference

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `lr` | 0.01 - 0.1 | SGD/Momentum: 0.05 works well; Adam: 0.001 to avoid instability |
| `batch_size` | 32 - 128 | 64 is standard, balance memory vs noise |
| `lam` | 1e-5 - 1e-3 | L2 regularization, prevent overfitting |
| `epochs` | 100 - 500 | Digits: 100-200; Synthetic: 100-500; Ablations: up to 500 |
| `h` (hidden dim) | 8 - 128 | 32 is default; ablate {2, 8, 32} for capacity study |

## Debugging Tips

**Loss not decreasing?**
- Check learning rate (try 10x lower)
- Verify data is normalized [-1, 1]
- Ensure labels are one-hot encoded

**High train loss, low val loss?**
- Increase regularization (`lam`)
- Add checkpoint_on_val=True
- Reduce hidden layer size

**Instability/spikes?**
- Reduce learning rate (especially for Adam)
- Increase batch size
- Add L2 regularization
