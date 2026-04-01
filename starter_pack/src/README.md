# `src/` - Core Implementation

## Overview
This directory contains all core ML implementation files built from scratch using NumPy only (no high-level frameworks like scikit-learn or TensorFlow).

## Core Modules

### Foundation Layer
- **softmax.py** - Softmax regression implementation
  - `softmax()`: Numerically stable softmax with max subtraction
  - `softmax_forward()`: Forward pass for softmax
  - `softmax_gradients()`: Backprop gradients for softmax
  - `cross_entropy_loss()`: Cross-entropy loss with epsilon clipping
  - `l2_regularization_loss()`: L2 regularization term
  - `l2_regularization_grad()`: L2 regularization gradients
  - `accuracy()`: Classification accuracy metric

- **neural_net.py** - One-hidden-layer neural network
  - `hidden_activation()`: Tanh hidden layer activation
  - `nn_forward()`: Forward pass through network
  - `nn_loss()`: Total loss (cross-entropy + L2)
  - `nn_gradients()`: Backprop through network
  - `initialize_nn()`: Parameter initialization
  - `train_nn()`: Training loop (deprecated - use train_nn_runner)
  - `nn_predict()`: Prediction function

### Training Utilities
- **training_utils.py** - Helper functions for training
  - `labels_to_onehot()`: Convert class labels to one-hot encoding
  - `accuracy_from_probs()`: Compute accuracy from probability matrix
  - `make_minibatches()`: Create mini-batches for SGD
  - `compute_ci95_for_five()`: Confidence intervals from 5 runs

- **train_softmax.py** - Softmax training wrapper
  - `initialize_softmax()`: Parameter initialization
  - `softmax_loss()`: Loss computation with regularization
  - `evaluate_softmax()`: Evaluation on a dataset
  - `train_softmax()`: Complete training loop for softmax

- **train_nn_runner.py** - Neural network training wrapper
  - `evaluate_nn()`: Evaluation on a dataset
  - `train_nn_runner()`: Complete training loop for NN

- **optimizers.py** - Optimization algorithms
  - `sgd_update()`: Stochastic gradient descent
  - `momentum_update()`: SGD with momentum
  - `adam_update()`: ADAM optimizer

### Visualization & Analysis
- **plot_decision_boundaries.py** - Decision boundary visualization
  - `predict_class()`: Batch prediction on mesh grid
  - `plot_decision_boundary()`: Plot decision boundaries for 2D datasets
  - `main()`: Generate all decision boundary plots

- **plot_training_curves.py** - Training curve visualization
  - `plot_training_curve()`: Plot loss and accuracy curves
  - `main()`: Generate all training curve plots

- **confidence_analysis.py** - Confidence calibration analysis (Track B)
  - `compute_confidence_and_predictions()`: Extract confidence scores
  - `compute_binned_accuracy()`: Calibration metrics
  - `load_digits_dataset()`: Load digits data
  - `main()`: Generate reliability diagrams

### Utilities
- **experiment_logger.py** - Logging utilities for experiment runners
  - `run_with_logging()`: Decorator for experiment logging

## Design Notes

### Numerical Stability
- Softmax uses max subtraction to prevent overflow
- Cross-entropy uses epsilon clipping to prevent log(0)

### Regularization
- L2 (weight decay) regularization on both layers
- Configurable regularization parameter (lambda)

### Training
- Mini-batch stochastic gradient descent (SGD)
- Optional momentum and ADAM optimizers
- Early stopping with validation set monitoring
- Configurable: learning rate, batch size, epochs

## Dependencies
- NumPy (only external numerical library)
- Plotting: matplotlib (for visualizations only, not core ML)
- Data: sklearn.model_selection (for standard train/val/test splits, not ML algorithms)

## Usage
All modules are designed to be imported by scripts in `../scripts/`.
Import example:
```python
import sys
sys.path.insert(0, '../src')
from softmax import softmax_forward, cross_entropy_loss
from neural_net import nn_forward, initialize_nn
```
