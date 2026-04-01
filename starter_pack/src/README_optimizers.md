# Optimizers (`src/optimizers.py`)

This module implements standard optimization algorithms used to update model parameters during training.

## Implemented Optimizers

### 1. Stochastic Gradient Descent (SGD)
- **`sgd_update(param, grad, lr)`**: The simplest update rule, moving parameters in the opposite direction of the gradient scaled by the learning rate.

### 2. Momentum
- **`init_momentum_state()`**: Initializes an empty dictionary to store velocity terms.
- **`momentum_update(param, grad, lr, velocity, beta=0.9)`**: Incorporates a moving average of past gradients to accelerate convergence and reduce oscillations.

### 3. Adam (Adaptive Moment Estimation)
- **`init_adam_state()`**: Initializes an empty dictionary for first and second moment estimates.
- **`adam_update(...)`**: A sophisticated optimizer that computes adaptive learning rates for each parameter by tracking both the mean (first moment) and uncentered variance (second moment) of the gradients.

## Usage

These functions are typically called within the training loop of `NeuralNet` or `Softmax` models when `optimizer != "sgd"`.
