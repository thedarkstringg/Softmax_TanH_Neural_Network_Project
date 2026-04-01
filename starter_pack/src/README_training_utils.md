# Training Utilities (`src/training_utils.py`)

A collection of helper functions for data processing, label manipulation, and metric computation during model training and evaluation.

## Key Functions

### Data Handling
- **`make_minibatches(X, Y, batch_size=64, shuffle=True, seed=None)`**: A generator that yields shuffled mini-batches of data for stochastic gradient descent.

### Label Processing
- **`labels_to_onehot(y, num_classes)`**: Converts integer class labels into one-hot encoded vectors.
- **`onehot_to_labels(Y_onehot)`**: Reverts one-hot encoded vectors back to integer labels (using `argmax`).

### Evaluation Metrics
- **`accuracy_from_probs(P, Y_onehot)`**: Computes the classification accuracy given predicted probabilities and ground truth one-hot labels.

### Statistical Analysis
- **`compute_ci95_for_five(values)`**: Calculates the mean, 95% confidence interval (using t-distribution for N=5), and standard deviation for a set of five experimental runs.

## Usage

These utilities are used extensively by the training runner scripts and model implementations to standardize data flow and evaluation.
