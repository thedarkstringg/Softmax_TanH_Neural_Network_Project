import numpy as np


# -------------------------
# Mini-batches
# -------------------------
def make_minibatches(X, Y, batch_size=64, shuffle=True, seed=None):
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
        X = X[idx]
        Y = Y[idx]

    # yield one batch at a time
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i + batch_size], Y[i:i + batch_size]


# -------------------------
# Labels
# -------------------------
def labels_to_onehot(y, num_classes):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


def onehot_to_labels(Y_onehot):
    return np.argmax(Y_onehot, axis=1)


# -------------------------
# Metrics
# -------------------------
def accuracy_from_probs(P, Y_onehot):
    y_pred = np.argmax(P, axis=1)
    y_true = np.argmax(Y_onehot, axis=1)
    return np.mean(y_pred == y_true)


# -------------------------
# Seed statistics
# -------------------------
def compute_ci95_for_five(values):
    values = np.asarray(values, dtype=float)

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    margin = 2.776 * std / np.sqrt(5)

    # confidence interval bounds
    lower = mean - margin
    upper = mean + margin

    return mean, lower, upper, std