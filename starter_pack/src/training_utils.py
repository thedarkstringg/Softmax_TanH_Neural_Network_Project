import numpy as np


class DataUtils:
    """Utility class for data transformations."""

    @staticmethod
    def make_minibatches(X, Y, batch_size=64, shuffle=True, seed=None):
        """
        Generate mini-batches from data.

        Args:
            X: Feature matrix
            Y: Labels or one-hot matrix
            batch_size: Batch size
            shuffle: Whether to shuffle
            seed: Random seed

        Yields:
            Tuples of (X_batch, Y_batch)
        """
        if shuffle:
            rng = np.random.default_rng(seed)
            idx = np.arange(X.shape[0])
            rng.shuffle(idx)
            X = X[idx]
            Y = Y[idx]

        for i in range(0, X.shape[0], batch_size):
            yield X[i:i + batch_size], Y[i:i + batch_size]

    @staticmethod
    def labels_to_onehot(y, num_classes):
        """Convert integer labels to one-hot encoding."""
        Y = np.zeros((len(y), num_classes))
        Y[np.arange(len(y)), y] = 1
        return Y

    @staticmethod
    def onehot_to_labels(Y_onehot):
        """Convert one-hot encoding back to integer labels."""
        return np.argmax(Y_onehot, axis=1)


class MetricsCalculator:
    """Utility class for metric calculations."""

    @staticmethod
    def accuracy_from_probs(P, Y_onehot):
        """
        Compute accuracy from probability matrix.

        Args:
            P: Probability matrix of shape (n, k)
            Y_onehot: One-hot label matrix of shape (n, k)

        Returns:
            Accuracy score
        """
        y_pred = np.argmax(P, axis=1)
        y_true = np.argmax(Y_onehot, axis=1)
        return np.mean(y_pred == y_true)

    @staticmethod
    def compute_ci95_for_five(values):
        """
        Compute 95% confidence interval for 5 samples.

        Args:
            values: List/array of 5 values

        Returns:
            Tuple of (mean, lower, upper, std)
        """
        values = np.asarray(values, dtype=float)

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        margin = 2.776 * std / np.sqrt(5)

        lower = mean - margin
        upper = mean + margin

        return mean, lower, upper, std