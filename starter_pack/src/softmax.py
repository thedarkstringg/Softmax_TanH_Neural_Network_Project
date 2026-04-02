"""
Matrix convention:
- X: (n, d) — n examples, d features
- W: (k, d) — k classes, d features
- b: (k,)
- S = X @ W.T + b → (n, k)
- dW: (k, d), db: (k,)
"""

import numpy as np


class SoftmaxLayer:
    """Softmax layer for multi-class classification."""

    def __init__(self, input_dim, num_classes, seed=0):
        """
        Initialize softmax layer with weights and biases.

        Args:
            input_dim: Input feature dimension (d)
            num_classes: Number of output classes (k)
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.standard_normal((num_classes, input_dim))
        self.b = np.zeros(num_classes)

    @staticmethod
    def softmax(S):
        """Compute softmax activation with numerical stability."""
        S_shifted = S - np.max(S, axis=1, keepdims=True)
        exp_S = np.exp(S_shifted)
        P = exp_S / np.sum(exp_S, axis=1, keepdims=True)
        return P

    def forward(self, X):
        """
        Forward pass: compute softmax probabilities.

        Args:
            X: Feature matrix of shape (n, d)

        Returns:
            P: Probability matrix of shape (n, k)
        """
        S = X @ self.W.T + self.b
        P = self.softmax(S)
        return P

    def backward(self, X, P, Y_onehot):
        """
        Compute gradients for softmax regression.

        Args:
            X: Feature matrix of shape (n, d)
            P: Probability matrix from forward pass, shape (n, k)
            Y_onehot: One-hot label matrix of shape (n, k)

        Returns:
            dW: Weight gradient of shape (k, d)
            db: Bias gradient of shape (k,)
        """
        n = X.shape[0]
        dS = (P - Y_onehot) / n
        dW = dS.T @ X
        db = dS.T @ np.ones(n)
        return dW, db

    def l2_loss(self, lam):
        """Compute L2 regularization loss."""
        return (lam / 2.0) * np.sum(self.W ** 2)

    def l2_grad(self, lam):
        """Compute L2 regularization gradient."""
        return lam * self.W

    def update_weights(self, dW, db, lr):
        """Update weights and biases."""
        self.W -= lr * dW
        self.b -= lr * db


class LossMetrics:
    """Utility class for loss and accuracy computation."""

    @staticmethod
    def cross_entropy_loss(P, Y_onehot):
        """Compute cross-entropy loss."""
        n = P.shape[0]
        P_clipped = np.clip(P, 1e-12, 1.0)
        L = -np.sum(Y_onehot * np.log(P_clipped)) / n
        return L

    @staticmethod
    def accuracy(P, Y):
        """Compute classification accuracy."""
        predictions = np.argmax(P, axis=1)
        acc = np.mean(predictions == Y)
        return acc


# Backward compatibility functions (deprecated)
def softmax(S):
    return SoftmaxLayer.softmax(S)

def cross_entropy_loss(P, Y_onehot):
    return LossMetrics.cross_entropy_loss(P, Y_onehot)

def accuracy(P, Y):
    return LossMetrics.accuracy(P, Y)

def softmax_forward(X, W, b):
    S = X @ W.T + b
    return SoftmaxLayer.softmax(S)

def softmax_gradients(X, P, Y_onehot):
    n = X.shape[0]
    dS = (P - Y_onehot) / n
    dW = dS.T @ X
    db = dS.T @ np.ones(n)
    return dW, db

def l2_regularization_loss(W, lam):
    return (lam / 2.0) * np.sum(W ** 2)

def l2_regularization_grad(W, lam):
    return lam * W
