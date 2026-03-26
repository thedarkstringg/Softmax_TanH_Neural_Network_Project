"""
Matrix convention:
- X: (n, d) — n examples, d features
- W: (k, d) — k classes, d features
- b: (k,)
- S = X @ W.T + b → (n, k)
- dW: (k, d), db: (k,)
"""

import numpy as np


def softmax(S):
    """
    To compute softmax activation with numerical stability.

    Args:
        S: Score matrix of shape (n, k)

    Returns:
        P: Probability matrix of shape (n, k), each row sums to 1
    """
    # log-sum-exp trick: subtract max per row for numerical stability
    S_shifted = S - np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S_shifted)
    P = exp_S / np.sum(exp_S, axis=1, keepdims=True)
    return P


def cross_entropy_loss(P, Y_onehot):
    """
    To compute cross-entropy loss.

    Args:
        P: Probability matrix of shape (n, k)
        Y_onehot: One-hot label matrix of shape (n, k)

    Returns:
        L: Scalar loss (mean over batch)
    """
    n = P.shape[0]
    # Clip to avoid log(0)
    P_clipped = np.clip(P, 1e-12, 1.0)
    # L = -(1/n) * sum(Y_onehot * log(P))
    L = -np.sum(Y_onehot * np.log(P_clipped)) / n
    return L


def accuracy(P, Y):
    """
    To compute classification accuracy.

    Args:
        P: Probability matrix of shape (n, k)
        Y: Integer label vector of shape (n,)

    Returns:
        acc: Scalar accuracy (fraction correct)
    """
    predictions = np.argmax(P, axis=1)
    acc = np.mean(predictions == Y)
    return acc


def softmax_forward(X, W, b):
    """
    Forward pass: compute softmax probabilities.

    Args:
        X: Feature matrix of shape (n, d)
        W: Weight matrix of shape (k, d)
        b: Bias vector of shape (k,)

    Returns:
        P: Probability matrix of shape (n, k)
    """
    S = X @ W.T + b  # (n, d) @ (d, k) + (k,) = (n, k)
    P = softmax(S)
    return P


def softmax_gradients(X, P, Y_onehot):
    """
    To compute gradients for softmax regression.

    Args:
        X: Feature matrix of shape (n, d)
        P: Probability matrix from forward pass, shape (n, k)
        Y_onehot: One-hot label matrix of shape (n, k)

    Returns:
        dW: Weight gradient of shape (k, d)
        db: Bias gradient of shape (k,)
    """
    n = X.shape[0]

    # Gradient w.r.t. logits
    dS = (P - Y_onehot) / n  # (n, k)

    # Gradient w.r.t. W
    dW = dS.T @ X  # (k, n) @ (n, d) = (k, d)

    # Gradient w.r.t. b
    db = dS.T @ np.ones(n)  # (k, n) @ (n,) = (k,)

    return dW, db


def l2_regularization_loss(W, lam):
    """
    To compute L2 regularization loss.

    Args:
        W: Weight matrix of shape (k, d)
        lam: Regularization coefficient (default 1e-4)

    Returns:
        reg_loss: Scalar (lam/2) * sum(W^2)
    """
    reg_loss = (lam / 2.0) * np.sum(W ** 2)
    return reg_loss


def l2_regularization_grad(W, lam):
    """
    To compute L2 regularization gradient.

    Args:
        W: Weight matrix of shape (k, d)
        lam: Regularization coefficient

    Returns:
        dW_reg: Gradient of shape (k, d), equals lam * W
    """
    dW_reg = lam * W
    return dW_reg
