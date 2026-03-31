import os
import sys
import numpy as np

# make src importable
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from softmax import (
    softmax_forward,
    softmax_gradients,
    cross_entropy_loss,
    l2_regularization_loss,
    l2_regularization_grad,
    accuracy,
)

from training_utils import make_minibatches


# -------------------------
# Initialization
# -------------------------
def initialize_softmax(d, k, seed=0):
    rng = np.random.default_rng(seed)

    W = 0.01 * rng.standard_normal((k, d))
    b = np.zeros(k)

    return W, b


# -------------------------
# Loss
# -------------------------
def softmax_loss(P, Y_onehot, W, lam):
    data_loss = cross_entropy_loss(P, Y_onehot)
    reg_loss = l2_regularization_loss(W, lam)
    return data_loss + reg_loss


# -------------------------
# Evaluation
# -------------------------
def evaluate_softmax(X, Y_onehot, y, W, b, lam):
    P = softmax_forward(X, W, b)
    loss = softmax_loss(P, Y_onehot, W, lam)
    acc = accuracy(P, y)
    return loss, acc


# -------------------------
# Training
# -------------------------
def train_softmax(
    X_train, Y_train_onehot, y_train,
    X_val, Y_val_onehot, y_val,
    d, k,
    epochs=100,
    lr=0.05,
    batch_size=64,
    lam=1e-4,
    seed=0,
    checkpoint_on_val=False,
):
    W, b = initialize_softmax(d, k, seed=seed)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best = {
        "val_loss": np.inf,
        "W": W.copy(),
        "b": b.copy(),
        "epoch": 0,
    }

    for epoch in range(epochs):
        for X_batch, Y_batch in make_minibatches(
            X_train, Y_train_onehot, batch_size=batch_size, shuffle=True, seed=seed + epoch
        ):
            P_batch = softmax_forward(X_batch, W, b)

            dW, db = softmax_gradients(X_batch, P_batch, Y_batch)
            dW += l2_regularization_grad(W, lam)

            W -= lr * dW
            b -= lr * db

        train_loss, train_acc = evaluate_softmax(
            X_train, Y_train_onehot, y_train, W, b, lam
        )
        val_loss, val_acc = evaluate_softmax(
            X_val, Y_val_onehot, y_val, W, b, lam
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if checkpoint_on_val and val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["W"] = W.copy()
            best["b"] = b.copy()
            best["epoch"] = epoch

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    if checkpoint_on_val:
        return best["W"], best["b"], history, best["epoch"]

    return W, b, history, None