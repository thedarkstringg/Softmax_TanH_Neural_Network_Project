import os
import sys
import numpy as np

# make src importable
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from training_utils import make_minibatches, accuracy_from_probs
from neural_net import nn_forward, nn_gradients, nn_loss, initialize_nn


# -------------------------
# Evaluation
# -------------------------
def evaluate_nn(X, Y_onehot, W1, b1, W2, b2, lam):
    H, P = nn_forward(X, W1, b1, W2, b2)
    loss = nn_loss(P, Y_onehot, W1, W2, lam)
    acc = accuracy_from_probs(P, Y_onehot)
    return loss, acc


# -------------------------
# Training
# -------------------------
def train_nn_runner(
    X_train, Y_train_onehot,
    X_val, Y_val_onehot,
    d, h, k,
    epochs=100,
    lr=0.05,
    batch_size=64,
    lam=1e-4,
    seed=0,
    checkpoint_on_val=False,
):
    W1, b1, W2, b2 = initialize_nn(d, h, k, seed=seed)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best = {
        "val_loss": np.inf,
        "W1": W1.copy(),
        "b1": b1.copy(),
        "W2": W2.copy(),
        "b2": b2.copy(),
        "epoch": 0,
    }

    for epoch in range(epochs):
        for X_batch, Y_batch in make_minibatches(
            X_train, Y_train_onehot, batch_size=batch_size, shuffle=True, seed=seed + epoch
        ):
            H_batch, P_batch = nn_forward(X_batch, W1, b1, W2, b2)

            dW1, db1, dW2, db2 = nn_gradients(
                X_batch, H_batch, P_batch, Y_batch, W1, W2, lam
            )

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        train_loss, train_acc = evaluate_nn(
            X_train, Y_train_onehot, W1, b1, W2, b2, lam
        )
        val_loss, val_acc = evaluate_nn(
            X_val, Y_val_onehot, W1, b1, W2, b2, lam
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if checkpoint_on_val and val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["W1"] = W1.copy()
            best["b1"] = b1.copy()
            best["W2"] = W2.copy()
            best["b2"] = b2.copy()
            best["epoch"] = epoch

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    if checkpoint_on_val:
        return best["W1"], best["b1"], best["W2"], best["b2"], history, best["epoch"]

    return W1, b1, W2, b2, history, None