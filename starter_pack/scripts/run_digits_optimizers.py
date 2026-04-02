import os
import sys
import numpy as np

# Add src to path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from training_utils import DataUtils, MetricsCalculator
from neural_net import nn_forward, nn_gradients, nn_loss, initialize_nn
from optimizers import sgd_update, momentum_update, adam_update
from experiment_logger import ExperimentLogger

# -------------------------
# Paths
# -------------------------
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data"))


# -------------------------
# Data loading
# -------------------------
def load_digits_dataset():
    data_path = os.path.join(DATA_DIR, "digits_data.npz")
    split_path = os.path.join(DATA_DIR, "digits_split_indices.npz")

    data = np.load(data_path)
    split = np.load(split_path)

    X = data["X"]
    y = data["y"]

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def accuracy_from_probs(P, Y_onehot):
    return MetricsCalculator.accuracy_from_probs(P, Y_onehot)

# -------------------------
# Evaluation
# -------------------------
def evaluate_nn(X, Y_onehot, W1, b1, W2, b2, lam):
    H, P = nn_forward(X, W1, b1, W2, b2)
    loss = nn_loss(P, Y_onehot, W1, W2, lam)
    acc = accuracy_from_probs(P, Y_onehot)
    return loss, acc


# -------------------------
# Training with optimizer
# -------------------------
def train_nn_with_optimizer(
    X_train, Y_train,
    X_val, Y_val,
    d, h, k,
    optimizer_name="sgd",
    epochs=200,
    lr=0.05,
    batch_size=64,
    lam=1e-4,
    seed=0,
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

    # optimizer states
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)

    mW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2)

    sW1 = np.zeros_like(W1)
    sb1 = np.zeros_like(b1)
    sW2 = np.zeros_like(W2)
    sb2 = np.zeros_like(b2)

    t = 0

    for epoch in range(epochs):
        for X_batch, Y_batch in DataUtils.make_minibatches(
            X_train, Y_train, batch_size=batch_size, shuffle=True, seed=seed + epoch
        ):
            H_batch, P_batch = nn_forward(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = nn_gradients(
                X_batch, H_batch, P_batch, Y_batch, W1, W2, lam
            )

            if optimizer_name == "sgd":
                W1 = sgd_update(W1, dW1, lr)
                b1 = sgd_update(b1, db1, lr)
                W2 = sgd_update(W2, dW2, lr)
                b2 = sgd_update(b2, db2, lr)

            elif optimizer_name == "momentum":
                W1, vW1 = momentum_update(W1, dW1, lr, vW1, beta=0.9)
                b1, vb1 = momentum_update(b1, db1, lr, vb1, beta=0.9)
                W2, vW2 = momentum_update(W2, dW2, lr, vW2, beta=0.9)
                b2, vb2 = momentum_update(b2, db2, lr, vb2, beta=0.9)

            elif optimizer_name == "adam":
                t += 1
                W1, mW1, sW1 = adam_update(W1, dW1, lr, mW1, sW1, t, beta1=0.9, beta2=0.999, eps=1e-8)
                b1, mb1, sb1 = adam_update(b1, db1, lr, mb1, sb1, t, beta1=0.9, beta2=0.999, eps=1e-8)
                W2, mW2, sW2 = adam_update(W2, dW2, lr, mW2, sW2, t, beta1=0.9, beta2=0.999, eps=1e-8)
                b2, mb2, sb2 = adam_update(b2, db2, lr, mb2, sb2, t, beta1=0.9, beta2=0.999, eps=1e-8)

        train_loss, train_acc = evaluate_nn(X_train, Y_train, W1, b1, W2, b2, lam)
        val_loss, val_acc = evaluate_nn(X_val, Y_val, W1, b1, W2, b2, lam)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["W1"] = W1.copy()
            best["b1"] = b1.copy()
            best["W2"] = W2.copy()
            best["b2"] = b2.copy()
            best["epoch"] = epoch

        if epoch % 10 == 0:
            print(
                f"[{optimizer_name}] Epoch {epoch}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    return best["W1"], best["b1"], best["W2"], best["b2"], history, best["epoch"]


# -------------------------
# Main
# -------------------------
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits_dataset()

    k = len(np.unique(y_train))
    d = X_train.shape[1]

    Y_train = DataUtils.labels_to_onehot(y_train, k)
    Y_val = DataUtils.labels_to_onehot(y_val, k)
    Y_test = DataUtils.labels_to_onehot(y_test, k)

    configs = [
        {"name": "sgd", "lr": 0.05},
        {"name": "momentum", "lr": 0.05},
        {"name": "adam", "lr": 0.001},
    ]

    results = []

    for cfg in configs:
        print(f"\n===== Optimizer: {cfg['name']} =====")

        W1, b1, W2, b2, history, best_epoch = train_nn_with_optimizer(
            X_train, Y_train,
            X_val, Y_val,
            d, 32, k,
            optimizer_name=cfg["name"],
            epochs=200,
            lr=cfg["lr"],
            batch_size=64,
            lam=1e-4,
            seed=0,
        )

        train_loss, train_acc = evaluate_nn(X_train, Y_train, W1, b1, W2, b2, 1e-4)
        val_loss, val_acc = evaluate_nn(X_val, Y_val, W1, b1, W2, b2, 1e-4)
        test_loss, test_acc = evaluate_nn(X_test, Y_test, W1, b1, W2, b2, 1e-4)

        print(f"{cfg['name']} Best Epoch: {best_epoch}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        results.append({
            "optimizer": cfg["name"],
            "best_epoch": best_epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history,
        })

    print("\n===== Summary =====")
    for r in results:
        print(
            f"{r['optimizer']}: "
            f"Val Loss={r['val_loss']:.4f}, Val Acc={r['val_acc']:.4f}, "
            f"Test Loss={r['test_loss']:.4f}, Test Acc={r['test_acc']:.4f}"
        )

    return results


if __name__ == "__main__":
    logger = ExperimentLogger(CURRENT_DIR, "run_digits_optimizers")
    logger.run(main)