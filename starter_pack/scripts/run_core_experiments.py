import os
import numpy as np

from training_utils import labels_to_onehot
from train_softmax import train_softmax, evaluate_softmax
from train_nn_runner import train_nn_runner, evaluate_nn
from experiment_logger import run_with_logging

# -------------------------
# Paths
# -------------------------
CURRENT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data"))


# -------------------------
# Data loading
# -------------------------
def load_synthetic_dataset(filename):
    path = os.path.join(DATA_DIR, filename)
    data = np.load(path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_val, y_val, X_test, y_test


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


# -------------------------
# Single experiment
# -------------------------
def run_dataset_experiment(
    name,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    nn_hidden=32,
    softmax_epochs=100,
    nn_epochs=100,
    softmax_lr=0.05,
    nn_lr=0.05,
    batch_size=64,
    lam=1e-4,
    seed=0,
    checkpoint_on_val=False,
):
    d = X_train.shape[1]
    k = len(np.unique(y_train))

    Y_train = labels_to_onehot(y_train, k)
    Y_val = labels_to_onehot(y_val, k)
    Y_test = labels_to_onehot(y_test, k)

    print(f"\n===== {name}: Softmax =====")
    W, b, softmax_history, softmax_best_epoch = train_softmax(
        X_train, Y_train, y_train,
        X_val, Y_val, y_val,
        d, k,
        epochs=softmax_epochs,
        lr=softmax_lr,
        batch_size=batch_size,
        lam=lam,
        seed=seed,
        checkpoint_on_val=checkpoint_on_val,
    )

    train_loss_sm, train_acc_sm = evaluate_softmax(X_train, Y_train, y_train, W, b, lam)
    val_loss_sm, val_acc_sm = evaluate_softmax(X_val, Y_val, y_val, W, b, lam)
    test_loss_sm, test_acc_sm = evaluate_softmax(X_test, Y_test, y_test, W, b, lam)

    print(f"Softmax Train  - Loss: {train_loss_sm:.4f}, Acc: {train_acc_sm:.4f}")
    print(f"Softmax Val    - Loss: {val_loss_sm:.4f}, Acc: {val_acc_sm:.4f}")
    print(f"Softmax Test   - Loss: {test_loss_sm:.4f}, Acc: {test_acc_sm:.4f}")

    print(f"\n===== {name}: Neural Net =====")
    W1, b1, W2, b2, nn_history, nn_best_epoch = train_nn_runner(
        X_train, Y_train,
        X_val, Y_val,
        d, nn_hidden, k,
        epochs=nn_epochs,
        lr=nn_lr,
        batch_size=batch_size,
        lam=lam,
        seed=seed,
        checkpoint_on_val=checkpoint_on_val,
    )

    train_loss_nn, train_acc_nn = evaluate_nn(X_train, Y_train, W1, b1, W2, b2, lam)
    val_loss_nn, val_acc_nn = evaluate_nn(X_val, Y_val, W1, b1, W2, b2, lam)
    test_loss_nn, test_acc_nn = evaluate_nn(X_test, Y_test, W1, b1, W2, b2, lam)

    print(f"NN Train       - Loss: {train_loss_nn:.4f}, Acc: {train_acc_nn:.4f}")
    print(f"NN Val         - Loss: {val_loss_nn:.4f}, Acc: {val_acc_nn:.4f}")
    print(f"NN Test        - Loss: {test_loss_nn:.4f}, Acc: {test_acc_nn:.4f}")

    return {
        "name": name,
        "softmax": {
            "train_loss": train_loss_sm,
            "train_acc": train_acc_sm,
            "val_loss": val_loss_sm,
            "val_acc": val_acc_sm,
            "test_loss": test_loss_sm,
            "test_acc": test_acc_sm,
            "history": softmax_history,
            "best_epoch": softmax_best_epoch,
        },
        "nn": {
            "train_loss": train_loss_nn,
            "train_acc": train_acc_nn,
            "val_loss": val_loss_nn,
            "val_acc": val_acc_nn,
            "test_loss": test_loss_nn,
            "test_acc": test_acc_nn,
            "history": nn_history,
            "best_epoch": nn_best_epoch,
        }
    }


# -------------------------
# Main
# -------------------------
def main():
    results = []

    # Linear Gaussian
    gaussian_data = load_synthetic_dataset("linear_gaussian.npz")
    results.append(
        run_dataset_experiment(
            "Linear Gaussian",
            *gaussian_data,
            nn_hidden=32,
            softmax_epochs=100,
            nn_epochs=100,
            softmax_lr=0.05,
            nn_lr=0.05,
            batch_size=64,
            lam=1e-4,
            seed=0,
            checkpoint_on_val=False,
        )
    )

    # Moons
    moons_data = load_synthetic_dataset("moons.npz")
    results.append(
        run_dataset_experiment(
            "Moons",
            *moons_data,
            nn_hidden=32,
            softmax_epochs=100,
            nn_epochs=100,
            softmax_lr=0.05,
            nn_lr=0.05,
            batch_size=64,
            lam=1e-4,
            seed=0,
            checkpoint_on_val=False,
        )
    )

    # Digits
    digits_data = load_digits_dataset()
    results.append(
        run_dataset_experiment(
            "Digits",
            *digits_data,
            nn_hidden=32,
            softmax_epochs=200,
            nn_epochs=200,
            softmax_lr=0.05,
            nn_lr=0.05,
            batch_size=64,
            lam=1e-4,
            seed=0,
            checkpoint_on_val=True,
        )
    )

    return results


if __name__ == "__main__":
    run_with_logging(main, CURRENT_DIR, "run_core_experiments")