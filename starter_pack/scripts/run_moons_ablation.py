import os
import numpy as np

from training_utils import labels_to_onehot
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
def load_moons_dataset():
    path = os.path.join(DATA_DIR, "moons.npz")
    data = np.load(path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -------------------------
# Main
# -------------------------
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_moons_dataset()

    k = len(np.unique(y_train))
    d = X_train.shape[1]

    Y_train = labels_to_onehot(y_train, k)
    Y_val = labels_to_onehot(y_val, k)
    Y_test = labels_to_onehot(y_test, k)

    widths = [2, 8, 32]
    results = []

    for h in widths:
        print(f"\n===== Moons Width Ablation: h = {h} =====")

        W1, b1, W2, b2, history, _ = train_nn_runner(
            X_train, Y_train,
            X_val, Y_val,
            d, h, k,
            epochs=100,
            lr=0.05,
            batch_size=64,
            lam=1e-4,
            seed=0,
            checkpoint_on_val=False,
        )

        train_loss, train_acc = evaluate_nn(X_train, Y_train, W1, b1, W2, b2, 1e-4)
        val_loss, val_acc = evaluate_nn(X_val, Y_val, W1, b1, W2, b2, 1e-4)
        test_loss, test_acc = evaluate_nn(X_test, Y_test, W1, b1, W2, b2, 1e-4)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        results.append({
            "hidden_width": h,
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
            f"h={r['hidden_width']}: "
            f"Test Loss={r['test_loss']:.4f}, Test Acc={r['test_acc']:.4f}"
        )

    return results


if __name__ == "__main__":
    run_with_logging(main, CURRENT_DIR, "run_moons_ablation")