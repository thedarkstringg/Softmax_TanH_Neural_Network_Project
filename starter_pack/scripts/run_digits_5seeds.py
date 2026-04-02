import os
import sys
import numpy as np

# Add src to path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from training_utils import DataUtils, MetricsCalculator
from train_softmax import SoftmaxTrainer
from train_nn_runner import NNTrainer
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

    seeds = [0, 1, 2, 3, 4]

    softmax_test_accs = []
    softmax_test_losses = []

    nn_test_accs = []
    nn_test_losses = []

    for seed in seeds:
        print(f"\n===== Seed {seed}: Softmax =====")
        softmax_trainer = SoftmaxTrainer(d, k, seed=seed)
        W, b, _, best_epoch_sm = softmax_trainer.train(
            X_train, Y_train, y_train,
            X_val, Y_val, y_val,
            epochs=200,
            lr=0.05,
            batch_size=64,
            lam=1e-4,
            checkpoint_on_val=True,
        )

        test_loss_sm, test_acc_sm = softmax_trainer.evaluate(
            X_test, Y_test, y_test, 1e-4
        )

        print(
            f"Softmax Seed {seed} - "
            f"Best Epoch: {best_epoch_sm}, Test Loss: {test_loss_sm:.4f}, Test Acc: {test_acc_sm:.4f}"
        )

        softmax_test_losses.append(test_loss_sm)
        softmax_test_accs.append(test_acc_sm)

        print(f"\n===== Seed {seed}: Neural Net =====")
        nn_trainer = NNTrainer(d, 32, k, seed=seed)
        W1, b1, W2, b2, _, best_epoch_nn = nn_trainer.train(
            X_train, Y_train,
            X_val, Y_val,
            epochs=200,
            lr=0.05,
            batch_size=64,
            lam=1e-4,
            checkpoint_on_val=True,
        )

        test_loss_nn, test_acc_nn = nn_trainer.evaluate(
            X_test, Y_test, 1e-4
        )

        print(
            f"NN Seed {seed} - "
            f"Best Epoch: {best_epoch_nn}, Test Loss: {test_loss_nn:.4f}, Test Acc: {test_acc_nn:.4f}"
        )

        nn_test_losses.append(test_loss_nn)
        nn_test_accs.append(test_acc_nn)

    sm_acc_mean, sm_acc_low, sm_acc_high, sm_acc_std = MetricsCalculator.compute_ci95_for_five(softmax_test_accs)
    sm_loss_mean, sm_loss_low, sm_loss_high, sm_loss_std = MetricsCalculator.compute_ci95_for_five(softmax_test_losses)

    nn_acc_mean, nn_acc_low, nn_acc_high, nn_acc_std = MetricsCalculator.compute_ci95_for_five(nn_test_accs)
    nn_loss_mean, nn_loss_low, nn_loss_high, nn_loss_std = MetricsCalculator.compute_ci95_for_five(nn_test_losses)

    print("\n===== 5-Seed Summary =====")
    print("Softmax Test Accs :", softmax_test_accs)
    print("Softmax Test Loss :", softmax_test_losses)
    print("NN Test Accs      :", nn_test_accs)
    print("NN Test Loss      :", nn_test_losses)

    print("\nSoftmax Accuracy")
    print(f"Mean: {sm_acc_mean:.4f}, 95% CI: [{sm_acc_low:.4f}, {sm_acc_high:.4f}], Std: {sm_acc_std:.4f}")

    print("Softmax Cross-Entropy")
    print(f"Mean: {sm_loss_mean:.4f}, 95% CI: [{sm_loss_low:.4f}, {sm_loss_high:.4f}], Std: {sm_loss_std:.4f}")

    print("\nNN Accuracy")
    print(f"Mean: {nn_acc_mean:.4f}, 95% CI: [{nn_acc_low:.4f}, {nn_acc_high:.4f}], Std: {nn_acc_std:.4f}")

    print("NN Cross-Entropy")
    print(f"Mean: {nn_loss_mean:.4f}, 95% CI: [{nn_loss_low:.4f}, {nn_loss_high:.4f}], Std: {nn_loss_std:.4f}")

    return {
        "softmax_test_accs": softmax_test_accs,
        "softmax_test_losses": softmax_test_losses,
        "nn_test_accs": nn_test_accs,
        "nn_test_losses": nn_test_losses,
    }


if __name__ == "__main__":
    logger = ExperimentLogger(CURRENT_DIR, "run_digits_5seeds")
    logger.run(main)